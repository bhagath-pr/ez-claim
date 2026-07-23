import os
import json
import requests
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from embedding.retriever import Retriever
except ImportError:
    Retriever = None

try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path)
except ImportError:
    pass

REASONER_API_URL = os.environ.get(
    "REASONER_API_URL", "https://api.groq.com/openai/v1/chat/completions"
)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
REASONER_MODEL = os.environ.get("REASONER_MODEL", "qwen/qwen3.6-27b")

def get_dummy_extracted_claim() -> dict:
    json_path = os.path.join(os.path.dirname(__file__), "..", "extracted_json", "extracted_claim.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to read {json_path}: {e}")
    
    return {
        "patient_age": 42,
        "policy_number": "HDFC-ERGO-XYZ",
        "policy_year": "2026",
        "annual_sum_insured": 200000,
        "diagnosis_code": "K80.20",
        "procedure_code": "47600",
        "treatment_category": "Laparoscopic Surgery",
        "claim_amount": 85000,
    }

def get_dummy_historical_examples(extracted_claim: dict = None) -> list:
    fallback_data = [
        {
            "diagnosis_code": "K80.20",
            "treatment_category": "Laparoscopic Surgery",
            "claim_amount": 78000,
            "outcome": "approved",
            "payout_ratio": 0.90,
        },
        {
            "diagnosis_code": "K80.20",
            "treatment_category": "Laparoscopic Surgery",
            "claim_amount": 92000,
            "outcome": "approved",
            "payout_ratio": 0.72,
        },
        {
            "diagnosis_code": "K80.00",
            "treatment_category": "Laparoscopic Surgery",
            "claim_amount": 65000,
            "outcome": "rejected",
            "payout_ratio": 0.0,
        },
    ]

    if not extracted_claim or not Retriever:
        return fallback_data

    try:
        retriever = Retriever(
            collection_name="insurance_claims",
            persist_directory=os.path.join(os.path.dirname(__file__), "..", "vector_db")
        )
        
        if retriever.store.count() == 0:
            return fallback_data

        results = retriever.search_by_claim(extracted_claim, top_k=5)
        
        if not results:
            return fallback_data

        historical_claims = []
        for res in results:
            meta = res.get("metadata", {})
            historical_claims.append({
                "diagnosis_code": meta.get("diagnosis_code", "UNKNOWN"),
                "treatment_category": meta.get("treatment_category", "UNKNOWN"),
                "claim_amount": meta.get("claim_amount", 0),
                "outcome": meta.get("outcome", "unknown"),
                "payout_ratio": meta.get("payout_ratio", 0.0),
            })
        return historical_claims

    except Exception as e:
        return fallback_data

def get_dummy_stats() -> dict:
    return {
        "approval_probability": 0.87,
        "predicted_payout_ratio": 0.78,
        "predicted_payout_amount": 66300,
        "remaining_annual_coverage": 115000,
        "hard_cap_applied": False,
    }

REVIEW_INSTRUCTIONS = """
You are writing for a hospital discharge desk administrator, not a doctor
or an insurance expert. They need to decide, right now, whether to let a
patient leave and how much (if any) safety deposit to collect.
 
Rules you MUST follow:
- Do NOT perform any arithmetic. All numbers below are already final.
  Only explain and narrate them.
- Do NOT invent numbers that are not given to you.
- Be direct and short: 4-6 sentences maximum.
- State clearly: (1) the likely approval outcome, (2) the recommended
  deposit amount if any, (3) one sentence citing why, based on the
  historical examples given.
- If approval_probability is below 0.5, or the diagnosis matches a
  rejected historical example, recommend the Escalation/Red path instead
  of authorizing discharge.
"""

def build_master_prompt(extracted_claim: dict, historical_examples: list, stats: dict) -> str:
    return f"""
### 1. EXTRACTED CLAIM DATA
{json.dumps(extracted_claim, indent=2)}
 
### 2. HISTORICAL SIMILAR CLAIMS (for context only, do not recalculate)
{json.dumps(historical_examples, indent=2)}
 
### 3. STATISTICAL CALCULATIONS (already computed deterministically — trust these numbers)
{json.dumps(stats, indent=2)}
 
### 4. YOUR INSTRUCTIONS
{REVIEW_INSTRUCTIONS}
 
Now write the explanation for the hospital administrator.
"""

def call_reasoner_llm(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": REASONER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful, concise hospital claims explainer. You never do math yourself.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
    }

    max_retries = 3
    for attempt in range(max_retries):
        response = requests.post(REASONER_API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 2))
            time.sleep(retry_after)
            continue
            
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        import re
        content = re.sub(r"<think>.*?(?:</think>|$)\s*", "", content, flags=re.DOTALL)
        
        return content.strip()
        
    raise RuntimeError("Exceeded maximum retries for Groq API due to rate limits.")

def main():
    extracted_claim = get_dummy_extracted_claim()
    historical_examples = get_dummy_historical_examples(extracted_claim)
    stats = get_dummy_stats()

    prompt = build_master_prompt(extracted_claim, historical_examples, stats)
    try:
        explanation = call_reasoner_llm(prompt)
    except RuntimeError as e:
        print(f"[Skipped actual API call] {e}")
        return

    os.makedirs("reasoner_output", exist_ok=True)
    with open("reasoner_output/latest_reasoning.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "extracted_claim": extracted_claim,
                "stats": stats,
                "explanation": explanation,
            },
            f,
            indent=2,
        )

if __name__ == "__main__":
    main()
