"""
reasoner.py — Track 3, Task 1: The Reasoner (LLM Call 2)
 
WHAT THIS DOES
---------------
Takes 4 pieces of information and asks a large cloud model (Qwen 3 32B)
to write a plain-language explanation for hospital admin staff.
 
The 4 inputs (per the project doc):
  1. Raw document text / extracted JSON      -> from Member A (real, already exists)
  2. Historical similar claims                -> from Member B's vector search (FAKED here)
  3. Statistical calculations                  -> from Track 1's ML models (FAKED here)
  4. Operational review instructions           -> written by you (Track 3)
 
GOLDEN RULE
-----------
This script NEVER computes anything itself. It only takes numbers that
were already calculated elsewhere and asks the model to explain them in
words. If you ever find yourself writing "amount * rate" in this file,
stop — that math belongs in Track 1's code, not here.
 
HOW TO WIRE THIS UP LATER
--------------------------
Right now `get_dummy_extracted_claim()`, `get_dummy_historical_examples()`,
and `get_dummy_stats()` return fake data so you can build and test this
file completely on your own. When your teammates' pieces are ready,
replace those 3 functions with real calls:
  - extracted claim  -> load from extracted_json/extracted_claim.json (Member A)
  - historical examples -> call VectorStore.search() (Member B, in embedding/)
  - stats -> call whatever Track 1 exposes (approval probability, payout
    ratio, remaining coverage cap)
Nothing else in this file needs to change.
"""
 
import os
import json
import requests
 
# ---------------------------------------------------------------------------
# CONFIG — fill these in with whichever cloud provider hosts Qwen 3 32B for
# you (OpenRouter, Together AI, Fireworks, etc). Most of these providers
# expose an OpenAI-compatible /chat/completions endpoint, so this script
# is written against that shape. Set these as environment variables so you
# never hardcode a key into the file.
# ---------------------------------------------------------------------------
REASONER_API_URL = os.environ.get(
    "REASONER_API_URL", "https://openrouter.ai/api/v1/chat/completions"
)
REASONER_API_KEY = os.environ.get("REASONER_API_KEY", "")
REASONER_MODEL = os.environ.get("REASONER_MODEL", "qwen/qwen3-32b")
 
 
# ---------------------------------------------------------------------------
# DUMMY DATA — stand-ins for your teammates' not-yet-built pieces
# ---------------------------------------------------------------------------
 
def get_dummy_extracted_claim() -> dict:
    """Stands in for Member A's extracted_json/extracted_claim.json."""
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
 
 
def get_dummy_historical_examples() -> list:
    """Stands in for Member B's VectorStore.search() results."""
    return [
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
 
 
def get_dummy_stats() -> dict:
    """Stands in for Track 1's classifier + regressor + hard cap output."""
    return {
        "approval_probability": 0.87,
        "predicted_payout_ratio": 0.78,
        "predicted_payout_amount": 66300,
        "remaining_annual_coverage": 115000,
        "hard_cap_applied": False,
    }
 
 
# ---------------------------------------------------------------------------
# PROMPT ASSEMBLY
# ---------------------------------------------------------------------------
 
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
    """Combines the 4 components into a single prompt for the Reasoner LLM."""
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
 
 
# ---------------------------------------------------------------------------
# LLM CALL
# ---------------------------------------------------------------------------
 
def call_reasoner_llm(prompt: str) -> str:
    """Sends the prompt to the cloud-hosted Qwen 3 32B model."""
    if not REASONER_API_KEY:
        raise RuntimeError(
            "REASONER_API_KEY is not set. Export it as an environment "
            "variable before running this script, e.g.:\n"
            "  export REASONER_API_KEY=sk-...\n"
            "Also double check REASONER_API_URL and REASONER_MODEL match "
            "whichever provider you're using to host Qwen 3 32B."
        )
 
    headers = {
        "Authorization": f"Bearer {REASONER_API_KEY}",
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
    }
 
    response = requests.post(REASONER_API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    extracted_claim = get_dummy_extracted_claim()
    historical_examples = get_dummy_historical_examples()
    stats = get_dummy_stats()
 
    prompt = build_master_prompt(extracted_claim, historical_examples, stats)
 
    print("----- PROMPT SENT TO REASONER -----")
    print(prompt)
    print("------------------------------------\n")
 
    try:
        explanation = call_reasoner_llm(prompt)
    except RuntimeError as e:
        print(f"[Skipped actual API call] {e}")
        return
 
    print("----- REASONER OUTPUT -----")
    print(explanation)
 
    # Handoff output, for later use by Track 2's LangGraph state
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
    print("\nSaved to reasoner_output/latest_reasoning.json")
 
 
if __name__ == "__main__":
    main()
 