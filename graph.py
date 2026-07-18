import os
import json
import uuid
from typing import Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from state import ClaimGraphState
from models import ClaimTransaction
from database import SessionLocal

from embedding.retriever import Retriever
from reasoner.reasoner import build_master_prompt

load_dotenv()

# Instantiate Qwen 3 32B via Groq
# Valid model names on Groq may vary depending on their exact Qwen 3 distribution.
# qwen2.5-32b-it is currently on Groq API. The user specified Qwen 3 32b, but Qwen 2.5 is the commonly hosted one. 
# We'll use the environment variable GROQ_MODEL or a descriptive default.
groq_model_name = os.getenv("GROQ_MODEL", "qwen/qwen3-32b") # Adjust as Groq updates
try:
    llm = ChatGroq(model_name=groq_model_name, temperature=0.1)
except Exception:
    llm = None

def ingestion_extraction_node(state: ClaimGraphState):
    """
    Simulates retrieving the extracted data from Member A's task.
    Reads extracted_claim.json and populates the graph state.
    """
    json_path = os.path.join("extracted_json", "extracted_claim.json")
    
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # Fallback to mock data if the file isn't there yet
        data = {
            "patient_age": 42,
            "policy_number": "HDFC-ERGO-XYZ",
            "policy_year": "2026",
            "annual_sum_insured": 200000,
            "diagnosis_code": "K80.20",
            "procedure_code": "47600",
            "treatment_category": "Laparoscopic Surgery",
            "claim_amount": 85000
        }
    
    return {
        "patient_age": data.get("patient_age"),
        "policy_number": data.get("policy_number"),
        "policy_year": data.get("policy_year"),
        "annual_sum_insured": data.get("annual_sum_insured"),
        "diagnosis_code": data.get("diagnosis_code"),
        "procedure_code": data.get("procedure_code"),
        "treatment_category": data.get("treatment_category"),
        "claim_amount": data.get("claim_amount"),
    }
    
def reference_lookup_node(state: ClaimGraphState):
    """
    Track B Vector Search.
    Finds similar cases.
    """
    claim_dict = {
        "patient_age": state.get("patient_age"),
        "policy_number": state.get("policy_number"),
        "policy_year": state.get("policy_year"),
        "annual_sum_insured": state.get("annual_sum_insured"),
        "diagnosis_code": state.get("diagnosis_code"),
        "procedure_code": state.get("procedure_code"),
        "treatment_category": state.get("treatment_category"),
        "claim_amount": state.get("claim_amount"),
    }
    
    try:
        retriever = Retriever()
        results = retriever.search_by_claim(claim_dict, top_k=5)
        
        if not results:
            context = "[]"
        else:
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
            context = json.dumps(historical_claims, indent=2)
    except Exception as e:
        context = "[]"
        
    return {
        "similar_cases_context": context
    }

import joblib
import pandas as pd

def ml_inference_node(state: ClaimGraphState):
    """
    Track 1 ML processing.
    Predicts approval prob, payout ratio, and calculates hard math cap.
    """
    claim_amount = state.get("claim_amount") or 0
    annual_sum = state.get("annual_sum_insured") or 0
    
    # Hard math calculation (Golden Rule: Done deterministically in python)
    amount_settled_ytd = 50000
    hard_math_cap = max(0, annual_sum - amount_settled_ytd)
    
    # ML Inference
    try:
        expected_cols = joblib.load(os.path.join("classifier", "model_features.joblib"))
        
        # Prepare sparse dataframe from available state
        df = pd.DataFrame([{
            'patient.age': state.get('patient_age') or 35,
            'policy.sum_insured': annual_sum,
            'claim.financial_breakdown.hospital_bill_amount': claim_amount
        }])
        
        X_new = pd.get_dummies(df, drop_first=True)
        X_new = X_new.reindex(columns=expected_cols, fill_value=0)
        
        # Load and predict classifier independently
        try:
            clf = joblib.load(os.path.join("classifier", "approval_classifier.joblib"))
            prob = clf.predict_proba(X_new)[0][1]
            prob = float(prob)
        except Exception as e:
            print(f"[Warning] Classifier load failed ({e}). Defaulting prob to 0.95.")
            prob = 0.95
            
        # Load and predict regressor independently
        try:
            reg = joblib.load(os.path.join("classifier", "payout_regressor.joblib"))
            payout = reg.predict(X_new)[0]
            payout = float(max(0.0, min(1.0, payout)))
        except Exception as e:
            print(f"[Warning] Regressor load failed ({e}). Defaulting payout to 0.85.")
            payout = 0.85
            
    except Exception as e:
        print(f"[Warning] ML features load failed: {e}. Falling back to full defaults.")
        prob = 0.95
        payout = 0.85
    
    return {
        "predicted_approval_prob": round(prob, 2),
        "predicted_payout_ratio": round(payout, 2),
        "hard_math_cap": hard_math_cap
    }

def route_triage(state: ClaimGraphState) -> Literal["green_path", "yellow_path", "red_path"]:
    """
    Conditional routing for triage Matrix.
    """
    prob = state.get("predicted_approval_prob", 0.0)
    payout = state.get("predicted_payout_ratio", 0.0)
    
    if prob >= 0.80 and payout >= 0.95:
        return "green_path"
    elif prob >= 0.80 and payout < 0.95:
        return "yellow_path"
    else:
        return "red_path"

def green_path_node(state: ClaimGraphState):
    return {
        "final_status": "APPROVED",
        "required_deposit": 0
    }

def yellow_path_node(state: ClaimGraphState):
    # Safety deposit is the difference between claim amount and predicted payout amount
    claim = state.get("claim_amount", 0)
    payout_ratio = state.get("predicted_payout_ratio", 0.0)
    cap = state.get("hard_math_cap", claim)
    
    predicted_coverage = min(claim * payout_ratio, cap)
    deposit = max(0, int(claim - predicted_coverage))
    
    return {
        "final_status": "PENDING_DEPOSIT",
        "required_deposit": deposit
    }

def red_path_node(state: ClaimGraphState):
    return {
        "final_status": "REJECTED_OR_ESCALATED",
        "required_deposit": state.get("claim_amount", 0)
    }

def reasoner_node(state: ClaimGraphState):
    """
    Track 3 integration. Uses LLM to provide a human-readable justification.
    """
    if llm is None:
        return {"reasoner_analysis": "[Mock Reasoner] GROQ_API_KEY not set or model error. Analysis skipped."}
        
    claim_dict = {
        "patient_age": state.get("patient_age"),
        "policy_number": state.get("policy_number"),
        "policy_year": state.get("policy_year"),
        "annual_sum_insured": state.get("annual_sum_insured"),
        "diagnosis_code": state.get("diagnosis_code"),
        "procedure_code": state.get("procedure_code"),
        "treatment_category": state.get("treatment_category"),
        "claim_amount": state.get("claim_amount"),
    }
    
    stats_dict = {
        "approval_probability": state.get("predicted_approval_prob"),
        "predicted_payout_ratio": state.get("predicted_payout_ratio"),
        "hard_math_cap": state.get("hard_math_cap"),
        "final_status": state.get("final_status"),
        "required_deposit": state.get("required_deposit")
    }
    
    try:
        historical_cases = json.loads(state.get("similar_cases_context") or "[]")
    except Exception:
        historical_cases = []
        
    master_prompt = build_master_prompt(claim_dict, historical_cases, stats_dict)
    
    messages = [
        SystemMessage(content="You are an insurance justification reasoning agent. You never do math yourself."),
        HumanMessage(content=master_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        # Remove <think> blocks if present
        import re
        content = re.sub(r"<think>.*?(?:</think>|$)\s*", "", response.content, flags=re.DOTALL).strip()
    except Exception as e:
        content = f"[Reasoner Error] {e}"
        
    return {"reasoner_analysis": content}
    
def persist_state_node(state: ClaimGraphState):
    """
    Saves the finalized state into the persistent DB (Postgres/SQLite).
    """
    db = SessionLocal()
    try:
        tx = ClaimTransaction(
            transaction_id=state["transaction_id"],
            patient_age=state.get("patient_age"),
            policy_number=state.get("policy_number"),
            policy_year=state.get("policy_year"),
            annual_sum_insured=state.get("annual_sum_insured"),
            diagnosis_code=state.get("diagnosis_code"),
            procedure_code=state.get("procedure_code"),
            treatment_category=state.get("treatment_category"),
            claim_amount=state.get("claim_amount"),
            predicted_approval_prob=state.get("predicted_approval_prob"),
            predicted_payout_ratio=state.get("predicted_payout_ratio"),
            hard_math_cap=state.get("hard_math_cap"),
            final_status=state.get("final_status"),
            required_deposit=state.get("required_deposit"),
            reasoner_analysis=state.get("reasoner_analysis")
        )
        db.add(tx)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Failed to persist state: {e}")
    finally:
        db.close()
        
    return {}

# ----------------- GRAPH COMPILATION -----------------
workflow = StateGraph(ClaimGraphState)

workflow.add_node("ingestion", ingestion_extraction_node)
workflow.add_node("reference", reference_lookup_node)
workflow.add_node("ml_inference", ml_inference_node)
workflow.add_node("green_path", green_path_node)
workflow.add_node("yellow_path", yellow_path_node)
workflow.add_node("red_path", red_path_node)
workflow.add_node("reasoner", reasoner_node)
workflow.add_node("persist", persist_state_node)

# Flow defined
workflow.set_entry_point("ingestion")
workflow.add_edge("ingestion", "reference")
workflow.add_edge("reference", "ml_inference")

workflow.add_conditional_edges(
    "ml_inference",
    route_triage,
    {
        "green_path": "green_path",
        "yellow_path": "yellow_path",
        "red_path": "red_path"
    }
)

workflow.add_edge("green_path", "reasoner")
workflow.add_edge("yellow_path", "reasoner")
workflow.add_edge("red_path", "reasoner")
workflow.add_edge("reasoner", "persist")
workflow.add_edge("persist", END)

claim_graph = workflow.compile()
