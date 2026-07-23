from typing import TypedDict, Optional

class ClaimGraphState(TypedDict):
    transaction_id: str
    
    # Ingestion Data (Member A)
    patient_age: Optional[int]
    policy_number: Optional[str]
    policy_year: Optional[str]
    annual_sum_insured: Optional[int]
    diagnosis_code: Optional[str]
    procedure_code: Optional[str]
    treatment_category: Optional[str]
    claim_amount: Optional[int]
    
    # Vector Search Context (Member B)
    similar_cases_context: Optional[str]
    
    # ML Prediction Data (Track 1)
    predicted_approval_prob: Optional[float]
    predicted_payout_ratio: Optional[float]
    hard_math_cap: Optional[int]
    
    # Triage Routing Outcomes (Track 2)
    final_status: Optional[str]  # e.g., "GREEN", "YELLOW", "RED"
    required_deposit: Optional[int]
    
    # Reasoner Output (Track 3)
    reasoner_analysis: Optional[str]
