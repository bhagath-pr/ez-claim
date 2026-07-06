from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, DateTime
from sqlalchemy.sql import func
from database import Base

class ClaimTransaction(Base):
    __tablename__ = "claim_transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True, nullable=False)
    
    # Original Data Primitives (from Member A)
    patient_age = Column(Integer, nullable=True)
    policy_number = Column(String, nullable=True)
    policy_year = Column(String, nullable=True)
    annual_sum_insured = Column(Integer, nullable=True)
    diagnosis_code = Column(String, nullable=True)
    procedure_code = Column(String, nullable=True)
    treatment_category = Column(String, nullable=True)
    claim_amount = Column(Integer, nullable=True)
    
    # Track 1 Output (ML Inference & Math Cap)
    predicted_approval_prob = Column(Float, nullable=True)
    predicted_payout_ratio = Column(Float, nullable=True)
    hard_math_cap = Column(Integer, nullable=True)
    
    # Triage State (Track 2)
    final_status = Column(String, nullable=True) # e.g. "APPROVED", "PENDING_DEPOSIT", "REJECTED"
    required_deposit = Column(Integer, default=0)
    
    # Track 3 (Reasoner Analysis)
    reasoner_analysis = Column(String, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
