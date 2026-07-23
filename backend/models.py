from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, DateTime
from sqlalchemy.sql import func
from database import Base

class ClaimTransaction(Base):
    """
    SQLAlchemy model representing an adjudicated claim transaction.
    Logs extracted fields, ML predictions, hard math caps, triage status, and LLM justifications.
    """
    __tablename__ = "claim_transactions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transaction_id = Column(String(64), unique=True, index=True, nullable=False)

    # Ingested Claim Primitives
    patient_age = Column(Integer, nullable=True)
    policy_number = Column(String(64), nullable=True)
    policy_year = Column(String(16), nullable=True)
    annual_sum_insured = Column(Integer, nullable=True)
    diagnosis_code = Column(String(32), nullable=True)
    procedure_code = Column(String(32), nullable=True)
    treatment_category = Column(String(64), nullable=True)
    claim_amount = Column(Integer, nullable=True)

    # ML Predictions & Hard Math Rules
    predicted_approval_prob = Column(Float, nullable=True)
    predicted_payout_ratio = Column(Float, nullable=True)
    hard_math_cap = Column(Integer, nullable=True)

    # Final Outcome & Deposit Logic
    final_status = Column(String(32), nullable=True)
    required_deposit = Column(Integer, nullable=True)

    # Reasoning Output
    reasoner_analysis = Column(JSON, nullable=True)

    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())
