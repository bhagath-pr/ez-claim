# pyrefly: ignore [missing-import]
import joblib
import pandas as pd
import re

def clean_financial(val):
    """Safely extract numeric values for incoming inference data."""
    if pd.isna(val) or val is None:
        return 0.0
    val_str = str(val).upper().strip()
    num = re.sub(r'[^\d.]', '', val_str)
    if 'LPA' in val_str:
        return float(num) * 100000 if num else 0.0
    return float(num) if num else 0.0

class ClaimRiskEngine:
    def __init__(self):
        # 1. Load the "Saved Brains" into memory (Regressor Only)
        print("Loading ML Models...")
        self.regressor = joblib.load('payout_regressor.joblib')
        self.expected_columns = joblib.load('model_features.joblib')
        
    def format_incoming_claim(self, raw_json_claim):
        # 2. Flatten the single JSON object exactly like we did in training
        df = pd.json_normalize([raw_json_claim])
        
        # Apply the exact same cleaning rules
        if 'claim.financial_breakdown.hospital_bill_amount' in df.columns:
            df['claim.financial_breakdown.hospital_bill_amount'] = df['claim.financial_breakdown.hospital_bill_amount'].apply(clean_financial)
            
        # Handle booleans
        boolean_columns = [
            'patient.health_profile.has_diabetes',
            'patient.health_profile.has_hypertension',
            'patient.health_profile.family_history_cardiac',
            'claim.is_cashless'
        ]
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(int)

        # 3. One-Hot Encode the new claim
        X_new = pd.get_dummies(df, drop_first=True)
        
        # 4. CRITICAL STEP: Align the new claim's columns perfectly with the trained model
        # If the new claim is missing a category (like a specific hospital_tier), it fills it with 0
        X_new = X_new.reindex(columns=self.expected_columns, fill_value=0)
        return X_new

    def evaluate_claim(self, raw_json_claim, annual_sum_insured, previously_settled_this_year):
        # Format the data
        X_matrix = self.format_incoming_claim(raw_json_claim)
        gross_bill = clean_financial(raw_json_claim['claim']['financial_breakdown']['hospital_bill_amount'])
        
        # --- ML INFERENCE (Payout Only) ---
        # Predict Payout Ratio
        predicted_ratio = self.regressor.predict(X_matrix)[0]
        uncapped_payout = gross_bill * predicted_ratio
        
        # --- DETERMINISTIC CAP (THE GOLDEN RULE) ---
        remaining_coverage = annual_sum_insured - previously_settled_this_year
        
        # The math forces the payout to be whichever is lower
        final_authorized_payout = min(uncapped_payout, remaining_coverage)
        
        # Calculate what the hospital must collect from the patient upfront
        safety_deposit_required = gross_bill - final_authorized_payout
        
        return {
            "hospital_bill": round(gross_bill, 2),
            "predicted_insurance_payout": round(final_authorized_payout, 2),
            "safety_deposit_required": round(max(0, safety_deposit_required), 2), # Deposit can't be negative
            "hit_coverage_ceiling": uncapped_payout > remaining_coverage
        }

# ==========================================
# TEST THE ENGINE WITH A FAKE INCOMING CLAIM
# ==========================================
if __name__ == "__main__":
    engine = ClaimRiskEngine()
    
    # This simulates the JSON your frontend/extraction engine will send
    mock_incoming_claim = {
        "patient": {
            "age": 42,
            "gender": "male",
            "marital_status": "Married",
            "health_profile": {
                "bmi": 28.5,
                "tobacco_usage": "none",
                "alcohol_units_per_week": 2,
                "physical_activity_level": "moderate",
                "diet_type": "veg",
                "has_diabetes": False,
                "has_hypertension": False,
                "family_history_cardiac": False,
                "stress_level_score": 4.5
            }
        },
        "policy": {
            "policy_type": "family_floater",
            "sum_insured": 100000
        },
        "claim": {
            "is_cashless": True,
            "hospital_details": {
                "tier": "private_corporate",
                "room_category": "private"
            },
            "stay_details": {
                "icu_days": 1,
                "length_of_stay_days": 4
            },
            "financial_breakdown": {
                "hospital_bill_amount": "₹250,000"
            }
        }
    }
    
    # Run the evaluation! 
    # Let's pretend the patient has a 500,000 policy and has already used 400,000 this year.
    results = engine.evaluate_claim(
        raw_json_claim=mock_incoming_claim,
        annual_sum_insured=500000,
        previously_settled_this_year=400000
    )
    
    print("\n=== EZ CLAIM TRIAGE RESULTS ===")
    for key, value in results.items():
        print(f"{key}: {value}")