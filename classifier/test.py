import json
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

def clean_financial(val):
    """Safely extracts numeric values, handling 'LPA' strings and missing data."""
    if pd.isna(val) or val is None:
        return 0.0
    val_str = str(val).upper().strip()
    num = re.sub(r'[^\d.]', '', val_str)
    if 'LPA' in val_str:
        return float(num) * 100000 if num else 0.0
    return float(num) if num else 0.0

def run_pipeline():
    print("1. Ingesting and Flattening JSON...")
    raw_records = []
    # Reads the line-delimited JSON safely
    with open('cleaned_insurance_claims.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))
    
    # Flattens nested JSON objects into standard column headers
    df = pd.json_normalize(raw_records)
    
    print("2. Sanitizing Data...")
    financial_cols = [
        'patient.annual_income_inr',
        'claim.financial_breakdown.hospital_bill_amount',
        'claim.financial_breakdown.total_claim_amount'
    ]
    for col in financial_cols:
        df[col] = df[col].apply(clean_financial)

    boolean_columns = [
        'patient.health_profile.has_diabetes',
        'patient.health_profile.has_hypertension',
        'patient.health_profile.family_history_cardiac',
        'claim.is_cashless'
    ]
    for col in boolean_columns:
        df[col] = df[col].fillna(False).astype(int)

    df['patient.health_profile.stress_level_score'] = df['patient.health_profile.stress_level_score'].fillna(0.0)
    df['patient.health_profile.bmi'] = df['patient.health_profile.bmi'].fillna(df['patient.health_profile.bmi'].mean())
    
    print("3. Defining Features and Splitting Data...")
    df['payout_ratio'] = df['claim.financial_breakdown.total_claim_amount'] / df['claim.financial_breakdown.hospital_bill_amount']
    df['payout_ratio'] = df['payout_ratio'].fillna(0.0)
    
    # Binary Target for the Classifier
    df['claim_status'] = np.where(df['claim.financial_breakdown.total_claim_amount'] > 0, 1, 0)
    
    features = [
        'patient.age', 'patient.gender', 'patient.marital_status',
        'patient.health_profile.bmi', 'patient.health_profile.tobacco_usage',
        'patient.health_profile.alcohol_units_per_week', 'patient.health_profile.physical_activity_level',
        'patient.health_profile.diet_type', 'patient.health_profile.has_diabetes',
        'patient.health_profile.has_hypertension', 'patient.health_profile.family_history_cardiac',
        'patient.health_profile.stress_level_score', 'policy.policy_type', 'policy.sum_insured',
        'claim.is_cashless', 'claim.hospital_details.tier', 'claim.hospital_details.room_category',
        'claim.stay_details.icu_days', 'claim.stay_details.length_of_stay_days',
        'claim.financial_breakdown.hospital_bill_amount'
    ]
    
    X_matrix = df[features].copy()
    X_encoded = pd.get_dummies(X_matrix, drop_first=True)
    
    # 70/15/15 Split 
    y_class = df['claim_status']
    X_temp, X_test, y_c_temp, y_c_test = train_test_split(X_encoded, y_class, test_size=0.15, random_state=42)
    X_train, X_val, y_c_train, y_c_val = train_test_split(X_temp, y_c_temp, test_size=0.176, random_state=42)
    
    print("4. Training XGBoost Classifier (Approvals)...")
    classifier = XGBClassifier(eval_metric='logloss', random_state=42)
    classifier.fit(X_train, y_c_train, eval_set=[(X_val, y_c_val)], verbose=False)
    
    print("5. Training XGBoost Regressor (Payouts)...")
    # Strictly isolate historically approved claims for the Regressor
    df_approved = df[df['claim_status'] == 1]
    X_reg = df_approved[features].copy()
    X_reg_encoded = pd.get_dummies(X_reg, drop_first=True)
    
    # Align the regressor matrix columns perfectly with the classifier matrix
    X_reg_encoded = X_reg_encoded.reindex(columns=X_encoded.columns, fill_value=0) 
    y_reg = df_approved['payout_ratio']
    
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_reg_encoded, y_reg, test_size=0.2, random_state=42)
    regressor = XGBRegressor(objective='reg:squarederror', random_state=42)
    regressor.fit(X_r_train, y_r_train)
    
    print("6. Exporting Trained Models for LangGraph Orchestrator...")
    joblib.dump(classifier, 'approval_classifier.joblib')
    joblib.dump(regressor, 'payout_regressor.joblib')
    joblib.dump(list(X_encoded.columns), 'model_features.joblib')
    
    print("Pipeline Complete! Artifacts generated successfully.")

if __name__ == "__main__":
    run_pipeline()