import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor

def generate_models():
    output_dir = os.path.dirname(__file__)

    # Define standard features expected by graph.py
    features = [
        'patient.age', 'patient.health_profile.bmi',
        'patient.health_profile.alcohol_units_per_week',
        'patient.health_profile.stress_level_score',
        'patient.health_profile.has_diabetes',
        'patient.health_profile.has_hypertension',
        'patient.health_profile.family_history_cardiac',
        'policy.sum_insured', 'claim.is_cashless',
        'claim.stay_details.icu_days', 'claim.stay_details.length_of_stay_days',
        'claim.financial_breakdown.hospital_bill_amount'
    ]

    # Create synthetic dataset for model fitting
    np.random.seed(42)
    n_samples = 100
    data = {
        'patient.age': np.random.randint(18, 80, size=n_samples),
        'patient.health_profile.bmi': np.random.uniform(18.5, 35.0, size=n_samples),
        'patient.health_profile.alcohol_units_per_week': np.random.randint(0, 15, size=n_samples),
        'patient.health_profile.stress_level_score': np.random.uniform(1.0, 10.0, size=n_samples),
        'patient.health_profile.has_diabetes': np.random.randint(0, 2, size=n_samples),
        'patient.health_profile.has_hypertension': np.random.randint(0, 2, size=n_samples),
        'patient.health_profile.family_history_cardiac': np.random.randint(0, 2, size=n_samples),
        'policy.sum_insured': np.random.choice([100000, 200000, 500000, 1000000], size=n_samples),
        'claim.is_cashless': np.random.randint(0, 2, size=n_samples),
        'claim.stay_details.icu_days': np.random.randint(0, 5, size=n_samples),
        'claim.stay_details.length_of_stay_days': np.random.randint(1, 14, size=n_samples),
        'claim.financial_breakdown.hospital_bill_amount': np.random.randint(10000, 150000, size=n_samples)
    }

    df = pd.DataFrame(data)
    X = pd.get_dummies(df, drop_first=True)
    expected_cols = list(X.columns)

    # Classification target: 1 = approved, 0 = rejected
    y_class = np.where(df['claim.financial_breakdown.hospital_bill_amount'] < df['policy.sum_insured'], 1, 0)
    
    # Regression target: payout ratio between 0.5 and 1.0
    y_reg = np.random.uniform(0.65, 0.95, size=n_samples)

    clf = XGBClassifier(eval_metric='logloss', random_state=42)
    clf.fit(X, y_class)

    reg = XGBRegressor(objective='reg:squarederror', random_state=42)
    reg.fit(X, y_reg)

    joblib.dump(clf, os.path.join(output_dir, 'approval_classifier.joblib'))
    joblib.dump(reg, os.path.join(output_dir, 'payout_regressor.joblib'))
    joblib.dump(expected_cols, os.path.join(output_dir, 'model_features.joblib'))

    print("✓ Model artifacts successfully generated in backend/classifier/")

if __name__ == "__main__":
    generate_models()
