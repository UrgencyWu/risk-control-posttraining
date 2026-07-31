"""
B1: Logistic Regression Baseline
Uses raw 20 features from normalized JSONL:
  - Categorical features → One-Hot encoding
  - Numerical features → StandardScaler
  - Model: LogisticRegression
Threshold selected on valid set, evaluated on test set.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import (
    load_ground_truth, compute_metrics, save_predictions, generate_metrics_table,
    select_cost_threshold,
)

NORMALIZED_DIR = "data/processed/german/normalized"
OUTPUT_DIR = "outputs/baselines"

# Numerical feature indices (from German Credit schema)
NUMERICAL_FEATURES = [
    "duration_month", "credit_amount", "installment_rate",
    "residence_since", "age_years", "existing_credits_count",
    "maintenance_liability_count"
]

# Categorical feature indices
CATEGORICAL_FEATURES = [
    "checking_account_status", "credit_history", "purpose",
    "savings_account", "employment_since", "personal_status_sex",
    "other_debtors", "property", "other_installment_plans",
    "housing", "job", "telephone", "foreign_worker"
]


def extract_feature_matrix(records):
    """Extract numerical and categorical features from normalized records."""
    num_data = []
    cat_data = []
    for r in records:
        f = r["features"]
        num_row = [float(f[name]) for name in NUMERICAL_FEATURES]
        cat_row = [f[name] for name in CATEGORICAL_FEATURES]
        num_data.append(num_row)
        cat_data.append(cat_row)
    return np.array(num_data), np.array(cat_data)


def run_logistic_regression():
    """B1: Logistic Regression with One-Hot + StandardScaler."""
    print("  Loading data...")
    train_records = load_ground_truth(f"{NORMALIZED_DIR}/train.jsonl")
    valid_records = load_ground_truth(f"{NORMALIZED_DIR}/valid.jsonl")
    test_records = load_ground_truth(f"{NORMALIZED_DIR}/test.jsonl")

    y_train = np.array([r["risk_label"] for r in train_records])
    y_valid = np.array([r["risk_label"] for r in valid_records])
    y_test = np.array([r["risk_label"] for r in test_records])

    X_train_num, X_train_cat = extract_feature_matrix(train_records)
    X_valid_num, X_valid_cat = extract_feature_matrix(valid_records)
    X_test_num, X_test_cat = extract_feature_matrix(test_records)

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), list(range(len(NUMERICAL_FEATURES)))),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(range(len(NUMERICAL_FEATURES), len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)))),
    ])

    X_train = np.hstack([X_train_num, X_train_cat])
    X_valid = np.hstack([X_valid_num, X_valid_cat])
    X_test = np.hstack([X_test_num, X_test_cat])

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=10000, random_state=10086)),
    ])

    print("  Training Logistic Regression...")
    pipe.fit(X_train, y_train)

    # Get probability scores
    valid_scores = pipe.predict_proba(X_valid)[:, 1]
    test_scores = pipe.predict_proba(X_test)[:, 1]

    # Select threshold on valid set
    threshold, valid_cost = select_cost_threshold(valid_scores, y_valid)
    print(f"  Optimal threshold (valid): {threshold:.2f}, cost={valid_cost}")

    results = {}

    # Valid set predictions
    valid_preds = (valid_scores >= threshold).astype(int)
    valid_pred_records = []
    for r, pred, score in zip(valid_records, valid_preds, valid_scores):
        error_type = None
        if r["risk_label"] == 1 and pred == 0:
            error_type = "false_negative"
        elif r["risk_label"] == 0 and pred == 1:
            error_type = "false_positive"
        valid_pred_records.append({
            "sample_id": r["sample_id"],
            "ground_truth": r["risk_label"],
            "prediction": int(pred),
            "risk_score": round(float(score), 6),
            "threshold": round(threshold, 2),
            "error_type": error_type,
            "cost": 5 if error_type == "false_negative" else (1 if error_type == "false_positive" else 0),
            "model": "LogisticRegression",
        })
    save_predictions(valid_pred_records, f"{OUTPUT_DIR}/logistic_regression_valid.jsonl")

    # Test set predictions
    test_preds = (test_scores >= threshold).astype(int)
    test_pred_records = []
    for r, pred, score in zip(test_records, test_preds, test_scores):
        error_type = None
        if r["risk_label"] == 1 and pred == 0:
            error_type = "false_negative"
        elif r["risk_label"] == 0 and pred == 1:
            error_type = "false_positive"
        test_pred_records.append({
            "sample_id": r["sample_id"],
            "ground_truth": r["risk_label"],
            "prediction": int(pred),
            "risk_score": round(float(score), 6),
            "threshold": round(threshold, 2),
            "error_type": error_type,
            "cost": 5 if error_type == "false_negative" else (1 if error_type == "false_positive" else 0),
            "model": "LogisticRegression",
        })
    save_predictions(test_pred_records, f"{OUTPUT_DIR}/logistic_regression_test.jsonl")

    test_metrics = compute_metrics(y_test, test_preds, test_scores)
    results["LogisticRegression"] = test_metrics
    print(f"  test: acc={test_metrics['accuracy']:.4f}, balanced_acc={test_metrics['balanced_accuracy']:.4f}, "
          f"high_risk_recall={test_metrics['high_risk_recall']:.4f}, "
          f"roc_auc={test_metrics['roc_auc']}, cost={test_metrics['cost']}")

    return results


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== B1: Logistic Regression Baseline ===")
    results = run_logistic_regression()
    print("\n" + generate_metrics_table(results))
