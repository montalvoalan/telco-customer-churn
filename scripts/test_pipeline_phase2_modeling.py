"""Exercise the reusable tuning, training, and evaluation modules."""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.evaluate import evaluate_model
from src.models.train import train_model
from src.models.tune import tune_model


DATA_PATH = "data/processed/telco_churn_features.csv"
TARGET = "Churn"
THRESHOLD = 0.35


def main():
    print("=== Phase 2: Tune, train, and evaluate XGBoost ===")
    df = pd.read_csv(DATA_PATH)
    assert df[TARGET].isna().sum() == 0, "Churn has missing values"
    assert set(df[TARGET].unique()) <= {0, 1}, "Churn is not encoded as 0/1"

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    best_params = tune_model(
        X_train, y_train, threshold=THRESHOLD, n_trials=5
    )
    model = train_model(X_train, y_train, params=best_params)
    results = evaluate_model(model, X_test, y_test, threshold=THRESHOLD)

    print(f"Test precision: {results['precision']:.3f}")
    print(f"Test recall: {results['recall']:.3f}")
    print(f"Test F1: {results['f1']:.3f}")
    print(f"Test ROC AUC: {results['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
