"""Threshold-aware evaluation for churn classifiers."""

from typing import Any

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.35,
) -> dict[str, Any]:
    """Evaluate probabilities at the same threshold used by the API."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "classification_report": classification_report(
            y_test, predictions, digits=3, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "predictions": predictions,
        "probabilities": probabilities,
    }
