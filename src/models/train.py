"""Model construction and training for the churn pipeline."""

from typing import Any

import pandas as pd
from xgboost import XGBClassifier


DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 301,
    "learning_rate": 0.034,
    "max_depth": 7,
    "subsample": 0.95,
    "colsample_bytree": 0.98,
    "n_jobs": -1,
    "random_state": 42,
    "eval_metric": "logloss",
}


def positive_class_weight(y: pd.Series) -> float:
    """Return the non-churner to churner ratio used by XGBoost."""
    negatives = int((y == 0).sum())
    positives = int((y == 1).sum())
    if positives == 0:
        raise ValueError("Training data has no positive churn examples")
    return negatives / positives


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any] | None = None,
) -> XGBClassifier:
    """Fit an XGBoost classifier using shared defaults and class weighting."""
    model_params = DEFAULT_MODEL_PARAMS.copy()
    if params:
        model_params.update(params)
    model_params["scale_pos_weight"] = positive_class_weight(y_train)

    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    return model
