"""Hyperparameter tuning on training data only."""

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold

from src.models.train import train_model


def tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.35,
    n_trials: int = 20,
) -> dict[str, Any]:
    """Choose XGBoost parameters by mean cross-validated churn recall."""
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        recalls = []
        for train_index, validation_index in folds.split(X, y):
            model = train_model(
                X.iloc[train_index], y.iloc[train_index], params=params
            )
            probability = model.predict_proba(X.iloc[validation_index])[:, 1]
            prediction = (probability >= threshold).astype(int)
            recalls.append(recall_score(y.iloc[validation_index], prediction))
        return float(np.mean(recalls))

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    print(f"Best cross-validated recall: {study.best_value:.3f}")
    print(f"Best parameters: {study.best_params}")
    return study.best_params
