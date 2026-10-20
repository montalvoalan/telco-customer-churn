"""Transform customer records and predict with the bundled training artifacts."""

import json
import os
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier


LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR = Path(os.environ.get("TELCO_MODEL_DIR", LOCAL_MODEL_DIR))

model = XGBClassifier()
model.load_model(MODEL_DIR / "model.json")
FEATURE_COLS = (MODEL_DIR / "feature_columns.txt").read_text(encoding="utf-8").splitlines()
CATEGORY_LEVELS = json.loads((MODEL_DIR / "category_levels.json").read_text(encoding="utf-8"))
THRESHOLD = float((MODEL_DIR / "threshold.txt").read_text(encoding="utf-8"))

BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Encode records with the categories and feature order saved during training."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    features = pd.DataFrame(0.0, index=df.index, columns=FEATURE_COLS)

    for column in NUMERIC_COLS:
        if column not in df:
            raise ValueError(f"Missing feature: {column}")
        features[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    for column, mapping in BINARY_MAP.items():
        if column not in df:
            raise ValueError(f"Missing feature: {column}")
        encoded = df[column].astype(str).str.strip().map(mapping)
        if encoded.isna().any():
            raise ValueError(f"Unexpected value for {column}")
        features[column] = encoded

    for column, levels in CATEGORY_LEVELS.items():
        if column not in df:
            raise ValueError(f"Missing feature: {column}")
        values = df[column].astype(str).str.strip()
        if not values.isin(levels).all():
            raise ValueError(f"Unexpected value for {column}")
        for level in levels[1:]:
            feature_name = f"{column}_{level}"
            if feature_name in features:
                features[feature_name] = values.eq(level).astype(int)

    return features


def predict(input_dict: dict) -> str:
    features = _serve_transform(pd.DataFrame([input_dict]))
    probability = float(model.predict_proba(features)[0, 1])
    return "Likely to churn" if probability >= THRESHOLD else "Not likely to churn"
