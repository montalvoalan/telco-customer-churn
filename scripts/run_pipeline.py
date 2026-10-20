"""Run data validation, feature engineering, training, evaluation, and export."""

import argparse
import json
import os
import sys
import time

import joblib
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.evaluate import evaluate_model
from src.models.train import DEFAULT_MODEL_PARAMS, positive_class_weight, train_model
from src.models.tune import tune_model
from src.utils.utils import default_mlflow_uri, get_project_root
from src.utils.validate_data import validate_telco_data


def export_serving_bundle(model, cleaned_df, feature_columns, threshold, serving_dir):
    """Save one self-contained model version for local and Docker serving."""
    serving_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(serving_dir / "model.json")
    (serving_dir / "feature_columns.txt").write_text(
        "\n".join(feature_columns) + "\n", encoding="utf-8"
    )
    category_levels = {
        column: sorted(cleaned_df[column].dropna().unique().tolist())
        for column in cleaned_df.select_dtypes(include=["object", "string"]).columns
        if cleaned_df[column].dropna().nunique() > 2
    }
    (serving_dir / "category_levels.json").write_text(
        json.dumps(category_levels, indent=2), encoding="utf-8"
    )
    (serving_dir / "threshold.txt").write_text(str(threshold), encoding="utf-8")


def main(args):
    project_root = get_project_root()
    mlflow.set_tracking_uri(args.mlflow_uri or default_mlflow_uri(project_root))
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": "xgboost",
                "threshold": args.threshold,
                "test_size": args.test_size,
                "tuning_enabled": args.tune,
                "tuning_trials": args.n_trials if args.tune else 0,
            }
        )

        print("Loading data...")
        raw_df = load_data(args.input)
        print(f"Data loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

        is_valid, failures = validate_telco_data(raw_df)
        mlflow.log_metric("data_quality_pass", int(is_valid))
        if not is_valid:
            mlflow.log_text(
                json.dumps(failures, indent=2), artifact_file="failed_expectations.json"
            )
            raise ValueError(f"Data quality checks failed: {failures}")

        cleaned_df = preprocess_data(raw_df, target_col=args.target)
        processed_path = project_root / "data" / "processed" / "telco_churn_processed.csv"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(processed_path, index=False)
        print(f"Cleaned data saved to {processed_path}")

        if args.target not in cleaned_df.columns:
            raise ValueError(f"Target column '{args.target}' not found in data")
        encoded_df = build_features(cleaned_df, target_col=args.target)
        X = encoded_df.drop(columns=[args.target])
        y = encoded_df[args.target]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            stratify=y,
            random_state=42,
        )
        print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

        selected_params = DEFAULT_MODEL_PARAMS.copy()
        if args.tune:
            print(f"Tuning on training data with {args.n_trials} Optuna trials...")
            selected_params.update(
                tune_model(
                    X_train,
                    y_train,
                    threshold=args.threshold,
                    n_trials=args.n_trials,
                )
            )
        mlflow.log_params({f"model_{key}": value for key, value in selected_params.items()})
        mlflow.log_param("scale_pos_weight", positive_class_weight(y_train))

        print("Training final model...")
        train_start = time.perf_counter()
        model = train_model(X_train, y_train, params=selected_params)
        train_time = time.perf_counter() - train_start

        prediction_start = time.perf_counter()
        results = evaluate_model(model, X_test, y_test, threshold=args.threshold)
        prediction_time = time.perf_counter() - prediction_start
        metrics = {
            name: results[name] for name in ("precision", "recall", "f1", "roc_auc")
        }
        metrics.update({"train_time": train_time, "pred_time": prediction_time})
        mlflow.log_metrics(metrics)

        print(
            f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | "
            f"F1: {metrics['f1']:.3f} | ROC AUC: {metrics['roc_auc']:.3f}"
        )
        print(results["classification_report"])
        print("Confusion matrix:")
        print(results["confusion_matrix"])

        artifacts_dir = project_root / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        feature_columns = list(X.columns)
        (artifacts_dir / "feature_columns.json").write_text(
            json.dumps(feature_columns, indent=2), encoding="utf-8"
        )
        preprocessing_path = artifacts_dir / "preprocessing.pkl"
        joblib.dump(
            {"feature_columns": feature_columns, "target": args.target},
            preprocessing_path,
        )
        mlflow.log_artifact(str(preprocessing_path))
        mlflow.log_text("\n".join(feature_columns), artifact_file="feature_columns.txt")
        mlflow.xgboost.log_model(model, name="model")

        serving_dir = project_root / "src" / "serving" / "model"
        export_serving_bundle(
            model, cleaned_df, feature_columns, args.threshold, serving_dir
        )
        print(f"Serving bundle saved to {serving_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and export the churn model")
    parser.add_argument("--input", required=True, help="path to the raw Telco CSV")
    parser.add_argument("--target", default="Churn")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--experiment", default="Telco Churn")
    parser.add_argument("--mlflow_uri", default=None)
    parser.add_argument(
        "--tune",
        action="store_true",
        help="run Optuna on training data before fitting the final model",
    )
    parser.add_argument("--n_trials", type=int, default=20)
    main(parser.parse_args())
