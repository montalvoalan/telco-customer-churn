# Telco Customer Churn Prediction

This project predicts whether a telecom customer is likely to churn. It starts with exploratory analysis of the [IBM Telco Customer Churn sample](https://www.ibm.com/docs/en/cognos-analytics/12.1.x?topic=samples-telco-customer-churn), trains an XGBoost classifier, and serves predictions through a FastAPI endpoint and a Gradio form.

## What is in the project

- `notebooks/EDA.ipynb`: data exploration and initial model comparisons.
- `src/data`, `src/features`, `src/utils`: data loading, cleaning, feature encoding, and Great Expectations checks.
- `scripts/run_pipeline.py`: coordinates validation, optional tuning, training, holdout evaluation, MLflow logging, and serving model export.
- `src/models/`: reusable tuning, training, and threshold-aware evaluation functions used by the pipeline.
- `src/serving/inference.py`: loads the exported model, training categories, feature order, and decision threshold.
- `src/app/main.py`: `/` health check, `/predict` API, and `/ui` Gradio form.
- `dockerfile` and `.github/workflows/ci.yml`: container build and Docker Hub publishing configuration.

## Run locally

Run these PowerShell commands from the repository root with a working Python environment. The training CSV must be at `data/raw/Telco-Customer-Churn.csv`; raw data is excluded from Git.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
.\.venv\Scripts\python.exe scripts/test_fast_api.py
.\.venv\Scripts\python.exe -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000
```

Add `--tune --n_trials 20` to the training command when you want Optuna to select model parameters using cross-validation on the training split. The untouched test split is evaluated once after tuning. Running without `--tune` uses the saved project defaults and is much faster.

Open `http://127.0.0.1:8000/ui` for the form or `http://127.0.0.1:8000/docs` for the API documentation. Training writes experiment records to the local `mlflow.db` and exports `model.json`, category levels, feature columns, and the threshold to `src/serving/model/`. The API requires the same 19 customer fields as the dataset, excluding `customerID` and `Churn`.

Optional checks:

```powershell
.\.venv\Scripts\python.exe scripts/test_pipeline_phase1_data_features.py
.\.venv\Scripts\python.exe scripts/prepare_processed_data.py
.\.venv\Scripts\python.exe scripts/test_pipeline_phase2_modeling.py
```

The phase 2 script runs a separate Optuna experiment. It does not replace the model exported by `run_pipeline.py`.

## Current local result

On the local 7,043-row CSV, the training script used a stratified 80/20 split and a 0.35 decision threshold. Its most recent local holdout results were precision **0.486**, recall **0.826**, F1 **0.612**, and ROC AUC **0.835** for the churn class. These numbers describe this sample and split; they are not a guarantee of performance on new customers. The threshold favors recall, so it flags more customers who will not churn.

## Deployment status

The Docker image uses a smaller pinned serving dependency file and includes the four files in `src/serving/model/`. On every push to `main`, GitHub Actions builds the image, starts a container, checks the health and prediction endpoints, and publishes the tested image as `montalvoalan/telco-fastapi:latest`. An AWS ECS Fargate service behind an Application Load Balancer is the intended deployment design. The local pipeline and in-process API checks have run; the GitHub Actions Docker build and AWS deployment still require external verification.
