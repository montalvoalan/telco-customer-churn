import os
import sys
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.app.main import app

sample_data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 350.75
}

client = TestClient(app)
assert client.get("/").json() == {"status": "ok"}
assert client.get("/ui").status_code == 200
response = client.post("/predict", json=sample_data)
print("Status Code:", response.status_code)
print("Response:", response.json())
assert response.status_code == 200
assert response.json()["prediction"] in {"Likely to churn", "Not likely to churn"}
