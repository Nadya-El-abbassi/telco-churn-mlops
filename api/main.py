import os
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import pandas as pd

from elasticsearch import Elasticsearch
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
from pathlib import Path
from fastapi.responses import FileResponse

app = FastAPI(
    title="Telco Customer Churn API",
    description="API for customer churn prediction",
    version="2.0.0"
)
STATIC_DIR = Path(__file__).resolve().parent / "static"

@app.get("/ui", include_in_schema=False)
def user_interface():
    return FileResponse(STATIC_DIR / "index.html")


# MLflow Tracking Server
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://host.docker.internal:5000"
)

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "TelcoChurnModel")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load the model through the MLflow Model Registry
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

client = MlflowClient()

champion_version = client.get_model_version_by_alias(
    MODEL_NAME,
    MODEL_ALIAS
)

model_run = client.get_run(champion_version.run_id)

MODEL_FAMILY = (
    model_run.data.params.get("model_family")
    or model_run.data.params.get("selected_model")
    or MODEL_NAME
)

MODEL_VERSION = str(champion_version.version)

model = mlflow.sklearn.load_model(MODEL_URI)

print(
    f"Loaded {MODEL_FAMILY}, "
    f"{MODEL_NAME}@{MODEL_ALIAS}, "
    f"version {MODEL_VERSION}"
)


# Elasticsearch connection
ES_URL = os.getenv(
    "ELASTICSEARCH_URL",
    "http://elasticsearch:9200"
)

es = Elasticsearch(ES_URL)
INDEX_NAME = "churn-predictions"


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Telco Customer Churn API is running",
        "model": MODEL_FAMILY,
        "model_version": MODEL_VERSION
    }


@app.get("/health")
def health():
    try:
        elasticsearch_connected = bool(es.ping())
    except Exception:
        elasticsearch_connected = False

    return {
        "status": "healthy",
        "model": MODEL_FAMILY,
        "model_registry": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": MODEL_VERSION,
        "elasticsearch_connected": elasticsearch_connected
    }


@app.post("/predict")
def predict(data: CustomerData):
    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    es.index(
        index=INDEX_NAME,
        document={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction": int(prediction),
            "churn_probability": float(probability),
            "model": MODEL_FAMILY,
            "model_registry": MODEL_NAME,
            "model_alias": MODEL_ALIAS,
            "model_version": MODEL_VERSION
        }
    )

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "model": MODEL_FAMILY,
        "model_version": MODEL_VERSION
    }