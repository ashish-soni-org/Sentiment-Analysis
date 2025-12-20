import os
import sys
import re
import time
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import mlflow
import dagshub

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from prometheus_client import (
    Counter,
    Histogram,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.constants import DAGSHUB_TOKEN
from src.exception import CustomException


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

ml_objects: dict = {}

SENTIMENT_MAP = {
    0: "Positive",
    1: "Negative"
}

TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------
# MLflow Setup
# ---------------------------------------------------------------------

def setup_mlflow_tracking() -> None:
    """
    Configure MLflow tracking with DagsHub.
    """
    try:
        dagshub_token = os.getenv(DAGSHUB_TOKEN)
        if not dagshub_token:
            raise EnvironmentError(
                "DAGSHUB_TOKEN environment variable is not set"
            )

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "ashishsoni295work"
        repo_name = "Sentiment-Analysis"

        mlflow.set_tracking_uri(
            f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
        )

    except Exception as e:
        raise CustomException(e, sys) from e


def load_model_and_vectorizer() -> None:
    """
    Load MLflow model and vectorizer into memory.
    """
    try:
        setup_mlflow_tracking()

        model_name = "sentiment_analysis_model"
        client = mlflow.MlflowClient()

        versions = (
            client.get_latest_versions(model_name, stages=["Production"])
            or client.get_latest_versions(model_name, stages=["None"])
        )

        if not versions:
            raise RuntimeError("No registered model versions found")

        model_uri = f"models:/{model_name}/{versions[0].version}"
        ml_objects["model"] = mlflow.pyfunc.load_model(model_uri)

        vectorizer_path = BASE_DIR / "models" / "vectorizer.pkl"
        with open(vectorizer_path, "rb") as file:
            ml_objects["vectorizer"] = pickle.load(file)

    except Exception as e:
        raise CustomException(e, sys) from e


# ---------------------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize input text for inference.
    """
    try:
        if not text:
            return ""

        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", "", text)

        words = [
            lemmatizer.lemmatize(word)
            for word in text.split()
            if word not in stop_words
        ]

        return " ".join(words)

    except Exception as e:
        raise CustomException(e, sys) from e


# ---------------------------------------------------------------------
# FastAPI App & Metrics
# ---------------------------------------------------------------------

app = FastAPI(title="Sentiment Analysis")

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total requests",
    ["method", "endpoint"],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Request latency",
    ["endpoint"],
    registry=registry
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Prediction count",
    ["prediction"],
    registry=registry
)


@app.on_event("startup")
def startup_event() -> None:
    """
    Load model artifacts before serving requests.
    """
    load_model_and_vectorizer()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()

    response = templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None}
    )

    REQUEST_LATENCY.labels(endpoint="/").observe(
        time.time() - start_time
    )
    return response


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    text: str = Form(...)
):
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    clean_text = normalize_text(text)

    vectorizer = ml_objects["vectorizer"]
    model = ml_objects["model"]

    features = vectorizer.transform([clean_text])
    features_df = pd.DataFrame(
        features.toarray(),
        columns=[str(i) for i in range(features.shape[1])]
    )

    raw_prediction = int(model.predict(features_df)[0])
    prediction_label = SENTIMENT_MAP.get(raw_prediction, "Unknown")

    PREDICTION_COUNT.labels(prediction=prediction_label).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(
        time.time() - start_time
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": prediction_label
        }
    )


@app.get("/metrics")
def metrics():
    """
    Expose Prometheus metrics.
    """
    return Response(
        generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )


# ---------------------------------------------------------------------
# Entry point (local run)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv() # To load variables from .env file

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=False
    )
