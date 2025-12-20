import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

import mlflow
import dagshub

from src.logger import logging
from src.constants import DAGSHUB_TOKEN
from src.exception import CustomException


def setup_mlflow_tracking() -> None:
    """
    Configure MLflow tracking with DagsHub.

    This method validates required environment variables and
    sets the MLflow tracking URI for experiment logging.
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

        logging.info("MLflow tracking configured with DagsHub")

    except Exception as e:
        raise CustomException(e, sys) from e


def load_model(file_path: str):
    """
    Load a trained model artifact from disk.

    Parameters
    ----------
    file_path : str
        Path to the serialized model file.

    Returns
    -------
    object
        Loaded model object.
    """
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)

        logging.info(f"Model loaded from: {file_path}")
        return model

    except Exception as e:
        raise CustomException(e, sys) from e


def load_data(file_path: str) -> DataFrame:
    """
    Load a dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    DataFrame
        Loaded dataset.
    """
    try:
        dataframe = pd.read_csv(file_path)
        logging.info(f"Test data loaded from: {file_path}")
        return dataframe

    except Exception as e:
        raise CustomException(e, sys) from e


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Evaluate the trained model on test data.

    This method computes classification metrics including
    accuracy, precision, recall, and ROC-AUC score.

    Parameters
    ----------
    model : object
        Trained classification model.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True labels for test data.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba)
        }

        logging.info("Model evaluation completed successfully")
        return metrics

    except Exception as e:
        raise CustomException(e, sys) from e


def save_metrics(metrics: dict, file_path: str) -> None:
    """
    Save evaluation metrics to disk as a JSON file.

    Parameters
    ----------
    metrics : dict
        Evaluation metrics.
    file_path : str
        Destination file path.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)

        logging.info(f"Metrics saved at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys) from e


def save_model_info(
    run_id: str,
    model_path: str,
    file_path: str
) -> None:
    """
    Save MLflow run metadata to disk.

    Parameters
    ----------
    run_id : str
        MLflow run ID.
    model_path : str
        Logged model path in MLflow.
    file_path : str
        Destination file path.
    """
    try:
        model_info = {
            "run_id": run_id,
            "model_path": model_path
        }

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            json.dump(model_info, file, indent=4)

        logging.info(f"Model run info saved at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys) from e


def main() -> None:
    """
    Execute the model evaluation pipeline.

    This function orchestrates:
    - MLflow and DagsHub setup
    - Model loading
    - Test data loading
    - Model evaluation
    - Metric logging and artifact persistence
    """
    logging.info("Starting model evaluation")
    try:
        setup_mlflow_tracking()

        mlflow.set_experiment("dvc-pipeline")

        with mlflow.start_run() as run:
            model = load_model("./models/model.pkl")
            test_data = load_data("./data/processed/test_bow.csv")

            X_test = test_data.iloc[:, :-1].to_numpy()
            y_test = test_data.iloc[:, -1].to_numpy()

            metrics = evaluate_model(model, X_test, y_test)

            save_metrics(
                metrics,
                file_path="./reports/metrics.json"
            )

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            if hasattr(model, "get_params"):
                for param, value in model.get_params().items():
                    mlflow.log_param(param, value)

            mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="sentiment_analysis_model")

            save_model_info(
                run_id=run.info.run_id,
                model_path="model",
                file_path="./reports/experiment_info.json"
            )

            mlflow.log_artifact("./reports/metrics.json")

            logging.info("Model evaluation completed successfully")

    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":

    from dotenv import load_dotenv
    load_dotenv() # To load variables from .env file
    
    main()