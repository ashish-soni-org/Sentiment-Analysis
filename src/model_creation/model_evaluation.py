import os
import sys
import json
import pickle
import numpy as np
import mlflow
import dagshub
from from_root import from_root
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.logger import logging
from src.constants import DAGSHUB_TOKEN_ENV_KEY
from src.exception import CustomException


def setup_mlflow_tracking() -> None:
    """Configure MLflow tracking with DagsHub."""
    try:
        dagshub_token = os.getenv(DAGSHUB_TOKEN_ENV_KEY)
        if not dagshub_token:
            raise EnvironmentError(f"{DAGSHUB_TOKEN_ENV_KEY} environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "ashishsoni295work" 
        repo_name = "Sentiment-Analysis"

        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        logging.info("MLflow tracking configured with DagsHub")

    except Exception as e:
        raise CustomException(e, sys) from e


def load_model(file_path: str):
    """Load model from disk."""
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        logging.info(f"Model loaded from: {file_path}")
        return model
    except Exception as e:
        raise CustomException(e, sys) from e


def load_processed_data(file_path: str):
    """
    Load sparse matrix test data from pickle.
    """
    try:
        with open(file_path, "rb") as f:
            X, y = pickle.load(f)
        logging.info(f"Test data loaded from: {file_path}")
        return X, y
    except Exception as e:
        raise CustomException(e, sys) from e


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute classification metrics."""
    try:
        y_pred = model.predict(X_test)
        
        # Check if model supports probability prediction
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
        else:
            # For models like SVM without probability enabled by default
            try:
                y_pred_proba = model.decision_function(X_test)
                auc_score = roc_auc_score(y_test, y_pred_proba)
            except:
                auc_score = 0.0

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": auc_score
        }

        logging.info("Model evaluation completed successfully")
        return metrics

    except Exception as e:
        raise CustomException(e, sys) from e


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save metrics to JSON."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logging.info(f"Metrics saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save MLflow run info."""
    try:
        model_info = {"run_id": run_id, "model_path": model_path}
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            json.dump(model_info, file, indent=4)
        logging.info(f"Model run info saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e


def main() -> None:
    """Execute evaluation pipeline."""
    logging.info("Starting model evaluation")
    try:
        setup_mlflow_tracking()
        mlflow.set_experiment("dvc-pipeline")

        model_path = os.path.join(from_root(), "models", "model.pkl")
        
        # CRITICAL FIX: Loading the .pkl file instead of the old .csv
        test_data_path = os.path.join(from_root(), "data", "processed", "test_data.pkl")
        
        model = load_model(model_path)
        X_test, y_test = load_processed_data(test_data_path)

        metrics = evaluate_model(model, X_test, y_test)

        reports_dir = os.path.join(from_root(), "reports")
        metrics_path = os.path.join(reports_dir, "metrics.json")
        info_path = os.path.join(reports_dir, "experiment_info.json")

        save_metrics(metrics, metrics_path)

        with mlflow.start_run() as run:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            if hasattr(model, "get_params"):
                for param, value in model.get_params().items():
                    mlflow.log_param(param, value)

            mlflow.sklearn.log_model(model, artifact_path="model")

            save_model_info(
                run_id=run.info.run_id,
                model_path="model",
                file_path=info_path
            )

            mlflow.log_artifact(metrics_path)

        logging.info("Model evaluation completed successfully")

    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()