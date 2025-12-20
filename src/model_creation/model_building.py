import os
import sys
import pickle

import numpy as np
import pandas as pd
from from_root import from_root
from sklearn.linear_model import LogisticRegression

from src.logger import logging
from src.exception import CustomException


def load_data(file_path: str) -> pd.DataFrame:
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
        logging.info(f"Training data loaded from: {file_path}")
        return dataframe

    except Exception as e:
        raise CustomException(e, sys) from e


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> LogisticRegression:
    """
    Train a Logistic Regression model.

    This method initializes a Logistic Regression classifier
    with predefined hyperparameters and fits it on the
    training dataset.

    Parameters
    ----------
    X_train : np.ndarray
        Feature matrix for training.
    y_train : np.ndarray
        Target labels for training.

    Returns
    -------
    LogisticRegression
        Trained Logistic Regression model.
    """
    try:
        logging.info("Starting model training")

        model = LogisticRegression(
            C=1.0,
            solver="liblinear",
            penalty="l1",
            random_state=42
        )

        model.fit(X_train, y_train)

        logging.info("Model training completed successfully")
        return model

    except Exception as e:
        raise CustomException(e, sys) from e


def save_model(model: LogisticRegression, file_path: str) -> None:
    """
    Persist the trained model to disk.

    Parameters
    ----------
    model : LogisticRegression
        Trained model to be saved.
    file_path : str
        Destination path for the model artifact.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(model, file)

        logging.info(f"Model artifact saved at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys) from e


def main() -> None:
    """
    Execute the model training.

    This function orchestrates:
    - Loading processed training data
    - Separating features and labels
    - Training the classification model
    - Persisting the trained model artifact
    """
    logging.info("Starting model training pipeline")
    try:
        
        train_data_path = os.path.join(
            from_root(),
            "data",
            "processed",
            "train_bow.csv"
        )

        model_output_path = os.path.join(
            from_root(),
            "models",
            "model.pkl"
        )

        train_data = load_data(train_data_path)

        X_train = train_data.iloc[:, :-1].to_numpy()
        y_train = train_data.iloc[:, -1].to_numpy()

        trained_model = train_model(
            X_train=X_train,
            y_train=y_train
        )

        save_model(
            model=trained_model,
            file_path=model_output_path
        )

        logging.info("Model training completed successfully")

    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    main()
