import os
import sys
import pickle
import numpy as np
from from_root import from_root
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomException


def load_processed_data(file_path: str):
    """Load sparse matrix data from pickle."""
    try:
        with open(file_path, "rb") as f:
            X, y = pickle.load(f)
        logging.info(f"Data loaded from: {file_path}")
        return X, y
    except Exception as e:
        raise CustomException(e, sys) from e


def train_model(X_train, y_train) -> SGDClassifier:
    """
    Train a Support Vector Machine (via SGD) for optimal text classification.
    """
    try:
        logging.info("Starting model training with GridSearchCV")

        # SGDClassifier with hinge loss = Linear SVM (Great for text)
        # modified_huber = Smooth SVM that supports probabilities
        base_model = SGDClassifier(
            loss='modified_huber', 
            random_state=42,
            max_iter=2000,
            class_weight='balanced',
            n_jobs=-1
        )

        param_grid = {
            'alpha': [1e-4, 1e-3, 1e-2], # Regularization strength
            'penalty': ['l2']
        }

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        
        logging.info(f"Best Params: {grid_search.best_params_}")
        logging.info(f"Best CV Score: {grid_search.best_score_:.4f}")

        return best_model

    except Exception as e:
        raise CustomException(e, sys) from e


def save_model(model, file_path: str) -> None:
    """Persist model to disk."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(model, file)

        logging.info(f"Model artifact saved at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys) from e


def main() -> None:
    """Execute model training pipeline."""
    logging.info("Starting model training pipeline")
    try:
        processed_path = os.path.join(from_root(), "data", "processed")
        
        # Load Sparse Matrices
        X_train, y_train = load_processed_data(os.path.join(processed_path, "train_data.pkl"))

        trained_model = train_model(X_train, y_train)

        model_output_path = os.path.join(from_root(), "models", "model.pkl")
        save_model(model=trained_model, file_path=model_output_path)

        logging.info("Model training completed successfully")

    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    main()