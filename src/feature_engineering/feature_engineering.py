import os
import sys
import pickle
import yaml
import pandas as pd
from from_root import from_root
from sklearn.feature_extraction.text import TfidfVectorizer

from src.logger import logging
from src.exception import CustomException


def load_params(params_path: str) -> dict:
    """Load configuration parameters."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from: {params_path}")
        return params
    except Exception as e:
        raise CustomException(e, sys) from e


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV dataset."""
    try:
        dataframe = pd.read_csv(file_path)
        dataframe.fillna("", inplace=True)
        logging.info(f"Data loaded from: {file_path}")
        return dataframe
    except Exception as e:
        raise CustomException(e, sys) from e


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, ngram_range: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply TF-IDF with optimized settings for maximum context capture.
    """
    try:
        logging.info("Starting TF-IDF feature extraction")

        ngram_range_tuple = tuple(ngram_range)

        # Removed max_features limit to capture all nuances
        # Increased min_df slightly to filter absolute noise
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range_tuple,
            sublinear_tf=True,
            strip_accents='unicode',
            min_df=3 
        )

        X_train = train_data["review"].astype(str).values
        y_train = train_data["sentiment"].values

        X_test = test_data["review"].astype(str).values
        y_test = test_data["sentiment"].values

        # Fit & Transform
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Use Sparse Matrix saving to handle large feature set efficiently
        # We save directly to avoid the memory overhead of converting to Dense DataFrame
        
        # Save Vectorizer
        model_dir = os.path.join(from_root(), "models")
        os.makedirs(model_dir, exist_ok=True)

        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        with open(vectorizer_path, "wb") as file:
            pickle.dump(vectorizer, file)
            
        logging.info(f"TF-IDF applied. Vectorizer saved at: {vectorizer_path}")

        # Return sparse matrices directly (Requires updating model_building to accept sparse)
        # But for compatibility with your existing structure, we will use a safe subset if sparse is too big,
        # OR we convert to dense if RAM allows. 
        # Given standard dev machines, let's keep it robust:
        
        # If dataset is huge, we should switch to sparse. For IMDB (50k rows), it fits in RAM.
        return X_train_tfidf, y_train, X_test_tfidf, y_test

    except Exception as e:
        raise CustomException(e, sys) from e

# NOTE: I updated the return type to be more memory efficient (Sparse Matrix)
# This requires a tiny update to main() below and model_building.py

def main() -> None:
    """Execute feature engineering pipeline."""
    logging.info("Starting feature engineering")
    try:
        params = load_params(params_path=os.path.join(from_root(), "params.yaml"))
        ngram_range = params["feature_engineering"]["ngram_range"]

        data_interim_path = os.path.join(from_root(), "data", "interim")
        
        train_data = load_data(os.path.join(data_interim_path, "train_processed.csv"))
        test_data = load_data(os.path.join(data_interim_path, "test_processed.csv"))

        X_train, y_train, X_test, y_test = apply_tfidf(
            train_data=train_data,
            test_data=test_data,
            ngram_range=ngram_range
        )

        # Save Processed Data as pickle (Efficient for Sparse Matrices)
        processed_path = os.path.join(from_root(), "data", "processed")
        os.makedirs(processed_path, exist_ok=True)
        
        with open(os.path.join(processed_path, "train_data.pkl"), "wb") as f:
            pickle.dump((X_train, y_train), f)
            
        with open(os.path.join(processed_path, "test_data.pkl"), "wb") as f:
            pickle.dump((X_test, y_test), f)

        logging.info("Feature engineering completed successfully")

    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    main()