import os
import sys
import pickle

import yaml
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.logger import logging
from src.exception import CustomException


def load_params(params_path: str) -> dict:
    """
    Load configuration parameters from a YAML file.

    Parameters
    ----------
    params_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Dictionary containing configuration parameters.
    """
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        logging.info(f"Parameters loaded from: {params_path}")
        return params

    except Exception as e:
        raise CustomException(e, sys) from e


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset from disk.

    This method reads a CSV file and fills missing values
    with empty strings for text processing.

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
        dataframe.fillna("", inplace=True)

        logging.info(f"Data loaded from: {file_path}")
        return dataframe

    except Exception as e:
        raise CustomException(e, sys) from e


def apply_bag_of_words(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply Bag-of-Words feature extraction on text data.

    This method:
    - Fits a CountVectorizer on training data
    - Transforms both train and test datasets
    - Appends labels to the transformed features
    - Persists the fitted vectorizer for downstream use

    Parameters
    ----------
    train_data : DataFrame
        Training dataset containing text and labels.
    test_data : DataFrame
        Testing dataset containing text and labels.
    max_features : int
        Maximum number of features for the vectorizer.

    Returns
    -------
    tuple[DataFrame, DataFrame]
        Transformed training and testing datasets.
    """
    try:
        logging.info("Starting Bag-of-Words feature extraction")

        vectorizer = CountVectorizer(max_features=max_features)

        X_train = train_data["review"].values
        y_train = train_data["sentiment"].values

        X_test = test_data["review"].values
        y_test = test_data["sentiment"].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df["label"] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df["label"] = y_test

        model_dir = os.path.join("models")
        os.makedirs(model_dir, exist_ok=True)

        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        with open(vectorizer_path, "wb") as file:
            pickle.dump(vectorizer, file)

        logging.info(
            f"Bag-of-Words applied successfully. "
            f"Vectorizer saved at: {vectorizer_path}"
        )

        return train_df, test_df

    except Exception as e:
        raise CustomException(e, sys) from e


def save_data(dataframe: pd.DataFrame, file_path: str) -> None:
    """
    Save a DataFrame to disk as a CSV file.

    Parameters
    ----------
    dataframe : DataFrame
        Dataset to be saved.
    file_path : str
        Destination file path.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        dataframe.to_csv(file_path, index=False, header=True)

        logging.info(f"Data saved at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys) from e


def main() -> None:
    """
    Execute the feature engineering workflow.

    This function orchestrates:
    - Loading configuration parameters
    - Loading preprocessed datasets
    - Applying Bag-of-Words feature extraction
    - Persisting transformed datasets
    """
    logging.info("Starting feature engineering")
    try:
        params = load_params(params_path= "params.yaml")
        max_features = params["feature_engineering"]["max_features"]

        train_data = load_data("./data/interim/train_processed.csv")
        test_data = load_data("./data/interim/test_processed.csv")

        train_bow_df, test_bow_df = apply_bag_of_words(
            train_data= train_data,
            test_data= test_data,
            max_features= max_features
        )

        save_data(train_bow_df, os.path.join("./data", "processed", "train_bow.csv"))
        save_data(test_bow_df, os.path.join("./data", "processed", "test_bow.csv"))

        logging.info("Feature engineering completed successfully")

    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    main()
