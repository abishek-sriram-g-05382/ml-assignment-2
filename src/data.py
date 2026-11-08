"""

This file is responsible for loading and cleaning the raw UCI Frogs MFCCs dataset, 
as well as split it into training and test sets. 
It also provides a function to save the test set to a CSV file.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = REPO_ROOT / "data" / "Frogs_MFCCs.csv"

TARGET_COL = "Species"
DROP_COLS = ["Family", "Genus", "RecordID"]

# Function to clean the column names
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # the source CSV has inconsistent spacing in MFCC headers
    # ("MFCCs_ 1" vs "MFCCs_10") — normalize so there's one naming scheme
    # used everywhere (training, test_data.csv, and anything a user uploads
    # to the app, including the raw UCI file itself).
    df = df.copy()
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    return df

# Funciton to load raw data from csv
def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    return clean_columns(df)

# Function to get the feature columns
def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("MFCCs_")]

# Function to load features and target from the raw data
def load_features_and_target(path: Path = RAW_CSV):
    df = load_raw(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    cols = feature_columns(df)
    return df[cols], df[TARGET_COL], cols

# Function to split the data into training and test sets
def split(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

# Function to save the test set to a CSV file
def save_test_csv(X_test: pd.DataFrame, y_test: pd.Series, out_path: Path) -> None:
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    test_df.to_csv(out_path, index=False)
