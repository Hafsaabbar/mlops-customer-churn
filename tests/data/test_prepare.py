"""
Tests for data preparation, cleaning, splitting, and manifest generation.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from src.data.prepare import clean_dataset, load_raw_dataset, prepare_splits
from src.data.manifest import compute_sha256, verify_manifest

def test_clean_dataset_total_charges_imputation():
    """Verify that empty TotalCharges for tenure=0 is imputed to 0.0 without dropping rows."""
    raw_sample = pd.DataFrame({
        "customerID": ["001-TEST", "002-TEST"],
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [0, 12],
        "PhoneService": ["No", "Yes"],
        "MultipleLines": ["No phone service", "No"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["No", "No"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["No", "No"],
        "StreamingMovies": ["No", "No"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [25.0, 75.0],
        "TotalCharges": [" ", "900.0"],
        "Churn": ["No", "Yes"]
    })

    cleaned = clean_dataset(raw_sample)
    assert len(cleaned) == 2
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert cleaned.loc[1, "TotalCharges"] == 900.0
    assert cleaned["TotalCharges"].dtype == np.float64

def test_split_files_exist_and_disjoint():
    """Verify all split CSVs exist and customer IDs are strictly disjoint across train, test, and stream."""
    splits_dir = Path("data/splits")
    train_x = pd.read_csv(splits_dir / "train" / "X_train.csv")
    train_y = pd.read_csv(splits_dir / "train" / "y_train.csv")
    test_x = pd.read_csv(splits_dir / "test" / "X_test.csv")
    test_y = pd.read_csv(splits_dir / "test" / "y_test.csv")
    stream_x = pd.read_csv(splits_dir / "stream" / "X_stream.csv")
    stream_y = pd.read_csv(splits_dir / "stream" / "y_stream.csv")

    assert len(train_x) == len(train_y)
    assert len(test_x) == len(test_y)
    assert len(stream_x) == len(stream_y)

    # Disjointness check
    ids_train = set(train_x["customerID"])
    ids_test = set(test_x["customerID"])
    ids_stream = set(stream_x["customerID"])

    assert len(ids_train.intersection(ids_test)) == 0, "Train and Test must be disjoint"
    assert len(ids_train.intersection(ids_stream)) == 0, "Train and Stream must be disjoint"
    assert len(ids_test.intersection(ids_stream)) == 0, "Test and Stream must be disjoint"

    # Total rows equal full dataset
    assert len(ids_train) + len(ids_test) + len(ids_stream) == 7043

def test_stream_features_and_labels_isolated():
    """Verify stream features do NOT include Churn target column."""
    stream_x = pd.read_csv("data/splits/stream/X_stream.csv")
    stream_y = pd.read_csv("data/splits/stream/y_stream.csv")

    assert "Churn" not in stream_x.columns
    assert "Churn" in stream_y.columns
    assert set(stream_y["Churn"].unique()).issubset({"Yes", "No"})

def test_manifest_verification():
    """Verify MANIFEST.json exists and all hashes match data files on disk."""
    assert Path("MANIFEST.json").exists()
    assert verify_manifest() is True

