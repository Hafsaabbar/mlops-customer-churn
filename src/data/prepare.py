"""
Data preparation and splitting module for IBM Telco Customer Churn.
Performs deterministic stratified splitting into Train (70%), Test (15%), and Stream (15%).
Separates features (X) and target (y) for all splits, notably for the stream simulation.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.manifest import build_manifest

def load_raw_dataset(raw_path: Path) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    if not raw_path.exists():
        fallback_root = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        if fallback_root.exists():
            raw_path = fallback_root
        else:
            raise FileNotFoundError(f"Raw dataset not found at {raw_path}")
    
    df = pd.read_csv(raw_path)
    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply essential cleaning justified in the data audit:
    - Trim whitespace in TotalCharges and convert to numeric.
    - Impute TotalCharges = 0.0 for records with tenure == 0 (new customers without closed bill).
    - Ensure SeniorCitizen is int.
    - Strip whitespace from all string columns.
    """
    df = df.copy()

    # Strip whitespaces from object columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # TotalCharges conversion and justified imputation
    total_charges_num = pd.to_numeric(df["TotalCharges"].replace("", np.nan), errors="coerce")
    # For tenure == 0, TotalCharges is 0.0
    zero_tenure_mask = (df["tenure"] == 0) & (total_charges_num.isna())
    total_charges_num = total_charges_num.mask(zero_tenure_mask, 0.0)
    # If any remaining NaN, fill with 0.0
    total_charges_num = total_charges_num.fillna(0.0)
    df["TotalCharges"] = total_charges_num

    # SeniorCitizen as integer (0 or 1)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    return df

def prepare_splits(
    raw_path: Path = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    splits_dir: Path = Path("data/splits"),
    seed: int = 42,
    train_ratio: float = 0.70,
    test_ratio: float = 0.15,
    stream_ratio: float = 0.15
):
    """
    Split the dataset into Train, Test, and Stream in a stratified, deterministic manner.
    Features (X) and labels (y) are saved separately for all sets.
    """
    assert round(train_ratio + test_ratio + stream_ratio, 4) == 1.0, "Split ratios must sum to 1.0"

    fallback_root = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if not raw_path.exists() and fallback_root.exists():
        raw_path = fallback_root

    # Graceful handling for clean clone where raw CSV is gitignored
    if not raw_path.exists():
        if (splits_dir / "train" / "X_train.csv").exists():
            print("Notice: Raw dataset file not present (excluded from git redistribution).")
            print("Verified split files detected in data/splits/. Rebuilding manifest.")
            return build_manifest(
                raw_path=raw_path,
                splits_dir=splits_dir,
                seed=seed,
                train_ratio=train_ratio,
                test_ratio=test_ratio,
                stream_ratio=stream_ratio
            )
        raise FileNotFoundError(f"Raw dataset not found at {raw_path} and no splits available.")

    df_raw = load_raw_dataset(raw_path)
    df_cleaned = clean_dataset(df_raw)

    target_col = "Churn"
    feature_cols = [c for c in df_cleaned.columns if c != target_col]

    X = df_cleaned[feature_cols]
    y = df_cleaned[[target_col]]

    # Step 1: Separate Train (70%) from the remaining 30% (Test + Stream)
    temp_ratio = test_ratio + stream_ratio  # 0.30
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_ratio,
        random_state=seed,
        stratify=y[target_col]
    )

    # Step 2: Split temp equally (50%/50% of the 30%, i.e. 15% Test, 15% Stream)
    stream_share_of_temp = stream_ratio / temp_ratio  # 0.50
    X_test, X_stream, y_test, y_stream = train_test_split(
        X_temp, y_temp,
        test_size=stream_share_of_temp,
        random_state=seed,
        stratify=y_temp[target_col]
    )

    # Save to directory structure
    for split_name, (X_s, y_s) in {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
        "stream": (X_stream, y_stream),
    }.items():
        out_dir = splits_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        X_s.to_csv(out_dir / f"X_{split_name}.csv", index=False)
        y_s.to_csv(out_dir / f"y_{split_name}.csv", index=False)

    print(f"Data splits successfully created with seed={seed}:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    print(f"  Stream: X={X_stream.shape}, y={y_stream.shape}")

    # Build manifest
    manifest = build_manifest(
        raw_path=raw_path,
        splits_dir=splits_dir,
        seed=seed,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        stream_ratio=stream_ratio
    )
    print("MANIFEST.json updated.")
    return manifest

if __name__ == "__main__":
    prepare_splits()
