"""
Manifest and data integrity module for Customer Churn MLOps project.
Computes SHA-256 hashes, generates MANIFEST.json and verifies file integrity.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found for hash calculation: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_git_commit_hash() -> str:
    """Retrieve current git commit hash if available, else return 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "uncommitted_or_git_unavailable"

def build_manifest(
    raw_path: Path,
    splits_dir: Path,
    seed: int,
    train_ratio: float,
    test_ratio: float,
    stream_ratio: float,
    manifest_path: Path = Path("MANIFEST.json")
) -> Dict[str, Any]:
    """Generate and write MANIFEST.json with metadata, split shapes and SHA-256 hashes."""
    import pandas as pd

    split_files = [
        splits_dir / "train" / "X_train.csv",
        splits_dir / "train" / "y_train.csv",
        splits_dir / "test" / "X_test.csv",
        splits_dir / "test" / "y_test.csv",
        splits_dir / "stream" / "X_stream.csv",
        splits_dir / "stream" / "y_stream.csv",
    ]

    hashes = {}
    shapes = {}
    columns_info = {}

    if raw_path.exists():
        hashes["raw_source"] = {
            "path": raw_path.as_posix(),
            "sha256": compute_sha256(raw_path)
        }

    for sf in split_files:
        if not sf.exists():
            raise FileNotFoundError(f"Split file missing: {sf}")
        rel_key = sf.relative_to(splits_dir).as_posix()
        file_hash = compute_sha256(sf)
        hashes[rel_key] = {
            "path": sf.as_posix(),
            "sha256": file_hash
        }
        df_tmp = pd.read_csv(sf)
        shapes[rel_key] = list(df_tmp.shape)
        if "X_" in sf.name:
            columns_info[rel_key] = list(df_tmp.columns)

    manifest_data = {
        "project": "mlops-customer-churn",
        "dataset_name": "IBM Telco Customer Churn",
        "source_url": "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "split_ratios": {
            "train": train_ratio,
            "test": test_ratio,
            "stream": stream_ratio
        },
        "git_commit": get_git_commit_hash(),
        "splits_dimensions": shapes,
        "feature_columns": columns_info.get("train/X_train.csv", []),
        "target_column": "Churn",
        "file_hashes": hashes,
        "reproduction_command": "python -m src.data.prepare"
    }

    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    return manifest_data

def verify_manifest(manifest_path: Path = Path("MANIFEST.json")) -> bool:
    """Verify that all recorded data files exist and match their SHA-256 hashes."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file {manifest_path} not found.")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_hashes = manifest_data.get("file_hashes", {})

    all_valid = True
    errors = []

    for key, info in file_hashes.items():
        file_path = Path(info["path"])
        expected_hash = info["sha256"]
        if not file_path.exists():
            errors.append(f"Missing file: {file_path}")
            all_valid = False
            continue
        actual_hash = compute_sha256(file_path)
        if actual_hash != expected_hash:
            errors.append(f"Hash mismatch for {file_path}: expected {expected_hash}, got {actual_hash}")
            all_valid = False

    if not all_valid:
        raise ValueError("Manifest verification failed:\n" + "\n".join(errors))

    return True

if __name__ == "__main__":
    verify_manifest()
    print("Manifest integrity verification SUCCESSFUL.")
