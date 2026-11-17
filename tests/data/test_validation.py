"""
Tests for Pandera data validation schemas, corruption scenarios, and schema.json contract.
Verifies nominal pass, explicit failure on corruptions, and business coherence rules.
"""

import json
from pathlib import Path
import pandas as pd
import pytest
import pandera.pandas as pa

from src.validation.schema import feature_schema, target_schema, export_schema_json
from src.validation.validate import validate_dataframe

@pytest.fixture
def sample_valid_data():
    """Load a real sample of valid features and target from train split."""
    x_path = Path("data/splits/train/X_train.csv")
    y_path = Path("data/splits/train/y_train.csv")
    df_x = pd.read_csv(x_path).head(50)
    df_y = pd.read_csv(y_path).head(50)
    return df_x, df_y

def test_nominal_data_passes(sample_valid_data):
    """Proves that uncorrupted data passes all schema checks."""
    df_x, df_y = sample_valid_data
    result = validate_dataframe(df_x, df_y)
    assert result["valid"] is True
    assert result["error_count"] == 0

def test_corrupted_column_name_fails(sample_valid_data):
    """Proves that renaming or removing a mandatory column causes validation failure."""
    df_x, _ = sample_valid_data
    corrupted_x = df_x.copy().rename(columns={"tenure": "anciennete_client"})
    result = validate_dataframe(corrupted_x)
    assert result["valid"] is False
    assert any("tenure" in str(e) or "column" in str(e) for e in result["errors"])

def test_corrupted_category_fails(sample_valid_data):
    """Proves that introducing an invalid/unknown category is rejected."""
    df_x, _ = sample_valid_data
    corrupted_x = df_x.copy()
    corrupted_x.loc[0, "Contract"] = "Infinite-Plan"  # Invalid contract category
    result = validate_dataframe(corrupted_x)
    assert result["valid"] is False
    assert any("Contract" in str(e) for e in result["errors"])

def test_corrupted_numeric_bounds_fails(sample_valid_data):
    """Proves that numeric values exceeding bounds (e.g. tenure > 72 or negative charges) fail."""
    df_x, _ = sample_valid_data
    corrupted_x = df_x.copy()
    corrupted_x.loc[0, "tenure"] = 999  # Max valid tenure is 72
    result = validate_dataframe(corrupted_x)
    assert result["valid"] is False
    assert any("tenure" in str(e) for e in result["errors"])

def test_corrupted_phone_coherence_fails(sample_valid_data):
    """Proves that business incoherence (PhoneService='No' with MultipleLines='Yes') is rejected."""
    df_x, _ = sample_valid_data
    corrupted_x = df_x.copy()
    corrupted_x.loc[0, "PhoneService"] = "No"
    corrupted_x.loc[0, "MultipleLines"] = "Yes"
    result = validate_dataframe(corrupted_x)
    assert result["valid"] is False
    assert any("phone_service_coherence" in str(e) for e in result["errors"])

def test_corrupted_target_fails(sample_valid_data):
    """Proves that targets outside {'Yes', 'No'} are rejected."""
    df_x, df_y = sample_valid_data
    corrupted_y = df_y.copy()
    corrupted_y.loc[0, "Churn"] = "Maybe"
    result = validate_dataframe(df_x, corrupted_y)
    assert result["valid"] is False
    assert any("target" in str(e) for e in result["errors"])

def test_schema_json_contract():
    """Verify schema.json conforms to contract requirements for Student B (FastAPI)."""
    schema_path = Path("schema.json")
    if not schema_path.exists():
        export_schema_json(schema_path)

    contract = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "properties" in contract
    assert "required" in contract
    # Target 'Churn' must NOT be in the input feature schema
    assert "Churn" not in contract["properties"]
    assert "Churn" not in contract["required"]
    # Key features must be present
    assert "tenure" in contract["properties"]
    assert "MonthlyCharges" in contract["properties"]
    assert "TotalCharges" in contract["properties"]
    assert "Contract" in contract["properties"]
