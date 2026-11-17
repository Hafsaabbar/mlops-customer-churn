"""
Pandera validation schemas and schema.json contract export.
Defines strict schemas for input features, targets, business rules,
and exports the formal contract for downstream API / Pydantic integration.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema

# Categorical allowed values from real dataset audit
ALLOWED_CATEGORIES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ],
    "Churn": ["Yes", "No"]
}

# Cross-column business coherence checks
def check_phone_coherence(df):
    """If PhoneService is 'No', MultipleLines must be 'No phone service'."""
    no_phone = df["PhoneService"] == "No"
    invalid = no_phone & (df["MultipleLines"] != "No phone service")
    return not invalid.any()

def check_internet_coherence(df):
    """If InternetService is 'No', internet add-ons must be 'No internet service'."""
    no_internet = df["InternetService"] == "No"
    addons = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for addon in addons:
        invalid = no_internet & (df[addon] != "No internet service")
        if invalid.any():
            return False
    return True

def check_tenure_total_charges(df):
    """If tenure is 0, TotalCharges must be 0.0."""
    zero_tenure = df["tenure"] == 0
    invalid = zero_tenure & (df["TotalCharges"] > 0.01)
    return not invalid.any()

# Pandera Feature Schema (X data)
feature_schema = DataFrameSchema(
    columns={
        "customerID": Column(pa.String, nullable=False, checks=Check(lambda s: s.str.len() > 3, name="valid_customer_id")),
        "gender": Column(pa.String, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["gender"])),
        "SeniorCitizen": Column(pa.Int, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["SeniorCitizen"])),
        "Partner": Column(pa.String, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["Partner"])),
        "Dependents": Column(pa.String, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["Dependents"])),
        "tenure": Column(pa.Int, nullable=False, checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(72)]),
        "PhoneService": Column(pa.String, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["PhoneService"])),
        "MultipleLines": Column(pa.String, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["MultipleLines"])),
        "InternetService": Column(pa.String, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["InternetService"])),
        "OnlineSecurity": Column(pa.String, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["OnlineSecurity"])),
        "OnlineBackup": Column(pa.String, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["OnlineBackup"])),
        "DeviceProtection": Column(pa.String, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["DeviceProtection"])),
        "TechSupport": Column(pa.String, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["TechSupport"])),
        "StreamingTV": Column(pa.String, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["StreamingTV"])),
        "StreamingMovies": Column(pa.String, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["StreamingMovies"])),
        "Contract": Column(pa.String, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["Contract"])),
        "PaperlessBilling": Column(pa.String, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["PaperlessBilling"])),
        "PaymentMethod": Column(pa.String, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["PaymentMethod"])),
        "MonthlyCharges": Column(pa.Float, nullable=False, checks=[Check.greater_than_or_equal_to(10.0), Check.less_than_or_equal_to(150.0)]),
        "TotalCharges": Column(pa.Float, nullable=False, checks=[Check.greater_than_or_equal_to(0.0), Check.less_than_or_equal_to(10000.0)])
    },
    checks=[
        Check(check_phone_coherence, name="phone_service_coherence"),
        Check(check_internet_coherence, name="internet_service_coherence"),
        Check(check_tenure_total_charges, name="tenure_totalcharges_coherence")
    ],
    coerce=True,
    strict=True
)

# Pandera Target Schema (y data)
target_schema = DataFrameSchema(
    columns={
        "Churn": Column(pa.String, nullable=False, checks=Check.isin(ALLOWED_CATEGORIES["Churn"]))
    },
    coerce=True,
    strict=True
)

def export_schema_json(output_path: Path = Path("schema.json")) -> Dict[str, Any]:
    """
    Export schema contract for downstream prediction service (FastAPI/Pydantic).
    Explicitly does NOT include the target 'Churn'.
    """
    schema_contract = {
        "title": "CustomerChurnPredictionInputSchema",
        "description": "Formal contract of input features for Customer Churn prediction API (Student B).",
        "version": "1.0.0",
        "type": "object",
        "required": [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"
        ],
        "optional": ["customerID"],
        "target": None,
        "properties": {
            "customerID": {
                "type": "string",
                "required": False,
                "description": "Pseudonymized customer identifier (optional in real-time prediction)."
            },
            "gender": {
                "type": "string",
                "required": True,
                "allowed_values": ALLOWED_CATEGORIES["gender"],
                "description": "Customer gender."
            },
            "SeniorCitizen": {
                "type": "integer",
                "required": True,
                "allowed_values": ALLOWED_CATEGORIES["SeniorCitizen"],
                "description": "Whether the customer is a senior citizen (1: Yes, 0: No)."
            },
            "Partner": {
                "type": "string",
                "required": True,
                "allowed_values": ALLOWED_CATEGORIES["Partner"],
                "description": "Whether the customer has a partner."
            },
            "Dependents": {
                "type": "string",
                "required": True,
                "allowed_values": ALLOWED_CATEGORIES["Dependents"],
                "description": "Whether the customer has dependents."
            },
            "tenure": {
                "type": "integer",
                "required": True,
                "min": 0,
                "max": 72,
                "description": "Number of months the customer has stayed with the company."
            },
            "PhoneService": {
                "type": "string",
                "required": True,
                "allowed_values": ALLOWED_CATEGORIES["PhoneService"],
                "description": "Whether the customer has a phone service."
            },
            "MultipleLines": {
                "type": "string",
                "required": True,
                "allowed_values": ALLOWED_CATEGORIES["MultipleLines"],
                "description": "Whether the customer has multiple lines."
            },
            "InternetService": {
                "type": "string",
                "required": True,
                "allowed_values": ALLOWED_CATEGORIES["InternetService"],
                "description": "Customer's internet service provider."
            },
            "OnlineSecurity": {
                "type": "string",
                "required": True,
                "allowed_values": ALLOWED_CATEGORIES["OnlineSecurity"],
                "description": "Whether the customer has online security."
            },
            "OnlineBackup": {
                "type": "string",
                "required": True,
                "allowed_values": ALLOWED_CATEGORIES["OnlineBackup"],
                "description": "Whether the customer has online backup."
            },
            "DeviceProtection": {
                "type": "string",
                "required": True,
                "allowed_values": ALLOWED_CATEGORIES["DeviceProtection"],
                "description": "Whether the customer has tech device protection."
            },
            "TechSupport": {
                "type": "string",
                "required": True,
                "allowed_values": ALLOWED_CATEGORIES["TechSupport"],
                "description": "Whether the customer has technical support."
            },
            "StreamingTV": {
                "type": "string",
                "required": True,
                "allowed_values": ALLOWED_CATEGORIES["StreamingTV"],
                "description": "Whether the customer has streaming TV."
            },
            "StreamingMovies": {
                "type": "string",
                "required": True,
                "allowed_values": ALLOWED_CATEGORIES["StreamingMovies"],
                "description": "Whether the customer has streaming movies."
            },
            "Contract": {
                "type": "string",
                "required": True,
                "allowed_values": ALLOWED_CATEGORIES["Contract"],
                "description": "Contract terms of the customer."
            },
            "PaperlessBilling": {
                "type": "string",
                "required": True,
                "allowed_values": ALLOWED_CATEGORIES["PaperlessBilling"],
                "description": "Whether the customer uses paperless billing."
            },
            "PaymentMethod": {
                "type": "string",
                "required": True,
                "allowed_values": ALLOWED_CATEGORIES["PaymentMethod"],
                "description": "The customer's payment method."
            },
            "MonthlyCharges": {
                "type": "number",
                "format": "float",
                "required": True,
                "min": 10.0,
                "max": 150.0,
                "description": "Amount billed to the customer monthly ($)."
            },
            "TotalCharges": {
                "type": "number",
                "format": "float",
                "required": True,
                "min": 0.0,
                "max": 10000.0,
                "description": "Total amount billed to customer ($). Must be 0.0 if tenure=0."
            }
        }
    }

    output_path.write_text(json.dumps(schema_contract, indent=2), encoding="utf-8")
    return schema_contract

if __name__ == "__main__":
    export_schema_json()
    print("schema.json exported successfully.")
