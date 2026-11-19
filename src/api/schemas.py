"""
Pydantic v2 schemas and validation models for Churn prediction API.
"""

from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator


class ChurnFeatures(BaseModel):
    """
    Input schema defining the 19 required features and optional customerID.
    Strictly aligned with schema.json and business validation rules.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        },
    )

    customerID: Optional[str] = Field(
        None,
        description="Pseudonymized customer identifier (optional in real-time prediction).",
    )
    gender: Literal["Female", "Male"] = Field(
        ...,
        description="Customer gender.",
    )
    SeniorCitizen: Literal[0, 1] = Field(
        ...,
        description="Whether the customer is a senior citizen (1: Yes, 0: No).",
    )
    Partner: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer has a partner.",
    )
    Dependents: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer has dependents.",
    )
    tenure: int = Field(
        ...,
        ge=0,
        le=72,
        description="Number of months the customer has stayed with the company (0 to 72).",
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer has a phone service.",
    )
    MultipleLines: Literal["No", "Yes", "No phone service"] = Field(
        ...,
        description="Whether the customer has multiple lines.",
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ...,
        description="Customer's internet service provider.",
    )
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether the customer has online security.",
    )
    OnlineBackup: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether the customer has online backup.",
    )
    DeviceProtection: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether the customer has tech device protection.",
    )
    TechSupport: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether the customer has technical support.",
    )
    StreamingTV: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether the customer has streaming TV.",
    )
    StreamingMovies: Literal["No", "Yes", "No internet service"] = Field(
        ...,
        description="Whether the customer has streaming movies.",
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ...,
        description="Contract terms of the customer.",
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer uses paperless billing.",
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        ...,
        description="The customer's payment method.",
    )
    MonthlyCharges: float = Field(
        ...,
        ge=10.0,
        le=150.0,
        description="Amount billed to the customer monthly ($).",
    )
    TotalCharges: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Total amount billed to customer ($). Must be 0.0 if tenure=0.",
    )

    @model_validator(mode="after")
    def validate_cross_rules(self) -> "ChurnFeatures":
        # Rule 1: PhoneService=="No" => MultipleLines=="No phone service"
        if self.PhoneService == "No" and self.MultipleLines != "No phone service":
            raise ValueError(
                "Incoherent phone services: when PhoneService is 'No', "
                "MultipleLines must be 'No phone service'."
            )

        # Rule 2: InternetService=="No" => Internet add-ons must be "No internet service"
        if self.InternetService == "No":
            addon_fields = {
                "OnlineSecurity": self.OnlineSecurity,
                "OnlineBackup": self.OnlineBackup,
                "DeviceProtection": self.DeviceProtection,
                "TechSupport": self.TechSupport,
                "StreamingTV": self.StreamingTV,
                "StreamingMovies": self.StreamingMovies,
            }
            invalids = [name for name, val in addon_fields.items() if val != "No internet service"]
            if invalids:
                raise ValueError(
                    f"Incoherent internet services: when InternetService is 'No', "
                    f"add-ons ({', '.join(invalids)}) must be 'No internet service'."
                )

        # Rule 3: tenure==0 => TotalCharges<=0.01
        if self.tenure == 0 and self.TotalCharges > 0.01:
            raise ValueError(
                "Incoherent charges: when tenure is 0, TotalCharges must be <= 0.01."
            )

        return self

    def to_schema_ordered_dict(self) -> Dict[str, Any]:
        """Return the 19 features in the exact required schema order."""
        return {
            "gender": self.gender,
            "SeniorCitizen": self.SeniorCitizen,
            "Partner": self.Partner,
            "Dependents": self.Dependents,
            "tenure": self.tenure,
            "PhoneService": self.PhoneService,
            "MultipleLines": self.MultipleLines,
            "InternetService": self.InternetService,
            "OnlineSecurity": self.OnlineSecurity,
            "OnlineBackup": self.OnlineBackup,
            "DeviceProtection": self.DeviceProtection,
            "TechSupport": self.TechSupport,
            "StreamingTV": self.StreamingTV,
            "StreamingMovies": self.StreamingMovies,
            "Contract": self.Contract,
            "PaperlessBilling": self.PaperlessBilling,
            "PaymentMethod": self.PaymentMethod,
            "MonthlyCharges": self.MonthlyCharges,
            "TotalCharges": self.TotalCharges,
        }


class PredictionResponse(BaseModel):
    """Prediction outcome returned by POST /predict."""

    churn_probability: float = Field(
        ...,
        description="Estimated churn probability P(Churn=Yes).",
    )
    prediction: Literal["Yes", "No"] = Field(
        ...,
        description="Binary churn prediction ('Yes' or 'No').",
    )
    threshold: float = Field(
        ...,
        description="Decision threshold applied to churn probability.",
    )
    model_version: str = Field(
        ...,
        description="Model or registry version utilized for inference.",
    )
    model_source: Literal["registry", "joblib"] = Field(
        ...,
        description="Source from which the model was loaded ('registry' or 'joblib').",
    )


class ModelInfo(BaseModel):
    """Metadata and operational configuration of the deployed model."""

    model_name: str = Field(..., description="Name of the deployed algorithm.")
    model_version: str = Field(..., description="Active model version.")
    registry_version: Optional[str] = Field(None, description="MLflow registry version if applicable.")
    run_id: Optional[str] = Field(None, description="Associated MLflow run ID.")
    threshold: float = Field(..., description="Optimal decision threshold configured.")
    source: Literal["registry", "joblib"] = Field(..., description="Active model artifact source.")
    registry_checked: bool = Field(..., description="Whether the MLflow registry was actively queried.")
    registry_consistent: Optional[bool] = Field(None, description="Whether registry status matches metadata run_id (null if registry was not checked).")
    sklearn_version: str = Field(..., description="Scikit-learn version in runtime environment.")
    metrics_optimal: Optional[Dict[str, Any]] = Field(None, description="Validation metrics achieved at optimal threshold.")
    schema_version: Optional[str] = Field("1.0.0", description="Version of the feature contract schema.")


class HealthResponse(BaseModel):
    """Health check status payload."""

    status: Literal["ok", "unavailable"] = Field(..., description="Operational status of the API.")
