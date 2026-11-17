"""
Machine learning preprocessing and model pipeline constructors.
Ensures zero data leakage via scikit-learn ColumnTransformer and Pipeline.
"""

from typing import List, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

def build_preprocessor() -> ColumnTransformer:
    """Build preprocessor for numeric and categorical features, dropping customerID."""
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"  # Safely ignores customerID and any unneeded columns
    )
    return preprocessor

def get_model_pipelines(seed: int = 42) -> Dict[str, Pipeline]:
    """
    Construct model pipelines for baselines and primary models:
    1. Baseline Always No
    2. Baseline Always Yes
    3. Logistic Regression
    4. Random Forest Classifier
    """
    pipelines = {
        "baseline_always_no": Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", DummyClassifier(strategy="constant", constant=0))
        ]),
        "baseline_always_yes": Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", DummyClassifier(strategy="constant", constant=1))
        ]),
        "logistic_regression": Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=seed,
                class_weight="balanced",
                solver="liblinear"
            ))
        ]),
        "random_forest": Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=seed,
                class_weight="balanced"
            ))
        ])
    }
    return pipelines

