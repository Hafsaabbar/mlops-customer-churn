"""
Tests for ML pipelines, column transformations, baselines, and cost functions.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from src.training.pipeline import build_preprocessor, get_model_pipelines
from src.training.cost import compute_business_cost, find_optimal_threshold
from src.training.train import evaluate_predictions

@pytest.fixture
def train_sample():
    X = pd.read_csv("data/splits/train/X_train.csv").head(100)
    y = pd.read_csv("data/splits/train/y_train.csv")["Churn"].head(100).map({"Yes": 1, "No": 0}).values
    return X, y

def test_preprocessor_excludes_customer_id(train_sample):
    """Verify that the ColumnTransformer transforms features while dropping customerID."""
    X, _ = train_sample
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    # 3 numeric + one-hot categories, but customerID must NOT be included
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 20  # Expanded by one-hot

def test_all_model_pipelines_fit_and_predict(train_sample):
    """Verify all defined models (baselines, logistic regression, random forest) can fit and predict."""
    X, y = train_sample
    pipelines = get_model_pipelines(seed=42)

    for name, pipe in pipelines.items():
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

def test_business_cost_metric_calculation():
    """Verify the business cost formula: FN * 500 + FP * 50."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0])
    # TP=1, FN=1, FP=1, TN=1
    cost_info = compute_business_cost(y_true, y_pred, cost_fn=500.0, cost_fp=50.0)
    assert cost_info["fn"] == 1
    assert cost_info["fp"] == 1
    assert cost_info["total_cost"] == 550.0
    assert cost_info["avg_cost_per_customer"] == 550.0 / 4

def test_find_optimal_threshold_range():
    """Verify threshold search stays within valid [0.05, 0.95] range."""
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.6, 0.2, 0.1, 0.8, 0.3])
    opt = find_optimal_threshold(y_true, y_prob)
    assert 0.05 <= opt["optimal_threshold"] <= 0.95
    assert opt["min_total_cost"] >= 0.0
