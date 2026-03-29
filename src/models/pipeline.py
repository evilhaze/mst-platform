"""
Pipeline construction — the single source of truth for all preprocessing.

Key invariants enforced here:
  1. All fit() calls happen inside cross-validation on train data only.
  2. ColumnTransformer handles numeric and categorical branches separately.
  3. The same Pipeline object is used for baseline, tuning, and serving.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.features.engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_RAW_FEATURES,
    RatioFeatureEngineer,
    TimeFeatureEngineer,
)

logger = logging.getLogger(__name__)


class ArrayToDataFrame(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        return X


# ---------------------------------------------------------------------------
# Sub-pipelines
# ---------------------------------------------------------------------------

def _build_numeric_pipeline() -> Pipeline:
    """
    Numeric branch:
      1. Custom ratio + log features
      2. Custom cyclical time encoding
      3. RobustScaler — handles outliers better than StandardScaler for ad data
      4. SimpleImputer as final safety net
    """
    return Pipeline([
        ("ratio_features", RatioFeatureEngineer()),
        ("time_features", TimeFeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])


def _build_categorical_pipeline() -> Pipeline:
    """
    Categorical branch:
      1. Impute unknowns as explicit 'missing' category
      2. One-hot encode with handle_unknown='ignore' for unseen categories
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )),
    ])


# ---------------------------------------------------------------------------
# Public pipeline builders
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer that handles all feature preprocessing."""
    return ColumnTransformer(
        transformers=[
            ("num", _build_numeric_pipeline(), NUMERIC_RAW_FEATURES),
            ("cat", _build_categorical_pipeline(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        n_jobs=-1,
    )


def build_baseline_pipeline(random_state: int = 42) -> Pipeline:
    """
    Baseline pipeline: preprocessing + LogisticRegression.

    Intentionally simple to establish a reproducible lower bound.
    """
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("to_dataframe", ArrayToDataFrame()),
        ("model", LogisticRegression(
            C=0.1,
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def build_lgbm_pipeline(params: dict | None = None, random_state: int = 42) -> Pipeline:
    """
    Main pipeline: preprocessing + LightGBM (with sklearn fallback).

    In production: LightGBM handles sparse OHE matrices efficiently.
    Fallback: sklearn HistGradientBoostingClassifier (identical API pattern).
    """
    try:
        from lightgbm import LGBMClassifier
        default_params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        if params:
            default_params.update(params)
        model = LGBMClassifier(**default_params)
        logger.info("Using LightGBM classifier")

    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        logger.warning("LightGBM not installed — using HistGradientBoostingClassifier")
        hgb_params = {
            "max_iter":          (params or {}).get("n_estimators", 1000),
            "learning_rate":     (params or {}).get("learning_rate", 0.05),
            "max_leaf_nodes":    (params or {}).get("num_leaves", 63),
            "min_samples_leaf":  (params or {}).get("min_child_samples", 50),
            "l2_regularization": (params or {}).get("reg_lambda", 0.1),
            "random_state":      random_state,
        }
        model = HistGradientBoostingClassifier(**hgb_params)

    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("to_dataframe", ArrayToDataFrame()),
        ("model", model),
    ])
