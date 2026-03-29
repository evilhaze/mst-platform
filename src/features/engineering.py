"""
Feature engineering transformers.

All transformers follow sklearn's TransformerMixin interface so they slot
cleanly into Pipeline without any special handling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Custom transformers
# ---------------------------------------------------------------------------


class RatioFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Compute derived ratio features BEFORE scaling.

    Must run on raw numeric columns — vectorized operations only.
    No .apply(), no row-wise lambdas.
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()

        # Guard against division by zero with .clip(lower=1)
        if "clicks" in X.columns and "impressions" in X.columns:
            X["ctr_engineered"] = X["clicks"] / X["impressions"].clip(lower=1)

        if "spend" in X.columns and "clicks" in X.columns:
            X["cpc_engineered"] = X["spend"] / X["clicks"].clip(lower=1)

        if "spend" in X.columns and "impressions" in X.columns:
            X["cpm"] = (X["spend"] / X["impressions"].clip(lower=1)) * 1000

        # Log-transform heavy-tailed features for linear models
        for col in ["impressions", "clicks", "spend"]:
            if col in X.columns:
                X[f"log_{col}"] = np.log1p(X[col])

        return X


class TimeFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Encode cyclical time features using sin/cos projection.

    Cyclical encoding prevents the model from treating hour 23 and hour 0
    as being 23 units apart when they're actually adjacent.
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()

        if "hour_of_day" in X.columns:
            X["hour_sin"] = np.sin(2 * np.pi * X["hour_of_day"] / 24)
            X["hour_cos"] = np.cos(2 * np.pi * X["hour_of_day"] / 24)

        if "day_of_week" in X.columns:
            X["dow_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
            X["dow_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)

            # Weekend flag
            X["is_weekend"] = (X["day_of_week"] >= 5).astype(np.int8)

        # Business hours flag
        if "hour_of_day" in X.columns:
            X["is_business_hours"] = (
                (X["hour_of_day"] >= 9) & (X["hour_of_day"] <= 18)
            ).astype(np.int8)

        return X


# ---------------------------------------------------------------------------
# Feature column definitions (used by ColumnTransformer)
# ---------------------------------------------------------------------------

NUMERIC_RAW_FEATURES = [
    "impressions",
    "clicks",
    "ctr",
    "spend",
    "cpc",
    "bid_amount",
    "hour_of_day",
    "day_of_week",
    "campaign_age_days",
    "avg_ctr_geo_7d",
    "avg_ctr_device_7d",
    "avg_ctr_creative_7d",
    "ctr_vs_geo_baseline",
    "ctr_vs_device_baseline",
    "ctr_vs_creative_hist",
    "spend_per_impression",
    "bid_to_cpc_ratio",
]

CATEGORICAL_FEATURES = [
    "geo",
    "device_type",
    "ad_format",
    "placement",
    # High-cardinality IDs — use target encoding or hashing in production
    # Excluded here to keep baseline simple; add back with TargetEncoder
]

# Features added by transformers above
ENGINEERED_NUMERIC = [
    "ctr_engineered",
    "cpc_engineered",
    "cpm",
    "log_impressions",
    "log_clicks",
    "log_spend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_business_hours",
]
