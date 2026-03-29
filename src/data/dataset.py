"""
Data generation and loading utilities.

In production, replace generate_synthetic_data() with actual data loaders
from your data warehouse (ClickHouse, BigQuery, etc.).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic data generation (replace with real ETL in production)
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_samples: int = 500_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic advertising event data.

    Feature correlations mirror real-world ad-tech patterns:
    - Higher CTR → higher conversion probability
    - Mobile devices convert at different rates than desktop
    - Geo clusters have different baseline CTRs
    - Time-of-day effects on conversion
    """
    rng = np.random.default_rng(random_seed)

    n = n_samples

    # --- Categorical features ---
    geo = rng.choice(
        ["US", "UK", "DE", "FR", "CA", "AU", "JP", "BR"],
        size=n,
        p=[0.30, 0.15, 0.12, 0.10, 0.08, 0.07, 0.10, 0.08],
    )
    device_type = rng.choice(
        ["mobile", "desktop", "tablet"],
        size=n,
        p=[0.55, 0.35, 0.10],
    )
    ad_format = rng.choice(
        ["banner", "video", "native", "interstitial"],
        size=n,
        p=[0.40, 0.25, 0.25, 0.10],
    )
    placement = rng.choice(
        ["top", "sidebar", "in-feed", "footer"],
        size=n,
        p=[0.30, 0.20, 0.40, 0.10],
    )
    campaign_id = rng.choice([f"camp_{i:04d}" for i in range(200)], size=n)
    creative_id = rng.choice([f"cre_{i:04d}" for i in range(500)], size=n)

    # --- Numeric features ---
    impressions = rng.integers(100, 50_000, size=n)
    # CTR base varies by geo and device
    geo_ctr_base = {"US": 0.025, "UK": 0.022, "DE": 0.020, "FR": 0.019,
                    "CA": 0.021, "AU": 0.023, "JP": 0.018, "BR": 0.015}
    device_ctr_mult = {"mobile": 1.1, "desktop": 1.0, "tablet": 0.95}

    ctr_base = np.array([geo_ctr_base[g] for g in geo])
    ctr_mult = np.array([device_ctr_mult[d] for d in device_type])
    ctr = np.clip(
        rng.normal(ctr_base * ctr_mult, 0.005, size=n), 0.001, 0.20
    )
    clicks = np.maximum(1, (impressions * ctr).astype(int))
    ctr_actual = clicks / impressions  # recompute after rounding

    spend = rng.uniform(0.5, 5.0, size=n) * clicks
    cpc = spend / clicks
    bid_amount = rng.uniform(0.10, 3.00, size=n)

    hour_of_day = rng.integers(0, 24, size=n)
    day_of_week = rng.integers(0, 7, size=n)
    campaign_age_days = rng.integers(1, 365, size=n)

    # Simulated feature-store aggregates
    avg_ctr_geo_7d = ctr_base + rng.normal(0, 0.002, size=n)
    avg_ctr_device_7d = ctr_base * ctr_mult + rng.normal(0, 0.002, size=n)
    avg_ctr_creative_7d = ctr + rng.normal(0, 0.003, size=n)

    # --- Interaction features ---
    ctr_vs_geo_baseline = ctr_actual / (avg_ctr_geo_7d + 1e-6)
    ctr_vs_device_baseline = ctr_actual / (avg_ctr_device_7d + 1e-6)
    ctr_vs_creative_hist = ctr_actual / (avg_ctr_creative_7d + 1e-6)
    spend_per_impression = spend / np.maximum(impressions, 1)
    bid_to_cpc_ratio = bid_amount / np.maximum(cpc, 0.01)

    # --- Target: conversion (binary) ---
    # Logistic model with realistic coefficients
    log_odds = (
        -3.5
        + 6.0 * ctr_actual
        + 0.3 * np.log1p(clicks)
        - 0.5 * cpc
        + 0.2 * (hour_of_day >= 9) * (hour_of_day <= 20)
        + 0.15 * (device_type == "desktop")
        - 0.1 * (device_type == "tablet")
        + 0.2 * (ad_format == "video")
        + 0.1 * (ad_format == "native")
        + 2.5 * (ctr_vs_geo_baseline - 1.0)
        + 1.8 * (ctr_vs_device_baseline - 1.0)
        + 1.2 * (ctr_vs_creative_hist - 1.0)
        + 0.20 * bid_to_cpc_ratio
        + rng.normal(0, 0.4, size=n)
    )
    prob_convert = 1 / (1 + np.exp(-log_odds))
    converted = rng.binomial(1, prob_convert, size=n)

    df = pd.DataFrame({
        # Categorical
        "geo": geo,
        "device_type": device_type,
        "ad_format": ad_format,
        "placement": placement,
        "campaign_id": campaign_id,
        "creative_id": creative_id,
        # Numeric
        "impressions": impressions,
        "clicks": clicks,
        "ctr": ctr_actual,
        "spend": spend.round(4),
        "cpc": cpc.round(4),
        "bid_amount": bid_amount.round(4),
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "campaign_age_days": campaign_age_days,
        # Feature-store aggregates
        "avg_ctr_geo_7d": avg_ctr_geo_7d.clip(0, 1).round(6),
        "avg_ctr_device_7d": avg_ctr_device_7d.clip(0, 1).round(6),
        "avg_ctr_creative_7d": avg_ctr_creative_7d.clip(0, 1).round(6),
        # Interaction features
        "ctr_vs_geo_baseline": ctr_vs_geo_baseline.round(6),
        "ctr_vs_device_baseline": ctr_vs_device_baseline.round(6),
        "ctr_vs_creative_hist": ctr_vs_creative_hist.round(6),
        "spend_per_impression": spend_per_impression.round(6),
        "bid_to_cpc_ratio": bid_to_cpc_ratio.round(6),
        # Target
        "converted": converted,
    })

    logger.info(
        "Generated %d samples | conversion rate: %.2f%%",
        n,
        converted.mean() * 100,
    )
    return df


def load_data(
    n_samples: int = 500_000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return train / val / test DataFrames with no leakage.

    Split strategy:
        80% train  →  used for CV and final fit
        10% val    →  early stopping / threshold tuning
        10% test   →  hold-out, touched ONCE at evaluation
    """
    df = generate_synthetic_data(n_samples=n_samples, random_seed=random_seed)

    target = "converted"
    df_train_val, df_test = train_test_split(
        df, test_size=0.10, stratify=df[target], random_state=random_seed
    )
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=0.111,  # 0.1 / 0.9 ≈ 0.111 → gives 10% of original
        stratify=df_train_val[target],
        random_state=random_seed,
    )

    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(df_train), len(df_val), len(df_test),
    )
    return df_train, df_val, df_test


def load_data_temporal(
    n_samples: int = 500_000,
    random_seed: int = 42,
    test_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return train / val / test DataFrames using temporal split.

    Split strategy:
        Train: events from days 1 to (90 - 2*test_days)
        Val:   events from days (90 - 2*test_days + 1) to (90 - test_days)
        Test:  events from days (90 - test_days + 1) to 90
    """
    df = generate_synthetic_data(n_samples=n_samples, random_seed=random_seed)

    rng = np.random.default_rng(random_seed)
    n = len(df)
    days = rng.integers(1, 91, size=n)
    df["event_day"] = days

    test_cutoff = 90 - test_days
    val_cutoff = test_cutoff - test_days

    df_train = df[df["event_day"] <= val_cutoff].drop(columns=["event_day"])
    df_val   = df[(df["event_day"] > val_cutoff) & (df["event_day"] <= test_cutoff)].drop(columns=["event_day"])
    df_test  = df[df["event_day"] > test_cutoff].drop(columns=["event_day"])

    target = "converted"
    logger.info(
        "Temporal split — train: %d (days 1-%d) | val: %d (days %d-%d) | test: %d (days %d-90)",
        len(df_train), val_cutoff,
        len(df_val), val_cutoff+1, test_cutoff,
        len(df_test), test_cutoff+1,
    )
    logger.info(
        "Conversion rates — train: %.2f%% | val: %.2f%% | test: %.2f%%",
        df_train[target].mean()*100,
        df_val[target].mean()*100,
        df_test[target].mean()*100,
    )
    return df_train, df_val, df_test


def split_features_target(
    df: pd.DataFrame,
    target: str = "converted",
) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target]), df[target]
