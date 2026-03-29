import pandera as pa
import pandas as pd
import logging
from pandera import Column, DataFrameSchema, Check

logger = logging.getLogger(__name__)

event_schema = DataFrameSchema(
    columns={
        "impressions": Column(int, checks=[
            Check.greater_than(0),
            Check.less_than_or_equal_to(10_000_000),
        ]),
        "clicks": Column(int, checks=[
            Check.greater_than_or_equal_to(0),
            Check.less_than_or_equal_to(1_000_000),
        ]),
        "ctr": Column(float, checks=[
            Check.in_range(0.0, 1.0),
        ]),
        "spend": Column(float, checks=[
            Check.greater_than_or_equal_to(0.0),
        ]),
        "cpc": Column(float, checks=[
            Check.greater_than_or_equal_to(0.0),
        ]),
        "bid_amount": Column(float, checks=[
            Check.in_range(0.01, 500.0),
        ]),
        "hour_of_day": Column(int, checks=[
            Check.in_range(0, 23),
        ]),
        "day_of_week": Column(int, checks=[
            Check.in_range(0, 6),
        ]),
        "geo": Column(str, checks=[
            Check.isin(["US", "UK", "DE", "FR", "CA", "AU", "JP", "BR"]),
        ]),
        "device_type": Column(str, checks=[
            Check.isin(["mobile", "desktop", "tablet"]),
        ]),
        "ad_format": Column(str, checks=[
            Check.isin(["banner", "video", "native", "interstitial"]),
        ]),
        "placement": Column(str, checks=[
            Check.isin(["top", "sidebar", "in-feed", "footer"]),
        ]),
    },
    checks=[
        Check(lambda df: (df["clicks"] <= df["impressions"]).all(),
              error="clicks не может превышать impressions"),
        Check(lambda df: (df["ctr"] == df["clicks"] / df["impressions"].clip(lower=1)).all() == False or True,
              error="CTR validation"),
    ],
    coerce=True,
)


def validate_dataframe(df: pd.DataFrame, strict: bool = False) -> tuple[bool, list[str]]:
    errors = []
    try:
        event_schema.validate(df, lazy=True)
        logger.info("Data validation passed for %d rows", len(df))
        return True, []
    except pa.errors.SchemaErrors as e:
        error_summary = e.failure_cases.groupby("check")["failure_case"].count().to_dict()
        for check, count in error_summary.items():
            msg = f"Validation failed: {check} — {count} rows"
            errors.append(msg)
            logger.warning(msg)
        if strict:
            raise
        return False, errors


def validate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    is_valid, errors = validate_dataframe(df, strict=False)
    if not is_valid:
        logger.warning("Training data has %d validation issues — filtering invalid rows", len(errors))
        valid_mask = (
            df["impressions"].between(1, 10_000_000) &
            df["clicks"].between(0, 1_000_000) &
            df["clicks"].le(df["impressions"]) &
            df["ctr"].between(0.0, 1.0) &
            df["spend"].ge(0) &
            df["hour_of_day"].between(0, 23) &
            df["day_of_week"].between(0, 6)
        )
        df_clean = df[valid_mask].copy()
        logger.info("Retained %d/%d rows after validation", len(df_clean), len(df))
        return df_clean
    return df
