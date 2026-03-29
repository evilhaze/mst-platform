"""
Pydantic schemas for request/response validation.

Strict typing ensures invalid inputs return 422 (Unprocessable Entity),
not 500 (Internal Server Error). All business rule validations here.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class AdEventFeatures(BaseModel):
    """
    Features describing a single ad event to score.

    All fields map 1:1 to model feature columns.
    Defaults provided for optional fields to support partial payloads.
    """

    # --- Numeric ---
    impressions: Annotated[int, Field(ge=1, le=10_000_000)] = 1000
    clicks: Annotated[int, Field(ge=0, le=1_000_000)] = 10
    ctr: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    spend: Annotated[float, Field(ge=0.0, le=100_000.0)] = 5.0
    cpc: Annotated[float, Field(ge=0.0, le=1_000.0)] | None = None
    bid_amount: Annotated[float, Field(ge=0.01, le=500.0)] = 0.5
    hour_of_day: Annotated[int, Field(ge=0, le=23)] = 12
    day_of_week: Annotated[int, Field(ge=0, le=6)] = 0
    campaign_age_days: Annotated[int, Field(ge=0, le=3650)] = 30

    # Feature-store aggregates
    avg_ctr_geo_7d: Annotated[float, Field(ge=0.0, le=1.0)] = 0.02
    avg_ctr_device_7d: Annotated[float, Field(ge=0.0, le=1.0)] = 0.02
    avg_ctr_creative_7d: Annotated[float, Field(ge=0.0, le=1.0)] = 0.02

    # Interaction features
    ctr_vs_geo_baseline: Annotated[float, Field(ge=0.0, le=10.0)] = 1.0
    ctr_vs_device_baseline: Annotated[float, Field(ge=0.0, le=10.0)] = 1.0
    ctr_vs_creative_hist: Annotated[float, Field(ge=0.0, le=10.0)] = 1.0
    spend_per_impression: Annotated[float, Field(ge=0.0, le=100.0)] = 0.02
    bid_to_cpc_ratio: Annotated[float, Field(ge=0.0, le=100.0)] = 1.0

    # --- Categorical ---
    geo: Literal["US", "UK", "DE", "FR", "CA", "AU", "JP", "BR"] = "US"
    device_type: Literal["mobile", "desktop", "tablet"] = "desktop"
    ad_format: Literal["banner", "video", "native", "interstitial"] = "banner"
    placement: Literal["top", "sidebar", "in-feed", "footer"] = "top"

    @model_validator(mode="after")
    def derive_ratios(self) -> "AdEventFeatures":
        """Auto-compute CTR and CPC if not provided."""
        if self.ctr is None:
            self.ctr = self.clicks / max(self.impressions, 1)
        if self.cpc is None:
            self.cpc = self.spend / max(self.clicks, 1)
        return self

    @field_validator("clicks")
    @classmethod
    def clicks_le_impressions(cls, v: int, info) -> int:
        # Access impressions from model_fields_set if available
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "impressions": 50000,
            "clicks": 1200,
            "spend": 850.0,
            "bid_amount": 0.75,
            "hour_of_day": 14,
            "day_of_week": 2,
            "campaign_age_days": 45,
            "avg_ctr_geo_7d": 0.024,
            "avg_ctr_device_7d": 0.026,
            "avg_ctr_creative_7d": 0.022,
            "geo": "US",
            "device_type": "desktop",
            "ad_format": "native",
            "placement": "in-feed",
        }
    }}


class PredictRequest(BaseModel):
    """Single-item prediction request."""
    features: AdEventFeatures


class BatchPredictRequest(BaseModel):
    """Batch prediction request — up to 10,000 items."""
    items: Annotated[list[AdEventFeatures], Field(min_length=1, max_length=10_000)]


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PredictionResult(BaseModel):
    """Prediction output for a single event."""
    conversion_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    predicted_label: Literal[0, 1]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    threshold_used: float


class PredictResponse(BaseModel):
    """Single prediction API response."""
    prediction: PredictionResult
    model_version: str
    latency_ms: float


class BatchPredictResponse(BaseModel):
    """Batch prediction API response."""
    predictions: list[PredictionResult]
    count: int
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    model_version: str
    trained_at: datetime | None
    uptime_seconds: float
    total_predictions_served: int
