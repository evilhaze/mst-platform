"""
Pydantic schemas for request/response validation.

Strict typing ensures invalid inputs return 422 (Unprocessable Entity),
not 500 (Internal Server Error). All business rule validations here.

Model v2.1.0 trained on Avazu-enriched dataset — features must match
the exact list in models/model_meta.json.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class AdEventFeatures(BaseModel):
    """
    Features for model v2.1.0 (Avazu-enriched dataset).

    Most fields are optional with sensible defaults so that a minimal
    request body (e.g. just ``{"features": {}}``) is valid.
    """

    # --- Core Avazu numeric ---
    hour_of_day: Annotated[int, Field(ge=0, le=23)] = 12
    day_of_week: Annotated[int, Field(ge=0, le=6)] = 3
    banner_pos: Annotated[int, Field(ge=0, le=7)] = 0
    device_type: Annotated[int, Field(ge=0, le=7)] = 1
    device_conn_type: Annotated[int, Field(ge=0, le=5)] = 0

    # Avazu anonymous features (C14–C21)
    C14: int = 15706
    C15: int = 320
    C16: int = 50
    C17: int = 1722
    C18: int = 0
    C19: int = 35
    C20: int = -1
    C21: int = 79

    # Campaign numeric
    bid: Annotated[float, Field(ge=0.0)] = 0.5
    impressions: Annotated[int, Field(ge=0)] = 1000
    spend: Annotated[float, Field(ge=0.0)] = 5.0

    # Hashed high-cardinality IDs
    site_id_hash: int = 500
    app_id_hash: int = 500
    device_model_hash: int = 500

    # Frequency-encoded features
    site_category_freq: float = 0.05
    app_category_freq: float = 0.05
    geo_freq: float = 0.1
    device_conn_type_freq: float = 0.3

    # Target-encoded features
    site_category_te: float = 0.17
    app_category_te: float = 0.17
    geo_te: float = 0.17
    vertical_te: float = 0.17
    device_te: float = 0.17

    # Interaction features
    hour_device: float = 12.0
    banner_conn: float = 0.0

    # --- Categorical ---
    geo: str = "US"
    traffic_source: str = "organic"
    vertical: str = "gambling"
    device: str = "Generic"
    site_category: str = "entertainment"
    app_category: str = "unknown"

    @model_validator(mode="after")
    def derive_interactions(self) -> "AdEventFeatures":
        """Auto-compute interaction features if left at defaults."""
        if self.hour_device == 12.0 and self.hour_of_day != 12:
            self.hour_device = float(self.hour_of_day * self.device_type)
        if self.banner_conn == 0.0 and self.banner_pos != 0:
            self.banner_conn = float(self.banner_pos * self.device_conn_type)
        return self

    model_config = {"json_schema_extra": {
        "example": {
            "hour_of_day": 14,
            "day_of_week": 2,
            "banner_pos": 0,
            "device_type": 1,
            "device_conn_type": 0,
            "C14": 15706,
            "C15": 320,
            "C16": 50,
            "C17": 1722,
            "C18": 0,
            "C19": 35,
            "C20": -1,
            "C21": 79,
            "bid": 0.75,
            "impressions": 50000,
            "spend": 850.0,
            "geo": "US",
            "traffic_source": "organic",
            "vertical": "gambling",
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
