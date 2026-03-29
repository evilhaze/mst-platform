"""
v1 API routes for ad-creative analysis & generation.

Re-exports the core router from the shared module with v1 prefix.
"""

from __future__ import annotations

# Re-export everything from the shared creatives module so existing
# imports (DriftMonitor, TokenBucketRateLimiter, verify_api_key, etc.)
# keep working from either import path.
from ..creatives import (  # noqa: F401
    DriftMonitor,
    PREDICTION_GAUGE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    TokenBucketRateLimiter,
    _analyze_limiter,
    _drift_monitor,
    _generate_limiter,
    _serialize,
    create_router,
    verify_api_key,
)
from ...errors import ErrorCode, ErrorResponse, make_error  # noqa: F401
