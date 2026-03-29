"""Structured error codes and error response model."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_API_KEY = "INVALID_API_KEY"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    SLA_BREACH = "SLA_BREACH"
    FEATURE_EXTRACTION_FAILED = "FEATURE_EXTRACTION_FAILED"
    CLASSIFIER_ERROR = "CLASSIFIER_ERROR"
    GENERATOR_ERROR = "GENERATOR_ERROR"
    CIRCUIT_BREAKER_OPEN = "CIRCUIT_BREAKER_OPEN"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# Map error codes to HTTP status codes
_STATUS_MAP: dict[ErrorCode, int] = {
    ErrorCode.VALIDATION_ERROR: 422,
    ErrorCode.INVALID_API_KEY: 401,
    ErrorCode.RATE_LIMIT_EXCEEDED: 429,
    ErrorCode.SLA_BREACH: 504,
    ErrorCode.FEATURE_EXTRACTION_FAILED: 500,
    ErrorCode.CLASSIFIER_ERROR: 500,
    ErrorCode.GENERATOR_ERROR: 500,
    ErrorCode.CIRCUIT_BREAKER_OPEN: 503,
    ErrorCode.INTERNAL_ERROR: 500,
}


class ErrorResponse(BaseModel):
    error_code: ErrorCode
    message: str
    request_id: str
    timestamp: str
    details: Optional[dict[str, Any]] = None


def make_error(
    code: ErrorCode,
    message: str,
    request_id: str | None = None,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build a structured JSON error response."""
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]

    body = ErrorResponse(
        error_code=code,
        message=message,
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        details=details,
    )

    status = _STATUS_MAP.get(code, 500)

    logger.warning(
        "error_code=%s request_id=%s status=%d message=%s",
        code.value, request_id, status, message,
    )

    return JSONResponse(
        status_code=status,
        content=body.model_dump(),
        headers=headers,
    )
