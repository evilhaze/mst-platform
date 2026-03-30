"""
MST Platform — unified FastAPI application.

Combines three microservices into a single API:
  - ROI/CTR prediction   (ml-roi-predictor)
  - Creative analysis     (ad_creative)
  - Cached inference      (mst-inference-api)

Startup lifecycle:
  1. Load ML model from MLflow (or local fallback)
  2. Connect to Redis
  3. Initialise CachedPredictor
  4. Initialise Anthropic client for creative endpoints
"""

from __future__ import annotations

import logging
import os
import pickle
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis as redis_lib
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from src.api.errors import ErrorCode, ErrorResponse, make_error
from src.api.model_loader import load_model, get_model
from src.api.schemas_predict import (
    AdEventFeatures,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictionResult,
)
from src.cache.cached_predictor import CachedPredictor
from src.creative.analyzer import CreativeAnalyzer
from src.creative.classifier import CreativeClassifier, get_classifier
from src.creative.generator import CreativeGenerator
from src.creative.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    GenerateRequest,
    GenerateResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env-driven)
# ---------------------------------------------------------------------------
MLFLOW_URL: str = os.getenv("MLFLOW_URL", "")
MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ALLOWED_API_KEYS: str = os.getenv("ALLOWED_API_KEYS", "")  # comma-separated
CREATIVE_CLASSIFIER_PATH: str = os.getenv(
    "CREATIVE_CLASSIFIER_PATH", "models/creative_classifier.joblib",
)
CREATIVE_DATASET_PATH: str = os.getenv(
    "CREATIVE_DATASET_PATH", "data/creatives.csv",
)

# ---------------------------------------------------------------------------
# App state (populated during lifespan)
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------
def _load_model_mlflow() -> Any:
    """Try loading the model from MLflow Model Registry."""
    try:
        import mlflow  # type: ignore[import-untyped]

        mlflow.set_tracking_uri(MLFLOW_URL)
        model_uri = os.getenv("MLFLOW_MODEL_URI", "models:/mst-roi/Production")
        logger.info("Loading model from MLflow: %s", model_uri)
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as exc:
        logger.warning("MLflow load failed (%s); falling back to local file.", exc)
        return None


def _load_model_local() -> Any:
    """Load the sklearn pipeline model from a local pickle file."""
    model_wrapper = load_model(MODEL_PATH)
    return model_wrapper


def _connect_redis() -> redis_lib.Redis:
    """Create a Redis client from REDIS_URL with a short connect timeout."""
    return redis_lib.Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )


def _init_creative_components() -> tuple[CreativeAnalyzer | None, CreativeGenerator | None]:
    """Initialise the creative classifier, analyzer, and generator."""
    import pandas as pd

    classifier_path = Path(CREATIVE_CLASSIFIER_PATH)
    dataset_path = Path(CREATIVE_DATASET_PATH)

    try:
        if classifier_path.exists():
            classifier = CreativeClassifier.load(classifier_path)
            logger.info("Creative classifier loaded from %s", classifier_path)
        elif dataset_path.exists():
            df = pd.read_csv(dataset_path)
            classifier = get_classifier(df)
            logger.info("Creative classifier trained from %s", dataset_path)
        else:
            logger.warning(
                "No creative classifier model (%s) or dataset (%s) found; "
                "creative endpoints will return 503.",
                classifier_path, dataset_path,
            )
            return None, None

        analyzer = CreativeAnalyzer(classifier=classifier)

        generator: CreativeGenerator | None = None
        if dataset_path.exists():
            df = pd.read_csv(dataset_path)
            generator = CreativeGenerator(classifier=classifier, df=df)
        else:
            logger.warning("Dataset not found at %s; /creatives/generate disabled.", dataset_path)

        return analyzer, generator

    except Exception as exc:
        logger.error("Failed to init creative components: %s", exc, exc_info=True)
        return None, None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    startup_ts = time.monotonic()
    logger.info("Starting MST Platform …")

    # 1. ML model
    model_wrapper = None
    if MLFLOW_URL:
        model_wrapper = _load_model_mlflow()
    if model_wrapper is None:
        model_wrapper = _load_model_local()
    _state["model"] = model_wrapper
    logger.info("Model loaded (version=%s)", getattr(model_wrapper, "version", "n/a"))

    # 2. Redis
    redis_client = _connect_redis()
    _state["redis"] = redis_client
    try:
        redis_client.ping()
        logger.info("Redis connected at %s", REDIS_URL)
    except redis_lib.RedisError as exc:
        logger.warning("Redis unavailable (%s); cache disabled.", exc)

    # 3. CachedPredictor
    _state["cached_predictor"] = CachedPredictor(
        model=model_wrapper,
        redis_client=redis_client,
        ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "300")),
        key_prefix="mst",
    )

    # 4. Creative components
    analyzer, generator = _init_creative_components()
    _state["creative_analyzer"] = analyzer
    _state["creative_generator"] = generator

    # 5. Anthropic client (for creative endpoints)
    if ANTHROPIC_API_KEY:
        os.environ.setdefault("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
        logger.info("Anthropic API key configured.")
    else:
        logger.info("ANTHROPIC_API_KEY not set; creative endpoints use rule-based fallback.")

    _state["startup_time"] = datetime.now(timezone.utc)
    _state["startup_monotonic"] = startup_ts
    logger.info("MST Platform ready (%.0f ms)", (time.monotonic() - startup_ts) * 1000)

    yield

    # Shutdown
    logger.info("Shutting down MST Platform …")
    try:
        redis_client.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MST Platform API",
    description="""
## MST Platform — AI-powered advertising intelligence

### Эндпоинты:
- **POST /predict** — предсказание ROI/CTR кампании (LightGBM, ROC-AUC 0.74)
- **POST /predict/batch** — батч до 10K объектов
- **POST /creatives/analyze** — анализ текста объявления + 3 совета
- **POST /creatives/generate** — генерация 5 вариантов через Claude API
- **GET /health** — статус всех компонентов + cache_hit_rate
- **GET /metrics** — метрики моделей + дата обучения
    """,
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware: X-Process-Time-Ms + request_id
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_process_time(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:8])
    request.state.request_id = request_id
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------
def _verify_api_key(x_api_key: str | None = Header(None)) -> str | None:
    """
    Validate X-API-Key header.

    If ALLOWED_API_KEYS is not set (empty) — dev mode, no auth required.
    """
    if not ALLOWED_API_KEYS:
        return None  # dev mode
    allowed = {k.strip() for k in ALLOWED_API_KEYS.split(",") if k.strip()}
    if x_api_key not in allowed:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return x_api_key


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:8])
    error_code = {
        401: ErrorCode.INVALID_API_KEY,
        422: ErrorCode.VALIDATION_ERROR,
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        503: ErrorCode.CIRCUIT_BREAKER_OPEN,
    }.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
    return make_error(
        code=error_code,
        message=str(exc.detail),
        request_id=request_id,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:8])
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return make_error(
        code=ErrorCode.INTERNAL_ERROR,
        message="Internal server error.",
        request_id=request_id,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/predict", response_model=PredictResponse, tags=["predict"])
async def predict(
    body: PredictRequest,
    x_api_key: str | None = Header(None),
):
    """Single ROI/CTR prediction."""
    _verify_api_key(x_api_key)

    model = _state["model"]
    start = time.perf_counter()
    result = model.predict_single(body.features.model_dump())
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        prediction=PredictionResult(**result),
        model_version=getattr(model, "version", "unknown"),
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["predict"])
async def predict_batch(
    body: BatchPredictRequest,
    x_api_key: str | None = Header(None),
):
    """Batch prediction — up to 10 000 items."""
    _verify_api_key(x_api_key)

    model = _state["model"]
    features_list = [item.model_dump() for item in body.items]

    start = time.perf_counter()
    results = model.predict_batch(features_list)
    latency_ms = (time.perf_counter() - start) * 1000

    return BatchPredictResponse(
        predictions=[PredictionResult(**r) for r in results],
        count=len(results),
        model_version=getattr(model, "version", "unknown"),
        latency_ms=round(latency_ms, 2),
    )


# ═══════════════════════════════════════════════════════════════════════════
# CREATIVE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/creatives/analyze", response_model=AnalyzeResponse, tags=["creatives"])
async def creatives_analyze(
    body: AnalyzeRequest,
    x_api_key: str | None = Header(None),
):
    """Analyze ad creative text: extract features, classify, generate tips."""
    _verify_api_key(x_api_key)

    analyzer: CreativeAnalyzer | None = _state.get("creative_analyzer")
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Creative analyzer not available.")

    return await analyzer.analyze(body)


@app.post("/creatives/generate", response_model=GenerateResponse, tags=["creatives"])
async def creatives_generate(
    body: GenerateRequest,
    x_api_key: str | None = Header(None),
):
    """Generate ad creative variants scored by the trained classifier."""
    _verify_api_key(x_api_key)

    generator: CreativeGenerator | None = _state.get("creative_generator")
    if generator is None:
        raise HTTPException(status_code=503, detail="Creative generator not available.")

    return await generator.generate(body)


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH & METRICS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["ops"])
async def health():
    """
    Health check: redis, mlflow reachability, model status, cache hit rate.

    Returns 200 (healthy/degraded) or 503 (unhealthy).
    """
    model = _state.get("model")
    redis_client: redis_lib.Redis = _state["redis"]
    cached_predictor: CachedPredictor = _state["cached_predictor"]

    # Redis check
    redis_ok = False
    try:
        redis_ok = redis_client.ping()
    except redis_lib.RedisError:
        pass

    # MLflow check (simple HTTP ping to avoid DNS rebinding issues)
    mlflow_ok = False
    if MLFLOW_URL:
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{MLFLOW_URL}/health",
                headers={"Host": "localhost:5000"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                mlflow_ok = resp.status == 200
        except Exception:
            pass
    else:
        mlflow_ok = True  # not configured — not a failure

    # Model check
    model_ok = model is not None

    # Cache metrics
    cache_metrics = cached_predictor.metrics.to_dict()

    # Overall status
    if not model_ok:
        status = "unhealthy"
    elif not redis_ok:
        status = "degraded"
    else:
        status = "healthy"

    uptime = 0.0
    if "startup_monotonic" in _state:
        uptime = time.monotonic() - _state["startup_monotonic"]

    body = {
        "status": status,
        "model_version": getattr(model, "version", "unknown"),
        "trained_at": str(getattr(model, "trained_at", None)),
        "uptime_seconds": round(uptime, 1),
        "total_predictions_served": getattr(model, "total_predictions", 0),
        "checks": {
            "redis": "ok" if redis_ok else "unavailable",
            "mlflow": "ok" if mlflow_ok else "unavailable",
            "model": "ok" if model_ok else "error",
            "creative_analyzer": "ok" if _state.get("creative_analyzer") else "unavailable",
            "creative_generator": "ok" if _state.get("creative_generator") else "unavailable",
        },
        "cache": {
            "hit_rate": cache_metrics["cache_hit_rate"],
            "hits": cache_metrics["hits"],
            "misses": cache_metrics["misses"],
            "errors": cache_metrics["errors"],
        },
    }

    http_status = 503 if status == "unhealthy" else 200
    return JSONResponse(content=body, status_code=http_status)


@app.get("/metrics", tags=["ops"])
async def metrics():
    """Model quality metrics: ROC-AUC, training date, feature importances."""
    model = _state.get("model")

    # ROI model info
    roi_metrics: dict[str, Any] = {
        "model_version": getattr(model, "version", "unknown"),
        "trained_at": str(getattr(model, "trained_at", None)),
        "threshold": getattr(model, "threshold", None),
        "total_predictions": getattr(model, "total_predictions", 0),
    }

    # Pull ROC-AUC from model metadata if available
    meta = getattr(model, "meta", {})
    if "metrics" in meta:
        roi_metrics["roc_auc"] = meta["metrics"].get("roc_auc")
        roi_metrics["pr_auc"] = meta["metrics"].get("pr_auc")
        roi_metrics["f1"] = meta["metrics"].get("f1")

    # Creative classifier info
    creative_metrics: dict[str, Any] = {}
    analyzer = _state.get("creative_analyzer")
    if analyzer and hasattr(analyzer, "classifier"):
        clf = analyzer.classifier
        creative_metrics["roc_auc"] = getattr(clf, "train_auc", None)
        creative_metrics["model_type"] = getattr(clf, "_best_model_type", None)
        creative_metrics["cv_results"] = getattr(clf, "_cv_results", None)

    return {
        "roi_predictor": roi_metrics,
        "creative_classifier": creative_metrics,
    }
