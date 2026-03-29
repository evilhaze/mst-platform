"""
CachedPredictor — Redis-backed prediction cache с метриками.
============================================================

Архитектурные решения:
  - MD5 как cache key (достаточно для коллизий при 10M/day, быстрее SHA256)
  - sort_keys=True для детерминированного хэширования dict
  - TTL по умолчанию 300s (5 минут) — баланс freshness vs hit rate
  - Thread-safe счётчики через threading.Lock (для sync использования)
  - /health эндпоинт совместим с Kubernetes liveness/readiness probe
  - Graceful degradation: если Redis недоступен — инференс без кэша,
    не крашим сервис из-за кэша

Зависимости:
  pip install redis fastapi uvicorn
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import redis as redis_lib

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Метрики
# ──────────────────────────────────────────────

@dataclass
class CacheMetrics:
    """Thread-safe счётчики для /health эндпоинта."""
    hits: int = 0
    misses: int = 0
    errors: int = 0       # Redis недоступен / сериализация упала
    model_calls: int = 0  # Сколько раз реально вызвали модель
    total_latency_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_hit(self) -> None:
        with self._lock:
            self.hits += 1

    def record_miss(self) -> None:
        with self._lock:
            self.misses += 1

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def record_model_call(self, latency_ms: float) -> None:
        with self._lock:
            self.model_calls += 1
            self.total_latency_ms += latency_ms

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate в диапазоне [0.0, 1.0]."""
        total = self.total_requests
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_model_latency_ms(self) -> float:
        calls = self.model_calls
        return self.total_latency_ms / calls if calls > 0 else 0.0

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "errors": self.errors,
                "model_calls": self.model_calls,
                "total_requests": self.total_requests,
                "cache_hit_rate": round(self.hit_rate, 4),
                "avg_model_latency_ms": round(self.avg_model_latency_ms, 2),
            }


# ──────────────────────────────────────────────
# CachedPredictor
# ──────────────────────────────────────────────

class CachedPredictor:
    """
    Обёртка над моделью с Redis-кэшом предсказаний.

    Паттерн: Cache-aside (lazy population).
    Fallback: если Redis недоступен — предсказание без кэша.

    Args:
        model:        Объект модели с методом predict(features) -> dict
        redis_client: Инициализированный redis.Redis / redis.StrictRedis
        ttl_seconds:  TTL записей в кэше (default 300s = 5 минут)
        key_prefix:   Префикс ключей (для namespace isolation в shared Redis)
        max_features_size_bytes: Защита от огромных feature-векторов
    """

    def __init__(
        self,
        model: Any,
        redis_client: redis_lib.Redis,
        ttl_seconds: int = 300,
        key_prefix: str = "pred",
        max_features_size_bytes: int = 4096,
    ):
        self.model = model
        self.redis = redis_client
        self.ttl = ttl_seconds
        self.key_prefix = key_prefix
        self.max_features_size_bytes = max_features_size_bytes
        self.metrics = CacheMetrics()

    # ── Публичный интерфейс ────────────────────

    def predict(self, features: dict) -> dict:
        """
        Возвращает предсказание (из кэша или от модели).

        Returns dict с полями:
          - ctr_predicted: float
          - cache_hit: bool
          - model_version: str (если от модели)
          - ... и любые другие поля модели
        """
        cache_key = self._build_cache_key(features)

        # 1. Попытка прочитать из кэша
        cached = self._cache_get(cache_key)
        if cached is not None:
            self.metrics.record_hit()
            cached["cache_hit"] = True
            return cached

        # 2. Cache miss — вызов модели
        self.metrics.record_miss()
        prediction = self._run_model(features)

        # 3. Запись в кэш (ошибка записи не прерывает работу)
        self._cache_set(cache_key, prediction)

        prediction["cache_hit"] = False
        return prediction

    def invalidate(self, features: dict) -> bool:
        """Явная инвалидация записи (например, после смены статуса кампании)."""
        cache_key = self._build_cache_key(features)
        try:
            deleted = self.redis.delete(cache_key)
            return deleted > 0
        except redis_lib.RedisError as exc:
            logger.warning("Redis delete failed for key %s: %s", cache_key, exc)
            return False

    def health(self) -> dict:
        """
        Состояние предиктора для /health эндпоинта.

        Совместимо с Kubernetes readiness probe:
          - status "healthy" → HTTP 200
          - status "degraded" → HTTP 200 (работаем без кэша)
          - status "unhealthy" → HTTP 503
        """
        redis_ok = self._check_redis()
        model_ok = self._check_model()

        if not model_ok:
            status = "unhealthy"
        elif not redis_ok:
            status = "degraded"  # Работаем, но без кэша
        else:
            status = "healthy"

        return {
            "status": status,
            "checks": {
                "redis": "ok" if redis_ok else "unavailable",
                "model": "ok" if model_ok else "error",
            },
            "cache": self.metrics.to_dict(),
            "config": {
                "ttl_seconds": self.ttl,
                "key_prefix": self.key_prefix,
            },
        }

    # ── Приватные методы ───────────────────────

    def _build_cache_key(self, features: dict) -> str:
        """
        Строит детерминированный cache key из feature-вектора.

        Защиты:
          - sort_keys=True: dict с одинаковыми парами → одинаковый ключ
          - Ограничение размера: защита от DoS через огромные features
        """
        serialized = json.dumps(features, sort_keys=True, default=str)

        if len(serialized.encode()) > self.max_features_size_bytes:
            raise ValueError(
                f"features size {len(serialized)} bytes exceeds "
                f"limit {self.max_features_size_bytes}"
            )

        digest = hashlib.md5(serialized.encode(), usedforsecurity=False).hexdigest()
        return f"{self.key_prefix}:{digest}"

    def _cache_get(self, key: str) -> Optional[dict]:
        """Читает из Redis. Возвращает None при любой ошибке (graceful degradation)."""
        try:
            raw = self.redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except redis_lib.RedisError as exc:
            logger.warning("Redis GET failed for key %s: %s", key, exc)
            self.metrics.record_error()
            return None
        except json.JSONDecodeError as exc:
            logger.error("Cache deserialization error for key %s: %s", key, exc)
            self.metrics.record_error()
            return None

    def _cache_set(self, key: str, value: dict) -> None:
        """Пишет в Redis. Ошибка записи не прерывает основной поток."""
        try:
            # Не кэшируем служебные поля
            payload = {k: v for k, v in value.items() if k != "cache_hit"}
            self.redis.setex(key, self.ttl, json.dumps(payload, default=str))
        except redis_lib.RedisError as exc:
            logger.warning("Redis SET failed for key %s: %s", key, exc)
            self.metrics.record_error()
        except (TypeError, ValueError) as exc:
            logger.error("Cache serialization error: %s", exc)
            self.metrics.record_error()

    def _run_model(self, features: dict) -> dict:
        """Вызывает модель с замером latency."""
        start = time.perf_counter()
        try:
            result = self.model.predict(features)
        except Exception as exc:
            logger.error("Model inference failed: %s", exc)
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            self.metrics.record_model_call(latency_ms)

        if not isinstance(result, dict):
            result = {"ctr_predicted": float(result)}

        return result

    def _check_redis(self) -> bool:
        try:
            return self.redis.ping()
        except redis_lib.RedisError:
            return False

    def _check_model(self) -> bool:
        try:
            # Пингуем модель минимальным запросом
            if hasattr(self.model, "health_check"):
                return self.model.health_check()
            return True  # Предполагаем OK если нет метода проверки
        except Exception:
            return False


# ──────────────────────────────────────────────
# FastAPI интеграция (production-ready /health)
# ──────────────────────────────────────────────

def create_app(predictor: CachedPredictor):
    """
    Создаёт FastAPI приложение с /predict и /health эндпоинтами.

    Пример:
        redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)
        predictor = CachedPredictor(model=my_model, redis_client=redis_client)
        app = create_app(predictor)
        # uvicorn src.cached_predictor:app --workers 4
    """
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError("fastapi required: pip install fastapi uvicorn")

    app = FastAPI(
        title="MST Inference API",
        version="1.0.0",
        description="Real-time CTR prediction with Redis caching",
    )

    @app.post("/predict")
    async def predict(request: Request) -> JSONResponse:
        body = await request.json()

        if not body:
            raise HTTPException(status_code=422, detail="Empty request body")

        try:
            result = predictor.predict(features=body)
            return JSONResponse(content=result)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error("Prediction error: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail="Inference failed")

    @app.get("/health")
    async def health() -> JSONResponse:
        health_data = predictor.health()

        # HTTP статус по состоянию сервиса
        if health_data["status"] == "unhealthy":
            return JSONResponse(content=health_data, status_code=503)

        return JSONResponse(content=health_data, status_code=200)

    @app.get("/metrics/cache")
    async def cache_metrics() -> JSONResponse:
        """Детальные метрики кэша (для Prometheus scraping или debugging)."""
        return JSONResponse(content=predictor.metrics.to_dict())

    return app


# ──────────────────────────────────────────────
# Демо / ручное тестирование
# ──────────────────────────────────────────────

class _DemoModel:
    """Заглушка модели для демонстрации."""

    def predict(self, features: dict) -> dict:
        # Имитируем вычисление CTR
        base_ctr = 0.05
        geo_boost = {"MSK": 1.2, "SPB": 1.1}.get(features.get("geo_id", ""), 1.0)
        hour_factor = 1.0 + 0.3 * (features.get("hour_of_day", 12) in range(9, 21))

        ctr = round(base_ctr * geo_boost * hour_factor, 4)
        return {
            "ctr_predicted": ctr,
            "model_version": "v2.3.1",
            "inference_ms": 12.4,
        }

    def health_check(self) -> bool:
        return True


if __name__ == "__main__":
    import fakeredis  # pip install fakeredis

    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    model = _DemoModel()
    predictor = CachedPredictor(model=model, redis_client=fake_redis, ttl_seconds=60)

    features = {
        "campaign_id": "camp_0001",
        "geo_id": "MSK",
        "device_type": "mobile",
        "hour_of_day": 14,
        "day_of_week": 2,
        "user_ctr_7d": 0.042,
    }

    print("=== CachedPredictor Demo ===\n")

    # Miss
    r1 = predictor.predict(features)
    print(f"1st call (miss):  ctr={r1['ctr_predicted']}, cache_hit={r1['cache_hit']}")

    # Hit
    r2 = predictor.predict(features)
    print(f"2nd call (hit):   ctr={r2['ctr_predicted']}, cache_hit={r2['cache_hit']}")

    # Health
    h = predictor.health()
    print(f"\n/health response:")
    print(json.dumps(h, indent=2))

    assert h["cache"]["cache_hit_rate"] == 0.5, "Expected 50% hit rate after 1 hit / 1 miss"
    print("\n✅ Assertions passed")
