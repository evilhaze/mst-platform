"""
Model loader — singleton pattern for process-level caching.

The model is loaded once at startup and reused across requests.
Thread-safe initialization via a lock.
"""
from __future__ import annotations

import json
import logging
import pickle
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.config import settings

logger = logging.getLogger(__name__)

_MODEL_LOCK = threading.Lock()
_MODEL_INSTANCE: "_ModelWrapper | None" = None

MODELS_DIR = Path("models")
DEFAULT_MODEL_PATH = settings.MODEL_PATH
DEFAULT_META_PATH  = MODELS_DIR / "model_meta.json"
THRESHOLD = 0.5  # Override with optimal_threshold from model_meta.json


class _ModelWrapper:
    """
    Wraps the sklearn Pipeline with metadata and prediction helpers.
    """

    def __init__(self, pipeline, meta: dict):
        self.pipeline = pipeline
        self.meta = meta
        self.version: str = meta.get("version", "unknown")
        self.trained_at: datetime | None = (
            datetime.fromisoformat(meta["trained_at"])
            if "trained_at" in meta else None
        )
        self.threshold: float = meta.get("metrics", {}).get("optimal_threshold", THRESHOLD)
        self.total_predictions: int = 0
        self._lock = threading.Lock()

    def predict_single(self, features: dict) -> dict:
        df = pd.DataFrame([features])
        prob = self.pipeline.predict_proba(df)[0, 1]
        label = int(prob >= self.threshold)
        confidence = prob if label == 1 else 1 - prob

        with self._lock:
            self.total_predictions += 1

        return {
            "conversion_probability": float(np.clip(prob, 0.0, 1.0)),
            "predicted_label": label,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "threshold_used": self.threshold,
        }

    def predict_batch(self, features_list: list[dict]) -> list[dict]:
        df = pd.DataFrame(features_list)
        probs = self.pipeline.predict_proba(df)[:, 1]
        labels = (probs >= self.threshold).astype(int)
        confidences = np.where(labels == 1, probs, 1 - probs)

        with self._lock:
            self.total_predictions += len(features_list)

        return [
            {
                "conversion_probability": float(np.clip(p, 0.0, 1.0)),
                "predicted_label": int(lbl),
                "confidence": float(np.clip(c, 0.0, 1.0)),
                "threshold_used": self.threshold,
            }
            for p, lbl, c in zip(probs, labels, confidences)
        ]


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> "_ModelWrapper":
    """
    Load the model (thread-safe singleton).

    Raises FileNotFoundError if model artifact is missing —
    this surfaces as a 503 at startup rather than a silent 500 during inference.
    """
    global _MODEL_INSTANCE

    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    with _MODEL_LOCK:
        if _MODEL_INSTANCE is not None:  # double-checked locking
            return _MODEL_INSTANCE

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run `python -m src.models.train` first."
            )

        logger.info("Loading model from %s ...", model_path)
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)

        pipeline = artifact["pipeline"]
        meta = artifact.get("meta", {})

        _MODEL_INSTANCE = _ModelWrapper(pipeline=pipeline, meta=meta)
        logger.info("model_loaded", extra={
            "version": _MODEL_INSTANCE.version,
            "trained_at": str(_MODEL_INSTANCE.trained_at),
            "path": str(model_path),
        })

    return _MODEL_INSTANCE


def get_model() -> "_ModelWrapper":
    """FastAPI dependency — returns the loaded model singleton."""
    return load_model()
