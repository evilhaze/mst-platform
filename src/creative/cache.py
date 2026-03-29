"""
Thread-safe LRU cache for feature extraction results.

Keys are SHA-256 hashes of ad text; entries expire after ``ttl_seconds``.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Optional

from .schemas import CreativeFeatures

logger = logging.getLogger(__name__)


class FeatureCache:
    """
    LRU cache with TTL for :class:`CreativeFeatures`.

    Parameters
    ----------
    max_size : int
        Maximum number of entries before LRU eviction.
    ttl_seconds : float
        Time-to-live in seconds (default 24 h).
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 86400,
    ) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._store: OrderedDict[str, tuple[CreativeFeatures, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    # ── Helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _is_expired(self, ts: float) -> bool:
        return (time.monotonic() - ts) > self.ttl_seconds

    # ── Public API ────────────────────────────────────────────────────
    def get(self, text: str) -> Optional[CreativeFeatures]:
        """
        Look up cached features for *text*.

        Returns ``None`` on miss or TTL expiry. Expired entries are
        removed automatically.
        """
        key = self._key(text)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            features, ts = entry
            if self._is_expired(ts):
                del self._store[key]
                self._misses += 1
                return None

            # Move to end (most-recently used)
            self._store.move_to_end(key)
            self._hits += 1
            return features

    def set(self, text: str, features: CreativeFeatures) -> None:
        """Insert or update *text* → *features* in the cache."""
        key = self._key(text)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (features, time.monotonic())

            # LRU eviction
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def invalidate(self, text: str) -> None:
        """Remove a specific entry."""
        key = self._key(text)
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Drop all entries and reset counters."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            }


# ── Singleton ──────────────────────────────────────────────────────────────
_cache_instance: Optional[FeatureCache] = None
_cache_lock = threading.Lock()


def get_cache() -> FeatureCache:
    """Return the module-level :class:`FeatureCache` singleton."""
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = FeatureCache()
    return _cache_instance
