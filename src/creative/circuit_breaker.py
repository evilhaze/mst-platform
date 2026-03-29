"""
Circuit Breaker for Claude API calls.

States: CLOSED → OPEN → HALF_OPEN → CLOSED (or back to OPEN).
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open and rejecting calls."""

    def __init__(self, retry_after: float) -> None:
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker is OPEN. Retry after {retry_after:.1f}s"
        )


class CircuitBreaker:
    """
    Thread-safe circuit breaker.

    Parameters
    ----------
    failure_threshold : int
        Consecutive failures before opening the circuit.
    recovery_timeout : float
        Seconds to wait in OPEN before transitioning to HALF_OPEN.
    half_open_max_calls : int
        Number of probe calls allowed in HALF_OPEN state.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[float] = None
        self._total_calls = 0
        self._rejected_calls = 0

    # ── State transitions ──────────────────────────────────────────────
    def _maybe_transition_to_half_open(self) -> None:
        """Called while holding the lock when state is OPEN."""
        if self._last_failure_time is None:
            return
        elapsed = time.monotonic() - self._last_failure_time
        if elapsed >= self.recovery_timeout:
            old = self._state
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            logger.warning("Circuit breaker: %s → HALF_OPEN", old.value)

    def record_success(self) -> None:
        """Record a successful call; close circuit if in HALF_OPEN."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    logger.warning("Circuit breaker: HALF_OPEN → CLOSED")
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._half_open_calls = 0
            else:
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call; open circuit if threshold is reached."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker: HALF_OPEN → OPEN (failure during probe)")
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.failure_threshold
            ):
                logger.warning(
                    "Circuit breaker: CLOSED → OPEN (failures=%d)",
                    self._failure_count,
                )
                self._state = CircuitState.OPEN

    # ── Guard ──────────────────────────────────────────────────────────
    def _guard(self) -> None:
        """Check whether a call is allowed. Raises on rejection."""
        with self._lock:
            self._total_calls += 1

            if self._state == CircuitState.CLOSED:
                return

            if self._state == CircuitState.OPEN:
                self._maybe_transition_to_half_open()
                if self._state == CircuitState.OPEN:
                    self._rejected_calls += 1
                    retry_after = 0.0
                    if self._last_failure_time is not None:
                        retry_after = max(
                            0.0,
                            self.recovery_timeout - (time.monotonic() - self._last_failure_time),
                        )
                    raise CircuitBreakerOpenError(retry_after)
                # Fell through to HALF_OPEN

            # HALF_OPEN — allow limited probe calls
            if self._half_open_calls >= self.half_open_max_calls:
                self._rejected_calls += 1
                raise CircuitBreakerOpenError(self.recovery_timeout)

    # ── Sync call ──────────────────────────────────────────────────────
    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *func* through the circuit breaker."""
        self._guard()
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    # ── Async call ─────────────────────────────────────────────────────
    async def async_call(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Execute an awaitable through the circuit breaker."""
        self._guard()
        try:
            result = await coro
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    # ── Properties ─────────────────────────────────────────────────────
    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                self._maybe_transition_to_half_open()
            return self._state

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state.value,
                "failures": self._failure_count,
                "total_calls": self._total_calls,
                "rejected_calls": self._rejected_calls,
                "last_failure_time": self._last_failure_time,
            }


# ── Singleton ──────────────────────────────────────────────────────────────
_cb_instance: Optional[CircuitBreaker] = None
_cb_lock = threading.Lock()


def get_claude_circuit_breaker() -> CircuitBreaker:
    """Return the module-level :class:`CircuitBreaker` singleton."""
    global _cb_instance
    if _cb_instance is None:
        with _cb_lock:
            if _cb_instance is None:
                _cb_instance = CircuitBreaker()
    return _cb_instance
