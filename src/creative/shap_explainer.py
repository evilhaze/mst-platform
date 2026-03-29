"""
SHAP-compatible explainer using Shapley Sampling Values.

Implements the Štrumbelj & Kononenko (2014) permutation-based
approximation of Shapley values **without** the ``shap`` package.

    φ_i ≈ (1/M) Σ_{π} [ f(x_{S∪{i}}) − f(x_S) ]

where S is the set of features preceding i in permutation π.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple

import numpy as np
import pandas as pd


# ── Encoding maps ──────────────────────────────────────────────────────────
_EMOTION_MAP: dict[str, float] = {
    "fear": 0.0,
    "greed": 1.0,
    "excitement": 2.0,
    "neutral": 3.0,
}
_EMOTION_INV: dict[float, str] = {v: k for k, v in _EMOTION_MAP.items()}

_LENGTH_MAP: dict[str, float] = {
    "short": 0.0,
    "medium": 1.0,
    "long": 2.0,
}
_LENGTH_INV: dict[float, str] = {v: k for k, v in _LENGTH_MAP.items()}


# ═══════════════════════════════════════════════════════════════════════════
# ShapExplanation
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ShapExplanation:
    """Container for a single-instance Shapley explanation."""

    feature_names: List[str]
    shap_values: np.ndarray          # shape (n_features,)
    base_value: float                # E[f(x)]
    predicted_prob: float            # f(x)

    # ── Derived properties ─────────────────────────────────────────────
    @property
    def top_positive(self) -> List[Tuple[str, float]]:
        """Features with positive SHAP values, sorted descending."""
        pairs = [
            (name, float(val))
            for name, val in zip(self.feature_names, self.shap_values)
            if val > 0
        ]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs

    @property
    def top_negative(self) -> List[Tuple[str, float]]:
        """Features with negative SHAP values, sorted ascending (most negative first)."""
        pairs = [
            (name, float(val))
            for name, val in zip(self.feature_names, self.shap_values)
            if val < 0
        ]
        pairs.sort(key=lambda p: p[1])
        return pairs

    def efficiency_error(self) -> float:
        """
        Shapley efficiency check: ``|Σφ − (f(x) − E[f(x)])|``.

        Should be close to zero.
        """
        return float(abs(self.shap_values.sum() - (self.predicted_prob - self.base_value)))

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Prediction: {self.predicted_prob:.4f}  (base={self.base_value:.4f})",
            f"Efficiency error: {self.efficiency_error():.6f}",
            "",
            "  Feature contributions:",
        ]
        order = np.argsort(-np.abs(self.shap_values))
        for idx in order:
            name = self.feature_names[idx]
            val = self.shap_values[idx]
            sign = "+" if val >= 0 else ""
            lines.append(f"    {name:<20s}  {sign}{val:.4f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# ShapleyExplainer
# ═══════════════════════════════════════════════════════════════════════════
class ShapleyExplainer:
    """
    Permutation-based Shapley value estimator.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Trained pipeline with ``predict_proba``.
    background : pd.DataFrame
        Background / reference dataset (used to marginalise features).
    feature_names : list[str]
        Column names matching the pipeline's expected input.
    n_samples : int
        Number of permutations M for the Monte-Carlo approximation.
    random_state : int
        Seed for reproducibility.
    """

    _EMOTION_MAP = _EMOTION_MAP
    _LENGTH_MAP = _LENGTH_MAP

    def __init__(
        self,
        pipeline: Any,
        background: pd.DataFrame,
        feature_names: list[str],
        n_samples: int = 200,
        random_state: int = 42,
    ) -> None:
        self.pipeline = pipeline
        self.feature_names = list(feature_names)
        self.n_samples = n_samples
        self.rng = np.random.default_rng(random_state)

        # Pre-compute numeric background matrix
        self._bg_numeric = self._to_numeric(background)
        # Base value: mean prediction across the background set
        self._base_value = float(self._predict_many(self._bg_numeric).mean())

    # ── Numeric encoding / decoding ────────────────────────────────────
    def _to_numeric(self, df: pd.DataFrame) -> np.ndarray:
        """Encode a DataFrame into a float ndarray (n_rows × n_features)."""
        out = np.empty((len(df), len(self.feature_names)), dtype=np.float64)
        for j, col in enumerate(self.feature_names):
            series = df[col]
            if col == "emotion":
                out[:, j] = series.map(self._EMOTION_MAP).values
            elif col == "length_category":
                out[:, j] = series.map(self._LENGTH_MAP).values
            else:
                out[:, j] = series.astype(float).values
        return out

    def _from_numeric(self, row: np.ndarray) -> pd.DataFrame:
        """Decode a single numeric row (1-d) back to a 1-row DataFrame."""
        record: dict[str, Any] = {}
        for j, col in enumerate(self.feature_names):
            val = row[j]
            if col == "emotion":
                record[col] = _EMOTION_INV.get(float(round(val)), "neutral")
            elif col == "length_category":
                record[col] = _LENGTH_INV.get(float(round(val)), "medium")
            elif col in ("has_number", "has_urgency", "has_social_proof"):
                record[col] = int(round(val))
            elif col == "cta_strength":
                record[col] = int(round(val))
            else:
                record[col] = float(val)
        return pd.DataFrame([record])

    # ── Prediction helpers ─────────────────────────────────────────────
    def _predict_arr(self, x_arr: np.ndarray) -> float:
        """Predict P(good) for a single numeric row."""
        df = self._from_numeric(x_arr)
        return float(self.pipeline.predict_proba(df)[0, 1])

    def _predict_many(self, arr: np.ndarray) -> np.ndarray:
        """Predict P(good) for many numeric rows at once."""
        rows = [self._from_numeric(arr[i]) for i in range(len(arr))]
        df = pd.concat(rows, ignore_index=True)
        return self.pipeline.predict_proba(df)[:, 1].astype(np.float64)

    # ── Core Shapley estimation ────────────────────────────────────────
    def explain(self, x: pd.DataFrame) -> ShapExplanation:
        """
        Estimate Shapley values for a single instance.

        Parameters
        ----------
        x : pd.DataFrame
            A 1-row DataFrame with the same columns as the training data.
        """
        x_num = self._to_numeric(x)[0]  # 1-d
        n_feat = len(self.feature_names)
        n_bg = len(self._bg_numeric)

        phi = np.zeros(n_feat, dtype=np.float64)

        for _ in range(self.n_samples):
            perm = self.rng.permutation(n_feat)
            # Pick a random background sample to fill absent features
            bg_idx = self.rng.integers(0, n_bg)
            bg_row = self._bg_numeric[bg_idx]

            # Build z_with and z_without incrementally
            z = bg_row.copy()
            prev_pred = self._predict_arr(z)

            for feat_idx in perm:
                z[feat_idx] = x_num[feat_idx]
                curr_pred = self._predict_arr(z)
                phi[feat_idx] += curr_pred - prev_pred
                prev_pred = curr_pred

        phi /= self.n_samples
        predicted_prob = self._predict_arr(x_num)

        return ShapExplanation(
            feature_names=self.feature_names,
            shap_values=phi,
            base_value=self._base_value,
            predicted_prob=predicted_prob,
        )

    def explain_batch(
        self,
        X: pd.DataFrame,
        n_instances: int = 20,
    ) -> list[ShapExplanation]:
        """Explain up to *n_instances* rows from *X*."""
        n = min(n_instances, len(X))
        return [self.explain(X.iloc[[i]]) for i in range(n)]

    def global_importance(
        self,
        X: pd.DataFrame,
        n_instances: int = 50,
    ) -> dict[str, float]:
        """
        Compute global feature importance as normalised mean |φ_i|.

        Returns a dict sorted by importance (descending), values sum to 1.
        """
        explanations = self.explain_batch(X, n_instances=n_instances)
        abs_shap = np.array([np.abs(e.shap_values) for e in explanations])
        mean_abs = abs_shap.mean(axis=0)
        total = mean_abs.sum()
        if total > 0:
            mean_abs /= total

        importance = {
            name: round(float(val), 4)
            for name, val in zip(self.feature_names, mean_abs)
        }
        return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))


# ── Factory ────────────────────────────────────────────────────────────────
def build_explainer(
    pipeline: Any,
    X_train: pd.DataFrame,
    feature_names: list[str],
    n_samples: int = 150,
) -> ShapleyExplainer:
    """
    Convenience factory: build a :class:`ShapleyExplainer` using a
    subsample of the training data as background.
    """
    bg_size = min(100, len(X_train))
    background = X_train.sample(n=bg_size, random_state=42)
    return ShapleyExplainer(
        pipeline=pipeline,
        background=background,
        feature_names=feature_names,
        n_samples=n_samples,
        random_state=42,
    )
