"""
Thread-safe singleton classifier for ad-creative quality prediction.

Supports LogisticRegression and RandomForest pipelines with automatic
model selection via stratified cross-validation (AUC-ROC).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schemas import (
    ClassifierPrediction,
    CreativeFeatures,
    CreativeLabel,
)

logger = logging.getLogger(__name__)

# ── Feature columns ───────────────────────────────────────────────────────
NUMERIC_FEATURES = ["has_number", "has_urgency", "has_social_proof", "cta_strength"]
CATEGORICAL_FEATURES = ["emotion", "length_category"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ── Pipeline builder ──────────────────────────────────────────────────────
def _build_pipeline(model_type: str = "logistic") -> Pipeline:
    """
    Build an sklearn Pipeline with preprocessing + classifier.

    Parameters
    ----------
    model_type : ``"logistic"`` or ``"random_forest"``
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    if model_type == "logistic":
        estimator = LogisticRegression(
            C=1.0, max_iter=500, class_weight="balanced", random_state=42,
        )
    elif model_type == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])


# ── Helpers: schema ↔ DataFrame ───────────────────────────────────────────
def features_to_df(feat: CreativeFeatures) -> pd.DataFrame:
    """Convert a single ``CreativeFeatures`` instance to a 1-row DataFrame."""
    return pd.DataFrame(
        [
            {
                "has_number": int(feat.has_number),
                "has_urgency": int(feat.has_urgency),
                "has_social_proof": int(feat.has_social_proof),
                "cta_strength": feat.cta_strength,
                "emotion": feat.emotion.value,
                "length_category": feat.length_category.value,
            }
        ]
    )


def training_df_to_X(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the feature matrix from a training DataFrame."""
    out = df[ALL_FEATURES].copy()
    for col in NUMERIC_FEATURES:
        out[col] = out[col].astype(int)
    return out


# ── Singleton classifier ──────────────────────────────────────────────────
class CreativeClassifier:
    """
    Thread-safe singleton classifier with double-checked locking.

    Usage::

        clf = get_classifier(df)   # first call trains
        clf = get_classifier()     # subsequent calls return the same instance
        pred = clf.predict(features)
    """

    _instance: Optional["CreativeClassifier"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "CreativeClassifier":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    # ── Training ──────────────────────────────────────────────────────
    def initialize(
        self,
        df: pd.DataFrame,
        force_retrain: bool = False,
    ) -> None:
        """
        Train the classifier on *df* (expects columns from ``generate_dataset``).

        Runs 5-fold stratified CV for both LR and RF, picks the model with
        the higher mean AUC, then fits it on all data.
        """
        if self._initialized and not force_retrain:
            logger.info("Classifier already trained; skipping (use force_retrain=True).")
            return

        with self._lock:
            if self._initialized and not force_retrain:
                return
            self._fit(df)
            self._initialized = True

    def _fit(self, df: pd.DataFrame) -> None:
        X = training_df_to_X(df)
        y = (df["label"] == CreativeLabel.good.value).astype(int)
        self._train_ctr_sorted = np.sort(df["ctr"].values)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Evaluate both model types
        results: dict[str, float] = {}
        for model_type in ("logistic", "random_forest"):
            pipe = _build_pipeline(model_type)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
            mean_auc = float(scores.mean())
            results[model_type] = mean_auc
            logger.info("CV AUC %-15s: %.4f ± %.4f", model_type, mean_auc, scores.std())

        # Pick the winner
        best_type = max(results, key=results.get)  # type: ignore[arg-type]
        self._best_model_type = best_type
        self._cv_results = results
        logger.info("Selected model: %s (AUC=%.4f)", best_type, results[best_type])

        # Final fit on all data
        self._pipeline = _build_pipeline(best_type)
        self._pipeline.fit(X, y)

        # Feature importances
        self._compute_importances(X)

        # Training metrics
        y_proba = self._pipeline.predict_proba(X)[:, 1]
        self.train_auc = roc_auc_score(y, y_proba)
        y_pred = self._pipeline.predict(X)
        self.train_report = classification_report(y, y_pred, target_names=["bad", "good"])
        logger.info("Train AUC: %.4f\n%s", self.train_auc, self.train_report)

    def _compute_importances(self, X: pd.DataFrame) -> None:
        """Compute per-feature importance, mapping OHE columns back."""
        clf = self._pipeline.named_steps["classifier"]
        preprocessor = self._pipeline.named_steps["preprocessor"]

        # Get feature names after transformation
        ohe = preprocessor.named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        all_names = NUMERIC_FEATURES + cat_names

        if isinstance(clf, LogisticRegression):
            raw = np.abs(clf.coef_[0])
        else:
            raw = clf.feature_importances_

        # Aggregate OHE columns back to original categorical features
        importance: dict[str, float] = {}
        for feat_name in NUMERIC_FEATURES:
            idx = all_names.index(feat_name)
            importance[feat_name] = float(raw[idx])

        for cat_feat in CATEGORICAL_FEATURES:
            prefix = f"{cat_feat}_"
            total = sum(
                float(raw[i])
                for i, name in enumerate(all_names)
                if name.startswith(prefix)
            )
            importance[cat_feat] = total

        self._feature_importances = dict(
            sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
        )

    # ── Prediction ────────────────────────────────────────────────────
    def predict(self, feat: CreativeFeatures) -> ClassifierPrediction:
        """
        Predict label, CTR-percentile, confidence and top active feature.

        Parameters
        ----------
        feat : CreativeFeatures
            Extracted features for a single ad creative.

        Returns
        -------
        ClassifierPrediction
        """
        if not self._initialized:
            raise RuntimeError("Classifier not trained — call initialize(df) first.")

        X = features_to_df(feat)
        proba = self._pipeline.predict_proba(X)[0]
        prob_good = float(proba[1])

        label = CreativeLabel.good if prob_good >= 0.5 else CreativeLabel.bad

        # CTR percentile via the training distribution
        ctr_percentile = 0.0
        if feat.ctr is not None:
            idx = np.searchsorted(self._train_ctr_sorted, feat.ctr, side="right")
            ctr_percentile = round(100.0 * idx / len(self._train_ctr_sorted), 2)
        else:
            # Estimate from prob_good
            ctr_percentile = round(prob_good * 100, 2)

        # Top active feature — most important feature that is "active"
        top_feature = self._top_active_feature(feat)

        return ClassifierPrediction(
            label=label,
            predicted_ctr_percentile=min(ctr_percentile, 100.0),
            confidence=round(max(prob_good, 1 - prob_good), 4),
            top_feature=top_feature,
        )

    def _top_active_feature(self, feat: CreativeFeatures) -> str:
        """Return the highest-importance feature that is 'active' for this ad."""
        active: dict[str, bool] = {
            "has_number": feat.has_number,
            "has_urgency": feat.has_urgency,
            "has_social_proof": feat.has_social_proof,
            "cta_strength": feat.cta_strength >= 3,
            "emotion": feat.emotion.value != "neutral",
            "length_category": feat.length_category.value == "medium",
        }
        for name in self._feature_importances:
            if active.get(name, False):
                return name
        # Fallback: return the most important feature regardless
        return next(iter(self._feature_importances))

    # ── Persistence ───────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Serialize the trained classifier to disk via joblib."""
        if not self._initialized:
            raise RuntimeError("Cannot save an untrained classifier.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "pipeline": self._pipeline,
            "best_model_type": self._best_model_type,
            "cv_results": self._cv_results,
            "feature_importances": self._feature_importances,
            "train_ctr_sorted": self._train_ctr_sorted,
            "train_auc": self.train_auc,
            "train_report": self.train_report,
        }
        joblib.dump(state, path)
        logger.info("Classifier saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "CreativeClassifier":
        """Load a previously saved classifier (restores singleton state)."""
        path = Path(path)
        state = joblib.load(path)
        instance = cls()
        instance._pipeline = state["pipeline"]
        instance._best_model_type = state["best_model_type"]
        instance._cv_results = state["cv_results"]
        instance._feature_importances = state["feature_importances"]
        instance._train_ctr_sorted = state["train_ctr_sorted"]
        instance.train_auc = state["train_auc"]
        instance.train_report = state["train_report"]
        instance._initialized = True
        logger.info("Classifier loaded from %s (AUC=%.4f)", path, instance.train_auc)
        return instance


# ── Module-level accessor ─────────────────────────────────────────────────
def get_classifier(df: Optional[pd.DataFrame] = None) -> CreativeClassifier:
    """
    Return the singleton ``CreativeClassifier``.

    On the first call, pass *df* to train the model.  Subsequent calls
    return the same (already-trained) instance.
    """
    clf = CreativeClassifier()
    if df is not None:
        clf.initialize(df)
    return clf
