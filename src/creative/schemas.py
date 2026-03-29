"""
Pydantic v2 schemas for AI ad-creative analysis.

Falls back to stdlib dataclasses when pydantic is not installed.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional

# ---------------------------------------------------------------------------
# Pydantic availability probe
# ---------------------------------------------------------------------------
try:
    from pydantic import (
        BaseModel,
        Field,
        field_validator,
        model_validator,
    )

    _PYDANTIC = True
except ImportError:
    _PYDANTIC = False

SCHEMA_BACKEND = "pydantic_v2" if _PYDANTIC else "dataclass_fallback"


# ── Enums ──────────────────────────────────────────────────────────────────
class Emotion(str, Enum):
    fear = "fear"
    greed = "greed"
    excitement = "excitement"
    neutral = "neutral"


class LengthCategory(str, Enum):
    short = "short"
    medium = "medium"
    long = "long"


class Vertical(str, Enum):
    gambling = "gambling"
    betting = "betting"
    casino = "casino"


class CreativeLabel(str, Enum):
    good = "good"
    bad = "bad"


# ── Pydantic v2 models ────────────────────────────────────────────────────
if _PYDANTIC:

    class CreativeFeatures(BaseModel):
        has_number: bool
        has_urgency: bool
        has_social_proof: bool
        emotion: Emotion
        cta_strength: int = Field(ge=1, le=5)
        length_category: LengthCategory
        key_benefit: str
        ctr: Optional[float] = None
        cr: Optional[float] = None
        text: Optional[str] = None

        def to_dict(self) -> dict:
            return self.model_dump()

    class ClassifierPrediction(BaseModel):
        label: CreativeLabel
        predicted_ctr_percentile: float = Field(ge=0, le=100)
        confidence: float = Field(ge=0, le=1)
        top_feature: str

    class ImprovementTip(BaseModel):
        feature: str
        suggestion: str
        impact: str

    class AnalyzeRequest(BaseModel):
        text: str = Field(min_length=5, max_length=2000)
        geo: Optional[str] = None
        vertical: Vertical

        @field_validator("text")
        @classmethod
        def _clean_text(cls, v: str) -> str:
            v = v.strip()
            if len(v) < 5:
                raise ValueError("text must be at least 5 characters after stripping whitespace")
            if not re.search(r"[a-zA-Zа-яА-ЯёЁ]", v):
                raise ValueError("text must contain at least one letter")
            return v

    class AnalyzeResponse(BaseModel):
        features: CreativeFeatures
        prediction: ClassifierPrediction
        improvement_tips: List[ImprovementTip] = Field(min_length=3, max_length=3)
        processing_time_ms: float
        api_version: str = "1.0"

    class GenerateRequest(BaseModel):
        original_text: str
        offer: str
        geo: str
        vertical: Vertical
        n_variants: int = Field(default=5, ge=1, le=10)

    class GeneratedVariant(BaseModel):
        text: str
        rationale: str
        predicted_ctr_percentile: float = Field(ge=0, le=100)
        feature_match_score: float = Field(ge=0, le=1)
        features: CreativeFeatures

    class GenerateResponse(BaseModel):
        variants: List[GeneratedVariant]
        top_performers_used: int
        processing_time_ms: float
        api_version: str = "1.0"

    class TrainingRow(BaseModel):
        ad_id: int
        text: str
        geo: str
        vertical: Vertical
        ctr: float
        cr: float
        label: Optional[CreativeLabel] = None
        features: Optional[CreativeFeatures] = None

        @model_validator(mode="after")
        def _cr_le_ctr(self) -> "TrainingRow":
            if self.cr > self.ctr:
                raise ValueError(
                    f"cr ({self.cr}) must be <= ctr ({self.ctr})"
                )
            return self

# ── Dataclass fallback ─────────────────────────────────────────────────────
else:
    from dataclasses import asdict, dataclass, field as dc_field

    @dataclass
    class CreativeFeatures:  # type: ignore[no-redef]
        has_number: bool
        has_urgency: bool
        has_social_proof: bool
        emotion: Emotion
        cta_strength: int
        length_category: LengthCategory
        key_benefit: str
        ctr: Optional[float] = None
        cr: Optional[float] = None
        text: Optional[str] = None

        def __post_init__(self) -> None:
            if not 1 <= self.cta_strength <= 5:
                raise ValueError("cta_strength must be between 1 and 5")

        def to_dict(self) -> dict:
            return asdict(self)

    @dataclass
    class ClassifierPrediction:  # type: ignore[no-redef]
        label: CreativeLabel
        predicted_ctr_percentile: float
        confidence: float
        top_feature: str

        def __post_init__(self) -> None:
            if not 0 <= self.predicted_ctr_percentile <= 100:
                raise ValueError("predicted_ctr_percentile must be 0-100")
            if not 0 <= self.confidence <= 1:
                raise ValueError("confidence must be 0-1")

    @dataclass
    class ImprovementTip:  # type: ignore[no-redef]
        feature: str
        suggestion: str
        impact: str

    @dataclass
    class AnalyzeRequest:  # type: ignore[no-redef]
        text: str
        vertical: Vertical
        geo: Optional[str] = None

        def __post_init__(self) -> None:
            self.text = self.text.strip()
            if len(self.text) < 5:
                raise ValueError("text must be at least 5 characters after stripping whitespace")
            if len(self.text) > 2000:
                raise ValueError("text must be at most 2000 characters")
            if not re.search(r"[a-zA-Zа-яА-ЯёЁ]", self.text):
                raise ValueError("text must contain at least one letter")

    @dataclass
    class AnalyzeResponse:  # type: ignore[no-redef]
        features: CreativeFeatures
        prediction: ClassifierPrediction
        improvement_tips: List[ImprovementTip]
        processing_time_ms: float
        api_version: str = "1.0"

        def __post_init__(self) -> None:
            if len(self.improvement_tips) != 3:
                raise ValueError("improvement_tips must contain exactly 3 items")

    @dataclass
    class GenerateRequest:  # type: ignore[no-redef]
        original_text: str
        offer: str
        geo: str
        vertical: Vertical
        n_variants: int = 5

        def __post_init__(self) -> None:
            if not 1 <= self.n_variants <= 10:
                raise ValueError("n_variants must be between 1 and 10")

    @dataclass
    class GeneratedVariant:  # type: ignore[no-redef]
        text: str
        rationale: str
        predicted_ctr_percentile: float
        feature_match_score: float
        features: CreativeFeatures

        def __post_init__(self) -> None:
            if not 0 <= self.predicted_ctr_percentile <= 100:
                raise ValueError("predicted_ctr_percentile must be 0-100")
            if not 0 <= self.feature_match_score <= 1:
                raise ValueError("feature_match_score must be 0-1")

    @dataclass
    class GenerateResponse:  # type: ignore[no-redef]
        variants: List[GeneratedVariant]
        top_performers_used: int
        processing_time_ms: float
        api_version: str = "1.0"

    @dataclass
    class TrainingRow:  # type: ignore[no-redef]
        ad_id: int
        text: str
        geo: str
        vertical: Vertical
        ctr: float
        cr: float
        label: Optional[CreativeLabel] = None
        features: Optional[CreativeFeatures] = None

        def __post_init__(self) -> None:
            if self.cr > self.ctr:
                raise ValueError(
                    f"cr ({self.cr}) must be <= ctr ({self.ctr})"
                )
