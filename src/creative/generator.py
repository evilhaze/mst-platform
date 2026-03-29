"""
Few-shot creative generator: produces ad-text variants via Claude
(with rule-based fallback) and scores them with the trained classifier.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

import numpy as np
import pandas as pd

from .analyzer import extract_batch_claude, _rule_based_extract
from .schemas import (
    CreativeFeatures,
    GenerateRequest,
    GenerateResponse,
    GeneratedVariant,
    Vertical,
)

logger = logging.getLogger(__name__)

# ── Anthropic probe ────────────────────────────────────────────────────────
try:
    from anthropic import AsyncAnthropic

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

CLAUDE_MODEL = "claude-sonnet-4-5"
MAX_CONCURRENT = 5
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0

# ── Encoding maps (same as shap_explainer) ─────────────────────────────────
_EMOTION_NUM = {"fear": 0.0, "greed": 1.0, "excitement": 2.0, "neutral": 3.0}
_LENGTH_NUM = {"short": 0.0, "medium": 1.0, "long": 2.0}


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════
def _feat_to_vector(feat: CreativeFeatures) -> np.ndarray:
    """Encode a CreativeFeatures into a 6-d float vector."""
    return np.array(
        [
            float(feat.has_number),
            float(feat.has_urgency),
            float(feat.has_social_proof),
            _EMOTION_NUM.get(feat.emotion.value, 3.0),
            feat.cta_strength / 5.0,
            _LENGTH_NUM.get(feat.length_category.value, 1.0),
        ],
        dtype=np.float64,
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (0 if either is zero-norm)."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _feature_match_score(
    feat: CreativeFeatures,
    top_profiles: list[np.ndarray],
) -> float:
    """Mean cosine similarity between *feat* and the top-performer profiles."""
    if not top_profiles:
        return 0.0
    vec = _feat_to_vector(feat)
    sims = [_cosine_similarity(vec, prof) for prof in top_profiles]
    return float(np.clip(np.mean(sims), 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# Few-shot prompt builder
# ═══════════════════════════════════════════════════════════════════════════
def _build_few_shot_prompt(
    offer: str,
    geo: str,
    vertical: Vertical,
    top_performers: list[dict[str, Any]],
    n_variants: int,
) -> str:
    """Build a few-shot prompt with real top-10 ads as examples."""
    examples_block = "\n".join(
        f'  {i+1}. (CTR={p["ctr"]:.4f}) "{p["text"]}"'
        for i, p in enumerate(top_performers[:10])
    )

    return f"""\
You are an expert performance-marketing copywriter for the {vertical.value} vertical.

Below are the top-performing ad creatives from real campaigns (sorted by CTR):

{examples_block}

PATTERNS that work best (learn from the examples above):
- Urgency / scarcity signals ("only today", "last chance", "hurry")
- Specific numbers (bonus %, dollar amounts, free-spin counts)
- Strong imperative CTA ("grab", "claim", "start winning now!")
- Medium length (7-12 words) — not too short, not too long
- Emotional triggers: greed or excitement work best
- Social proof when possible ("5000+ players already won")

YOUR TASK:
Write {n_variants} new ad creative variants for:
- Offer: {offer}
- Geo: {geo}
- Vertical: {vertical.value}
- Language: Russian

Return ONLY a JSON array (no markdown, no commentary):
[
  {{"text": "...", "rationale": "..."}},
  ...
]

Each "rationale" should briefly explain which winning patterns the variant uses.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Claude generation with retry
# ═══════════════════════════════════════════════════════════════════════════
async def _generate_via_claude(
    client: "AsyncAnthropic",
    prompt: str,
    n_variants: int,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, str]]:
    """Call Claude to generate variants; returns list of {{text, rationale}}."""
    async with semaphore:
        last_exc: Exception | None = None
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                message = await client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                body = message.content[0].text.strip()
                body = re.sub(r"^```(?:json)?\s*", "", body)
                body = re.sub(r"\s*```$", "", body.strip())
                variants = json.loads(body)
                if isinstance(variants, list):
                    return [
                        {
                            "text": v.get("text", ""),
                            "rationale": v.get("rationale", ""),
                        }
                        for v in variants
                        if isinstance(v, dict) and v.get("text")
                    ][:n_variants]
            except Exception as exc:
                last_exc = exc
                if attempt < RETRY_ATTEMPTS:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Claude gen attempt %d/%d failed (%s), retry in %.1fs",
                        attempt, RETRY_ATTEMPTS, exc, delay,
                    )
                    await asyncio.sleep(delay)

        logger.error("Claude generation failed after %d attempts: %s", RETRY_ATTEMPTS, last_exc)
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Rule-based fallback generator
# ═══════════════════════════════════════════════════════════════════════════
_RULE_TEMPLATES: list[dict[str, str]] = [
    {
        "tpl": "Только сегодня! {offer} — бонус {num}%! Успей забрать!",
        "rationale": "Urgency ('только сегодня', 'успей') + specific number + strong CTA.",
    },
    {
        "tpl": "Уже {big_num}+ игроков забрали {offer}! Присоединяйся и выигрывай!",
        "rationale": "Social proof (player count) + imperative CTA + excitement.",
    },
    {
        "tpl": "Забери свой {offer} до {num}$! Жми и начинай выигрывать прямо сейчас!",
        "rationale": "Greed trigger ('забери') + specific dollar amount + double CTA.",
    },
    {
        "tpl": "Осталось {small_num} мест! {offer} — получи {num}% бонус!",
        "rationale": "Scarcity ('осталось N мест') + number + urgency.",
    },
    {
        "tpl": "Не упусти {offer}! Бонус {num}% сгорает через {hours} часа!",
        "rationale": "Fear of missing out + urgency timer + specific bonus percentage.",
    },
    {
        "tpl": "Почувствуй адреналин! {offer} — ставь и забирай до {num}$!",
        "rationale": "Excitement trigger ('адреналин') + imperative CTA + dollar amount.",
    },
    {
        "tpl": "{offer} — {num} фриспинов бесплатно! Активируй и выводи выигрыш!",
        "rationale": "Greed ('бесплатно') + specific freespin count + double CTA.",
    },
    {
        "tpl": "Более {big_num} выплат сегодня! {offer} ждёт тебя — регистрируйся!",
        "rationale": "Social proof (payout count) + urgency ('сегодня') + CTA.",
    },
    {
        "tpl": "Последний шанс! {offer} + кэшбэк {num}%! Жми — не пожалеешь!",
        "rationale": "Urgency ('последний шанс') + cashback number + fear reversal + CTA.",
    },
    {
        "tpl": "Удвой свой депозит с {offer}! До {num}$ бонус — начни прямо сейчас!",
        "rationale": "Greed ('удвой') + specific amount + urgency ('прямо сейчас') + CTA.",
    },
]


def _generate_rule_based(
    offer: str,
    geo: str,
    vertical: Vertical,
    top_performers: list[dict[str, Any]],
    n_variants: int,
) -> list[dict[str, str]]:
    """Deterministic template-based fallback generator."""
    rng = np.random.default_rng(hash((offer, geo, vertical.value)) % (2**31))
    indices = rng.permutation(len(_RULE_TEMPLATES))[:n_variants]

    results: list[dict[str, str]] = []
    for idx in indices:
        tpl_info = _RULE_TEMPLATES[idx]
        text = tpl_info["tpl"].format(
            offer=offer,
            num=int(rng.integers(50, 500)),
            big_num=int(rng.integers(1000, 50000)),
            small_num=int(rng.integers(3, 30)),
            hours=int(rng.integers(1, 24)),
        )
        results.append({"text": text, "rationale": tpl_info["rationale"]})
    return results


# ═══════════════════════════════════════════════════════════════════════════
# CreativeGenerator
# ═══════════════════════════════════════════════════════════════════════════
class CreativeGenerator:
    """
    Few-shot ad-creative generator backed by Claude + trained classifier.

    Parameters
    ----------
    classifier : CreativeClassifier
        Trained classifier for scoring generated variants.
    df : pd.DataFrame
        Training dataset (used to select top performers as few-shot examples).
    """

    def __init__(self, classifier: Any, df: pd.DataFrame) -> None:
        self.classifier = classifier
        self._top_performers = self._select_top_performers(df)
        self._top_profiles = self._build_top_profiles(df)

    @staticmethod
    def _select_top_performers(df: pd.DataFrame) -> list[dict[str, Any]]:
        """Select top-10 ads by CTR."""
        top = df.nlargest(10, "ctr")
        return top[["text", "ctr"]].to_dict("records")

    def _build_top_profiles(self, df: pd.DataFrame) -> list[np.ndarray]:
        """Build feature vectors for the top-10 performers."""
        top = df.nlargest(10, "ctr")
        profiles: list[np.ndarray] = []
        for _, row in top.iterrows():
            feat = _rule_based_extract(row["text"])
            profiles.append(_feat_to_vector(feat))
        return profiles

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Generate, extract features, score, and rank ad-creative variants.

        1. Generate text variants (Claude → rule-based fallback).
        2. Extract features for each variant via Claude / rule-based.
        3. Predict label + CTR percentile via the classifier.
        4. Compute feature_match_score vs. top performers.
        5. Return variants sorted by predicted_ctr_percentile descending.
        """
        t0 = time.perf_counter()

        # Step 1: generate raw variants
        raw_variants = await self._generate_raw(request)

        # Step 2: extract features for all variants in parallel
        texts = [v["text"] for v in raw_variants]
        features_list = await extract_batch_claude(texts)

        # Steps 3-4: score each variant
        scored: list[GeneratedVariant] = []
        for raw_v, feat in zip(raw_variants, features_list):
            prediction = self.classifier.predict(feat)
            match_score = _feature_match_score(feat, self._top_profiles)

            scored.append(
                GeneratedVariant(
                    text=raw_v["text"],
                    rationale=raw_v["rationale"],
                    predicted_ctr_percentile=prediction.predicted_ctr_percentile,
                    feature_match_score=round(match_score, 4),
                    features=feat,
                )
            )

        # Step 5: sort by predicted CTR percentile descending
        scored.sort(key=lambda v: v.predicted_ctr_percentile, reverse=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return GenerateResponse(
            variants=scored,
            top_performers_used=len(self._top_performers),
            processing_time_ms=round(elapsed_ms, 2),
        )

    async def _generate_raw(self, request: GenerateRequest) -> list[dict[str, str]]:
        """Try Claude generation, fall back to rule-based."""
        prompt = _build_few_shot_prompt(
            offer=request.offer,
            geo=request.geo,
            vertical=request.vertical,
            top_performers=self._top_performers,
            n_variants=request.n_variants,
        )

        if _HAS_ANTHROPIC:
            try:
                client = AsyncAnthropic()
                semaphore = asyncio.Semaphore(MAX_CONCURRENT)
                variants = await _generate_via_claude(
                    client, prompt, request.n_variants, semaphore,
                )
                if variants:
                    return variants
                logger.warning("Claude returned empty; using rule-based fallback.")
            except Exception as exc:
                logger.warning("Claude unavailable (%s); using rule-based fallback.", exc)

        return _generate_rule_based(
            offer=request.offer,
            geo=request.geo,
            vertical=request.vertical,
            top_performers=self._top_performers,
            n_variants=request.n_variants,
        )
