"""
Creative analyzer: async Claude API feature extraction, statistical
pattern analysis, and improvement-tip generation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .cache import get_cache
from .circuit_breaker import CircuitBreakerOpenError, get_claude_circuit_breaker
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClassifierPrediction,
    CreativeFeatures,
    CreativeLabel,
    Emotion,
    ImprovementTip,
    LengthCategory,
    Vertical,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
MAX_CONCURRENT = 10
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0

CLAUDE_MODEL = "claude-sonnet-4-5"

CLAUDE_SYSTEM = (
    "You are a senior performance-marketing analyst specialising in "
    "gambling, betting and casino verticals.  You evaluate ad creatives "
    "and extract structured feature vectors used for CTR prediction.  "
    "Always respond with valid JSON only — no markdown, no commentary."
)

CLAUDE_PROMPT = """\
Analyze the following ad creative and return a JSON object with exactly \
these keys (no extra keys):

{{
  "has_number": <bool — true if the text contains any digit>,
  "has_urgency": <bool — true if it creates urgency / scarcity>,
  "has_social_proof": <bool — true if it references other users / popularity>,
  "emotion": <"fear" | "greed" | "excitement" | "neutral">,
  "cta_strength": <int 1-5 — how strong is the call-to-action>,
  "length_category": <"short" | "medium" | "long">,
  "key_benefit": <string — main benefit offered, e.g. "bonus", "freespins">
}}

Ad creative text:
\"\"\"{text}\"\"\"
"""

# ── Anthropic client probe ─────────────────────────────────────────────────
try:
    from anthropic import AsyncAnthropic

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


# ── Keyword lists (mirrored from dataset.py for standalone use) ────────────
_URGENCY_KW = [
    "только сегодня", "осталось", "последний шанс", "торопись",
    "успей", "сейчас", "прямо сейчас", "сгорает", "истекает",
    "заканчивается", "через час", "не упусти", "упустишь",
    "пока ты думаешь",
]

_SOCIAL_PROOF_KW = [
    "уже", "игроков", "человек", "нас уже", "довольных",
    "выплат за", "присоединяйся", "сообщество",
    "не могут ошибаться",
]

_FEAR_KW = [
    "не упусти", "теряешь", "пока ты думаешь", "упустишь",
    "пожалеешь", "в минус", "сгорает",
]

_GREED_KW = [
    "забери", "удвой", "бесплатно", "без риска", "куш",
    "выводи", "получи", "забирай",
]

_EXCITEMENT_KW = [
    "адреналин", "удачу", "драйв", "эмоции", "азарт",
    "невероятные", "на максимум", "джекпот",
]

_STRONG_CTA_KW = [
    "жми", "регистрируйся", "активируй", "начинай",
    "забирай", "начни", "делай ставку",
]

_BENEFIT_KW = {
    "бонус": "bonus",
    "фриспин": "freespins",
    "фрибет": "freebet",
    "кэшбэк": "cashback",
    "депозит": "deposit_bonus",
    "джекпот": "jackpot",
    "выигрыш": "winnings",
    "выплат": "fast_payouts",
    "лицензи": "licensed",
}


# ── Rule-based extraction (keyword / regex fallback) ───────────────────────
def _has_keywords(text_lower: str, keywords: list[str]) -> bool:
    return any(kw in text_lower for kw in keywords)


def _rule_based_extract(text: str) -> CreativeFeatures:
    """Keyword / regex feature extraction — used when Claude is unavailable."""
    low = text.lower()

    has_number = bool(re.search(r"\d+", text))
    has_urgency = _has_keywords(low, _URGENCY_KW)
    has_social_proof = _has_keywords(low, _SOCIAL_PROOF_KW)

    fear_score = sum(1 for kw in _FEAR_KW if kw in low)
    greed_score = sum(1 for kw in _GREED_KW if kw in low)
    excitement_score = sum(1 for kw in _EXCITEMENT_KW if kw in low)

    scores = {
        Emotion.fear: fear_score,
        Emotion.greed: greed_score,
        Emotion.excitement: excitement_score,
    }
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    emotion = best if scores[best] > 0 else Emotion.neutral

    cta_hits = sum(1 for kw in _STRONG_CTA_KW if kw in low)
    has_excl = "!" in text
    has_imp = bool(re.search(r"(жми|забери|начни|играй|делай|крути|ставь|испытай)", low))
    cta_strength = max(1, min(5, cta_hits + int(has_excl) + int(has_imp)))

    wc = len(text.split())
    if wc <= 6:
        length_category = LengthCategory.short
    elif wc <= 12:
        length_category = LengthCategory.medium
    else:
        length_category = LengthCategory.long

    key_benefit = "general"
    for kw, benefit in _BENEFIT_KW.items():
        if kw in low:
            key_benefit = benefit
            break

    return CreativeFeatures(
        has_number=has_number,
        has_urgency=has_urgency,
        has_social_proof=has_social_proof,
        emotion=emotion,
        cta_strength=cta_strength,
        length_category=length_category,
        key_benefit=key_benefit,
        text=text,
    )


# ── Claude response parsing ───────────────────────────────────────────────
def _parse_claude_response(raw: dict[str, Any], text: str) -> CreativeFeatures:
    """
    Parse the JSON dict returned by Claude into ``CreativeFeatures``.

    Every field has a defensive fallback so a partial / malformed response
    still produces a valid object.
    """
    fallback = _rule_based_extract(text)

    def _bool(key: str, default: bool) -> bool:
        v = raw.get(key)
        return bool(v) if v is not None else default

    def _enum(key: str, enum_cls: type, default: Any) -> Any:
        v = raw.get(key)
        if v is None:
            return default
        try:
            return enum_cls(v)
        except (ValueError, KeyError):
            return default

    has_number = _bool("has_number", fallback.has_number)
    has_urgency = _bool("has_urgency", fallback.has_urgency)
    has_social_proof = _bool("has_social_proof", fallback.has_social_proof)
    emotion = _enum("emotion", Emotion, fallback.emotion)
    length_category = _enum("length_category", LengthCategory, fallback.length_category)

    cta_raw = raw.get("cta_strength")
    if isinstance(cta_raw, (int, float)) and 1 <= int(cta_raw) <= 5:
        cta_strength = int(cta_raw)
    else:
        cta_strength = fallback.cta_strength

    key_benefit = raw.get("key_benefit")
    if not isinstance(key_benefit, str) or not key_benefit.strip():
        key_benefit = fallback.key_benefit

    return CreativeFeatures(
        has_number=has_number,
        has_urgency=has_urgency,
        has_social_proof=has_social_proof,
        emotion=emotion,
        cta_strength=cta_strength,
        length_category=length_category,
        key_benefit=key_benefit,
        text=text,
    )


# ── Async Claude call with retry ──────────────────────────────────────────
async def _call_claude_with_retry(
    client: "AsyncAnthropic",
    text: str,
    semaphore: asyncio.Semaphore,
) -> CreativeFeatures:
    """
    Call Claude with exponential back-off, semaphore-based concurrency,
    and circuit breaker protection.

    Falls back to rule-based extraction after all retries are exhausted
    or when the circuit breaker is open.
    """
    cb = get_claude_circuit_breaker()

    # Fast-fail if circuit is open
    try:
        cb._guard()
    except CircuitBreakerOpenError:
        logger.info("circuit_breaker_fallback=True text=%s", text[:40])
        return _rule_based_extract(text)

    async with semaphore:
        last_exc: Optional[Exception] = None
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                async def _do_call() -> CreativeFeatures:
                    message = await client.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=512,
                        system=CLAUDE_SYSTEM,
                        messages=[
                            {"role": "user", "content": CLAUDE_PROMPT.format(text=text)},
                        ],
                    )
                    body = message.content[0].text
                    body = re.sub(r"^```(?:json)?\s*", "", body.strip())
                    body = re.sub(r"\s*```$", "", body.strip())
                    raw = json.loads(body)
                    return _parse_claude_response(raw, text)

                result = await cb.async_call(_do_call())
                return result

            except CircuitBreakerOpenError:
                logger.info("circuit_breaker_fallback=True text=%s", text[:40])
                return _rule_based_extract(text)

            except Exception as exc:
                last_exc = exc
                if attempt < RETRY_ATTEMPTS:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Claude attempt %d/%d failed (%s), retrying in %.1fs …",
                        attempt, RETRY_ATTEMPTS, exc, delay,
                    )
                    await asyncio.sleep(delay)

        logger.error(
            "Claude extraction failed after %d attempts (%s); using rule-based fallback.",
            RETRY_ATTEMPTS, last_exc,
        )
        return _rule_based_extract(text)


# ── Batch extraction ──────────────────────────────────────────────────────
async def extract_batch_claude(texts: list[str]) -> list[CreativeFeatures]:
    """
    Extract features for a batch of ad texts via Claude (async, parallel).

    Results are cached (LRU, 24 h TTL).  Falls back entirely to rule-based
    extraction when the ``anthropic`` package is not installed or
    ``ANTHROPIC_API_KEY`` is not set.
    """
    cache = get_cache()
    results: list[Optional[CreativeFeatures]] = [None] * len(texts)
    texts_to_fetch: list[tuple[int, str]] = []  # (original_index, text)

    # Check cache first
    for i, text in enumerate(texts):
        cached = cache.get(text)
        if cached is not None:
            results[i] = cached
            logger.debug("cache_hit=True text=%s", text[:40])
        else:
            texts_to_fetch.append((i, text))
            logger.debug("cache_hit=False text=%s", text[:40])

    if not texts_to_fetch:
        return results  # type: ignore[return-value]

    # Fetch uncached texts
    if not _HAS_ANTHROPIC:
        logger.info("anthropic package not available — using rule-based extraction.")
        for idx, text in texts_to_fetch:
            feat = _rule_based_extract(text)
            cache.set(text, feat)
            results[idx] = feat
        return results  # type: ignore[return-value]

    try:
        client = AsyncAnthropic()
    except Exception as exc:
        logger.warning("Cannot create AsyncAnthropic client (%s); rule-based fallback.", exc)
        for idx, text in texts_to_fetch:
            feat = _rule_based_extract(text)
            cache.set(text, feat)
            results[idx] = feat
        return results  # type: ignore[return-value]

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [_call_claude_with_retry(client, t, semaphore) for _, t in texts_to_fetch]
    fetched = await asyncio.gather(*tasks)

    for (idx, text), feat in zip(texts_to_fetch, fetched):
        cache.set(text, feat)
        results[idx] = feat

    return results  # type: ignore[return-value]


def extract_batch_sync(texts: list[str]) -> list[CreativeFeatures]:
    """Synchronous wrapper around :func:`extract_batch_claude`."""
    return asyncio.run(extract_batch_claude(texts))


# ═══════════════════════════════════════════════════════════════════════════
# PatternAnalyzer — statistical pattern mining
# ═══════════════════════════════════════════════════════════════════════════
class PatternAnalyzer:
    """
    Run statistical tests that relate creative features to CTR.

    Parameters
    ----------
    df : pd.DataFrame
        Training dataset (must contain ``ctr`` and the feature columns).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self._results: list[dict[str, Any]] | None = None

    # ── Statistical tests ──────────────────────────────────────────────
    def _test_binary(self, col: str) -> dict[str, Any]:
        """Mann-Whitney U + point-biserial correlation for a boolean column."""
        mask = self.df[col].astype(bool)
        group_1 = self.df.loc[mask, "ctr"]
        group_0 = self.df.loc[~mask, "ctr"]

        if len(group_1) < 2 or len(group_0) < 2:
            return {
                "feature": col, "test": "mann_whitney_u",
                "statistic": np.nan, "p_value": 1.0,
                "effect_size": 0.0, "direction": "n/a",
            }

        u_stat, p_val = sp_stats.mannwhitneyu(group_1, group_0, alternative="two-sided")
        corr, _ = sp_stats.pointbiserialr(mask.astype(int), self.df["ctr"])

        return {
            "feature": col,
            "test": "mann_whitney_u",
            "statistic": float(u_stat),
            "p_value": float(p_val),
            "effect_size": round(float(corr), 4),
            "direction": "positive" if corr > 0 else "negative",
        }

    def _test_ordinal(self, col: str) -> dict[str, Any]:
        """Spearman correlation for an ordinal column."""
        rho, p_val = sp_stats.spearmanr(self.df[col], self.df["ctr"])
        return {
            "feature": col,
            "test": "spearman",
            "statistic": round(float(rho), 4),
            "p_value": float(p_val),
            "effect_size": round(float(rho), 4),
            "direction": "positive" if rho > 0 else "negative",
        }

    def _test_categorical(self, col: str) -> dict[str, Any]:
        """Kruskal-Wallis H test for a categorical column."""
        groups = [g["ctr"].values for _, g in self.df.groupby(col) if len(g) >= 2]
        if len(groups) < 2:
            return {
                "feature": col, "test": "kruskal_wallis",
                "statistic": np.nan, "p_value": 1.0,
                "effect_size": 0.0, "direction": "n/a",
            }

        h_stat, p_val = sp_stats.kruskal(*groups)
        # Eta-squared approximation: η² = (H - k + 1) / (N - k)
        k = len(groups)
        n = len(self.df)
        eta_sq = (h_stat - k + 1) / (n - k) if n > k else 0.0
        eta_sq = max(eta_sq, 0.0)

        return {
            "feature": col,
            "test": "kruskal_wallis",
            "statistic": float(h_stat),
            "p_value": float(p_val),
            "effect_size": round(eta_sq, 4),
            "direction": "varies",
        }

    # ── Public API ─────────────────────────────────────────────────────
    def analyze(self) -> list[dict[str, Any]]:
        """Run all statistical tests and return results sorted by p-value."""
        results: list[dict[str, Any]] = []

        for col in ("has_number", "has_urgency", "has_social_proof"):
            results.append(self._test_binary(col))

        results.append(self._test_ordinal("cta_strength"))

        for col in ("emotion", "length_category"):
            results.append(self._test_categorical(col))

        results.sort(key=lambda r: r["p_value"])
        self._results = results
        return results

    def significant_patterns(self, alpha: float = 0.05) -> list[dict[str, Any]]:
        """Return only patterns with ``p_value < alpha``."""
        if self._results is None:
            self.analyze()
        return [r for r in self._results if r["p_value"] < alpha]  # type: ignore[union-attr]

    def summary(self) -> str:
        """Human-readable summary of all tests."""
        if self._results is None:
            self.analyze()

        lines = ["Feature Pattern Analysis", "=" * 50]
        for r in self._results:  # type: ignore[union-attr]
            sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else ""))
            lines.append(
                f"  {r['feature']:<20s}  p={r['p_value']:<10.2e}  "
                f"effect={r['effect_size']:+.4f}  {r['direction']:>8s}  {sig}"
            )
        sig_count = sum(1 for r in self._results if r["p_value"] < 0.05)  # type: ignore[union-attr]
        lines.append(f"\nSignificant features (p<0.05): {sig_count}/{len(self._results)}")  # type: ignore[arg-type]

        cache_stats = get_cache().stats
        lines.append(
            f"Cache: size={cache_stats['size']}, hits={cache_stats['hits']}, "
            f"misses={cache_stats['misses']}, hit_rate={cache_stats['hit_rate']:.1%}"
        )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CreativeAnalyzer — full analysis pipeline
# ═══════════════════════════════════════════════════════════════════════════
class CreativeAnalyzer:
    """
    High-level analyzer: feature extraction → classification → tips.

    Parameters
    ----------
    classifier : CreativeClassifier
        A trained classifier instance.
    """

    def __init__(self, classifier: Any) -> None:
        self.classifier = classifier
        self._tip_rules = self._build_tip_rules()

    # ── Tip rules ──────────────────────────────────────────────────────
    @staticmethod
    def _build_tip_rules() -> list[dict[str, Any]]:
        """
        Return a priority-ordered list of improvement rules.

        Each rule contains:
        - ``check``: callable(CreativeFeatures) → bool  (True = tip applies)
        - ``feature``: feature name
        - ``suggestion``: actionable advice
        - ``impact``: expected impact description
        """
        return [
            {
                "check": lambda f: not f.has_urgency,
                "feature": "has_urgency",
                "suggestion": (
                    "Add urgency signals: time limits ('only today', "
                    "'last chance') or scarcity ('only N spots left')."
                ),
                "impact": "CTR +30-50% — urgency is the strongest single driver.",
            },
            {
                "check": lambda f: not f.has_number,
                "feature": "has_number",
                "suggestion": (
                    "Include specific numbers: bonus percentages, free-spin "
                    "counts, or dollar amounts to increase credibility."
                ),
                "impact": "CTR +20-35% — numbers boost perceived value and specificity.",
            },
            {
                "check": lambda f: f.emotion == Emotion.neutral,
                "feature": "emotion",
                "suggestion": (
                    "Replace neutral tone with greed ('grab your bonus') or "
                    "excitement ('feel the thrill') emotional triggers."
                ),
                "impact": "CTR +25-35% — emotional ads outperform neutral by a wide margin.",
            },
            {
                "check": lambda f: f.cta_strength < 3,
                "feature": "cta_strength",
                "suggestion": (
                    "Strengthen the call-to-action: use imperative verbs "
                    "('claim', 'grab', 'start winning') and exclamation marks."
                ),
                "impact": "CTR +8% per CTA level — strong CTAs can double engagement.",
            },
            {
                "check": lambda f: not f.has_social_proof,
                "feature": "has_social_proof",
                "suggestion": (
                    "Add social proof: mention the number of active players, "
                    "recent winners, or total payouts."
                ),
                "impact": "CTR +15-25% — social proof builds trust and FOMO.",
            },
            {
                "check": lambda f: f.length_category != LengthCategory.medium,
                "feature": "length_category",
                "suggestion": (
                    "Aim for medium length (7-12 words). Too short lacks "
                    "persuasion; too long loses attention."
                ),
                "impact": "CTR +10% — medium-length creatives have optimal engagement.",
            },
        ]

    def get_improvement_tips(self, feat: CreativeFeatures) -> list[ImprovementTip]:
        """Return exactly 3 improvement tips for the given features."""
        tips: list[ImprovementTip] = []

        # First pass: tips whose condition fires
        for rule in self._tip_rules:
            if len(tips) >= 3:
                break
            if rule["check"](feat):
                tips.append(
                    ImprovementTip(
                        feature=rule["feature"],
                        suggestion=rule["suggestion"],
                        impact=rule["impact"],
                    )
                )

        # Second pass: fill remaining slots from unused rules (always-useful advice)
        for rule in self._tip_rules:
            if len(tips) >= 3:
                break
            if not rule["check"](feat):
                tips.append(
                    ImprovementTip(
                        feature=rule["feature"],
                        suggestion=f"Keep leveraging {rule['feature']} — it's already strong. "
                        f"Consider A/B testing variants to push it further.",
                        impact=rule["impact"],
                    )
                )

        return tips[:3]

    # ── Full analysis ──────────────────────────────────────────────────
    async def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        """
        Full analysis pipeline for a single ad creative.

        1. Extract features (Claude with rule-based fallback).
        2. Classify via the trained model.
        3. Generate exactly 3 improvement tips.
        """
        t0 = time.perf_counter()

        # Feature extraction
        features_list = await extract_batch_claude([request.text])
        features = features_list[0]

        # Classification
        prediction: ClassifierPrediction = self.classifier.predict(features)

        # Improvement tips
        tips = self.get_improvement_tips(features)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return AnalyzeResponse(
            features=features,
            prediction=prediction,
            improvement_tips=tips,
            processing_time_ms=round(elapsed_ms, 2),
        )
