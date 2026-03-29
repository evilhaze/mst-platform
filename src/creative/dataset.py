"""
Synthetic dataset generator for gambling / betting / casino ad creatives.

Generates 520 labelled ads with rule-based feature extraction and
realistic CTR / CR simulation.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .schemas import (
    CreativeFeatures,
    CreativeLabel,
    Emotion,
    LengthCategory,
    Vertical,
)

# ── Constants ──────────────────────────────────────────────────────────────
SEED = 42
DEFAULT_N = 520

GEOS = ["RU", "UA", "KZ", "BY", "PL", "DE", "TR"]

VERTICAL_WEIGHTS = {
    Vertical.gambling: 0.50,
    Vertical.betting: 0.35,
    Vertical.casino: 0.15,
}

GEO_MULTIPLIERS = {
    "RU": 1.00,
    "KZ": 0.95,
    "UA": 0.90,
    "BY": 0.88,
    "PL": 0.80,
    "DE": 0.75,
    "TR": 0.85,
}

BASE_CTR = 0.032

# ── Ad-text templates (6 types) ───────────────────────────────────────────
TEMPLATES: dict[str, list[str]] = {
    "urgency_number": [
        "Только сегодня! Бонус {num}% на первый депозит — успей забрать!",
        "Осталось {num} мест — зарегистрируйся прямо сейчас и получи фрибет!",
        "Последний шанс! {num} фриспинов ждут тебя — предложение истекает через час!",
        "Торопись! Бонус до {num}$ — акция заканчивается сегодня!",
        "Успей за {num} минут! Эксклюзивный кэшбэк только для новых игроков!",
    ],
    "social_proof_number": [
        "Уже {num}+ игроков выиграли на этой неделе — присоединяйся!",
        "{num} человек забрали бонус за последний час! Ты следующий?",
        "Нас уже {num}K — стань частью сообщества победителей!",
        "Более {num} выплат за сегодня — убедись сам!",
        "{num}+ довольных игроков не могут ошибаться. Начни выигрывать!",
    ],
    "greed_strong_cta": [
        "Забери свой бонус {num}% — жми и начинай выигрывать прямо сейчас!",
        "Удвой депозит до {num}$! Жми на кнопку — деньги ждут!",
        "{num} фриспинов бесплатно — активируй и забирай выигрыш!",
        "Получи до {num}$ без риска! Регистрируйся и выводи!",
        "Бонус {num}% + фрибет — не упусти шанс сорвать куш!",
    ],
    "excitement_cta": [
        "Почувствуй адреналин! Крути слоты и выигрывай до {num}$!",
        "Испытай удачу — ставь на спорт и получи заряд эмоций!",
        "Драйв начинается здесь! Делай ставку и побеждай!",
        "Невероятные эмоции ждут — играй и выигрывай каждый день!",
        "Азарт на максимум! Начни игру и сорви джекпот до {num}$!",
    ],
    "neutral_weak": [
        "Онлайн-казино с лицензией. Широкий выбор игр.",
        "Ставки на спорт — удобно и надёжно.",
        "Игровые автоматы от ведущих провайдеров. Попробуй бесплатно.",
        "Букмекерская контора с быстрыми выплатами.",
        "Большой выбор слотов и настольных игр для любого вкуса.",
    ],
    "fear_urgency": [
        "Не упусти! Бонус {num}% сгорает через {hours} часа!",
        "Ты теряешь деньги каждый день без кэшбэка {num}%!",
        "Пока ты думаешь — другие забирают бонусы до {num}$!",
        "Упустишь — пожалеешь! Только {num} бонусов осталось!",
        "Без этого бонуса ты играешь в минус. Активируй {num}% сейчас!",
    ],
}

# ── Keyword lists for rule-based extraction ────────────────────────────────
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


# ── Rule-based feature extraction ─────────────────────────────────────────
def _has_keywords(text_lower: str, keywords: list[str]) -> bool:
    return any(kw in text_lower for kw in keywords)


def _extract_rule_based_features(text: str) -> CreativeFeatures:
    """Extract all 7 creative features from ad text using regex + keyword lists."""
    low = text.lower()

    # 1. has_number
    has_number = bool(re.search(r"\d+", text))

    # 2. has_urgency
    has_urgency = _has_keywords(low, _URGENCY_KW)

    # 3. has_social_proof
    has_social_proof = _has_keywords(low, _SOCIAL_PROOF_KW)

    # 4. emotion — pick the strongest signal
    fear_score = sum(1 for kw in _FEAR_KW if kw in low)
    greed_score = sum(1 for kw in _GREED_KW if kw in low)
    excitement_score = sum(1 for kw in _EXCITEMENT_KW if kw in low)

    scores = {
        Emotion.fear: fear_score,
        Emotion.greed: greed_score,
        Emotion.excitement: excitement_score,
    }
    best_emotion = max(scores, key=scores.get)  # type: ignore[arg-type]
    emotion = best_emotion if scores[best_emotion] > 0 else Emotion.neutral

    # 5. cta_strength (1-5)
    cta_hits = sum(1 for kw in _STRONG_CTA_KW if kw in low)
    has_excl = "!" in text
    has_imperative = bool(
        re.search(r"(жми|забери|начни|играй|делай|крути|ставь|испытай)", low)
    )
    raw_cta = cta_hits + int(has_excl) + int(has_imperative)
    cta_strength = max(1, min(5, raw_cta))

    # 6. length_category
    word_count = len(text.split())
    if word_count <= 6:
        length_category = LengthCategory.short
    elif word_count <= 12:
        length_category = LengthCategory.medium
    else:
        length_category = LengthCategory.long

    # 7. key_benefit
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


# ── CTR / CR simulation ───────────────────────────────────────────────────
def _compute_ctr(
    feat: CreativeFeatures,
    geo: str,
    vertical: Vertical,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Compute realistic (CTR, CR) pair from features, geo and vertical.

    Returns
    -------
    (ctr, cr) — both clipped to sensible ranges.
    """
    multiplier = 1.0

    # has_urgency: 1.30–1.50
    if feat.has_urgency:
        multiplier *= rng.uniform(1.30, 1.50)

    # has_number: 1.20–1.35
    if feat.has_number:
        multiplier *= rng.uniform(1.20, 1.35)

    # has_social_proof: 1.15–1.25
    if feat.has_social_proof:
        multiplier *= rng.uniform(1.15, 1.25)

    # emotion
    emotion_mult = {
        Emotion.greed: 1.35,
        Emotion.excitement: 1.25,
        Emotion.fear: 1.10,
        Emotion.neutral: 0.85,
    }
    multiplier *= emotion_mult[feat.emotion]

    # cta_strength: +8% per level above 1
    multiplier *= 1.0 + 0.08 * (feat.cta_strength - 1)

    # length_category
    length_mult = {
        LengthCategory.medium: 1.10,
        LengthCategory.short: 0.90,
        LengthCategory.long: 0.95,
    }
    multiplier *= length_mult[feat.length_category]

    # geo
    multiplier *= GEO_MULTIPLIERS.get(geo, 0.85)

    # log-normal noise (std=0.15)
    noise = rng.lognormal(mean=0.0, sigma=0.15)
    ctr = BASE_CTR * multiplier * noise
    ctr = float(np.clip(ctr, 0.002, 0.35))

    # CR = CTR * uniform(0.08, 0.15)
    cr = ctr * rng.uniform(0.08, 0.15)
    cr = float(np.clip(cr, 0.0001, ctr))

    return round(ctr, 6), round(cr, 6)


# ── Dataset generation ─────────────────────────────────────────────────────
def generate_dataset(n: int = DEFAULT_N) -> pd.DataFrame:
    """
    Generate a synthetic ad-creative dataset.

    Parameters
    ----------
    n : int
        Number of rows (default 520).

    Returns
    -------
    pd.DataFrame with columns:
        ad_id, text, geo, vertical, ctr, cr, label,
        has_number, has_urgency, has_social_proof, emotion,
        cta_strength, length_category, key_benefit
    """
    rng = np.random.default_rng(SEED)

    # Pre-compute vertical distribution
    vert_values = [v.value for v in VERTICAL_WEIGHTS]
    vert_probs = np.array(list(VERTICAL_WEIGHTS.values()))
    verticals = rng.choice(vert_values, size=n, p=vert_probs)

    # Pre-compute template type weights (uniform across 6 types)
    tpl_names = list(TEMPLATES.keys())

    rows: list[dict] = []
    for i in range(n):
        geo = str(rng.choice(GEOS))
        vertical = Vertical(str(verticals[i]))

        # Pick template type, then a specific variant
        tpl_type = str(rng.choice(tpl_names))
        tpl_variants = TEMPLATES[tpl_type]
        tpl = str(rng.choice(tpl_variants))

        # Fill placeholders
        text = tpl.format(
            num=rng.integers(3, 500),
            hours=rng.integers(1, 24),
        )

        feat = _extract_rule_based_features(text)
        ctr, cr = _compute_ctr(feat, geo, vertical, rng)

        rows.append(
            {
                "ad_id": i + 1,
                "text": text,
                "geo": geo,
                "vertical": vertical.value,
                "ctr": ctr,
                "cr": cr,
                "has_number": feat.has_number,
                "has_urgency": feat.has_urgency,
                "has_social_proof": feat.has_social_proof,
                "emotion": feat.emotion.value,
                "cta_strength": feat.cta_strength,
                "length_category": feat.length_category.value,
                "key_benefit": feat.key_benefit,
            }
        )

    df = pd.DataFrame(rows)

    # Label: top-40% CTR → "good", rest → "bad"
    threshold = df["ctr"].quantile(0.60)
    df["label"] = df["ctr"].apply(
        lambda x: CreativeLabel.good.value if x >= threshold else CreativeLabel.bad.value
    )

    return df


def save_dataset(path: str | Path) -> pd.DataFrame:
    """Generate the dataset, save to *path* (CSV), and return the DataFrame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(path, index=False)
    return df
