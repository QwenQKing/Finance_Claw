from __future__ import annotations

from typing import Sequence, Dict, Any
import re

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

_SENTIMENT_LABEL_ALIASES = {
    "strong negative": "negative",
    "strongly negative": "negative",
    "moderately negative": "negative",
    "mildly negative": "negative",
    "very negative": "negative",
    "bearish": "negative",
    "negative": "negative",
    "neg": "negative",
    "消极": "negative",
    "负面": "negative",
    "利空": "negative",
    "strong positive": "positive",
    "strongly positive": "positive",
    "moderately positive": "positive",
    "mildly positive": "positive",
    "very positive": "positive",
    "bullish": "positive",
    "positive": "positive",
    "pos": "positive",
    "积极": "positive",
    "正面": "positive",
    "利好": "positive",
    "neutral": "neutral",
    "neut": "neutral",
    "中性": "neutral",
    "中立": "neutral",
    "mixed": "neutral",
    "mixed sentiment": "neutral",
    "neutral to mixed": "neutral",
    "true": "true",
    "yes": "true",
    "supported": "true",
    "正确": "true",
    "false": "false",
    "no": "false",
    "refuted": "false",
    "错误": "false",
}

_OPTION_PATTERN = re.compile(r"\b(?:final answer|answer|option|choice|correct answer)\s*(?:is|:)?\s*\(?([a-h])\)?\b", re.I)
_LEADING_OPTION_PATTERN = re.compile(r"^\s*\(?([a-h])\)?(?:[\s\.\):,-]|$)", re.I)
_TRAILING_OPTION_PATTERN = re.compile(r"\b([a-h])\b(?:[\s\.\):,-]*$)", re.I)

def _extract_bool_label(text: str) -> str | None:
    for raw, normalized in sorted(
        {k: v for k, v in _SENTIMENT_LABEL_ALIASES.items() if v in {"true", "false"}}.items(),
        key=lambda item: -len(item[0]),
    ):
        if re.search(rf"(?<![a-z]){re.escape(raw)}(?![a-z])", text):
            return normalized
    return None

def _extract_sentiment_label(text: str) -> str | None:
    hits: set[str] = set()
    for raw, normalized in sorted(
        {k: v for k, v in _SENTIMENT_LABEL_ALIASES.items() if v in {"positive", "negative", "neutral"}}.items(),
        key=lambda item: -len(item[0]),
    ):
        if re.search(rf"(?<![a-z]){re.escape(raw)}(?![a-z])", text):
            hits.add(normalized)

    if "positive" in hits and "negative" in hits:
        return "neutral"
    if "positive" in hits:
        return "positive"
    if "negative" in hits:
        return "negative"
    if "neutral" in hits:
        return "neutral"

    if any(token in text for token in ["optimistic", "bullish", "improved prospects", "cautious optimism"]):
        return "positive"
    if any(token in text for token in ["pessimistic", "bearish", "deteriorating outlook"]):
        return "negative"
    return None

def _normalize_label(value: Any) -> str:
    raw_text = str(value or "").strip()
    text = raw_text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;!?()[]{}\"'")
    if text in _SENTIMENT_LABEL_ALIASES:
        return _SENTIMENT_LABEL_ALIASES[text]

    if re.fullmatch(r"[a-h]", text):
        return text.upper()

    match = (
        _OPTION_PATTERN.search(raw_text)
        or _LEADING_OPTION_PATTERN.search(raw_text)
        or _TRAILING_OPTION_PATTERN.search(raw_text)
        or _OPTION_PATTERN.search(text)
        or _LEADING_OPTION_PATTERN.search(text)
        or _TRAILING_OPTION_PATTERN.search(text)
    )
    if match:
        return str(match.group(1)).upper()

    bool_label = _extract_bool_label(text)
    if bool_label is not None:
        return bool_label

    sentiment_label = _extract_sentiment_label(text)
    if sentiment_label is not None:
        return sentiment_label
    return "__other__" if text else ""

def normalize_classification_labels(values: Sequence[Any]) -> list[str]:
    return [_normalize_label(v) for v in values]

def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    if not y_true or not y_pred:
        return 0.0
    return float(accuracy_score(normalize_classification_labels(y_true), normalize_classification_labels(y_pred)))

def macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    if not y_true or not y_pred:
        return 0.0
    return float(f1_score(normalize_classification_labels(y_true), normalize_classification_labels(y_pred), average="macro"))

def auroc(y_true: Sequence[Any], y_score: Sequence[float]) -> float | None:
    y_true_arr = np.asarray(normalize_classification_labels(y_true))
    y_score_arr = np.asarray(list(y_score), dtype=float)

    if y_true_arr.size == 0:
        return None
    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    labels = np.unique(y_true_arr)
    if labels.size <= 1:
        return None

    if labels.size != 2:
        return None

    pos_label = labels[-1]
    y_bin = (y_true_arr == pos_label).astype(int)
    if len(np.unique(y_bin)) < 2:
        return None
    try:
        return float(roc_auc_score(y_bin, y_score_arr))
    except ValueError:
        return None

def compute_task_metrics(y_true: Sequence[Any], y_pred: Sequence[Any], y_score: Sequence[float]) -> Dict[str, float | None]:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "auroc": auroc(y_true, y_score),
    }
