from __future__ import annotations

from typing import Sequence, Dict, Any, Optional, List
import math
import re
from collections import Counter

import numpy as np

_BOOL_ALIASES = {
    "true": "true",
    "yes": "true",
    "supported": "true",
    "support": "true",
    "正确": "true",
    "是": "true",
    "真": "true",
    "false": "false",
    "no": "false",
    "refuted": "false",
    "refute": "false",
    "incorrect": "false",
    "错误": "false",
    "否": "false",
    "假": "false",
}

_SENTIMENT_ALIASES = {
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
}

_LIST_DELIMS = r"[，,、|/;；]"
_NUMBER_RE = re.compile(r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?%?")
_OPTION_PATTERN = re.compile(r"\b(?:final answer|answer|option|choice|correct answer)\s*(?:is|:)?\s*\(?([a-h])\)?\b", re.I)
_LEADING_OPTION_PATTERN = re.compile(r"^\s*\(?([a-h])\)?(?:[\s\.\):,-]|$)", re.I)
_TRAILING_OPTION_PATTERN = re.compile(r"\b([a-h])\b(?:[\s\.\):,-]*$)", re.I)

def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions or not references:
        return 0.0

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    matches = sum(1 for p, r in zip(predictions, references) if answers_equivalent(p, r))
    return float(matches / len(predictions))

def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[%％]", " % ", text)
    text = re.sub(r"[:：]", " : ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff:%\.\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _normalize_label_text(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip(" .,:;!?()[]{}\"'")
    return text

def _extract_bool_label(text: str) -> Optional[str]:
    for raw, normalized in sorted(_BOOL_ALIASES.items(), key=lambda item: -len(item[0])):
        if re.search(rf"(?<![a-z]){re.escape(raw)}(?![a-z])", text):
            return normalized
    return None

def _extract_sentiment_label(text: str) -> Optional[str]:
    hits: set[str] = set()
    for raw, normalized in sorted(_SENTIMENT_ALIASES.items(), key=lambda item: -len(item[0])):
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

def normalize_scalar_label(text: Any) -> Optional[str]:
    raw_text = str(text or "").strip()
    normalized = _normalize_label_text(raw_text)
    if not normalized:
        return None
    if normalized in _BOOL_ALIASES:
        return _BOOL_ALIASES[normalized]
    if normalized in _SENTIMENT_ALIASES:
        return _SENTIMENT_ALIASES[normalized]
    if re.fullmatch(r"[a-h]", normalized):
        return normalized.upper()
    match = (
        _OPTION_PATTERN.search(raw_text)
        or _LEADING_OPTION_PATTERN.search(raw_text)
        or _TRAILING_OPTION_PATTERN.search(raw_text)
        or _OPTION_PATTERN.search(normalized)
        or _LEADING_OPTION_PATTERN.search(normalized)
        or _TRAILING_OPTION_PATTERN.search(normalized)
    )
    if match:
        return str(match.group(1)).upper()
    bool_label = _extract_bool_label(normalized)
    if bool_label is not None:
        return bool_label
    sentiment_label = _extract_sentiment_label(normalized)
    if sentiment_label is not None:
        return sentiment_label
    return None

def extract_numeric_value(text: Any) -> Optional[tuple[float, bool]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    matches = list(_NUMBER_RE.finditer(raw.replace("−", "-")))
    if not matches:
        return None
    percent_matches = [match for match in matches if "%" in match.group(0)]
    match = percent_matches[-1] if percent_matches else matches[-1]
    token = match.group(0)
    has_percent = "%" in token or bool(re.search(r"\b(percent|percentage|pct)\b", raw.lower())) or "百分" in raw
    cleaned = token.replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
    try:
        return float(cleaned), has_percent
    except ValueError:
        return None

def numeric_values_equivalent(pred: Any, ref: Any, tolerance: float = 0.01) -> bool:
    pred_info = extract_numeric_value(pred)
    ref_info = extract_numeric_value(ref)
    if pred_info is None or ref_info is None:
        return False

    pred_val, pred_pct = pred_info
    ref_val, ref_pct = ref_info
    candidate_pairs = [(pred_val, ref_val)]
    if pred_pct or ref_pct:
        candidate_pairs.extend(
            [
                (pred_val * 100.0, ref_val),
                (pred_val, ref_val * 100.0),
                (pred_val / 100.0, ref_val),
                (pred_val, ref_val / 100.0),
            ]
        )

    for left, right in candidate_pairs:
        if math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
            return True
        if right != 0 and abs(left - right) / abs(right) <= tolerance:
            return True
    return False

def _normalize_list_items(text: Any) -> List[str]:
    parts = [part.strip() for part in re.split(_LIST_DELIMS, str(text or "")) if part.strip()]
    return [normalize_text(part) for part in parts if normalize_text(part)]

def list_answers_equivalent(pred: Any, ref: Any) -> bool:
    pred_items = _normalize_list_items(pred)
    ref_items = _normalize_list_items(ref)
    if len(pred_items) <= 1 or len(ref_items) <= 1:
        return False
    return Counter(pred_items) == Counter(ref_items)

def answers_equivalent(pred: Any, ref: Any) -> bool:
    pred_label = normalize_scalar_label(pred)
    ref_label = normalize_scalar_label(ref)
    if pred_label is not None and ref_label is not None:
        return pred_label == ref_label
    if numeric_values_equivalent(pred, ref):
        return True
    if list_answers_equivalent(pred, ref):
        return True
    return normalize_text(pred) == normalize_text(ref)

def _metric_tokens(text: str) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return re.findall(r"[a-z0-9]+(?:[:.%\-][a-z0-9]+)*|[\u4e00-\u9fff]", normalized)

def _f1_from_tokens(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

def token_f1_score(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions or not references:
        return 0.0

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    scores = []
    for pred, ref in zip(predictions, references):
        if answers_equivalent(pred, ref):
            scores.append(1.0)
            continue
        pred_tokens = _metric_tokens(pred)
        ref_tokens = _metric_tokens(ref)
        scores.append(_f1_from_tokens(pred_tokens, ref_tokens))

    return float(np.mean(scores))

def rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions or not references:
        return 0.0

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    def compute_lcs_length(pred_tokens: List[str], ref_tokens: List[str]) -> int:
        m, n = len(pred_tokens), len(ref_tokens)
        if m == 0 or n == 0:
            return 0

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i - 1] == ref_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    scores = []
    for pred, ref in zip(predictions, references):
        if answers_equivalent(pred, ref):
            scores.append(1.0)
            continue
        pred_tokens = _metric_tokens(pred)
        ref_tokens = _metric_tokens(ref)
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue

        lcs_len = compute_lcs_length(pred_tokens, ref_tokens)
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(ref_tokens)
        scores.append(0.0 if precision + recall == 0 else (2 * precision * recall / (precision + recall)))

    return float(np.mean(scores))

def numerical_consistency(predictions: Sequence[Any], references: Sequence[Any], tolerance: float = 0.01) -> float:
    if not predictions or not references:
        return 0.0

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    consistent = 0
    total_numeric = 0

    for pred, ref in zip(predictions, references):
        pred_info = extract_numeric_value(pred)
        ref_info = extract_numeric_value(ref)
        if pred_info is None or ref_info is None:
            continue
        total_numeric += 1
        if numeric_values_equivalent(pred, ref, tolerance=tolerance):
            consistent += 1

    return float(consistent / total_numeric) if total_numeric > 0 else 0.0

def citation_faithfulness(
    predictions: Sequence[str],
    references: Sequence[str],
    evidence_texts: Optional[Sequence[str]] = None
) -> Dict[str, float]:
    if not predictions or not references:
        return {"citation_faithfulness": 0.0}

    def extract_claims(text: str) -> List[str]:
        import re
        sentences = re.split(r'[.!?]', str(text))
        return [s.strip() for s in sentences if s.strip()]

    faithfulness_scores = []
    for pred, ref in zip(predictions, references):
        pred_claims = set(extract_claims(pred))
        ref_claims = set(extract_claims(ref))

        if not ref_claims:
            faithfulness_scores.append(1.0)
            continue

        supported = sum(1 for claim in ref_claims if any(claim in pc for pc in pred_claims))
        faithfulness_scores.append(supported / len(ref_claims))

    return {"citation_faithfulness": float(np.mean(faithfulness_scores))}

def directional_accuracy(predictions: Sequence[float], references: Sequence[float]) -> float:
    if not predictions or not references:
        return 0.0

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    if len(predictions) < 2:
        return 0.0

    pred_changes = np.diff(np.asarray(predictions, dtype=float))
    ref_changes = np.diff(np.asarray(references, dtype=float))
    pred_sign = np.sign(pred_changes)
    ref_sign = np.sign(ref_changes)

    return float(np.mean(pred_sign == ref_sign))

def hit_at_k(predictions: Sequence[List[Any]], references: Sequence[Any], k: int = 5) -> float:
    if not predictions or not references:
        return 0.0

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    hits = 0
    for pred_list, ref in zip(predictions, references):
        if not isinstance(pred_list, list):
            pred_list = list(pred_list) if hasattr(pred_list, '__iter__') else [pred_list]

        top_k = pred_list[:k]
        if ref in top_k:
            hits += 1

    return float(hits / len(predictions))

def mape(predictions: Sequence[float], references: Sequence[float]) -> float:
    if not predictions or not references:
        return float('inf')

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    errors = []
    for pred, ref in zip(predictions, references):
        try:
            pred_val = float(pred)
            ref_val = float(ref)

            if ref_val == 0:
                continue

            errors.append(abs((ref_val - pred_val) / ref_val))
        except (TypeError, ValueError):
            continue

    return float(np.mean(errors)) if errors else float('inf')

def kg_coverage_at_k(
    predictions: Sequence[Any],
    references: Sequence[Any],
    kg_evidence_ids: Optional[Sequence[List[str]]] = None,
    k: int = 5
) -> Dict[str, float]:
    if not kg_evidence_ids:
        return {"kg_evidence_recall_at_k": 0.0, "kg_evidence_used_ratio": 0.0}

    recalls = []
    used_ratios = []

    for i, _ in enumerate(zip(predictions, references)):
        evidence_list = kg_evidence_ids[i] if i < len(kg_evidence_ids) else []
        if not evidence_list:
            recalls.append(0.0)
            used_ratios.append(0.0)
            continue

        used_evidence = min(len(evidence_list), k)
        recalls.append(used_evidence / len(evidence_list))
        used_ratios.append(used_evidence / k if k > 0 else 0.0)

    return {
        "kg_evidence_recall_at_k": float(np.mean(recalls)),
        "kg_evidence_used_ratio": float(np.mean(used_ratios)),
    }

def path_faithfulness(
    predictions: Sequence[Any],
    kg_paths: Optional[Sequence[List[Dict[str, Any]]]] = None
) -> float:
    if not kg_paths or not predictions:
        return 0.0

    if len(predictions) != len(kg_paths):
        raise ValueError("predictions and kg_paths must have the same length")

    faithful = 0
    for pred, paths in zip(predictions, kg_paths):
        if not paths:
            faithful += 1
            continue

        supported = False
        pred_norm = normalize_text(pred)
        for path in paths:
            if not isinstance(path, dict):
                continue
            if path.get("supports_prediction") is True:
                supported = True
                break
            path_answer = normalize_text(path.get("answer", ""))
            if path_answer and path_answer == pred_norm:
                supported = True
                break
        if supported:
            faithful += 1

    return float(faithful / len(predictions))

def temporal_kg_validity(
    kg_temporal_edges: Optional[Sequence[List[Dict[str, Any]]]] = None,
    temporal_violations: Optional[Sequence[int]] = None
) -> Dict[str, float]:
    if not kg_temporal_edges and not temporal_violations:
        return {"temporal_kg_validity": 0.0, "temporal_violation_ratio": 0.0}

    if temporal_violations is not None:
        total_samples = len(temporal_violations)
        if total_samples == 0:
            return {"temporal_kg_validity": 0.0, "temporal_violation_ratio": 0.0}
        if kg_temporal_edges:
            total_edges = sum(len(edges) for edges in kg_temporal_edges)
        else:
            total_edges = total_samples
        total_violations = int(sum(temporal_violations))
        validity = 1.0 - (total_violations / total_edges) if total_edges > 0 else 0.0
        violation_ratio = total_violations / total_samples
        return {
            "temporal_kg_validity": float(max(0.0, min(1.0, validity))),
            "temporal_violation_ratio": float(max(0.0, min(1.0, violation_ratio))),
        }

    return {"temporal_kg_validity": 0.0, "temporal_violation_ratio": 0.0}

def conflict_edge_utilization(
    predictions: Sequence[Any],
    conflict_edges_used: Optional[Sequence[int]] = None,
    total_conflict_edges: Optional[Sequence[int]] = None
) -> float:
    if not conflict_edges_used or not total_conflict_edges:
        return 0.0

    if len(predictions) != len(conflict_edges_used) or len(conflict_edges_used) != len(total_conflict_edges):
        raise ValueError("predictions, conflict_edges_used, and total_conflict_edges must have the same length")

    ratios = []
    for used, total in zip(conflict_edges_used, total_conflict_edges):
        if total <= 0:
            ratios.append(0.0)
        else:
            ratios.append(min(max(float(used) / float(total), 0.0), 1.0))

    return float(np.mean(ratios))

def compute_qa_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    evidence_texts: Optional[Sequence[str]] = None
) -> Dict[str, float]:
    result = {
        "exact_match": exact_match(predictions, references),
        "token_f1": token_f1_score(predictions, references),
        "rouge_l": rouge_l(predictions, references),
        "numerical_consistency": numerical_consistency(predictions, references),
    }
    result.update(citation_faithfulness(predictions, references, evidence_texts))
    return result

def compute_forecasting_metrics(
    predictions: Sequence[float],
    references: Sequence[float]
) -> Dict[str, float]:
    return {
        "directional_accuracy": directional_accuracy(predictions, references),
        "mape": mape(predictions, references),
    }

def compute_hit_at_k_metric(
    predictions: Sequence[List[Any]],
    references: Sequence[Any],
    k: int = 5
) -> Dict[str, float]:
    return {"hit_at_k": hit_at_k(predictions, references, k)}

def compute_kg_metrics(
    predictions: Sequence[Any],
    kg_evidence_ids: Optional[Sequence[List[str]]] = None,
    kg_paths: Optional[Sequence[List[Dict[str, Any]]]] = None,
    kg_temporal_edges: Optional[Sequence[List[Dict[str, Any]]]] = None,
    temporal_violations: Optional[Sequence[int]] = None,
    conflict_edges_used: Optional[Sequence[int]] = None,
    total_conflict_edges: Optional[Sequence[int]] = None,
    k: int = 5
) -> Dict[str, float]:
    result: Dict[str, float] = {}

    result.update(kg_coverage_at_k(predictions, predictions, kg_evidence_ids, k))
    result["path_faithfulness"] = path_faithfulness(predictions, kg_paths)
    result.update(temporal_kg_validity(kg_temporal_edges, temporal_violations))
    result["conflict_edge_utilization"] = conflict_edge_utilization(
        predictions, conflict_edges_used, total_conflict_edges
    )

    return result
