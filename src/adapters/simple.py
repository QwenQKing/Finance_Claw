from __future__ import annotations

import base64
import hashlib
import io
import heapq
import json
import mimetypes
import os
import random
import re
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence

import requests
from urllib import error, request

from ..profiling import attach_stage_timings, merge_stage_timings
from ..retrieval import DenseHybridRetriever
from .base import BaselineAdapter
from .token_estimation import estimate_usage_from_prompt_completion

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DENSE_RETRIEVER_LOCAL = threading.local()
_PREBUILT_RAG_ENV_KEYS = {}
_CONTEXT_FIRST_RAG_DATASETS = {
    "ConvFinQA",
    "FinChart-Bench",
    "FinMME",
    "FinMTM",
    "TAT-QA",
}

def _ensure_think_answer_tags(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "<think></think>\n<answer></answer>\n<confidence></confidence>"
    think_matches = re.findall(r"<think>\s*(.*?)\s*</think>", raw, flags=re.I | re.S)
    answer_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.I | re.S)
    confidence_matches = re.findall(r"<confidence>\s*(.*?)\s*</confidence>", raw, flags=re.I | re.S)

    def _strip_inline_tags(value: str) -> str:
        cleaned = re.sub(r"</?[^>]+>", "", str(value or "")).strip()
        cleaned = re.sub(r"</?(?:think|answer)\b[^>]*>?", "", cleaned, flags=re.I).strip()
        return cleaned

    if think_matches or answer_matches:
        think_text = _strip_inline_tags(think_matches[-1] if think_matches else "")
        answer_text = _strip_inline_tags(answer_matches[-1] if answer_matches else "")
        confidence_text = _strip_inline_tags(confidence_matches[-1] if confidence_matches else "")
        if not think_text:
            think_text = "brief reasoning omitted by upstream adapter"
        return f"<think>{think_text}</think>\n<answer>{answer_text}</answer>\n<confidence>{confidence_text}</confidence>"

    return f"<think>brief reasoning omitted by upstream adapter</think>\n<answer>{raw}</answer>\n<confidence></confidence>"

def _load_dotenv() -> None:
    candidates: List[Path] = []

    candidates.append(_PROJECT_ROOT)

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    here = Path(__file__).resolve()
    candidates.extend([here.parent, *here.parents])

    seen: set[Path] = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        env_path = base / ".env"
        if not env_path.exists():
            continue

        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
        return

def _extract_primary_text(sample: Dict[str, Any]) -> str:
    modalities = sample.get("modalities", []) or []
    for m in modalities:
        if m.get("modality") != "text":
            continue
        data = m.get("data", {}) or {}
        parts: List[str] = []
        content = data.get("content", "")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        table = data.get("table")
        if isinstance(table, dict):
            table_rows = table.get("table")
            if isinstance(table_rows, list) and table_rows:
                lines: List[str] = []
                for row in table_rows:
                    if isinstance(row, list):
                        cells = [str(cell).strip() for cell in row]
                        lines.append(" | ".join(cells))
                if lines:
                    parts.append("[TABLE]\n" + "\n".join(lines))
        paragraphs = data.get("paragraphs")
        if isinstance(paragraphs, list) and paragraphs:
            para_lines = [str(item.get("text") or "").strip() for item in paragraphs if isinstance(item, dict) and str(item.get("text") or "").strip()]
            if para_lines:
                parts.append("[PARAGRAPHS]\n" + "\n".join(para_lines))
        merged = "\n\n".join(part for part in parts if part)
        if merged.strip():
            return merged.strip()
    return ""

def _prefer_sample_context_for_external_rag(sample: Dict[str, Any]) -> bool:
    dataset = str(sample.get("dataset") or sample.get("source_dataset") or "").strip()
    if dataset in _CONTEXT_FIRST_RAG_DATASETS:
        return True
    sample_id = str(sample.get("id") or sample.get("sample_id") or "").strip().lower()
    if sample_id.startswith("convfinqa_") or sample_id.startswith("tatqa_") or sample_id.startswith("finmtm_") or sample_id.startswith("finchart_") or sample_id.startswith("finmme_"):
        return True
    topic = str(sample.get("topic") or "").strip().lower()
    if topic in {"table_text_qa", "chart_qa", "multimodal_temporal_matching"}:
        return True
    for modality in sample.get("modalities", []) or []:
        data = modality.get("data", {}) or {}
        if isinstance(data.get("table"), dict) or isinstance(data.get("paragraphs"), list):
            return True
    return False

def _extract_question(sample: Dict[str, Any]) -> str:
    return str(sample.get("question", "")).strip()

def _extract_instruction(sample: Dict[str, Any]) -> str:
    return str(sample.get("instruction", "")).strip()

def _extract_option_labels(text: str) -> List[str]:
    def _normalize_letter_labels(labels: List[str]) -> List[str]:
        if len(labels) < 2:
            return []
        normalized = [label.strip().upper() for label in labels if label.strip()]
        if len(normalized) < 2 or any(len(label) != 1 or not label.isalpha() for label in normalized):
            return []
        if normalized[0] != "A":
            return []
        expected = [chr(ord("A") + idx) for idx in range(len(normalized))]
        return normalized if normalized == expected else []

    option_labels = re.findall(r"(?m)^\s*([A-Z])(?:[.:：]|\s)", text)
    if not option_labels:
        option_labels = re.findall(r"\b([A-Z])\.\s+(?=\S)", text)
    if option_labels:
        ordered: List[str] = []
        seen: set[str] = set()
        for label in option_labels:
            if label not in seen:
                seen.add(label)
                ordered.append(label)
        normalized = _normalize_letter_labels(ordered)
        if normalized:
            return normalized

    grouped_options: List[str] = []
    for pattern in [
        r"\{([^{}]+)\}",
        r"（([^（）]{1,300})）",
        r"\(([^()]{1,300})\)",
    ]:
        grouped_options.extend(match.group(1) for match in re.finditer(pattern, text))

    for raw in grouped_options:
        if not re.search(
            r"[/,、，]|negative|neutral|positive|bullish|bearish|true|false|yes|no|积极|中性|消极|正面|负面|利好|利空|是|否|可选|选项|分类",
            raw,
            flags=re.IGNORECASE,
        ):
            continue
        parts = [part.strip() for part in re.split(r"[/,、，]", raw) if part.strip()]
        if parts:
            if all(len(label) == 1 and label.isalpha() for label in parts):
                return _normalize_letter_labels(parts)
            return parts

    return []

_LABEL_ALIAS_GROUPS: Dict[str, Tuple[str, ...]] = {
    "positive": ("positive", "bullish", "pos", "积极", "正面", "利好", "看涨"),
    "negative": ("negative", "bearish", "neg", "消极", "负面", "利空", "看跌"),
    "neutral": ("neutral", "mixed", "mixed sentiment", "中性", "中立", "持平"),
    "true": ("true", "yes", "是", "正确", "对", "成立"),
    "false": ("false", "no", "否", "错误", "不对", "不成立"),
}
_DATASET_DEFAULT_LABELS: Dict[str, List[str]] = {
    "fpb": ["positive", "neutral", "negative"],
    "tfns": ["bullish", "neutral", "bearish"],
    "fin-fact": ["true", "neutral", "false"],
    "finfact": ["true", "neutral", "false"],
}
_NUMERIC_ANSWER_TYPES = {"arithmetic", "decimal", "float", "integer", "number", "numeric"}

def _normalize_label_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()

def _compact_label_match_text(text: str) -> str:
    normalized = _normalize_label_match_text(text)
    return re.sub(r"[\s,，、。.;:：/\\|()\[\]{}<>\"'`~!@#$%^&*+=?-]+", "", normalized)

def _canonical_label_alias(text: str) -> Optional[str]:
    normalized = _normalize_label_match_text(text)
    compact = _compact_label_match_text(text)
    for canonical, aliases in _LABEL_ALIAS_GROUPS.items():
        for alias in sorted(aliases, key=len, reverse=True):
            alias_norm = _normalize_label_match_text(alias)
            alias_compact = _compact_label_match_text(alias)
            if not alias_compact:
                continue
            if any("\u4e00" <= ch <= "\u9fff" for ch in alias):
                if alias in normalized or alias_compact == compact:
                    return canonical
                continue
            if re.search(rf"\b{re.escape(alias_norm)}\b", normalized):
                return canonical
            if alias_compact == compact:
                return canonical
    return None

def _text_matches_label(answer_text: str, label_text: str) -> bool:
    answer_norm = _normalize_label_match_text(answer_text)
    label_norm = _normalize_label_match_text(label_text)
    answer_compact = _compact_label_match_text(answer_text)
    label_compact = _compact_label_match_text(label_text)
    if not answer_compact or not label_compact:
        return False
    if answer_compact == label_compact:
        return True
    if len(label_compact) >= 2 and label_compact in answer_compact:
        return True
    if len(answer_compact) >= 2 and answer_compact in label_compact:
        return True
    answer_alias = _canonical_label_alias(answer_norm)
    label_alias = _canonical_label_alias(label_norm)
    return answer_alias is not None and answer_alias == label_alias

def _extract_option_text_map(text: str) -> Dict[str, str]:
    option_map: Dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^([A-Z])(?:[.:：]|\s)\s*(.+?)\s*$", line)
        if not match:
            continue
        option_map[match.group(1).upper()] = match.group(2).strip()
    return option_map

def _extract_tag_block(text: str, tag: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", raw, flags=re.I | re.S)
    return match.group(1).strip() if match else ""

def _extract_labels_from_question(question: str) -> List[str]:
    return _extract_option_labels(question)

def _default_labels_from_sample(sample: Dict[str, Any]) -> List[str]:
    dataset = str(sample.get("dataset") or sample.get("source_dataset") or "").strip().lower()
    if dataset in _DATASET_DEFAULT_LABELS:
        return list(_DATASET_DEFAULT_LABELS[dataset])

    question = _extract_question(sample)
    q_lower = question.lower()
    topic = str(sample.get("topic") or "").strip().lower()
    answer_type = str(sample.get("answer_type") or "").strip().lower()

    if topic in {"sentiment", "sentiment_analysis"} or "情感" in question or "sentiment" in q_lower:
        if any(token in q_lower for token in ["bullish", "bearish"]) or any(token in question for token in ["看涨", "看跌"]):
            return ["bullish", "neutral", "bearish"]
        return ["positive", "neutral", "negative"]

    if (
        dataset in {"fin-fact", "finfact"}
        or "true or false" in q_lower
        or "exactly: true or false" in q_lower
        or "claim supported" in q_lower
    ):
        return ["true", "neutral", "false"]

    if answer_type in {"yes_no", "bool"}:
        return ["true", "false"]

    return []

def _extract_labels_from_sample(sample: Dict[str, Any]) -> List[str]:
    dataset = str(sample.get("dataset") or sample.get("source_dataset") or "").strip().lower()
    if dataset in _DATASET_DEFAULT_LABELS:
        return list(_DATASET_DEFAULT_LABELS[dataset])

    answer_type = str(sample.get("answer_type") or "").strip().lower()
    if answer_type in _NUMERIC_ANSWER_TYPES:
        return []
    if answer_type in {"yes_no", "bool"}:
        return ["true", "false"]

    question = _extract_question(sample)
    labels = _extract_option_labels(question)
    if labels:
        return labels
    labels = _extract_option_labels(_extract_primary_text(sample))
    if labels:
        return labels
    return _default_labels_from_sample(sample)

def _sample_allows_multi_label(sample: Dict[str, Any]) -> bool:
    answer_type = str(sample.get("answer_type", "")).strip().lower()
    if answer_type in {"labels", "multi_label", "multiple_choice"}:
        return True
    answer = sample.get("answer")
    return isinstance(answer, list)

def _extract_image_modalities(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []
    for modality in sample.get("modalities", []) or []:
        if modality.get("modality") != "image":
            continue
        data = modality.get("data", {}) or {}
        if data:
            images.append(data)
    return images

def _image_data_url(image_data: Dict[str, Any]) -> str:
    def _int_env(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)) or default)
        except Exception:
            return default

    def _maybe_reencode_image(raw: bytes, fmt: str) -> Tuple[bytes, str]:
        max_side = _int_env("RUNNER_IMAGE_MAX_SIDE", 0)
        force_format = str(os.getenv("RUNNER_IMAGE_FORCE_FORMAT", "") or "").strip().lower()
        if not max_side and not force_format:
            return raw, fmt

        try:
            from PIL import Image

            with Image.open(io.BytesIO(raw)) as image:
                target_format = force_format or ("jpeg" if fmt.lower() in {"jpg", "jpeg"} else fmt.lower())
                if max_side and max(image.size) > max_side:
                    image = image.copy()
                    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                elif force_format:
                    image = image.copy()

                if target_format in {"jpg", "jpeg"} and image.mode not in {"RGB", "L"}:
                    image = image.convert("RGB")
                elif image.mode == "P":
                    image = image.convert("RGBA")

                out = io.BytesIO()
                save_format = "JPEG" if target_format in {"jpg", "jpeg"} else target_format.upper()
                save_kwargs: Dict[str, Any] = {}
                if save_format == "JPEG":
                    save_kwargs.update(
                        {
                            "quality": max(1, min(95, _int_env("RUNNER_IMAGE_JPEG_QUALITY", 85))),
                            "optimize": True,
                        }
                    )
                    target_format = "jpg"
                image.save(out, format=save_format, **save_kwargs)
                return out.getvalue(), target_format
        except Exception:
            return raw, fmt

    inline_base64 = str(image_data.get("base64") or "").strip()
    fmt = str(image_data.get("format") or "").strip().lower() or "png"
    if inline_base64:
        try:
            raw = base64.b64decode(inline_base64)
            raw, fmt = _maybe_reencode_image(raw, fmt)
            inline_base64 = base64.b64encode(raw).decode("ascii")
        except Exception:
            pass
        mime = mimetypes.types_map.get(f".{fmt}", f"image/{fmt}")
        return f"data:{mime};base64,{inline_base64}"

    path_str = str(image_data.get("path") or "").strip()
    if not path_str:
        return ""
    cache_key = "|".join(
        [
            path_str,
            str(os.getenv("RUNNER_IMAGE_MAX_SIDE", "") or ""),
            str(os.getenv("RUNNER_IMAGE_FORCE_FORMAT", "") or ""),
            str(os.getenv("RUNNER_IMAGE_JPEG_QUALITY", "") or ""),
        ]
    )
    cached = _IMAGE_DATA_URL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    raw = Path(path_str).read_bytes()
    raw, fmt = _maybe_reencode_image(raw, fmt)
    mime = mimetypes.types_map.get(f".{fmt}", f"image/{fmt}")
    encoded = base64.b64encode(raw).decode("ascii")
    data_url = f"data:{mime};base64,{encoded}"
    _IMAGE_DATA_URL_CACHE[cache_key] = data_url
    return data_url

def _model_accepts_openai_image_url() -> bool:

    override = str(os.getenv("FORCE_OPENAI_IMAGE_URL", "") or "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True

    model = (
        os.getenv("MODEL_NAME")
        or os.getenv("LLM_MODEL_NAME")
        or os.getenv("KG_LLM_MODEL_NAME")
        or ""
    ).strip().lower()
    if model.startswith("deepseek-"):
        return False
    return True

def _multimodal_message_content(
    prompt_text: str,
    sample: Dict[str, Any],
    budget: Dict[str, Any],
) -> str | List[Dict[str, Any]]:
    image_modalities = _extract_image_modalities(sample)
    supports_vision = str(os.getenv("MODEL_SUPPORTS_VISION", "") or "").strip().lower() in {"1", "true", "yes", "on"}
    if not image_modalities or not supports_vision or not _model_accepts_openai_image_url():
        return prompt_text

    max_images = int(budget.get("max_input_images", os.getenv("RUNNER_MAX_INPUT_IMAGES", "6")) or 6)
    detail = str(budget.get("image_detail", os.getenv("RUNNER_IMAGE_DETAIL", "high")) or "high").strip()
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    included = 0
    for image_data in image_modalities:
        if included >= max_images:
            break
        data_url = _image_data_url(image_data)
        if not data_url:
            continue
        content.append({"type": "image_url", "image_url": {"url": data_url, "detail": detail}})
        included += 1
    return content if included else prompt_text

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _prediction_in_label_space(prediction: str, labels: Sequence[str], *, allow_multi: bool = False) -> bool:
    value = str(prediction or "").strip()
    if not labels:
        return bool(value)
    if value in labels:
        return True
    if allow_multi and all(len(label) == 1 and label.isalpha() for label in labels):
        allowed = {label.upper() for label in labels}
        return bool(value) and all(char in allowed for char in value.upper())
    return False

def _requires_strict_label_space(sample: Dict[str, Any]) -> bool:
    dataset = str(sample.get("dataset") or sample.get("source_dataset") or "").strip().lower()
    answer_type = str(sample.get("answer_type") or "").strip().lower()
    return dataset in _DATASET_DEFAULT_LABELS or answer_type in {
        "bool",
        "label",
        "labels",
        "multi_label",
        "multiple_choice",
        "yes_no",
    }

def _response_is_complete(raw_text: str) -> bool:
    text = str(raw_text or "").strip()
    if not text:
        return False
    if not text.lower().startswith("<think"):
        return True
    return bool(re.search(r"<answer>\s*.+?\s*</answer>", text, flags=re.I | re.S))

def _extract_tag_block(text: str, tag: str) -> str:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, str(text or ""), flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""

def _strip_inline_tags(text: str) -> str:
    cleaned = re.sub(r"</?(?:think|answer)\b[^>]*>", " ", str(text or ""), flags=re.IGNORECASE)
    return _normalize_text(cleaned)

def _extract_answer_text(text: str) -> str:
    answer = _extract_tag_block(text, "answer")
    if answer:
        nested_answer = _extract_tag_block(answer, "answer")
        if nested_answer:
            answer = nested_answer
        answer = _strip_inline_tags(answer)
        if answer:
            return answer
    return _strip_inline_tags(str(text or "").strip())

def _sample_numeric_like(sample: Dict[str, Any]) -> bool:
    q = _extract_question(sample)
    q_lower = q.lower()
    answer_type = str(sample.get("answer_type", "")).strip().lower()
    dataset = str(sample.get("dataset") or sample.get("source_dataset") or "").strip()
    return (
        dataset in {"ConvFinQA", "FinMTM", "TAT-QA"}
        or
        answer_type in _NUMERIC_ANSWER_TYPES
        or any(token in q_lower for token in ["percentage", "percent", "ratio", "change", "difference", "total", "net income", "how much"])
        or any(token in q for token in ["百分比", "比例", "变化", "差值", "总计", "多少"])
    )

_NUMERIC_SPAN_RE = re.compile(
    r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|percent|percentage|bp|bps|x|倍|万元|亿元|元|美元|million|billion|trillion|mn|bn))?",
    flags=re.IGNORECASE,
)

def _compress_numeric_answer(sample: Dict[str, Any], answer_text: str) -> str:
    cleaned = _normalize_text(answer_text)
    if not cleaned or not _sample_numeric_like(sample):
        return cleaned
    if len(cleaned) <= 48 and len(cleaned.split()) <= 6:
        return cleaned

    tail = cleaned
    markers = [
        "final answer",
        "answer is",
        "therefore",
        "thus",
        "so the answer is",
        "calculated as",
        "equals",
        "结果是",
        "答案是",
        "因此",
        "所以",
    ]
    lowered = cleaned.lower()
    for marker in markers:
        idx = lowered.rfind(marker)
        if idx >= 0:
            tail = cleaned[idx + len(marker) :].strip(" :：,.;")
            break

    candidates = _NUMERIC_SPAN_RE.findall(tail) or _NUMERIC_SPAN_RE.findall(cleaned)
    if not candidates:
        return cleaned
    return _normalize_text(candidates[-1])

def _ensure_nonempty_answer_tagged(text: str) -> str:
    raw = _ensure_think_answer_tags(str(text or "").strip())
    if "<think></think>" in raw:
        raw = raw.replace(
            "<think></think>",
            "<think>brief reasoning omitted by upstream adapter</think>",
            1,
        )
    answer = _extract_tag_block(raw, "answer")
    if not answer:
        raise ValueError("Answer block is empty after extraction")
    return raw

def _split_chunks(text: str, max_len: int = 260) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []

    sents = re.split(r"(?<=[.!?。！？])\s+", text)
    chunks: List[str] = []
    cur = ""
    for s in sents:
        if not s:
            continue
        cand = f"{cur} {s}".strip() if cur else s
        if len(cand) <= max_len:
            cur = cand
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)

    if not chunks:
        return [text[:max_len]]
    return chunks

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", text.lower())

def _truncate_chunk_text(text: str, max_chars: int = 420) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."

def _budget_int(
    budget: Dict[str, Any],
    key: str,
    env_name: str,
    default: int,
) -> int:
    if key in budget and budget.get(key) is not None:
        try:
            return int(budget.get(key))
        except Exception:
            return default
    raw = os.getenv(env_name, "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            return default
    return default

def _keyword_overlap_score(query: str, text: str) -> float:
    q = set(_tokenize(query))
    t = set(_tokenize(text))
    if not q or not t:
        return 0.0
    inter = len(q.intersection(t))
    return inter / max(1, len(q))

def _retrieve_top_k(question: str, text: str, top_k: int) -> List[str]:
    chunks = _split_chunks(text)
    scored = [(c, _keyword_overlap_score(question, c)) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[: max(1, top_k)]]

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows

def _stream_top_k_chunk_rows(
    chunks_path: Path,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    heap: List[Tuple[float, int, Dict[str, Any]]] = []
    seq = 0
    with chunks_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            score = _keyword_overlap_score(query, text)
            entry = (score, seq, row)
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
            seq += 1
    if not heap:
        return []
    heap.sort(reverse=True)
    return [row for _, _, row in heap]

def _dense_retriever_for_asset(asset_dir: Path) -> Optional[DenseHybridRetriever]:
    if os.getenv("DISABLE_DENSE_RETRIEVAL", "").strip() == "1":
        return None
    dense_manifest = asset_dir / "dense_index" / "manifest.json"
    if not dense_manifest.exists():
        return None
    cache = getattr(_DENSE_RETRIEVER_LOCAL, "cache", None)
    if cache is None:
        cache = {}
        _DENSE_RETRIEVER_LOCAL.cache = cache
    cache_key = str(asset_dir.resolve())
    retriever = cache.get(cache_key)
    if retriever is None:
        retriever = DenseHybridRetriever(
            asset_dir,
            candidate_limit=max(256, int(os.getenv("DENSE_CANDIDATE_LIMIT", "2048") or "2048")),
        )
        cache[cache_key] = retriever
    return retriever

def _retriever_stage_timings_ms(retriever: Optional[DenseHybridRetriever]) -> Dict[str, float]:
    stats = dict(getattr(retriever, "last_stats", {}) or {})
    return {
        "rag_candidate_recall_ms": float(stats.get("candidate_recall_seconds", 0.0) or 0.0) * 1000.0,
        "rag_rerank_ms": float(stats.get("rerank_seconds", 0.0) or 0.0) * 1000.0,
        "rag_row_load_ms": float(stats.get("row_load_seconds", 0.0) or 0.0) * 1000.0,
        "rag_query_embed_ms": float(stats.get("query_embed_seconds", 0.0) or 0.0) * 1000.0,
    }

def _retrieve_prebuilt_rows(asset_dir: Path, chunks_path: Path, query: str, top_k: int) -> tuple[List[Dict[str, Any]], str, Dict[str, float]]:
    try:
        retriever = _dense_retriever_for_asset(asset_dir)
    except Exception:
        retriever = None
    if retriever is not None:
        try:
            rows = retriever.retrieve(query, top_k=top_k)
            if rows:
                return rows, "prebuilt_asset_dense_hybrid_topk", _retriever_stage_timings_ms(retriever)
        except Exception:
            pass
    return _stream_top_k_chunk_rows(chunks_path, query, top_k=top_k), "prebuilt_asset_keyword_topk", {}

def _coerce_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (_REPO_ROOT / path)

def _resolve_prebuilt_rag_asset_dir(sample: Dict[str, Any], method_name: str) -> Optional[Path]:
    explicit_sample_dir = str(sample.get("_finclaw_explicit_asset_dir") or "").strip()
    if explicit_sample_dir:
        path = _coerce_path(explicit_sample_dir)
        if path.exists():
            return path
    env_keys = _PREBUILT_RAG_ENV_KEYS.get(method_name)
    if not env_keys:
        return None
    dir_key, root_key = env_keys
    explicit = os.getenv(dir_key, "").strip()
    if explicit:
        path = _coerce_path(explicit)
        if path.exists():
            return path
    root = os.getenv(root_key, "").strip()
    if not root:
        return None
    base = _coerce_path(root)
    if not base.exists():
        return None
    asset_dataset = (
        str(sample.get("_finclaw_asset_dataset") or "").strip()
        or os.getenv("ASSET_DATASET", "").strip()
        or os.getenv("DATASET_NAME", "").strip()
        or str(sample.get("source_dataset") or sample.get("dataset") or "").strip()
    )
    if not asset_dataset:
        return None
    candidate = base / asset_dataset
    return candidate if candidate.exists() else None

def _prebuilt_rag_payload(
    sample: Dict[str, Any],
    budget: Dict[str, Any],
    *,
    method_name: str,
) -> Optional[Tuple[str, List[str], List[str], str, Dict[str, float]]]:
    if _prefer_sample_context_for_external_rag(sample):
        return None
    started = time.perf_counter()
    asset_dir = _resolve_prebuilt_rag_asset_dir(sample, method_name)
    if asset_dir is None:
        return None
    chunks_path = asset_dir / "corpus" / "chunks.jsonl"
    if not chunks_path.exists():
        return None
    question = "\n".join(part for part in [_extract_instruction(sample), _extract_question(sample)] if part).strip()
    top_k = max(1, int(budget.get("max_retrieval_top_k", 6)))
    selected, retrieval_strategy, dense_stage_timings = _retrieve_prebuilt_rows(asset_dir, chunks_path, question, top_k=top_k)
    if not selected:
        return None
    retrieval_ms = (time.perf_counter() - started) * 1000.0
    retrieved_chunks = [str(row.get("text") or "").strip() for row in selected]
    evidence_ids = [str(row.get("chunk_id") or f"chunk_{idx+1}") for idx, row in enumerate(selected)]
    labels = _extract_labels_from_sample(sample)
    prompt_started = time.perf_counter()
    message = _benchmark_retrieval_blob(sample, labels, retrieved_chunks)
    prompt_build_ms = (time.perf_counter() - prompt_started) * 1000.0
    return message, labels, evidence_ids, retrieval_strategy, {
        "rag_retrieval_ms": retrieval_ms,
        "rag_prompt_build_ms": prompt_build_ms,
        **dense_stage_timings,
    }

def _as_bullets(items: Iterable[str], prefix: str = "- ") -> str:
    return "\n".join(f"{prefix}{x}" for x in items)

def _append_confidence_instruction(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if os.getenv("REQUIRE_CONFIDENCE_TAG", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return messages
    patched = [dict(message) for message in messages]
    instruction = (
        "\n\nFor the final benchmark response, output exactly these tags:\n"
        "<think>brief reasoning</think>\n"
        "<answer>final answer only</answer>\n"
        "<confidence>integer from 0 to 100 estimating the probability that the answer is correct</confidence>\n"
        "The <confidence> tag is mandatory and must contain digits only, for example <confidence>80</confidence>. "
        "Never leave <confidence> empty. Do not put confidence text inside <answer>."
    )
    for idx in range(len(patched) - 1, -1, -1):
        content = patched[idx].get("content")
        if not isinstance(content, str) or "<answer>" not in content.lower():
            continue
        patched[idx]["content"] = content.rstrip() + instruction
        break
    return patched

def _benchmark_user_blob(sample: Dict[str, Any], labels: List[str], max_context_chars: int = 12000) -> str:
    instruction = _extract_instruction(sample)
    q = _extract_question(sample)
    ctx = _extract_primary_text(sample)
    dataset = str(sample.get("dataset") or sample.get("source_dataset") or "").strip().lower()
    topic = str(sample.get("topic", "")).strip().lower()
    answer_type = str(sample.get("answer_type", "")).strip().lower()
    parts: List[str] = []
    if instruction:
        parts.append(f"Instruction:\n{instruction}")
    parts.append(f"Context:\n{ctx[:max_context_chars]}")
    parts.append(f"Question:\n{q}")
    rules: List[str] = [
        "Use only the provided Context.",
        "Do not ask follow-up questions.",
        "Do not mention tools, missing files, missing reports, system prompts, sessions, or limitations.",
        "Output exactly two tags and nothing else: <think>...</think><answer>...</answer>.",
    ]
    if labels:
        if _sample_allows_multi_label(sample):
            rules.append(
                "Inside <answer>, return one or more option labels from the allowed set, "
                "concatenated in option order with no separators (example: AC)."
            )
            rules.append(f"Allowed labels: {', '.join(labels)}")
        else:
            rules.append(f"Inside <answer>, return exactly one label from: {', '.join(labels)}")
        if dataset in {"fin-fact", "finfact"}:
            rules.append(
                "For Fin-Fact, ignore any wording that restricts answers to only true/false; "
                "the valid labels are true, neutral, and false. Use neutral when the context is insufficient, mixed, or inconclusive."
            )
    else:
        q_lower = q.lower()
        numeric_like = (
            answer_type == "arithmetic"
            or any(token in q_lower for token in ["percentage", "percent", "ratio", "change", "difference", "total", "net income", "how much"])
            or any(token in q for token in ["百分比", "比例", "变化", "差值", "总计", "多少"])
        )
        if (
            topic in {"company", "product", "sector"}
            or "公司名称" in q
            or "找出" in q
            or "识别" in q
            or "提取" in q
            or "which company" in q_lower
            or "company name" in q_lower
        ):
            rules.append("Inside <answer>, return only the extracted answer span or comma-separated items from the Context.")
        elif topic == "sentiment" or "情感" in q or "sentiment" in q_lower:
            rules.append("Inside <answer>, return only the sentiment answer.")
        elif topic == "risk" or "风险" in q:
            rules.append("Inside <answer>, return only the concise risk summary, with no bullets.")
        elif topic == "suggestion" or "评级" in q or "建议" in q or "rating" in q_lower:
            rules.append("Inside <answer>, return only the concise investment suggestion or rating summary, with no bullets.")
        elif numeric_like:
            rules.append("Inside <answer>, return only the final numeric answer or short value span needed by the question, with no explanation.")
        else:
            rules.append("Inside <answer>, return only the final answer, as short as possible.")

        if answer_type == "text":
            rules.append("Do not wrap the answer inside <answer> with markdown, quotes, or leading phrases like '答案是' or '根据内容'.")
    parts.append("Rules:\n" + _as_bullets(rules))
    parts.append("Use a brief reasoning trace in <think> and place the final answer in <answer>.")
    return "\n\n".join(parts)

def _benchmark_retrieval_blob(
    sample: Dict[str, Any],
    labels: List[str],
    retrieved_chunks: Sequence[str],
) -> str:
    instruction = _extract_instruction(sample)
    q = _extract_question(sample)
    dataset = str(sample.get("dataset") or sample.get("source_dataset") or "").strip().lower()
    topic = str(sample.get("topic", "")).strip().lower()
    answer_type = str(sample.get("answer_type", "")).strip().lower()

    parts: List[str] = []
    if instruction:
        parts.append(f"Instruction:\n{instruction}")
    parts.append(f"Question:\n{q}")
    parts.append(
        "Retrieved Passages:\n"
        + (
            _as_bullets([f"[chunk_{i}] {chunk}" for i, chunk in enumerate(retrieved_chunks, 1)])
            if retrieved_chunks
            else "- (no retrieved passages)"
        )
    )

    rules: List[str] = [
        "Use only the provided Retrieved Passages.",
        "Do not ask follow-up questions.",
        "Do not mention tools, retrieval, missing files, missing reports, system prompts, sessions, or limitations.",
        "Output exactly two tags and nothing else: <think>...</think><answer>...</answer>.",
    ]
    if labels:
        if _sample_allows_multi_label(sample):
            rules.append(
                "Inside <answer>, return one or more option labels from the allowed set, "
                "concatenated in option order with no separators (example: AC)."
            )
            rules.append(f"Allowed labels: {', '.join(labels)}")
        else:
            rules.append(f"Inside <answer>, return exactly one label from: {', '.join(labels)}")
        if dataset in {"fin-fact", "finfact"}:
            rules.append(
                "For Fin-Fact, ignore any wording that restricts answers to only true/false; "
                "the valid labels are true, neutral, and false. Use neutral when the context is insufficient, mixed, or inconclusive."
            )
    else:
        q_lower = q.lower()
        numeric_like = (
            answer_type == "arithmetic"
            or any(token in q_lower for token in ["percentage", "percent", "ratio", "change", "difference", "total", "net income", "how much"])
            or any(token in q for token in ["百分比", "比例", "变化", "差值", "总计", "多少"])
        )
        if (
            topic in {"company", "product", "sector"}
            or "公司名称" in q
            or "找出" in q
            or "识别" in q
            or "提取" in q
            or "which company" in q_lower
            or "company name" in q_lower
        ):
            rules.append("Inside <answer>, return only the extracted answer span or comma-separated items from the Retrieved Passages.")
        elif topic == "sentiment" or "情感" in q or "sentiment" in q_lower:
            rules.append("Inside <answer>, return only the sentiment answer.")
        elif topic == "risk" or "风险" in q:
            rules.append("Inside <answer>, return only the concise risk summary, with no bullets.")
        elif topic == "suggestion" or "评级" in q or "建议" in q or "rating" in q_lower:
            rules.append("Inside <answer>, return only the concise investment suggestion or rating summary, with no bullets.")
        elif numeric_like:
            rules.append("Inside <answer>, return only the final numeric answer or short value span needed by the question, with no explanation.")
        else:
            rules.append("Inside <answer>, return only the final answer, as short as possible.")

        if answer_type == "text":
            rules.append("Do not wrap the answer inside <answer> with markdown, quotes, or leading phrases like '答案是' or '根据内容'.")

    parts.append("Rules:\n" + _as_bullets(rules))
    parts.append("Use a brief reasoning trace in <think> and place the final answer in <answer>.")
    return "\n\n".join(parts)

def _require_official_external_baselines() -> bool:
    return os.getenv("REQUIRE_OFFICIAL_EXTERNAL_BASELINES", "").strip().lower() in {
        "1", "true", "yes", "on",
    }

def _external_agent_retrieval_payload(
    sample: Dict[str, Any],
    budget: Dict[str, Any],
    *,
    default_top_k: int = 6,
    max_top_k: int = 8,
) -> Tuple[str, List[str], List[str], Dict[str, float]]:
    retrieval_started = time.perf_counter()
    labels = _extract_labels_from_sample(sample)
    if _prefer_sample_context_for_external_rag(sample):
        question = _extract_question(sample)
        context = _extract_primary_text(sample)
        top_k = min(int(budget.get("max_retrieval_top_k", 4)), 4)
        chunks = [_truncate_chunk_text(chunk) for chunk in (_retrieve_top_k(question, context, top_k=top_k) if context else [])]
        prompt_started = time.perf_counter()
        message = _benchmark_retrieval_blob(sample, labels, chunks)
        prompt_build_ms = (time.perf_counter() - prompt_started) * 1000.0
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
        evidence_ids = [f"chunk_{i}" for i in range(1, len(chunks) + 1)]
        return message, labels, evidence_ids, {
            "rag_retrieval_ms": retrieval_ms,
            "rag_prompt_build_ms": prompt_build_ms,
            "rag_context_fallback": 1.0,
        }
    question = _extract_question(sample)
    context = _extract_primary_text(sample)
    top_k = min(int(budget.get("max_retrieval_top_k", default_top_k)), max_top_k)
    chunks = _retrieve_top_k(question, context, top_k=top_k) if context else []
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
    prompt_started = time.perf_counter()
    message = _benchmark_retrieval_blob(sample, labels, chunks)
    prompt_build_ms = (time.perf_counter() - prompt_started) * 1000.0
    evidence_ids = [f"chunk_{i}" for i in range(1, len(chunks) + 1)]
    return message, labels, evidence_ids, {
        "rag_retrieval_ms": retrieval_ms,
        "rag_prompt_build_ms": prompt_build_ms,
    }

class OpenAICompatibleLLM:
    def __init__(self) -> None:
        _load_dotenv()

        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.base_url = os.getenv("LLM_BASE_URL", "").rstrip("/").strip()
        self.model = os.getenv("LLM_MODEL_NAME", "").strip()
        self._session_local = threading.local()

        missing = [
            name
            for name, val in [
                ("LLM_API_KEY", self.api_key),
                ("LLM_BASE_URL", self.base_url),
                ("LLM_MODEL_NAME", self.model),
            ]
            if not val
        ]
        if missing:
            raise RuntimeError(f"Missing required .env variables: {', '.join(missing)}")

    @staticmethod
    def validate_env() -> List[str]:
        _load_dotenv()
        checks = {
            "LLM_API_KEY": os.getenv("LLM_API_KEY", "").strip(),
            "LLM_BASE_URL": os.getenv("LLM_BASE_URL", "").rstrip("/").strip(),
            "LLM_MODEL_NAME": os.getenv("LLM_MODEL_NAME", "").strip(),
        }
        return [k for k, v in checks.items() if not v]

    def _thread_session(self) -> requests.Session:
        session = getattr(self._session_local, "session", None)
        if session is None:
            session = requests.Session()
            session.trust_env = False
            session.headers.update(
                {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "Connection": "close",
                }
            )
            self._session_local.session = session
        return session

    def _reset_thread_session(self) -> None:
        session = getattr(self._session_local, "session", None)
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        self._session_local.session = None

    def chat(self, messages: List[Dict[str, Any]], budget: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        messages = _append_confidence_instruction(messages)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(budget.get("temperature", 0.0)),
            "top_p": float(budget.get("top_p", 1.0)),
            "max_tokens": int(budget.get("max_output_tokens", 8192)),
        }
        started = time.perf_counter()
        timeout_seconds = int(budget.get("timeout_seconds", 120) or 120)
        connect_timeout = float(
            budget.get(
                "llm_connect_timeout_seconds",
                os.getenv("LLM_CONNECT_TIMEOUT_SECONDS", "10"),
            )
            or 10
        )
        read_timeout = float(
            budget.get(
                "llm_read_timeout_seconds",
                os.getenv("LLM_READ_TIMEOUT_SECONDS", str(timeout_seconds)),
            )
            or timeout_seconds
        )
        max_attempts = max(
            1,
            int(budget.get("llm_retry_attempts", os.getenv("LLM_RETRY_ATTEMPTS", "2")) or 2),
        )
        retry_backoff_seconds = float(
            budget.get("llm_retry_backoff_seconds", os.getenv("LLM_RETRY_BACKOFF_SECONDS", "1.5")) or 1.5
        )
        body = ""
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            session = self._thread_session()
            resp: requests.Response | None = None
            try:
                resp = session.post(
                    url,
                    json=payload,
                    timeout=(max(1.0, connect_timeout), max(1.0, read_timeout)),
                )
                if resp.status_code in {408, 429, 500, 502, 503, 504}:
                    if attempt < max_attempts:
                        last_exc = RuntimeError(f"LLM HTTPError: {resp.status_code} {resp.text[:1000]}")
                        self._reset_thread_session()
                        time.sleep(retry_backoff_seconds * attempt)
                        continue
                    raise RuntimeError(f"LLM HTTPError: {resp.status_code} {resp.text[:4000]}")
                resp.raise_for_status()
                body = resp.text
                last_exc = None
                break
            except requests.HTTPError as exc:
                self._reset_thread_session()
                status_code = exc.response.status_code if exc.response is not None else "unknown"
                detail = exc.response.text[:4000] if exc.response is not None else str(exc)
                raise RuntimeError(f"LLM HTTPError: {status_code} {detail}") from exc
            except (requests.Timeout, requests.ConnectionError) as exc:
                self._reset_thread_session()
                last_exc = exc
                if attempt < max_attempts:
                    time.sleep(retry_backoff_seconds * attempt)
                    continue
                raise RuntimeError(f"LLM request failed: {exc}") from exc
            except Exception as exc:
                self._reset_thread_session()
                raise RuntimeError(f"LLM request failed: {exc}") from exc
            finally:
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass

        if last_exc is not None:
            raise RuntimeError(f"LLM request failed after retries: {last_exc}") from last_exc

        data = json.loads(body)
        choices = data.get("choices", []) or []
        if not choices:
            raise RuntimeError("LLM response has no choices")

        content = str((choices[0].get("message", {}) or {}).get("content", "")).strip()

        usage = data.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
        latency_ms = (time.perf_counter() - started) * 1000.0
        confidence_only_request = any(
            "Return only one tag: <confidence>N</confidence>" in str(message.get("content") or "")
            for message in messages
        )
        if (
            os.getenv("REQUIRE_CONFIDENCE_TAG", "").strip().lower() in {"1", "true", "yes", "on"}
            and not confidence_only_request
            and not re.search(r"<confidence>\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*</confidence>", content, flags=re.I | re.S)
        ):
            retry_budget = dict(budget)
            retry_budget["max_output_tokens"] = min(32, int(retry_budget.get("max_output_tokens", 32) or 32))
            retry_out = self.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "Return only one tag: <confidence>N</confidence>, where N is an integer from 0 to 100.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Estimate the probability that the previous benchmark answer is correct. "
                            "Return only <confidence>N</confidence> with digits only."
                        ),
                    },
                ],
                budget=retry_budget,
            )
            content = content + "\n" + str(retry_out.get("content") or "").strip()
            prompt_tokens += int(retry_out.get("prompt_tokens", 0) or 0)
            completion_tokens += int(retry_out.get("completion_tokens", 0) or 0)
            total_tokens += int(retry_out.get("total_tokens", 0) or 0)
            usage = {
                "primary_usage": usage,
                "confidence_retry_usage": retry_out.get("raw_usage") or {},
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            latency_ms += float(retry_out.get("latency_ms", 0.0) or 0.0)

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "raw_usage": usage,
            "latency_ms": latency_ms,
        }

class _BaseRealAdapter(BaselineAdapter):
    system_prompt = "You are a precise financial reasoning assistant."

    def __init__(self, name: str) -> None:
        self.name = name
        self._llm: Optional[OpenAICompatibleLLM] = None

    @property
    def llm(self) -> OpenAICompatibleLLM:
        if self._llm is None:
            self._llm = OpenAICompatibleLLM()
        return self._llm

    def _enforce_label_space(
        self,
        raw_pred: str,
        labels: List[str],
        *,
        sample: Optional[Dict[str, Any]] = None,
        allow_multi: bool = False,
    ) -> str:
        answer_text = _normalize_text(_extract_answer_text(raw_pred))
        if not labels:
            return answer_text

        question = _extract_question(sample or {})
        primary_text = _extract_primary_text(sample or {})
        option_map = _extract_option_text_map(question)
        if not option_map and primary_text:
            option_map = _extract_option_text_map(primary_text)

        if allow_multi:
            multi = self._extract_multi_label_prediction(raw_pred, labels)
            if multi:
                return multi

        single = self._extract_single_letter_label(raw_pred, labels)
        if single:
            return single

        if option_map and all(len(label) == 1 and label.isalpha() for label in labels):
            matched = [label for label in labels if _text_matches_label(answer_text, option_map.get(label.upper(), ""))]
            if matched:
                return "".join(matched) if allow_multi else matched[0]

        direct_matches: List[str] = []
        if not all(len(label) == 1 and label.isalpha() for label in labels):
            direct_matches = [label for label in labels if _text_matches_label(answer_text, label)]
        if direct_matches:
            return "".join(direct_matches) if allow_multi else direct_matches[0]

        answer_alias = _canonical_label_alias(answer_text)
        if answer_alias is not None:
            alias_matches = [label for label in labels if _canonical_label_alias(label) == answer_alias]
            if alias_matches:
                return "".join(alias_matches) if allow_multi else alias_matches[0]

        return answer_text

    def _extract_single_letter_label(self, raw_pred: str, labels: List[str]) -> str:
        allowed = {label.upper(): label for label in labels}
        upper_text = _extract_answer_text(raw_pred).upper()
        match = re.search(r"\b([A-Z])\b", upper_text)
        if match and match.group(1) in allowed:
            return allowed[match.group(1)]
        match = re.match(r"\s*([A-Z])(?:[.:：\s]|$)", upper_text)
        if match and match.group(1) in allowed:
            return allowed[match.group(1)]
        return ""

    def _extract_multi_label_prediction(self, raw_pred: str, labels: List[str]) -> str:
        allowed = {label.upper(): label for label in labels}
        upper_text = _extract_answer_text(raw_pred).upper()
        labelish_text = re.sub(r"\b(?:AND|OR|AND/OR|OPTION|OPTIONS)\b", " ", upper_text)
        labelish_text = re.sub(r"[和及与]", " ", labelish_text)
        compact_letters = re.sub(r"[^A-Z]", "", labelish_text)
        if not compact_letters or any(char not in allowed for char in compact_letters):
            return ""
        direct_concat = re.findall(r"\b[" + "".join(re.escape(k) for k in allowed) + r"]+\b", upper_text)
        found: List[str] = []
        for token in direct_concat:
            for char in token:
                if char in allowed and char not in found:
                    found.append(char)

        if not found:
            tokenized = re.findall(r"[A-Z]", upper_text)
            for token in tokenized:
                if token in allowed and token not in found:
                    found.append(token)

        if not found:
            array_like = re.findall(r"[\"']([A-Z])[\"']", upper_text)
            for token in array_like:
                if token in allowed and token not in found:
                    found.append(token)

        if not found:
            return ""
        ordered = [label for label in labels if label.upper() in found]
        return "".join(ordered)

    def _ask_prediction(
        self,
        sample: Dict[str, Any],
        budget: Dict[str, Any],
        method_brief: str,
        context_sections: List[Tuple[str, str]],
        extra_rules: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        question = _extract_question(sample)
        instruction = _extract_instruction(sample)
        labels = _extract_labels_from_sample(sample)
        allow_multi = _sample_allows_multi_label(sample)
        strict_label_space = _requires_strict_label_space(sample)

        sections: List[str] = [f"Method: {method_brief}"]
        if instruction:
            sections.append(f"Instruction:\n{instruction}")
        sections.append(f"Question:\n{question}")

        for title, content in context_sections:
            sections.append(f"{title}:\n{content}")

        rules = [
            "Output exactly two tags and nothing else: <think>...</think><answer>...</answer>.",
            "Do not invent unsupported facts.",
        ]
        if labels:
            if allow_multi:
                rules.append(
                    "Inside <answer>, output one or more option labels from the allowed set, concatenated in option order with no separators (example: AC)."
                )
                rules.append(f"Allowed labels: {', '.join(labels)}")
            else:
                rules.append(f"Inside <answer>, output exactly one label from: {', '.join(labels)}")
            if strict_label_space:
                rules.append("You must choose from the allowed labels even if evidence is incomplete; never return N/A.")
        if extra_rules:
            rules.extend(extra_rules)

        sections.append("Rules:\n" + _as_bullets(rules))
        sections.append("Keep <think> brief and put the benchmark-facing answer inside <answer>.")

        user_prompt = "\n\n".join(sections)
        user_content = _multimodal_message_content(user_prompt, sample, budget)
        retry_messages: Optional[List[Dict[str, Any]]] = None
        answer_only_messages: Optional[List[Dict[str, Any]]] = None
        model_name = (
            os.getenv("MODEL_NAME")
            or os.getenv("LLM_MODEL_NAME")
            or os.getenv("KG_LLM_MODEL_NAME")
            or ""
        ).strip().lower()
        has_visual_label_task = bool(labels and _extract_image_modalities(sample))
        needs_concise_fallback = bool(
            has_visual_label_task
            or _requires_strict_label_space(sample)
            or str(sample.get("answer_type") or "").strip().lower() in _NUMERIC_ANSWER_TYPES
        )
        if needs_concise_fallback:
            concise_action = (
                "Read the attached image and answer the benchmark question."
                if _extract_image_modalities(sample)
                else "Answer the benchmark question from the provided information."
            )
            concise_rules = [
                concise_action,
                "Output exactly: <think>brief reason</think><answer>final answer</answer>.",
                "Keep <think> very short.",
            ]
            if labels:
                if allow_multi:
                    concise_rules.append(
                        "Inside <answer>, output only the correct option label(s), concatenated in option order with no separators."
                    )
                    concise_rules.append(f"Allowed labels: {', '.join(labels)}")
                else:
                    concise_rules.append(f"Inside <answer>, output exactly one label from: {', '.join(labels)}")
                if strict_label_space:
                    concise_rules.append("You must choose from the allowed labels; never return N/A.")
            concise_prompt = "\n\n".join(
                [
                    f"Question:\n{question}",
                    "Rules:\n" + _as_bullets(concise_rules),
                ]
            )
            retry_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": _multimodal_message_content(concise_prompt, sample, budget)},
            ]
            answer_only_rules = [
                concise_action,
                "Return only the final benchmark answer. Do not explain.",
            ]
            if labels:
                if allow_multi:
                    answer_only_rules.append(
                        "Return only the correct option label(s), concatenated in option order with no separators."
                    )
                    answer_only_rules.append(f"Allowed labels: {', '.join(labels)}")
                else:
                    answer_only_rules.append(f"Return exactly one label from: {', '.join(labels)}")
                if strict_label_space:
                    answer_only_rules.append("You must choose from the allowed labels; never return N/A.")
            answer_only_prompt = "\n\n".join(
                [
                    f"Question:\n{question}",
                    "Rules:\n" + _as_bullets(answer_only_rules),
                ]
            )
            answer_only_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": _multimodal_message_content(answer_only_prompt, sample, budget)},
            ]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        out: Dict[str, Any] = {}
        prediction = ""
        semantic_retry_count = 0
        response_complete = False
        answer_type = str(sample.get("answer_type") or "").strip().lower()
        normal_attempts = 1 if model_name.startswith("deepseek-") and answer_type in _NUMERIC_ANSWER_TYPES else 4
        for attempt in range(normal_attempts):
            active_messages = retry_messages if retry_messages is not None and attempt >= 2 else messages
            out = self.llm.chat(messages=active_messages, budget=budget)
            prediction = self._enforce_label_space(out["content"], labels, sample=sample, allow_multi=allow_multi)
            response_complete = _response_is_complete(out.get("content", ""))
            label_valid = not strict_label_space or _prediction_in_label_space(prediction, labels, allow_multi=allow_multi)
            if response_complete and label_valid:
                break
            semantic_retry_count = attempt + 1
        if (not response_complete or (strict_label_space and not label_valid)) and answer_only_messages is not None:
            fallback_budget = {**budget, "max_output_tokens": max(8192, int(budget.get("max_output_tokens", 8192) or 8192))}
            fallback_out = self.llm.chat(messages=answer_only_messages, budget=fallback_budget)
            fallback_raw = str(fallback_out.get("content", "")).strip()
            fallback_prediction = self._enforce_label_space(
                fallback_raw,
                labels,
                sample=sample,
                allow_multi=allow_multi,
            )
            fallback_label_valid = not strict_label_space or _prediction_in_label_space(
                fallback_prediction,
                labels,
                allow_multi=allow_multi,
            )
            if fallback_raw and fallback_label_valid:
                prediction = fallback_prediction
                wrapped = (
                    "<think>Concise fallback was used because the strict tagged response was incomplete.</think>"
                    f"<answer>{prediction}</answer>"
                )
                out = fallback_out
                out["content"] = wrapped
                out["prediction"] = prediction
                out["concise_fallback"] = True
                out["concise_fallback_raw"] = fallback_raw
                if has_visual_label_task:
                    out["concise_visual_fallback"] = True
                    out["concise_visual_fallback_raw"] = fallback_raw
                if model_name.startswith("gemini-") and has_visual_label_task:
                    out["gemini_concise_visual_fallback"] = True
                    out["gemini_concise_visual_fallback_raw"] = fallback_raw
                response_complete = True
                label_valid = True
        if not response_complete:
            raise RuntimeError("model response is empty or incomplete after semantic retries")
        if strict_label_space and not _prediction_in_label_space(prediction, labels, allow_multi=allow_multi):
            raise RuntimeError(f"model prediction is outside label space {labels}: {prediction!r}")
        out["semantic_retry_count"] = semantic_retry_count
        out["prediction"] = prediction
        return out

    def _build_output(
        self,
        llm_out: Dict[str, Any],
        *,
        steps_used: int,
        top_k_used: int,
        evidence_ids: Optional[List[str]] = None,
        confidence: float = 0.7,
        temporal_valid: bool = True,
        violation: bool = False,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        stage_timings = {}
        if isinstance(llm_out.get("latency_ms"), (int, float)):
            stage_timings["llm_generate_ms"] = float(llm_out["latency_ms"])
        raw_model_output = _ensure_nonempty_answer_tagged(
            str(llm_out.get("content", "") or "").strip(),
        )
        x = {
            "prediction": llm_out["prediction"],
            "confidence": confidence,
            "evidence_ids": evidence_ids or [],
            "temporal_valid": temporal_valid,
            "violation": violation,
            "steps_used": steps_used,
            "top_k_used": top_k_used,
            "prompt_tokens": llm_out["prompt_tokens"],
            "completion_tokens": llm_out["completion_tokens"],
            "total_tokens": llm_out["total_tokens"],
            "extras": {
                "adapter_mode": self.name,
                "real_llm_enabled": True,
                "execution_mode": "openai_compatible_http",
                "raw_model_output": raw_model_output,
                "raw_upstream_output": dict(llm_out),
                "raw_usage": llm_out.get("raw_usage", {}),
            },
        }
        if extras:
            x["extras"].update(extras)
        x["extras"] = attach_stage_timings(x["extras"], stage_timings)
        return x

def build_default_registry(instance_tag: str | None = None) -> Dict[str, BaselineAdapter]:
    from .unified_method import UnifiedPaperMethodAdapter

    return {
        "financeclaw": UnifiedPaperMethodAdapter("financeclaw", instance_tag=instance_tag),
    }
def _methods_need_openai_compatible_env(methods: Sequence[str]) -> bool:
    return True

def validate_runtime_requirements(
    methods: Sequence[str], cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    missing_env: List[str] = []
    if _methods_need_openai_compatible_env(methods):
        missing_env = OpenAICompatibleLLM.validate_env()
    issues: List[str] = []
    if missing_env:
        issues.append(f"Missing env vars: {', '.join(missing_env)}")
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "missing_env": missing_env,
        "kg_required": False,
        "kg_dir": "",
        "kg_manifest_exists": True,
    }
