from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

from bridge_common import (
    OpenAICompatibleLLM,
    as_bullets,
    ensure_think_answer_tags,
    extract_labels,
)
from retrieval.dense_hybrid import DenseHybridRetriever
from adapters.simple import build_default_registry

ROOT = Path(__file__).resolve().parents[1]

OUR_METHOD = "financeclaw"
_RUNTIME_ADAPTER_LOCAL = threading.local()
_DENSE_RETRIEVER_LOCAL = threading.local()
_ASSET_CACHE_LOCK = threading.Lock()
_ASSET_FILE_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_CHUNK_ROWS_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_KEYWORD_INDEX_CACHE: Dict[str, Dict[str, List[int]]] = {}
_SHARED_FILE_SIGNATURE_CACHE: Dict[str, tuple[float, int]] = {}
_SHARED_ROWS_BY_SIGNATURE: Dict[tuple[float, int], List[Dict[str, Any]]] = {}
_ASSET_EXISTS_CACHE: Dict[str, bool] = {}

def _repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (ROOT / path)

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _is_retrieval_disabled(retrieval_mode: str) -> bool:
    return str(retrieval_mode or "").strip().lower() in {"disabled", "off", "none", "no_rag", "no-rag"}

def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _append_jsonl_row(handle, row: Dict[str, Any], *, durable: bool = False) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    if durable:
        os.fsync(handle.fileno())

def _group_key(row: Dict[str, Any]) -> tuple[str, str]:
    dataset = str(row.get("dataset") or "").strip()
    method = str(row.get("method") or "").strip()
    return dataset, method

def _prepare_row_for_persistence(row: Dict[str, Any]) -> Dict[str, Any]:
    prepared = dict(row)
    extras = dict(prepared.get("extras") or {})
    raw_model_output = str(extras.get("raw_model_output") or "").strip()
    if raw_model_output:
        extras["raw_model_output"] = raw_model_output
        extras.setdefault("raw_model_output_source", "upstream")
    if extras:
        prepared["extras"] = extras
    return prepared

def _init_group_state() -> Dict[str, Any]:
    return {
        "count": 0,
        "completed": 0,
        "missing_assets": 0,
        "failed": 0,
        "unsupported_method": 0,
        "embedding_requests": 0,
        "chat_requests": 0,
        "retrieval_seconds": 0.0,
        "generation_seconds": 0.0,
        "sample_ids": [],
        "handle": None,
    }

def _sample_memory_mb() -> float:
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss) / (1024 * 1024)
    except Exception:
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if os.name == "nt":
                return float(rss) / (1024 * 1024)
            return float(rss) / 1024
        except Exception:
            return 0.0

def _capture_run_metrics(
    *,
    method: str,
    dataset: str,
    count: int,
    completed: int,
    failed: int,
    unsupported_method: int,
    retrieval_seconds: float,
    generation_seconds: float,
    max_inflight_eval: int,
    max_inflight_prefetch: int,
    memory_before_mb: float,
    memory_after_mb: float,
) -> Dict[str, Any]:
    total_eval_seconds = float(retrieval_seconds) + float(generation_seconds)
    return {
        "method": method,
        "dataset": dataset,
        "count": int(count),
        "completed": int(completed),
        "failed": int(failed),
        "unsupported_method": int(unsupported_method),
        "total_eval_seconds": round(total_eval_seconds, 6),
        "retrieval_seconds": round(float(retrieval_seconds), 6),
        "generation_seconds": round(float(generation_seconds), 6),
        "retrieval_share": round(float(retrieval_seconds) / max(1e-9, total_eval_seconds), 6),
        "memory_before_mb": round(float(memory_before_mb), 3),
        "memory_after_mb": round(float(memory_after_mb), 3),
        "memory_delta_mb": round(float(memory_after_mb) - float(memory_before_mb), 3),
        "max_inflight_eval": int(max_inflight_eval),
        "max_inflight_prefetch": int(max_inflight_prefetch),
    }

def _write_group_outputs(output_root: Path, grouped_state: Dict[tuple[str, str], Dict[str, Any]]) -> None:
    for (dataset, method), state in grouped_state.items():
        group_dir = output_root / dataset / method
        handle = state.get("handle")
        if handle is not None:
            handle.close()
        summary = {
            "dataset": dataset,
            "method": method,
            "count": int(state.get("count", 0) or 0),
            "completed": int(state.get("completed", 0) or 0),
            "missing_assets": int(state.get("missing_assets", 0) or 0),
            "failed": int(state.get("failed", 0) or 0),
            "unsupported_method": int(state.get("unsupported_method", 0) or 0),
            "embedding_requests": int(state.get("embedding_requests", 0) or 0),
            "chat_requests": int(state.get("chat_requests", 0) or 0),
            "retrieval_seconds": round(float(state.get("retrieval_seconds", 0.0) or 0.0), 6),
            "generation_seconds": round(float(state.get("generation_seconds", 0.0) or 0.0), 6),
            "avg_retrieval_seconds_per_completed": round(float(state.get("retrieval_seconds", 0.0) or 0.0) / max(1, int(state.get("completed", 0) or 0)), 6),
            "avg_generation_seconds_per_completed": round(float(state.get("generation_seconds", 0.0) or 0.0) / max(1, int(state.get("completed", 0) or 0)), 6),
            "sample_ids": list(state.get("sample_ids") or []),
            "generated_at": _utc_now(),
        }
        _write_json(group_dir / "summary.json", summary)

def _sample_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": _stable_sample_id(row),
        "question": _extract_question(row),
        "gold_answer": str(row.get("answer") or "").strip(),
        "answer_type": str(row.get("answer_type") or "").strip(),
        "topic": str(row.get("topic") or "").strip(),
        "domain": str(row.get("domain") or "").strip(),
    }

def _extract_question(row: Dict[str, Any]) -> str:
    return str(row.get("question") or "").strip()

def _extract_primary_text(row: Dict[str, Any]) -> str:
    retrieval_text = str(row.get("retrieval_text") or "").strip()
    if retrieval_text:
        return retrieval_text
    texts: List[str] = []
    for modality in row.get("modalities") or []:
        if str(modality.get("modality", "")).lower() != "text":
            data = modality.get("data") or {}
            if isinstance(data, dict):
                for key in ("image_summary", "summary", "image_caption", "caption", "image_ocr", "ocr"):
                    value = str(data.get(key) or "").strip()
                    if value:
                        texts.append(value)
            continue
        data = modality.get("data") or {}
        content = str(data.get("content") or "").strip()
        if content:
            texts.append(content)
    if texts:
        return "\n".join(texts)
    return "\n".join(
        x
        for x in [
            str(row.get("text") or "").strip(),
            str(row.get("context") or "").strip(),
            str(row.get("question") or "").strip(),
            str(row.get("instruction") or "").strip(),
        ]
        if x
    )

def _compact_text(text: str, limit: int = 900) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + " ..."

def _build_task_context(sample: Dict[str, Any]) -> str:
    parts = [
        str(sample.get("instruction") or "").strip(),
        _extract_primary_text(sample),
    ]
    return "\n".join(part for part in parts if part).strip()

def _build_retrieval_query(sample: Dict[str, Any]) -> str:
    question = _extract_question(sample)
    context = _compact_text(_build_task_context(sample), limit=480)
    return "\n".join(part for part in [question, context] if part).strip()

def _normalize_prediction(sample: Dict[str, Any], raw_prediction: str, labels: List[str]) -> str:
    del sample
    del labels
    text = str(raw_prediction or "").strip()
    tag = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    if tag:
        return tag.group(1)
    return text

class OutputFormatValidationError(RuntimeError):
    def __init__(self, message: str, *, raw_model_output: str = "") -> None:
        super().__init__(message)
        self.raw_model_output = str(raw_model_output or "")

def _require_nonempty_tagged_raw_output(raw_text: str, *, method: str) -> str:
    text = ensure_think_answer_tags(str(raw_text or "").strip())
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.I | re.S)
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    think_text = str(think_match.group(1) or "").strip() if think_match else ""
    answer_text = str(answer_match.group(1) or "").strip() if answer_match else ""

    def _strip_inline_tags(value: str) -> str:
        cleaned = re.sub(r"</?[^>]+>", "", str(value or "")).strip()
        cleaned = re.sub(r"</?(?:think|answer)\b[^>]*>?", "", cleaned, flags=re.I).strip()
        return cleaned

    def _has_tag_fragment(value: str) -> bool:
        return bool(re.search(r"</?(?:think|answer)\b", str(value or ""), flags=re.I))

    if not think_text or _has_tag_fragment(think_text):
        think_text = _strip_inline_tags(think_text) or "brief reasoning omitted by upstream adapter"
    if not think_match:
        text = f"<think>{think_text}</think>\n<answer>{answer_text}</answer>" if answer_text else f"<think>{think_text}</think>\n<answer></answer>"
    else:
        text = re.sub(
            r"<think>\s*.*?\s*</think>",
            lambda _match: f"<think>{think_text}</think>",
            text,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )

    if not answer_text or _has_tag_fragment(answer_text):
        raise OutputFormatValidationError(
            f"{method}: raw_model_output <answer> block is empty or contains tag fragments",
            raw_model_output=text,
        )
    return text

def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if len(token) >= 2]

def _keyword_overlap_score(query: str, text: str) -> float:
    q = set(_tokenize(query))
    t = set(_tokenize(text))
    if not q or not t:
        return 0.0
    return len(q.intersection(t)) / max(1, len(q))

def _llm_budget(max_output_tokens: int = 256) -> Dict[str, Any]:
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": max_output_tokens,
        "timeout_seconds": 120,
    }

def _chat(llm: OpenAICompatibleLLM, system: str, user: str, max_output_tokens: int = 256) -> Dict[str, Any]:
    return llm.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        _llm_budget(max_output_tokens=max_output_tokens),
    )

def _tagged_output_instruction() -> str:
    return (
        "Output format is mandatory.\n"
        "Return exactly two XML-style tags and nothing else.\n"
        "Do not add preamble, markdown, code fences, or extra explanation outside the tags.\n"
        "<think>Put the brief reasoning here, grounded only in the provided material.</think>\n"
        "<answer>Put the final answer only here.</answer>"
    )

def _stage_metrics(
    *,
    retrieval_seconds: float = 0.0,
    recall_seconds: float = 0.0,
    rerank_seconds: float = 0.0,
    row_load_seconds: float = 0.0,
    query_embed_seconds: float = 0.0,
    generation_seconds: float = 0.0,
    embedding_requests: int = 0,
    chat_requests: int = 0,
    retrieval_strategy: str = "",
    candidate_count: int = 0,
    selected_count: int = 0,
) -> Dict[str, Any]:
    return {
        "retrieval_seconds": round(float(retrieval_seconds), 6),
        "recall_seconds": round(float(recall_seconds), 6),
        "rerank_seconds": round(float(rerank_seconds), 6),
        "row_load_seconds": round(float(row_load_seconds), 6),
        "query_embed_seconds": round(float(query_embed_seconds), 6),
        "generation_seconds": round(float(generation_seconds), 6),
        "total_stage_seconds": round(float(retrieval_seconds + generation_seconds), 6),
        "embedding_requests": int(embedding_requests),
        "chat_requests": int(chat_requests),
        "retrieval_strategy": str(retrieval_strategy or ""),
        "candidate_count": int(candidate_count),
        "selected_count": int(selected_count),
    }

def _evidence_records_from_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        records.append(
            {
                "evidence_id": str(row.get("chunk_id") or f"chunk_{idx}"),
                "text": str(row.get("text") or ""),
                "source_chunk_id": str(row.get("chunk_id") or ""),
                "source_doc_id": str(row.get("full_doc_id") or row.get("doc_id") or row.get("sample_id") or ""),
                "chunk_order_index": row.get("chunk_order_index"),
            }
        )
    return records

def _pick_rows(rows: List[Dict[str, Any]], count: int, seed: int) -> List[Dict[str, Any]]:
    if not rows:
        return []
    idxs = list(range(len(rows)))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    return [rows[i] for i in idxs[: min(count, len(rows))]]

def _selection_score(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{sample_id}".encode("utf-8")).hexdigest()

def _dataset_seed_offset(dataset: str) -> int:
    return int(hashlib.sha256(dataset.encode("utf-8")).hexdigest()[:8], 16) % 100000

def _select_val_rows_incremental(
    rows: List[Dict[str, Any]],
    count: int,
    seed: int,
    previous_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    row_map = {_stable_sample_id(row): row for row in rows}
    desired = min(count, len(row_map))
    selected_ids: List[str] = []
    seen = set()
    for sample_id in previous_ids:
        if sample_id in row_map and sample_id not in seen and len(selected_ids) < desired:
            selected_ids.append(sample_id)
            seen.add(sample_id)
    if len(selected_ids) < desired:
        remaining = [sample_id for sample_id in row_map.keys() if sample_id not in seen]
        remaining.sort(key=lambda sid: _selection_score(sid, seed))
        selected_ids.extend(remaining[: desired - len(selected_ids)])
    return [row_map[sample_id] for sample_id in selected_ids]

def _retrieve_chunks(query: str, chunks: Sequence[Dict[str, Any]], top_k: int, *, asset_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    top_k = max(1, top_k)
    if not chunks:
        return []
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [dict(chunk) for chunk in chunks[:top_k]]

    index: Optional[Dict[str, List[int]]] = None
    if asset_dir is not None:
        index = _get_keyword_index(asset_dir, chunks)

    candidate_idxs: set[int] = set()
    if index:
        for token in query_tokens[:16]:
            for idx in index.get(token, []):
                candidate_idxs.add(idx)
                if len(candidate_idxs) >= max(128, top_k * 16):
                    break
            if len(candidate_idxs) >= max(128, top_k * 16):
                break
    if not candidate_idxs:
        candidate_idxs = set(range(min(len(chunks), max(128, top_k * 32))))

    scored: List[tuple[float, int, Dict[str, Any]]] = []
    for idx in sorted(candidate_idxs):
        chunk = chunks[idx]
        text = str(chunk.get("text") or "")
        score = _keyword_overlap_score(query, text)
        if re.search(r"\b(19\d{2}|20\d{2}|q[1-4]|yoy|mom)\b", text.lower()):
            score += 0.05
        if re.search(r"\d", text):
            score += 0.03
        scored.append((score, idx, chunk))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [chunk for _, _, chunk in scored[:top_k]]

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

def _retriever_stage_metrics(retriever: Optional[DenseHybridRetriever]) -> Dict[str, Any]:
    stats = dict(getattr(retriever, "last_stats", {}) or {})
    return {
        "recall_seconds": float(stats.get("candidate_recall_seconds", 0.0) or 0.0),
        "rerank_seconds": float(stats.get("rerank_seconds", 0.0) or 0.0),
        "row_load_seconds": float(stats.get("row_load_seconds", 0.0) or 0.0),
        "query_embed_seconds": float(stats.get("query_embed_seconds", 0.0) or 0.0),
        "candidate_count": int(stats.get("candidate_count", 0) or 0),
        "selected_count": int(stats.get("selected_count", 0) or 0),
    }

def _slim_chunk_row(chunk: Dict[str, Any], *, idx: int, text_limit: int = 256) -> Dict[str, Any]:
    text = str(chunk.get("text") or "")
    slim = {
        "chunk_id": str(chunk.get("chunk_id") or f"chunk_{idx + 1}"),
        "text": text[:max(32, int(text_limit))],
    }
    score = chunk.get("score")
    if score is not None:
        slim["score"] = score
    return slim

def _keyword_index_cache() -> Dict[str, Dict[str, List[int]]]:
    cache = getattr(_KEYWORD_INDEX_CACHE_LOCAL, "cache", None)
    if cache is None:
        cache = {}
        _KEYWORD_INDEX_CACHE_LOCAL.cache = cache
    return cache

def _get_chunk_rows(asset_dir: Path) -> List[Dict[str, Any]]:
    cache = _chunk_rows_cache()
    corpus_chunks = asset_dir / "corpus" / "chunks.jsonl"
    cache_key = str(corpus_chunks.resolve())
    rows = cache.get(cache_key)
    if rows is None:
        rows = _load_asset_file(asset_dir, "corpus/chunks.jsonl")
        cache[cache_key] = rows
    return rows

def _build_keyword_index(chunks: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = collections.defaultdict(list)
    for idx, chunk in enumerate(chunks):
        for token in set(_tokenize(str(chunk.get("text") or ""))):
            index[token].append(idx)
    return index

def _get_keyword_index(asset_dir: Path, chunks: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
    cache = _keyword_index_cache()
    corpus_chunks = asset_dir / "corpus" / "chunks.jsonl"
    cache_key = str(corpus_chunks.resolve())
    index = cache.get(cache_key)
    if index is None:
        index = _build_keyword_index(chunks)
        cache[cache_key] = index
    return index

def _retrieve_asset_chunks(asset_dir: Path, query: str, top_k: int) -> tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    retriever = _dense_retriever_for_asset(asset_dir)
    rows = retriever.retrieve(query, top_k=top_k)
    if rows:
        slim_rows = [_slim_chunk_row(row, idx=i, text_limit=256) for i, row in enumerate(rows)]
        return slim_rows, "dense_hybrid_sqlite_cosine_topk", _retriever_stage_metrics(retriever)
    return [], "dense_hybrid_sqlite_cosine_topk", _retriever_stage_metrics(retriever)

def _prefetch_textual_retrieval_sample(
    method: str,
    dataset: str,
    asset_dir: Path,
    top_k: int,
    sample_id: str,
    query: str,
    query_vector: Any,
) -> tuple[tuple[str, str, str], Optional[Dict[str, Any]], Dict[str, Any]]:
    stats = {
        "retrieval_seconds": 0.0,
        "recall_seconds": 0.0,
        "samples_prefetched": 0,
        "samples_failed": 0,
    }
    retriever = _dense_retriever_for_asset(asset_dir)
    if retriever is None:
        stats["samples_failed"] = 1
        return (method, dataset, sample_id), None, stats
    retrieve_t0 = time.perf_counter()
    try:
        rows = retriever.retrieve_with_vector(query, query_vector, top_k=top_k)
    except Exception:
        rows = []
    retrieve_elapsed = time.perf_counter() - retrieve_t0
    retriever_metrics = _retriever_stage_metrics(retriever)
    if not rows:
        stats["samples_failed"] = 1
        return (method, dataset, sample_id), None, stats
    slim_rows = [_slim_chunk_row(row, idx=i, text_limit=256) for i, row in enumerate(rows)]
    stats["samples_prefetched"] = 1
    stats["retrieval_seconds"] = retrieve_elapsed
    stats["recall_seconds"] = float(retriever_metrics.get("recall_seconds", 0.0) or 0.0)
    return (
        (method, dataset, sample_id),
        {
            "top_chunks": slim_rows,
            "retrieval_strategy": "dense_hybrid_sqlite_cosine_topk_prefetched",
            "retrieval_seconds": retrieve_elapsed,
            **retriever_metrics,
        },
        stats,
    )

def _prefetch_query_vectors(
    grouped: Dict[tuple[str, str, str, int], List[tuple[str, Dict[str, Any], str]]],
    group_models: Dict[tuple[str, str, str, int], str],
    model_asset_dirs: Dict[str, Path],
    *,
    query_embed_batch_size: int,
) -> tuple[Dict[tuple[str, str], Any], Dict[str, Any]]:
    query_vectors: Dict[tuple[str, str], Any] = {}
    stats = {
        "embedding_requests": 0,
        "unique_queries": 0,
    }
    by_model: Dict[str, List[str]] = collections.defaultdict(list)
    seen_by_model: Dict[str, set[str]] = collections.defaultdict(set)
    for group_key, items in grouped.items():
        model_name = str(group_models.get(group_key) or "").strip()
        if not model_name:
            continue
        seen = seen_by_model[model_name]
        for _sample_id, _row, query in items:
            if query in seen:
                continue
            seen.add(query)
            by_model[model_name].append(query)
    batch_size = max(1, int(query_embed_batch_size))
    for model_name, raw_queries in by_model.items():
        if not raw_queries:
            continue
        asset_dir = model_asset_dirs.get(model_name)
        if asset_dir is None:
            continue
        retriever = _dense_retriever_for_asset(asset_dir)
        if retriever is None:
            continue
        client = retriever.client
        for start in range(0, len(raw_queries), batch_size):
            batch = raw_queries[start : start + batch_size]
            vectors = client.embed_texts(batch)
            stats["embedding_requests"] += 1
            stats["unique_queries"] += len(batch)
            for query, vector in zip(batch, vectors):
                query_vectors[(model_name, query)] = vector
    return query_vectors, stats

def _prefetch_textual_retrievals(
    tasks: Sequence[tuple[str, str, Dict[str, Any]]],
    asset_root: Path,
    asset_dataset: Optional[str],
    *,
    query_embed_batch_size: int,
    retrieval_mode: str,
    max_workers: int,
    show_progress: bool = True,
) -> tuple[Dict[tuple[str, str, str], Dict[str, Any]], Dict[str, Any]]:
    cache: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    stats = {
        "embedding_requests": 0,
        "retrieval_seconds": 0.0,
        "samples_prefetched": 0,
        "samples_failed": 0,
        "groups_prefetched": 0,
        "max_inflight_prefetch": 0,
    }
    if _is_retrieval_disabled(retrieval_mode):
        return cache, stats
    grouped: Dict[tuple[str, str, str, int], List[tuple[str, Dict[str, Any], str]]] = collections.defaultdict(list)
    group_models: Dict[tuple[str, str, str, int], str] = {}
    model_asset_dirs: Dict[str, Path] = {}
    for method, dataset, row in tasks:
        if method not in TEXTUAL_METHODS:
            continue
        asset_dir = _resolve_asset_dir(asset_root, method, dataset, asset_dataset)
        if not asset_dir.exists():
            continue
        retriever = _dense_retriever_for_asset(asset_dir)
        if retriever is None:
            continue
        sample_id = _stable_sample_id(row)
        top_k = 4
        group_key = (method, dataset, str(asset_dir.resolve()), top_k)
        grouped[group_key].append((sample_id, row, _build_retrieval_query(row)))
        model_name = str(retriever.client.model or "").strip()
        group_models[group_key] = model_name
        if model_name and model_name not in model_asset_dirs:
            model_asset_dirs[model_name] = asset_dir

    total_prefetch_samples = sum(len(items) for items in grouped.values())
    if total_prefetch_samples <= 0:
        return cache, stats

    query_vectors, embed_stats = _prefetch_query_vectors(
        grouped,
        group_models,
        model_asset_dirs,
        query_embed_batch_size=query_embed_batch_size,
    )
    stats["embedding_requests"] += int(embed_stats.get("embedding_requests", 0) or 0)
    sample_jobs: List[tuple[str, str, Path, int, str, str, Any]] = []
    for (method, dataset, asset_dir_raw, top_k), items in grouped.items():
        model_name = str(group_models.get((method, dataset, asset_dir_raw, top_k)) or "").strip()
        if not model_name:
            continue
        stats["groups_prefetched"] += 1
        for sample_id, _row, query in items:
            vector = query_vectors.get((model_name, query))
            if vector is None:
                stats["samples_failed"] += 1
                continue
            sample_jobs.append((method, dataset, Path(asset_dir_raw), top_k, sample_id, query, vector))
    worker_count = min(max(1, int(max_workers)), max(1, len(sample_jobs)))
    inflight_lock = threading.Lock()
    inflight_prefetch = 0
    missing_prefetch_jobs = total_prefetch_samples - len(sample_jobs)
    progress_bar = tqdm(total=total_prefetch_samples, desc="Prefetch retrieval", unit="sample") if show_progress else None
    try:
        if progress_bar is not None:
            if missing_prefetch_jobs > 0:
                progress_bar.update(missing_prefetch_jobs)
            progress_bar.set_postfix_str(
                f"ok={stats['samples_prefetched']} failed={stats['samples_failed']} embed_reqs={stats['embedding_requests']} retrieval_s={stats['retrieval_seconds']:.1f}"
            )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            def _submit_prefetch(job: tuple[str, str, Path, int, str, str, Any]):
                nonlocal inflight_prefetch
                with inflight_lock:
                    inflight_prefetch += 1
                    stats["max_inflight_prefetch"] = max(stats["max_inflight_prefetch"], inflight_prefetch)
                try:
                    return _prefetch_textual_retrieval_sample(*job)
                finally:
                    with inflight_lock:
                        inflight_prefetch -= 1

            prefetch_limit = max(1, min(len(sample_jobs), max(1, worker_count * 2)))
            future_map = {}
            job_iter = iter(sample_jobs)
            for _ in range(prefetch_limit):
                try:
                    job = next(job_iter)
                except StopIteration:
                    break
                future_map[executor.submit(_submit_prefetch, job)] = job
            while future_map:
                done, _ = wait(set(future_map.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    future_map.pop(future, None)
                    try:
                        cache_key, cache_value, sample_stats = future.result()
                    except Exception:
                        cache_key, cache_value, sample_stats = None, None, {"retrieval_seconds": 0.0, "samples_prefetched": 0, "samples_failed": 1}
                    if cache_key is not None and cache_value is not None:
                        cache[cache_key] = cache_value
                    stats["retrieval_seconds"] += float(sample_stats.get("retrieval_seconds", 0.0) or 0.0)
                    stats["samples_prefetched"] += int(sample_stats.get("samples_prefetched", 0) or 0)
                    stats["samples_failed"] += int(sample_stats.get("samples_failed", 0) or 0)
                    if progress_bar is not None:
                        progress_bar.update(1)
                        progress_bar.set_postfix_str(
                            f"ok={stats['samples_prefetched']} failed={stats['samples_failed']} embed_reqs={stats['embedding_requests']} retrieval_s={stats['retrieval_seconds']:.1f}"
                        )
                    try:
                        job = next(job_iter)
                    except StopIteration:
                        continue
                    future_map[executor.submit(_submit_prefetch, job)] = job
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return cache, stats

def _chunked_tasks(
    tasks: Sequence[tuple[str, str, Dict[str, Any]]],
    chunk_size: int,
) -> Iterable[List[tuple[str, str, Dict[str, Any]]]]:
    size = max(1, int(chunk_size))
    for start in range(0, len(tasks), size):
        yield list(tasks[start : start + size])

def _interleave_task_buckets(
    buckets: Sequence[Sequence[tuple[str, str, Dict[str, Any]]]],
) -> List[tuple[str, str, Dict[str, Any]]]:
    queues: List[collections.deque[tuple[str, str, Dict[str, Any]]]] = [
        collections.deque(bucket) for bucket in buckets if bucket
    ]
    tasks: List[tuple[str, str, Dict[str, Any]]] = []
    while queues:
        next_round: List[collections.deque[tuple[str, str, Dict[str, Any]]]] = []
        for queue in queues:
            if not queue:
                continue
            tasks.append(queue.popleft())
            if queue:
                next_round.append(queue)
        queues = next_round
    return tasks

def _bounded_as_completed_map(
    executor: ThreadPoolExecutor,
    items: Sequence[Any],
    submit_fn,
    *,
    max_in_flight: int,
) -> Iterable[tuple[Any, Any]]:
    limit = max(1, int(max_in_flight))
    iterator = iter(items)
    future_map: Dict[Any, Any] = {}

    def _fill() -> None:
        while len(future_map) < limit:
            try:
                item = next(iterator)
            except StopIteration:
                return
            future = executor.submit(submit_fn, item)
            future_map[future] = item

    _fill()
    while future_map:
        done, _ = wait(set(future_map.keys()), return_when=FIRST_COMPLETED)
        for future in done:
            item = future_map.pop(future, None)
            if item is None:
                continue
            yield future, item
        _fill()

def _resolve_eval_inflight_limit(method: str, default_limit: int) -> int:
    return max(1, int(default_limit))

def _asset_file_cache() -> Dict[str, List[Dict[str, Any]]]:
    return _ASSET_FILE_CACHE

def _chunk_rows_cache() -> Dict[str, List[Dict[str, Any]]]:
    return _CHUNK_ROWS_CACHE

def _keyword_index_cache() -> Dict[str, Dict[str, List[int]]]:
    return _KEYWORD_INDEX_CACHE

def _shared_file_signature_cache() -> Dict[str, tuple[float, int]]:
    return _SHARED_FILE_SIGNATURE_CACHE

def _shared_rows_by_signature() -> Dict[tuple[float, int], List[Dict[str, Any]]]:
    return _SHARED_ROWS_BY_SIGNATURE

def _load_asset_file(asset_dir: Path, relative_path: str) -> List[Dict[str, Any]]:
    path = asset_dir / relative_path
    cache_key = str(path.resolve())
    with _ASSET_CACHE_LOCK:
        exists_cache = _ASSET_EXISTS_CACHE
        exists = exists_cache.get(cache_key)
        if exists is None:
            exists = path.exists()
            exists_cache[cache_key] = exists
        if not exists:
            return []
        cache = _asset_file_cache()
        signature_cache = _shared_file_signature_cache()
        shared_rows_cache = _shared_rows_by_signature()
        signature = (path.stat().st_mtime, path.stat().st_size)
        rows = cache.get(cache_key)
        cached_signature = signature_cache.get(cache_key)
        if rows is None or cached_signature != signature:
            rows = shared_rows_cache.get(signature)
            if rows is None:
                rows = _load_jsonl(path)
                shared_rows_cache[signature] = rows
            cache[cache_key] = rows
            signature_cache[cache_key] = signature
        return rows

def _get_cached_chunk_texts(asset_dir: Path, relative_path: str) -> List[Dict[str, Any]]:
    return _load_asset_file(asset_dir, relative_path)

def _evict_asset_caches(asset_dir: Path, *, keep_paths: Optional[set[str]] = None) -> None:
    keep_paths = set(keep_paths or set())
    resolved_asset = str(asset_dir.resolve())
    with _ASSET_CACHE_LOCK:
        file_cache = _asset_file_cache()
        signature_cache = _shared_file_signature_cache()
        shared_rows_cache = _shared_rows_by_signature()
        chunk_cache = _chunk_rows_cache()
        keyword_cache = _keyword_index_cache()
        dense_cache = getattr(_DENSE_RETRIEVER_LOCAL, "cache", None)
        exists_cache = _ASSET_EXISTS_CACHE

        for path_str in list(file_cache.keys()):
            if not path_str.startswith(resolved_asset) or any(path_str.endswith(keep) for keep in keep_paths):
                continue
            rows = file_cache.pop(path_str, None)
            signature = signature_cache.pop(path_str, None)
            exists_cache.pop(path_str, None)
            if signature is not None and rows is shared_rows_cache.get(signature):
                shared_rows_cache.pop(signature, None)
        for path_str in list(chunk_cache.keys()):
            if path_str.startswith(resolved_asset):
                chunk_cache.pop(path_str, None)
        for path_str in list(keyword_cache.keys()):
            if path_str.startswith(resolved_asset):
                keyword_cache.pop(path_str, None)
        if dense_cache is not None:
            dense_cache.pop(resolved_asset, None)
def _evaluate_runtime_adapter(
    method: str,
    dataset: str,
    asset_dir: Optional[Path],
    sample: Dict[str, Any],
    *,
    disable_retrieval: bool = False,
) -> Dict[str, Any]:
    asset_required = not disable_retrieval
    runtime_method = method
    if asset_required and asset_dir is None:
        raise FileNotFoundError(f"missing asset directory for method={method} dataset={dataset}")
    adapter_cache = getattr(_RUNTIME_ADAPTER_LOCAL, "adapter_cache", None)
    if adapter_cache is None:
        adapter_cache = {}
        _RUNTIME_ADAPTER_LOCAL.adapter_cache = adapter_cache
    cache_key = (runtime_method, str(asset_dir.resolve()) if asset_dir is not None else "__no_asset__", "no_rag" if disable_retrieval else "rag")
    adapter = adapter_cache.get(cache_key)
    if adapter is None:
        adapter = build_default_registry(instance_tag=f"built-eval-{runtime_method}-{threading.get_ident()}")[runtime_method]
        adapter_cache[cache_key] = adapter
    runtime_sample = dict(sample)
    runtime_sample.setdefault("dataset", dataset)
    runtime_sample.setdefault("source_dataset", dataset)
    runtime_sample["_finclaw_disable_rag_retrieval"] = bool(disable_retrieval)
    runtime_sample["_finclaw_asset_dataset"] = asset_dir.name if asset_dir is not None else dataset
    if asset_required and asset_dir is not None:
        runtime_sample["_finclaw_explicit_asset_dir"] = str(asset_dir)
    run_t0 = time.perf_counter()
    out = adapter.run_sample(
        runtime_sample,
        budget={"timeout_seconds": 120, "max_retrieval_top_k": 6, "max_output_tokens": 8192},
    )
    run_seconds = time.perf_counter() - run_t0
    extras = dict(out.get("extras") or {})
    if not extras.get("evidence_records") and out.get("evidence_ids"):
        evidence_lines = str(out.get("evidence_text") or "").split("\n")
        extras["evidence_records"] = [
            {
                "evidence_id": str(eid),
                "text": evidence_lines[idx] if idx < len(evidence_lines) else "",
            }
            for idx, eid in enumerate(out.get("evidence_ids") or [])
        ]
    retrieval_strategy = str(extras.get("retrieval_strategy") or ("disabled_no_rag" if disable_retrieval else "runtime_adapter"))
    embedding_requests = int(extras.get("embedding_requests", 0) or 0)
    chat_requests = int(extras.get("chat_requests", 0) or 0)
    if not disable_retrieval and retrieval_strategy.startswith("prebuilt_asset_dense_hybrid"):
        embedding_requests = max(1, embedding_requests)
    if disable_retrieval:
        embedding_requests = 0
    extras.update(
        _stage_metrics(
            retrieval_seconds=float(extras.get("retrieval_seconds", 0.0) or 0.0),
            recall_seconds=float(extras.get("recall_seconds", 0.0) or 0.0),
            rerank_seconds=float(extras.get("rerank_seconds", 0.0) or 0.0),
            row_load_seconds=float(extras.get("row_load_seconds", 0.0) or 0.0),
            query_embed_seconds=float(extras.get("query_embed_seconds", 0.0) or 0.0),
            generation_seconds=float(extras.get("generation_seconds", run_seconds) or run_seconds),
            embedding_requests=embedding_requests,
                chat_requests=max(chat_requests, 0),
            retrieval_strategy=retrieval_strategy,
            candidate_count=int(extras.get("candidate_count", 0) or 0),
            selected_count=int(extras.get("selected_count", 0) or 0),
        )
    )
    raw_model_output = str(extras.get("raw_model_output") or "").strip()
    extras["raw_model_output"] = _require_nonempty_tagged_raw_output(
        raw_model_output,
        method=method,
    )
    out["extras"] = extras
    return out

def _stable_sample_id(row: Dict[str, Any]) -> str:
    return str(row.get("id") or row.get("sample_id") or "unknown").strip() or "unknown"

def _load_existing_results(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()

    def _load_one(file_path: Path) -> None:
        resolved = str(file_path.resolve())
        if resolved in seen_paths or not file_path.exists():
            return
        seen_paths.add(resolved)
        rows.extend(_load_jsonl(file_path))

    if path.exists():
        _load_one(path)

    parent = path.parent
    if parent.exists():
        for nested in sorted(parent.glob("**/predictions.jsonl")):
            relative_parts = nested.relative_to(parent).parts
            if any(part.startswith("_") for part in relative_parts):
                continue
            _load_one(nested)

    return rows

def _completed_keys(rows: Sequence[Dict[str, Any]]) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for row in rows:
        if row.get("status") != "completed":
            continue
        out.add((str(row.get("method") or ""), str(row.get("dataset") or ""), str(row.get("sample_id") or "")))
    return out

def _asset_method_name(method: str) -> str:
    return method

def _method_specific_asset_roots(asset_root: Path, method: str) -> List[Path]:
    return roots

def _resolve_asset_dir(asset_root: Path, method: str, dataset: str, asset_dataset: Optional[str]) -> Path:
    method_name = _asset_method_name(method)
    target_name = asset_dataset or dataset
    for root in _method_specific_asset_roots(asset_root, method):
        candidate = root / method_name / target_name
        if candidate.exists():
            return candidate
    return asset_root / method_name / target_name

def _candidate_asset_dirs(asset_root: Path, method: str, dataset: str, asset_dataset: Optional[str]) -> List[Path]:
    direct_candidates: List[Path] = []
    shared_candidates: List[Path] = []
    method_name = _asset_method_name(method)
    target_name = asset_dataset or dataset
    for root in _method_specific_asset_roots(asset_root, method):
        candidate = root / method_name / target_name
        if candidate not in direct_candidates:
            direct_candidates.append(candidate)
        shared_dir = root / "_shared" / target_name
        if shared_dir not in shared_candidates:
            shared_candidates.append(shared_dir)
    return direct_candidates + shared_candidates

def _cleanup_asset_caches_for_dataset(asset_root: Path, dataset: str, asset_dataset: Optional[str]) -> None:
    candidates = _candidate_asset_dirs(asset_root, "", dataset, asset_dataset)
    for path in candidates:
        if path.exists():
            _evict_asset_caches(path, keep_paths={"corpus/chunks.jsonl"})

def _select_eval_rows(rows: List[Dict[str, Any]], count: int, seed: int, previous_ids: Sequence[str], dataset: str) -> List[Dict[str, Any]]:
    if count <= 0:
        return rows
    return _select_val_rows_incremental(rows, count, seed + _dataset_seed_offset(dataset), previous_ids)

def _evaluate_one(
    method: str,
    dataset: str,
    asset_root: Path,
    sample: Dict[str, Any],
    llm: OpenAICompatibleLLM,
    asset_dataset: Optional[str],
    retrieval_mode: str = "enabled",
) -> Dict[str, Any]:
    disable_retrieval = _is_retrieval_disabled(retrieval_mode)
    sample_meta = _sample_record(sample)
    asset_dir = _resolve_asset_dir(asset_root, method, dataset, asset_dataset)
    candidate_dirs = _candidate_asset_dirs(asset_root, method, dataset, asset_dataset)
    available_asset_dir = next((path for path in candidate_dirs if path.exists()), None)
    if not disable_retrieval and available_asset_dir is None:
        return {
            **sample_meta,
            "dataset": dataset,
            "method": method,
            "status": "missing_assets",
            "asset_dataset": asset_dataset or dataset,
        }

    try:
        out = _evaluate_runtime_adapter(method, dataset, available_asset_dir, sample, disable_retrieval=disable_retrieval)
    except FileNotFoundError as exc:
        return {
            **sample_meta,
            "dataset": dataset,
            "method": method,
            "status": "missing_assets",
            "error": str(exc),
            "asset_dataset": asset_dataset or dataset,
        }
    except Exception as exc:
        failed_row = {
            **sample_meta,
            "dataset": dataset,
            "method": method,
            "status": "failed",
            "error": str(exc),
            "asset_dataset": asset_dataset or dataset,
        }
        raw_model_output = str(getattr(exc, "raw_model_output", "") or "")
        failed_row["extras"] = {"raw_model_output": raw_model_output}
        return failed_row
    finally:
        if available_asset_dir is not None and available_asset_dir.name == (asset_dataset or dataset):
            _evict_asset_caches(available_asset_dir)

    return {
        **sample_meta,
        "dataset": dataset,
        "method": method,
        "status": "completed",
        "asset_dataset": asset_dataset or dataset,
        **out,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate built retrieval assets using 5 random val samples per dataset.")
    parser.add_argument("--dataset-root", default="datasets/latest/kg")
    parser.add_argument("--asset-root", default="kg")
    parser.add_argument("--output-root", default="experiments/outputs/built_asset_eval")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--exclude-datasets", default="")
    parser.add_argument("--samples-per-dataset", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--max-workers-per-sample", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=1)
    parser.add_argument("--benchmark-output", default="")
    parser.add_argument("--benchmark-label", default="single_sample_benchmark")
    parser.add_argument("--asset-dataset", default="")
    parser.add_argument("--retrieval-mode", choices=["enabled", "disabled"], default="enabled")
    args = parser.parse_args()

    dataset_root = _repo_path(args.dataset_root)
    asset_root = _repo_path(args.asset_root)
    output_root = _repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    methods = ["financeclaw"]
    dataset_filter = {item.strip() for item in str(args.datasets).split(",") if item.strip()} or None
    exclude_dataset_filter = {item.strip() for item in str(args.exclude_datasets).split(",") if item.strip()}
    asset_dataset = str(args.asset_dataset).strip() or None

    llm = OpenAICompatibleLLM()
    sample_manifest_path = output_root / "sample_manifest.json"
    existing_sample_manifest = _load_json(sample_manifest_path) if sample_manifest_path.exists() else {}
    predictions_path = output_root / "predictions.jsonl"
    existing_results = _load_existing_results(predictions_path)
    done_keys = _completed_keys(existing_results)
    task_buckets: List[List[tuple[str, str, Dict[str, Any]]]] = []
    sample_manifest: Dict[str, List[str]] = {}
    dataset_selection_stats: Dict[str, Dict[str, int]] = {}
    total_dataset_rows = 0
    total_selected_samples = 0
    total_pending_tasks = 0
    for dataset_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        dataset = dataset_dir.name
        if dataset in exclude_dataset_filter:
            continue
        if dataset_filter and dataset not in dataset_filter:
            continue
        val_path = dataset_dir / "val.jsonl"
        if not val_path.exists():
            continue
        rows = _load_jsonl(val_path)
        row_count = len(rows)
        previous_ids = existing_sample_manifest.get(dataset, []) if isinstance(existing_sample_manifest, dict) else []
        sample_count = 1 if args.sample_limit == 1 else args.samples_per_dataset
        selected = _select_eval_rows(rows, sample_count, args.seed, previous_ids, dataset)
        if not selected:
            continue
        sample_manifest[dataset] = [_stable_sample_id(row) for row in selected]
        dataset_selection_stats[dataset] = {
            "rows": row_count,
            "selected": len(selected),
        }
        total_dataset_rows += row_count
        total_selected_samples += len(selected)
        for method in methods:
            method_tasks: List[tuple[str, str, Dict[str, Any]]] = []
            for row in selected:
                key = (method, dataset, _stable_sample_id(row))
                if key in done_keys:
                    continue
                method_tasks.append((method, dataset, row))
                total_pending_tasks += 1
            if method_tasks:
                task_buckets.append(method_tasks)

    tasks = _interleave_task_buckets(task_buckets)

    _write_json(sample_manifest_path, sample_manifest)

    results_by_group: Dict[tuple[str, str], Dict[str, Any]] = collections.defaultdict(_init_group_state)
    for row in existing_results:
        key = _group_key(row)
        state = results_by_group[key]
        state["count"] += 1
        state["completed"] += int(row.get("status") == "completed")
        state["missing_assets"] += int(row.get("status") == "missing_assets")
        state["failed"] += int(row.get("status") == "failed")
        state["unsupported_method"] += int(row.get("status") == "unsupported_method")
        extras = row.get("extras") or {}
        state["embedding_requests"] += int(extras.get("embedding_requests", 0) or 0)
        state["chat_requests"] += int(extras.get("chat_requests", 0) or 0)
        state["retrieval_seconds"] += float(extras.get("retrieval_seconds", 0.0) or 0.0)
        state["generation_seconds"] += float(extras.get("generation_seconds", 0.0) or 0.0)
        state["sample_ids"].append(str(row.get("sample_id") or ""))
    progress_lock = threading.Lock()
    aggregate_embedding_requests = sum(int((row.get("extras") or {}).get("embedding_requests", 0) or 0) for row in existing_results)
    aggregate_chat_requests = sum(int((row.get("extras") or {}).get("chat_requests", 0) or 0) for row in existing_results)
    aggregate_retrieval_seconds = sum(float((row.get("extras") or {}).get("retrieval_seconds", 0.0) or 0.0) for row in existing_results)
    aggregate_generation_seconds = sum(float((row.get("extras") or {}).get("generation_seconds", 0.0) or 0.0) for row in existing_results)
    aggregate_completed = sum(1 for row in existing_results if row.get("status") == "completed")
    aggregate_failed = sum(1 for row in existing_results if row.get("status") == "failed")
    eval_t0 = time.perf_counter()
    eval_inflight_lock = threading.Lock()
    eval_inflight = 0
    max_inflight_eval = 0
    prediction_handles: Dict[tuple[str, str], Any] = {}
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            def _submit_eval(
                item: tuple[str, str, Dict[str, Any]],
            ):
                method, dataset, row = item
                nonlocal eval_inflight, max_inflight_eval
                with eval_inflight_lock:
                    eval_inflight += 1
                    max_inflight_eval = max(max_inflight_eval, eval_inflight)
                try:
                    return _evaluate_one(method, dataset, asset_root, row, llm, asset_dataset, args.retrieval_mode)
                finally:
                    with eval_inflight_lock:
                        eval_inflight -= 1

            with tqdm(total=len(tasks), desc="Evaluate assets", unit="task") as pbar:
                def _update_postfix() -> None:
                    progress_meta = (
                        f"datasets={len(dataset_selection_stats)} "
                        f"rows={total_dataset_rows} selected={total_selected_samples} pending={total_pending_tasks} "
                        f"ok={aggregate_completed} failed={aggregate_failed} "
                        f"embed_reqs={aggregate_embedding_requests} chat_reqs={aggregate_chat_requests} "
                        f"retrieval_s={aggregate_retrieval_seconds:.1f} generation_s={aggregate_generation_seconds:.1f}"
                    )
                    pbar.set_postfix_str(progress_meta)

                _update_postfix()
                futures = [executor.submit(_submit_eval, (method, dataset, row)) for method, dataset, row in tasks]
                for future in as_completed(futures):
                    result = _prepare_row_for_persistence(future.result())
                    key = _group_key(result)
                    state = results_by_group[key]
                    if key not in prediction_handles:
                        group_dir = output_root / key[0] / key[1]
                        group_dir.mkdir(parents=True, exist_ok=True)
                        handle = open(group_dir / "predictions.jsonl", "a", encoding="utf-8", buffering=1024 * 1024)
                        prediction_handles[key] = handle
                        state["handle"] = handle
                    _append_jsonl_row(prediction_handles[key], result, durable=False)
                    state["count"] += 1
                    state["completed"] += int(result.get("status") == "completed")
                    state["missing_assets"] += int(result.get("status") == "missing_assets")
                    state["failed"] += int(result.get("status") == "failed")
                    state["unsupported_method"] += int(result.get("status") == "unsupported_method")
                    extras = result.get("extras") or {}
                    state["embedding_requests"] += int(extras.get("embedding_requests", 0) or 0)
                    state["chat_requests"] += int(extras.get("chat_requests", 0) or 0)
                    state["retrieval_seconds"] += float(extras.get("retrieval_seconds", 0.0) or 0.0)
                    state["generation_seconds"] += float(extras.get("generation_seconds", 0.0) or 0.0)
                    state["sample_ids"].append(str(result.get("sample_id") or ""))
                    aggregate_embedding_requests += int(extras.get("embedding_requests", 0) or 0)
                    aggregate_chat_requests += int(extras.get("chat_requests", 0) or 0)
                    aggregate_retrieval_seconds += float(extras.get("retrieval_seconds", 0.0) or 0.0)
                    aggregate_generation_seconds += float(extras.get("generation_seconds", 0.0) or 0.0)
                    if result.get("status") == "completed":
                        aggregate_completed += 1
                    elif result.get("status") == "failed":
                        aggregate_failed += 1
                    _update_postfix()
                    tqdm.write(
                        f"[{result['status']}] {result['method']} {result['dataset']} {result['sample_id']} "
                        f"({pbar.n + 1}/{pbar.total} tasks, selected={total_selected_samples}, rows={total_dataset_rows})"
                    )
                    pbar.update(1)
    finally:
        for handle in prediction_handles.values():
            try:
                handle.close()
            except Exception:
                pass
    evaluation_wall_seconds = time.perf_counter() - eval_t0
    _write_group_outputs(output_root, results_by_group)
    total_wall_seconds = evaluation_wall_seconds

    benchmark_memory_after_mb = _sample_memory_mb() if args.benchmark_output else 0.0
    benchmark_records: List[Dict[str, Any]] = []
    if args.benchmark_output:
        for (dataset, method), state in results_by_group.items():
            benchmark_records.append(
                _capture_run_metrics(
                    method=method,
                    dataset=dataset,
                    count=int(state.get("count", 0) or 0),
                    completed=int(state.get("completed", 0) or 0),
                    failed=int(state.get("failed", 0) or 0),
                    unsupported_method=int(state.get("unsupported_method", 0) or 0),
                    retrieval_seconds=float(state.get("retrieval_seconds", 0.0) or 0.0),
                    generation_seconds=float(state.get("generation_seconds", 0.0) or 0.0),
                    max_inflight_eval=max_inflight_eval,
                    max_inflight_prefetch=0,
                    memory_before_mb=0.0,
                    memory_after_mb=benchmark_memory_after_mb,
                )
            )
        tqdm.write(
            f"[summary] datasets={len(dataset_selection_stats)} rows={total_dataset_rows} selected={total_selected_samples} "
            f"pending={total_pending_tasks} completed={aggregate_completed} failed={aggregate_failed}"
        )
        benchmark_path = _repo_path(args.benchmark_output)
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            benchmark_path,
            {
                "label": args.benchmark_label,
                "generated_at": _utc_now(),
                "records": benchmark_records,
            },
        )

    summary: Dict[str, Any] = {
        "generated_at": _utc_now(),
        "methods": methods,
        "datasets": sorted(sample_manifest.keys()),
        "exclude_datasets": sorted(exclude_dataset_filter),
        "samples_per_dataset": args.samples_per_dataset,
        "asset_dataset": asset_dataset,
        "retrieval_mode": args.retrieval_mode,
        "evaluation_wall_seconds": round(evaluation_wall_seconds, 6),
        "total_wall_seconds": round(total_wall_seconds, 6),
        "submitted_tasks": len(tasks),
        "throughput_samples_per_second": round(len(tasks) / max(1e-9, total_wall_seconds), 6),
        "max_inflight_eval": int(max_inflight_eval),
        "result_count": sum(int(state.get("count", 0) or 0) for state in results_by_group.values()),
        "completed": sum(int(state.get("completed", 0) or 0) for state in results_by_group.values()),
        "missing_assets": sum(int(state.get("missing_assets", 0) or 0) for state in results_by_group.values()),
        "failed": sum(int(state.get("failed", 0) or 0) for state in results_by_group.values()),
        "unsupported_method": sum(int(state.get("unsupported_method", 0) or 0) for state in results_by_group.values()),
        "embedding_requests": aggregate_embedding_requests,
        "chat_requests": aggregate_chat_requests,
        "retrieval_seconds": round(aggregate_retrieval_seconds, 6),
        "generation_seconds": round(aggregate_generation_seconds, 6),
        "avg_retrieval_seconds_per_completed": round(aggregate_retrieval_seconds / max(1, aggregate_completed), 6),
        "avg_generation_seconds_per_completed": round(aggregate_generation_seconds / max(1, aggregate_completed), 6),
    }
    _write_json(output_root / "summary.json", summary)

if __name__ == "__main__":
    main()
