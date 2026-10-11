from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from numpy.lib.format import open_memmap

from .dense_hybrid import (
    EmbeddingClient,
    _build_corpus_key,
    _iter_jsonl_with_offsets,
    _normalize_vector,
    _normalize_vectors,
    _read_json,
    _repo_path,
    _tokenize,
    _write_json,
)

try:
    import faiss
except Exception:
    faiss = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None

try:
    import tantivy
except Exception:
    tantivy = None

FAST_BACKEND_DEPENDENCIES_AVAILABLE = faiss is not None and pa is not None and pq is not None and tantivy is not None
FAST_BACKEND_RETRIEVAL_MODE = "tantivy_bm25_faiss_hnsw_parquet"

def _require_fast_backend_dependencies() -> None:
    if FAST_BACKEND_DEPENDENCIES_AVAILABLE:
        return
    raise RuntimeError(
        "Fast retrieval backend requires optional dependencies `faiss-cpu`, `pyarrow`, and `tantivy`. "
        "Install the retrieval-native environment or add those packages before using backend=fast."
    )

@dataclass
class FastIndexPaths:
    asset_dir: Path
    manifest_path: Path
    shared_manifest_path: Path
    shared_cache_dir: Path
    chunks_path: Path
    faiss_path: Path
    vectors_path: Path
    metadata_path: Path
    tantivy_dir: Path
    failed_rows_path: Path

def _resolve_fast_index_paths(asset_dir: Path, shared_cache_root: Path | None = None) -> FastIndexPaths:
    asset_dir = _repo_path(asset_dir)
    if shared_cache_root is None:
        shared_cache_root = asset_dir.parents[1] / "_dense_cache"
    corpus_key = _build_corpus_key(asset_dir)
    shared_cache_dir = _repo_path(shared_cache_root) / corpus_key
    return FastIndexPaths(
        asset_dir=asset_dir,
        manifest_path=asset_dir / "dense_index" / "manifest.json",
        shared_manifest_path=shared_cache_dir / "manifest.json",
        shared_cache_dir=shared_cache_dir,
        chunks_path=asset_dir / "corpus" / "chunks.jsonl",
        faiss_path=shared_cache_dir / "faiss_hnsw_ip.index",
        vectors_path=shared_cache_dir / "vectors.f16.npy",
        metadata_path=shared_cache_dir / "metadata.parquet",
        tantivy_dir=shared_cache_dir / "tantivy_index",
        failed_rows_path=shared_cache_dir / "failed_rows.jsonl",
    )

def _resolve_fast_index_paths_with_corpus_key(
    asset_dir: Path,
    *,
    corpus_key_asset_dir: Path,
    shared_cache_root: Path | None = None,
) -> FastIndexPaths:
    asset_dir = _repo_path(asset_dir)
    corpus_key_asset_dir = _repo_path(corpus_key_asset_dir)
    if shared_cache_root is None:
        shared_cache_root = asset_dir.parents[1] / "_dense_cache"
    corpus_key = _build_corpus_key(corpus_key_asset_dir)
    shared_cache_dir = _repo_path(shared_cache_root) / corpus_key
    return FastIndexPaths(
        asset_dir=asset_dir,
        manifest_path=asset_dir / "dense_index" / "manifest.json",
        shared_manifest_path=shared_cache_dir / "manifest.json",
        shared_cache_dir=shared_cache_dir,
        chunks_path=asset_dir / "corpus" / "chunks.jsonl",
        faiss_path=shared_cache_dir / "faiss_hnsw_ip.index",
        vectors_path=shared_cache_dir / "vectors.f16.npy",
        metadata_path=shared_cache_dir / "metadata.parquet",
        tantivy_dir=shared_cache_dir / "tantivy_index",
        failed_rows_path=shared_cache_dir / "failed_rows.jsonl",
    )

def _write_method_manifest(
    paths: FastIndexPaths,
    *,
    asset_dir: Path,
    embedding_model: str,
    success_chunk_count: int = 0,
    failed_chunk_count: int = 0,
    state: str = "completed",
) -> None:
    method_manifest = {
        "state": state,
        "asset_dir": str(asset_dir),
        "shared_cache_dir": str(paths.shared_cache_dir),
        "shared_manifest_path": str(paths.shared_manifest_path),
        "embedding_model": embedding_model,
        "success_chunk_count": success_chunk_count,
        "failed_chunk_count": failed_chunk_count,
        "retrieval_mode": FAST_BACKEND_RETRIEVAL_MODE,
    }
    _write_json(paths.manifest_path, method_manifest)

def _replace_file_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(src), str(dst))

def _replace_dir_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        backup = dst.with_name(f"{dst.name}.bak-{uuid.uuid4().hex}")
        os.replace(str(dst), str(backup))
        try:
            os.replace(str(src), str(dst))
        except Exception:
            if not dst.exists():
                os.replace(str(backup), str(dst))
            raise
        finally:
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
        return
    os.replace(str(src), str(dst))

def build_fast_index_for_asset(
    asset_dir: str | Path,
    *,
    shared_cache_root: str | Path | None = None,
    shared_asset_dir: str | Path | None = None,
    batch_size: int = 32,
    embed_workers: int = 1,
    force: bool = False,
    attach_only: bool = False,
    progress_cb=None,
) -> Dict[str, Any]:
    _require_fast_backend_dependencies()
    asset_dir = _repo_path(asset_dir)
    if shared_asset_dir is not None:
        paths = _resolve_fast_index_paths_with_corpus_key(
            asset_dir,
            corpus_key_asset_dir=_repo_path(shared_asset_dir),
            shared_cache_root=_repo_path(shared_cache_root) if shared_cache_root else None,
        )
    else:
        paths = _resolve_fast_index_paths(asset_dir, _repo_path(shared_cache_root) if shared_cache_root else None)
    client = EmbeddingClient()
    shared_manifest = _read_json(paths.shared_manifest_path) if paths.shared_manifest_path.exists() else {}
    shared_state = str(shared_manifest.get("state") or "").strip()
    shared_success_chunk_count = int(shared_manifest.get("success_chunk_count", 0) or 0)
    shared_failed_chunk_count = int(shared_manifest.get("failed_chunk_count", 0) or 0)
    shared_is_usable = shared_state in {"completed", "completed_with_failures"}
    shared_is_fully_complete = shared_state == "completed" and shared_failed_chunk_count == 0
    if attach_only and shared_is_usable and paths.faiss_path.exists() and paths.metadata_path.exists() and paths.tantivy_dir.exists():
        _write_method_manifest(
            paths,
            asset_dir=asset_dir,
            embedding_model=str(shared_manifest.get("embedding_model") or client.model),
            success_chunk_count=shared_success_chunk_count,
            failed_chunk_count=shared_failed_chunk_count,
            state="completed_with_failures" if shared_failed_chunk_count else "completed",
        )
        return {
            "status": "attached_shared_cache",
            "asset_dir": str(asset_dir),
            "shared_cache_dir": str(paths.shared_cache_dir),
            "success_chunk_count": shared_success_chunk_count,
            "failed_chunk_count": shared_failed_chunk_count,
        }
    if attach_only:
        raise RuntimeError(
            f"Shared dense index is not ready for attach-only asset {asset_dir}. "
            f"shared_manifest={paths.shared_manifest_path}"
        )
    if shared_is_fully_complete and not force:
        _write_method_manifest(
            paths,
            asset_dir=asset_dir,
            embedding_model=str(shared_manifest.get("embedding_model") or client.model),
            success_chunk_count=shared_success_chunk_count,
            failed_chunk_count=0,
            state="completed",
        )
        return {
            "status": "skipped_resume",
            "asset_dir": str(asset_dir),
            "shared_cache_dir": str(paths.shared_cache_dir),
            "success_chunk_count": shared_success_chunk_count,
            "failed_chunk_count": 0,
        }

    chunks_path = paths.chunks_path
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    build_manifest = _read_json(asset_dir / "build_manifest.json") if (asset_dir / "build_manifest.json").exists() else {}
    expected_chunk_count = int(build_manifest.get("chunk_count", 0) or 0)
    _write_json(
        paths.shared_manifest_path,
        {
            "state": "running",
            "asset_dir": str(asset_dir),
            "embedding_model": client.model,
            "chunks_path": str(chunks_path),
            "expected_chunk_count": expected_chunk_count,
            "retrieval_mode": FAST_BACKEND_RETRIEVAL_MODE,
            "backend": "fast",
        },
    )

    if force:
        for path in (paths.faiss_path, paths.vectors_path, paths.metadata_path, paths.failed_rows_path):
            if path.exists():
                path.unlink()
        if paths.tantivy_dir.exists():
            for child in sorted(paths.tantivy_dir.glob("**/*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            if paths.tantivy_dir.exists():
                paths.tantivy_dir.rmdir()

    effective_embed_workers = max(1, int(embed_workers))
    metadata_rows: List[Dict[str, Any]] = []
    vectors_out: List[Optional[np.ndarray]] = []
    failed_rows: List[Dict[str, Any]] = []
    texts_batch: List[str] = []
    row_ids_batch: List[int] = []
    row_meta_batch: List[Dict[str, Any]] = []

    def _embed_batch_resilient(batch_texts: Sequence[str], batch_row_ids: Sequence[int], batch_meta: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not batch_texts:
            return []
        try:
            vectors = client.embed_texts(batch_texts)
            out: List[Dict[str, Any]] = []
            for idx, row_id in enumerate(batch_row_ids):
                out.append(
                    {
                        "row_id": int(row_id),
                        "chunk_id": str(batch_meta[idx].get("chunk_id") or ""),
                        "vector": np.asarray(vectors[idx], dtype=np.float32),
                    }
                )
            return out
        except Exception as batch_exc:
            out = []
            for single_text, row_id, meta in zip(batch_texts, batch_row_ids, batch_meta):
                try:
                    vector = client.embed_texts([single_text])[0]
                    out.append(
                        {
                            "row_id": int(row_id),
                            "chunk_id": str(meta.get("chunk_id") or ""),
                            "vector": np.asarray(vector, dtype=np.float32),
                        }
                    )
                except Exception as single_exc:
                    out.append(
                        {
                            "row_id": int(row_id),
                            "chunk_id": str(meta.get("chunk_id") or ""),
                            "error": str(single_exc),
                            "batch_error": str(batch_exc),
                        }
                    )
            return out

    def _consume_embed_results(results: Sequence[Dict[str, Any]]) -> None:
        for item in results:
            row_id = int(item.get("row_id", -1))
            if row_id < 0 or row_id >= len(vectors_out):
                continue
            if "vector" in item:
                vectors_out[row_id] = np.asarray(item["vector"], dtype=np.float32)
                if progress_cb:
                    progress_cb(1)
            else:
                failed_rows.append(
                    {
                        "row_id": row_id,
                        "chunk_id": str(item.get("chunk_id") or ""),
                        "error": str(item.get("error") or "embedding_error"),
                        "batch_error": str(item.get("batch_error") or "batch_error"),
                    }
                )

    executor: ThreadPoolExecutor | None = None
    inflight: Dict[Any, tuple[List[int], List[Dict[str, Any]]]] = {}

    def _wait_for_one() -> None:
        if not inflight:
            return
        done, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
        for future in done:
            inflight.pop(future, None)
            _consume_embed_results(future.result())

    def _process_batch() -> None:
        nonlocal texts_batch, row_ids_batch, row_meta_batch, executor
        if not texts_batch:
            return
        batch_texts = texts_batch
        batch_row_ids = row_ids_batch
        batch_meta = row_meta_batch
        texts_batch = []
        row_ids_batch = []
        row_meta_batch = []
        if effective_embed_workers <= 1:
            _consume_embed_results(_embed_batch_resilient(batch_texts, batch_row_ids, batch_meta))
            return
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=effective_embed_workers)
        while len(inflight) >= effective_embed_workers:
            _wait_for_one()
        future = executor.submit(_embed_batch_resilient, batch_texts, batch_row_ids, batch_meta)
        inflight[future] = (batch_row_ids, batch_meta)

    try:
        row_count = 0
        for _offset, row in _iter_jsonl_with_offsets(chunks_path):
            chunk_id = str(row.get("chunk_id") or "").strip()
            text = str(row.get("text") or "").strip()
            if not chunk_id or not text:
                continue
            metadata_rows.append(
                {
                    "row_id": row_count,
                    "chunk_id": chunk_id,
                    "doc_id": str(row.get("doc_id") or "").strip(),
                    "sample_id": str(row.get("sample_id") or "").strip(),
                    "chunk_index": int(row.get("chunk_index", 0) or 0),
                    "text": text,
                    "tokens_json": json.dumps(list(dict.fromkeys([str(token).lower() for token in (row.get("tokens") or []) if str(token).strip()] or _tokenize(text))), ensure_ascii=False),
                }
            )
            vectors_out.append(None)
            texts_batch.append(text)
            row_ids_batch.append(row_count)
            row_meta_batch.append({"chunk_id": chunk_id})
            row_count += 1
            if len(texts_batch) >= max(1, batch_size):
                _process_batch()
        if texts_batch:
            _process_batch()
        while inflight:
            _wait_for_one()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if not any(vector is not None for vector in vectors_out):
        raise RuntimeError("Fast index build produced no vectors")
    first_vector = next(vector for vector in vectors_out if vector is not None)
    dim = int(first_vector.shape[0])
    build_token = uuid.uuid4().hex
    tmp_root = paths.shared_cache_dir / f".tmp-build-{build_token}"
    tmp_tantivy_dir = tmp_root / "tantivy_index"
    tmp_vectors_path = tmp_root / "vectors.f16.npy"
    tmp_faiss_path = tmp_root / "faiss_hnsw_ip.index"
    tmp_metadata_path = tmp_root / "metadata.parquet"
    tmp_failed_rows_path = tmp_root / "failed_rows.jsonl"
    tmp_shared_manifest_path = tmp_root / "manifest.json"
    tmp_root.mkdir(parents=True, exist_ok=True)
    memmap = open_memmap(tmp_vectors_path, mode="w+", dtype=np.float16, shape=(len(metadata_rows), dim))
    success_rows: List[Dict[str, Any]] = []
    success_row_ids: List[int] = []
    success_vectors: List[np.ndarray] = []
    for meta, vector in zip(metadata_rows, vectors_out):
        row_id = int(meta["row_id"])
        if vector is None:
            memmap[row_id] = np.zeros((dim,), dtype=np.float16)
            continue
        memmap[row_id] = vector.astype(np.float16)
        success_rows.append(meta)
        success_row_ids.append(row_id)
        success_vectors.append(vector)
    memmap.flush()
    del memmap
    matrix = _normalize_vectors(np.asarray(success_vectors, dtype=np.float32))
    m = max(16, min(48, dim // 8 if dim >= 128 else 16))
    base_index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    base_index.hnsw.efConstruction = max(80, min(400, len(success_rows)))
    base_index.hnsw.efSearch = max(64, min(256, max(64, len(success_rows) // 8)))
    index = faiss.IndexIDMap2(base_index)
    index.add_with_ids(matrix, np.asarray(success_row_ids, dtype=np.int64))
    table = pa.table(
        {
            "row_id": [int(row["row_id"]) for row in metadata_rows],
            "chunk_id": [str(row["chunk_id"]) for row in metadata_rows],
            "doc_id": [str(row["doc_id"]) for row in metadata_rows],
            "sample_id": [str(row["sample_id"]) for row in metadata_rows],
            "chunk_index": [int(row["chunk_index"]) for row in metadata_rows],
            "text": [str(row["text"]) for row in metadata_rows],
            "tokens_json": [str(row["tokens_json"]) for row in metadata_rows],
            "tokens_text": [" ".join(json.loads(str(row["tokens_json"]))) for row in metadata_rows],
            "has_vector": [vector is not None for vector in vectors_out],
        }
    )
    faiss.write_index(index, str(tmp_faiss_path))
    pq.write_table(table, tmp_metadata_path, compression="zstd")
    tmp_tantivy_dir.mkdir(parents=True, exist_ok=True)
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_integer_field("row_id", stored=True)
    schema_builder.add_text_field("tokens_text", stored=False)
    schema = schema_builder.build()
    text_index = tantivy.Index(schema, path=str(tmp_tantivy_dir))
    writer = text_index.writer()
    for meta, vector in zip(metadata_rows, vectors_out):
        if vector is None:
            continue
        writer.add_document(
            tantivy.Document(
                row_id=int(meta["row_id"]),
                tokens_text=[" ".join(json.loads(str(meta["tokens_json"])))],
            )
        )
    writer.commit()
    text_index.reload()

    if failed_rows:
        with tmp_failed_rows_path.open("w", encoding="utf-8") as handle:
            for row in failed_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    shared_manifest = {
        "state": "completed_with_failures" if failed_rows else "completed",
        "asset_dir": str(asset_dir),
        "shared_cache_dir": str(paths.shared_cache_dir),
        "embedding_model": client.model,
        "chunk_count": len(metadata_rows),
        "success_chunk_count": len(success_rows),
        "failed_chunk_count": len(failed_rows),
        "dimension": dim,
        "faiss_path": str(paths.faiss_path),
        "vectors_path": str(paths.vectors_path),
        "metadata_path": str(paths.metadata_path),
        "tantivy_dir": str(paths.tantivy_dir),
        "chunks_path": str(chunks_path),
        "failed_rows_path": str(paths.failed_rows_path),
        "retrieval_mode": FAST_BACKEND_RETRIEVAL_MODE,
    }
    _write_json(tmp_shared_manifest_path, shared_manifest)
    paths.shared_cache_dir.mkdir(parents=True, exist_ok=True)
    _replace_file_atomic(tmp_vectors_path, paths.vectors_path)
    _replace_file_atomic(tmp_faiss_path, paths.faiss_path)
    _replace_file_atomic(tmp_metadata_path, paths.metadata_path)
    _replace_dir_atomic(tmp_tantivy_dir, paths.tantivy_dir)
    if failed_rows:
        _replace_file_atomic(tmp_failed_rows_path, paths.failed_rows_path)
    elif paths.failed_rows_path.exists():
        paths.failed_rows_path.unlink()
    _replace_file_atomic(tmp_shared_manifest_path, paths.shared_manifest_path)
    shutil.rmtree(tmp_root, ignore_errors=True)
    _write_method_manifest(
        paths,
        asset_dir=asset_dir,
        embedding_model=client.model,
        success_chunk_count=len(success_rows),
        failed_chunk_count=len(failed_rows),
        state="completed_with_failures" if failed_rows else "completed",
    )
    return {
        "status": "completed_with_failures" if failed_rows else "completed",
        "asset_dir": str(asset_dir),
        "shared_cache_dir": str(paths.shared_cache_dir),
        "chunk_count": len(metadata_rows),
        "success_chunk_count": len(success_rows),
        "failed_chunk_count": len(failed_rows),
    }

class FastHybridRetriever:
    def __init__(self, asset_dir: str | Path, *, candidate_limit: int = 2048) -> None:
        _require_fast_backend_dependencies()
        self.asset_dir = _repo_path(asset_dir)
        self.candidate_limit = max(32, int(candidate_limit))
        self.paths = _resolve_fast_index_paths(self.asset_dir)
        if not self.paths.manifest_path.exists() or not self.paths.shared_manifest_path.exists():
            raise FileNotFoundError(f"Fast index manifest missing under {self.asset_dir}")
        self.method_manifest = _read_json(self.paths.manifest_path)
        self.shared_manifest = _read_json(self.paths.shared_manifest_path)
        self.client = EmbeddingClient(model=str(self.shared_manifest.get("embedding_model") or "").strip() or None)
        self.index = faiss.read_index(str(self.paths.faiss_path))
        hnsw_index = faiss.downcast_index(self.index.index) if hasattr(self.index, "index") else self.index
        if hasattr(hnsw_index, "hnsw"):
            hnsw_index.hnsw.efSearch = max(hnsw_index.hnsw.efSearch, min(max(64, self.candidate_limit), 512))
        self.vectors = np.load(self.paths.vectors_path, mmap_mode="r")
        self.table = pq.read_table(self.paths.metadata_path)
        self.row_id_col = self.table.column("row_id").to_pylist()
        self.chunk_id_col = self.table.column("chunk_id").to_pylist()
        self.doc_id_col = self.table.column("doc_id").to_pylist()
        self.sample_id_col = self.table.column("sample_id").to_pylist()
        self.chunk_index_col = self.table.column("chunk_index").to_pylist()
        self.text_col = self.table.column("text").to_pylist()
        self.has_vector_col = self.table.column("has_vector").to_pylist()
        self.text_index = tantivy.Index.open(path=str(self.paths.tantivy_dir))
        self.text_searcher = self.text_index.searcher()
        self.last_stats: Dict[str, Any] = {}

    def close(self) -> None:
        return None

    def retrieve(self, query: str, *, top_k: int = 6, candidate_limit: int | None = None) -> List[Dict[str, Any]]:
        search_limit = max(int(top_k), int(candidate_limit or self.candidate_limit))
        started = time.perf_counter()
        query_embed_started = time.perf_counter()
        query_vector = self.client.embed_query(query).astype(np.float32, copy=False)
        query_embed_seconds = time.perf_counter() - query_embed_started
        recall_started = time.perf_counter()
        candidate_ids = self._sparse_candidate_row_ids(query, search_limit)
        sparse_recall_seconds = time.perf_counter() - recall_started
        rerank_started = time.perf_counter()
        if candidate_ids:
            selected_ids, score_map = self._rerank_candidates(candidate_ids, query_vector, top_k=max(1, int(top_k)))
        else:
            scores, ids = self.index.search(np.asarray([_normalize_vector(query_vector)], dtype=np.float32), search_limit)
            selected_ids = [int(idx) for idx in ids[0].tolist() if int(idx) >= 0][: max(1, int(top_k))]
            score_map = {int(idx): float(score) for idx, score in zip(ids[0].tolist(), scores[0].tolist()) if int(idx) >= 0}
        rerank_seconds = time.perf_counter() - rerank_started
        row_load_started = time.perf_counter()
        rows = self._load_rows(selected_ids)
        row_load_seconds = time.perf_counter() - row_load_started
        for row in rows:
            row["score"] = score_map.get(int(row.get("_faiss_row_id", -1)), 0.0)
        self.last_stats = {
            "candidate_recall_seconds": round(float(sparse_recall_seconds), 6),
            "rerank_seconds": round(float(rerank_seconds), 6),
            "row_load_seconds": round(float(row_load_seconds), 6),
            "query_embed_seconds": round(float(query_embed_seconds), 6),
            "end_to_end_retrieval_seconds": round(float(time.perf_counter() - started), 6),
            "candidate_count": len(candidate_ids) if candidate_ids else len(selected_ids),
            "selected_count": len(rows),
            "top_k": int(top_k),
        }
        return rows

    def _sparse_candidate_row_ids(self, query: str, limit: int) -> List[int]:
        normalized_tokens = _tokenize(query)
        query_text = " ".join(normalized_tokens) if normalized_tokens else str(query or "")
        parsed_query = self.text_index.parse_query(query_text, ["tokens_text"])
        result = self.text_searcher.search(parsed_query, int(limit))
        row_ids: List[int] = []
        for _score, doc_address in result.hits:
            doc = self.text_searcher.doc(doc_address)
            try:
                value = doc.get_first("row_id")
                if value is None:
                    continue
                row_ids.append(int(value))
            except Exception:
                continue
        return row_ids

    def _rerank_candidates(self, candidate_ids: Sequence[int], query_vector: np.ndarray, *, top_k: int) -> tuple[List[int], Dict[int, float]]:
        valid_ids = [int(row_id) for row_id in candidate_ids if 0 <= int(row_id) < len(self.has_vector_col) and bool(self.has_vector_col[int(row_id)])]
        if not valid_ids:
            return [], {}
        matrix = np.asarray(self.vectors[np.asarray(valid_ids, dtype=np.int64)], dtype=np.float32)
        matrix = _normalize_vectors(matrix)
        query_vector = _normalize_vector(np.asarray(query_vector, dtype=np.float32))
        scores = matrix @ query_vector
        k = min(max(1, int(top_k)), len(valid_ids))
        if len(valid_ids) > k:
            idx = np.argpartition(scores, -k)[-k:]
            idx = idx[np.argsort(scores[idx])[::-1]]
        else:
            idx = np.argsort(scores)[::-1]
        selected_ids = [valid_ids[int(i)] for i in idx[:k]]
        score_map = {valid_ids[int(i)]: float(scores[int(i)]) for i in idx[:k]}
        return selected_ids, score_map

    def _load_rows(self, row_ids: Sequence[int]) -> List[Dict[str, Any]]:
        if not row_ids:
            return []
        out: List[Dict[str, Any]] = []
        for row_id in row_ids:
            if row_id < 0 or row_id >= len(self.row_id_col):
                continue
            out.append(
                {
                    "_faiss_row_id": int(self.row_id_col[row_id]),
                    "chunk_id": str(self.chunk_id_col[row_id] or ""),
                    "doc_id": str(self.doc_id_col[row_id] or ""),
                    "sample_id": str(self.sample_id_col[row_id] or ""),
                    "chunk_index": int(self.chunk_index_col[row_id] or 0),
                    "text": str(self.text_col[row_id] or ""),
                }
            )
        return out
