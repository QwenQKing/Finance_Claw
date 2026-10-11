from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import hashlib
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib import error, request

import numpy as np
from numpy.lib.format import open_memmap

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./-]{1,}|[\u4e00-\u9fff]{2,}")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_QUERY_EMBED_CACHE: dict[tuple[str, str], np.ndarray] = {}
_QUERY_EMBED_CACHE_LOCK = threading.Lock()
_EMBED_TIMEOUT_SECONDS = max(30, int(os.getenv("EMBED_TIMEOUT_SECONDS", "180") or "180"))
_EMBED_RETRY_ATTEMPTS = max(1, int(os.getenv("EMBED_RETRY_ATTEMPTS", "4") or "4"))
_EMBED_RETRY_BACKOFF_SECONDS = max(0.0, float(os.getenv("EMBED_RETRY_BACKOFF_SECONDS", "5") or "5"))

def _repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (_REPO_ROOT / path)

def load_dotenv() -> None:
    candidates = [_REPO_ROOT, Path.cwd().resolve(), *Path.cwd().resolve().parents, _REPO_ROOT]
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
            if not line or line.startswith("#") or "=" not in line or line.startswith("export "):
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return

def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "") if len(token) >= 2]

def _normalize_vectors(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms

def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _iter_jsonl_with_offsets(path: Path) -> Iterable[tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            yield offset, json.loads(stripped)

def _resolve_embed_config() -> tuple[str, str, str]:
    load_dotenv()
    api_key = (
        os.getenv("EMBEDDING_API_KEY", "").strip()
        or os.getenv("KG_EMBED_API_KEY", "").strip()
    )
    base_url = (
        os.getenv("EMBEDDING_BASE_URL", "").rstrip("/").strip()
        or os.getenv("KG_EMBED_BASE_URL", "").rstrip("/").strip()
    )
    model = (
        os.getenv("EMBEDDING_MODEL", "").strip()
        or os.getenv("KG_EMBED_MODEL", "").strip()
    )
    missing = [
        name
        for name, value in [
            ("EMBEDDING_API_KEY|KG_EMBED_API_KEY", api_key),
            ("EMBEDDING_BASE_URL|KG_EMBED_BASE_URL", base_url),
            ("EMBEDDING_MODEL|KG_EMBED_MODEL", model),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required embedding env vars: {', '.join(missing)}")
    return api_key, base_url, model

def _normalize_openai_base_url(raw: str) -> str:
    base = (raw or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/v1/chat/completions", "/embeddings", "/v1/embeddings"):
        if base.endswith(suffix):
            return base[: -len(suffix)].rstrip("/")
    return base

class EmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 180,
    ) -> None:
        resolved_api_key, resolved_base_url, resolved_model = _resolve_embed_config()
        self.api_key = (api_key or resolved_api_key).strip()
        self.base_url = _normalize_openai_base_url(base_url or resolved_base_url)
        self.model = (model or resolved_model).strip()
        self.timeout_seconds = max(30, int(timeout_seconds or _EMBED_TIMEOUT_SECONDS))
        self.retry_attempts = _EMBED_RETRY_ATTEMPTS
        self.retry_backoff_seconds = _EMBED_RETRY_BACKOFF_SECONDS

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        resolved: List[Optional[np.ndarray]] = [None] * len(texts)
        missing_positions: dict[str, List[int]] = {}
        with _QUERY_EMBED_CACHE_LOCK:
            for idx, text in enumerate(texts):
                cache_key = (self.model, text)
                cached = _QUERY_EMBED_CACHE.get(cache_key)
                if cached is not None:
                    resolved[idx] = cached
                    continue
                missing_positions.setdefault(text, []).append(idx)
        if not missing_positions:
            return np.asarray(resolved, dtype=np.float32)
        missing_texts = list(missing_positions.keys())
        payload = {"model": self.model, "input": missing_texts}
        req = request.Request(
            url=f"{self.base_url}/embeddings",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_exc: Exception | None = None
        body = ""
        for attempt in range(1, self.retry_attempts + 1):
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    body = resp.read().decode("utf-8")
                break
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else str(exc)
                last_exc = RuntimeError(f"Embedding HTTPError: {exc.code} {detail}")
            except Exception as exc:
                last_exc = RuntimeError(f"Embedding request failed: {exc}")
            if attempt < self.retry_attempts and self.retry_backoff_seconds > 0:
                time.sleep(self.retry_backoff_seconds * attempt)
        else:
            assert last_exc is not None
            raise last_exc
        data = json.loads(body)
        rows = data.get("data") or []
        if not rows:
            raise RuntimeError("Embedding response has no data rows")
        ordered = sorted(rows, key=lambda item: int(item.get("index", 0)))
        matrix = np.asarray([item.get("embedding") or [] for item in ordered], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(missing_texts):
            raise RuntimeError("Embedding response shape mismatch")
        normalized = _normalize_vectors(matrix)
        with _QUERY_EMBED_CACHE_LOCK:
            for text, vector in zip(missing_texts, normalized):
                cache_key = (self.model, text)
                _QUERY_EMBED_CACHE[cache_key] = vector
                for idx in missing_positions[text]:
                    resolved[idx] = vector
        return np.asarray(resolved, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        cache_key = (self.model, text)
        with _QUERY_EMBED_CACHE_LOCK:
            cached = _QUERY_EMBED_CACHE.get(cache_key)
        if cached is not None:
            return cached
        matrix = self.embed_texts([text])
        vector = _normalize_vector(matrix[0].astype(np.float32, copy=False))
        with _QUERY_EMBED_CACHE_LOCK:
            _QUERY_EMBED_CACHE[cache_key] = vector
        return vector

@dataclass
class DenseIndexPaths:
    asset_dir: Path
    manifest_path: Path
    shared_manifest_path: Path
    shared_cache_dir: Path
    chunks_path: Path
    sqlite_path: Path
    vectors_path: Path
    failed_rows_path: Path

def _build_corpus_key(asset_dir: Path) -> str:
    build_manifest_path = asset_dir / "build_manifest.json"
    if build_manifest_path.exists():
        build_manifest = _read_json(build_manifest_path)
        shared_asset_dir = str(build_manifest.get("shared_asset_dir") or "").strip()
        if shared_asset_dir:
            shared_build_manifest_path = _repo_path(shared_asset_dir) / "build_manifest.json"
            if shared_build_manifest_path.exists():
                build_manifest = _read_json(shared_build_manifest_path)
        payload = {
            "dataset": build_manifest.get("dataset"),
            "train_path": build_manifest.get("train_path"),
            "chunk_count": build_manifest.get("chunk_count"),
            "max_samples": build_manifest.get("max_samples"),
            "requested_max_samples": build_manifest.get("requested_max_samples"),
            "selected_count": len(build_manifest.get("selected_sample_ids") or []),
            "selected_sample_ids": build_manifest.get("selected_sample_ids") or [],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    chunks_path = asset_dir / "corpus" / "chunks.jsonl"
    stat = chunks_path.stat()
    raw = f"{asset_dir.name}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

def _resolve_dense_index_paths(asset_dir: Path, shared_cache_root: Path | None = None) -> DenseIndexPaths:
    asset_dir = _repo_path(asset_dir)
    if shared_cache_root is None:
        shared_cache_root = asset_dir.parents[1] / "_dense_cache"
    corpus_key = _build_corpus_key(asset_dir)
    shared_cache_dir = _repo_path(shared_cache_root) / corpus_key
    return DenseIndexPaths(
        asset_dir=asset_dir,
        manifest_path=asset_dir / "dense_index" / "manifest.json",
        shared_manifest_path=shared_cache_dir / "manifest.json",
        shared_cache_dir=shared_cache_dir,
        chunks_path=asset_dir / "corpus" / "chunks.jsonl",
        sqlite_path=shared_cache_dir / "hybrid_index.sqlite3",
        vectors_path=shared_cache_dir / "vectors.f16.npy",
        failed_rows_path=shared_cache_dir / "failed_rows.jsonl",
    )

def resolve_dense_shared_cache_dir(asset_dir: str | Path, shared_cache_root: str | Path | None = None) -> Path:
    return _resolve_dense_index_paths(
        _repo_path(asset_dir),
        _repo_path(shared_cache_root) if shared_cache_root else None,
    ).shared_cache_dir

def _init_sqlite(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_rows (
            row_id INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            doc_id TEXT,
            sample_id TEXT,
            file_offset INTEGER NOT NULL,
            has_vector INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS postings (
            token TEXT NOT NULL,
            row_id INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_postings_token ON postings(token)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_postings_token_row ON postings(token, row_id)")
    return conn

def _write_method_manifest(
    paths: DenseIndexPaths,
    *,
    asset_dir: Path,
    embedding_model: str,
    retrieval_mode: str,
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
        "retrieval_mode": retrieval_mode,
    }
    _write_json(paths.manifest_path, method_manifest)

def build_dense_index_for_asset(
    asset_dir: str | Path,
    *,
    shared_cache_root: str | Path | None = None,
    batch_size: int = 32,
    embed_workers: int = 1,
    force: bool = False,
    attach_only: bool = False,
    progress_cb=None,
) -> Dict[str, Any]:
    asset_dir = _repo_path(asset_dir)
    paths = _resolve_dense_index_paths(asset_dir, _repo_path(shared_cache_root) if shared_cache_root else None)
    client = EmbeddingClient()
    shared_manifest = _read_json(paths.shared_manifest_path) if paths.shared_manifest_path.exists() else {}
    shared_state = str(shared_manifest.get("state") or "").strip()
    shared_success_chunk_count = int(shared_manifest.get("success_chunk_count", 0) or 0)
    shared_failed_chunk_count = int(shared_manifest.get("failed_chunk_count", 0) or 0)
    shared_is_usable = shared_state in {"completed", "completed_with_failures"}
    shared_is_fully_complete = shared_state == "completed" and shared_failed_chunk_count == 0
    if attach_only and shared_is_usable and paths.sqlite_path.exists():
        _write_method_manifest(
            paths,
            asset_dir=asset_dir,
            embedding_model=str(shared_manifest.get("embedding_model") or client.model),
            retrieval_mode=str(shared_manifest.get("retrieval_mode") or "dense_hybrid_sqlite_cosine"),
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
    if shared_is_fully_complete and not force:
        _write_method_manifest(
            paths,
            asset_dir=asset_dir,
            embedding_model=str(shared_manifest.get("embedding_model") or client.model),
            retrieval_mode=str(shared_manifest.get("retrieval_mode") or "dense_hybrid_sqlite_cosine"),
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
    current_shared_state = {
        "state": "running",
        "asset_dir": str(asset_dir),
        "embedding_model": client.model,
        "chunks_path": str(chunks_path),
        "expected_chunk_count": expected_chunk_count,
    }
    _write_json(paths.shared_manifest_path, current_shared_state)

    resume_mode = (
        not force
        and paths.sqlite_path.exists()
        and shared_state in {"running", "completed_with_failures"}
    )
    if force:
        if paths.sqlite_path.exists():
            paths.sqlite_path.unlink()
        if paths.vectors_path.exists():
            paths.vectors_path.unlink()
        if paths.failed_rows_path.exists():
            paths.failed_rows_path.unlink()
    conn = _init_sqlite(paths.sqlite_path)
    rows_payload: List[List[Any]] = []
    postings_payload: List[tuple[str, int]] = []
    texts_batch: List[str] = []
    row_ids_batch: List[int] = []
    row_meta_batch: List[Dict[str, Any]] = []
    vectors_memmap = None
    dim: Optional[int] = None
    row_count = 0
    success_chunk_count = 0
    failed_rows: List[Dict[str, Any]] = []
    pending_row_indexes: Dict[int, int] = {}
    existing_rows: Dict[int, tuple[str, int]] = {}
    if resume_mode:
        for row_id, chunk_id, has_vector in conn.execute("SELECT row_id, chunk_id, has_vector FROM chunk_rows"):
            existing_rows[int(row_id)] = (str(chunk_id), int(has_vector))
        success_chunk_count = int(
            conn.execute("SELECT COUNT(*) FROM chunk_rows WHERE has_vector = 1").fetchone()[0]
        )
        if paths.vectors_path.exists():
            vectors_memmap = open_memmap(paths.vectors_path, mode="r+")
            if len(vectors_memmap.shape) == 2:
                dim = int(vectors_memmap.shape[1])

    effective_embed_workers = max(1, int(embed_workers))

    def _flush_pending_rows() -> None:
        nonlocal rows_payload, postings_payload, pending_row_indexes
        if not rows_payload:
            return
        conn.executemany(
            "INSERT OR REPLACE INTO chunk_rows(row_id, chunk_id, doc_id, sample_id, file_offset, has_vector) VALUES (?, ?, ?, ?, ?, ?)",
            [tuple(row) for row in rows_payload],
        )
        if postings_payload:
            conn.executemany("INSERT OR IGNORE INTO postings(token, row_id) VALUES (?, ?)", postings_payload)
        conn.commit()
        rows_payload = []
        postings_payload = []
        pending_row_indexes = {}

    def _ensure_memmap(vector_dim: int) -> Any:
        nonlocal vectors_memmap, dim
        if vectors_memmap is not None:
            return vectors_memmap
        dim = int(vector_dim)
        total_rows = expected_chunk_count if expected_chunk_count > 0 else max(row_count, len(texts_batch), 1)
        vectors_memmap = open_memmap(paths.vectors_path, mode="w+", dtype=np.float16, shape=(total_rows, dim))
        return vectors_memmap

    def _mark_success(row_id: int, vector: np.ndarray) -> None:
        nonlocal success_chunk_count
        memmap = _ensure_memmap(int(vector.shape[0]))
        memmap[int(row_id)] = vector.astype(np.float16)
        idx = pending_row_indexes.get(int(row_id))
        if idx is not None:
            rows_payload[idx][5] = 1
        else:
            conn.execute("UPDATE chunk_rows SET has_vector = 1 WHERE row_id = ?", [int(row_id)])
            conn.commit()
        success_chunk_count += 1
        if progress_cb:
            progress_cb(1)

    def _record_failure(row_id: int, chunk_id: str, batch_exc: Exception, single_exc: Exception) -> None:
        failed_rows.append(
            {
                "row_id": int(row_id),
                "chunk_id": chunk_id,
                "error": str(single_exc),
                "batch_error": str(batch_exc),
            }
        )

    def _embed_batch_resilient(batch_texts: Sequence[str], batch_row_ids: Sequence[int], batch_meta: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not batch_texts:
            return []
        try:
            vectors = client.embed_texts(batch_texts)
            out: List[Dict[str, Any]] = []
            for idx, row_id_in_batch in enumerate(batch_row_ids):
                out.append(
                    {
                        "row_id": int(row_id_in_batch),
                        "chunk_id": str(batch_meta[idx].get("chunk_id") or ""),
                        "vector": np.asarray(vectors[idx], dtype=np.float32),
                    }
                )
            return out
        except Exception:
            raise

    executor: ThreadPoolExecutor | None = None
    inflight: Dict[Any, tuple[List[int], List[Dict[str, Any]]]] = {}

    def _consume_embed_results(results: Sequence[Dict[str, Any]]) -> None:
        for item in results:
            row_id_in_batch = int(item.get("row_id", -1))
            if "vector" in item:
                _mark_success(row_id_in_batch, np.asarray(item["vector"], dtype=np.float32))
            else:
                _record_failure(
                    row_id_in_batch,
                    str(item.get("chunk_id") or ""),
                    RuntimeError(str(item.get("batch_error") or "batch_error")),
                    RuntimeError(str(item.get("error") or "embedding_error")),
                )

    def _wait_for_one() -> None:
        if not inflight:
            return
        done, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
        for future in done:
            batch_row_ids, _batch_meta = inflight.pop(future)
            _consume_embed_results(future.result())
            if row_count > 0 and len(rows_payload) >= 4096 and not inflight:
                _flush_pending_rows()

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
        for offset, row in _iter_jsonl_with_offsets(chunks_path):
            chunk_id = str(row.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            row_id = row_count
            row_count += 1
            existing = existing_rows.get(row_id)
            if existing is not None:
                existing_chunk_id, existing_has_vector = existing
                if existing_chunk_id != chunk_id:
                    raise RuntimeError(
                        f"Resume state mismatch at row_id={row_id}: existing chunk_id={existing_chunk_id} current chunk_id={chunk_id}; rerun with --force"
                    )
                if existing_has_vector == 1:
                    continue
            rows_payload.append(
                [
                    row_id,
                    chunk_id,
                    str(row.get("doc_id") or "").strip(),
                    str(row.get("sample_id") or "").strip(),
                    int(offset),
                    0,
                ]
            )
            pending_row_indexes[row_id] = len(rows_payload) - 1
            seen_tokens: set[str] = set()
            tokens = [str(token).lower() for token in (row.get("tokens") or []) if str(token).strip()]
            if not tokens:
                tokens = _tokenize(text)
            for token in tokens:
                if token in seen_tokens:
                    continue
                seen_tokens.add(token)
                postings_payload.append((token, row_id))
            texts_batch.append(text)
            row_ids_batch.append(row_id)
            row_meta_batch.append({"chunk_id": chunk_id})
            if len(texts_batch) >= max(1, batch_size):
                _process_batch()
            if len(rows_payload) >= 4096 and not texts_batch:
                if not inflight:
                    _flush_pending_rows()
        if texts_batch:
            _process_batch()
        while inflight:
            _wait_for_one()
        _flush_pending_rows()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        conn.close()

    if failed_rows:
        paths.failed_rows_path.parent.mkdir(parents=True, exist_ok=True)
        with paths.failed_rows_path.open("w", encoding="utf-8") as handle:
            for row in failed_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif paths.failed_rows_path.exists():
        paths.failed_rows_path.unlink()

    shared_manifest = {
        "state": "completed_with_failures" if failed_rows else "completed",
        "asset_dir": str(asset_dir),
        "shared_cache_dir": str(paths.shared_cache_dir),
        "embedding_model": client.model,
        "chunk_count": row_count,
        "success_chunk_count": success_chunk_count,
        "failed_chunk_count": len(failed_rows),
        "dimension": dim or 0,
        "sqlite_path": str(paths.sqlite_path),
        "vectors_path": str(paths.vectors_path),
        "chunks_path": str(chunks_path),
        "failed_rows_path": str(paths.failed_rows_path),
        "retrieval_mode": "dense_hybrid_sqlite_cosine",
    }
    _write_json(paths.shared_manifest_path, shared_manifest)
    _write_method_manifest(
        paths,
        asset_dir=asset_dir,
        embedding_model=client.model,
        retrieval_mode="dense_hybrid_sqlite_cosine",
        success_chunk_count=success_chunk_count,
        failed_chunk_count=len(failed_rows),
        state="completed_with_failures" if failed_rows else "completed",
    )
    return {
        "status": "completed_with_failures" if failed_rows else "completed",
        "asset_dir": str(asset_dir),
        "shared_cache_dir": str(paths.shared_cache_dir),
        "chunk_count": row_count,
        "success_chunk_count": success_chunk_count,
        "failed_chunk_count": len(failed_rows),
    }

class DenseHybridRetriever:
    def __init__(self, asset_dir: str | Path, *, candidate_limit: int = 2048) -> None:
        self.asset_dir = _repo_path(asset_dir)
        self.candidate_limit = max(32, int(candidate_limit))
        self.paths = _resolve_dense_index_paths(self.asset_dir)
        if not self.paths.manifest_path.exists() or not self.paths.shared_manifest_path.exists():
            raise FileNotFoundError(f"Dense index manifest missing under {self.asset_dir}")
        self.method_manifest = _read_json(self.paths.manifest_path)
        self.shared_manifest = _read_json(self.paths.shared_manifest_path)
        self.client = EmbeddingClient(model=str(self.shared_manifest.get("embedding_model") or "").strip() or None)
        self._conn = sqlite3.connect(self.paths.sqlite_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._vectors = np.load(self.paths.vectors_path, mmap_mode="r")
        self._chunks_path = _repo_path(self.shared_manifest.get("chunks_path") or self.paths.chunks_path)
        self.last_stats: Dict[str, Any] = {}

    def close(self) -> None:
        self._conn.close()

    def _candidate_row_ids(self, query: str, limit: int) -> List[int]:
        tokens = list(dict.fromkeys(_tokenize(query)))
        if not tokens:
            return []
        placeholders = ",".join("?" for _ in tokens)
        sql = (
            f"SELECT row_id, COUNT(*) AS hits FROM postings "
            f"WHERE token IN ({placeholders}) AND row_id IN (SELECT row_id FROM chunk_rows WHERE has_vector = 1) "
            f"GROUP BY row_id ORDER BY hits DESC, row_id ASC LIMIT ?"
        )
        rows = self._conn.execute(sql, [*tokens, int(limit)]).fetchall()
        return [int(row_id) for row_id, _ in rows]

    def _load_rows_by_offsets(self, row_ids: Sequence[int]) -> List[Dict[str, Any]]:
        if not row_ids:
            return []
        mapping = self._row_mappings(row_ids)
        return self._load_rows_from_mapping(row_ids, mapping)

    def _row_mappings(self, row_ids: Sequence[int]) -> Dict[int, tuple[str, int]]:
        if not row_ids:
            return {}
        placeholders = ",".join("?" for _ in row_ids)
        rows = self._conn.execute(
            f"SELECT row_id, chunk_id, file_offset FROM chunk_rows WHERE row_id IN ({placeholders})",
            [int(row_id) for row_id in row_ids],
        ).fetchall()
        return {int(row_id): (str(chunk_id), int(file_offset)) for row_id, chunk_id, file_offset in rows}

    def _load_rows_from_mapping(self, row_ids: Sequence[int], mapping: Dict[int, tuple[str, int]]) -> List[Dict[str, Any]]:
        if not row_ids or not mapping:
            return []
        rows: List[Dict[str, Any]] = []
        with self._chunks_path.open("r", encoding="utf-8-sig") as handle:
            for row_id in row_ids:
                item = mapping.get(int(row_id))
                if item is None:
                    continue
                _chunk_id, file_offset = item
                handle.seek(file_offset)
                line = handle.readline()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def retrieve(self, query: str, *, top_k: int = 6, candidate_limit: int | None = None) -> List[Dict[str, Any]]:
        started = time.perf_counter()
        query_embed_started = time.perf_counter()
        query_vector = self.client.embed_query(query).astype(np.float32, copy=False)
        query_embed_seconds = time.perf_counter() - query_embed_started
        rows = self.retrieve_with_vector(query, query_vector, top_k=top_k, candidate_limit=candidate_limit)
        self.last_stats = {
            **dict(self.last_stats or {}),
            "query_embed_seconds": round(float(query_embed_seconds), 6),
            "end_to_end_retrieval_seconds": round(float(time.perf_counter() - started), 6),
        }
        return rows

    def retrieve_with_vector(
        self,
        query: str,
        query_vector: np.ndarray,
        *,
        top_k: int = 6,
        candidate_limit: int | None = None,
    ) -> List[Dict[str, Any]]:
        started = time.perf_counter()
        recall_started = time.perf_counter()
        candidate_ids = self._candidate_row_ids(query, candidate_limit or self.candidate_limit)
        recall_seconds = time.perf_counter() - recall_started
        if not candidate_ids:
            self.last_stats = {
                "candidate_recall_seconds": round(float(recall_seconds), 6),
                "rerank_seconds": 0.0,
                "row_load_seconds": 0.0,
                "query_embed_seconds": 0.0,
                "end_to_end_retrieval_seconds": round(float(time.perf_counter() - started), 6),
                "candidate_count": 0,
                "selected_count": 0,
                "top_k": int(top_k),
            }
            return []
        query_vector = np.asarray(query_vector, dtype=np.float32)
        query_vector = _normalize_vector(query_vector)
        rerank_started = time.perf_counter()
        candidate_matrix = np.asarray(self._vectors[np.asarray(candidate_ids, dtype=np.int64)], dtype=np.float32)
        scores = candidate_matrix @ query_vector
        k = min(max(1, int(top_k)), len(candidate_ids))
        if len(candidate_ids) > k:
            idx = np.argpartition(scores, -k)[-k:]
            idx = idx[np.argsort(scores[idx])[::-1]]
        else:
            idx = np.argsort(scores)[::-1]
        selected_row_ids = [candidate_ids[int(i)] for i in idx]
        rerank_seconds = time.perf_counter() - rerank_started
        row_load_started = time.perf_counter()
        mapping = self._row_mappings(selected_row_ids)
        rows = self._load_rows_from_mapping(selected_row_ids, mapping)
        row_map = {str(row.get("chunk_id") or ""): row for row in rows}
        ordered_rows: List[Dict[str, Any]] = []
        for row_id in selected_row_ids:
            item = mapping.get(int(row_id))
            if not item:
                continue
            chunk_id = str(item[0])
            row = row_map.get(chunk_id)
            if row is not None:
                ordered_rows.append(row)
        row_load_seconds = time.perf_counter() - row_load_started
        self.last_stats = {
            "candidate_recall_seconds": round(float(recall_seconds), 6),
            "rerank_seconds": round(float(rerank_seconds), 6),
            "row_load_seconds": round(float(row_load_seconds), 6),
            "query_embed_seconds": 0.0,
            "end_to_end_retrieval_seconds": round(float(time.perf_counter() - started), 6),
            "candidate_count": int(len(candidate_ids)),
            "selected_count": int(len(ordered_rows)),
            "top_k": int(k),
        }
        return ordered_rows
