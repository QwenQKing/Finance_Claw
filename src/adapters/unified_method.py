from __future__ import annotations

import datetime as dt
import heapq
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evolution import EvolutionOrchestrator, PerformanceTracker, SessionMetrics
from src.governance import (
    ComplianceGate,
    GovernanceLog,
    JurisdictionRegistry,
)
from src.temporal import (
    EvidenceChainExplorer,
    RuleMiner,
)
from src.temporal.time_model import TemporalConstraint
from src.methods import UnifiedMethodPipeline

from ..retrieval import DenseHybridRetriever
from .real_fr_ci import RealFRCIAdapter

_DATE_PATTERNS = [
    re.compile(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b"),
    re.compile(r"\b(20\d{2})年(\d{1,2})月(\d{1,2})日"),
]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DENSE_RETRIEVER_LOCAL = threading.local()
_PRESEEDED_GRAPH_CACHE: Dict[str, Dict[str, Any]] = {}
_PRESEEDED_GRAPH_LOCK = threading.Lock()
_RETRIEVED_EVIDENCE_CHUNK_MAX_CHARS = int(os.getenv("KB_CHUNK_MAX_CHARS", "800") or "800")
_RETRIEVED_EVIDENCE_TOTAL_MAX_CHARS = int(os.getenv("KB_TOTAL_MAX_CHARS", "3000") or "3000")

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
    *,
    score_fn,
    query: str,
    top_k: int,
    dataset_chunk_prefix: str = "",
    dataset_bonus: float = 0.0,
) -> List[Dict[str, Any]]:
    heap: List[tuple[float, int, Dict[str, Any]]] = []
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
            score = float(score_fn(query, text))
            chunk_id = str(row.get("chunk_id") or "").strip()
            if dataset_chunk_prefix and chunk_id.startswith(dataset_chunk_prefix):
                score += dataset_bonus
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

def _build_retrieval_query(sample: Dict[str, Any]) -> str:
    parts: List[str] = []
    question = str(sample.get("question") or "").strip()
    instruction = str(sample.get("instruction") or "").strip()
    original_text = str(sample.get("_original_text_content") or "").strip()
    if not original_text:
        for modality in sample.get("modalities") or []:
            if str(modality.get("modality") or "").lower() != "text":
                continue
            data = modality.get("data") or {}
            original_text = str(data.get("content") or "").strip()
            if original_text:
                break
    if question:
        parts.append(question)
    if instruction:
        parts.append(instruction)
    if original_text:
        parts.append(original_text[:1200])
    return "\n".join(parts).strip()

class UnifiedPaperMethodAdapter(RealFRCIAdapter):

    name = "financeclaw"

    def __init__(
        self,
        name: str = "financeclaw",
        instance_tag: str | None = None,
        *,
        enable_fr: bool = True,
        enable_ci: bool = True,
        enable_rc: bool = True,
        fr_variant: str = "default",
        ci_variant: str = "default",
        rc_variant: str = "default",
        allow_pseudo_rules: bool = False,
    ) -> None:
        super().__init__(instance_tag=instance_tag)
        self.name = name
        artifact_root = self._artifact_dir("experiments/artifacts/unified_paper_method")
        artifact_root.mkdir(parents=True, exist_ok=True)
        self._rule_miner = RuleMiner()
        self._explorer = EvidenceChainExplorer()
        self._constraint = TemporalConstraint(max_release_lag_days=45)
        self._governance_log = GovernanceLog(artifact_root / "governance")
        self._compliance_gate = ComplianceGate()
        self._jurisdiction_registry = JurisdictionRegistry()
        self._tracker = PerformanceTracker(log_dir=artifact_root / "evolution")
        self._evolution = EvolutionOrchestrator(
            tracker=self._tracker,
            hypergraph=self._memory,
            evolution_log_dir=artifact_root / "evolution",
        )
        self._pipeline = UnifiedMethodPipeline(
            name=self.name,
            memory=self._memory,
            llm=self.llm,
            rule_miner=self._rule_miner,
            explorer=self._explorer,
            constraint=self._constraint,
            governance_log=self._governance_log,
            compliance_gate=self._compliance_gate,
            jurisdiction_registry=self._jurisdiction_registry,
            tracker=self._tracker,
            evolution=self._evolution,
            score_chunk_fn=self._score_chunk,
            parse_anchor_time_fn=self._parse_anchor_time,
            finalize_prediction_fn=self._finalize_prediction,
            candidate_labels_fn=self._candidate_labels,
            extract_chunks_fn=self._extract_chunks,
            enable_fr=enable_fr,
            enable_ci=enable_ci,
            enable_rc=enable_rc,
            fr_variant=fr_variant,
            ci_variant=ci_variant,
            rc_variant=rc_variant,
            allow_pseudo_rules=allow_pseudo_rules,
        )
        self._prebuilt_seeded_assets: set[str] = set()

    @staticmethod
    def _coerce_path(raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else (_REPO_ROOT / path)

    def _resolve_asset_dir(self, sample: Dict[str, Any]) -> Optional[Path]:
        explicit_sample_dir = str(sample.get("_finclaw_explicit_asset_dir") or "").strip()
        if explicit_sample_dir:
            path = self._coerce_path(explicit_sample_dir)
            if path.exists():
                return path
        explicit = os.getenv("KB_DIR", "").strip()
        if explicit:
            path = self._coerce_path(explicit)
            if path.exists():
                return path
        root = os.getenv("KB_ROOT", "").strip()
        if not root:
            return None
        base = self._coerce_path(root)
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

    def _seed_memory_from_asset(self, asset_dir: Path) -> None:
        cache_key = str(asset_dir.resolve())
        if cache_key in self._prebuilt_seeded_assets:
            return
        with _PRESEEDED_GRAPH_LOCK:
            snapshot = _PRESEEDED_GRAPH_CACHE.get(cache_key)
            if snapshot is None:
                seed_path = asset_dir / "hypergraph_seed" / "findings.jsonl"
                if not seed_path.exists():
                    return
                findings = _load_jsonl(seed_path)
                if not findings:
                    return
                self._memory.ingest_findings(findings, session_id=f"prebuilt::{asset_dir.name}")
                snapshot = self._memory.snapshot_state()
                _PRESEEDED_GRAPH_CACHE[cache_key] = snapshot
            else:
                self._memory.restore_state(snapshot)
        self._prebuilt_seeded_assets.add(cache_key)

    def _prepare_sample_with_prebuilt_asset(self, sample: Dict[str, Any], budget: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        disable_flag = sample.get("_finclaw_disable_rag_retrieval")
        if disable_flag is None:
            disable_flag = os.getenv("DISABLE_RAG_RETRIEVAL", "").strip().lower() in {"1", "true", "yes", "on"}
        if bool(disable_flag):
            return sample, {
                "prebuilt_asset_used": False,
                "prebuilt_asset_disabled": True,
                "retrieval_strategy": "disabled_no_rag",
            }
        asset_dir = self._resolve_asset_dir(sample)
        if asset_dir is None:
            return sample, {"prebuilt_asset_used": False}
        self._seed_memory_from_asset(asset_dir)
        chunks_path = asset_dir / "corpus" / "chunks.jsonl"
        if not chunks_path.exists():
            return sample, {"prebuilt_asset_used": False, "asset_dir": str(asset_dir)}
        question = _build_retrieval_query(sample)
        sample_dataset = str(sample.get("dataset") or sample.get("source_dataset") or "").strip()
        dataset_chunk_prefix = f"doc_{sample_dataset}__" if sample_dataset else ""
        top_k = max(1, int(budget.get("max_retrieval_top_k", 6)))
        retrieval_strategy = "prebuilt_asset_keyword_topk_append"
        selected: List[Dict[str, Any]] = []
        dense_stage_timings: Dict[str, float] = {}
        retrieval_started = dt.datetime.now()
        retriever = _dense_retriever_for_asset(asset_dir)
        selected = retriever.retrieve(query=question, top_k=top_k)
        if selected:
            retrieval_strategy = "prebuilt_asset_dense_hybrid_topk_append"
            dense_stage_timings = _retriever_stage_timings_ms(retriever)
        if not selected:
            raise RuntimeError(f"Dense retrieval returned no chunks for asset_dir={asset_dir}")
        retrieved_parts: List[str] = []
        for row in selected:
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            if _RETRIEVED_EVIDENCE_CHUNK_MAX_CHARS > 0 and len(text) > _RETRIEVED_EVIDENCE_CHUNK_MAX_CHARS:
                text = text[:_RETRIEVED_EVIDENCE_CHUNK_MAX_CHARS].rstrip()
            retrieved_parts.append(text)
        retrieved_text = "\n".join(retrieved_parts)
        if _RETRIEVED_EVIDENCE_TOTAL_MAX_CHARS > 0 and len(retrieved_text) > _RETRIEVED_EVIDENCE_TOTAL_MAX_CHARS:
            retrieved_text = retrieved_text[:_RETRIEVED_EVIDENCE_TOTAL_MAX_CHARS].rstrip()
        if not retrieved_text:
            return sample, {"prebuilt_asset_used": False, "asset_dir": str(asset_dir)}

        prepared = dict(sample)
        modalities = list(prepared.get("modalities") or [])
        replaced = False
        original_text = ""
        original_instruction = str(prepared.get("instruction") or "").strip()
        for modality in modalities:
            if str(modality.get("modality") or "").lower() != "text":
                continue
            data = dict(modality.get("data") or {})
            original_text = str(data.get("content") or "").strip()
            merged_parts = [part for part in [original_text] if part]
            if retrieved_text and retrieved_text != original_text:
                merged_parts.append(f"[Retrieved Evidence Pool]\n{retrieved_text}")
            data["content"] = "\n\n".join(merged_parts) if merged_parts else retrieved_text
            data["retrieved_evidence"] = retrieved_text
            modality["data"] = data
            replaced = True
            break
        if not replaced:
            modalities.insert(0, {"modality": "text", "data": {"content": retrieved_text}})
        prepared["modalities"] = modalities
        prepared["source_dataset"] = prepared.get("source_dataset") or prepared.get("dataset") or asset_dir.name
        prepared["_original_text_content"] = original_text
        prepared["_original_instruction"] = original_instruction
        prepared["_retrieved_evidence_text"] = retrieved_text
        retrieval_seconds = (dt.datetime.now() - retrieval_started).total_seconds()
        return prepared, {
            "prebuilt_asset_used": True,
            "asset_dir": str(asset_dir),
            "retrieved_chunk_ids": [str(row.get("chunk_id") or "") for row in selected],
            "retrieval_strategy": retrieval_strategy,
            "retrieval_seconds": retrieval_seconds,
            "recall_seconds": dense_stage_timings.get("rag_candidate_recall_ms", 0.0) / 1000.0,
            "rerank_seconds": dense_stage_timings.get("rag_rerank_ms", 0.0) / 1000.0,
            "row_load_seconds": dense_stage_timings.get("rag_row_load_ms", 0.0) / 1000.0,
            "query_embed_seconds": dense_stage_timings.get("rag_query_embed_ms", 0.0) / 1000.0,
            "profiling": {"stage_timings_ms": dense_stage_timings} if dense_stage_timings else {},
        }

    def _parse_anchor_time(self, sample: Dict[str, Any], text: str) -> dt.datetime:
        candidates = [
            sample.get("data_cutoff"),
            sample.get("resolved_at"),
            sample.get("created_at"),
        ]
        for modality in sample.get("modalities") or []:
            candidates.append(modality.get("timestamp"))
            data = modality.get("data") or {}
            candidates.append(data.get("timestamp"))
            candidates.append(data.get("data_timestamp"))

        for raw in candidates:
            s = str(raw or "").strip()
            if not s:
                continue
            try:
                parsed = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
                if parsed.tzinfo is not None:
                    parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
                return parsed
            except ValueError:
                continue

        for pattern in _DATE_PATTERNS:
            m = pattern.search(text)
            if not m:
                continue
            year, month, day = [int(x) for x in m.groups()]
            return dt.datetime(year, month, day)

        year_match = re.search(r"\b(20\d{2})\b", text)
        year = int(year_match.group(1)) if year_match else dt.datetime.now(dt.timezone.utc).year
        return dt.datetime(year, 1, 1)

    def run_sample(self, sample: Dict[str, Any], budget: Dict[str, Any]) -> Dict[str, Any]:
        prepared_sample, asset_info = self._prepare_sample_with_prebuilt_asset(sample, budget)
        out = self._pipeline.run_sample(prepared_sample, budget)
        extras = dict(out.get("extras") or {})
        extras.update(asset_info)
        out["extras"] = extras
        if asset_info.get("prebuilt_asset_used"):
            out["evidence_ids"] = asset_info.get("retrieved_chunk_ids") or out.get("evidence_ids") or []
        return out
