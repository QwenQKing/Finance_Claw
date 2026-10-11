from __future__ import annotations

import base64
import datetime as dt
import hashlib
import io
import json
import mimetypes
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Sequence, Tuple

from src.evolution import SessionMetrics
from src.governance import (
    ResearchTrace,
    ToolCallRecord,
    ValidationRecord,
)
from src.temporal import (
    ExplorationConfig,
    FinancialKGBuilder,
    TemporalEdge,
    TemporalKG,
    TemporalRule,
    TextSource,
)
from src.temporal.time_model import TemporalEdgeScore, TriClock

def _normalize_tagged_output(raw_text: str, fallback_answer: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        answer = str(fallback_answer or "").strip() or "N/A"
        return f"<think>brief reasoning omitted by upstream adapter</think>\n<answer>{answer}</answer>\n<confidence></confidence>"

    think_matches = re.findall(r"<think>\s*(.*?)\s*</think>", text, flags=re.I | re.S)
    answer_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    confidence_matches = re.findall(r"<confidence>\s*(.*?)\s*</confidence>", text, flags=re.I | re.S)

    def _strip_inline_tags(value: str) -> str:
        return re.sub(r"</?[^>]+>", "", str(value or "")).strip()

    think_text = _strip_inline_tags(think_matches[-1] if think_matches else "")
    answer_text = _strip_inline_tags(answer_matches[-1] if answer_matches else "")
    if not think_text:
        think_text = "brief reasoning omitted by upstream adapter"
    if not answer_text:
        answer_text = str(fallback_answer or "").strip() or "N/A"
    confidence_text = _strip_inline_tags(confidence_matches[-1] if confidence_matches else "")
    return f"<think>{think_text}</think>\n<answer>{answer_text}</answer>\n<confidence>{confidence_text}</confidence>"

def _extract_primary_text(sample: Dict[str, Any]) -> str:
    modalities = sample.get("modalities", []) or []
    for modality in modalities:
        if modality.get("modality") != "text":
            continue
        data = modality.get("data", {}) or {}
        content = data.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""

def _extract_question(sample: Dict[str, Any]) -> str:
    return str(sample.get("question", "")).strip()

def _extract_instruction(sample: Dict[str, Any]) -> str:
    return str(sample.get("instruction", "")).strip()

def _extract_image_modalities(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []
    for modality in sample.get("modalities", []) or []:
        if str(modality.get("modality") or "").lower() != "image":
            continue
        data = modality.get("data", {}) or {}
        if isinstance(data, dict) and data:
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
            raise

    fmt = str(image_data.get("format") or "").strip().lower() or "png"
    if fmt == "jpg":
        fmt = "jpeg"

    inline_base64 = str(image_data.get("base64") or "").strip()
    if not inline_base64:
        raise ValueError("image_data missing base64 data")
    raw = base64.b64decode(inline_base64)
    raw, fmt = _maybe_reencode_image(raw, fmt)
    inline_base64 = base64.b64encode(raw).decode("ascii")
    mime = mimetypes.types_map.get(f".{fmt}", f"image/{fmt}")
    return f"data:{mime};base64,{inline_base64}"

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
    if not image_modalities or not _model_accepts_openai_image_url():
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

def _count_included_images(message_content: str | List[Dict[str, Any]]) -> int:
    if not isinstance(message_content, list):
        return 0
    return sum(1 for item in message_content if isinstance(item, dict) and item.get("type") == "image_url")

def _as_bullets(items: Sequence[str], prefix: str = "- ") -> str:
    return "\n".join(f"{prefix}{item}" for item in items)

_SENTIMENT_TASK_MARKERS = (
    "情感",
    "sentiment",
)
_COMPANY_TASK_MARKERS = (
    "公司名称",
    "涉及到的公司名称",
)
_PRODUCT_TASK_MARKERS = (
    "哪些产品",
    "与哪些产品有关",
)
_TAXONOMY_TASK_MARKERS = (
    "一级分类",
    "一级行业",
    "二级标题",
    "二级分类",
)
_RISK_TASK_MARKERS = (
    "投资风险",
    "风险进行深入分析",
    "潜在风险",
)
_RATING_TASK_MARKERS = (
    "投资建议",
    "投资评级",
    "评级",
)
_SUMMARY_TASK_MARKERS = (
    "摘要",
    "总结",
    "提炼",
    "研究背景",
    "结论和建议",
)
_MARKET_DIRECTION_MARKERS = (
    "股价",
    "走势",
    "上涨",
    "下跌",
    "涨跌",
    "forecast",
    "predict",
    "higher",
    "lower",
    "up or down",
)

def _contains_marker(text: str, markers: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)

def _select_prompt_profile(question: str, instruction: str, labels: Sequence[str]) -> str:
    combined = "\n".join(part for part in (question, instruction) if part)
    if labels:
        if _contains_marker(combined, _SENTIMENT_TASK_MARKERS):
            return "sentiment_label"
        if _contains_marker(combined, _TAXONOMY_TASK_MARKERS):
            return "taxonomy_label"
        if _contains_marker(combined, _MARKET_DIRECTION_MARKERS):
            return "market_direction"
        return "label_choice"
    if _contains_marker(combined, _COMPANY_TASK_MARKERS):
        return "company_extract"
    if _contains_marker(combined, _PRODUCT_TASK_MARKERS):
        return "product_extract"
    if _contains_marker(combined, _RISK_TASK_MARKERS):
        return "risk_analysis"
    if _contains_marker(combined, _RATING_TASK_MARKERS):
        return "investment_advice"
    if _contains_marker(combined, _SUMMARY_TASK_MARKERS):
        return "report_summary"
    return "free_form"

@dataclass(frozen=True)
class _PromptPlan:
    profile: str
    system_prompt: str
    rules: Tuple[str, ...]
    final_instruction: str
    include_temporal_verdict: bool
    include_chain_reports: bool
    include_rule_lines: bool
    include_memory: bool
    include_regime: bool
    include_governance: bool
    evidence_with_temporal_scores: bool

def _build_prompt_plan(question: str, instruction: str, labels: Sequence[str]) -> _PromptPlan:
    profile = _select_prompt_profile(question, instruction, labels)
    allowed = f"Allowed labels: {', '.join(labels)}" if labels else ""
    tagged_format_rule = (
        "Output exactly three tags and nothing else: <think>...</think><answer>...</answer><confidence>80</confidence>. "
        "Inside <confidence>, return digits only: an integer from 0 to 100 estimating the probability that the answer is correct; never leave it empty."
    )

    if profile == "market_direction":
        return _PromptPlan(
            profile=profile,
            system_prompt=(
                "You are the unified FinanceClaw paper method. "
                "Reason over temporal KG, conflict-aware memory, and governance constraints. "
                "Return exactly one allowed label."
            ),
            rules=(
                "Use the temporal KG verdict, conflict-aware memory, and regime/compliance signals jointly.",
                "Prefer temporally legal and conflict-aware evidence.",
                tagged_format_rule,
                "Inside <answer>, return exactly one final label from the allowed set.",
                allowed,
            ),
            final_instruction="Keep <think> brief and place exactly one label in <answer>.",
            include_temporal_verdict=True,
            include_chain_reports=True,
            include_rule_lines=True,
            include_memory=True,
            include_regime=True,
            include_governance=True,
            evidence_with_temporal_scores=True,
        )

    if profile == "sentiment_label":
        return _PromptPlan(
            profile=profile,
            system_prompt=(
                "You are a financial sentiment classifier. "
                "Use only the provided evidence. "
                "Do not let auxiliary NEUTRAL/BULLISH/BEARISH tool outputs override the task definition. "
                "Return exactly one allowed label."
            ),
            rules=(
                "Focus on the sentiment expressed in the source text itself.",
                tagged_format_rule,
                "Inside <answer>, return exactly one final label from the allowed set.",
                allowed,
            ),
            final_instruction="Keep <think> brief and place exactly one label in <answer>.",
            include_temporal_verdict=False,
            include_chain_reports=False,
            include_rule_lines=False,
            include_memory=False,
            include_regime=False,
            include_governance=False,
            evidence_with_temporal_scores=False,
        )

    if profile == "taxonomy_label":
        return _PromptPlan(
            profile=profile,
            system_prompt=(
                "You are a financial taxonomy classifier. "
                "Use only the supplied evidence and return the benchmark-facing category path."
            ),
            rules=(
                "Select the most specific category path supported by the evidence.",
                "When the task is hierarchical, preserve the path using Chinese comma separators.",
                "Do not collapse a supported path to only the first or broadest label.",
                tagged_format_rule,
                allowed,
            ),
            final_instruction="Keep <think> brief and place only the category path inside <answer>.",
            include_temporal_verdict=False,
            include_chain_reports=False,
            include_rule_lines=False,
            include_memory=False,
            include_regime=False,
            include_governance=False,
            evidence_with_temporal_scores=False,
        )

    if profile == "label_choice":
        return _PromptPlan(
            profile=profile,
            system_prompt=(
                "You are a financial classification model. "
                "Use only the supplied evidence and return exactly one allowed label."
            ),
            rules=(
                tagged_format_rule,
                "Inside <answer>, return exactly one final label from the allowed set.",
                allowed,
            ),
            final_instruction="Keep <think> brief and place exactly one label in <answer>.",
            include_temporal_verdict=False,
            include_chain_reports=False,
            include_rule_lines=False,
            include_memory=False,
            include_regime=False,
            include_governance=False,
            evidence_with_temporal_scores=False,
        )

    if profile == "company_extract":
        return _PromptPlan(
            profile=profile,
            system_prompt="You extract company names from financial text.",
            rules=(
                tagged_format_rule,
                "Inside <answer>, return only the company name or company names supported by the evidence.",
                "Deduplicate entities and use Chinese comma separators when multiple names are present.",
            ),
            final_instruction="Keep <think> brief and place only the extracted company name text in <answer>.",
            include_temporal_verdict=False,
            include_chain_reports=False,
            include_rule_lines=False,
            include_memory=False,
            include_regime=False,
            include_governance=False,
            evidence_with_temporal_scores=False,
        )

    if profile == "product_extract":
        return _PromptPlan(
            profile=profile,
            system_prompt="You extract product mentions from financial text.",
            rules=(
                tagged_format_rule,
                "Inside <answer>, return only the products explicitly supported by the evidence.",
                "Use Chinese comma separators for multiple products.",
                "Do not add commentary or generic category words unless they appear in the evidence.",
            ),
            final_instruction="Keep <think> brief and place only the extracted product text in <answer>.",
            include_temporal_verdict=False,
            include_chain_reports=False,
            include_rule_lines=False,
            include_memory=False,
            include_regime=False,
            include_governance=False,
            evidence_with_temporal_scores=False,
        )

    if profile == "risk_analysis":
        return _PromptPlan(
            profile=profile,
            system_prompt="You summarize investment risks from financial research notes.",
            rules=(
                tagged_format_rule,
                "Extract the concrete risk items supported by the evidence.",
                "Prefer compact benchmark-style risk phrases.",
                "Inside <answer>, return only the risk items, separated by Chinese semicolons when multiple items are present.",
            ),
            final_instruction="Keep <think> brief and place only the risk summary in <answer>.",
            include_temporal_verdict=False,
            include_chain_reports=False,
            include_rule_lines=False,
            include_memory=False,
            include_regime=False,
            include_governance=False,
            evidence_with_temporal_scores=False,
        )

    if profile == "investment_advice":
        return _PromptPlan(
            profile=profile,
            system_prompt="You summarize investment advice from Chinese financial research reports.",
            rules=(
                "State the investment rating explicitly when the evidence supports one.",
                "Mirror benchmark style: concise rating-first summary, then brief rationale.",
                "Do not replace a report rating with auxiliary temporal verdict words such as NEUTRAL.",
                tagged_format_rule,
            ),
            final_instruction="Keep <think> brief and place only the investment advice summary in <answer>.",
            include_temporal_verdict=False,
            include_chain_reports=False,
            include_rule_lines=False,
            include_memory=False,
            include_regime=False,
            include_governance=False,
            evidence_with_temporal_scores=False,
        )

    if profile == "report_summary":
        return _PromptPlan(
            profile=profile,
            system_prompt="You summarize Chinese financial research reports for benchmark evaluation.",
            rules=(
                "Prefer 2-4 concise benchmark-style findings.",
                "When appropriate, separate key findings with the Chinese vertical bar character.",
                "Keep the answer information-dense and grounded in the evidence.",
                tagged_format_rule,
            ),
            final_instruction="Keep <think> brief and place only the summary text in <answer>.",
            include_temporal_verdict=False,
            include_chain_reports=False,
            include_rule_lines=False,
            include_memory=False,
            include_regime=False,
            include_governance=False,
            evidence_with_temporal_scores=False,
        )

    return _PromptPlan(
        profile=profile,
        system_prompt="You answer financial benchmark questions using only the supplied evidence.",
        rules=(
            "Use only the supplied evidence.",
            tagged_format_rule,
            "Inside <answer>, return the final answer only.",
        ),
        final_instruction="Keep <think> brief and place the final answer in <answer>.",
        include_temporal_verdict=False,
        include_chain_reports=False,
        include_rule_lines=False,
        include_memory=False,
        include_regime=False,
        include_governance=False,
        evidence_with_temporal_scores=False,
    )

@dataclass(frozen=True)
class _TemporalChunk:
    chunk_id: str
    text: str
    score: float
    tri_clock: TriClock
    score_detail: TemporalEdgeScore

class UnifiedMethodPipeline:

    def __init__(
        self,
        *,
        name: str,
        memory: Any,
        llm: Any,
        rule_miner: Any,
        explorer: Any,
        constraint: Any,
        governance_log: Any,
        compliance_gate: Any,
        jurisdiction_registry: Any,
        tracker: Any,
        evolution: Any,
        score_chunk_fn: Callable[[str, str], float],
        parse_anchor_time_fn: Callable[[Dict[str, Any], str], dt.datetime],
        finalize_prediction_fn: Callable[[Dict[str, Any], str, Sequence[str]], str],
        candidate_labels_fn: Callable[[Dict[str, Any]], List[str]],
        extract_chunks_fn: Callable[[Dict[str, Any]], List[str]],
        enable_fr: bool = True,
        enable_ci: bool = True,
        enable_rc: bool = True,
        fr_variant: str = "default",
        ci_variant: str = "default",
        rc_variant: str = "default",
        allow_pseudo_rules: bool = False,
    ) -> None:
        self.name = name
        self._memory = memory
        self._llm = llm
        self._rule_miner = rule_miner
        self._explorer = explorer
        self._constraint = constraint
        self._governance_log = governance_log
        self._compliance_gate = compliance_gate
        self._jurisdiction_registry = jurisdiction_registry
        self._tracker = tracker
        self._evolution = evolution
        self._score_chunk = score_chunk_fn
        self._parse_anchor_time = parse_anchor_time_fn
        self._finalize_prediction = finalize_prediction_fn
        self._candidate_labels = candidate_labels_fn
        self._extract_chunks = extract_chunks_fn
        self._enable_fr = bool(enable_fr)
        self._enable_ci = bool(enable_ci)
        self._enable_rc = bool(enable_rc)
        self._fr_variant = fr_variant
        self._ci_variant = ci_variant
        self._rc_variant = rc_variant
        self._allow_pseudo_rules = bool(allow_pseudo_rules)
        self._ingested_samples: set[str] = set()
        self._trace_history: List[Dict[str, Any]] = []

    def _sample_id(self, sample: Dict[str, Any]) -> str:
        return str(sample.get("id") or sample.get("sample_id") or "anon").strip() or "anon"

    @staticmethod
    def _stable_rng(*parts: Any) -> random.Random:
        raw = "||".join(str(p) for p in parts)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _build_tri_clock(self, base_time: dt.datetime, idx: int, chunk: str) -> TriClock:
        lag_days = 0 if re.search(r"\b(today|latest|current|今日|最新)\b", chunk, re.IGNORECASE) else min(21, idx * 2)
        event_time = base_time - dt.timedelta(days=max(1, idx + 1))
        release_time = event_time + dt.timedelta(days=lag_days)
        effect_start = release_time
        effect_end = effect_start + dt.timedelta(days=7 if re.search(r"\b(q[1-4]|季度|year|年度)\b", chunk, re.IGNORECASE) else 14)
        if self._fr_variant == "event_time_off":
            event_time = base_time
        if self._fr_variant == "disclosure_time_off":
            release_time = event_time
        if self._fr_variant == "effect_window_off":
            effect_start = release_time
            effect_end = effect_start
        return TriClock(
            event_time=event_time,
            release_time=release_time,
            effect_start=effect_start,
            effect_end=effect_end,
        )

    def _register_entity(self, kg: TemporalKG, entities: Dict[str, Dict[str, str]]) -> None:
        for surface, meta in entities.items():
            eid = str(meta.get("id") or "").strip()
            if not eid:
                continue
            if eid not in kg.entities:
                kg.entities[eid] = {
                    "name": surface,
                    "name_zh": meta.get("name_zh", ""),
                    "type": meta.get("type", "UNKNOWN"),
                }

    def _ingest_chunk_into_kg(
        self,
        builder: FinancialKGBuilder,
        kg: TemporalKG,
        sample_id: str,
        chunk_id: str,
        chunk_text: str,
        published_at: dt.datetime,
    ) -> list[TemporalEdge]:
        entities = builder.extract_entities(chunk_text)
        relations = builder.extract_relations(chunk_text, entities)
        self._register_entity(kg, entities)

        if not entities:
            synthetic_id = f"ent-query-{sample_id}"
            kg.entities.setdefault(synthetic_id, {"name": synthetic_id, "type": "QUERY"})
            entities = {synthetic_id: {"id": synthetic_id, "type": "QUERY"}}
        if not relations:
            relations = ["mentions"]

        source = TextSource(
            source_id=chunk_id,
            source_type="article",
            title=chunk_id,
            published_at=published_at,
            snippet=chunk_text[:120] + ("..." if len(chunk_text) > 120 else ""),
        )

        entity_values = list(entities.values())
        edges: list[TemporalEdge] = []
        for rel in relations:
            if len(entity_values) == 1:
                pairs = [(entity_values[0], entity_values[0])]
            else:
                head = entity_values[0]
                pairs = [(head, tail) for tail in entity_values[1:]]

            for src, tgt in pairs:
                tri_clock = self._build_tri_clock(published_at, len(edges), chunk_text)
                temporal_valid, temporal_reason = self._constraint.validate_edge(tri_clock)
                score = TemporalEdgeScore.from_triclock(tri_clock, self._constraint)
                edge = TemporalEdge(
                    source_entity=str(src["id"]),
                    target_entity=str(tgt["id"]),
                    relation=rel,
                    timestamp=published_at,
                    confidence=min(0.95, 0.55 + self._score_chunk(chunk_text, chunk_text)),
                    provenance=source,
                    tri_clock=tri_clock,
                    temporal_valid=temporal_valid,
                    temporal_reason=temporal_reason,
                    order_score=score.order_score,
                    window_score=score.window_score,
                    lag_penalty=score.lag_penalty,
                )
                kg.edges.append(edge)
                edges.append(edge)
        return edges

    def _build_kg(self, sample: Dict[str, Any], chunks: Sequence[str]) -> Tuple[TemporalKG, List[_TemporalChunk]]:
        builder = FinancialKGBuilder()
        sample_id = self._sample_id(sample)
        question = _extract_question(sample)
        anchor_time = self._parse_anchor_time(sample, " ".join(chunks) or question)
        working_chunks = list(chunks)
        if self._fr_variant == "tri_clock_shuffle" and working_chunks:
            rng = self._stable_rng(self.name, sample_id, "tri_clock_shuffle")
            rng.shuffle(working_chunks)
        temporal_chunks: List[_TemporalChunk] = []
        for idx, chunk in enumerate(working_chunks):
            chunk_id = f"{sample_id}#chunk_{idx+1}"
            tri_clock = self._build_tri_clock(anchor_time, idx, chunk)
            score_detail = TemporalEdgeScore.from_triclock(tri_clock, self._constraint)
            temporal_chunks.append(
                _TemporalChunk(
                    chunk_id=chunk_id,
                    text=chunk,
                    score=self._score_chunk(question, chunk),
                    tri_clock=tri_clock,
                    score_detail=score_detail,
                )
            )
            published_at = tri_clock.release_time
            self._ingest_chunk_into_kg(builder, builder.kg, sample_id, chunk_id, chunk, published_at)

        if not builder.kg.edges:
            raise RuntimeError(f"KG edges empty for sample {sample_id}: no chunks scored or ingested")

        return builder.kg, temporal_chunks

    def _pick_target_entity(self, kg: TemporalKG, question: str) -> str:
        if not kg.entities:
            return "ent-query"
        company_entities = [eid for eid, attrs in kg.entities.items() if attrs.get("type") == "COMPANY"]
        if company_entities:
            return company_entities[0]
        for eid, attrs in kg.entities.items():
            name = str(attrs.get("name") or "").lower()
            if name and name in question.lower():
                return eid
        degrees = {eid: len(kg.edges_for_entity(eid)) for eid in kg.entities}
        return max(degrees, key=degrees.get)

    def _mine_rules(self, kg: TemporalKG) -> tuple[List[TemporalRule], str]:
        timestamps = sorted({edge.timestamp for edge in kg.edges})
        snapshots = [kg.create_snapshot(ts) for ts in timestamps] or [kg]
        rules = self._rule_miner.mine_rules(
            snapshots,
            max_body_length=2,
            min_confidence=0.3,
            min_support=0.0,
        )
        if rules:
            return rules, "mined"
        if self._allow_pseudo_rules:
            return self._rule_miner.mine_rules(None, max_body_length=2, min_confidence=0.3, min_support=0.0), "pseudo"
        return [], "none"

    def _memory_findings(self, sample: Dict[str, Any], chunks: Sequence[str], sample_id: str) -> List[Dict[str, Any]]:
        question = _extract_question(sample)
        findings = []
        for idx, chunk in enumerate(chunks[:8], 1):
            ent_names = re.findall(r"[A-Z][A-Za-z0-9&._-]{2,}|[\u4e00-\u9fff]{2,}", chunk)
            entities = [{"name": name, "type": "company"} for name in ent_names[:4]]
            findings.append(
                {
                    "sentence": chunk,
                    "entities": entities,
                    "confidence": min(0.95, 0.55 + idx * 0.04),
                    "source_tool": "unified_paper_method",
                    "data_timestamp": self._parse_anchor_time(sample, chunk).isoformat(),
                }
            )
        findings.append(
            {
                "sentence": question,
                "entities": [{"name": sample_id, "type": "event"}],
                "confidence": 0.65,
                "source_tool": "unified_paper_method_question",
                "data_timestamp": self._parse_anchor_time(sample, question).isoformat(),
            }
        )
        return findings

    def _partition_memory_hits(
        self,
        hits: Sequence[Dict[str, Any]],
        sample_id: str,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        in_sample: List[Dict[str, Any]] = []
        cross_sample: List[Dict[str, Any]] = []
        for row in hits:
            if str(row.get("source_session") or "").strip() == sample_id:
                in_sample.append(dict(row))
            else:
                cross_sample.append(dict(row))
        return in_sample, cross_sample

    @staticmethod
    def _source_trust_score(row: Dict[str, Any]) -> float:
        sentence = str(row.get("sentence") or "")
        source_tool = str(row.get("source_tool") or "").lower()
        trust_hits = sum(source_tool.count(token) for token in ("sec", "filing", "report", "official", "audited", "guidance"))
        return float(trust_hits * 2 + min(len(sentence), 400) / 400.0)

    def _apply_ci_variant(
        self,
        hits: List[Dict[str, Any]],
        *,
        sample_id: str,
        scope: str,
    ) -> List[Dict[str, Any]]:
        if not hits:
            return hits
        if self._ci_variant == "no_contradiction_edge":
            return []
        if self._ci_variant == "latest_only":
            return hits[-1:]
        if self._ci_variant == "source_trust_only":
            ranked = sorted(hits, key=self._source_trust_score, reverse=True)
            return ranked[:1]
        if self._ci_variant == "majority_collapse":
            return hits[:1]
        if self._ci_variant == "random_edges":
            rng = self._stable_rng(self.name, sample_id, scope, self._ci_variant)
            picks = max(1, min(len(hits), 3))
            return rng.sample(list(hits), k=picks)
        if self._ci_variant == "no_conflict_weighting":
            return hits[: max(1, min(len(hits), 2))]
        return hits

    def _retrieve_memory(
        self,
        sample: Dict[str, Any],
        chunks: Sequence[str],
        sample_id: str,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        question = _extract_question(sample)
        top_k = 4
        retrieve_mode = {
            "ci_keyword_only": "keyword_only",
            "ci_entity_only": "entity_only",
            "ci_rrf_fused": "rrf_fused",
            "ci_conflict_bonus": "rrf_conflict",
        }.get(self._ci_variant, "rrf_conflict")

        if sample_id in self._ingested_samples:
            hits = self._memory.retrieve(question, top_k=top_k * 2, mode=retrieve_mode)
            return self._partition_memory_hits(hits, sample_id)

        cross_sample_hits = self._memory.retrieve(question, top_k=top_k, mode=retrieve_mode)

        findings = self._memory_findings(sample, chunks, sample_id)
        self._memory.ingest_findings(findings, session_id=sample_id)
        self._memory.save()
        self._ingested_samples.add(sample_id)

        post_ingest_hits = self._memory.retrieve(question, top_k=top_k * 2, mode=retrieve_mode)
        in_sample_hits, _ = self._partition_memory_hits(post_ingest_hits, sample_id)
        return in_sample_hits[:top_k], cross_sample_hits[:top_k]

    def _regime_and_update(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        recent = [m.__dict__ for m in self._tracker._history[-10:]]
        if self._rc_variant == "no_regime_detector":
            regime = SimpleNamespace(regime_label="detector_off", shift_confidence=0.0, trigger="rc_variant:no_regime_detector")
        elif self._rc_variant == "fixed_regime":
            regime = SimpleNamespace(regime_label="fixed_neutral", shift_confidence=1.0, trigger="rc_variant:fixed_regime")
        elif self._rc_variant == "random_regime":
            rng = self._stable_rng(self.name, "random_regime", len(recent))
            regime = SimpleNamespace(regime_label=rng.choice(["bullish", "neutral", "stress"]), shift_confidence=0.5, trigger="rc_variant:random_regime")
        else:
            regime = self._evolution.regime_detector.detect_shift(recent)
        if recent:
            avg_conf = sum(r.get("confidence", 0.0) for r in recent) / len(recent)
            avg_viol = sum((r.get("compliance_violations", 0) > 0) for r in recent) / len(recent)
        else:
            avg_conf = 0.65
            avg_viol = 0.0
        if self._rc_variant == "no_constrained_updater":
            update = SimpleNamespace(
                action="updater_disabled",
                objective=avg_conf,
                lagrangian_lambda=0.0,
                smoothed_reward=avg_conf,
                smoothed_violation=avg_viol,
            )
        else:
            update = self._evolution.constrained_updater.update_policy(reward=avg_conf, violation=avg_viol)
        regime_payload = {
            "label": regime.regime_label,
            "shift_confidence": regime.shift_confidence,
            "trigger": regime.trigger,
        }
        update_payload = {
            "action": update.action,
            "objective": update.objective,
            "lambda": update.lagrangian_lambda,
            "reward_ema": update.smoothed_reward,
            "violation_ema": update.smoothed_violation,
        }
        return regime_payload, update_payload

    def _build_trace(self, sample: Dict[str, Any], question: str, task_type: str, model_id: str) -> ResearchTrace:
        risk_tier = self._jurisdiction_registry.classify_risk_tier(task_type)
        return ResearchTrace(
            session_id=self._sample_id(sample),
            user_prompt=question,
            research_depth="benchmark",
            inferred_risk_tier=risk_tier.value,
            model_id=model_id,
            model_version="unified_paper_method_v1",
        )

    @staticmethod
    def _compact_rows(rows: Sequence[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
        compact: List[Dict[str, Any]] = []
        for row in list(rows)[:limit]:
            compact.append(
                {
                    "proposition_id": row.get("proposition_id"),
                    "source_session": row.get("source_session"),
                    "source_tool": row.get("source_tool"),
                    "confidence": row.get("confidence"),
                    "conflict_density": row.get("conflict_density"),
                    "sentence_preview": str(row.get("sentence") or "")[:160],
                }
            )
        return compact

    @staticmethod
    def _compact_temporal_chunks(chunks: Sequence[_TemporalChunk], limit: int = 3) -> List[Dict[str, Any]]:
        compact: List[Dict[str, Any]] = []
        for chunk in list(chunks)[:limit]:
            compact.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "score": round(chunk.score, 4),
                    "temporal_score": round(chunk.score_detail.total, 4),
                    "lag_penalty": round(chunk.score_detail.lag_penalty, 4),
                    "event_time": chunk.tri_clock.event_time.isoformat(),
                    "release_time": chunk.tri_clock.release_time.isoformat(),
                    "effect_start": chunk.tri_clock.effect_start.isoformat(),
                    "effect_end": chunk.tri_clock.effect_end.isoformat(),
                    "text_preview": chunk.text[:160],
                }
            )
        return compact

    def _implementation_snapshot(
        self,
        *,
        sample_id: str,
        kg: TemporalKG,
        temporal_chunks: Sequence[_TemporalChunk],
        rules: Sequence[TemporalRule],
        rule_source: str,
        verdict: Any,
        in_sample_memory_hits: Sequence[Dict[str, Any]],
        cross_sample_memory_hits: Sequence[Dict[str, Any]],
        regime_payload: Dict[str, Any],
        update_payload: Dict[str, Any],
        compliance_decision: Any,
    ) -> Dict[str, Any]:
        return {
            "signature_version": "ours_pipeline_v1",
            "sample_id": sample_id,
            "canonical_pipeline": True,
            "allow_pseudo_rules": self._allow_pseudo_rules,
            "stages": {
                "fr": {
                    "enabled": self._enable_fr,
                    "variant": self._fr_variant,
                    "kg_entities": len(kg.entities),
                    "kg_edges": len(kg.edges),
                    "temporal_chunks": len(temporal_chunks),
                    "rules_mined": len(rules),
                    "rule_source": rule_source,
                    "top_temporal_chunks": self._compact_temporal_chunks(
                        sorted(temporal_chunks, key=lambda x: x.score + x.score_detail.total, reverse=True),
                        limit=3,
                    ),
                    "verdict_summary": verdict.summary() if verdict is not None else "FR stage disabled",
                },
                "ci": {
                    "enabled": self._enable_ci,
                    "variant": self._ci_variant,
                    "in_sample_hits": len(in_sample_memory_hits),
                    "cross_sample_hits": len(cross_sample_memory_hits),
                    "top_in_sample_hits": self._compact_rows(in_sample_memory_hits, limit=3),
                    "top_cross_sample_hits": self._compact_rows(cross_sample_memory_hits, limit=3),
                },
                "rc": {
                    "enabled": self._enable_rc,
                    "variant": self._rc_variant,
                    "regime": regime_payload,
                    "update": update_payload,
                    "compliance": {
                        "passed": bool(getattr(compliance_decision, "passed", False)),
                        "soft_penalty": float(getattr(compliance_decision, "soft_penalty", 0.0)),
                        "violated_rules": list(getattr(compliance_decision, "violated_rules", []) or []),
                    },
                },
            },
        }

    def run_sample(self, sample: Dict[str, Any], budget: Dict[str, Any]) -> Dict[str, Any]:
        stage_timings_ms: Dict[str, float] = {}
        run_started = dt.datetime.now()

        def _mark_stage(stage_name: str, started_at: dt.datetime) -> None:
            stage_timings_ms[stage_name] = round((dt.datetime.now() - started_at).total_seconds() * 1000.0, 3)

        question = _extract_question(sample)
        instruction = _extract_instruction(sample)
        labels = self._candidate_labels(sample)
        sample_id = self._sample_id(sample)
        context = _extract_primary_text(sample)
        chunks = self._extract_chunks(sample)
        if self._fr_variant == "textify_only":
            chunks = [context[:1800]] if context else []
        if self._enable_fr:
            fr_started = dt.datetime.now()
            kg, temporal_chunks = self._build_kg(sample, chunks)
            target_entity = self._pick_target_entity(kg, question)
            rules, rule_source = self._mine_rules(kg)
            prediction_date = self._parse_anchor_time(sample, context or question).date()
            verdict = self._explorer.explore(
                stock_entity_id=target_entity,
                prediction_date=prediction_date,
                kg=kg,
                rules=rules,
                config=ExplorationConfig(
                    beam_width=4,
                    max_depth=3,
                    lookback_days=90,
                    min_edge_confidence=0.0,
                    min_chain_confidence=0.0,
                    max_chains=5,
                ),
            )
            _mark_stage("fr_total", fr_started)
        else:
            kg = TemporalKG()
            temporal_chunks = []
            rules = []
            rule_source = "disabled"
            verdict = None

        ci_started = dt.datetime.now()
        in_sample_memory_hits, cross_sample_memory_hits = self._retrieve_memory(sample, chunks, sample_id) if self._enable_ci else ([], [])
        if self._enable_ci:
            in_sample_memory_hits = self._apply_ci_variant(in_sample_memory_hits, sample_id=sample_id, scope="in_sample")
            cross_sample_memory_hits = self._apply_ci_variant(cross_sample_memory_hits, sample_id=sample_id, scope="cross_sample")
        _mark_stage("ci_total", ci_started)
        memory_hits = [*in_sample_memory_hits[:2], *cross_sample_memory_hits[:2]]
        rc_started = dt.datetime.now()
        if self._enable_rc:
            regime_payload, update_payload = self._regime_and_update()
        else:
            regime_payload = {"label": "disabled", "shift_confidence": 0.0, "trigger": "rc_disabled"}
            update_payload = {"action": "disabled", "objective": 0.0, "lambda": 0.0, "reward_ema": 0.0, "violation_ema": 0.0}
        _mark_stage("rc_total", rc_started)

        trace_started = dt.datetime.now()
        jurisdictions = self._jurisdiction_registry.infer_jurisdictions(markets=[str(sample.get("domain") or "")])
        disclaimer = self._jurisdiction_registry.get_disclaimer(jurisdictions)
        task_type = str(sample.get("topic") or sample.get("domain") or "analysis")
        trace = self._build_trace(sample, question, task_type, self._llm.model)
        trace.jurisdictions = jurisdictions
        trace.disclaimers_applied = [disclaimer]

        if self._enable_fr:
            trace.record_tool_call(ToolCallRecord(call_id="", tool_name="temporal_kg_builder", source="src.temporal", output_summary=kg.summary()))
            trace.record_tool_call(ToolCallRecord(call_id="", tool_name="rule_miner", source="src.temporal", output_summary=f"rules={len(rules)} source={rule_source}"))
            trace.record_tool_call(
                ToolCallRecord(
                    call_id="",
                    tool_name="evidence_chain_explorer",
                    source="src.temporal",
                    output_summary=verdict.summary() if verdict is not None else "No evidence chains found.",
                )
            )
            trace.record_validation(
                ValidationRecord(
                    check_type="temporal_validity",
                    target=sample_id,
                    status="pass" if all(c.score_detail.total >= 0.0 for c in temporal_chunks) else "warn",
                    actual=str([round(c.score_detail.total, 3) for c in temporal_chunks]),
                )
            )
        if self._enable_ci:
            trace.record_tool_call(
                ToolCallRecord(
                    call_id="",
                    tool_name="memory_hypergraph",
                    source="src.memory",
                    output_summary=(
                        f"memory_hits={len(memory_hits)} "
                        f"(in_sample={len(in_sample_memory_hits)}, cross_sample={len(cross_sample_memory_hits)})"
                    ),
                )
            )
        _mark_stage("trace_setup", trace_started)

        prompt_started = dt.datetime.now()
        chain_reports = [chain.summary() for chain in verdict.chains[:3]] if verdict else []
        verdict_chain_count = len(verdict.chains) if verdict is not None else 0
        verdict_direction = verdict.direction.value if verdict is not None else "disabled"
        verdict_confidence = float(verdict.aggregated_confidence) if verdict is not None else 0.0
        if not chain_reports:
            chain_reports = ["FR stage disabled." if not self._enable_fr else "No evidence chains found."]
        prompt_plan = _build_prompt_plan(question, instruction, labels)
        in_sample_memory_lines = [
            f"[{row.get('proposition_id', row.get('source_session', 'mem'))}] {row.get('sentence', '')} "
            f"(conf={row.get('confidence', 0):.2f}, conflict={row.get('conflict_density', 0):.2f})"
            for row in in_sample_memory_hits[:4]
        ] or ["CI stage disabled." if not self._enable_ci else "- none"]
        cross_sample_memory_lines = [
            f"[{row.get('proposition_id', row.get('source_session', 'mem'))}] {row.get('sentence', '')} "
            f"(conf={row.get('confidence', 0):.2f}, conflict={row.get('conflict_density', 0):.2f})"
            for row in cross_sample_memory_hits[:4]
        ] or ["CI stage disabled." if not self._enable_ci else "- none"]
        if prompt_plan.evidence_with_temporal_scores:
            top_chunk_lines = [
                f"[{c.chunk_id}] {c.text} "
                f"(temporal={c.score_detail.total:.2f}, lag_penalty={c.score_detail.lag_penalty:.2f})"
                for c in sorted(temporal_chunks, key=lambda x: x.score + x.score_detail.total, reverse=True)[:5]
            ]
        else:
            top_chunk_lines = [
                f"[{c.chunk_id}] {c.text}"
                for c in sorted(temporal_chunks, key=lambda x: x.score + x.score_detail.total, reverse=True)[:5]
            ]
        if not top_chunk_lines:
            top_chunk_lines = [f"[chunk_{idx+1}] {chunk}" for idx, chunk in enumerate(chunks[:5])]
        rule_lines = [str(rule) for rule in rules[:5]] or ["FR stage disabled." if not self._enable_fr else "- none"]

        prompt_sections = [
            f"Question:\n{question}",
            f"Instruction:\n{instruction}" if instruction else "",
            f"Core Evidence:\n{_as_bullets(top_chunk_lines)}",
        ]
        if self._enable_fr and prompt_plan.include_temporal_verdict:
            prompt_sections.extend(
                [
                    f"Temporal KG Verdict:\n{verdict.summary()}",
                ]
            )
        if self._enable_fr and prompt_plan.include_chain_reports:
            prompt_sections.append(f"Evidence Chains:\n{_as_bullets(chain_reports)}")
        if self._enable_fr and prompt_plan.include_rule_lines:
            prompt_sections.append(f"Mined Temporal Rules:\n{_as_bullets(rule_lines)}")
        if self._enable_ci and prompt_plan.include_memory:
            prompt_sections.append(f"In-Sample Conflict Memory:\n{_as_bullets(in_sample_memory_lines)}")
            prompt_sections.append(f"Cross-Sample Conflict Memory:\n{_as_bullets(cross_sample_memory_lines)}")
        if self._enable_rc and prompt_plan.include_regime:
            prompt_sections.append(
                (
                    "Regime / Constraint Update:\n"
                    f"- regime={regime_payload['label']} (shift_confidence={regime_payload['shift_confidence']:.2f}, trigger={regime_payload['trigger']})\n"
                    f"- updater_action={update_payload['action']} (lambda={update_payload['lambda']:.3f}, objective={update_payload['objective']:.3f})"
                )
            )
        if self._enable_rc and prompt_plan.include_governance:
            prompt_sections.append(
                (
                    "Governance / Compliance:\n"
                    f"- jurisdictions={', '.join(jurisdictions)}\n"
                    f"- disclaimer={disclaimer}"
                )
            )
        prompt_sections.extend(
            [
                "Rules:\n" + _as_bullets([rule for rule in prompt_plan.rules if rule]),
                prompt_plan.final_instruction,
            ]
        )
        prompt_text = "\n\n".join(section for section in prompt_sections if section)
        _mark_stage("prompt_build", prompt_started)

        llm_budget = dict(budget)
        llm_budget["max_output_tokens"] = int(budget.get("max_output_tokens", 8192))
        llm_budget["temperature"] = float(budget.get("temperature", 0.0))
        llm_budget["top_p"] = float(budget.get("top_p", 1.0))
        user_content = _multimodal_message_content(prompt_text, sample, llm_budget)
        included_image_count = _count_included_images(user_content)
        llm_started = dt.datetime.now()
        llm_out = self._llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": prompt_plan.system_prompt,
                },
                {"role": "user", "content": user_content},
            ],
            budget=llm_budget,
        )
        _mark_stage("llm_chat", llm_started)
        trace.record_tool_call(
            ToolCallRecord(
                call_id="",
                tool_name="llm_finalize",
                source="openai_compatible",
                output_summary=str(llm_out.get("content") or "")[:200],
                output_hash=hashlib.sha256(str(llm_out.get("content") or "").encode("utf-8")).hexdigest(),
            )
        )

        compliance_started = dt.datetime.now()
        raw_pred = str(llm_out.get("content") or "").strip()
        llm_pred = self._finalize_prediction(sample, raw_pred, chunks)
        confidence_llm_out: Dict[str, Any] = {}
        if os.getenv("REQUIRE_CONFIDENCE_TAG", "").strip().lower() in {"1", "true", "yes", "on"} and not re.search(
            r"<confidence>\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*</confidence>",
            raw_pred,
            flags=re.I | re.S,
        ):
            confidence_budget = dict(llm_budget)
            confidence_budget["max_output_tokens"] = min(32, int(confidence_budget.get("max_output_tokens", 32) or 32))
            confidence_llm_out = self._llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "Return only one tag: <confidence>N</confidence>, where N is an integer from 0 to 100.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Estimate the probability that this benchmark answer is correct.\n"
                            f"Question:\n{question}\n\n"
                            f"Answer:\n{llm_pred}\n\n"
                            "Return only <confidence>N</confidence> with digits only."
                        ),
                    },
                ],
                budget=confidence_budget,
            )
            raw_pred = raw_pred + "\n" + str(confidence_llm_out.get("content", "") or "").strip()

        if self._enable_rc and self._rc_variant != "no_compliance_gate":
            compliance_decision = self._compliance_gate.evaluate(
                f"{question}\n{llm_pred}",
                context={"audience": "public"},
            )
            trace.record_validation(
                ValidationRecord(
                    check_type="compliance_gate",
                    target=sample_id,
                    status="pass" if compliance_decision.passed else "fail",
                    actual=json.dumps({"soft_penalty": compliance_decision.soft_penalty, "rules": compliance_decision.violated_rules}, ensure_ascii=False),
                )
            )
        else:
            compliance_decision = SimpleNamespace(passed=True, soft_penalty=0.0, violated_rules=[])
        _mark_stage("post_llm_finalize", compliance_started)

        wrapup_started = dt.datetime.now()
        lag_penalty_gate = self._fr_variant != "no_lag_penalty"
        temporal_valid = (
            all((not c.score_detail.lag_penalty > 0.3) if lag_penalty_gate else True for c in temporal_chunks)
            and all(c.tri_clock.release_time >= c.tri_clock.event_time for c in temporal_chunks)
        ) if self._enable_fr else True
        conflict_density = max((row.get("conflict_density", 0.0) for row in memory_hits), default=0.0)
        in_sample_conflict_density = max((row.get("conflict_density", 0.0) for row in in_sample_memory_hits), default=0.0)
        cross_sample_conflict_density = max((row.get("conflict_density", 0.0) for row in cross_sample_memory_hits), default=0.0)
        effective_conflict_density = in_sample_conflict_density
        confidence = 0.45
        if self._enable_fr and verdict is not None:
            confidence += min(0.2, verdict.aggregated_confidence * 0.35)
            confidence += min(0.1, (sum(c.score_detail.total for c in temporal_chunks) / max(1, len(temporal_chunks))) * 0.15)
        if self._enable_rc and regime_payload["label"] == "stable":
            confidence += 0.05
        if self._enable_ci:
            confidence -= min(0.15, effective_conflict_density * 0.2)
        if self._enable_rc:
            confidence -= float(compliance_decision.soft_penalty)
        if self._enable_fr and not temporal_valid:
            confidence -= 0.1
        confidence = max(0.2, min(0.9, confidence))
        violation = (not compliance_decision.passed) or (
            self._enable_ci and bool(in_sample_memory_hits) and in_sample_conflict_density > 0.55
        )

        trace.output_risk_tier = trace.inferred_risk_tier
        trace.finalize(llm_pred)
        self._governance_log.append(trace)

        metrics = SessionMetrics(
            session_id=sample_id,
            trace_id=trace.trace_id,
            research_depth=trace.research_depth,
            task_type=task_type,
            confidence=confidence,
            hallucination_rate=max(0.0, 1.0 - confidence) * 0.1,
            compliance_violations=int(violation),
            tool_calls=len(trace.tool_calls),
            sources_used=trace.total_sources,
            dimensions_covered=4,
            validation_pass_rate=trace.overall_confidence or 0.0,
            total_latency_ms=0,
            tokens_used=int(llm_out.get("total_tokens", 0)) + int(confidence_llm_out.get("total_tokens", 0) or 0),
        )
        if self._enable_rc and self._rc_variant != "no_constrained_updater":
            self._evolution.record_session(metrics)

        health_started = dt.datetime.now()
        memory_health = self._memory.health()
        _mark_stage("memory_health", health_started)

        snapshot_started = dt.datetime.now()
        implementation_snapshot = self._implementation_snapshot(
            sample_id=sample_id,
            kg=kg,
            temporal_chunks=temporal_chunks,
            rules=rules,
            rule_source=rule_source,
            verdict=verdict,
            in_sample_memory_hits=in_sample_memory_hits,
            cross_sample_memory_hits=cross_sample_memory_hits,
            regime_payload=regime_payload,
            update_payload=update_payload,
            compliance_decision=compliance_decision,
        )
        _mark_stage("implementation_snapshot", snapshot_started)

        extras = {
            "method_impl": "unified_paper_method",
            "method_name": self.name,
            "raw_model_output": _normalize_tagged_output(
                raw_pred,
                str(llm_pred or "").strip(),
            ),
            "multimodal_input_images": included_image_count,
            "multimodal_input_enabled": included_image_count > 0,
            "trace_id": trace.trace_id,
            "enable_fr": self._enable_fr,
            "enable_ci": self._enable_ci,
            "enable_rc": self._enable_rc,
            "variant_fr": self._fr_variant,
            "variant_ci": self._ci_variant,
            "variant_rc": self._rc_variant,
            "jurisdictions": jurisdictions,
            "kg_entities": len(kg.entities),
            "kg_edges": len(kg.edges),
            "rules_mined": len(rules),
            "rule_source": rule_source,
            "evidence_chains": verdict_chain_count,
            "verdict_direction": verdict_direction,
            "verdict_confidence": verdict_confidence,
            "memory_hits": len(memory_hits),
            "memory_hits_in_sample": len(in_sample_memory_hits),
            "memory_hits_cross_sample": len(cross_sample_memory_hits),
            "conflict_density": conflict_density,
            "conflict_density_in_sample": in_sample_conflict_density,
            "conflict_density_cross_sample": cross_sample_conflict_density,
            "effective_conflict_density": effective_conflict_density,
            "regime_signal": regime_payload,
            "constraint_update": update_payload,
            "compliance_passed": compliance_decision.passed,
            "compliance_soft_penalty": compliance_decision.soft_penalty,
            "compliance_rules": compliance_decision.violated_rules,
            "disclaimer": disclaimer,
            "governance_confidence": trace.overall_confidence,
            "memory_health": memory_health,
            "integrated_modules": ["temporal", "rule_miner", "evidence_chain", "memory_hypergraph", "regime_detector", "constrained_updater", "governance"],
            "implementation_snapshot": implementation_snapshot,
        }
        if confidence_llm_out:
            extras["confidence_retry_used"] = True
            extras["confidence_retry_raw_model_output"] = str(confidence_llm_out.get("content", "") or "")
        _mark_stage("wrapup_total", wrapup_started)
        stage_timings_ms["run_total"] = round((dt.datetime.now() - run_started).total_seconds() * 1000.0, 3)
        extras["stage_timings_ms"] = stage_timings_ms

        self._trace_history.append(
            {
                "sample_id": sample_id,
                "trace_id": trace.trace_id,
                "rule_source": rule_source,
                "rules_mined": len(rules),
            }
        )

        return {
            "prediction": llm_pred,
            "confidence": confidence,
            "evidence_ids": [c.chunk_id for c in temporal_chunks[:5]],
            "temporal_valid": temporal_valid,
            "violation": violation,
            "steps_used": 5,
            "top_k_used": len(chunks),
            "prompt_tokens": int(llm_out.get("prompt_tokens", max(1, len(prompt_text) // 4)))
            + int(confidence_llm_out.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(llm_out.get("completion_tokens", 1))
            + int(confidence_llm_out.get("completion_tokens", 0) or 0),
            "total_tokens": int(llm_out.get("total_tokens", max(1, len(prompt_text) // 4) + 1))
            + int(confidence_llm_out.get("total_tokens", 0) or 0),
            "extras": extras,
        }
