from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.memory.hypergraph import FinancialMemoryHypergraph, HypergraphProposition
from src.temporal.time_model import TemporalConstraint, TriClock, TemporalEdgeScore

from .base import BaselineAdapter
from ..profiling import attach_stage_timings
from .simple import (
    OpenAICompatibleLLM,
    _as_bullets,
    _ensure_nonempty_answer_tagged,
    _extract_answer_text,
    _extract_instruction,
    _multimodal_message_content,
    _extract_primary_text,
    _extract_question,
    _extract_labels_from_question,
    _keyword_overlap_score,
    _normalize_text,
    _sample_allows_multi_label,
    _split_chunks,
    _tokenize,
)

_TEMPORAL_MARKERS = re.compile(
    r"\b(19\d{2}|20\d{2}|q[1-4]|yoy|mom|mtd|ytd|latest|previous|before|after|during|since|next)\b",
    re.IGNORECASE,
)
_CONFLICT_MARKERS = ("but", "however", "although", "yet", "despite", "while", "whereas", "contrary")
_UP_MARKERS = ("up", "increase", "growth", "beat", "surge", "improve", "strong", "上涨", "增长", "改善", "利好")
_DOWN_MARKERS = ("down", "decline", "drop", "fall", "weaken", "miss", "weak", "下跌", "下降", "恶化", "利空")
_TAXONOMY_MARKERS = ("一级分类", "一级行业", "二级标题", "二级分类")

@dataclass
class _FREvidence:
    chunk_id: str
    text: str
    score: float
    tri_clock: TriClock
    score_detail: TemporalEdgeScore

class RealFRCIAdapter(BaselineAdapter):

    name = "real_fr_ci"

    def __init__(self, instance_tag: str | None = None) -> None:
        self._instance_tag = self._sanitize_instance_tag(instance_tag)
        self._memory = FinancialMemoryHypergraph(
            storage_dir=self._artifact_dir("experiments/artifacts/real_fr_ci_memory")
        )
        self._temporal_constraint = TemporalConstraint(max_release_lag_days=45)
        self._llm: Optional[OpenAICompatibleLLM] = None
        self._kg_enabled = os.getenv("REAL_FR_CI_ENABLE_KG", "1").strip().lower() not in {"0", "false", "no", "off"}
        self._ingested_samples: set[str] = set()
        self._memory_ingest_enabled = os.getenv("REAL_FR_CI_ENABLE_MEMORY_INGEST", "1").strip().lower() not in {"0", "false", "no", "off"}
        self._kg_seed_cache: Dict[str, List[Dict[str, Any]]] = {}

    @staticmethod
    def _sanitize_instance_tag(instance_tag: str | None) -> str:
        raw = str(instance_tag or "").strip()
        if not raw:
            return ""
        return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._")[:80]

    def _artifact_dir(self, base: str | Path) -> Path:
        root = Path(base)
        if not self._instance_tag:
            return root
        return root / "isolated" / self._instance_tag

    @property
    def llm(self) -> OpenAICompatibleLLM:
        if self._llm is None:
            self._llm = OpenAICompatibleLLM()
        return self._llm

    def _candidate_labels(self, sample: Dict[str, Any]) -> List[str]:
        question = _extract_question(sample)
        answer_type = str(sample.get("answer_type") or "").strip().lower()
        labels = _extract_labels_from_question(question)
        if labels:
            return labels
        instruction = _extract_instruction(sample)
        if instruction:
            labels = _extract_labels_from_question(instruction)
            if labels:
                return labels

        meta = sample.get("meta") or {}
        options = meta.get("options")
        if isinstance(options, dict):
            ordered: List[str] = []
            for key in options.keys():
                label = str(key or "").strip()
                if label and label not in ordered:
                    ordered.append(label)
            if ordered:
                return ordered

        lower_question = question.lower()
        if "sentiment" in lower_question:
            return ["negative", "neutral", "positive"]
        if re.search(r"\btrue\s+or\s+false\b", lower_question):
            return ["true", "false"]
        if re.search(r"\byes\s+or\s+no\b", lower_question):
            return ["yes", "no"]
        if answer_type == "yes_no":
            return ["True", "False"]

        for text in [
            str(sample.get("_original_text_content") or "").strip(),
            str(sample.get("_original_instruction") or "").strip(),
        ]:
            if not text:
                continue
            labels = _extract_labels_from_question(text)
            if labels:
                return labels

        patterns = [
            r"从（([^（）]+)）中选取答案",
            r"从\(([^()]+)\)中选取答案",
            r"请从（([^（）]+)）中选取",
            r"请从\(([^()]+)\)中选取",
            r"可选的[^：:]*[：:]\s*（([^（）]+)）",
            r"可选的[^：:]*[：:]\s*\(([^()]+)\)",
        ]
        for pattern in patterns:
            m = re.search(pattern, question)
            if not m:
                continue
            raw = m.group(1)
            parts = [p.strip() for p in re.split(r"[，,、/]", raw) if p.strip()]
            if len(parts) >= 2:
                return parts
        return []

    @staticmethod
    def _lexical_terms(text: str) -> set[str]:
        terms = {token for token in _tokenize(text) if token}
        terms.update(match.lower() for match in re.findall(r"\d+(?:\.\d+)?%?", text))
        for seq in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            seq = seq.strip()
            if len(seq) <= 4:
                terms.add(seq)
                continue
            max_n = min(4, len(seq))
            for n in range(2, max_n + 1):
                for idx in range(len(seq) - n + 1):
                    terms.add(seq[idx : idx + n])
        return terms

    def _clean_free_form_prediction(self, question: str, raw_prediction: str) -> str:
        text = _extract_answer_text(raw_prediction)
        final_match = re.search(r"FINAL\s*=\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
        if final_match:
            text = final_match.group(1).strip()
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        lines = [line.strip().lstrip("> ").strip() for line in text.splitlines() if line.strip()]
        text = " ".join(lines).strip()
        text = re.sub(r"^(最终答案|答案|预测|输出|结论|公司名称|涉及产品|投资评级|情感分析|情感倾向)\s*[:：]\s*", "", text)
        text = re.sub(r"^(最终答案是|答案是|结论是|预测为|投资评级为|情感为)\s*", "", text)
        if "公司名称" in question or "涉及到的公司名称" in question:
            quoted = re.findall(r"[“\"]([^”\"]+)[”\"]", text)
            if quoted:
                return quoted[0].strip()
            if "是" in text and len(text) < 80:
                return text.split("是")[-1].strip("：:。.;； ")
        if "哪些产品" in question or "与哪些产品有关" in question:
            if "：" in text and len(text) < 120:
                return text.split("：")[-1].strip("。.;； ")
        if "情感" in question:
            for label in ("中性", "积极", "消极", "neutral", "positive", "negative"):
                if label in text:
                    return label
        return text

    @staticmethod
    def _refine_company_prediction(sample: Dict[str, Any], prediction: str) -> str:
        original_text = str(sample.get("_original_text_content") or _extract_primary_text(sample) or "").strip()
        if not original_text:
            return prediction
        candidates = [part.strip() for part in re.split(r"[，,、|/;；]", prediction) if part.strip()]
        if len(candidates) <= 1:
            return prediction
        ranked = []
        for candidate in candidates:
            count = original_text.count(candidate)
            if count > 0:
                ranked.append((count, -original_text.find(candidate), candidate))
        if not ranked:
            return prediction
        ranked.sort(reverse=True)
        return ranked[0][2]

    @staticmethod
    def _expects_numeric_answer(sample: Dict[str, Any], question: str) -> bool:
        answer_type = str(sample.get("answer_type", "")).strip().lower()
        topic = str(sample.get("topic", "")).strip().lower()
        if answer_type in {"number", "arithmetic", "count"}:
            return True
        if topic in {"qa", "conversational_qa", "calcu", "numerical", "table_text_qa"}:
            return True
        lower_question = question.lower()
        return any(
            hint in lower_question
            for hint in (
                "what was",
                "what is",
                "how much",
                "how many",
                "difference",
                "percentage",
                "percent",
            )
        )

    @staticmethod
    def _normalize_numeric_prediction(text: str) -> str:
        matches = re.findall(r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?%?", text)
        if not matches:
            return text
        value = matches[0].replace("$", "").replace(",", "").replace(" ", "").strip()
        return value

    def _finalize_prediction(self, sample: Dict[str, Any], raw_prediction: str, chunks: Sequence[str]) -> str:
        question = _extract_question(sample)
        labels = self._candidate_labels(sample)
        if labels:
            if any(marker in question for marker in _TAXONOMY_MARKERS):
                return self._clean_free_form_prediction(question, raw_prediction)
            if _sample_allows_multi_label(sample):
                return self._pick_multi_label(labels, [raw_prediction, *chunks])
            return self._pick_label(labels, question, [raw_prediction, *chunks])
        cleaned = self._clean_free_form_prediction(question, raw_prediction)
        if "公司名称" in question or "涉及到的公司名称" in question:
            cleaned = self._refine_company_prediction(sample, cleaned)
        if self._expects_numeric_answer(sample, question):
            return self._normalize_numeric_prediction(cleaned)
        return cleaned

    def _build_tri_clock(self, text: str, idx: int, total: int) -> TriClock:
        now = datetime.now(timezone.utc)
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
        year = int(year_match.group(1)) if year_match else now.year
        month = min(12, max(1, (idx % 12) + 1))
        day = min(28, max(1, (idx % 28) + 1))
        event_time = datetime(year, month, day, tzinfo=timezone.utc)
        lag_days = min(60, max(0, (idx * 7) % 60))
        release_time = event_time + timedelta(days=lag_days)
        effect_start = release_time
        effect_end = effect_start + timedelta(days=max(1, 3 + (total - idx) % 7))
        return TriClock(event_time=event_time, release_time=release_time, effect_start=effect_start, effect_end=effect_end)

    def _extract_chunks(self, sample: Dict[str, Any]) -> List[str]:
        ctx = _extract_primary_text(sample)
        chunks = _split_chunks(ctx, max_len=220)
        if chunks:
            return chunks[:8]
        ctx = _normalize_text(ctx)
        return [ctx[:500]] if ctx else []

    def _flag_enabled(self, budget: Dict[str, Any], key: str, env_name: str, default: bool = True) -> bool:
        if key in budget and budget.get(key) is not None:
            val = str(budget.get(key)).strip().lower()
            return val not in {"0", "false", "no", "off"}
        val = os.getenv(env_name, "").strip().lower()
        if val:
            return val not in {"0", "false", "no", "off"}
        return default

    def _kg_query_enabled(self, budget: Dict[str, Any]) -> bool:
        return self._flag_enabled(budget, "real_fr_ci_enable_kg", "REAL_FR_CI_ENABLE_KG", default=True) and bool(os.getenv("KG_ARTIFACT_DIR", "").strip())

    def _kg_retrieve(self, question: str, chunks: Sequence[str], budget: Dict[str, Any], top_k: int = 4) -> List[Dict[str, Any]]:
        if not self._kg_query_enabled(budget):
            return []
        try:
            cache_key = question + "||" + "||".join(chunks[:3])
            cached = self._kg_seed_cache.get(cache_key)
            if cached is not None:
                return [dict(x) for x in cached[:top_k]]

            results: List[Dict[str, Any]] = []
            seeds: List[str] = []
            for raw in [question, *chunks[:3]]:
                for token in re.findall(r"[A-Z][A-Za-z0-9_&.-]{2,}|[\u4e00-\u9fff]{2,}", raw):
                    token = token.strip(".,;:()[]{}<>")
                    if len(token) >= 2 and token not in seeds:
                        seeds.append(token)
                    if len(seeds) >= 4:
                        break
                if len(seeds) >= 4:
                    break

            seen_prop_ids: set[str] = set()
            for seed in seeds:
                entity = self._memory.find_entity(seed)
                if not entity:
                    continue
                for prop in self._memory.propositions_for_entity(entity.entity_id):
                    if prop.proposition_id in seen_prop_ids:
                        continue
                    seen_prop_ids.add(prop.proposition_id)
                    results.append(
                        {
                            "proposition_id": prop.proposition_id,
                            "sentence": prop.sentence,
                            "confidence": prop.confidence,
                            "conflict_density": self._memory.conflict_density_for_proposition(prop.proposition_id),
                            "source_session": prop.source_session,
                            "source_tool": prop.source_tool,
                            "grounding_level": prop.grounding_level,
                            "entities": [{"name": e.name, "type": e.entity_type} for e in self._memory.entities_for_proposition(prop.proposition_id)],
                        }
                    )
                    if len(results) >= top_k:
                        self._kg_seed_cache[cache_key] = list(results)
                        return results

            if not results:
                for item in self._memory.retrieve(question, top_k=top_k):
                    results.append(item)
            self._kg_seed_cache[cache_key] = list(results)
            return results[:top_k]
        except Exception:
            return []

    def _score_chunk(self, question: str, chunk: str) -> float:
        q_terms = self._lexical_terms(question)
        c_terms = self._lexical_terms(chunk)
        if q_terms and c_terms:
            overlap = len(q_terms.intersection(c_terms))
            base = overlap / max(1, len(q_terms))
        else:
            base = _keyword_overlap_score(question, chunk)
        exact_bonus = 0.0
        for term in sorted(q_terms, key=len, reverse=True):
            if len(term) < 3:
                continue
            if term in chunk.lower() or term in chunk:
                exact_bonus = 0.12
                break
        temporal_bonus = 0.12 if _TEMPORAL_MARKERS.search(chunk) else 0.0
        numeric_bonus = 0.06 if re.search(r"\d+(?:\.\d+)?%?|\$\d+|\b\d+[kbm]\b", chunk, re.IGNORECASE) else 0.0
        return base + exact_bonus + temporal_bonus + numeric_bonus

    def _find_conflict_chunks(self, chunks: Sequence[str]) -> List[str]:
        conflicts = [c for c in chunks if any(m in c.lower() for m in _CONFLICT_MARKERS)]
        if len(conflicts) >= 2:
            return conflicts
        labeled = []
        for c in chunks:
            lower = c.lower()
            up = sum(1 for m in _UP_MARKERS if m.lower() in lower)
            down = sum(1 for m in _DOWN_MARKERS if m.lower() in lower)
            if up and down:
                labeled.append(c)
        return conflicts + [c for c in labeled if c not in conflicts]

    def _infer_stance(self, text: str) -> str:
        lower = text.lower()
        up = sum(1 for m in _UP_MARKERS if m.lower() in lower)
        down = sum(1 for m in _DOWN_MARKERS if m.lower() in lower)
        if up > down:
            return "bullish"
        if down > up:
            return "bearish"
        return "neutral"

    def _pick_label(self, labels: Sequence[str], question: str, chunks: Sequence[str]) -> str:
        if not labels:
            return ""
        if len(labels) == 1:
            return labels[0]

        raw_blob = str(chunks[0] if chunks else "").lower()
        evidence_blob = " ".join(str(chunk) for chunk in chunks[1:]).lower()
        label_scores: List[Tuple[float, str]] = []
        for label in labels:
            score = 0.0
            label_l = label.lower()
            if label_l in raw_blob:
                score += 4.0
            elif label_l in evidence_blob:
                score += 1.0
            tokens = _tokenize(label)
            if tokens:
                score += sum(1 for t in tokens if t in raw_blob) * 1.5
                score += sum(1 for t in tokens if t in evidence_blob) * 0.35
            if any(k in label_l for k in ("up", "increase", "bull", "positive", "yes", "true")):
                score += sum(1 for m in _UP_MARKERS if m.lower() in raw_blob) * 0.4
                score += sum(1 for m in _UP_MARKERS if m.lower() in evidence_blob) * 0.1
            if any(k in label_l for k in ("down", "decrease", "bear", "negative", "no", "false")):
                score += sum(1 for m in _DOWN_MARKERS if m.lower() in raw_blob) * 0.4
                score += sum(1 for m in _DOWN_MARKERS if m.lower() in evidence_blob) * 0.1
            label_scores.append((score, label))

        label_scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_label = label_scores[0]
        if best_score <= 0:
            raise ValueError(f"Cannot classify: no label matched any chunk for labels={labels}")
        return best_label

    def _pick_multi_label(self, labels: Sequence[str], chunks: Sequence[str]) -> str:
        if not labels:
            return ""
        allowed = {label.upper(): label for label in labels}
        upper_blob = " ".join(str(chunk) for chunk in chunks).upper()
        seen: List[str] = []
        for token in re.findall(r"[A-Z]", upper_blob):
            if token in allowed and token not in seen:
                seen.append(token)
        if not seen:
            return labels[0]
        return "".join(label for label in labels if label.upper() in seen)

    def _memory_ingest(self, sample: Dict[str, Any], chunks: Sequence[str], question: str) -> None:
        sample_ref = str(sample.get("id") or sample.get("sample_id") or "anon")
        findings = []
        for i, chunk in enumerate(chunks[:6], 1):
            findings.append(
                {
                    "sentence": chunk,
                    "entities": [{"name": f"Sample_{sample_ref}", "type": "event"}],
                    "confidence": min(1.0, 0.55 + 0.05 * i),
                    "source_tool": "real_fr_ci_chunk",
                    "data_timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        if findings:
            self._memory.ingest_findings(findings, session_id=sample_ref)
        prop_id = f"q_{abs(int.from_bytes(question.encode('utf-8')[:8].ljust(8, b'0'), 'little')) % 10**12}"
        self._memory.add_proposition(
            HypergraphProposition(
                proposition_id=prop_id,
                sentence=question,
                confidence=0.6,
                source_session=sample_ref,
                source_tool="question_anchor",
                stance=self._infer_stance(question),
            )
        )

    def run_sample(self, sample: Dict[str, Any], budget: Dict[str, Any]) -> Dict[str, Any]:
        extract_started = time.perf_counter()
        question = _extract_question(sample)
        instruction = _extract_instruction(sample)
        labels = self._candidate_labels(sample)
        chunks = self._extract_chunks(sample)
        chunk_extract_ms = (time.perf_counter() - extract_started) * 1000.0
        retrieval_started = time.perf_counter()
        scored = sorted(((c, self._score_chunk(question, c)) for c in chunks), key=lambda x: x[1], reverse=True)
        top_chunks = [c for c, _ in scored[: min(8, max(3, int(budget.get("max_retrieval_top_k", 6))))]]
        ranking_ms = (time.perf_counter() - retrieval_started) * 1000.0

        temporal_hits: List[_FREvidence] = []
        for idx, (chunk, score) in enumerate(scored[: len(top_chunks)]):
            tri_clock = self._build_tri_clock(chunk, idx, len(top_chunks))
            detail = TemporalEdgeScore.from_triclock(tri_clock, self._temporal_constraint)
            temporal_hits.append(_FREvidence(f"chunk_{idx+1}", chunk, score, tri_clock, detail))

        conflict_chunks = self._find_conflict_chunks(top_chunks)
        conflict_density = 0.0
        kg_started = time.perf_counter()
        kg_retrieval = self._kg_retrieve(question, top_chunks, budget, top_k=min(4, len(top_chunks)))
        kg_retrieval_ms = (time.perf_counter() - kg_started) * 1000.0
        memory_ms = 0.0
        if chunks and self._memory_ingest_enabled:
            memory_started = time.perf_counter()
            sample_ref = str(sample.get("id") or sample.get("sample_id") or "anon")
            if sample_ref not in self._ingested_samples:
                self._memory_ingest(sample, top_chunks, question)
                self._ingested_samples.add(sample_ref)
            retrieval = self._memory.retrieve(question, top_k=min(4, len(top_chunks)))
            conflict_density = max((item.get("conflict_density", 0.0) for item in retrieval), default=0.0)
            if kg_retrieval:
                conflict_density = max(conflict_density, max((item.get("conflict_density", 0.0) for item in kg_retrieval), default=0.0))
            memory_ms = (time.perf_counter() - memory_started) * 1000.0

        temporal_valid = True
        temporal_issues: List[str] = []
        prev_event: Optional[datetime] = None
        for evidence in temporal_hits:
            ok, reason = self._temporal_constraint.validate_edge(evidence.tri_clock)
            if not ok:
                temporal_valid = False
                temporal_issues.append(reason)
            if prev_event is not None:
                step_ok, step_reason = self._temporal_constraint.validate_step_transition(prev_event, evidence.tri_clock.event_time)
                if not step_ok:
                    temporal_valid = False
                    temporal_issues.append(step_reason)
            prev_event = evidence.tri_clock.event_time

        retrieved_lines = [f"[{e.chunk_id}] {e.text}" for e in temporal_hits]
        conflict_lines = [f"- {c}" for c in conflict_chunks] if conflict_chunks else ["- none"]
        kg_lines = [f"[{item.get('proposition_id', 'kg')}] {item.get('sentence', '')}" for item in kg_retrieval[:4]] if kg_retrieval else ["- none"]
        temporal_lines = [
            f"[{e.chunk_id}] event={e.tri_clock.event_time.date().isoformat()} release={e.tri_clock.release_time.date().isoformat()} "
            f"effect={e.tri_clock.effect_start.date().isoformat()}~{e.tri_clock.effect_end.date().isoformat()} "
            f"score={e.score_detail.total:.2f}"
            for e in temporal_hits
        ] or ["- none"]
        allow_multi = _sample_allows_multi_label(sample)

        context_sections: List[Tuple[str, str]] = [
            ("Core Evidence", _as_bullets(retrieved_lines)),
            ("Conflict Resolution", _as_bullets(conflict_lines)),
            ("KG Retrieval", _as_bullets(kg_lines)),
            ("Temporal Alignment", _as_bullets(temporal_lines)),
        ]
        if instruction:
            context_sections.append(("Instruction", instruction))

        prompt = [
            "You are a financial reasoning model.",
            (
                "Use the evidence sections to produce one or more final option labels, concatenated in option order with no separators."
                if allow_multi
                else "Use the evidence sections to produce exactly one final label."
            ),
            f"Question:\n{question}",
            *[f"{title}:\n{content}" for title, content in context_sections],
            (
                "Rules:\n- prefer temporally legal evidence\n- if conflict exists, prefer the most source-consistent synthesis\n- output exactly three tags and nothing else: <think>...</think><answer>...</answer><confidence>80</confidence>\n- inside <answer>, return only allowed option labels concatenated in option order with no separators\n- inside <confidence>, return digits only: an integer from 0 to 100 estimating the probability that the answer is correct; never leave it empty"
                if allow_multi
                else "Rules:\n- prefer temporally legal evidence\n- if conflict exists, prefer the most source-consistent synthesis\n- output exactly three tags and nothing else: <think>...</think><answer>...</answer><confidence>80</confidence>\n- inside <answer>, return only one label\n- inside <confidence>, return digits only: an integer from 0 to 100 estimating the probability that the answer is correct; never leave it empty"
            ),
            (f"Allowed labels: {', '.join(labels)}" if labels else ""),
            "Keep <think> brief, place the final benchmark answer in <answer>, and always include a non-empty numeric <confidence> tag.",
        ]
        prompt_text = "\n\n".join([p for p in prompt if p])
        prompt_build_ms = 0.0
        prompt_build_started = time.perf_counter()
        llm_budget = dict(budget)
        llm_budget["max_output_tokens"] = int(budget.get("max_output_tokens", 8192))
        llm_budget["temperature"] = float(budget.get("temperature", 0.0))
        llm_budget["top_p"] = float(budget.get("top_p", 1.0))
        user_content = _multimodal_message_content(prompt_text, sample, llm_budget)
        prompt_build_ms = (time.perf_counter() - prompt_build_started) * 1000.0

        llm_out = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a finance reasoning model for a publication-grade benchmark. "
                        + (
                            "Use only the supplied evidence. Return one or more labels from the allowed set, concatenated in option order with no separators."
                            if allow_multi
                            else "Use only the supplied evidence. Return a single label from the allowed set."
                        )
                        + " Always include <think>, <answer>, and a non-empty numeric <confidence> tag."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            budget=llm_budget,
        )
        llm_pred = self._finalize_prediction(sample, str(llm_out.get("content", "")), top_chunks)
        raw_model_text = str(llm_out.get("content", "") or "").strip()
        confidence_llm_out: Dict[str, Any] = {}
        if os.getenv("REQUIRE_CONFIDENCE_TAG", "").strip().lower() in {"1", "true", "yes", "on"} and not re.search(
            r"<confidence>\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*</confidence>",
            raw_model_text,
            flags=re.I | re.S,
        ):
            confidence_budget = dict(llm_budget)
            confidence_budget["max_output_tokens"] = min(32, int(confidence_budget.get("max_output_tokens", 32) or 32))
            confidence_llm_out = self.llm.chat(
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
            raw_model_text = raw_model_text + "\n" + str(confidence_llm_out.get("content", "") or "").strip()
        if temporal_hits and conflict_chunks and len(labels) > 1:
            best = max(temporal_hits, key=lambda e: e.score_detail.total + e.score)
            if best.score < 0.25:
                llm_pred = labels[min(1, len(labels) - 1)]
        if temporal_hits and not conflict_chunks and len(labels) > 1:
            best = max(temporal_hits, key=lambda e: e.score_detail.total)
            if best.score_detail.total < 0.3:
                llm_pred = labels[0]

        confidence = 0.55 + min(0.20, sum(e.score_detail.total for e in temporal_hits) / max(1, len(temporal_hits)) * 0.1)
        if not temporal_valid:
            confidence -= 0.08
        if conflict_chunks:
            confidence -= 0.04
        if conflict_density > 0.4:
            confidence -= 0.04
        confidence = max(0.35, min(0.9, confidence))

        extras = {
            "method_impl": "real_fr_ci",
            "raw_model_output": _ensure_nonempty_answer_tagged(raw_model_text),
            "retrieved_chunks": len(top_chunks),
            "kg_retrieved_chunks": len(kg_retrieval),
            "conflict_chunks": len(conflict_chunks),
            "conflict_density": conflict_density,
            "temporal_issues": temporal_issues,
            "temporal_scores": [e.score_detail.total for e in temporal_hits],
            "temporal_cues": [e.chunk_id for e in temporal_hits],
            "prompt_preview": prompt_text[:1200],
            "memory_health": self._memory.health(),
        }
        if confidence_llm_out:
            extras["confidence_retry_raw_model_output"] = str(confidence_llm_out.get("content", "") or "")
            extras["confidence_retry_used"] = True
        extras = attach_stage_timings(
            extras,
            {
                "rag_chunk_extract_ms": chunk_extract_ms,
                "rag_rank_ms": ranking_ms,
                "rag_kg_retrieval_ms": kg_retrieval_ms,
                "rag_memory_ms": memory_ms,
                "rag_prompt_build_ms": prompt_build_ms,
                "llm_generate_ms": float(llm_out.get("latency_ms", 0.0) or 0.0),
            },
        )

        return {
            "prediction": llm_pred,
            "confidence": confidence,
            "evidence_ids": [e.chunk_id for e in temporal_hits],
            "temporal_valid": temporal_valid,
            "violation": conflict_density > 0.5,
            "steps_used": 2,
            "top_k_used": len(top_chunks),
            "prompt_tokens": int(llm_out.get("prompt_tokens", max(1, len(prompt_text) // 4)))
            + int(confidence_llm_out.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(llm_out.get("completion_tokens", 1))
            + int(confidence_llm_out.get("completion_tokens", 0) or 0),
            "total_tokens": int(llm_out.get("total_tokens", max(1, len(prompt_text) // 4) + 1))
            + int(confidence_llm_out.get("total_tokens", 0) or 0),
            "extras": extras,
        }
