
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

FIELD_SEP = "<SEP>"

class FinancialEntityType(Enum):
    COMPANY = "company"
    INDICATOR = "indicator"
    SECTOR = "sector"
    PERSON = "person"
    REGULATION = "regulation"
    EVENT = "event"
    INSTRUMENT = "instrument"
    MARKET = "market"

@dataclass
class HypergraphEntity:
    entity_id: str
    name: str
    name_zh: str = ""
    entity_type: str = "company"
    confidence: float = 1.0
    first_seen: str = ""
    last_updated: str = ""
    source_sessions: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.first_seen:
            self.first_seen = datetime.now(timezone.utc).isoformat()
        if not self.last_updated:
            self.last_updated = self.first_seen

    def freshness(self, half_life_days: float = 30.0) -> float:
        try:
            updated = datetime.fromisoformat(self.last_updated)
            age_days = (datetime.now(timezone.utc) - updated).total_seconds() / 86400
            return 0.5 ** (age_days / half_life_days)
        except (ValueError, TypeError):
            return 0.5

@dataclass
class HypergraphProposition:

    proposition_id: str
    sentence: str
    sentence_zh: str = ""
    entity_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source_session: str = ""
    source_tool: str = ""
    timestamp: str = ""
    data_timestamp: str = ""
    grounding_level: str = "L1"

    valid_from: str = ""
    valid_to: str = ""
    superseded_by: str = ""
    validity_type: str = "generic"
    stance: str = "neutral"

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.valid_from:
            self.valid_from = self.data_timestamp or self.timestamp

    def freshness(self, half_life_days: float = 14.0) -> float:
        try:
            ts = datetime.fromisoformat(self.timestamp)
            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            return 0.5 ** (age_days / half_life_days)
        except (ValueError, TypeError):
            return 0.5

    def validity_state(self, as_of: Optional[datetime] = None) -> str:
        if as_of is None:
            as_of = datetime.now(timezone.utc)

        if self.superseded_by:
            return "superseded"

        if self.valid_to:
            try:
                end = datetime.fromisoformat(self.valid_to.replace("Z", "+00:00"))
                if as_of > end:
                    return "expired"
            except ValueError:
                pass

        return "active"

@dataclass
class ContradictionEdge:
    proposition_a: str
    proposition_b: str
    conflict_type: str = "directional"
    strength: float = 0.0
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

class FinancialMemoryHypergraph:

    def __init__(self, storage_dir: str | Path = "~/financeclaw/memory/hypergraph/") -> None:
        self._storage_dir = Path(storage_dir).expanduser()
        self._source_credibility: dict[str, float] = {
            "sec-edgar": 0.98,
            "exchange_filing": 0.95,
            "akshare": 0.90,
            "fred": 0.92,
            "reuters": 0.88,
            "caixin": 0.85,
            "news_sentiment": 0.72,
            "unknown": 0.65,
        }
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._entities_file = self._storage_dir / "entities.json"
        self._propositions_file = self._storage_dir / "propositions.json"

        self._entities: dict[str, HypergraphEntity] = {}
        self._propositions: dict[str, HypergraphProposition] = {}
        self._edges: dict[str, set[str]] = {}
        self._reverse_edges: dict[str, set[str]] = {}

        self._contradictions: list[ContradictionEdge] = []
        self._contradictions_by_prop: dict[str, list[ContradictionEdge]] = {}
        self._health_cache: Optional[dict[str, Any]] = None
        self._conflict_density_cache: dict[tuple[str, int], float] = {}
        self._conflict_density_prop_cache: dict[str, float] = {}

        self._load()

    def _load(self) -> None:
        if self._entities_file.exists():
            try:
                raw = self._entities_file.read_text(encoding="utf-8").strip()
                data = json.loads(raw) if raw else {}
                for eid, edata in data.items():
                    self._entities[eid] = HypergraphEntity(**edata)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Could not load entities: %s", e)

        if self._propositions_file.exists():
            try:
                raw = self._propositions_file.read_text(encoding="utf-8").strip()
                data = json.loads(raw) if raw else {}
                for pid, pdata in data.items():
                    prop = HypergraphProposition(**pdata)
                    self._propositions[pid] = prop
                    self._edges[pid] = set(prop.entity_ids)
                    for eid in prop.entity_ids:
                        self._reverse_edges.setdefault(eid, set()).add(pid)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Could not load propositions: %s", e)

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)

    def save(self) -> None:
        self._atomic_write_json(
            self._entities_file,
            {eid: asdict(e) for eid, e in self._entities.items()},
        )
        self._atomic_write_json(
            self._propositions_file,
            {pid: asdict(p) for pid, p in self._propositions.items()},
        )
        logger.info(
            "Hypergraph saved: %d entities, %d propositions",
            len(self._entities), len(self._propositions),
        )

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "entities": copy.deepcopy(self._entities),
            "propositions": copy.deepcopy(self._propositions),
            "edges": copy.deepcopy(self._edges),
            "reverse_edges": copy.deepcopy(self._reverse_edges),
            "contradictions": copy.deepcopy(self._contradictions),
            "contradictions_by_prop": copy.deepcopy(self._contradictions_by_prop),
            "health_cache": copy.deepcopy(self._health_cache),
            "conflict_density_cache": copy.deepcopy(self._conflict_density_cache),
            "conflict_density_prop_cache": copy.deepcopy(self._conflict_density_prop_cache),
        }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        self._entities = copy.deepcopy(snapshot.get("entities", {}))
        self._propositions = copy.deepcopy(snapshot.get("propositions", {}))
        self._edges = copy.deepcopy(snapshot.get("edges", {}))
        self._reverse_edges = copy.deepcopy(snapshot.get("reverse_edges", {}))
        self._contradictions = copy.deepcopy(snapshot.get("contradictions", []))
        self._contradictions_by_prop = copy.deepcopy(snapshot.get("contradictions_by_prop", {}))
        self._health_cache = copy.deepcopy(snapshot.get("health_cache"))
        self._conflict_density_cache = copy.deepcopy(snapshot.get("conflict_density_cache", {}))
        self._conflict_density_prop_cache = copy.deepcopy(snapshot.get("conflict_density_prop_cache", {}))

    def _invalidate_caches(self) -> None:
        self._health_cache = None
        self._conflict_density_cache.clear()
        self._conflict_density_prop_cache.clear()

    def upsert_entity(self, entity: HypergraphEntity) -> str:
        existing = self._entities.get(entity.entity_id)
        if existing:
            existing.last_updated = datetime.now(timezone.utc).isoformat()
            existing.confidence = max(existing.confidence, entity.confidence)
            for sid in entity.source_sessions:
                if sid not in existing.source_sessions:
                    existing.source_sessions.append(sid)
            for alias in entity.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
            existing.attributes.update(entity.attributes)
        else:
            self._entities[entity.entity_id] = entity

        return entity.entity_id

    def get_entity(self, entity_id: str) -> Optional[HypergraphEntity]:
        return self._entities.get(entity_id)

    def find_entity(self, name: str) -> Optional[HypergraphEntity]:
        name_lower = name.lower().strip()
        for e in self._entities.values():
            if (
                e.name.lower() == name_lower
                or e.name_zh == name
                or e.entity_id.lower() == name_lower
                or any(a.lower() == name_lower for a in e.aliases)
            ):
                return e
        return None

    def entities_by_type(self, entity_type: str) -> list[HypergraphEntity]:
        return [e for e in self._entities.values() if e.entity_type == entity_type]

    def add_proposition(self, proposition: HypergraphProposition) -> str:
        self._propositions[proposition.proposition_id] = proposition
        entity_set = set(proposition.entity_ids)
        self._edges[proposition.proposition_id] = entity_set
        for eid in entity_set:
            self._reverse_edges.setdefault(eid, set()).add(proposition.proposition_id)
        self._invalidate_caches()
        return proposition.proposition_id

    def get_proposition(self, prop_id: str) -> Optional[HypergraphProposition]:
        return self._propositions.get(prop_id)

    def propositions_for_entity(self, entity_id: str) -> list[HypergraphProposition]:
        prop_ids = self._reverse_edges.get(entity_id, set())
        return [self._propositions[pid] for pid in prop_ids if pid in self._propositions]

    def entities_for_proposition(self, prop_id: str) -> list[HypergraphEntity]:
        entity_ids = self._edges.get(prop_id, set())
        return [self._entities[eid] for eid in entity_ids if eid in self._entities]

    def ingest_findings(
        self,
        findings: list[dict[str, Any]],
        session_id: str = "",
    ) -> dict[str, int]:
        stats = {"entities_added": 0, "entities_updated": 0, "propositions_added": 0}

        for finding in findings:
            sentence = finding.get("sentence", "").strip()
            if not sentence:
                continue

            entity_ids: list[str] = []
            for ent_data in finding.get("entities", []):
                name = ent_data.get("name", "").strip()
                if not name:
                    continue
                eid = _entity_id(name)
                entity_ids.append(eid)

                existing = self._entities.get(eid)
                entity = HypergraphEntity(
                    entity_id=eid,
                    name=name,
                    name_zh=ent_data.get("name_zh", name),
                    entity_type=ent_data.get("type", "company"),
                    confidence=finding.get("confidence", 1.0),
                    source_sessions=[session_id] if session_id else [],
                    aliases=ent_data.get("aliases", []),
                )
                self.upsert_entity(entity)
                if existing:
                    stats["entities_updated"] += 1
                else:
                    stats["entities_added"] += 1

            prop = HypergraphProposition(
                proposition_id=f"prop_{uuid.uuid4().hex[:10]}",
                sentence=sentence,
                sentence_zh=finding.get("sentence_zh", ""),
                entity_ids=entity_ids,
                confidence=finding.get("confidence", 1.0),
                source_session=session_id,
                source_tool=finding.get("source_tool", ""),
                data_timestamp=finding.get("data_timestamp", ""),
                valid_from=finding.get("valid_from", finding.get("data_timestamp", "")),
                valid_to=finding.get("valid_to", ""),
                validity_type=finding.get("validity_type", "generic"),
                stance=self._infer_stance(sentence),
            )
            self.add_proposition(prop)
            self._update_contradictions_for(prop)
            stats["propositions_added"] += 1

        return stats

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        entity_type: Optional[str] = None,
        min_confidence: float = 0.0,
        mode: str = "rrf_conflict",
    ) -> list[dict[str, Any]]:
        mode = str(mode or "rrf_conflict").strip().lower()
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        rrf_k = 4

        entity_ranked: list[str] = []
        matched_entities = self._search_entities(query_lower, entity_type)
        for entity in matched_entities:
            for prop in self.propositions_for_entity(entity.entity_id):
                if prop.confidence >= min_confidence and prop.proposition_id not in entity_ranked:
                    entity_ranked.append(prop.proposition_id)

        prop_ranked: list[str] = []
        scored_props: list[tuple[float, str]] = []
        for pid, prop in self._propositions.items():
            if prop.confidence < min_confidence:
                continue
            prop_lower = prop.sentence.lower()
            overlap = sum(1 for t in query_terms if t in prop_lower)
            if overlap > 0:
                score = overlap * prop.freshness()
                scored_props.append((score, pid))

        scored_props.sort(reverse=True)
        prop_ranked = [pid for _, pid in scored_props]

        rrf_scores: dict[str, float] = {}
        if mode in {"entity", "entity_only", "entity-only"}:
            for rank, pid in enumerate(entity_ranked):
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rank + rrf_k)
        elif mode in {"keyword", "keyword_only", "keyword-only", "proposition", "prop"}:
            for rank, pid in enumerate(prop_ranked):
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rank + rrf_k)
        else:
            for rank, pid in enumerate(entity_ranked):
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rank + rrf_k)
            for rank, pid in enumerate(prop_ranked):
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rank + rrf_k)

        use_conflict_bonus = mode not in {"rrf", "rrf_fused", "rrf-fused", "entity", "entity_only", "entity-only", "keyword", "keyword_only", "keyword-only", "proposition", "prop"}
        reranked = sorted(
            rrf_scores,
            key=lambda pid: rrf_scores[pid] + (self._conflict_info_gain(pid) if use_conflict_bonus else 0.0),
            reverse=True,
        )[:top_k]

        results: list[dict[str, Any]] = []
        for pid in reranked:
            prop = self._propositions[pid]
            entities = self.entities_for_proposition(pid)
            conflict_density = self.conflict_density_for_proposition(pid)
            results.append({
                "proposition_id": pid,
                "sentence": prop.sentence,
                "sentence_zh": prop.sentence_zh,
                "confidence": prop.confidence,
                "freshness": prop.freshness(),
                "validity_state": prop.validity_state(),
                "stance": prop.stance,
                "conflict_density": conflict_density,
                "conflict_info_gain": self._conflict_info_gain(pid) if use_conflict_bonus else 0.0,
                "rrf_score": rrf_scores[pid],
                "retrieval_mode": mode,
                "entities": [{"name": e.name, "type": e.entity_type} for e in entities],
                "source_session": prop.source_session,
                "source_tool": prop.source_tool,
                "grounding_level": prop.grounding_level,
            })

        return results

    def _search_entities(
        self,
        query_lower: str,
        entity_type: Optional[str] = None,
    ) -> list[HypergraphEntity]:
        results: list[tuple[float, HypergraphEntity]] = []
        query_terms = set(query_lower.split())

        for entity in self._entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            names = [entity.name.lower(), entity.name_zh.lower()] + [a.lower() for a in entity.aliases]
            score = 0.0
            for name in names:
                if query_lower in name or name in query_lower:
                    score += 2.0
                else:
                    overlap = sum(1 for t in query_terms if t in name)
                    score += overlap * 0.5
            if score > 0:
                results.append((score * entity.freshness(), entity))

        results.sort(reverse=True, key=lambda x: x[0])
        return [e for _, e in results[:20]]

    def _infer_stance(self, sentence: str) -> str:
        text = sentence.lower()
        up_terms = (
            "up", "grow", "growth", "beat", "surge", "increase", "bull",
            "上涨", "增长", "提升", "改善", "利好",
        )
        down_terms = (
            "down", "decline", "drop", "fall", "slow", "burn", "bear",
            "下跌", "下降", "放缓", "恶化", "利空",
        )
        up = sum(1 for t in up_terms if t in text)
        down = sum(1 for t in down_terms if t in text)
        if up > down:
            return "bullish"
        if down > up:
            return "bearish"
        return "neutral"

    def _update_contradictions_for(self, new_prop: HypergraphProposition) -> None:
        candidate_ids: set[str] = set()
        for entity_id in new_prop.entity_ids:
            candidate_ids.update(self._reverse_edges.get(entity_id, set()))
        for pid in candidate_ids:
            if pid == new_prop.proposition_id:
                continue
            other = self._propositions.get(pid)
            if other is None:
                continue
            if not (set(other.entity_ids) & set(new_prop.entity_ids)):
                continue
            if new_prop.stance == "neutral" or other.stance == "neutral":
                continue
            if new_prop.stance == other.stance:
                continue

            strength = self._contradiction_strength(new_prop, other)
            edge = ContradictionEdge(
                proposition_a=new_prop.proposition_id,
                proposition_b=other.proposition_id,
                conflict_type="directional",
                strength=strength,
            )
            self._contradictions.append(edge)
            self._contradictions_by_prop.setdefault(new_prop.proposition_id, []).append(edge)
            self._contradictions_by_prop.setdefault(other.proposition_id, []).append(edge)
        self._invalidate_caches()

    def _contradiction_strength(
        self,
        a: HypergraphProposition,
        b: HypergraphProposition,
    ) -> float:
        conf = (a.confidence + b.confidence) / 2.0
        dt_days = 30.0
        try:
            ta = datetime.fromisoformat(a.timestamp.replace("Z", "+00:00"))
            tb = datetime.fromisoformat(b.timestamp.replace("Z", "+00:00"))
            dt_days = abs((ta - tb).total_seconds()) / 86400.0
        except ValueError:
            pass
        proximity = max(0.0, 1.0 - min(dt_days, 60.0) / 60.0)

        src_a = self._source_credibility.get((a.source_tool or "unknown").lower(), self._source_credibility["unknown"])
        src_b = self._source_credibility.get((b.source_tool or "unknown").lower(), self._source_credibility["unknown"])
        source_weight = (src_a + src_b) / 2.0

        score = 0.5 * conf + 0.3 * proximity + 0.2 * source_weight
        return round(max(0.0, min(1.0, score)), 3)

    def conflict_density(self, entity_id: str, window_days: int = 90) -> float:
        cache_key = (entity_id, int(window_days))
        cached = self._conflict_density_cache.get(cache_key)
        if cached is not None:
            return cached
        now = datetime.now(timezone.utc)
        related = self._reverse_edges.get(entity_id, set())
        if not related:
            return 0.0

        weighted_conflicts = 0.0
        relevant_props = 0
        for pid in related:
            prop = self._propositions.get(pid)
            if not prop:
                continue
            relevant_props += 1
            for ce in self._contradictions_by_prop.get(pid, []):
                try:
                    ts = datetime.fromisoformat(prop.timestamp.replace("Z", "+00:00"))
                    if (now - ts) > timedelta(days=window_days):
                        continue
                except ValueError:
                    pass
                weighted_conflicts += ce.strength

        if relevant_props == 0:
            return 0.0
        value = round(min(1.0, weighted_conflicts / max(1, relevant_props)), 3)
        self._conflict_density_cache[cache_key] = value
        return value

    def conflict_density_for_proposition(self, proposition_id: str) -> float:
        cached = self._conflict_density_prop_cache.get(proposition_id)
        if cached is not None:
            return cached
        prop = self._propositions.get(proposition_id)
        if not prop or not prop.entity_ids:
            return 0.0
        vals = [self.conflict_density(eid) for eid in prop.entity_ids]
        value = round(sum(vals) / len(vals), 3) if vals else 0.0
        self._conflict_density_prop_cache[proposition_id] = value
        return value

    def _conflict_info_gain(self, proposition_id: str) -> float:
        return 0.15 * self.conflict_density_for_proposition(proposition_id)

    def prune(
        self,
        min_freshness: float = 0.05,
        min_confidence: float = 0.1,
    ) -> dict[str, int]:
        stats = {"entities_pruned": 0, "propositions_pruned": 0}

        prune_props: list[str] = []
        for pid, prop in self._propositions.items():
            if prop.freshness() < min_freshness or prop.confidence < min_confidence:
                prune_props.append(pid)

        for pid in prune_props:
            prop = self._propositions.pop(pid)
            self._edges.pop(pid, None)
            for eid in prop.entity_ids:
                if eid in self._reverse_edges:
                    self._reverse_edges[eid].discard(pid)
            stats["propositions_pruned"] += 1

        prune_entities: list[str] = []
        for eid in list(self._entities.keys()):
            connected_props = self._reverse_edges.get(eid, set())
            if not connected_props:
                prune_entities.append(eid)

        for eid in prune_entities:
            self._entities.pop(eid, None)
            self._reverse_edges.pop(eid, None)
            stats["entities_pruned"] += 1

        if stats["entities_pruned"] or stats["propositions_pruned"]:
            logger.info(
                "Hypergraph pruned: %d entities, %d propositions removed",
                stats["entities_pruned"], stats["propositions_pruned"],
            )

        return stats

    def health(self) -> dict[str, Any]:
        if self._health_cache is not None:
            return copy.deepcopy(self._health_cache)

        total_entities = len(self._entities)
        total_propositions = len(self._propositions)

        isolated = [
            eid for eid in self._entities
            if not self._reverse_edges.get(eid)
        ]

        if total_entities > 0:
            degrees = [len(self._reverse_edges.get(eid, set())) for eid in self._entities]
            avg_degree = sum(degrees) / total_entities
        else:
            avg_degree = 0.0

        entity_degrees = [
            (self._entities[eid].name, len(self._reverse_edges.get(eid, set())))
            for eid in self._entities
        ]
        entity_degrees.sort(key=lambda x: x[1], reverse=True)
        top_entities = [{"name": n, "degree": d} for n, d in entity_degrees[:10]]

        type_dist: dict[str, int] = {}
        for e in self._entities.values():
            type_dist[e.entity_type] = type_dist.get(e.entity_type, 0) + 1

        fresh_props = sum(1 for p in self._propositions.values() if p.freshness() > 0.5)
        stale_props = total_propositions - fresh_props

        total_conflicts = len(self._contradictions)
        avg_conflict_strength = (
            round(sum(c.strength for c in self._contradictions) / total_conflicts, 3)
            if total_conflicts > 0 else 0.0
        )

        isolation_ratio = len(isolated) / max(1, total_entities)
        freshness_ratio = fresh_props / max(1, total_propositions)
        health_score = (1.0 - isolation_ratio) * 0.5 + freshness_ratio * 0.5

        health = {
            "total_entities": total_entities,
            "total_propositions": total_propositions,
            "isolated_entities": len(isolated),
            "avg_degree": round(avg_degree, 2),
            "top_entities": top_entities,
            "entity_type_distribution": type_dist,
            "fresh_propositions": fresh_props,
            "stale_propositions": stale_props,
            "total_conflicts": total_conflicts,
            "avg_conflict_strength": avg_conflict_strength,
            "health_score": round(health_score, 3),
        }
        self._health_cache = copy.deepcopy(health)
        return health

    def summary(self) -> str:
        h = self.health()
        return (
            f"MemoryHypergraph: {h['total_entities']} entities, "
            f"{h['total_propositions']} propositions, "
            f"health={h['health_score']:.2f}"
        )

def _entity_id(name: str) -> str:
    return name.upper().strip().replace(" ", "_")
