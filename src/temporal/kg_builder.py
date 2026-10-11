
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Sequence

from .time_model import TriClock, TemporalConstraint, TemporalEdgeScore
from .pseudo_data import (
    PseudoArticle,
    PseudoReport,
    get_pseudo_articles,
    get_pseudo_reports,
)

@dataclass(frozen=True)
class TextSource:
    source_id: str
    source_type: str
    title: str
    published_at: dt.datetime
    snippet: str = ""

@dataclass(frozen=True)
class TemporalEdge:

    source_entity: str
    target_entity: str
    relation: str
    timestamp: dt.datetime
    confidence: float
    provenance: TextSource
    tri_clock: TriClock | None = None
    temporal_valid: bool = True
    temporal_reason: str = "ok"
    order_score: float = 1.0
    window_score: float = 1.0
    lag_penalty: float = 0.0
    attributes: dict[str, str] = field(default_factory=dict)

@dataclass
class TemporalKG:
    entities: dict[str, dict[str, str]] = field(default_factory=dict)
    edges: list[TemporalEdge] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def entities_of_type(self, entity_type: str) -> dict[str, dict[str, str]]:
        return {
            eid: attrs
            for eid, attrs in self.entities.items()
            if attrs.get("type") == entity_type
        }

    def edges_for_entity(self, entity_id: str) -> list[TemporalEdge]:
        return [
            e for e in self.edges
            if e.source_entity == entity_id or e.target_entity == entity_id
        ]

    def edges_in_window(
        self,
        start: dt.datetime,
        end: dt.datetime,
    ) -> list[TemporalEdge]:
        return [e for e in self.edges if start <= e.timestamp <= end]

    def create_snapshot(self, as_of: dt.datetime) -> "TemporalKG":
        snapshot_edges = [e for e in self.edges if e.timestamp <= as_of]
        referenced: set[str] = set()
        for e in snapshot_edges:
            referenced.add(e.source_entity)
            referenced.add(e.target_entity)
        snapshot_entities = {
            eid: attrs
            for eid, attrs in self.entities.items()
            if eid in referenced
        }
        return TemporalKG(
            entities=snapshot_entities,
            edges=snapshot_edges,
            metadata={**self.metadata, "snapshot_as_of": as_of.isoformat()},
        )

    def __len__(self) -> int:
        return len(self.edges)

    def summary(self) -> str:
        return (
            f"TemporalKG: {len(self.entities)} entities, "
            f"{len(self.edges)} edges"
        )

_ENTITY_GAZETTEER: dict[str, dict[str, str]] = {
    "BYD":        {"id": "ent-byd",        "type": "COMPANY", "name_zh": "比亚迪"},
    "CATL":       {"id": "ent-catl",       "type": "COMPANY", "name_zh": "宁德时代"},
    "Alibaba":    {"id": "ent-alibaba",    "type": "COMPANY", "name_zh": "阿里巴巴"},
    "Tencent":    {"id": "ent-tencent",    "type": "COMPANY", "name_zh": "腾讯"},
    "PetroChina": {"id": "ent-petrochina", "type": "COMPANY", "name_zh": "中国石油"},
    "NIO":        {"id": "ent-nio",        "type": "COMPANY", "name_zh": "蔚来"},
    "BMW":        {"id": "ent-bmw",        "type": "COMPANY", "name_zh": "宝马"},
    "Guoxuan":    {"id": "ent-guoxuan",    "type": "COMPANY", "name_zh": "国轩高科"},
    "Tesla":      {"id": "ent-tesla",      "type": "COMPANY", "name_zh": "特斯拉"},
}

_RELATION_PATTERNS: list[tuple[str, str]] = [
    (r"surpass(?:es|ed)?",              "surpassed"),
    (r"supply\s+(?:chain\s+)?disrupt",  "supply_chain_disruption"),
    (r"supply\s+(?:deal|agreement)",    "signed_supply_deal_with"),
    (r"revenue\s+(?:grew|grows|growth)","revenue_growth"),
    (r"revenue\s+(?:slow|declin)",      "revenue_slowdown"),
    (r"invest(?:s|ed)?",                "invested_in"),
    (r"announc(?:es|ed)?.*(?:battery|breakthrough)", "technology_breakthrough"),
    (r"buyback|repurchase",             "share_buyback"),
    (r"cash\s+(?:burn|reserves?\s+fell)","cash_burn_acceleration"),
    (r"fire\b",                         "factory_incident"),
]

class FinancialKGBuilder:

    def __init__(self) -> None:
        self._kg = TemporalKG()
        self._constraint = TemporalConstraint(max_release_lag_days=45)

    @property
    def kg(self) -> TemporalKG:
        return self._kg

    def build_from_news(
        self,
        articles: Sequence[PseudoArticle] | None = None,
    ) -> TemporalKG:
        if articles is None:
            articles = get_pseudo_articles()

        for article in articles:
            source = TextSource(
                source_id=article.article_id,
                source_type="article",
                title=article.title,
                published_at=article.published_at,
                snippet=article.content[:120] + "...",
            )
            entities = self.extract_entities(article.content)
            relations = self.extract_relations(article.content, entities)

            for name, meta in entities.items():
                eid = meta["id"]
                if eid not in self._kg.entities:
                    self._kg.entities[eid] = {
                        "name": name,
                        "name_zh": meta.get("name_zh", ""),
                        "type": meta.get("type", "UNKNOWN"),
                    }

            entity_list = list(entities.values())
            for rel in relations:
                if len(entity_list) >= 2:
                    src = entity_list[0]
                    tgt = entity_list[1]
                else:
                    src = tgt = entity_list[0] if entity_list else None
                if src is None:
                    continue

                event_time = article.published_at - dt.timedelta(days=1)
                release_time = article.published_at
                effect_start = release_time
                effect_end = release_time + dt.timedelta(days=14)
                tri_clock = TriClock(
                    event_time=event_time,
                    release_time=release_time,
                    effect_start=effect_start,
                    effect_end=effect_end,
                )
                temporal_valid, temporal_reason = self._constraint.validate_edge(tri_clock)
                temporal_score = TemporalEdgeScore.from_triclock(tri_clock, self._constraint)

                edge = TemporalEdge(
                    source_entity=src["id"],
                    target_entity=tgt["id"],
                    relation=rel,
                    timestamp=article.published_at,
                    confidence=0.80,
                    provenance=source,
                    tri_clock=tri_clock,
                    temporal_valid=temporal_valid,
                    temporal_reason=temporal_reason,
                    order_score=temporal_score.order_score,
                    window_score=temporal_score.window_score,
                    lag_penalty=temporal_score.lag_penalty,
                )
                self._kg.edges.append(edge)

        return self._kg

    def build_from_reports(
        self,
        reports: Sequence[PseudoReport] | None = None,
    ) -> TemporalKG:
        if reports is None:
            reports = get_pseudo_reports()

        for report in reports:
            source = TextSource(
                source_id=report.report_id,
                source_type="report",
                title=f"{report.company} {report.fiscal_period}",
                published_at=report.published_at,
                snippet=report.summary[:120] + "...",
            )
            eid = self._company_to_entity_id(report.company)
            if eid not in self._kg.entities:
                self._kg.entities[eid] = {
                    "name": report.company,
                    "type": "COMPANY",
                    "ticker": report.ticker,
                }

            growth_tag = (
                "revenue_growth" if report.yoy_revenue_growth > 0
                else "revenue_decline"
            )
            event_time = report.published_at - dt.timedelta(days=7)
            release_time = report.published_at
            effect_start = release_time
            effect_end = release_time + dt.timedelta(days=30)
            tri_clock = TriClock(
                event_time=event_time,
                release_time=release_time,
                effect_start=effect_start,
                effect_end=effect_end,
            )
            temporal_valid, temporal_reason = self._constraint.validate_edge(tri_clock)
            temporal_score = TemporalEdgeScore.from_triclock(tri_clock, self._constraint)

            edge = TemporalEdge(
                source_entity=eid,
                target_entity=eid,
                relation=growth_tag,
                timestamp=report.published_at,
                confidence=0.95,
                provenance=source,
                tri_clock=tri_clock,
                temporal_valid=temporal_valid,
                temporal_reason=temporal_reason,
                order_score=temporal_score.order_score,
                window_score=temporal_score.window_score,
                lag_penalty=temporal_score.lag_penalty,
                attributes={
                    "revenue_billion_cny": str(report.revenue_billion_cny),
                    "yoy_growth": f"{report.yoy_revenue_growth:.1%}",
                },
            )
            self._kg.edges.append(edge)

        return self._kg

    def extract_entities(self, text: str) -> dict[str, dict[str, str]]:
        found: dict[str, dict[str, str]] = {}
        for name, meta in _ENTITY_GAZETTEER.items():
            if name.lower() in text.lower():
                found[name] = meta
        return found

    def extract_relations(
        self,
        text: str,
        entities: dict[str, dict[str, str]],
    ) -> list[str]:
        relations: list[str] = []
        text_lower = text.lower()
        for pattern, label in _RELATION_PATTERNS:
            if re.search(pattern, text_lower):
                relations.append(label)
        return relations

    def create_snapshot(self, date: dt.datetime) -> TemporalKG:
        return self._kg.create_snapshot(date)

    @staticmethod
    def _company_to_entity_id(company_name: str) -> str:
        name_lower = company_name.lower()
        for surface, meta in _ENTITY_GAZETTEER.items():
            if surface.lower() in name_lower:
                return meta["id"]
        raise ValueError(f"Unknown entity: {company_name}")

def build_pseudo_kg() -> TemporalKG:
    builder = FinancialKGBuilder()
    builder.build_from_news()
    builder.build_from_reports()
    return builder.kg

if __name__ == "__main__":
    kg = build_pseudo_kg()
    print(kg.summary())
    print(f"\nSample edges (first 5):")
    for edge in kg.edges[:5]:
        print(
            f"  {edge.source_entity} --[{edge.relation}]--> "
            f"{edge.target_entity}  @ {edge.timestamp.date()}  "
            f"(conf={edge.confidence:.2f})"
        )
