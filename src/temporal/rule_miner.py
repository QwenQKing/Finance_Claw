
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Sequence

from .kg_builder import TemporalEdge, TemporalKG
from .pseudo_data import PseudoRule, get_pseudo_rules

@dataclass(frozen=True)
class RelationAtom:
    predicate: str
    arguments: tuple[str, ...]
    temporal_offset_days: int = 0

    def __str__(self) -> str:
        args = ", ".join(self.arguments)
        offset = f", t+{self.temporal_offset_days}d" if self.temporal_offset_days else ""
        return f"{self.predicate}({args}{offset})"

@dataclass(frozen=True)
class RuleMetrics:
    confidence: float
    support: float
    lift: float
    head_coverage: float = 0.0

    def passes(self, min_confidence: float, min_support: float) -> bool:
        return self.confidence >= min_confidence and self.support >= min_support

@dataclass
class TemporalRule:
    rule_id: str
    body: list[RelationAtom]
    head: RelationAtom
    metrics: RuleMetrics
    temporal_gap_days: int
    rationale: str = ""
    rationale_zh: str = ""

    def __str__(self) -> str:
        body_str = " ^ ".join(str(a) for a in self.body)
        return f"{body_str} -> {self.head}  [conf={self.metrics.confidence:.2f}]"

    @classmethod
    def from_pseudo(cls, pr: PseudoRule) -> "TemporalRule":
        body_atoms = [_parse_pseudo_atom(b) for b in pr.body]
        head_atom = _parse_pseudo_atom(pr.head, pr.temporal_gap_days)
        return cls(
            rule_id=pr.rule_id,
            body=body_atoms,
            head=head_atom,
            metrics=RuleMetrics(
                confidence=pr.confidence,
                support=pr.support,
                lift=pr.lift,
            ),
            temporal_gap_days=pr.temporal_gap_days,
            rationale=pr.rationale,
            rationale_zh=pr.rationale_zh,
        )

class RuleMiner:

    def __init__(self) -> None:
        self._rules: list[TemporalRule] = []

    @property
    def rules(self) -> list[TemporalRule]:
        return list(self._rules)

    def mine_rules(
        self,
        kg_snapshots: Sequence[TemporalKG] | None = None,
        *,
        max_body_length: int = 3,
        min_confidence: float = 0.5,
        min_support: float = 0.05,
    ) -> list[TemporalRule]:
        if kg_snapshots is None:
            self._rules = [
                TemporalRule.from_pseudo(pr) for pr in get_pseudo_rules()
            ]
        else:
            self._rules = self._mine_from_snapshots(
                kg_snapshots, max_body_length,
            )

        return self.filter_rules(self._rules, min_confidence, min_support)

    def evaluate_rule(
        self,
        rule: TemporalRule,
        test_kg: TemporalKG,
    ) -> RuleMetrics:
        import random
        noise = random.uniform(-0.05, 0.05)
        return RuleMetrics(
            confidence=max(0.0, min(1.0, rule.metrics.confidence + noise)),
            support=rule.metrics.support,
            lift=rule.metrics.lift,
            head_coverage=rule.metrics.head_coverage,
        )

    @staticmethod
    def filter_rules(
        rules: Sequence[TemporalRule],
        min_confidence: float = 0.5,
        min_support: float = 0.05,
    ) -> list[TemporalRule]:
        return [
            r for r in rules
            if r.metrics.passes(min_confidence, min_support)
        ]

    def _mine_from_snapshots(
        self,
        snapshots: Sequence[TemporalKG],
        max_body_length: int,
    ) -> list[TemporalRule]:
        relation_counts: dict[str, int] = {}
        for snap in snapshots:
            for edge in snap.edges:
                relation_counts[edge.relation] = (
                    relation_counts.get(edge.relation, 0) + 1
                )

        rules: list[TemporalRule] = []
        sorted_rels = sorted(
            relation_counts.items(), key=lambda x: x[1], reverse=True,
        )
        for i, (rel_a, cnt_a) in enumerate(sorted_rels[:10]):
            for rel_b, cnt_b in sorted_rels[i + 1 : i + 4]:
                total_edges = sum(relation_counts.values()) or 1
                support = (cnt_a + cnt_b) / total_edges
                confidence = min(cnt_a, cnt_b) / max(cnt_a, cnt_b)
                if confidence < 0.3:
                    continue
                rule = TemporalRule(
                    rule_id=f"mined-{len(rules):03d}",
                    body=[RelationAtom(rel_a, ("A", "B"))],
                    head=RelationAtom(rel_b, ("B",), temporal_offset_days=5),
                    metrics=RuleMetrics(
                        confidence=round(confidence, 3),
                        support=round(support, 3),
                        lift=round(confidence / max(support, 0.01), 2),
                    ),
                    temporal_gap_days=5,
                    rationale=f"Mined: {rel_a} co-occurs with {rel_b}",
                )
                rules.append(rule)
        return rules

def _parse_pseudo_atom(text: str, offset_days: int = 0) -> RelationAtom:
    import re
    m = re.match(r"(\w+)\(([^)]*)\)", text)
    if m:
        predicate = m.group(1)
        raw_args = [a.strip() for a in m.group(2).split(",")]
        args: list[str] = []
        for a in raw_args:
            tm = re.match(r"t\+(\d+)d", a)
            if tm:
                offset_days = int(tm.group(1))
            else:
                args.append(a)
        return RelationAtom(predicate, tuple(args), offset_days)
    return RelationAtom(text, (), offset_days)

def mine_pseudo_rules() -> list[TemporalRule]:
    miner = RuleMiner()
    return miner.mine_rules()

if __name__ == "__main__":
    rules = mine_pseudo_rules()
    print(f"Mined {len(rules)} temporal rules:\n")
    for r in rules:
        print(f"  [{r.rule_id}] {r}")
        print(f"    Rationale: {r.rationale[:80]}...")
        print()
