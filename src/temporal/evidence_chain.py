
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from .kg_builder import TemporalEdge, TemporalKG, TextSource
from .rule_miner import TemporalRule
from .time_model import TemporalConstraint

class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"

@dataclass
class ExplorationConfig:
    beam_width: int = 5
    max_depth: int = 4
    lookback_days: int = 30
    min_edge_confidence: float = 0.5
    min_chain_confidence: float = 0.4
    max_chains: int = 10

@dataclass(frozen=True)
class EvidenceStep:

    step_index: int
    source_entity: str
    relation: str
    target_entity: str
    timestamp: dt.datetime
    confidence: float
    text_source: TextSource
    temporal_valid: bool = True
    temporal_reason: str = "ok"
    order_score: float = 1.0
    window_score: float = 1.0
    lag_penalty: float = 0.0
    transition_reason: str = "ok"

    def __str__(self) -> str:
        temporal_tag = "valid" if self.temporal_valid else f"invalid:{self.temporal_reason}"
        return (
            f"[{self.step_index}] {self.source_entity} "
            f"--[{self.relation}]--> {self.target_entity} "
            f"@ {self.timestamp.date()} (conf={self.confidence:.2f}, "
            f"time={temporal_tag}, order={self.order_score:.2f}, window={self.window_score:.2f}, "
            f"lag_penalty={self.lag_penalty:.2f}, transition={self.transition_reason})"
        )

@dataclass
class EvidenceChain:
    chain_id: str
    steps: list[EvidenceStep] = field(default_factory=list)
    triggered_rules: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        if not self.steps:
            return 0.0
        c = 1.0
        for s in self.steps:
            c *= s.confidence
        return round(c, 4)

    @property
    def direction(self) -> Direction:
        if not self.steps:
            return Direction.NEUTRAL
        last_rel = self.steps[-1].relation.lower()
        if any(w in last_rel for w in ("up", "growth", "expansion", "buyback", "deal")):
            return Direction.UP
        if any(w in last_rel for w in ("down", "decline", "disruption", "burn", "slow")):
            return Direction.DOWN
        return Direction.NEUTRAL

    def summary(self) -> str:
        lines = [f"Chain {self.chain_id} ({self.direction.value}, conf={self.confidence:.3f}):"]
        for s in self.steps:
            lines.append(f"  {s}")
        if self.triggered_rules:
            lines.append(f"  Rules: {', '.join(self.triggered_rules)}")
        return "\n".join(lines)

@dataclass
class Verdict:
    stock: str
    prediction_date: dt.date
    direction: Direction
    aggregated_confidence: float
    chains: list[EvidenceChain] = field(default_factory=list)
    explanation: str = ""

    def summary(self) -> str:
        return (
            f"Verdict for {self.stock} on {self.prediction_date}: "
            f"{self.direction.value} (conf={self.aggregated_confidence:.2f})\n"
            f"  Based on {len(self.chains)} evidence chain(s).\n"
            f"  {self.explanation}"
        )

class EvidenceChainExplorer:

    def __init__(self) -> None:
        self._chains: list[EvidenceChain] = []
        self._constraint = TemporalConstraint(max_release_lag_days=45)

    def explore(
        self,
        stock_entity_id: str,
        prediction_date: dt.date,
        kg: TemporalKG,
        rules: Sequence[TemporalRule],
        config: ExplorationConfig | None = None,
    ) -> Verdict:
        if config is None:
            config = ExplorationConfig()

        cutoff = dt.datetime.combine(prediction_date, dt.time.min)
        lookback_start = cutoff - dt.timedelta(days=config.lookback_days)
        seed_edges = [
            e for e in kg.edges_for_entity(stock_entity_id)
            if lookback_start <= e.timestamp <= cutoff
            and e.confidence >= config.min_edge_confidence
            and e.temporal_valid
        ]

        chains: list[EvidenceChain] = []
        for i, seed in enumerate(seed_edges[: config.beam_width]):
            chain = self._expand_seed(
                seed, kg, rules, config, chain_idx=i,
            )
            if chain.confidence >= config.min_chain_confidence:
                chains.append(chain)

        chains.sort(key=lambda c: c.confidence, reverse=True)
        chains = chains[: config.max_chains]
        self._chains = chains

        return self.aggregate(stock_entity_id, prediction_date, chains)

    def aggregate(
        self,
        stock: str,
        prediction_date: dt.date,
        chains: Sequence[EvidenceChain],
    ) -> Verdict:
        if not chains:
            return Verdict(
                stock=stock,
                prediction_date=prediction_date,
                direction=Direction.NEUTRAL,
                aggregated_confidence=0.0,
                explanation="No evidence chains found.",
            )

        votes: dict[Direction, float] = {d: 0.0 for d in Direction}
        for c in chains:
            votes[c.direction] += c.confidence

        total = sum(votes.values()) or 1.0
        best_dir = max(votes, key=lambda d: votes[d])
        agg_conf = round(votes[best_dir] / total, 3)

        up_n = sum(1 for c in chains if c.direction == Direction.UP)
        down_n = sum(1 for c in chains if c.direction == Direction.DOWN)
        explanation = (
            f"{up_n} chain(s) support UP, {down_n} chain(s) support DOWN. "
            f"Weighted vote: {best_dir.value} with confidence {agg_conf:.2f}."
        )

        return Verdict(
            stock=stock,
            prediction_date=prediction_date,
            direction=best_dir,
            aggregated_confidence=agg_conf,
            chains=list(chains),
            explanation=explanation,
        )

    def format_report(self, verdict: Verdict) -> str:
        lines: list[str] = [
            "=" * 60,
            "TRACE Temporal Reasoning Report",
            "=" * 60,
            f"Stock:      {verdict.stock}",
            f"Date:       {verdict.prediction_date}",
            f"Prediction: {verdict.direction.value}",
            f"Confidence: {verdict.aggregated_confidence:.2f}",
            f"Chains:     {len(verdict.chains)}",
            "-" * 60,
            "Temporal audit legend: order/window in [0,1], lower lag_penalty is better",
            "-" * 60,
        ]
        for chain in verdict.chains:
            lines.append(chain.summary())
            lines.append("")
        lines.append("-" * 60)
        lines.append(verdict.explanation)
        lines.append("=" * 60)
        return "\n".join(lines)

    def _expand_seed(
        self,
        seed_edge: TemporalEdge,
        kg: TemporalKG,
        rules: Sequence[TemporalRule],
        config: ExplorationConfig,
        chain_idx: int,
    ) -> EvidenceChain:
        chain = EvidenceChain(chain_id=f"chain-{chain_idx:03d}")

        chain.steps.append(
            EvidenceStep(
                step_index=0,
                source_entity=seed_edge.source_entity,
                relation=seed_edge.relation,
                target_entity=seed_edge.target_entity,
                timestamp=seed_edge.timestamp,
                confidence=max(0.0, seed_edge.confidence - seed_edge.lag_penalty),
                text_source=seed_edge.provenance,
                temporal_valid=seed_edge.temporal_valid,
                temporal_reason=seed_edge.temporal_reason,
                order_score=seed_edge.order_score,
                window_score=seed_edge.window_score,
                lag_penalty=seed_edge.lag_penalty,
                transition_reason="seed",
            )
        )

        current_entity = seed_edge.target_entity
        current_time = seed_edge.timestamp
        for depth in range(1, config.max_depth):
            candidates = [
                e for e in kg.edges_for_entity(current_entity)
                if e.timestamp >= current_time
                and e.confidence >= config.min_edge_confidence
                and e.source_entity == current_entity
                and e.temporal_valid
            ]
            if not candidates:
                break

            scored: list[tuple[float, TemporalEdge, list[str]]] = []
            prev_event_time = current_time
            if chain.steps and chain.steps[-1].timestamp:
                prev_event_time = chain.steps[-1].timestamp

            for e in candidates:
                curr_event_time = e.tri_clock.event_time if e.tri_clock else e.timestamp
                ok_transition, transition_reason = self._constraint.validate_step_transition(
                    prev_event_time=prev_event_time,
                    curr_event_time=curr_event_time,
                )
                if not ok_transition:
                    continue

                rule_bonus, matched_rules = self._rule_score(
                    e, chain.steps, rules,
                )
                temporal_component = max(0.0, 1.0 - e.lag_penalty)
                score = (e.confidence * temporal_component) + rule_bonus
                scored.append((score, e, matched_rules))

            if not scored:
                break

            scored.sort(key=lambda t: t[0], reverse=True)
            best_score, best_edge, matched = scored[0]

            chain.steps.append(
                EvidenceStep(
                    step_index=depth,
                    source_entity=best_edge.source_entity,
                    relation=best_edge.relation,
                    target_entity=best_edge.target_entity,
                    timestamp=best_edge.timestamp,
                    confidence=max(0.0, best_edge.confidence - best_edge.lag_penalty),
                    text_source=best_edge.provenance,
                    temporal_valid=best_edge.temporal_valid,
                    temporal_reason=best_edge.temporal_reason,
                    order_score=best_edge.order_score,
                    window_score=best_edge.window_score,
                    lag_penalty=best_edge.lag_penalty,
                    transition_reason=transition_reason,
                )
            )
            chain.triggered_rules.extend(matched)
            current_entity = best_edge.target_entity
            current_time = best_edge.timestamp

        return chain

    @staticmethod
    def _rule_score(
        candidate_edge: TemporalEdge,
        existing_steps: Sequence[EvidenceStep],
        rules: Sequence[TemporalRule],
    ) -> tuple[float, list[str]]:
        bonus = 0.0
        matched: list[str] = []
        existing_rels = {s.relation for s in existing_steps}
        for rule in rules:
            body_preds = {a.predicate for a in rule.body}
            candidate_rels = existing_rels | {candidate_edge.relation}
            overlap = body_preds & candidate_rels
            if len(overlap) >= 1:
                bonus += rule.metrics.confidence * (len(overlap) / len(body_preds))
                matched.append(rule.rule_id)
        return bonus, matched

def get_pseudo_exploration() -> Verdict:
    from .kg_builder import build_pseudo_kg
    from .rule_miner import mine_pseudo_rules

    kg = build_pseudo_kg()
    rules = mine_pseudo_rules()

    explorer = EvidenceChainExplorer()
    verdict = explorer.explore(
        stock_entity_id="ent-byd",
        prediction_date=dt.date(2024, 12, 10),
        kg=kg,
        rules=rules,
        config=ExplorationConfig(
            beam_width=5,
            max_depth=3,
            lookback_days=60,
        ),
    )
    return verdict

if __name__ == "__main__":
    verdict = get_pseudo_exploration()
    explorer = EvidenceChainExplorer()
    print(explorer.format_report(verdict))
