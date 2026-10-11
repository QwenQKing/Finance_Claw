
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .constrained_updater import ConstrainedUpdater
from .regime_detector import RegimeDetector

logger = logging.getLogger(__name__)

@dataclass
class EvolutionEvent:
    event_id: str = ""
    event_type: str = ""
    timestamp: str = ""
    description: str = ""
    trigger: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            self.event_id = f"evo_{uuid.uuid4().hex[:10]}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class SessionMetrics:

    session_id: str
    trace_id: str = ""
    research_depth: str = ""
    task_type: str = ""
    confidence: float = 0.0
    hallucination_rate: float = 0.0
    compliance_violations: int = 0
    tool_calls: int = 0
    sources_used: int = 0
    dimensions_covered: int = 0
    validation_pass_rate: float = 1.0
    total_latency_ms: int = 0
    tokens_used: int = 0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

@dataclass
class FinancialSkill:

    skill_id: str
    name: str
    valid_regimes: list[str] = field(default_factory=lambda: ["stable"])
    performance_by_regime: dict[str, float] = field(default_factory=dict)
    risk_of_violation: float = 0.0
    parent_skill: str = ""
    lineage: list[str] = field(default_factory=list)

class PerformanceTracker:

    def __init__(self, log_dir: str | Path = "~/financeclaw/evolution/") -> None:
        self._log_dir = Path(log_dir).expanduser()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_file = self._log_dir / "session_metrics.jsonl"
        self._history: list[SessionMetrics] = []
        self._load()

    def _load(self) -> None:
        if self._metrics_file.exists():
            for line in self._metrics_file.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    try:
                        data = json.loads(line)
                        self._history.append(SessionMetrics(**data))
                    except (json.JSONDecodeError, TypeError):
                        continue

    def record(self, metrics: SessionMetrics) -> None:
        self._history.append(metrics)
        with open(self._metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(metrics), ensure_ascii=False) + "\n")

    def get_task_type_stats(self) -> dict[str, dict[str, Any]]:
        by_type: dict[str, list[SessionMetrics]] = {}
        for m in self._history:
            task = m.task_type or "unknown"
            by_type.setdefault(task, []).append(m)

        stats: dict[str, dict[str, Any]] = {}
        for task_type, sessions in by_type.items():
            confs = [s.confidence for s in sessions]
            hall_rates = [s.hallucination_rate for s in sessions]
            stats[task_type] = {
                "session_count": len(sessions),
                "avg_confidence": sum(confs) / len(confs) if confs else 0,
                "avg_hallucination_rate": sum(hall_rates) / len(hall_rates) if hall_rates else 0,
                "min_confidence": min(confs) if confs else 0,
                "avg_tokens": sum(s.tokens_used for s in sessions) / len(sessions) if sessions else 0,
            }

        return stats

    def detect_underperformers(
        self,
        min_sessions: int = 3,
        confidence_threshold: float = 0.60,
        hallucination_threshold: float = 0.03,
    ) -> list[dict[str, Any]]:
        stats = self.get_task_type_stats()
        underperformers: list[dict[str, Any]] = []

        for task_type, s in stats.items():
            if s["session_count"] < min_sessions:
                continue

            reasons: list[str] = []
            if s["avg_confidence"] < confidence_threshold:
                reasons.append(f"low confidence ({s['avg_confidence']:.2f} < {confidence_threshold})")
            if s["avg_hallucination_rate"] > hallucination_threshold:
                reasons.append(f"high hallucination ({s['avg_hallucination_rate']:.2%} > {hallucination_threshold:.2%})")

            if reasons:
                underperformers.append({
                    "task_type": task_type,
                    "reason": "; ".join(reasons),
                    **s,
                })

        return underperformers

    @property
    def total_sessions(self) -> int:
        return len(self._history)

class SkillGapDetector:

    def __init__(self, tracker: PerformanceTracker) -> None:
        self._tracker = tracker

    def detect_gaps(self) -> list[dict[str, Any]]:
        gaps: list[dict[str, Any]] = []

        underperformers = self._tracker.detect_underperformers()
        for up in underperformers:
            severity = "high" if up.get("avg_confidence", 0) < 0.4 else "medium"
            gaps.append({
                "gap_type": "underperforming_skill",
                "task_type": up["task_type"],
                "severity": severity,
                "metrics": up,
                "recommendation": (
                    f"Skill for '{up['task_type']}' needs improvement. "
                    f"Consider: refine prompts, add data sources, or create specialized sub-skills."
                ),
            })

        stats = self._tracker.get_task_type_stats()
        known_skills = {
            "stock_price", "financial_statement", "macro_indicator", "news_sentiment",
            "ratio_calculation", "dcf_valuation", "risk_var", "peer_comparison",
            "trend_analysis", "portfolio_optimizer", "equity_research",
            "industry_analysis", "macro_weekly", "temporal_kg_builder",
            "rule_miner", "evidence_chain", "scenario_builder",
            "agent_persona_gen", "swarm_runner",
        }

        for task_type, s in stats.items():
            normalized = task_type.lower().replace(" ", "_").replace("-", "_")
            if normalized not in known_skills and s["session_count"] >= 5:
                gaps.append({
                    "gap_type": "missing_skill",
                    "task_type": task_type,
                    "severity": "medium",
                    "metrics": s,
                    "recommendation": (
                        f"'{task_type}' has been requested {s['session_count']} times "
                        f"but has no dedicated skill. Auto-generate a SKILL.md."
                    ),
                })

        return gaps

SKILL_TEMPLATE = """# {skill_name}

{when_to_use}

{workflow}

{data_sources}

{output_format}

{rules}

---
*Auto-generated by FinanceClaw Self-Evolution Engine*
*Trigger: {trigger}*
*Generated: {timestamp}*
*Based on {session_count} successful sessions*
"""

class SkillEvolver:

    def __init__(
        self,
        skills_dir: str | Path = "skills/",
        evolution_log_dir: str | Path = "~/financeclaw/evolution/",
    ) -> None:
        self._skills_dir = Path(skills_dir)
        self._log_dir = Path(evolution_log_dir).expanduser()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._evolution_log_file = self._log_dir / "evolution_log.jsonl"

    def generate_skill(
        self,
        gap: dict[str, Any],
        successful_patterns: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        task_type = gap.get("task_type", "unknown")
        skill_name = task_type.replace("_", "-").lower()
        skill_dir = self._skills_dir / f"fin-evolved/{skill_name}"

        workflow_steps: list[str] = ["1. Identify relevant data sources"]
        data_sources: list[str] = ["- AKShare (default)"]
        rules: list[str] = ["- Follow FINANCE.md §1 Zero Hallucination rule"]

        if successful_patterns:
            tools_seen: dict[str, int] = {}
            sources_seen: dict[str, int] = {}
            for pattern in successful_patterns:
                for tool in pattern.get("tools_used", []):
                    tools_seen[tool] = tools_seen.get(tool, 0) + 1
                for source in pattern.get("sources_used", []):
                    sources_seen[source] = sources_seen.get(source, 0) + 1

            if tools_seen:
                top_tools = sorted(tools_seen, key=tools_seen.get, reverse=True)[:5]
                for i, tool in enumerate(top_tools, 2):
                    workflow_steps.append(f"{i}. Use `{tool}` to gather data")
            if sources_seen:
                data_sources = [f"- {src}" for src in sorted(sources_seen, key=sources_seen.get, reverse=True)[:5]]

        workflow_steps.append(f"{len(workflow_steps) + 1}. Validate via ETL pipeline (§14)")
        workflow_steps.append(f"{len(workflow_steps) + 1}. Generate report with citations")

        content = SKILL_TEMPLATE.format(
            skill_name=task_type.replace("_", " ").title(),
            when_to_use=f"Use when the user requests {task_type.replace('_', ' ')} analysis.",
            workflow="\n".join(workflow_steps),
            data_sources="\n".join(data_sources),
            output_format="- Follow §9 output templates\n- Include data source citations (§1.1)",
            rules="\n".join(rules),
            trigger=gap.get("recommendation", "Performance gap detected"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_count=gap.get("metrics", {}).get("session_count", 0),
        )

        event = EvolutionEvent(
            event_type="skill_generated",
            description=f"Auto-generated skill: {skill_name}",
            trigger=gap.get("recommendation", ""),
            details={
                "task_type": task_type,
                "gap": gap,
                "skill_path": str(skill_dir / "SKILL.md"),
            },
        )
        self._log_event(event)

        return {
            "skill_name": skill_name,
            "skill_path": str(skill_dir / "SKILL.md"),
            "content": content,
            "event": event.to_dict(),
        }

    def _log_event(self, event: EvolutionEvent) -> None:
        with open(self._evolution_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

class EvolutionOrchestrator:

    def __init__(
        self,
        tracker: PerformanceTracker | None = None,
        evolver: SkillEvolver | None = None,
        hypergraph: Any = None,
        evolution_log_dir: str | Path = "~/financeclaw/evolution/",
    ) -> None:
        log_dir = Path(evolution_log_dir).expanduser()
        self.tracker = tracker or PerformanceTracker(log_dir=log_dir)
        self.gap_detector = SkillGapDetector(self.tracker)
        self.evolver = evolver or SkillEvolver(evolution_log_dir=log_dir)
        self.hypergraph = hypergraph
        self._log_dir = log_dir

        self.regime_detector = RegimeDetector()
        self.constrained_updater = ConstrainedUpdater()

    def run_evolution_cycle(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_sessions_analyzed": self.tracker.total_sessions,
            "regime_signal": {},
            "constraint_update": {},
            "gaps_detected": [],
            "skills_generated": [],
            "memory_pruned": {},
            "events": [],
        }

        recent = [asdict(m) for m in self.tracker._history[-10:]]
        regime = self.regime_detector.detect_shift(recent)
        results["regime_signal"] = {
            "label": regime.regime_label,
            "shift_confidence": regime.shift_confidence,
            "trigger": regime.trigger,
        }

        if recent:
            avg_conf = sum(r.get("confidence", 0.0) for r in recent) / len(recent)
            avg_viol = sum((r.get("compliance_violations", 0) > 0) for r in recent) / len(recent)
            update = self.constrained_updater.update_policy(reward=avg_conf, violation=avg_viol)
            results["constraint_update"] = asdict(update)
            results["events"].append(
                EvolutionEvent(
                    event_type="constraint_update",
                    description=f"RCNSL update action={update.action}, lambda={update.lagrangian_lambda}",
                    trigger=regime.trigger,
                    details=asdict(update),
                ).to_dict()
            )

        gaps = self.gap_detector.detect_gaps()
        results["gaps_detected"] = gaps

        for gap in gaps:
            if gap["gap_type"] == "missing_skill":
                skill = self.evolver.generate_skill(gap)
                results["skills_generated"].append(skill)
                results["events"].append(skill["event"])

        if self.hypergraph is not None:
            prune_stats = self.hypergraph.prune()
            results["memory_pruned"] = prune_stats
            if prune_stats.get("entities_pruned") or prune_stats.get("propositions_pruned"):
                event = EvolutionEvent(
                    event_type="memory_pruned",
                    description=(
                        f"Pruned {prune_stats.get('entities_pruned', 0)} entities, "
                        f"{prune_stats.get('propositions_pruned', 0)} propositions"
                    ),
                    trigger="Periodic evolution cycle",
                    details=prune_stats,
                )
                results["events"].append(event.to_dict())

        logger.info(
            "Evolution cycle complete: %d gaps, %d skills generated, %s pruned",
            len(gaps), len(results["skills_generated"]),
            results["memory_pruned"],
        )

        return results

    def record_session(self, metrics: SessionMetrics) -> None:
        self.tracker.record(metrics)

        if self.tracker.total_sessions % 10 == 0:
            logger.info("Auto-triggering evolution cycle (every 10 sessions)")
            self.run_evolution_cycle()

    def get_evolution_history(self, limit: int = 50) -> list[dict[str, Any]]:
        log_file = self._log_dir / "evolution_log.jsonl"
        if not log_file.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in log_file.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events[-limit:]
