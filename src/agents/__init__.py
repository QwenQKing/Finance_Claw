
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from src.governance import ComplianceGate

logger = logging.getLogger(__name__)

class ResearchDepth(Enum):
    QUICK_CHECK = "quick_check"
    THEMATIC = "thematic"
    DEEP_DIVE = "deep_dive"
    SYSTEMATIC = "systematic"

@dataclass
class DepthThreshold:
    min_tool_calls: int
    min_sources: int
    min_analysis_dimensions: int
    required_checklist: list[str]
    early_exit_confidence: float
    escalation_confidence: float

    @property
    def allows_early_exit(self) -> bool:
        return self.early_exit_confidence < 1.0

DEPTH_THRESHOLDS: dict[ResearchDepth, DepthThreshold] = {
    ResearchDepth.QUICK_CHECK: DepthThreshold(
        min_tool_calls=3,
        min_sources=1,
        min_analysis_dimensions=1,
        required_checklist=[
            "data_fetched",
            "timestamp_annotated",
            "source_declared",
        ],
        early_exit_confidence=0.90,
        escalation_confidence=0.50,
    ),
    ResearchDepth.THEMATIC: DepthThreshold(
        min_tool_calls=15,
        min_sources=3,
        min_analysis_dimensions=3,
        required_checklist=[
            "multiple_databases_searched",
            "full_metadata_retrieved",
            "cross_referenced",
            "contradictions_checked",
        ],
        early_exit_confidence=0.85,
        escalation_confidence=0.40,
    ),
    ResearchDepth.DEEP_DIVE: DepthThreshold(
        min_tool_calls=30,
        min_sources=5,
        min_analysis_dimensions=5,
        required_checklist=[
            "peer_comparison_done",
            "macro_context_assessed",
            "risk_factors_evaluated",
            "primary_source_verified",
            "validation_phase_passed",
        ],
        early_exit_confidence=0.80,
        escalation_confidence=0.35,
    ),
    ResearchDepth.SYSTEMATIC: DepthThreshold(
        min_tool_calls=60,
        min_sources=8,
        min_analysis_dimensions=6,
        required_checklist=[
            "methodology_established",
            "multi_timeframe_covered",
            "scenario_analysis_done",
            "data_gaps_documented",
            "methodology_section_written",
            "critique_refine_completed",
        ],
        early_exit_confidence=0.75,
        escalation_confidence=0.30,
    ),
}

class AgentRole(Enum):
    COORDINATOR = "coordinator"
    DATA_AGENT = "data_agent"
    VALIDATOR = "validator"
    ANALYST = "analyst"
    SYNTHESIZER = "synthesizer"
    CRITIC = "critic"

@dataclass
class AgentHandoff:
    handoff_id: str = ""
    from_agent: str = ""
    to_agent: str = ""
    reason: str = ""
    context_keys: list[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.handoff_id:
            self.handoff_id = f"hoff_{uuid.uuid4().hex[:8]}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

@dataclass
class AgentTask:
    task_id: str = ""
    agent_role: str = ""
    description: str = ""
    status: str = "pending"
    result_summary: str = ""
    confidence: float = 0.0
    started_at: str = ""
    completed_at: str = ""

    def __post_init__(self) -> None:
        if not self.task_id:
            self.task_id = f"task_{uuid.uuid4().hex[:8]}"

@dataclass
class CritiqueResult:
    iteration: int
    critique: str
    gaps_identified: list[str] = field(default_factory=list)
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    refinements_made: list[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

@dataclass
class FinanceClawAgentConfig:
    timeout: int = 3600
    heartbeat_interval: int = 1800

    research_depth: ResearchDepth = ResearchDepth.THEMATIC
    allow_dynamic_adaptation: bool = True
    max_escalation_level: ResearchDepth = ResearchDepth.DEEP_DIVE

    enable_multi_agent: bool = True
    agent_roles: list[str] = field(default_factory=lambda: [
        AgentRole.DATA_AGENT.value,
        AgentRole.VALIDATOR.value,
        AgentRole.ANALYST.value,
        AgentRole.SYNTHESIZER.value,
    ])
    max_parallel_agents: int = 3

    critique_refine_rounds: int = 2
    min_critique_confidence_gain: float = 0.05

    @property
    def threshold(self) -> DepthThreshold:
        return DEPTH_THRESHOLDS[self.research_depth]

    @property
    def min_tool_calls(self) -> int:
        return self.threshold.min_tool_calls

    @property
    def min_sources(self) -> int:
        return self.threshold.min_sources

    system_prompt_files: list[str] = field(default_factory=lambda: ["FINANCE.md"])

    inject_portfolio_context: bool = True
    inject_recent_opinions: bool = True
    max_memory_context_tokens: int = 4000

    require_disclaimer: bool = True
    block_investment_advice: bool = True

    token_budget: int = 0
    api_call_budget: int = 0

    @classmethod
    def for_depth(cls, depth: ResearchDepth) -> "FinanceClawAgentConfig":
        return cls(research_depth=depth)

@dataclass
class ToolCallTracker:
    calls: list[dict[str, Any]] = field(default_factory=list)
    sources_used: set = field(default_factory=set)
    dimensions_covered: set = field(default_factory=set)
    tokens_used: int = 0
    total_latency_ms: int = 0

    DIMENSION_KEYWORDS: dict[str, list[str]] = field(default_factory=lambda: {
        "financial": ["revenue", "profit", "eps", "roe", "debt", "cash_flow", "ratio"],
        "industry": ["peer", "competitor", "market_share", "industry", "sector"],
        "macro": ["gdp", "cpi", "interest_rate", "unemployment", "pmi", "fred"],
        "risk": ["var", "volatility", "beta", "drawdown", "risk"],
        "valuation": ["dcf", "pe", "pb", "ev_ebitda", "target_price", "fair_value"],
        "sentiment": ["sentiment", "news", "social", "analyst_rating"],
    })

    def record_call(
        self,
        tool_name: str,
        source: str = "",
        tokens: int = 0,
        latency_ms: int = 0,
    ) -> None:
        self.calls.append({
            "tool": tool_name,
            "source": source,
            "tokens": tokens,
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if source:
            self.sources_used.add(source)
        self.tokens_used += tokens
        self.total_latency_ms += latency_ms

        tool_lower = tool_name.lower()
        for dim, keywords in self.DIMENSION_KEYWORDS.items():
            if any(kw in tool_lower for kw in keywords):
                self.dimensions_covered.add(dim)

    def add_dimension(self, dimension: str) -> None:
        self.dimensions_covered.add(dimension)

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    @property
    def total_sources(self) -> int:
        return len(self.sources_used)

    @property
    def total_dimensions(self) -> int:
        return len(self.dimensions_covered)

    def meets_threshold(self, config: FinanceClawAgentConfig) -> bool:
        t = config.threshold
        return (
            self.total_calls >= t.min_tool_calls
            and self.total_sources >= t.min_sources
            and self.total_dimensions >= t.min_analysis_dimensions
        )

    def should_early_exit(self, config: FinanceClawAgentConfig, confidence: float) -> bool:
        if not config.allow_dynamic_adaptation:
            return False
        t = config.threshold
        if self.total_calls < t.min_tool_calls or self.total_sources < t.min_sources:
            return False
        return confidence >= t.early_exit_confidence

    def should_escalate(self, config: FinanceClawAgentConfig, confidence: float) -> Optional[ResearchDepth]:
        if not config.allow_dynamic_adaptation:
            return None
        t = config.threshold
        if confidence > t.escalation_confidence:
            return None

        depth_order = [
            ResearchDepth.QUICK_CHECK,
            ResearchDepth.THEMATIC,
            ResearchDepth.DEEP_DIVE,
            ResearchDepth.SYSTEMATIC,
        ]
        current_idx = depth_order.index(config.research_depth)
        max_idx = depth_order.index(config.max_escalation_level)

        if current_idx < max_idx:
            next_depth = depth_order[current_idx + 1]
            logger.info(
                "Confidence %.2f below escalation threshold %.2f — "
                "recommending escalation from %s to %s",
                confidence, t.escalation_confidence,
                config.research_depth.value, next_depth.value,
            )
            return next_depth
        return None

    def get_deficit_report(self, config: FinanceClawAgentConfig) -> str:
        t = config.threshold
        lines: list[str] = []
        if self.total_calls < t.min_tool_calls:
            lines.append(
                f"Tool calls: {self.total_calls}/{t.min_tool_calls} "
                f"(need {t.min_tool_calls - self.total_calls} more)"
            )
        if self.total_sources < t.min_sources:
            lines.append(
                f"Data sources: {self.total_sources}/{t.min_sources} "
                f"(need {t.min_sources - self.total_sources} more)"
            )
        if self.total_dimensions < t.min_analysis_dimensions:
            missing = set(self.DIMENSION_KEYWORDS.keys()) - self.dimensions_covered
            lines.append(
                f"Analysis dimensions: {self.total_dimensions}/{t.min_analysis_dimensions} "
                f"(missing: {', '.join(sorted(missing)[:3])}...)"
            )
        if not lines:
            return "All thresholds met ✅"

        if self.tokens_used > 0:
            lines.append(f"Tokens used: ~{self.tokens_used:,}")
        if self.total_latency_ms > 0:
            lines.append(f"Elapsed: {self.total_latency_ms / 1000:.1f}s")

        return "\n".join(lines)

    def cost_summary(self) -> dict[str, Any]:
        return {
            "total_tool_calls": self.total_calls,
            "unique_sources": self.total_sources,
            "dimensions_covered": sorted(self.dimensions_covered),
            "tokens_used": self.tokens_used,
            "total_latency_ms": self.total_latency_ms,
            "sources_list": sorted(self.sources_used),
        }

class ResearchOrchestrator:

    def __init__(
        self,
        config: FinanceClawAgentConfig | None = None,
        *,
        temporal_engine: Any = None,
        memory_engine: Any = None,
        evolution_engine: Any = None,
        governance_gate: ComplianceGate | None = None,
    ) -> None:
        self.config = config or FinanceClawAgentConfig()
        self.session_id = f"session_{uuid.uuid4().hex[:10]}"
        self.tasks: list[AgentTask] = []
        self.handoffs: list[AgentHandoff] = []
        self.critiques: list[CritiqueResult] = []
        self._current_confidence: float = 0.0

        self.temporal_engine = temporal_engine
        self.memory_engine = memory_engine
        self.evolution_engine = evolution_engine
        self.governance_gate = governance_gate or ComplianceGate()

        self.temporal_validity_ratio: float = 0.0
        self.conflict_density: float = 0.0
        self.compliance_penalty_total: float = 0.0

        self._metrics_dir = Path(".financeclaw").expanduser()
        self._metrics_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_file = self._metrics_dir / "统一编排指标.jsonl"

    @property
    def confidence(self) -> float:
        return self._current_confidence

    def decompose(self, user_prompt: str, depth: ResearchDepth) -> list[AgentTask]:
        tasks = [
            AgentTask(
                agent_role=AgentRole.COORDINATOR.value,
                description="Plan research scope, identify data sources, create research outline",
            ),
            AgentTask(
                agent_role=AgentRole.DATA_AGENT.value,
                description="Fetch data from identified sources via MCP servers",
            ),
            AgentTask(
                agent_role=AgentRole.VALIDATOR.value,
                description="Run ETL pipeline: normalize units/currency, detect anomalies, cross-check sources",
            ),
            AgentTask(
                agent_role=AgentRole.ANALYST.value,
                description="Compute ratios, run models, generate analytical insights",
            ),
            AgentTask(
                agent_role=AgentRole.SYNTHESIZER.value,
                description="Integrate findings, generate report with citations and disclaimers",
            ),
        ]

        if depth in (ResearchDepth.DEEP_DIVE, ResearchDepth.SYSTEMATIC):
            for i in range(self.config.critique_refine_rounds):
                tasks.append(AgentTask(
                    agent_role=AgentRole.CRITIC.value,
                    description=f"Critique-refine iteration {i + 1}: identify gaps, challenge conclusions",
                ))

        self.tasks = tasks
        return tasks

    def record_handoff(
        self,
        from_role: AgentRole,
        to_role: AgentRole,
        reason: str,
        context_keys: list[str] | None = None,
    ) -> AgentHandoff:
        handoff = AgentHandoff(
            from_agent=from_role.value,
            to_agent=to_role.value,
            reason=reason,
            context_keys=context_keys or [],
        )
        self.handoffs.append(handoff)
        return handoff

    def record_critique(
        self,
        iteration: int,
        critique: str,
        gaps: list[str],
        confidence_before: float,
        confidence_after: float,
        refinements: list[str],
    ) -> CritiqueResult:
        result = CritiqueResult(
            iteration=iteration,
            critique=critique,
            gaps_identified=gaps,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            refinements_made=refinements,
        )
        self.critiques.append(result)
        self._current_confidence = confidence_after
        return result

    def should_continue_critique(self) -> bool:
        if not self.critiques:
            return True
        if len(self.critiques) >= self.config.critique_refine_rounds:
            return False
        last = self.critiques[-1]
        gain = last.confidence_after - last.confidence_before
        return gain >= self.config.min_critique_confidence_gain

    def run_unified_pipeline(self, output_text: str, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        context = context or {}
        out: dict[str, Any] = {
            "temporal": {},
            "memory": {},
            "evolution": {},
            "governance": {},
        }

        decision = self.governance_gate.evaluate(output_text, context=context)
        self.compliance_penalty_total += decision.soft_penalty
        out["governance"] = {
            "passed": decision.passed,
            "soft_penalty": decision.soft_penalty,
            "hard_fail_reasons": decision.hard_fail_reasons,
            "violated_rules": decision.violated_rules,
        }

        if self.temporal_engine and hasattr(self.temporal_engine, "temporal_validity_ratio"):
            self.temporal_validity_ratio = float(getattr(self.temporal_engine, "temporal_validity_ratio"))
            out["temporal"]["temporal_validity_ratio"] = self.temporal_validity_ratio

        if self.memory_engine and hasattr(self.memory_engine, "conflict_density"):
            try:
                entity_id = context.get("entity_id", "")
                if entity_id:
                    self.conflict_density = float(self.memory_engine.conflict_density(entity_id))
                out["memory"]["conflict_density"] = self.conflict_density
            except Exception:
                out["memory"]["conflict_density"] = self.conflict_density

        if self.evolution_engine and hasattr(self.evolution_engine, "run_evolution_cycle"):
            try:
                evo = self.evolution_engine.run_evolution_cycle()
                out["evolution"] = {
                    "regime_signal": evo.get("regime_signal", {}),
                    "constraint_update": evo.get("constraint_update", {}),
                }
            except Exception:
                out["evolution"] = {"status": "skipped"}

        return out

    def summary(self) -> dict[str, Any]:
        summary = {
            "session_id": self.session_id,
            "research_depth": self.config.research_depth.value,
            "multi_agent_enabled": self.config.enable_multi_agent,
            "total_tasks": len(self.tasks),
            "completed_tasks": sum(1 for t in self.tasks if t.status == "completed"),
            "total_handoffs": len(self.handoffs),
            "critique_rounds": len(self.critiques),
            "final_confidence": self._current_confidence,
            "temporal_validity_ratio": self.temporal_validity_ratio,
            "conflict_density": self.conflict_density,
            "compliance_penalty_total": self.compliance_penalty_total,
            "agent_roles_used": sorted(set(t.agent_role for t in self.tasks)),
            "handoff_log": [asdict(h) for h in self.handoffs],
        }

        self._persist_metrics(summary)
        return summary

    def _persist_metrics(self, summary: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **summary,
        }
        with open(self._metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
