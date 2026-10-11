
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

class RiskTier(Enum):
    LOW = "low"
    LIMITED = "limited"
    HIGH = "high"
    UNACCEPTABLE = "unacceptable"

class Jurisdiction(Enum):
    CN = "cn"
    US = "us"
    HK = "hk"
    SG = "sg"
    EU = "eu"
    GLOBAL = "global"

@dataclass
class ToolCallRecord:
    call_id: str
    tool_name: str
    source: str
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    latency_ms: int = 0
    status: str = "success"
    error_message: str = ""
    output_summary: str = ""
    output_hash: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.call_id:
            self.call_id = f"tc_{uuid.uuid4().hex[:10]}"

@dataclass
class ValidationRecord:
    check_type: str
    target: str
    status: str
    expected: str = ""
    actual: str = ""
    details: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

@dataclass
class HumanOversightPoint:
    point_id: str
    description: str
    required: bool = True
    reviewed: bool = False
    reviewer: str = ""
    timestamp: str = ""
    notes: str = ""

@dataclass
class ResearchTrace:
    trace_id: str = ""
    session_id: str = ""
    created_at: str = ""

    user_prompt: str = ""
    research_depth: str = ""
    inferred_risk_tier: str = ""

    model_id: str = ""
    model_version: str = ""

    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    agent_roles: list[dict[str, str]] = field(default_factory=list)
    agent_handoffs: list[dict[str, Any]] = field(default_factory=list)

    validations: list[ValidationRecord] = field(default_factory=list)
    hallucination_check: dict[str, Any] = field(default_factory=dict)

    oversight_points: list[HumanOversightPoint] = field(default_factory=list)

    output_hash: str = ""
    output_risk_tier: str = ""
    disclaimers_applied: list[str] = field(default_factory=list)
    jurisdictions: list[str] = field(default_factory=list)

    total_tool_calls: int = 0
    total_sources: int = 0
    total_tokens_estimated: int = 0
    total_latency_ms: int = 0

    overall_confidence: float = 0.0
    confidence_breakdown: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trace_id:
            self.trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def record_tool_call(self, record: ToolCallRecord) -> None:
        self.tool_calls.append(record)
        self.total_tool_calls = len(self.tool_calls)
        sources = {tc.source for tc in self.tool_calls if tc.source}
        self.total_sources = len(sources)
        self.total_latency_ms += record.latency_ms

    def record_validation(self, record: ValidationRecord) -> None:
        self.validations.append(record)

    def record_agent_handoff(
        self,
        from_agent: str,
        to_agent: str,
        reason: str,
        context_summary: str = "",
    ) -> None:
        self.agent_handoffs.append({
            "from": from_agent,
            "to": to_agent,
            "reason": reason,
            "context_summary": context_summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def finalize(self, output_text: str) -> None:
        self.output_hash = hashlib.sha256(output_text.encode()).hexdigest()

        if self.validations:
            passes = sum(1 for v in self.validations if v.status == "pass")
            self.overall_confidence = passes / len(self.validations)
            self.confidence_breakdown["validation_pass_rate"] = self.overall_confidence

        if self.tool_calls:
            successes = sum(1 for tc in self.tool_calls if tc.status == "success")
            tc_rate = successes / len(self.tool_calls)
            self.confidence_breakdown["tool_success_rate"] = tc_rate
            self.overall_confidence = (
                self.overall_confidence * 0.6 + tc_rate * 0.4
            ) if self.validations else tc_rate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

class GovernanceLog:

    def __init__(self, log_dir: str | Path = "~/financeclaw/governance/") -> None:
        self._log_dir = Path(log_dir).expanduser()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "audit_trail.jsonl"

    @property
    def log_file(self) -> Path:
        return self._log_file

    def append(self, trace: ResearchTrace) -> None:
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(trace.to_json(indent=0).replace("\n", " ") + "\n")
        logger.info(
            "Governance trace %s appended (%d tool calls, confidence=%.2f)",
            trace.trace_id,
            trace.total_tool_calls,
            trace.overall_confidence,
        )

    def read_all(self) -> list[dict[str, Any]]:
        if not self._log_file.exists():
            return []
        traces = []
        with open(self._log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(json.loads(line))
        return traces

    def search(
        self,
        *,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        risk_tier: Optional[str] = None,
        since: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        results = self.read_all()
        if trace_id:
            results = [t for t in results if t.get("trace_id") == trace_id]
        if session_id:
            results = [t for t in results if t.get("session_id") == session_id]
        if risk_tier:
            results = [t for t in results if t.get("output_risk_tier") == risk_tier]
        if since:
            results = [t for t in results if t.get("created_at", "") >= since]
        return results

@dataclass
class ComplianceDecision:
    passed: bool
    hard_fail_reasons: list[str] = field(default_factory=list)
    soft_penalty: float = 0.0
    violated_rules: list[str] = field(default_factory=list)

class ComplianceGate:

    def __init__(self) -> None:
        self._hard_banned_terms = ["insider trading", "market manipulation", "pump and dump", "内幕交易", "操纵市场"]
        self._soft_risk_terms = ["guaranteed return", "稳赚", "稳赚不赔", "zero risk", "无风险"]

    def evaluate(self, content: str, context: Optional[dict[str, Any]] = None) -> ComplianceDecision:
        text = (content or "").lower()
        context = context or {}

        hard_fail_reasons: list[str] = []
        soft_penalty = 0.0
        violated_rules: list[str] = []

        for term in self._hard_banned_terms:
            if term in text:
                hard_fail_reasons.append(f"contains prohibited term: {term}")
                violated_rules.append("hard_prohibited_content")

        for term in self._soft_risk_terms:
            if term in text:
                soft_penalty += 0.1
                violated_rules.append("soft_exaggerated_claim")

        audience = str(context.get("audience", "")).lower()
        if audience in {"retail", "public"} and "target price" in text:
            soft_penalty += 0.1
            violated_rules.append("soft_retail_suitability")

        return ComplianceDecision(
            passed=len(hard_fail_reasons) == 0,
            hard_fail_reasons=hard_fail_reasons,
            soft_penalty=round(min(1.0, soft_penalty), 3),
            violated_rules=sorted(set(violated_rules)),
        )

JURISDICTION_DISCLAIMERS: dict[str, str] = {
    "cn": (
        "⚠️ 免责声明: 以上内容仅供研究参考，不构成任何投资建议。"
        "投资有风险，入市需谨慎。请在做出投资决策前咨询持牌金融顾问。"
        "本报告由 AI 辅助生成 (FinanceClaw)，符合中国证监会关于 AI 辅助研究的信息披露要求。"
    ),
    "us": (
        "Disclaimer: This content is for research purposes only and does not "
        "constitute investment advice. This report was generated with AI assistance "
        "(FinanceClaw). Per SEC 2026 Examination Priorities, AI-generated financial "
        "analysis must be supervised and its limitations disclosed. Consult a "
        "licensed financial advisor before making investment decisions."
    ),
    "hk": (
        "Disclaimer: This material is for informational purposes only and does not "
        "constitute an offer or solicitation under the Securities and Futures "
        "Ordinance (Cap. 571). AI-assisted analysis by FinanceClaw. Consult an SFC-"
        "licensed adviser before acting on this information."
    ),
    "sg": (
        "Disclaimer: This content does not constitute regulated financial advice "
        "under the Financial Advisers Act (Cap. 110) or Securities and Futures Act "
        "(Cap. 289) of Singapore. AI-assisted analysis by FinanceClaw, in accordance "
        "with MAS guidelines on the use of AI in financial services. Please consult "
        "a MAS-licensed financial adviser."
    ),
    "eu": (
        "Disclaimer: This AI-generated analysis is classified as a high-risk AI "
        "system output under the EU AI Act (Regulation 2024/1689). It does not "
        "constitute investment advice under MiFID II. A complete audit trail is "
        "available. Consult a regulated financial adviser in your jurisdiction."
    ),
    "global": (
        "⚠️ Disclaimer / 免责声明:\n"
        "This content is AI-generated (FinanceClaw) for research purposes only. "
        "It does not constitute investment advice in any jurisdiction. "
        "投资有风险，入市需谨慎。\n"
        "Consult a licensed financial adviser before making investment decisions."
    ),
}

class JurisdictionRegistry:

    def __init__(self) -> None:
        self._disclaimers = dict(JURISDICTION_DISCLAIMERS)

    def get_disclaimer(
        self,
        jurisdictions: list[str] | None = None,
    ) -> str:
        if not jurisdictions:
            jurisdictions = ["global"]

        parts: list[str] = []
        seen: set[str] = set()
        for j in jurisdictions:
            j_lower = j.lower()
            if j_lower in self._disclaimers and j_lower not in seen:
                parts.append(self._disclaimers[j_lower])
                seen.add(j_lower)

        if not parts:
            parts.append(self._disclaimers["global"])

        return "\n---\n".join(parts)

    def infer_jurisdictions(
        self,
        stock_codes: list[str] | None = None,
        markets: list[str] | None = None,
    ) -> list[str]:
        jurisdictions: set[str] = set()
        stock_codes = stock_codes or []
        markets = markets or []

        for code in stock_codes:
            code_upper = code.upper()
            if code_upper.endswith((".SH", ".SZ")):
                jurisdictions.add("cn")
            elif code_upper.endswith(".HK"):
                jurisdictions.add("hk")
            elif code_upper.endswith((".SI", ".SGX")):
                jurisdictions.add("sg")
            elif any(code_upper.endswith(s) for s in (".L", ".PA", ".DE", ".AS")):
                jurisdictions.add("eu")
            else:
                jurisdictions.add("us")

        market_map = {
            "a-share": "cn", "a股": "cn", "沪深": "cn", "sh": "cn", "sz": "cn",
            "hk": "hk", "港股": "hk", "hkex": "hk",
            "us": "us", "美股": "us", "nyse": "us", "nasdaq": "us",
            "sg": "sg", "sgx": "sg", "新加坡": "sg",
            "eu": "eu", "europe": "eu", "lse": "eu", "euronext": "eu",
        }
        for m in markets:
            m_lower = m.lower()
            if m_lower in market_map:
                jurisdictions.add(market_map[m_lower])

        return sorted(jurisdictions) if jurisdictions else ["global"]

    def classify_risk_tier(self, task_type: str) -> RiskTier:
        high_risk_keywords = [
            "predict", "forecast", "valuation", "recommend",
            "dcf", "target_price", "rating", "预测", "估值", "评级",
        ]
        unacceptable_keywords = [
            "insider", "manipulat", "front_run", "内幕", "操纵",
        ]

        task_lower = task_type.lower()
        if any(k in task_lower for k in unacceptable_keywords):
            return RiskTier.UNACCEPTABLE
        if any(k in task_lower for k in high_risk_keywords):
            return RiskTier.HIGH
        if any(k in task_lower for k in ["analysis", "research", "report", "分析", "研报"]):
            return RiskTier.LIMITED
        return RiskTier.LOW
