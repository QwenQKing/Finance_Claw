from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

@dataclass
class RegimeSignal:
    regime_label: str
    shift_confidence: float
    trigger: str

class RegimeDetector:

    def __init__(
        self,
        confidence_drop_threshold: float = 0.12,
        hallucination_rise_threshold: float = 0.02,
        compliance_violation_rise_threshold: float = 0.05,
        latency_rise_threshold: float = 500.0,
    ) -> None:
        self.confidence_drop_threshold = confidence_drop_threshold
        self.hallucination_rise_threshold = hallucination_rise_threshold
        self.compliance_violation_rise_threshold = compliance_violation_rise_threshold
        self.latency_rise_threshold = latency_rise_threshold

    def detect_shift(self, recent_metrics: Iterable[dict]) -> RegimeSignal:
        rows = list(recent_metrics)
        if len(rows) < 6:
            return RegimeSignal("stable", 0.0, "insufficient_history")

        split = len(rows) // 2
        prev, curr = rows[:split], rows[split:]

        prev_conf = mean(r.get("confidence", 0.0) for r in prev)
        curr_conf = mean(r.get("confidence", 0.0) for r in curr)
        prev_hall = mean(r.get("hallucination_rate", 0.0) for r in prev)
        curr_hall = mean(r.get("hallucination_rate", 0.0) for r in curr)

        prev_compliance = mean((r.get("compliance_violations", 0) > 0) for r in prev)
        curr_compliance = mean((r.get("compliance_violations", 0) > 0) for r in curr)

        prev_latency = mean(r.get("total_latency_ms", 0) for r in prev)
        curr_latency = mean(r.get("total_latency_ms", 0) for r in curr)

        conf_drop = prev_conf - curr_conf
        hall_rise = curr_hall - prev_hall
        compliance_rise = curr_compliance - prev_compliance
        latency_rise = curr_latency - prev_latency

        score = 0.0
        triggers: list[str] = []

        if conf_drop >= self.confidence_drop_threshold:
            score += 0.4 * min(1.0, conf_drop / max(self.confidence_drop_threshold, 1e-6))
            triggers.append("confidence_drop")
        if hall_rise >= self.hallucination_rise_threshold:
            score += 0.25 * min(1.0, hall_rise / max(self.hallucination_rise_threshold, 1e-6))
            triggers.append("hallucination_rise")
        if compliance_rise >= self.compliance_violation_rise_threshold:
            score += 0.2 * min(1.0, compliance_rise / max(self.compliance_violation_rise_threshold, 1e-6))
            triggers.append("compliance_violation_rise")
        if latency_rise >= self.latency_rise_threshold:
            score += 0.15 * min(1.0, latency_rise / max(self.latency_rise_threshold, 1e-6))
            triggers.append("latency_rise")

        score = round(min(1.0, score), 3)
        if score >= 0.75:
            return RegimeSignal("stressed", score, "+".join(triggers) or "multi_signal")
        if score >= 0.35:
            return RegimeSignal("transition", score, "+".join(triggers) or "partial_signal")
        return RegimeSignal("stable", score, "no_shift")
