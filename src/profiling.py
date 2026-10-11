from __future__ import annotations

from typing import Any, Dict, Iterable

def merge_stage_timings(*parts: Dict[str, float] | None) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    for part in parts:
        if not isinstance(part, dict):
            continue
        for key, value in part.items():
            try:
                numeric = float(value)
            except Exception:
                continue
            merged[key] = merged.get(key, 0.0) + numeric
    return merged

def attach_stage_timings(extras: Dict[str, Any] | None, stage_timings_ms: Dict[str, float] | None) -> Dict[str, Any]:
    merged = dict(extras or {})
    if not stage_timings_ms:
        return merged
    profiling = dict(merged.get("profiling") or {})
    profiling["stage_timings_ms"] = merge_stage_timings(
        profiling.get("stage_timings_ms") if isinstance(profiling.get("stage_timings_ms"), dict) else None,
        stage_timings_ms,
    )
    merged["profiling"] = profiling
    return merged

def summarize_stage_timings(
    extras_rows: Iterable[Dict[str, Any]],
    *,
    completed_samples: int,
    total_latency_ms: float,
) -> Dict[str, Any]:
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    maxes: Dict[str, float] = {}
    profiled_samples = 0

    for extras in extras_rows:
        profiling = (extras or {}).get("profiling") or {}
        stage_map = profiling.get("stage_timings_ms") if isinstance(profiling, dict) else None
        if not isinstance(stage_map, dict) or not stage_map:
            continue
        profiled_samples += 1
        for stage, raw_value in stage_map.items():
            try:
                value = float(raw_value)
            except Exception:
                continue
            totals[stage] = totals.get(stage, 0.0) + value
            counts[stage] = counts.get(stage, 0) + 1
            maxes[stage] = max(maxes.get(stage, value), value)

    if not totals:
        return {}

    avg_present = {stage: totals[stage] / max(1, counts[stage]) for stage in totals}
    avg_completed = {stage: totals[stage] / max(1, completed_samples) for stage in totals}
    share_total_latency = {
        stage: (totals[stage] / total_latency_ms) if total_latency_ms > 0 else 0.0 for stage in totals
    }
    dominant = sorted(totals.items(), key=lambda item: item[1], reverse=True)

    return {
        "profiled_samples": profiled_samples,
        "stage_totals_ms": totals,
        "stage_avg_ms_present": avg_present,
        "stage_avg_ms_completed": avg_completed,
        "stage_max_ms": maxes,
        "stage_presence_count": counts,
        "stage_share_of_total_latency": share_total_latency,
        "dominant_stages": [stage for stage, _ in dominant[:8]],
    }
