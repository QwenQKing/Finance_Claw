from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

@dataclass
class RunnerOutput:
    sample_id: str
    method: str
    prediction: Any
    confidence: Optional[float]
    evidence_ids: List[str] = field(default_factory=list)
    temporal_valid: Optional[bool] = None
    violation: Optional[bool] = None
    latency_ms: float = 0.0
    steps_used: int = 0
    top_k_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RunManifest:
    protocol_version: str
    method: str
    seed: int
    config_hash: str
    git_commit: str

def to_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)
