from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaselineAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def run_sample(self, sample: Dict[str, Any], budget: Dict[str, Any]) -> Dict[str, Any]:
                ...

    def warmup(self, budget: Dict[str, Any]) -> None:
                return None
