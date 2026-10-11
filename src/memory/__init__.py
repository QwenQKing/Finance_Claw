
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class MemoryTier(Enum):
    PORTFOLIO = "portfolio"
    OPINION = "opinion"
    DATA = "data"
    METHODOLOGY = "methodology"

class DecayPolicy(Enum):
    NONE = "none"
    SLOW = "slow"
    FAST = "fast"

@dataclass
class FinancialMemoryEntry:
    id: str
    tier: MemoryTier
    content: str
    tags: list[str] = field(default_factory=list)
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    decay_policy: DecayPolicy = DecayPolicy.SLOW
    confidence: float = 1.0
    related_stocks: list[str] = field(default_factory=list)
    related_sectors: list[str] = field(default_factory=list)

    def relevance_score(self, query_time: Optional[datetime] = None) -> float:
        if query_time is None:
            query_time = datetime.now()

        age_days = (query_time - self.timestamp).days

        if self.decay_policy == DecayPolicy.NONE:
            decay = 1.0
        elif self.decay_policy == DecayPolicy.SLOW:
            half_life = 90
            decay = 0.5 ** (age_days / half_life)
        else:
            half_life = 7
            decay = 0.5 ** (age_days / half_life)

        return self.confidence * decay

@dataclass
class FinancialMemoryStore:
    entries: list[FinancialMemoryEntry] = field(default_factory=list)

    def add(self, entry: FinancialMemoryEntry) -> None:
        self.entries.append(entry)

    def search(
        self,
        query: str,
        tier: Optional[MemoryTier] = None,
        stock: Optional[str] = None,
        top_k: int = 10,
    ) -> list[FinancialMemoryEntry]:
        results = self.entries

        if tier:
            results = [e for e in results if e.tier == tier]

        if stock:
            results = [e for e in results if stock in e.related_stocks]

        if query:
            query_lower = query.lower()
            results = [
                e for e in results
                if query_lower in e.content.lower()
                or any(query_lower in tag.lower() for tag in e.tags)
            ]

        results.sort(key=lambda e: e.relevance_score(), reverse=True)

        return results[:top_k]

    def get_portfolio_context(self) -> list[FinancialMemoryEntry]:
        return self.search("", tier=MemoryTier.PORTFOLIO, top_k=100)
