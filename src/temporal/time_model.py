
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

@dataclass(frozen=True)
class TriClock:

    event_time: dt.datetime
    release_time: dt.datetime
    effect_start: dt.datetime
    effect_end: dt.datetime

    def is_valid(self) -> bool:
        return self.effect_start <= self.effect_end

    def lag_days(self) -> int:
        return max(0, (self.release_time - self.event_time).days)

@dataclass(frozen=True)
class TemporalConstraint:

    max_release_lag_days: int = 30
    require_non_decreasing_event_time: bool = True

    def validate_edge(self, tri_clock: TriClock) -> tuple[bool, str]:
        if not tri_clock.is_valid():
            return False, "invalid_effect_window"
        if tri_clock.lag_days() > self.max_release_lag_days:
            return False, "release_lag_exceeded"
        return True, "ok"

    def validate_step_transition(
        self,
        prev_event_time: dt.datetime,
        curr_event_time: dt.datetime,
    ) -> tuple[bool, str]:
        if self.require_non_decreasing_event_time and curr_event_time < prev_event_time:
            return False, "event_time_reversed"
        return True, "ok"

@dataclass(frozen=True)
class TemporalEdgeScore:

    order_score: float
    window_score: float
    lag_penalty: float

    @property
    def legality(self) -> float:
        return min(1.0, self.order_score * self.window_score)

    @property
    def total(self) -> float:
        return max(0.0, self.legality - self.lag_penalty)

    @staticmethod
    def from_triclock(
        tri_clock: TriClock,
        constraint: TemporalConstraint,
    ) -> "TemporalEdgeScore":
        valid_window = tri_clock.is_valid()
        window_score = 1.0 if valid_window else 0.0

        order_score = 1.0

        lag = tri_clock.lag_days()
        lag_ratio = min(1.0, lag / max(1, constraint.max_release_lag_days))
        lag_penalty = round(lag_ratio * 0.3, 4)

        return TemporalEdgeScore(
            order_score=order_score,
            window_score=window_score,
            lag_penalty=lag_penalty,
        )
