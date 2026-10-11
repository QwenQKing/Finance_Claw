from __future__ import annotations

from dataclasses import dataclass

@dataclass
class ConstraintUpdate:
    reward: float
    violation: float
    lagrangian_lambda: float
    objective: float
    action: str
    smoothed_reward: float
    smoothed_violation: float
    cooldown_left: int

class ConstrainedUpdater:

    def __init__(
        self,
        target_violation: float = 0.02,
        beta: float = 0.1,
        ema_alpha: float = 0.4,
        cooldown_steps: int = 2,
    ) -> None:
        self.target_violation = target_violation
        self.beta = beta
        self.ema_alpha = ema_alpha
        self.cooldown_steps = cooldown_steps

        self._lambda = 0.5
        self._reward_ema = 0.0
        self._violation_ema = 0.0
        self._initialized = False
        self._cooldown_left = 0

    @property
    def lagrangian_lambda(self) -> float:
        return self._lambda

    def _ema(self, prev: float, val: float) -> float:
        return (1 - self.ema_alpha) * prev + self.ema_alpha * val

    def update_policy(self, reward: float, violation: float) -> ConstraintUpdate:
        if not self._initialized:
            self._reward_ema = reward
            self._violation_ema = violation
            self._initialized = True
        else:
            self._reward_ema = self._ema(self._reward_ema, reward)
            self._violation_ema = self._ema(self._violation_ema, violation)

        objective = self._reward_ema - self._lambda * self._violation_ema

        self._lambda = max(0.0, self._lambda + self.beta * (self._violation_ema - self.target_violation))

        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            action = "hold"
        else:
            if self._violation_ema > self.target_violation:
                action = "tighten"
                self._cooldown_left = self.cooldown_steps
            elif self._reward_ema < 0.5:
                action = "explore"
                self._cooldown_left = self.cooldown_steps
            else:
                action = "exploit"

        return ConstraintUpdate(
            reward=reward,
            violation=violation,
            lagrangian_lambda=round(self._lambda, 4),
            objective=round(objective, 4),
            action=action,
            smoothed_reward=round(self._reward_ema, 4),
            smoothed_violation=round(self._violation_ema, 4),
            cooldown_left=self._cooldown_left,
        )
