"""Small state machines for generation budgets and recovery routing."""

from dataclasses import dataclass, field
from enum import Enum

MAX_VARIANT_ATTEMPTS = 3
"""Maximum model outputs evaluated by the ordinary quality loop per chart."""

MAX_CRASH_ATTEMPTS = 3
"""Maximum output-less process failures tolerated per chart."""

MAX_TOTAL_ATTEMPTS = MAX_VARIANT_ATTEMPTS + MAX_CRASH_ATTEMPTS
"""Hard ceiling for ordinary quality and crash attempts combined."""


@dataclass(slots=True)
class AdditionalInferenceBudget:
    """Song-wide capacity for explicitly admitted recovery inference."""

    limit: int = 1
    used: int = 0

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("limit must be non-negative")
        if self.used < 0 or self.used > self.limit:
            raise ValueError("used must be between zero and limit")

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def consume(self) -> bool:
        if self.remaining <= 0:
            return False
        self.used += 1
        return True


@dataclass(slots=True)
class AttemptBudgetState:
    max_quality_attempts: int
    max_crash_attempts: int
    max_total_attempts: int
    next_attempt: int = 1
    quality_attempts: int = 0
    crash_attempts: int = 0
    attempted_seeds: list[int] = field(default_factory=list)

    @property
    def can_attempt(self) -> bool:
        return (
            self.quality_attempts < self.max_quality_attempts
            and self.crash_attempts < self.max_crash_attempts
            and self.next_attempt <= self.max_total_attempts
        )

    def reserve_attempt(self, *, seed: int) -> int:
        if not self.can_attempt:
            raise RuntimeError("attempt budget exhausted")
        return self._reserve(seed=seed)

    def reserve_additional_attempt(self, *, seed: int) -> int:
        """Number a separately authorized recovery without reusing primary limits."""
        return self._reserve(seed=seed)

    def _reserve(self, *, seed: int) -> int:
        attempt = self.next_attempt
        self.next_attempt += 1
        self.attempted_seeds.append(seed)
        return attempt

    def record_quality_attempt(self) -> None:
        self.quality_attempts += 1

    def record_crash_attempt(self) -> None:
        self.crash_attempts += 1


class RecoveryKind(str, Enum):
    PARTIAL_REMAP = "PARTIAL_REMAP"
    INTRO = "INTRO"
    TIMING_FAMILY = "TIMING_FAMILY"


@dataclass(slots=True)
class RecoveryRouterState:
    attempted: set[RecoveryKind] = field(default_factory=set)

    def was_attempted(self, kind: RecoveryKind) -> bool:
        return kind in self.attempted

    def claim(self, kind: RecoveryKind) -> bool:
        if self.was_attempted(kind):
            return False
        self.attempted.add(kind)
        return True
