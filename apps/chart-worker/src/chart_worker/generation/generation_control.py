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
    """Song-wide capacity for explicitly admitted recovery inference.

    ``limit`` remains a defensive call-count ceiling.  Production also sets a
    song-equivalent ``work_limit_ms`` so several short suffix repairs can fit
    where a second full-song retry cannot.
    """

    limit: int = 1
    used: int = 0
    work_limit_ms: int | None = None
    used_work_ms: int = 0

    def __post_init__(self) -> None:
        if type(self.limit) is not int or self.limit < 0:
            raise ValueError("limit must be non-negative")
        if type(self.used) is not int or self.used < 0 or self.used > self.limit:
            raise ValueError("used must be between zero and limit")
        if self.work_limit_ms is not None and (
            type(self.work_limit_ms) is not int or self.work_limit_ms < 0
        ):
            raise ValueError("work_limit_ms must be non-negative")
        if type(self.used_work_ms) is not int or self.used_work_ms < 0:
            raise ValueError("used_work_ms must be non-negative")
        if self.work_limit_ms is None and self.used_work_ms != 0:
            raise ValueError("used_work_ms requires work_limit_ms")
        if (
            self.work_limit_ms is not None
            and self.used_work_ms > self.work_limit_ms
        ):
            raise ValueError("used_work_ms must not exceed work_limit_ms")

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    @property
    def remaining_work_ms(self) -> int | None:
        if self.work_limit_ms is None:
            return None
        return self.work_limit_ms - self.used_work_ms

    def consume(self, estimated_generation_ms: int | None = None) -> bool:
        if estimated_generation_ms is not None and (
            type(estimated_generation_ms) is not int
            or estimated_generation_ms < 0
        ):
            raise ValueError("estimated_generation_ms must be a non-negative int")
        if self.remaining <= 0:
            return False
        if self.work_limit_ms is not None:
            if estimated_generation_ms is None:
                raise ValueError(
                    "estimated_generation_ms is required for a work-limited budget"
                )
            assert self.remaining_work_ms is not None
            if estimated_generation_ms > self.remaining_work_ms:
                return False
            self.used_work_ms += estimated_generation_ms
        self.used += 1
        return True

    def to_report(self) -> dict[str, object]:
        return {
            "policyVersion": (
                "SONG_EQUIVALENT_WORK_V1"
                if self.work_limit_ms is not None
                else "CALL_COUNT_V1"
            ),
            "callLimit": self.limit,
            "callsUsed": self.used,
            "workLimitMs": self.work_limit_ms,
            "workUsedMs": self.used_work_ms,
            "workRemainingMs": self.remaining_work_ms,
        }


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
    DIFFICULTY_SHADOW = "DIFFICULTY_SHADOW"


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
