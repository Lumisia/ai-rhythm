"""Pure prioritization for song-wide recovery inference.

The router does not inspect charts, mutate candidate state, or call a model.
Callers must first translate validated domain evidence into requests. Keeping
that boundary explicit prevents loop order from silently becoming policy.
"""

from dataclasses import dataclass
from enum import IntEnum

from chart_worker.generation.generation_control import RecoveryKind


class RecoveryPriority(IntEnum):
    """User-visible consequence if a recovery is not attempted."""

    ADVISORY = 100
    QUALITY_BLOCKING = 200
    COMPLETENESS_BLOCKING = 300


@dataclass(frozen=True, slots=True)
class RecoveryRequest:
    request_id: str
    kind: RecoveryKind
    key_mode: int
    difficulty: str
    priority: RecoveryPriority
    estimated_generation_ms: int
    reason: str

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.key_mode <= 0:
            raise ValueError("key_mode must be positive")
        if not self.difficulty:
            raise ValueError("difficulty must not be empty")
        if self.estimated_generation_ms < 0:
            raise ValueError("estimated_generation_ms must be non-negative")
        if not self.reason:
            raise ValueError("reason must not be empty")

    def to_report(self) -> dict[str, object]:
        return {
            "requestId": self.request_id,
            "kind": self.kind.value,
            "keyMode": self.key_mode,
            "difficulty": self.difficulty,
            "priority": self.priority.name,
            "estimatedGenerationMs": self.estimated_generation_ms,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    selected: tuple[RecoveryRequest, ...]
    deferred: tuple[RecoveryRequest, ...]
    available_generation_ms: int
    selected_generation_ms: int

    @property
    def remaining_generation_ms(self) -> int:
        return self.available_generation_ms - self.selected_generation_ms

    def to_report(self) -> dict[str, object]:
        return {
            "policyVersion": "RECOVERY_WORK_BUDGET_V2",
            "availableGenerationMs": self.available_generation_ms,
            "selectedGenerationMs": self.selected_generation_ms,
            "remainingGenerationMs": self.remaining_generation_ms,
            "selectedRequestIds": [request.request_id for request in self.selected],
            "deferredRequestIds": [request.request_id for request in self.deferred],
            "requests": [
                {
                    **request.to_report(),
                    "decision": (
                        "SELECTED"
                        if request in self.selected
                        else "DEFERRED_WORK_BUDGET"
                    ),
                }
                for request in (*self.selected, *self.deferred)
            ],
        }


_KIND_ORDER = {
    RecoveryKind.PARTIAL_REMAP: 0,
    RecoveryKind.INTRO: 1,
    RecoveryKind.TIMING_FAMILY: 2,
}


def _rank_key(request: RecoveryRequest) -> tuple[int, int, int, int, str, str]:
    return (
        -int(request.priority),
        request.estimated_generation_ms,
        _KIND_ORDER[request.kind],
        request.key_mode,
        request.difficulty,
        request.request_id,
    )


def plan_recoveries(
    requests: tuple[RecoveryRequest, ...],
    *,
    available_generation_ms: int,
) -> RecoveryPlan:
    """Choose recovery work independently of discovery or iteration order."""

    if available_generation_ms < 0:
        raise ValueError("available_generation_ms must be non-negative")
    request_ids = [request.request_id for request in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("duplicate recovery request id")

    ranked = tuple(sorted(requests, key=_rank_key))
    selected: list[RecoveryRequest] = []
    deferred: list[RecoveryRequest] = []
    remaining = available_generation_ms
    higher_priority_blocked = False
    for request in ranked:
        if not higher_priority_blocked and request.estimated_generation_ms <= remaining:
            selected.append(request)
            remaining -= request.estimated_generation_ms
        else:
            deferred.append(request)
            # Requests are sorted by priority and then increasing cost.  Once
            # a completeness request cannot fit, spending the leftover on a
            # lower-priority quality retry would invert the user-visible goal.
            higher_priority_blocked = True
    return RecoveryPlan(
        selected=tuple(selected),
        deferred=tuple(deferred),
        available_generation_ms=available_generation_ms,
        selected_generation_ms=available_generation_ms - remaining,
    )


def partial_remap_recovery_request(
    *,
    key_mode: int,
    difficulty: str,
    window_ms: int,
) -> RecoveryRequest:
    return RecoveryRequest(
        request_id=f"partial:{key_mode}k:{difficulty}",
        kind=RecoveryKind.PARTIAL_REMAP,
        key_mode=key_mode,
        difficulty=difficulty,
        priority=RecoveryPriority.COMPLETENESS_BLOCKING,
        estimated_generation_ms=window_ms,
        reason="ACTIVE_COVERAGE_GAP_WITHOUT_ADMITTED_CANDIDATE",
    )


def intro_phrase_recovery_request(
    *,
    key_mode: int,
    song_duration_ms: int,
) -> RecoveryRequest:
    return RecoveryRequest(
        request_id=f"intro:{key_mode}k:EXPERT",
        kind=RecoveryKind.INTRO,
        key_mode=key_mode,
        difficulty="EXPERT",
        priority=RecoveryPriority.COMPLETENESS_BLOCKING,
        estimated_generation_ms=song_duration_ms,
        reason="CONFIRMED_INTRO_PHRASE_FAMILY_DEFECT",
    )


def timing_family_recovery_request(
    *,
    key_mode: int,
    difficulty: str,
    song_duration_ms: int,
) -> RecoveryRequest:
    return RecoveryRequest(
        request_id=f"timing:{key_mode}k:{difficulty}",
        kind=RecoveryKind.TIMING_FAMILY,
        key_mode=key_mode,
        difficulty=difficulty,
        priority=RecoveryPriority.QUALITY_BLOCKING,
        estimated_generation_ms=song_duration_ms,
        reason="CORROBORATED_TIMING_FAMILY_OUTLIER",
    )
