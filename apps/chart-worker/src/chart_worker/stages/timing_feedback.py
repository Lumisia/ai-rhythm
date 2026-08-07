"""Internal MAP evidence that can invalidate a shared timing authority."""

from dataclasses import dataclass
from typing import Literal

TimingFailureFamily = Literal[
    "DUPLICATE_NOTE",
    "ACTIVE_MIDDLE_GAP",
    "RESNAP_COLLISION",
]


@dataclass(frozen=True, slots=True)
class MapTimingFailureSignature:
    authority_sha256: str
    key_mode: int
    difficulty: str
    seed: int
    timing_segment_id: int
    failure_family: TimingFailureFamily
    time_ms: int
    grid_aligned: bool
    evidence: dict[str, object] | None = None

    @property
    def corroboration_key(self) -> tuple[str, int, str, int, str]:
        return (
            self.authority_sha256,
            self.key_mode,
            self.difficulty,
            self.timing_segment_id,
            self.failure_family,
        )

    def to_report(self) -> dict[str, object]:
        return {
            "authoritySha256": self.authority_sha256,
            "keyMode": self.key_mode,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "timingSegmentId": self.timing_segment_id,
            "failureFamily": self.failure_family,
            "timeMs": self.time_ms,
            "gridAligned": self.grid_aligned,
            "evidence": self.evidence,
        }


class RetryTimingSignal(Exception):
    """Two MAP seeds independently rejected the same timing segment."""

    def __init__(self, signatures: tuple[MapTimingFailureSignature, ...]) -> None:
        self.signatures = signatures
        super().__init__(
            "distinct MAP seeds failed in the same shared timing segment"
        )

    def to_context(self) -> dict[str, object]:
        first = self.signatures[0]
        return {
            "authoritySha256": first.authority_sha256,
            "keyMode": first.key_mode,
            "difficulty": first.difficulty,
            "timingSegmentId": first.timing_segment_id,
            "failureFamily": first.failure_family,
            "seeds": [signature.seed for signature in self.signatures],
            "signatures": [signature.to_report() for signature in self.signatures],
        }


def record_timing_failure(
    signatures: list[MapTimingFailureSignature],
    candidate: MapTimingFailureSignature,
) -> None:
    """Append evidence and escalate only corroborating, grid-aligned seeds."""
    signatures.append(candidate)
    if not candidate.grid_aligned:
        return
    corroborating = tuple(
        signature
        for signature in signatures
        if signature.grid_aligned
        and signature.corroboration_key == candidate.corroboration_key
    )
    if len({signature.seed for signature in corroborating}) >= 2:
        raise RetryTimingSignal(corroborating)
