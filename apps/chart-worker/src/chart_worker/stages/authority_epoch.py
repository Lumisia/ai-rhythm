"""Audit records for bounded shared-timing authority replacement."""

from dataclasses import dataclass
from typing import Literal

AuthorityEpochStatus = Literal[
    "SELECTED",
    "REJECTED_MAP_TIMING_FEEDBACK",
    "FAILED",
]


@dataclass(frozen=True, slots=True)
class AuthorityEpochRecord:
    epoch: int
    authority_sha256: str
    mode: Literal["STANDARD", "SUPER_TIMING"]
    status: AuthorityEpochStatus
    escalation: dict[str, object] | None = None

    def to_report(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "authoritySha256": self.authority_sha256,
            "mode": self.mode,
            "status": self.status,
            "escalation": self.escalation,
        }
