"""Optional diagnostics emitted by the pinned Mapperatorinator resnap patch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RESNAP_DIAGNOSTICS_VERSION = "resnap-collisions-v1"
ResnapDiagnosticsStatus = Literal[
    "OBSERVED",
    "NO_COLLISIONS",
    "UNOBSERVED",
    "INVALID",
]
ResnapNoteKind = Literal["TAP", "HOLD"]


@dataclass(frozen=True, slots=True)
class ResnapCollision:
    seed: int
    lane: int
    note_kind: ResnapNoteKind
    pre_time_ms: int
    post_time_ms: int
    snap_divisor: int

    def to_report(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "lane": self.lane,
            "noteKind": self.note_kind,
            "preTimeMs": self.pre_time_ms,
            "postTimeMs": self.post_time_ms,
            "snapDivisor": self.snap_divisor,
        }


@dataclass(frozen=True, slots=True)
class ResnapDiagnostics:
    status: ResnapDiagnosticsStatus
    collisions: tuple[ResnapCollision, ...] = ()
    error: str | None = None

    @classmethod
    def unobserved(cls) -> ResnapDiagnostics:
        return cls(status="UNOBSERVED")

    def to_report(self) -> dict[str, object]:
        return {
            "version": RESNAP_DIAGNOSTICS_VERSION,
            "status": self.status,
            "collisions": [collision.to_report() for collision in self.collisions],
            "error": self.error,
        }


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _collision(payload: object, *, seed: int) -> ResnapCollision:
    if not isinstance(payload, dict):
        raise TypeError("collision must be an object")
    lane = _integer(payload.get("lane"), "lane")
    note_kind = payload.get("noteKind")
    pre_time_ms = _integer(payload.get("preTimeMs"), "preTimeMs")
    post_time_ms = _integer(payload.get("postTimeMs"), "postTimeMs")
    snap_divisor = _integer(payload.get("snapDivisor"), "snapDivisor")
    if lane < 0:
        raise ValueError("lane must not be negative")
    if note_kind not in ("TAP", "HOLD"):
        raise ValueError("noteKind must be TAP or HOLD")
    if post_time_ms < 0:
        raise ValueError("postTimeMs must not be negative")
    if snap_divisor <= 0:
        raise ValueError("snapDivisor must be positive")
    return ResnapCollision(
        seed=seed,
        lane=lane,
        note_kind=note_kind,
        pre_time_ms=pre_time_ms,
        post_time_ms=post_time_ms,
        snap_divisor=snap_divisor,
    )


def read_resnap_diagnostics(osu_path: Path) -> ResnapDiagnostics:
    """Read an adjacent sidecar without turning diagnostics into a hard dependency."""
    sidecar_path = Path(osu_path).with_suffix(".resnap.json")
    if not sidecar_path.is_file():
        return ResnapDiagnostics.unobserved()
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("sidecar root must be an object")
        if payload.get("version") != RESNAP_DIAGNOSTICS_VERSION:
            raise ValueError("unsupported resnap diagnostics version")
        seed = _integer(payload.get("seed"), "seed")
        raw_collisions = payload.get("collisions")
        if not isinstance(raw_collisions, list):
            raise TypeError("collisions must be an array")
        collisions = tuple(
            _collision(item, seed=seed) for item in raw_collisions
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        return ResnapDiagnostics(status="INVALID", error=str(error))
    return ResnapDiagnostics(
        status="OBSERVED" if collisions else "NO_COLLISIONS",
        collisions=collisions,
    )
