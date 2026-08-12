"""Shadow-only lane state trace for TAP/HOLD integrity evidence."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from chart_worker.generation.resnap_diagnostics import (
    ResnapDiagnostics,
    mania_object_mismatch,
)
from chart_worker.schema.note import Chart

TRACE_VERSION = "hold-lane-state-shadow-v1"


@dataclass(frozen=True, slots=True)
class HoldLaneViolation:
    code: Literal["DUPLICATE_LANE_TIME", "NOTE_WHILE_HOLD_ACTIVE"]
    lane: int
    time_ms: int
    note_kind: str
    active_hold_end_ms: int | None = None

    def to_report(self) -> dict[str, object]:
        return {
            "code": self.code,
            "lane": self.lane,
            "timeMs": self.time_ms,
            "noteKind": self.note_kind,
            "activeHoldEndMs": self.active_hold_end_ms,
        }


@dataclass(frozen=True, slots=True)
class HoldLaneStateTrace:
    status: Literal["PASS", "VIOLATION"]
    lane_count: int
    hold_count: int
    tap_count: int
    transition_count: int
    violations: tuple[HoldLaneViolation, ...]
    sidecar_evidence_status: Literal["AVAILABLE", "UNAVAILABLE", "INVALID"]
    sidecar_object_count: int
    sidecar_hold_count: int
    origin_backed_hold_count: int
    sidecar_mismatch: str | None

    def to_report(self) -> dict[str, object]:
        return {
            "version": TRACE_VERSION,
            "enforcement": "SHADOW",
            "status": self.status,
            "laneCount": self.lane_count,
            "holdCount": self.hold_count,
            "tapCount": self.tap_count,
            "transitionCount": self.transition_count,
            "violations": [item.to_report() for item in self.violations],
            "sidecarEvidenceStatus": self.sidecar_evidence_status,
            "sidecarObjectCount": self.sidecar_object_count,
            "sidecarHoldCount": self.sidecar_hold_count,
            "originBackedHoldCount": self.origin_backed_hold_count,
            "sidecarMismatch": self.sidecar_mismatch,
        }


def analyze_hold_lane_state(
    notes: Chart,
    diagnostics: ResnapDiagnostics,
) -> HoldLaneStateTrace:
    by_lane = defaultdict(list)
    for note in notes:
        by_lane[note.lane].append(note)

    violations: list[HoldLaneViolation] = []
    for lane, lane_notes in sorted(by_lane.items()):
        active_hold_end = -1
        previous_time = -1
        for note in sorted(lane_notes, key=lambda item: item.time_ms):
            if note.time_ms == previous_time:
                violations.append(
                    HoldLaneViolation(
                        "DUPLICATE_LANE_TIME",
                        lane,
                        note.time_ms,
                        note.kind,
                        active_hold_end if active_hold_end >= 0 else None,
                    )
                )
            elif note.time_ms < active_hold_end:
                violations.append(
                    HoldLaneViolation(
                        "NOTE_WHILE_HOLD_ACTIVE",
                        lane,
                        note.time_ms,
                        note.kind,
                        active_hold_end,
                    )
                )
            previous_time = note.time_ms
            if note.kind == "HOLD":
                active_hold_end = max(
                    active_hold_end,
                    note.time_ms + (note.duration_ms or 0),
                )

    sidecar_status: Literal["AVAILABLE", "UNAVAILABLE", "INVALID"]
    if diagnostics.status == "UNOBSERVED":
        sidecar_status = "UNAVAILABLE"
    elif diagnostics.status == "INVALID":
        sidecar_status = "INVALID"
    else:
        sidecar_status = "AVAILABLE"
    sidecar_holds = tuple(
        item for item in diagnostics.mania_objects if item.kind == "HOLD"
    )
    return HoldLaneStateTrace(
        status="VIOLATION" if violations else "PASS",
        lane_count=len(by_lane),
        hold_count=sum(note.kind == "HOLD" for note in notes),
        tap_count=sum(note.kind == "TAP" for note in notes),
        transition_count=sum(2 if note.kind == "HOLD" else 1 for note in notes),
        violations=tuple(violations),
        sidecar_evidence_status=sidecar_status,
        sidecar_object_count=len(diagnostics.mania_objects),
        sidecar_hold_count=len(sidecar_holds),
        origin_backed_hold_count=sum(
            bool(item.start_origins) and bool(item.end_origins) for item in sidecar_holds
        ),
        sidecar_mismatch=mania_object_mismatch(diagnostics, notes),
    )
