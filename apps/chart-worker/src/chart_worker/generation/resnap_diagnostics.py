"""Optional diagnostics emitted by the pinned Mapperatorinator resnap patch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from chart_worker.hashing import sha256_file
from chart_worker.schema.note import Chart

LEGACY_ORIGIN_DIAGNOSTICS_VERSION = "mania-origin-v1-canonical-hold-ir"
RESNAP_DIAGNOSTICS_VERSION = "mania-origin-v2-osu-bound"
_OBJECT_DIAGNOSTICS_VERSIONS = frozenset(
    {LEGACY_ORIGIN_DIAGNOSTICS_VERSION, RESNAP_DIAGNOSTICS_VERSION}
)
SUPPORTED_RESNAP_DIAGNOSTICS_VERSIONS = frozenset(
    {
        "resnap-collisions-v2-preserve-raw",
        "mania-resnap-v3-hold-pairs",
        LEGACY_ORIGIN_DIAGNOSTICS_VERSION,
        RESNAP_DIAGNOSTICS_VERSION,
    }
)
ResnapDiagnosticsStatus = Literal[
    "OBSERVED",
    "NO_COLLISIONS",
    "UNOBSERVED",
    "INVALID",
]
ResnapNoteKind = Literal[
    "TAP",
    "HOLD",
    "HOLD_START",
    "HOLD_END",
    "HOLD_PAIR",
]
ResnapCollisionReason = Literal[
    "SNAP_TARGET_CONFLICT_PRESERVED",
    "RAW_TIME_COLLISION_PRESERVED",
    "START_BOUNDARY_RAW_RESTORED",
    "HOLD_PAIR_RAW_RESTORED",
    "HOLD_PAIR_START_BOUNDARY_RAW_RESTORED",
    "HOLD_PAIR_INVALID_RAW",
    "LANE_ORDER_RAW_RESTORED",
]


@dataclass(frozen=True, slots=True)
class ResnapCollision:
    seed: int
    lane: int
    note_kind: ResnapNoteKind | None
    reason: ResnapCollisionReason
    pre_time_ms: int | None = None
    post_time_ms: int | None = None
    snap_divisor: int | None = None
    raw_start_ms: int | None = None
    raw_end_ms: int | None = None
    proposed_start_ms: int | None = None
    proposed_end_ms: int | None = None
    previous_object_id: int | None = None
    current_object_id: int | None = None
    proposed_previous_end_ms: int | None = None
    proposed_current_start_ms: int | None = None
    raw_previous_end_ms: int | None = None
    raw_current_start_ms: int | None = None

    def to_report(self) -> dict[str, object]:
        report: dict[str, object] = {
            "seed": self.seed,
            "lane": self.lane,
            "reason": self.reason,
        }
        if self.reason == "LANE_ORDER_RAW_RESTORED":
            report.update(
                {
                    "previousObjectId": self.previous_object_id,
                    "currentObjectId": self.current_object_id,
                    "proposedPreviousEndMs": self.proposed_previous_end_ms,
                    "proposedCurrentStartMs": self.proposed_current_start_ms,
                    "rawPreviousEndMs": self.raw_previous_end_ms,
                    "rawCurrentStartMs": self.raw_current_start_ms,
                }
            )
        elif self.note_kind == "HOLD_PAIR":
            report["noteKind"] = self.note_kind
            report.update(
                {
                    "rawStartMs": self.raw_start_ms,
                    "rawEndMs": self.raw_end_ms,
                    "proposedStartMs": self.proposed_start_ms,
                    "proposedEndMs": self.proposed_end_ms,
                }
            )
        else:
            report["noteKind"] = self.note_kind
            report.update(
                {
                    "preTimeMs": self.pre_time_ms,
                    "postTimeMs": self.post_time_ms,
                    "snapDivisor": self.snap_divisor,
                }
            )
        return report


@dataclass(frozen=True, slots=True)
class ManiaEventOrigin:
    kind: Literal["GENERATED", "REFERENCE"]
    source_window_id: int | None
    source_token_index: int | None
    reference_event_index: int | None

    def to_report(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "sourceWindowId": self.source_window_id,
            "sourceTokenIndex": self.source_token_index,
            "referenceEventIndex": self.reference_event_index,
        }


@dataclass(frozen=True, slots=True)
class ManiaObjectDiagnostic:
    object_id: int
    lane: int
    kind: Literal["TAP", "HOLD"]
    start_time_ms: int
    end_time_ms: int | None
    start_group_id: int
    end_group_id: int | None
    start_origins: tuple[ManiaEventOrigin, ...]
    end_origins: tuple[ManiaEventOrigin, ...]

    def to_report(self) -> dict[str, object]:
        return {
            "objectId": self.object_id,
            "lane": self.lane,
            "kind": self.kind,
            "startTimeMs": self.start_time_ms,
            "endTimeMs": self.end_time_ms,
            "startGroupId": self.start_group_id,
            "endGroupId": self.end_group_id,
            "startOrigins": [origin.to_report() for origin in self.start_origins],
            "endOrigins": [origin.to_report() for origin in self.end_origins],
        }


@dataclass(frozen=True, slots=True)
class ManiaDuplicateDiagnostic:
    kept_group_id: int
    dropped_group_id: int
    reason: Literal["EXACT_CROSS_WINDOW_DUPLICATE"]

    def to_report(self) -> dict[str, object]:
        return {
            "keptGroupId": self.kept_group_id,
            "droppedGroupId": self.dropped_group_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ResnapDiagnostics:
    status: ResnapDiagnosticsStatus
    collisions: tuple[ResnapCollision, ...] = ()
    error: str | None = None
    version: str = RESNAP_DIAGNOSTICS_VERSION
    osu_sha256: str | None = None
    mania_objects: tuple[ManiaObjectDiagnostic, ...] = ()
    duplicates: tuple[ManiaDuplicateDiagnostic, ...] = ()

    @classmethod
    def unobserved(cls) -> ResnapDiagnostics:
        return cls(status="UNOBSERVED")

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "osuSha256": self.osu_sha256,
            "status": self.status,
            "collisions": [collision.to_report() for collision in self.collisions],
            "maniaObjects": [obj.to_report() for obj in self.mania_objects],
            "duplicates": [duplicate.to_report() for duplicate in self.duplicates],
            "error": self.error,
        }


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _optional_integer(value: object, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _collision(payload: object, *, seed: int) -> ResnapCollision:
    if not isinstance(payload, dict):
        raise TypeError("collision must be an object")
    lane = _integer(payload.get("lane"), "lane")
    note_kind = payload.get("noteKind")
    reason = payload.get("reason")
    if lane < 0:
        raise ValueError("lane must not be negative")
    if reason == "LANE_ORDER_RAW_RESTORED":
        previous_object_id = _integer(
            payload.get("previousObjectId"), "previousObjectId"
        )
        current_object_id = _integer(payload.get("currentObjectId"), "currentObjectId")
        proposed_previous_end_ms = _integer(
            payload.get("proposedPreviousEndMs"), "proposedPreviousEndMs"
        )
        proposed_current_start_ms = _integer(
            payload.get("proposedCurrentStartMs"), "proposedCurrentStartMs"
        )
        raw_previous_end_ms = _optional_integer(
            payload.get("rawPreviousEndMs"), "rawPreviousEndMs"
        )
        raw_current_start_ms = _integer(
            payload.get("rawCurrentStartMs"), "rawCurrentStartMs"
        )
        if previous_object_id < 0 or current_object_id <= previous_object_id:
            raise ValueError("lane-order object IDs must be increasing and non-negative")
        if min(
            proposed_previous_end_ms,
            proposed_current_start_ms,
            raw_current_start_ms,
        ) < 0 or (raw_previous_end_ms is not None and raw_previous_end_ms < 0):
            raise ValueError("lane-order times must not be negative")
        return ResnapCollision(
            seed=seed,
            lane=lane,
            note_kind=None,
            reason=reason,
            previous_object_id=previous_object_id,
            current_object_id=current_object_id,
            proposed_previous_end_ms=proposed_previous_end_ms,
            proposed_current_start_ms=proposed_current_start_ms,
            raw_previous_end_ms=raw_previous_end_ms,
            raw_current_start_ms=raw_current_start_ms,
        )
    if note_kind not in ("TAP", "HOLD", "HOLD_START", "HOLD_END", "HOLD_PAIR"):
        raise ValueError("unsupported noteKind")
    if reason not in (
        "SNAP_TARGET_CONFLICT_PRESERVED",
        "RAW_TIME_COLLISION_PRESERVED",
        "START_BOUNDARY_RAW_RESTORED",
        "HOLD_PAIR_RAW_RESTORED",
        "HOLD_PAIR_START_BOUNDARY_RAW_RESTORED",
        "HOLD_PAIR_INVALID_RAW",
        "LANE_ORDER_RAW_RESTORED",
    ):
        raise ValueError("unsupported collision reason")
    if note_kind == "HOLD_PAIR":
        if reason not in (
            "HOLD_PAIR_RAW_RESTORED",
            "HOLD_PAIR_START_BOUNDARY_RAW_RESTORED",
            "HOLD_PAIR_INVALID_RAW",
        ):
            raise ValueError("HOLD_PAIR requires a HOLD pair reason")
        raw_start_ms = _optional_integer(payload.get("rawStartMs"), "rawStartMs")
        raw_end_ms = _optional_integer(payload.get("rawEndMs"), "rawEndMs")
        proposed_start_ms = _optional_integer(
            payload.get("proposedStartMs"), "proposedStartMs"
        )
        proposed_end_ms = _optional_integer(
            payload.get("proposedEndMs"), "proposedEndMs"
        )
        if raw_start_ms is None and raw_end_ms is None:
            raise ValueError("HOLD_PAIR must contain at least one raw endpoint")
        return ResnapCollision(
            seed=seed,
            lane=lane,
            note_kind=note_kind,
            reason=reason,
            raw_start_ms=raw_start_ms,
            raw_end_ms=raw_end_ms,
            proposed_start_ms=proposed_start_ms,
            proposed_end_ms=proposed_end_ms,
        )
    if reason in (
        "HOLD_PAIR_RAW_RESTORED",
        "HOLD_PAIR_START_BOUNDARY_RAW_RESTORED",
        "HOLD_PAIR_INVALID_RAW",
    ):
        raise ValueError("HOLD pair reason requires noteKind HOLD_PAIR")
    pre_time_ms = _integer(payload.get("preTimeMs"), "preTimeMs")
    post_time_ms = _integer(payload.get("postTimeMs"), "postTimeMs")
    snap_divisor = _integer(payload.get("snapDivisor"), "snapDivisor")
    if post_time_ms < 0:
        raise ValueError("postTimeMs must not be negative")
    if snap_divisor < 0:
        raise ValueError("snapDivisor must not be negative")
    return ResnapCollision(
        seed=seed,
        lane=lane,
        note_kind=note_kind,
        reason=reason,
        pre_time_ms=pre_time_ms,
        post_time_ms=post_time_ms,
        snap_divisor=snap_divisor,
    )


def _origin(payload: object) -> ManiaEventOrigin:
    if not isinstance(payload, dict):
        raise TypeError("origin must be an object")
    kind = payload.get("kind")
    source_window_id = _optional_integer(
        payload.get("sourceWindowId"), "sourceWindowId"
    )
    source_token_index = _optional_integer(
        payload.get("sourceTokenIndex"), "sourceTokenIndex"
    )
    reference_event_index = _optional_integer(
        payload.get("referenceEventIndex"), "referenceEventIndex"
    )
    if kind == "GENERATED":
        if source_window_id is None or source_token_index is None:
            raise ValueError("GENERATED origin requires window and token indices")
        if source_window_id < 0 or source_token_index < 0:
            raise ValueError("generated origin indices must not be negative")
        if reference_event_index is not None:
            raise ValueError("GENERATED origin cannot contain a reference index")
    elif kind == "REFERENCE":
        if reference_event_index is None or reference_event_index < 0:
            raise ValueError("REFERENCE origin requires a non-negative reference index")
        if source_window_id is not None or source_token_index is not None:
            raise ValueError("REFERENCE origin cannot contain generated indices")
    else:
        raise ValueError("unsupported origin kind")
    return ManiaEventOrigin(
        kind=kind,
        source_window_id=source_window_id,
        source_token_index=source_token_index,
        reference_event_index=reference_event_index,
    )


def _origins(payload: object, name: str) -> tuple[ManiaEventOrigin, ...]:
    if not isinstance(payload, list):
        raise TypeError(f"{name} must be an array")
    return tuple(_origin(item) for item in payload)


def _mania_object(payload: object, *, expected_id: int) -> ManiaObjectDiagnostic:
    if not isinstance(payload, dict):
        raise TypeError("mania object must be an object")
    object_id = _integer(payload.get("objectId"), "objectId")
    lane = _integer(payload.get("lane"), "lane")
    kind = payload.get("kind")
    start_time_ms = _integer(payload.get("startTimeMs"), "startTimeMs")
    end_time_ms = _optional_integer(payload.get("endTimeMs"), "endTimeMs")
    start_group_id = _integer(payload.get("startGroupId"), "startGroupId")
    end_group_id = _optional_integer(payload.get("endGroupId"), "endGroupId")
    if object_id != expected_id:
        raise ValueError("mania object IDs must be contiguous and ordered")
    if lane < 0 or start_time_ms < 0 or start_group_id < 0:
        raise ValueError("mania object indices and times must not be negative")
    start_origins = _origins(payload.get("startOrigins"), "startOrigins")
    end_origins = _origins(payload.get("endOrigins"), "endOrigins")
    if kind == "TAP":
        if end_time_ms is not None or end_group_id is not None or end_origins:
            raise ValueError("TAP cannot contain HOLD end fields")
    elif kind == "HOLD":
        if end_time_ms is None or end_group_id is None:
            raise ValueError("HOLD requires end time and group")
        if end_time_ms <= start_time_ms:
            raise ValueError("HOLD duration must be positive")
        if end_group_id < 0:
            raise ValueError("HOLD end group must not be negative")
    else:
        raise ValueError("unsupported mania object kind")
    return ManiaObjectDiagnostic(
        object_id=object_id,
        lane=lane,
        kind=kind,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        start_group_id=start_group_id,
        end_group_id=end_group_id,
        start_origins=start_origins,
        end_origins=end_origins,
    )


def _duplicate(payload: object) -> ManiaDuplicateDiagnostic:
    if not isinstance(payload, dict):
        raise TypeError("duplicate must be an object")
    kept = _integer(payload.get("keptGroupId"), "keptGroupId")
    dropped = _integer(payload.get("droppedGroupId"), "droppedGroupId")
    reason = payload.get("reason")
    if kept < 0 or dropped < 0 or kept == dropped:
        raise ValueError("duplicate group IDs must be distinct and non-negative")
    if reason != "EXACT_CROSS_WINDOW_DUPLICATE":
        raise ValueError("unsupported duplicate reason")
    return ManiaDuplicateDiagnostic(kept, dropped, reason)


def read_resnap_diagnostics(osu_path: Path) -> ResnapDiagnostics:
    """Read an adjacent sidecar without turning diagnostics into a hard dependency."""
    sidecar_path = Path(osu_path).with_suffix(".resnap.json")
    if not sidecar_path.is_file():
        return ResnapDiagnostics.unobserved()
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("sidecar root must be an object")
        version = payload.get("version")
        if version not in SUPPORTED_RESNAP_DIAGNOSTICS_VERSIONS:
            raise ValueError("unsupported resnap diagnostics version")
        seed = _integer(payload.get("seed"), "seed")
        osu_sha256: str | None = None
        if version == RESNAP_DIAGNOSTICS_VERSION:
            raw_osu_sha256 = payload.get("osuSha256")
            if (
                not isinstance(raw_osu_sha256, str)
                or len(raw_osu_sha256) != 64
                or any(character not in "0123456789abcdef" for character in raw_osu_sha256)
            ):
                raise ValueError("osuSha256 must be a lowercase SHA-256 hex digest")
            if not osu_path.is_file() or sha256_file(osu_path) != raw_osu_sha256:
                raise ValueError("resnap sidecar SHA-256 does not match .osu")
            osu_sha256 = raw_osu_sha256
        raw_collisions = payload.get("collisions")
        if not isinstance(raw_collisions, list):
            raise TypeError("collisions must be an array")
        collisions = tuple(
            _collision(item, seed=seed) for item in raw_collisions
        )
        mania_objects: tuple[ManiaObjectDiagnostic, ...] = ()
        duplicates: tuple[ManiaDuplicateDiagnostic, ...] = ()
        if version in _OBJECT_DIAGNOSTICS_VERSIONS:
            raw_objects = payload.get("maniaObjects")
            raw_duplicates = payload.get("duplicates")
            if not isinstance(raw_objects, list):
                raise TypeError("maniaObjects must be an array")
            if not isinstance(raw_duplicates, list):
                raise TypeError("duplicates must be an array")
            mania_objects = tuple(
                _mania_object(item, expected_id=index)
                for index, item in enumerate(raw_objects)
            )
            duplicates = tuple(_duplicate(item) for item in raw_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        return ResnapDiagnostics(status="INVALID", error=str(error))
    return ResnapDiagnostics(
        status="OBSERVED" if collisions else "NO_COLLISIONS",
        collisions=collisions,
        version=version,
        osu_sha256=osu_sha256,
        mania_objects=mania_objects,
        duplicates=duplicates,
    )


def mania_object_mismatch(
    diagnostics: ResnapDiagnostics,
    notes: Chart,
) -> str | None:
    """Return direct evidence when v4 canonical objects differ from .osu output."""
    if diagnostics.status == "UNOBSERVED":
        return None
    if diagnostics.version not in _OBJECT_DIAGNOSTICS_VERSIONS:
        return None
    if diagnostics.status == "INVALID":
        return diagnostics.error or "invalid v4 sidecar"
    sidecar = sorted(
        (
            item.kind,
            item.lane,
            item.start_time_ms,
            item.end_time_ms,
        )
        for item in diagnostics.mania_objects
    )
    serialized = sorted(
        (
            note.kind,
            note.lane,
            note.time_ms,
            (
                note.time_ms + (note.duration_ms or 0)
                if note.kind == "HOLD"
                else None
            ),
        )
        for note in notes
    )
    if sidecar == serialized:
        return None
    limit = min(len(sidecar), len(serialized))
    first_difference = next(
        (
            index
            for index in range(limit)
            if sidecar[index] != serialized[index]
        ),
        limit,
    )
    return (
        "canonical mania object sidecar does not match serialized .osu: "
        f"sidecar={len(sidecar)}, osu={len(serialized)}, "
        f"firstDifference={first_difference}"
    )
