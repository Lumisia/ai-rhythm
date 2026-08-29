"""Evidence-bound gameplay interval planning for bounded shadow generation.

The policy is deliberately independent of song identity.  It consumes only
typed timing, audio, and cross-slot evidence and does not mutate a chart.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from hashlib import sha256

from chart_worker.analysis.intro_anchor import GRID_SUPPORT_WINDOW_MS
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.generation.partial_remap import MAX_REPAIR_FRACTION, PartialRemapWindow
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES

REQUIRED_GAMEPLAY_INTERVAL_POLICY_VERSION = "required-gameplay-interval-policy-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_PULSE_OPPORTUNITIES = 4
_MINIMUM_PULSE_SUPPORT = 2
_MINIMUM_FAMILY_SLOTS = 3
_MINIMUM_FAMILY_KEY_MODES = 2
_MINIMUM_PARTIAL_CONTEXT_BEATS = 4.0


class RequiredGameplayIntervalMode(str, Enum):
    OBSERVE = "OBSERVE"
    SHADOW_ENFORCE = "SHADOW_ENFORCE"


class RequiredGameplayEvidenceClass(str, Enum):
    BROADBAND_ATTACK = "BROADBAND_ATTACK"
    INTRO_REGION_CORROBORATED = "INTRO_REGION_CORROBORATED"
    PULSE_FAMILY_CORROBORATED = "PULSE_FAMILY_CORROBORATED"


class RequiredGameplayGroupType(str, Enum):
    TAP = "TAP"
    HOLD_START = "HOLD_START"


def _require_exact_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return value


def _require_exact_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be an exact boolean")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be an exact string")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class RequiredGameplayIntervalV1:
    start_ms: int
    end_ms: int
    minimum_complete_groups: int
    allowed_group_types: tuple[RequiredGameplayGroupType, ...]
    evidence_class: RequiredGameplayEvidenceClass
    evidence_digest: str
    mode: RequiredGameplayIntervalMode

    def __post_init__(self) -> None:
        _require_exact_int(self.start_ms, "start_ms")
        _require_exact_int(self.end_ms, "end_ms")
        if self.start_ms >= self.end_ms:
            raise ValueError("start_ms must be before end_ms")
        _require_exact_int(
            self.minimum_complete_groups,
            "minimum_complete_groups",
            minimum=1,
        )
        if type(self.allowed_group_types) is not tuple or not self.allowed_group_types:
            raise ValueError("allowed_group_types must be a non-empty tuple")
        if any(type(item) is not RequiredGameplayGroupType for item in self.allowed_group_types):
            raise TypeError("allowed_group_types must contain group type enum values")
        if len(set(self.allowed_group_types)) != len(self.allowed_group_types):
            raise ValueError("allowed_group_types must not contain duplicates")
        object.__setattr__(
            self,
            "allowed_group_types",
            tuple(sorted(self.allowed_group_types, key=lambda item: item.value)),
        )
        if type(self.evidence_class) is not RequiredGameplayEvidenceClass:
            raise TypeError("evidence_class must be a RequiredGameplayEvidenceClass")
        _require_sha256(self.evidence_digest, "evidence_digest")
        if type(self.mode) is not RequiredGameplayIntervalMode:
            raise TypeError("mode must be a RequiredGameplayIntervalMode")
        if (
            self.mode is RequiredGameplayIntervalMode.SHADOW_ENFORCE
            and self.minimum_complete_groups != 1
        ):
            raise ValueError(
                "SHADOW_ENFORCE requires exactly one complete group"
            )


@dataclass(frozen=True, slots=True)
class RequiredGameplayFamilySlotV1:
    key_mode: int
    difficulty: str
    supports_anchor: bool

    def __post_init__(self) -> None:
        if type(self.key_mode) is not int or self.key_mode not in KEY_MODES:
            raise ValueError("key_mode is unsupported")
        if type(self.difficulty) is not str or self.difficulty not in DIFFICULTIES:
            raise ValueError("difficulty is unsupported")
        _require_exact_bool(self.supports_anchor, "supports_anchor")


@dataclass(frozen=True, slots=True)
class RequiredGameplayEvidenceV1:
    anchor_status: str
    anchor_ms: int
    anchor_grid_ms: int
    aggregate_rank: float
    prominent_band_count: int
    pulse_support_count: int
    family_slots: tuple[RequiredGameplayFamilySlotV1, ...]
    local_audio_supported: bool
    reference_first_row_supported: bool
    repeated_high_confidence_refusal: bool
    timing_authority_valid: bool
    timing_authority_digest: str
    anchor_evidence_digest: str

    def __post_init__(self) -> None:
        if type(self.anchor_status) is not str or self.anchor_status not in {
            "CONFIRMED",
            "UNCERTAIN",
            "NON_RHYTHMIC",
        }:
            raise ValueError("anchor_status is unsupported")
        _require_exact_int(self.anchor_ms, "anchor_ms")
        _require_exact_int(self.anchor_grid_ms, "anchor_grid_ms")
        if type(self.aggregate_rank) is not float or not math.isfinite(
            self.aggregate_rank
        ):
            raise TypeError("aggregate_rank must be a finite exact float")
        if not 0.0 <= self.aggregate_rank <= 1.0:
            raise ValueError("aggregate_rank must be within 0..1")
        _require_exact_int(self.prominent_band_count, "prominent_band_count")
        _require_exact_int(self.pulse_support_count, "pulse_support_count")
        if self.pulse_support_count > _PULSE_OPPORTUNITIES:
            raise ValueError(
                f"pulse_support_count must not exceed {_PULSE_OPPORTUNITIES}"
            )
        if type(self.family_slots) is not tuple:
            raise TypeError("family_slots must be a tuple")
        if len(self.family_slots) > len(KEY_MODES) * len(DIFFICULTIES):
            raise ValueError("family_slots must not exceed the 12 public slots")
        if any(type(item) is not RequiredGameplayFamilySlotV1 for item in self.family_slots):
            raise TypeError("family_slots must contain typed slot evidence")
        identities = [(item.key_mode, item.difficulty) for item in self.family_slots]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate family slot evidence is not allowed")
        difficulty_order = {value: index for index, value in enumerate(DIFFICULTIES)}
        object.__setattr__(
            self,
            "family_slots",
            tuple(
                sorted(
                    self.family_slots,
                    key=lambda item: (
                        item.key_mode,
                        difficulty_order[item.difficulty],
                    ),
                )
            ),
        )
        for field_name in (
            "local_audio_supported",
            "reference_first_row_supported",
            "repeated_high_confidence_refusal",
            "timing_authority_valid",
        ):
            _require_exact_bool(getattr(self, field_name), field_name)
        _require_sha256(self.timing_authority_digest, "timing_authority_digest")
        _require_sha256(self.anchor_evidence_digest, "anchor_evidence_digest")


@dataclass(frozen=True, slots=True)
class RequiredGameplayIntervalDecision:
    interval: RequiredGameplayIntervalV1 | None
    reason: str

    def __post_init__(self) -> None:
        if self.interval is not None and type(self.interval) is not RequiredGameplayIntervalV1:
            raise TypeError("interval must be RequiredGameplayIntervalV1 or None")
        if type(self.reason) is not str or not self.reason:
            raise TypeError("reason must be a non-empty exact string")


def _validate_bpm_event_sequence(
    bpm_events: tuple[OsuBpmEvent, ...],
) -> None:
    if type(bpm_events) is not tuple or not bpm_events:
        raise ValueError("tempo-map advancement requires BPM events")
    previous_time: int | None = None
    for event in bpm_events:
        if type(event) is not OsuBpmEvent:
            raise TypeError("bpm_events must contain OsuBpmEvent values")
        if type(event.time_ms) is not int:
            raise TypeError("BPM event time_ms must be an exact integer")
        if type(event.bpm) is not float or not math.isfinite(event.bpm):
            raise ValueError("BPM event bpm must be a finite exact float")
        if event.bpm <= 0:
            raise ValueError("BPM event bpm must be positive")
        if previous_time is not None and event.time_ms <= previous_time:
            raise ValueError("BPM event times must be strictly increasing")
        previous_time = event.time_ms


def tempo_map_addresses(
    start_ms: int,
    bpm_events: tuple[OsuBpmEvent, ...],
) -> bool:
    """Return whether a structurally valid tempo map defines ``start_ms``."""

    _require_exact_int(start_ms, "start_ms")
    _validate_bpm_event_sequence(bpm_events)
    return bpm_events[0].time_ms <= start_ms


def _validate_bpm_events(
    bpm_events: tuple[OsuBpmEvent, ...], *, start_ms: int
) -> None:
    if not tempo_map_addresses(start_ms, bpm_events):
        raise ValueError("tempo map does not address start_ms")


def advance_tempo_map_beats(
    start_ms: int,
    beats: float,
    bpm_events: tuple[OsuBpmEvent, ...],
) -> int:
    """Advance fractional beats across every BPM boundary in the timing map."""

    _require_exact_int(start_ms, "start_ms")
    if type(beats) is not float or not math.isfinite(beats) or beats < 0:
        raise ValueError("beats must be a non-negative finite exact float")
    _validate_bpm_events(bpm_events, start_ms=start_ms)

    index = max(
        position
        for position, event in enumerate(bpm_events)
        if event.time_ms <= start_ms
    )
    current_ms = Decimal(start_ms)
    remaining_beats = Decimal(str(beats))
    while remaining_beats > 0:
        event = bpm_events[index]
        beat_length_ms = Decimal(60_000) / Decimal(str(event.bpm))
        if index + 1 >= len(bpm_events):
            current_ms += remaining_beats * beat_length_ms
            break

        next_time_ms = Decimal(bpm_events[index + 1].time_ms)
        available_beats = (next_time_ms - current_ms) / beat_length_ms
        if remaining_beats <= available_beats:
            current_ms += remaining_beats * beat_length_ms
            break
        remaining_beats -= available_beats
        current_ms = next_time_ms
        index += 1

    return int(current_ms.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _validate_partial_window(
    partial_window: PartialRemapWindow,
    *,
    duration_ms: int | None = None,
) -> None:
    if type(partial_window) is not PartialRemapWindow:
        raise TypeError("partial_window must be a PartialRemapWindow")
    _require_exact_int(partial_window.start_ms, "partial_window.start_ms")
    _require_exact_int(partial_window.end_ms, "partial_window.end_ms")
    if partial_window.start_ms >= partial_window.end_ms:
        raise ValueError("partial window start must be before end")
    if duration_ms is not None and partial_window.end_ms > duration_ms:
        raise ValueError("partial window must fit within duration_ms")


def required_gameplay_evidence_payload(
    evidence: RequiredGameplayEvidenceV1,
    *,
    partial_window: PartialRemapWindow,
    minimum_complete_groups: int = 1,
    allowed_group_types: tuple[RequiredGameplayGroupType, ...] = (
        RequiredGameplayGroupType.TAP,
        RequiredGameplayGroupType.HOLD_START,
    ),
) -> dict[str, object]:
    """Return the explicit identity-free payload bound by the evidence digest."""

    if type(evidence) is not RequiredGameplayEvidenceV1:
        raise TypeError("evidence must be RequiredGameplayEvidenceV1")
    _validate_partial_window(partial_window)
    _require_exact_int(
        minimum_complete_groups,
        "minimum_complete_groups",
        minimum=1,
    )
    if type(allowed_group_types) is not tuple or not allowed_group_types:
        raise ValueError("allowed_group_types must be a non-empty tuple")
    if any(type(item) is not RequiredGameplayGroupType for item in allowed_group_types):
        raise TypeError("allowed_group_types must contain group type enum values")
    if len(set(allowed_group_types)) != len(allowed_group_types):
        raise ValueError("allowed_group_types must not contain duplicates")

    return {
        "policyVersion": REQUIRED_GAMEPLAY_INTERVAL_POLICY_VERSION,
        "anchorEvidence": {
            "status": evidence.anchor_status,
            "anchorMs": evidence.anchor_ms,
            "anchorGridMs": evidence.anchor_grid_ms,
            "aggregateRank": evidence.aggregate_rank,
            "prominentBandCount": evidence.prominent_band_count,
            "pulseSupportCount": evidence.pulse_support_count,
            "familySlots": [
                {
                    "keyMode": slot.key_mode,
                    "difficulty": slot.difficulty,
                    "supportsAnchor": slot.supports_anchor,
                }
                for slot in evidence.family_slots
            ],
            "localAudioSupported": evidence.local_audio_supported,
            "referenceFirstRowSupported": evidence.reference_first_row_supported,
            "repeatedHighConfidenceRefusal": evidence.repeated_high_confidence_refusal,
            "timingAuthorityValid": evidence.timing_authority_valid,
            "anchorEvidenceDigest": evidence.anchor_evidence_digest,
        },
        "timingAuthorityDigest": evidence.timing_authority_digest,
        "partialWindow": {
            "startMs": partial_window.start_ms,
            "endMs": partial_window.end_ms,
        },
        "minimumCompleteGroups": minimum_complete_groups,
        "allowedGroupTypes": sorted(item.value for item in allowed_group_types),
    }


def required_gameplay_evidence_digest(
    evidence: RequiredGameplayEvidenceV1,
    *,
    partial_window: PartialRemapWindow,
    minimum_complete_groups: int = 1,
    allowed_group_types: tuple[RequiredGameplayGroupType, ...] = (
        RequiredGameplayGroupType.TAP,
        RequiredGameplayGroupType.HOLD_START,
    ),
) -> str:
    payload = required_gameplay_evidence_payload(
        evidence,
        partial_window=partial_window,
        minimum_complete_groups=minimum_complete_groups,
        allowed_group_types=allowed_group_types,
    )
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _decline(reason: str) -> RequiredGameplayIntervalDecision:
    return RequiredGameplayIntervalDecision(interval=None, reason=reason)


def plan_required_gameplay_interval(
    evidence: RequiredGameplayEvidenceV1,
    *,
    partial_window: PartialRemapWindow,
    bpm_events: tuple[OsuBpmEvent, ...],
    second_distinct_row_ms: int,
    duration_ms: int,
    mode: RequiredGameplayIntervalMode,
) -> RequiredGameplayIntervalDecision:
    """Plan one generic evidence-backed interval without mutating a chart."""

    if type(evidence) is not RequiredGameplayEvidenceV1:
        raise TypeError("evidence must be RequiredGameplayEvidenceV1")
    if type(mode) is not RequiredGameplayIntervalMode:
        raise TypeError("mode must be RequiredGameplayIntervalMode")
    _require_exact_int(duration_ms, "duration_ms", minimum=1)
    _require_exact_int(second_distinct_row_ms, "second_distinct_row_ms")
    if second_distinct_row_ms > duration_ms:
        raise ValueError("second_distinct_row_ms must fit within duration_ms")
    _validate_partial_window(partial_window, duration_ms=duration_ms)
    _validate_bpm_events(bpm_events, start_ms=second_distinct_row_ms)

    if (
        (partial_window.end_ms - partial_window.start_ms) / duration_ms
        >= MAX_REPAIR_FRACTION
    ):
        return _decline("PARTIAL_WINDOW_TOO_LARGE")
    minimum_partial_end_ms = min(
        duration_ms,
        advance_tempo_map_beats(
            second_distinct_row_ms,
            _MINIMUM_PARTIAL_CONTEXT_BEATS,
            bpm_events,
        ),
    )
    if partial_window.end_ms < minimum_partial_end_ms:
        return _decline("PARTIAL_WINDOW_TOO_SHORT")

    if evidence.anchor_status != "CONFIRMED":
        return _decline("ANCHOR_NOT_CONFIRMED")
    if abs(evidence.anchor_ms - evidence.anchor_grid_ms) > GRID_SUPPORT_WINDOW_MS:
        return _decline("ANCHOR_GRID_UNSUPPORTED")
    if not evidence.local_audio_supported:
        return _decline("LOCAL_AUDIO_UNSUPPORTED")
    if not evidence.reference_first_row_supported:
        return _decline("REFERENCE_FIRST_ROW_UNSUPPORTED")
    if evidence.repeated_high_confidence_refusal:
        return _decline("REPEATED_MODEL_REFUSAL")
    if not evidence.timing_authority_valid:
        return _decline("TIMING_AUTHORITY_INVALID")

    if evidence.aggregate_rank >= 0.9 and evidence.prominent_band_count >= 2:
        evidence_class = RequiredGameplayEvidenceClass.BROADBAND_ATTACK
        reason = "BROADBAND_ATTACK_SUPPORTED"
    else:
        if evidence.pulse_support_count < _MINIMUM_PULSE_SUPPORT:
            return _decline("PULSE_SUPPORT_INSUFFICIENT")
        supported_slots = tuple(
            slot for slot in evidence.family_slots if slot.supports_anchor
        )
        minimum_support = max(
            _MINIMUM_FAMILY_SLOTS,
            math.ceil(len(evidence.family_slots) * 0.25),
        )
        if len(supported_slots) < minimum_support:
            return _decline("FAMILY_SLOT_SUPPORT_INSUFFICIENT")
        if len({slot.key_mode for slot in supported_slots}) < _MINIMUM_FAMILY_KEY_MODES:
            return _decline("FAMILY_KEY_MODE_SUPPORT_INSUFFICIENT")
        evidence_class = RequiredGameplayEvidenceClass.PULSE_FAMILY_CORROBORATED
        reason = "PULSE_FAMILY_CORROBORATED"

    interval_start_ms = max(
        partial_window.start_ms,
        0,
        evidence.anchor_grid_ms - GRID_SUPPORT_WINDOW_MS,
    )
    interval_end_ms = min(
        partial_window.end_ms,
        duration_ms,
        evidence.anchor_grid_ms + GRID_SUPPORT_WINDOW_MS,
    )
    if interval_start_ms >= interval_end_ms:
        return _decline("REQUIRED_INTERVAL_OUTSIDE_PARTIAL_WINDOW")

    allowed_group_types = (
        RequiredGameplayGroupType.TAP,
        RequiredGameplayGroupType.HOLD_START,
    )
    digest = required_gameplay_evidence_digest(
        evidence,
        partial_window=partial_window,
        allowed_group_types=allowed_group_types,
    )
    return RequiredGameplayIntervalDecision(
        interval=RequiredGameplayIntervalV1(
            start_ms=interval_start_ms,
            end_ms=interval_end_ms,
            minimum_complete_groups=1,
            allowed_group_types=allowed_group_types,
            evidence_class=evidence_class,
            evidence_digest=digest,
            mode=mode,
        ),
        reason=reason,
    )
