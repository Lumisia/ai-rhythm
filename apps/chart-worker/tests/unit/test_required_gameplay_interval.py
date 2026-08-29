from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.generation.partial_remap import PartialRemapWindow
from chart_worker.generation.required_gameplay_interval import (
    RequiredGameplayEvidenceClass,
    RequiredGameplayEvidenceV1,
    RequiredGameplayFamilySlotV1,
    RequiredGameplayGroupType,
    RequiredGameplayIntervalMode,
    RequiredGameplayIntervalV1,
    advance_tempo_map_beats,
    plan_required_gameplay_interval,
    required_gameplay_evidence_digest,
    required_gameplay_evidence_payload,
    tempo_map_addresses,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64


def _slots() -> tuple[RequiredGameplayFamilySlotV1, ...]:
    return (
        RequiredGameplayFamilySlotV1(4, "EASY", True),
        RequiredGameplayFamilySlotV1(4, "NORMAL", False),
        RequiredGameplayFamilySlotV1(6, "HARD", True),
        RequiredGameplayFamilySlotV1(7, "EXPERT", True),
    )


def _evidence(**overrides: object) -> RequiredGameplayEvidenceV1:
    values: dict[str, object] = {
        "anchor_status": "CONFIRMED",
        "anchor_ms": 500,
        "anchor_grid_ms": 500,
        "aggregate_rank": 0.95,
        "prominent_band_count": 2,
        "pulse_support_count": 2,
        "family_slots": _slots(),
        "local_audio_supported": True,
        "reference_first_row_supported": True,
        "repeated_high_confidence_refusal": False,
        "timing_authority_valid": True,
        "timing_authority_digest": _SHA_A,
        "anchor_evidence_digest": _SHA_B,
    }
    values.update(overrides)
    return RequiredGameplayEvidenceV1(**values)  # type: ignore[arg-type]


def _plan(
    evidence: RequiredGameplayEvidenceV1 | None = None,
    **overrides: object,
):
    values: dict[str, object] = {
        "partial_window": PartialRemapWindow(0, 4_500),
        "bpm_events": (OsuBpmEvent(0, 120.0),),
        "second_distinct_row_ms": 1_000,
        "duration_ms": 10_000,
        "mode": RequiredGameplayIntervalMode.OBSERVE,
    }
    values.update(overrides)
    return plan_required_gameplay_interval(
        evidence or _evidence(),
        **values,  # type: ignore[arg-type]
    )


def test_interval_rejects_boolean_bounds_and_non_enum_contract_values():
    with pytest.raises(TypeError, match="start_ms"):
        RequiredGameplayIntervalV1(
            True, 100, 1, (RequiredGameplayGroupType.TAP,),
            RequiredGameplayEvidenceClass.BROADBAND_ATTACK, _SHA_A,
            RequiredGameplayIntervalMode.OBSERVE,
        )
    with pytest.raises(TypeError, match="mode"):
        RequiredGameplayIntervalV1(
            0, 100, 1, (RequiredGameplayGroupType.TAP,),
            RequiredGameplayEvidenceClass.BROADBAND_ATTACK, _SHA_A,
            "OBSERVE",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"start_ms": 100, "end_ms": 100}, "start_ms.*end_ms"),
        ({"minimum_complete_groups": 0}, "minimum_complete_groups"),
        ({"allowed_group_types": ()}, "allowed_group_types"),
        ({"allowed_group_types": ("TAP",)}, "allowed_group_types"),
        ({"evidence_digest": "not-a-sha"}, "evidence_digest"),
    ],
)
def test_interval_rejects_malformed_contract(overrides, message):
    values = {
        "start_ms": 0,
        "end_ms": 100,
        "minimum_complete_groups": 1,
        "allowed_group_types": (RequiredGameplayGroupType.TAP,),
        "evidence_class": RequiredGameplayEvidenceClass.BROADBAND_ATTACK,
        "evidence_digest": _SHA_A,
        "mode": RequiredGameplayIntervalMode.OBSERVE,
    }
    values.update(overrides)

    with pytest.raises((TypeError, ValueError), match=message):
        RequiredGameplayIntervalV1(**values)


def test_family_slots_reject_duplicate_key_difficulty_votes():
    duplicated = (
        RequiredGameplayFamilySlotV1(4, "HARD", True),
        RequiredGameplayFamilySlotV1(4, "HARD", False),
    )

    with pytest.raises(ValueError, match="duplicate family slot"):
        _evidence(family_slots=duplicated)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"anchor_ms": True}, "anchor_ms"),
        ({"aggregate_rank": math.nan}, "aggregate_rank"),
        ({"aggregate_rank": 1.01}, "aggregate_rank"),
        ({"prominent_band_count": -1}, "prominent_band_count"),
        ({"pulse_support_count": 5}, "pulse_support_count"),
        ({"local_audio_supported": 1}, "local_audio_supported"),
        ({"timing_authority_digest": ""}, "timing_authority_digest"),
    ],
)
def test_evidence_rejects_malformed_values(overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _evidence(**overrides)


def test_constant_tempo_advances_four_beats():
    assert advance_tempo_map_beats(1_000, 4.0, (OsuBpmEvent(0, 120.0),)) == 3_000


def test_speedup_integrates_across_the_tempo_boundary():
    # 750ms at 120 BPM consumes 1.5 beats; 2.5 beats at 240 BPM consume 625ms.
    events = (OsuBpmEvent(0, 120.0), OsuBpmEvent(750, 240.0))

    assert advance_tempo_map_beats(0, 4.0, events) == 1_375


def test_slowdown_integrates_across_the_tempo_boundary():
    # 250ms at 240 BPM consumes 1 beat; 1 beat at 60 BPM consumes 1000ms.
    events = (OsuBpmEvent(0, 240.0), OsuBpmEvent(1_250, 60.0))

    assert advance_tempo_map_beats(1_000, 2.0, events) == 2_250


def test_tempo_change_exactly_at_start_uses_the_new_tempo():
    events = (OsuBpmEvent(0, 120.0), OsuBpmEvent(1_000, 180.0))

    assert advance_tempo_map_beats(1_000, 1.0, events) == 1_333


def test_fractional_beat_rounds_half_up_once_at_the_result_boundary():
    assert advance_tempo_map_beats(0, 0.125, (OsuBpmEvent(0, 120.0),)) == 63


def test_tempo_map_addressability_distinguishes_scope_from_validity():
    events = (OsuBpmEvent(500, 120.0), OsuBpmEvent(1_000, 180.0))

    assert tempo_map_addresses(499, events) is False
    assert tempo_map_addresses(500, events) is True


def test_tempo_map_addressability_rejects_malformed_events_before_scope_result():
    with pytest.raises(ValueError, match="positive"):
        tempo_map_addresses(0, (OsuBpmEvent(500, 0.0),))


@pytest.mark.parametrize(
    "events",
    [
        (),
        (OsuBpmEvent(0, 0.0),),
        (OsuBpmEvent(0, math.inf),),
        (OsuBpmEvent(500, 120.0),),
        (OsuBpmEvent(0, 120.0), OsuBpmEvent(0, 140.0)),
        (OsuBpmEvent(100, 120.0), OsuBpmEvent(0, 140.0)),
    ],
)
def test_tempo_advancement_rejects_unaddressable_or_malformed_maps(events):
    with pytest.raises(ValueError):
        advance_tempo_map_beats(0, 1.0, events)


def test_evidence_digest_is_order_independent_for_unique_family_slots_and_groups():
    evidence = _evidence()
    reversed_evidence = replace(evidence, family_slots=tuple(reversed(evidence.family_slots)))
    window = PartialRemapWindow(0, 4_500)

    first = required_gameplay_evidence_digest(
        evidence,
        partial_window=window,
        allowed_group_types=(
            RequiredGameplayGroupType.TAP,
            RequiredGameplayGroupType.HOLD_START,
        ),
    )
    second = required_gameplay_evidence_digest(
        reversed_evidence,
        partial_window=window,
        allowed_group_types=(
            RequiredGameplayGroupType.HOLD_START,
            RequiredGameplayGroupType.TAP,
        ),
    )

    assert first == second
    assert len(first) == 64


def test_evidence_digest_changes_when_semantic_evidence_changes():
    first = required_gameplay_evidence_digest(
        _evidence(), partial_window=PartialRemapWindow(0, 4_500)
    )
    second = required_gameplay_evidence_digest(
        _evidence(anchor_ms=501), partial_window=PartialRemapWindow(0, 4_500)
    )

    assert first != second


def test_evidence_payload_has_an_explicit_identity_free_schema():
    payload = required_gameplay_evidence_payload(
        _evidence(), partial_window=PartialRemapWindow(0, 4_500)
    )
    serialized = json.dumps(payload, sort_keys=True)

    assert set(payload) == {
        "allowedGroupTypes",
        "anchorEvidence",
        "minimumCompleteGroups",
        "partialWindow",
        "policyVersion",
        "timingAuthorityDigest",
    }
    assert not any(word in serialized.lower() for word in ("title", "path", "seed", "songid"))


def test_broadband_evidence_activates_a_bounded_interval():
    decision = _plan()

    assert decision.reason == "BROADBAND_ATTACK_SUPPORTED"
    assert decision.interval is not None
    assert decision.interval.start_ms == 430
    assert decision.interval.end_ms == 570
    assert decision.interval.evidence_class is RequiredGameplayEvidenceClass.BROADBAND_ATTACK
    assert decision.interval.mode is RequiredGameplayIntervalMode.OBSERVE


def test_pulse_family_evidence_activates_without_a_broadband_attack():
    decision = _plan(_evidence(aggregate_rank=0.5, prominent_band_count=0))

    assert decision.reason == "PULSE_FAMILY_CORROBORATED"
    assert decision.interval is not None
    assert (
        decision.interval.evidence_class
        is RequiredGameplayEvidenceClass.PULSE_FAMILY_CORROBORATED
    )


@pytest.mark.parametrize(
    ("evidence", "reason"),
    [
        (_evidence(anchor_status="UNCERTAIN"), "ANCHOR_NOT_CONFIRMED"),
        (_evidence(anchor_ms=571), "ANCHOR_GRID_UNSUPPORTED"),
        (_evidence(local_audio_supported=False), "LOCAL_AUDIO_UNSUPPORTED"),
        (
            _evidence(reference_first_row_supported=False),
            "REFERENCE_FIRST_ROW_UNSUPPORTED",
        ),
        (
            _evidence(repeated_high_confidence_refusal=True),
            "REPEATED_MODEL_REFUSAL",
        ),
        (_evidence(timing_authority_valid=False), "TIMING_AUTHORITY_INVALID"),
    ],
)
def test_common_vetoes_do_not_activate(evidence, reason):
    decision = _plan(evidence)

    assert decision.interval is None
    assert decision.reason == reason


def test_one_key_mode_family_support_cannot_activate_pulse_evidence():
    slots = (
        RequiredGameplayFamilySlotV1(4, "EASY", True),
        RequiredGameplayFamilySlotV1(4, "NORMAL", True),
        RequiredGameplayFamilySlotV1(4, "HARD", True),
        RequiredGameplayFamilySlotV1(6, "EXPERT", False),
    )
    decision = _plan(
        _evidence(
            aggregate_rank=0.5,
            prominent_band_count=0,
            family_slots=slots,
        )
    )

    assert decision.interval is None
    assert decision.reason == "FAMILY_KEY_MODE_SUPPORT_INSUFFICIENT"


def test_two_of_twelve_family_support_cannot_activate_pulse_evidence():
    slots = tuple(
        RequiredGameplayFamilySlotV1(key_mode, difficulty, index < 2)
        for index, (key_mode, difficulty) in enumerate(
            (key_mode, difficulty)
            for key_mode in (4, 6, 7)
            for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
        )
    )
    decision = _plan(
        _evidence(
            aggregate_rank=0.5,
            prominent_band_count=0,
            family_slots=slots,
        )
    )

    assert decision.interval is None
    assert decision.reason == "FAMILY_SLOT_SUPPORT_INSUFFICIENT"


def test_variable_bpm_partial_window_is_checked_by_integrated_beats():
    events = (OsuBpmEvent(0, 120.0), OsuBpmEvent(750, 240.0))

    accepted = _plan(
        partial_window=PartialRemapWindow(0, 1_375),
        bpm_events=events,
        second_distinct_row_ms=0,
        duration_ms=4_000,
    )
    rejected = _plan(
        partial_window=PartialRemapWindow(0, 1_374),
        bpm_events=events,
        second_distinct_row_ms=0,
        duration_ms=4_000,
    )

    assert accepted.interval is not None
    assert rejected.interval is None
    assert rejected.reason == "PARTIAL_WINDOW_TOO_SHORT"


def test_near_full_partial_window_is_vetoed_at_eighty_percent():
    decision = _plan(partial_window=PartialRemapWindow(0, 8_000))

    assert decision.interval is None
    assert decision.reason == "PARTIAL_WINDOW_TOO_LARGE"


def test_interval_outside_the_partial_window_is_declined_after_clipping():
    decision = _plan(
        _evidence(anchor_ms=500, anchor_grid_ms=500),
        partial_window=PartialRemapWindow(1_000, 4_500),
    )

    assert decision.interval is None
    assert decision.reason == "REQUIRED_INTERVAL_OUTSIDE_PARTIAL_WINDOW"
