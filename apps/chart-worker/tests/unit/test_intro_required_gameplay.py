from __future__ import annotations

from dataclasses import replace

from chart_worker.generation.intro_required_gameplay import (
    plan_intro_required_gameplay_interval,
)
from chart_worker.generation.partial_remap import PartialRemapWindow
from chart_worker.generation.required_gameplay_interval import (
    RequiredGameplayEvidenceClass,
    RequiredGameplayIntervalMode,
)
from chart_worker.validation.intro_region_contract import IntroRegionContract

_SHA_A = "a" * 64
_SHA_B = "b" * 64


def _contract() -> IntroRegionContract:
    return IntroRegionContract(
        version="intro-region-contract-v1",
        status="CONFIRMED",
        allowed_first_row_ms=(0, 993),
        leading_silence_end_ms=None,
        anchor_ms=21,
        anchor_grid_ms=0,
        supported_pulse_ms=(0, 231, 462, 692, 923),
        quantization_tolerance_ms=70,
        reasons=("CONFIRMED_AUDIO_GRID_PULSE_SEQUENCE",),
    )


def test_confirmed_region_becomes_one_generated_group_obligation() -> None:
    decision = plan_intro_required_gameplay_interval(
        _contract(),
        partial_window=PartialRemapWindow(0, 5_000),
        timing_authority_digest=_SHA_A,
        mode=RequiredGameplayIntervalMode.SHADOW_ENFORCE,
    )

    assert decision.reason == "INTRO_REGION_CORROBORATED"
    assert decision.interval is not None
    assert (decision.interval.start_ms, decision.interval.end_ms) == (0, 993)
    assert decision.interval.minimum_complete_groups == 1
    assert (
        decision.interval.evidence_class
        is RequiredGameplayEvidenceClass.INTRO_REGION_CORROBORATED
    )


def test_unknown_or_truncated_region_never_authorizes_decoder_enforcement() -> None:
    unknown = plan_intro_required_gameplay_interval(
        replace(_contract(), status="UNKNOWN", allowed_first_row_ms=None),
        partial_window=PartialRemapWindow(0, 5_000),
        timing_authority_digest=_SHA_A,
        mode=RequiredGameplayIntervalMode.SHADOW_ENFORCE,
    )
    truncated = plan_intro_required_gameplay_interval(
        _contract(),
        partial_window=PartialRemapWindow(0, 900),
        timing_authority_digest=_SHA_A,
        mode=RequiredGameplayIntervalMode.SHADOW_ENFORCE,
    )

    assert unknown.interval is None
    assert unknown.reason == "INTRO_REGION_NOT_CONFIRMED"
    assert truncated.interval is None
    assert truncated.reason == "INTRO_REGION_OUTSIDE_PARTIAL_WINDOW"


def test_digest_binds_timing_and_region_evidence() -> None:
    first = plan_intro_required_gameplay_interval(
        _contract(),
        partial_window=PartialRemapWindow(0, 5_000),
        timing_authority_digest=_SHA_A,
        mode=RequiredGameplayIntervalMode.OBSERVE,
    ).interval
    changed_timing = plan_intro_required_gameplay_interval(
        _contract(),
        partial_window=PartialRemapWindow(0, 5_000),
        timing_authority_digest=_SHA_B,
        mode=RequiredGameplayIntervalMode.OBSERVE,
    ).interval
    changed_region = plan_intro_required_gameplay_interval(
        replace(_contract(), allowed_first_row_ms=(0, 1_000)),
        partial_window=PartialRemapWindow(0, 5_000),
        timing_authority_digest=_SHA_A,
        mode=RequiredGameplayIntervalMode.OBSERVE,
    ).interval

    assert first is not None and changed_timing is not None and changed_region is not None
    assert first.evidence_digest != changed_timing.evidence_digest
    assert first.evidence_digest != changed_region.evidence_digest
