from dataclasses import replace

from chart_worker.analysis.grid_alignment import TempoCandidateMetrics
from chart_worker.analysis.local_timing import LocalTimingSegmentMetrics
from chart_worker.validation.local_timing_review import (
    LocalTimingAuthorityReview,
    LocalTimingSegmentReview,
)
from chart_worker.validation.timing_candidate_selector import (
    build_timing_candidate_evidence,
    select_timing_candidate,
    timing_candidates_need_external_corroboration,
)
from chart_worker.validation.timing_review import TimingAuthorityAction


def _tempo() -> TempoCandidateMetrics:
    return TempoCandidateMetrics(
        base_pulse_support=0.8,
        half_pulse_support=0.4,
        double_pulse_support=0.3,
        base_supported_pulses=80,
        half_supported_pulses=40,
        double_supported_pulses=30,
        pulse_best_alternative=None,
        pulse_alternative_margin=-0.4,
        base_periodicity_support=0.7,
        half_periodicity_support=0.3,
        double_periodicity_support=0.2,
        periodicity_frame_count=100,
        periodicity_best_alternative=None,
        periodicity_margin=-0.4,
        evidence_agrees=False,
        evidence_status="SUFFICIENT",
    )


def _segment(
    index: int,
    start_ms: int,
    end_ms: int,
    *,
    contradicted: bool = False,
    sufficient: bool = True,
) -> LocalTimingSegmentReview:
    metrics = LocalTimingSegmentMetrics(
        index=index,
        start_ms=start_ms,
        end_ms=end_ms,
        bpm=120.0,
        onset_count=20 if sufficient else 2,
        active_onset_count=20 if sufficient else 2,
        active_frame_ratio=1.0,
        active_confident=sufficient,
        current_grid_support=0.2 if contradicted else 0.8,
        neighbor_grid_support=0.8 if contradicted else 0.4,
        current_residual_p95_ms=150.0 if contradicted else 10.0,
        neighbor_residual_p95_ms=10.0,
        isolated_metrical_outlier=contradicted,
        pulse_conflict=contradicted,
        phase_conflict=contradicted,
        evidence_status="SUFFICIENT" if sufficient else "INSUFFICIENT",
    )
    return LocalTimingSegmentReview(
        metrics=metrics,
        grid_damage=contradicted,
        action=(
            TimingAuthorityAction.RETRY_TIMING
            if contradicted
            else (
                TimingAuthorityAction.PASS
                if sufficient
                else TimingAuthorityAction.REVIEW
            )
        ),
        reasons=(),
    )


def _review(*segments: LocalTimingSegmentReview) -> LocalTimingAuthorityReview:
    return LocalTimingAuthorityReview(
        action=TimingAuthorityAction.REVIEW,
        reasons=(),
        segments=segments,
    )


def test_selector_uses_contradicted_duration_not_segment_count():
    short_damage = build_timing_candidate_evidence(
        epoch=1,
        mode="STANDARD",
        structurally_valid=True,
        local_review=_review(
            _segment(0, 0, 8_000, contradicted=True),
            _segment(1, 8_000, 80_000),
        ),
        tempo_metrics=_tempo(),
    )
    long_damage = build_timing_candidate_evidence(
        epoch=1,
        mode="SUPER_TIMING",
        structurally_valid=True,
        local_review=_review(
            _segment(0, 0, 20_000, contradicted=True),
            _segment(1, 20_000, 40_000),
        ),
        tempo_metrics=_tempo(),
    )

    selection = select_timing_candidate((short_damage, long_damage))

    assert short_damage.contradicted_active_ms == 8_000
    assert short_damage.contradicted_ratio == 0.1
    assert long_damage.contradicted_active_ms == 20_000
    assert long_damage.contradicted_ratio == 0.5
    assert selection.selected_index == 0
    assert selection.reason == "LOWER_CONTRADICTED_ACTIVE_RATIO"


def test_insufficient_evidence_is_confidence_not_candidate_rejection():
    insufficient = build_timing_candidate_evidence(
        epoch=1,
        mode="STANDARD",
        structurally_valid=True,
        local_review=_review(_segment(0, 0, 90_000, sufficient=False)),
        tempo_metrics=_tempo(),
    )

    selection = select_timing_candidate((insufficient,))

    assert insufficient.insufficient_active_ms == 90_000
    assert insufficient.confidence == "LOW"
    assert selection.selected_index == 0


def test_metrical_level_consensus_is_reported_without_bpm_number_rejection():
    evidence = build_timing_candidate_evidence(
        epoch=1,
        mode="STANDARD",
        structurally_valid=True,
        local_review=_review(_segment(0, 0, 60_000)),
        tempo_metrics=_tempo(),
    )

    assert evidence.beat_consensus_by_level == {
        "HALF": 0.3,
        "BASE": 0.7,
        "DOUBLE": 0.2,
    }
    assert evidence.best_metrical_level == "BASE"
    assert evidence.structurally_valid is True


def test_close_internal_candidates_use_external_beat_f1_as_a_tie_breaker():
    standard = build_timing_candidate_evidence(
        epoch=1,
        mode="STANDARD",
        structurally_valid=True,
        local_review=_review(_segment(0, 0, 60_000)),
        tempo_metrics=_tempo(),
    )
    super_timing = replace(standard, mode="SUPER_TIMING")
    assert timing_candidates_need_external_corroboration((standard, super_timing))

    selection = select_timing_candidate(
        (
            replace(
                standard,
                external_beat_f1_by_level={"HALF": 0.90, "BASE": 0.65, "DOUBLE": 0.39},
                best_external_beat_f1=0.90,
            ),
            replace(
                super_timing,
                external_beat_f1_by_level={"HALF": 0.92, "BASE": 0.66, "DOUBLE": 0.40},
                best_external_beat_f1=0.92,
            ),
        )
    )

    assert selection.selected_index == 1
    assert selection.reason == "HIGHER_EXTERNAL_BEAT_F1"


def test_clear_internal_winner_does_not_request_the_optional_model():
    standard = build_timing_candidate_evidence(
        epoch=1,
        mode="STANDARD",
        structurally_valid=True,
        local_review=_review(_segment(0, 0, 60_000)),
        tempo_metrics=_tempo(),
    )
    super_timing = replace(
        standard,
        mode="SUPER_TIMING",
        beat_consensus_by_level={"HALF": 0.1, "BASE": 0.1, "DOUBLE": 0.1},
        best_metrical_level="BASE",
    )

    assert not timing_candidates_need_external_corroboration(
        (standard, super_timing)
    )
