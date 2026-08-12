from chart_worker.analysis.timing_diagnostics import (
    TimingDiagnostics,
    TimingMetrics,
    TimingSection,
)
from chart_worker.validation.timing_family_review import (
    TimingFamilyCandidate,
    review_timing_family,
)


def _metrics(rows: int, matched_precision: float) -> TimingMetrics:
    return TimingMetrics(
        row_count=rows,
        precision_20=matched_precision,
        precision_50=matched_precision,
        signed_median_ms=0.0,
        absolute_p95_ms=0.0,
        absolute_p99_ms=0.0,
        matched_count_50=round(rows * matched_precision),
        matched_precision_50=matched_precision,
        matched_recall_50=matched_precision,
        matched_f1_50=matched_precision,
        onset_reuse_inflation_50=0.0,
    )


def _diagnostics(overall: float, sections: tuple[tuple[int, float], ...]) -> TimingDiagnostics:
    return TimingDiagnostics(
        status="REVIEW" if overall < 0.7 else "PASS",
        onset_count=500,
        active_onset_count=500,
        first_note_time_ms=0,
        max_gap_ms=500,
        coverage_gaps=(),
        quiet_coverage_gaps=(),
        overall=_metrics(sum(rows for rows, _ in sections), overall),
        sections=tuple(
            TimingSection(
                start_ms=index * 15_000,
                end_ms=(index + 1) * 15_000,
                status="PASS",
                metrics=_metrics(rows, precision),
                phase_delta_ms=0.0,
            )
            for index, (rows, precision) in enumerate(sections)
        ),
    )


def _candidate(key_mode: int, overall: float, sections: tuple[tuple[int, float], ...]):
    return TimingFamilyCandidate(
        key_mode=key_mode,
        difficulty="EXPERT",
        diagnostics=_diagnostics(overall, sections),
    )


def test_flags_only_the_cross_key_local_density_outlier():
    review = review_timing_family(
        (
            _candidate(4, 0.62, ((70, 0.8), (105, 0.42), (110, 0.40))),
            _candidate(6, 0.72, ((68, 0.8), (70, 0.70), (65, 0.72))),
            _candidate(7, 0.73, ((66, 0.8), (68, 0.72), (67, 0.74))),
        )
    )

    assert review.status == "OUTLIER"
    assert review.target_key_mode == 4
    assert review.overall_sibling_gap == 0.105
    assert review.longest_local_outlier_run == 2
    assert review.outlier_section_indices == (1, 2)


def test_common_low_onset_support_is_review_evidence_not_a_family_outlier():
    review = review_timing_family(
        tuple(
            _candidate(key_mode, 0.40, ((100, 0.2), (100, 0.45)))
            for key_mode in (4, 6, 7)
        )
    )

    assert review.status == "CONSISTENT"
    assert review.target_key_mode is None


def test_normal_key_mode_bias_below_the_measured_tail_is_not_retried():
    review = review_timing_family(
        (
            _candidate(4, 0.66, ((80, 0.55), (80, 0.55))),
            _candidate(6, 0.70, ((60, 0.75), (60, 0.75))),
            _candidate(7, 0.71, ((60, 0.75), (60, 0.75))),
        )
    )

    assert review.status == "CONSISTENT"


def test_one_local_section_is_not_enough_to_spend_song_inference_budget():
    review = review_timing_family(
        (
            _candidate(4, 0.60, ((60, 0.80), (100, 0.40), (60, 0.80))),
            _candidate(6, 0.72, ((60, 0.80), (60, 0.75), (60, 0.80))),
            _candidate(7, 0.73, ((60, 0.80), (60, 0.75), (60, 0.80))),
        )
    )

    assert review.status == "CONSISTENT"
    assert review.longest_local_outlier_run == 1
