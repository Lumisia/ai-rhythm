from chart_worker.analysis.grid_alignment import TempoCandidateMetrics
from chart_worker.validation.timing_review import TimingAuthorityAction, review_timing_authority


def _metrics(**changes):
    values = {
        "base_pulse_support": 0.5,
        "half_pulse_support": 0.8,
        "double_pulse_support": 0.2,
        "base_supported_pulses": 20,
        "half_supported_pulses": 24,
        "double_supported_pulses": 0,
        "pulse_best_alternative": "HALF",
        "pulse_alternative_margin": 0.3,
        "base_periodicity_support": 0.3,
        "half_periodicity_support": 0.6,
        "double_periodicity_support": 0.0,
        "periodicity_frame_count": 300,
        "periodicity_best_alternative": "HALF",
        "periodicity_margin": 0.3,
        "evidence_agrees": True,
        "evidence_status": "SUFFICIENT",
    }
    values.update(changes)
    return TempoCandidateMetrics(**values)


def test_same_strong_half_tempo_evidence_requests_one_retry():
    review = review_timing_authority(_metrics())

    assert review.action is TimingAuthorityAction.RETRY_TIMING
    assert review.reasons == ("STRONG_HALF_TEMPO_ALTERNATIVE",)


def test_timing_authority_review_report_serializes_action_and_reasons():
    review = review_timing_authority(
        _metrics(evidence_status="INSUFFICIENT")
    )

    assert review.to_report() == {
        "action": "REVIEW",
        "reasons": ["INSUFFICIENT_TEMPO_EVIDENCE"],
    }


def test_close_high_quality_candidates_pass_with_ambiguity_diagnostic():
    review = review_timing_authority(
        _metrics(
            base_pulse_support=0.7,
            half_pulse_support=0.73,
            pulse_alternative_margin=0.03,
            base_periodicity_support=0.7,
            half_periodicity_support=0.72,
            periodicity_margin=0.02,
        )
    )

    assert review.action is TimingAuthorityAction.PASS
    assert review.reasons == ("TEMPO_CANDIDATE_AMBIGUOUS",)


def test_insufficient_evidence_requires_review():
    review = review_timing_authority(_metrics(evidence_status="INSUFFICIENT"))

    assert review.action is TimingAuthorityAction.REVIEW
    assert review.reasons == ("INSUFFICIENT_TEMPO_EVIDENCE",)


def test_disagreeing_or_single_strong_axes_require_review():
    disagreeing = review_timing_authority(
        _metrics(periodicity_best_alternative="DOUBLE")
    )
    one_strong = review_timing_authority(
        _metrics(periodicity_margin=0.01)
    )

    assert disagreeing.action is TimingAuthorityAction.REVIEW
    assert disagreeing.reasons == ("TEMPO_EVIDENCE_DISAGREES",)
    assert one_strong.action is TimingAuthorityAction.REVIEW
    assert one_strong.reasons == ("TEMPO_EVIDENCE_DISAGREES",)


def test_weak_disagreeing_axes_require_review_even_without_strong_margins():
    review = review_timing_authority(
        _metrics(
            base_pulse_support=0.7,
            half_pulse_support=0.72,
            pulse_alternative_margin=0.02,
            base_periodicity_support=0.5,
            half_periodicity_support=0.49,
            double_periodicity_support=0.52,
            periodicity_best_alternative="DOUBLE",
            periodicity_margin=0.02,
        )
    )

    assert review.action is TimingAuthorityAction.REVIEW
    assert review.reasons == ("TEMPO_EVIDENCE_DISAGREES",)


def test_low_base_pulse_support_without_strong_corroboration_requires_review():
    review = review_timing_authority(
        _metrics(
            base_pulse_support=0.54,
            half_pulse_support=0.56,
            pulse_alternative_margin=0.02,
            base_periodicity_support=0.5,
            half_periodicity_support=0.52,
            periodicity_margin=0.02,
        )
    )

    assert review.action is TimingAuthorityAction.REVIEW
    assert review.reasons == ("WEAK_BASE_TEMPO_SUPPORT",)


def test_exact_strong_margins_are_inclusive_for_half_and_double_retries():
    half = review_timing_authority(
        _metrics(
            half_pulse_support=0.65,
            pulse_alternative_margin=0.15,
            half_periodicity_support=0.4,
            periodicity_margin=0.10,
        )
    )
    double = review_timing_authority(
        _metrics(
            half_pulse_support=0.2,
            double_pulse_support=0.65,
            pulse_best_alternative="DOUBLE",
            pulse_alternative_margin=0.15,
            half_periodicity_support=0.0,
            double_periodicity_support=0.4,
            periodicity_best_alternative="DOUBLE",
            periodicity_margin=0.10,
        )
    )

    assert half.reasons == ("STRONG_HALF_TEMPO_ALTERNATIVE",)
    assert double.action is TimingAuthorityAction.RETRY_TIMING
    assert double.reasons == ("STRONG_DOUBLE_TEMPO_ALTERNATIVE",)
