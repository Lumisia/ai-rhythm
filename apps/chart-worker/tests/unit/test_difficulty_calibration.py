from dataclasses import replace

import pytest

from chart_worker.analysis.chart_profile import build_chart_quality_profile
from chart_worker.schema.note import NoteEvent
from chart_worker.validation.difficulty_calibration import (
    CALIBRATION_FEATURE_NAMES_V3,
    CalibrationFeatureV3,
    FrozenCandidateFeatureSourceV3,
    PairwisePreferenceV3,
    PairwiseQualityPreferenceV3,
    build_calibration_feature_from_frozen_source_v3,
    build_calibration_feature_v3,
    fit_pairwise_calibration_v3,
    group_disjoint_folds,
    project_pairwise_preferences_v3,
)
from chart_worker.validation.family_evidence_v3 import (
    CandidateFamilyEvidenceV3,
    CandidateSafetyEvidenceV3,
)
from chart_worker.validation.mania_star_evidence import ManiaStarEvidenceV1
from chart_worker.validation.pairwise_labels import (
    CandidateLabelBindingV1,
    PairwiseLabelV1,
    build_pairwise_task,
)

FEATURE_NAMES = ("projectRating", "peakSkill")
SCHEMA_SHA = "1" * 64


def _feature(
    candidate_id: str,
    audio: str,
    values: tuple[float, ...],
    *,
    provenance: str = "PRIMARY",
) -> CalibrationFeatureV3:
    return CalibrationFeatureV3(
        candidate_id=candidate_id,
        audio_sha256=audio,
        key_mode=4,
        provenance=provenance,
        feature_schema_sha256=SCHEMA_SHA,
        source_evidence_sha256="0" * 64,
        feature_names=FEATURE_NAMES,
        values=values,
    )


def test_group_folds_never_split_one_audio_and_are_order_invariant():
    groups = ("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64, "f" * 64)

    first = group_disjoint_folds(groups, n_splits=3, seed="fold-v1")
    second = group_disjoint_folds(tuple(reversed(groups)), n_splits=3, seed="fold-v1")

    assert first == second
    assert set(first) == set(groups)
    assert set(first.values()) == {0, 1, 2}


def test_pairwise_model_learns_simple_order_without_using_validation_song():
    train_a = "a" * 64
    train_b = "b" * 64
    held_out = "c" * 64
    train_d = "d" * 64
    features = (
        _feature("a-low", train_a, (1.0, 1.0)),
        _feature("a-high", train_a, (4.0, 4.0)),
        _feature("b-low", train_b, (2.0, 2.0)),
        _feature("b-high", train_b, (5.0, 5.0)),
        _feature("c-low", held_out, (1.5, 1.5)),
        _feature("c-high", held_out, (4.5, 4.5)),
        _feature("d-low", train_d, (1.25, 2.5)),
        _feature("d-high", train_d, (4.25, 5.5)),
    )
    preferences = (
        PairwisePreferenceV3(train_a, "a-high", "a-low", 5),
        PairwisePreferenceV3(train_b, "b-high", "b-low", 4),
        PairwisePreferenceV3(held_out, "c-high", "c-low", 5),
        PairwisePreferenceV3(train_d, "d-high", "d-low", 5),
    )

    result = fit_pairwise_calibration_v3(
        key_mode=4,
        features=features,
        preferences=preferences,
        validation_audio_sha256=(held_out,),
        label_sha256="2" * 64,
    )

    assert result.model.training_audio_sha256 == (train_a, train_b, train_d)
    assert result.model.validation_audio_sha256 == (held_out,)
    low = result.model.predict(features[4])
    high = result.model.predict(features[5])
    assert low.state == high.state == "IN_DOMAIN"
    assert high.score > low.score
    assert result.validation_pair_count == 1
    assert result.validation_total_pair_count == 1
    assert result.validation_unknown_pair_count == 0
    assert result.training_pair_count == 3
    assert result.training_audio_group_count == 3
    assert result.validation_audio_group_count == 1
    assert result.effective_parameter_count == 2
    assert result.validation_disagreement_rate == 0.0
    assert result.activation_state == "REPORT_ONLY"


def test_training_rejects_audio_leakage_and_hash_mismatch():
    audio = "a" * 64
    features = (
        _feature("low", audio, (1.0, 1.0)),
        _feature("high", audio, (3.0, 3.0)),
    )
    preferences = (PairwisePreferenceV3(audio, "high", "low", 4),)

    with pytest.raises(ValueError, match="validation audio cannot train"):
        fit_pairwise_calibration_v3(
            key_mode=4,
            features=features,
            preferences=preferences,
            validation_audio_sha256=(audio,),
            label_sha256="2" * 64,
        )
    with pytest.raises(ValueError, match="feature schema"):
        fit_pairwise_calibration_v3(
            key_mode=4,
            features=(features[0], replace(features[1], feature_schema_sha256="3" * 64)),
            preferences=preferences,
            validation_audio_sha256=(),
            label_sha256="2" * 64,
        )


def test_fit_rejects_more_parameters_than_independent_training_pairs():
    audio_a = "a" * 64
    audio_b = "b" * 64
    features = (
        _feature("a-low", audio_a, (1.0, 1.0)),
        _feature("a-high", audio_a, (3.0, 3.0)),
        _feature("b-low", audio_b, (1.5, 1.0)),
        _feature("b-high", audio_b, (3.5, 3.0)),
    )

    with pytest.raises(ValueError, match="insufficient independent pairwise evidence"):
        fit_pairwise_calibration_v3(
            key_mode=4,
            features=features,
            preferences=(
                PairwisePreferenceV3(audio_a, "a-high", "a-low", 5),
                PairwisePreferenceV3(audio_b, "b-high", "b-low", 5),
            ),
            validation_audio_sha256=(),
            label_sha256="2" * 64,
        )


def test_missing_provenance_and_out_of_domain_feature_fail_closed():
    audio_a = "a" * 64
    audio_b = "b" * 64
    audio_c = "c" * 64
    features = (
        _feature("low", audio_a, (1.0, 1.0)),
        _feature("high", audio_a, (3.0, 3.0)),
        _feature("b-low", audio_b, (1.5, 1.5)),
        _feature("b-high", audio_b, (2.5, 2.5)),
        _feature("c-low", audio_c, (1.25, 1.75)),
        _feature("c-high", audio_c, (2.75, 2.25)),
    )
    model = fit_pairwise_calibration_v3(
        key_mode=4,
        features=features,
        preferences=(
            PairwisePreferenceV3(audio_a, "high", "low", 4),
            PairwisePreferenceV3(audio_b, "b-high", "b-low", 4),
            PairwisePreferenceV3(audio_c, "c-high", "c-low", 4),
        ),
        validation_audio_sha256=(),
        label_sha256="2" * 64,
    ).model

    unsupported = model.predict(
        _feature("fallback", "c" * 64, (2.0, 2.0), provenance="SAFE_FALLBACK")
    )
    out_of_domain = model.predict(_feature("extreme", "c" * 64, (30.0, 30.0)))

    assert unsupported.state == "UNKNOWN"
    assert out_of_domain.state == "UNKNOWN"
    assert unsupported.score is out_of_domain.score is None


def test_model_rejects_wrong_feature_schema_instead_of_clipping():
    audio = "a" * 64
    audio_b = "b" * 64
    audio_c = "c" * 64
    features = (
        _feature("low", audio, (1.0, 1.0)),
        _feature("high", audio, (3.0, 3.0)),
        _feature("b-low", audio_b, (1.25, 1.75)),
        _feature("b-high", audio_b, (3.25, 3.75)),
        _feature("c-low", audio_c, (1.5, 1.25)),
        _feature("c-high", audio_c, (3.5, 3.25)),
    )
    model = fit_pairwise_calibration_v3(
        key_mode=4,
        features=features,
        preferences=(
            PairwisePreferenceV3(audio, "high", "low", 4),
            PairwisePreferenceV3(audio_b, "b-high", "b-low", 4),
            PairwisePreferenceV3(audio_c, "c-high", "c-low", 4),
        ),
        validation_audio_sha256=(),
        label_sha256="2" * 64,
    ).model

    with pytest.raises(ValueError, match="feature schema"):
        model.predict(replace(features[0], feature_schema_sha256="9" * 64))


def test_validation_reports_unknown_pairs_instead_of_hiding_them_from_denominator():
    train_audio = "a" * 64
    held_out = "b" * 64
    train_c = "c" * 64
    train_d = "d" * 64
    features = (
        _feature("train-low", train_audio, (1.0, 1.0)),
        _feature("train-high", train_audio, (3.0, 3.0)),
        _feature("held-low", held_out, (10.0, 10.0)),
        _feature("held-high", held_out, (11.0, 11.0)),
        _feature("c-low", train_c, (1.25, 1.5)),
        _feature("c-high", train_c, (3.25, 3.5)),
        _feature("d-low", train_d, (1.5, 1.25)),
        _feature("d-high", train_d, (3.5, 3.25)),
    )

    result = fit_pairwise_calibration_v3(
        key_mode=4,
        features=features,
        preferences=(
            PairwisePreferenceV3(train_audio, "train-high", "train-low", 4),
            PairwisePreferenceV3(held_out, "held-high", "held-low", 4),
            PairwisePreferenceV3(train_c, "c-high", "c-low", 4),
            PairwisePreferenceV3(train_d, "d-high", "d-low", 4),
        ),
        validation_audio_sha256=(held_out,),
        label_sha256="2" * 64,
    )

    assert result.validation_total_pair_count == 1
    assert result.validation_pair_count == 0
    assert result.validation_unknown_pair_count == 1
    assert result.validation_disagreement_rate is None


def test_projection_keeps_difficulty_when_rater_prefers_other_side_quality():
    audio = "a" * 64
    left_feature = _feature("left", audio, (3.0, 3.0))
    right_feature = _feature("right", audio, (1.0, 1.0))
    task = build_pairwise_task(
        CandidateLabelBindingV1(
            "left",
            audio,
            4,
            "a" * 64,
            left_feature.stable_sha256(),
        ),
        CandidateLabelBindingV1(
            "right",
            audio,
            4,
            "b" * 64,
            right_feature.stable_sha256(),
        ),
        presentation_seed="seed",
        force_left_candidate_id="left",
    )
    label = PairwiseLabelV1(
        task_sha256=task.stable_sha256(),
        rater_sha256="f" * 64,
        harder_answer="LEFT",
        musical_quality_answer="RIGHT",
        confidence=5,
    )

    projection = project_pairwise_preferences_v3(
        ((task, label),),
        features=(left_feature, right_feature),
    )

    assert projection.preferences == (PairwisePreferenceV3(audio, "left", "right", 5),)
    assert projection.quality_preferences == (
        PairwiseQualityPreferenceV3(audio, "right", "left", 5),
    )
    assert projection.quality_conflict_count == 1
    assert projection.difficulty_uncertain_or_tie_count == 0
    assert projection.quality_uncertain_or_tie_count == 0


def test_projection_accepts_quality_tie_and_rejects_feature_hash_mismatch():
    audio = "a" * 64
    left_feature = _feature("left", audio, (3.0, 3.0))
    right_feature = _feature("right", audio, (1.0, 1.0))
    left_binding = CandidateLabelBindingV1(
        "left",
        audio,
        4,
        "a" * 64,
        left_feature.stable_sha256(),
    )
    right_binding = CandidateLabelBindingV1(
        "right",
        audio,
        4,
        "b" * 64,
        right_feature.stable_sha256(),
    )
    task = build_pairwise_task(
        left_binding,
        right_binding,
        presentation_seed="seed",
        force_left_candidate_id="left",
    )
    label = PairwiseLabelV1(
        task.stable_sha256(),
        "f" * 64,
        "LEFT",
        "TIE",
        4,
    )

    projection = project_pairwise_preferences_v3(
        ((task, label),),
        features=(left_feature, right_feature),
    )

    assert projection.preferences == (PairwisePreferenceV3(audio, "left", "right", 4),)
    assert projection.quality_preferences == ()
    assert projection.difficulty_uncertain_or_tie_count == 0
    assert projection.quality_uncertain_or_tie_count == 1
    assert len(projection.label_sha256) == 64
    with pytest.raises(ValueError, match="feature digest"):
        project_pairwise_preferences_v3(
            ((replace(task, left=replace(left_binding, feature_sha256="9" * 64)), label),),
            features=(left_feature, right_feature),
        )


def _candidate_evidence(payload_sha256: str) -> CandidateFamilyEvidenceV3:
    safety = CandidateSafetyEvidenceV3(
        candidate_id="candidate-a",
        structure_safe=True,
        timing_identity_safe=True,
        song_bounds_safe=True,
        serialization_safe=True,
        publication_tier="PRODUCTION_CANDIDATE",
        model_backed=True,
        recovery_trust_rank=0,
        active_gaps=(),
    )
    return CandidateFamilyEvidenceV3(
        candidate_id="candidate-a",
        key_mode=4,
        difficulty="HARD",
        provenance="PRIMARY",
        candidate_payload_ref=f"raw/candidates/sha256/{payload_sha256}.osu",
        candidate_payload_sha256=payload_sha256,
        safety=safety,
        first_row_ms=1_000,
        first_row_audio_supported=True,
        first_row_grid_distance_ms=0,
        intro_reference_state="CONFIRMED_AUDIO",
        matched_f1_50=0.9,
        matched_precision_50=0.9,
        review_rank=0,
    )


def _verified_star(payload_sha256: str) -> ManiaStarEvidenceV1:
    return ManiaStarEvidenceV1(
        input_osu_sha256=payload_sha256,
        tool_binary_sha256="1" * 64,
        osu_tools_source_commit="2" * 40,
        osu_source_commit="3" * 40,
        calculator_version=20241007,
        star_rating=2.5,
        attributes_sha256="4" * 64,
        mods=(),
        verification_state="VERIFIED_PINNED_TOOL_EXECUTION",
    )


def test_fixed_feature_builder_binds_candidate_profile_and_verified_official_input():
    payload_sha = "a" * 64
    profile = build_chart_quality_profile(
        [
            NoteEvent(time_ms=1_000, lane=0),
            NoteEvent(time_ms=1_250, lane=1, kind="HOLD", duration_ms=750),
            NoteEvent(time_ms=2_000, lane=2),
        ],
        key_mode=4,
        duration_ms=5_000,
        beat_ms=500.0,
        activity=None,
    )

    feature = build_calibration_feature_v3(
        candidate=_candidate_evidence(payload_sha),
        profile=profile,
        official_star=_verified_star(payload_sha),
        audio_sha256="b" * 64,
    )

    assert feature.feature_names == CALIBRATION_FEATURE_NAMES_V3
    assert feature.values[0] == 2.5
    assert len(feature.values) == len(CALIBRATION_FEATURE_NAMES_V3)
    assert feature.source_evidence_sha256 != payload_sha
    assert len(feature.source_evidence_sha256) == 64


def test_feature_builder_rejects_unverified_or_different_raw_osu_evidence():
    payload_sha = "a" * 64
    candidate = _candidate_evidence(payload_sha)
    profile = build_chart_quality_profile(
        [NoteEvent(time_ms=1_000, lane=0)],
        key_mode=4,
        duration_ms=2_000,
        beat_ms=500.0,
        activity=None,
    )
    verified = _verified_star(payload_sha)

    with pytest.raises(ValueError, match="verified pinned"):
        build_calibration_feature_v3(
            candidate=candidate,
            profile=profile,
            official_star=replace(
                verified,
                verification_state="PINNED_TOOL_OUTPUT_UNVERIFIED",
            ),
            audio_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="payload digest"):
        build_calibration_feature_v3(
            candidate=candidate,
            profile=profile,
            official_star=replace(verified, input_osu_sha256="c" * 64),
            audio_sha256="b" * 64,
        )


def test_frozen_historical_source_is_distinct_and_bound_to_report_and_payload():
    payload_sha = "a" * 64
    source = FrozenCandidateFeatureSourceV3(
        candidate_id="historical-candidate",
        key_mode=6,
        provenance="SAFE_FALLBACK",
        candidate_payload_sha256=payload_sha,
        generation_report_sha256="c" * 64,
    )
    profile = build_chart_quality_profile(
        [
            NoteEvent(time_ms=1_000, lane=0),
            NoteEvent(time_ms=1_500, lane=1, kind="HOLD", duration_ms=500),
        ],
        key_mode=6,
        duration_ms=3_000,
        beat_ms=500.0,
        activity=None,
    )

    feature = build_calibration_feature_from_frozen_source_v3(
        source=source,
        profile=profile,
        official_star=_verified_star(payload_sha),
        audio_sha256="b" * 64,
    )

    assert feature.candidate_id == "historical-candidate"
    assert feature.key_mode == 6
    assert feature.provenance == "SAFE_FALLBACK"
    assert feature.values[0] == 2.5
    assert feature.source_evidence_sha256 not in {
        payload_sha,
        source.generation_report_sha256,
    }
    with pytest.raises(ValueError, match="payload digest"):
        build_calibration_feature_from_frozen_source_v3(
            source=source,
            profile=profile,
            official_star=replace(_verified_star(payload_sha), input_osu_sha256="d" * 64),
            audio_sha256="b" * 64,
        )
