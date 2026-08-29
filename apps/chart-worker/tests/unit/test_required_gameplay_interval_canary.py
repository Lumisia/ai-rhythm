import importlib
import sys
from pathlib import Path

import pytest

from chart_worker.generation.generation_origin_diagnostics import (
    GameplayGroupStageCount,
    GenerationOriginDiagnostics,
)
from chart_worker.generation.params import GenerationRequest
from chart_worker.hashing import sha256_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_canary = importlib.import_module("scripts.run_required_gameplay_interval_canary")
CanaryComparison = _canary.CanaryComparison
ShadowEvaluation = _canary.ShadowEvaluation
QualitySummary = _canary.QualitySummary
canonical_chart_summary = _canary.canonical_chart_summary
classify_observation = _canary.classify_observation
compare_canary_runs = _canary.compare_canary_runs
evaluate_shadow_run = _canary.evaluate_shadow_run
run_canary = _canary.run_canary
stage_registered_inputs = _canary._stage_registered_inputs


def test_shadow_report_schema_version_changes_when_quality_fields_are_added() -> None:
    assert _canary.SHADOW_REPORT_VERSION == "required-gameplay-shadow-canary-report-v4"
    assert (
        _canary._report_filename(include_shadow=True)
        == "required-gameplay-shadow-canary-report-v4.json"
    )
    assert (
        _canary._report_filename(include_shadow=False)
        == "required-gameplay-canary-report-v1.json"
    )


def _write_osu(path: Path, *, timing: str = "0,500,4,2,1,100,1,0") -> Path:
    path.write_text(
        f"""osu file format v14

[General]
Mode:3

[Difficulty]
CircleSize:4

[TimingPoints]
{timing}

[HitObjects]
64,192,479,1,0,0:0:0:0:
192,192,800,128,0,1000:0:0:0:0:
320,192,18000,1,0,0:0:0:0:
""",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _count(value: int) -> GameplayGroupStageCount:
    return GameplayGroupStageCount(
        total_generated_complete_groups=value,
        interval_generated_complete_groups=value,
        tap_groups=value,
        hold_start_groups=0,
    )


def _diagnostics(
    counts: tuple[int, int, int, int, int],
    *,
    first_loss_stage: str | None,
) -> GenerationOriginDiagnostics:
    return GenerationOriginDiagnostics(
        evidence_digest="a" * 64,
        invocation_digest="b" * 64,
        decoder=_count(counts[0]),
        window_merge=_count(counts[1]),
        canonical=_count(counts[2]),
        resnap=_count(counts[3]),
        final_serialization=_count(counts[4]),
        first_loss_stage=first_loss_stage,  # type: ignore[arg-type]
    )


def _quality(
    *,
    action: str = "REVIEW",
    disposition: str = "REVIEW",
    leading_gap_count: int = 0,
    leading_gap_total_ms: int = 0,
    precision: float = 0.75,
    f1: float = 0.70,
    rating: float = 3.0,
) -> QualitySummary:
    return QualitySummary(
        action=action,
        disposition=disposition,
        first_note_time_ms=479,
        active_leading_gap_count=leading_gap_count,
        active_leading_gap_total_ms=leading_gap_total_ms,
        matched_precision_50=precision,
        matched_f1_50=f1,
        project_rating=rating,
    )


def test_chart_summary_is_path_free_and_counts_complete_groups(tmp_path: Path) -> None:
    first = _write_osu(tmp_path / "first.osu")
    second = _write_osu(tmp_path / "renamed.osu")

    one = canonical_chart_summary(
        first,
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )
    two = canonical_chart_summary(
        second,
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )

    assert one.semantic_sha256 == two.semantic_sha256
    assert one.suffix_semantic_sha256 == two.suffix_semantic_sha256
    assert one.note_count == 3
    assert one.hold_count == 1
    assert one.interval_complete_group_count == 1
    assert one.timing_section_sha256 == two.timing_section_sha256


def test_chart_summary_preserves_raw_timing_section_evidence(tmp_path: Path) -> None:
    first = _write_osu(tmp_path / "first.osu")
    second = _write_osu(
        tmp_path / "second.osu",
        timing="0,500.0,4,2,1,100,1,0",
    )

    one = canonical_chart_summary(
        first,
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )
    two = canonical_chart_summary(
        second,
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )

    assert one.semantic_sha256 == two.semantic_sha256
    assert one.timing_section_sha256 != two.timing_section_sha256
    assert one.timing_semantic_sha256 == two.timing_semantic_sha256


def test_chart_summary_timing_semantics_detect_inherited_sv_changes(
    tmp_path: Path,
) -> None:
    first = _write_osu(tmp_path / "first.osu")
    second = _write_osu(
        tmp_path / "second.osu",
        timing="0,500,4,2,1,100,1,0\n500,-100,4,2,1,100,0,0",
    )

    one = canonical_chart_summary(
        first,
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )
    two = canonical_chart_summary(
        second,
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )

    assert one.timing_semantic_sha256 != two.timing_semantic_sha256


def test_registered_inputs_are_staged_beneath_the_resident_job_root(
    tmp_path: Path,
) -> None:
    source = _write_osu(tmp_path / "source.osu")
    timing = _write_osu(tmp_path / "timing.osu")
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"frozen audio fixture")
    hashes = {
        "audioSha256": sha256_file(audio),
        "sourceCandidateSha256": sha256_file(source),
        "timingAuthoritySha256": sha256_file(timing),
    }
    request = GenerationRequest(
        audio_path=audio,
        timing_reference_path=source,
        key_mode=4,
        difficulty="EXPERT",
        seed=15,
        duration_ms=20_000,
        partial_start_ms=0,
        partial_end_ms=10_000,
        add_to_beatmap=True,
    )
    registration = {"input": {"timingAuthorityPath": str(timing.resolve())}}
    output_root = tmp_path / "canary"
    output_root.mkdir()

    staged = stage_registered_inputs(registration, request, output_root, hashes)

    assert staged.audio_path.is_relative_to(output_root)
    assert staged.timing_reference_path.is_relative_to(output_root)
    assert sha256_file(staged.audio_path) == hashes["audioSha256"]
    assert sha256_file(staged.timing_reference_path) == hashes["sourceCandidateSha256"]
    assert sha256_file(output_root / "input" / "timing-authority.osu") == hashes[
        "timingAuthoritySha256"
    ]


def test_run_canary_rejects_cached_report_for_another_registration(
    tmp_path: Path,
) -> None:
    first_registration = tmp_path / "registration-a.json"
    second_registration = tmp_path / "registration-b.json"
    first_registration.write_text('{"fixture":"a"}\n', encoding="utf-8")
    second_registration.write_text('{"fixture":"b"}\n', encoding="utf-8")
    output_root = tmp_path / "canary"
    output_root.mkdir()
    report_path = output_root / _canary._report_filename(include_shadow=False)
    report_path.write_text(
        "{\n"
        f'  "version": "{_canary.REPORT_VERSION}",\n'
        f'  "registrationSha256": "{sha256_file(first_registration)}",\n'
        '  "frozenInputs": {},\n'
        '  "runtime": {}\n'
        "}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different registration"):
        run_canary(second_registration, output_root)


@pytest.mark.parametrize(
    ("diagnostics", "expected"),
    (
        (_diagnostics((0, 0, 0, 0, 0), first_loss_stage=None), "CONTENT_OBLIGATION_UNSATISFIED"),
        (_diagnostics((1, 0, 0, 0, 0), first_loss_stage="windowMerge"), "WINDOW_MERGE_DELETION"),
        (_diagnostics((1, 1, 0, 0, 0), first_loss_stage="canonical"), "CANONICAL_DELETION"),
        (_diagnostics((1, 1, 1, 0, 0), first_loss_stage="resnap"), "RESNAP_DELETION"),
        (
            _diagnostics((1, 1, 1, 1, 0), first_loss_stage="finalSerialization"),
            "SERIALIZATION_DELETION",
        ),
        (_diagnostics((1, 1, 1, 1, 1), first_loss_stage=None), "NO_OBSERVED_LOSS"),
    ),
)
def test_classification_selects_one_stage(
    diagnostics: GenerationOriginDiagnostics,
    expected: str,
) -> None:
    assert classify_observation(diagnostics) == expected


def test_comparison_rejects_observe_output_interference(tmp_path: Path) -> None:
    off = canonical_chart_summary(
        _write_osu(tmp_path / "off.osu"),
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )
    observe_path = _write_osu(tmp_path / "observe.osu")
    observe_path.write_text(
        observe_path.read_text(encoding="utf-8").replace(
            "320,192,18000,1,0,0:0:0:0:",
            "448,192,18000,1,0,0:0:0:0:",
        ),
        encoding="utf-8",
        newline="\n",
    )
    observe = canonical_chart_summary(
        observe_path,
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )

    comparison = compare_canary_runs(
        off,
        observe,
        off_failure_class=None,
        observe_failure_class=None,
        off_exit_code=0,
        observe_exit_code=0,
        off_invocation_count=1,
        observe_invocation_count=1,
    )

    assert comparison == CanaryComparison(
        equivalent=False,
        classification="OBSERVE_INTERFERENCE",
        mismatches=("OSU_BYTES", "SEMANTIC_DIGEST", "SUFFIX_SEMANTIC_DIGEST"),
    )


def test_comparison_accepts_identical_successes(tmp_path: Path) -> None:
    off = canonical_chart_summary(
        _write_osu(tmp_path / "off.osu"),
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )
    observe = canonical_chart_summary(
        _write_osu(tmp_path / "observe.osu"),
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )

    comparison = compare_canary_runs(
        off,
        observe,
        off_failure_class=None,
        observe_failure_class=None,
        off_exit_code=0,
        observe_exit_code=0,
        off_invocation_count=1,
        observe_invocation_count=1,
    )

    assert comparison.equivalent is True
    assert comparison.classification == "EQUIVALENT"
    assert comparison.mismatches == ()


def test_comparison_never_treats_matching_execution_failures_as_equivalent() -> None:
    comparison = compare_canary_runs(
        None,
        None,
        off_failure_class="PermissionError",
        observe_failure_class="PermissionError",
        off_exit_code=1,
        observe_exit_code=1,
        off_invocation_count=0,
        observe_invocation_count=0,
    )

    assert comparison.equivalent is False
    assert comparison.classification == "CANARY_EXECUTION_FAILURE"
    assert comparison.mismatches == (
        "OFF_EXIT_CODE",
        "OFF_FAILURE_CLASS",
        "OFF_INVOCATION_COUNT",
        "OFF_MISSING_CHART",
        "OBSERVE_EXIT_CODE",
        "OBSERVE_FAILURE_CLASS",
        "OBSERVE_INVOCATION_COUNT",
        "OBSERVE_MISSING_CHART",
    )


def test_shadow_evaluation_requires_origin_group_and_preserved_suffix_and_timing(
    tmp_path: Path,
) -> None:
    baseline = canonical_chart_summary(
        _write_osu(tmp_path / "baseline.osu"),
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )
    shadow = canonical_chart_summary(
        _write_osu(tmp_path / "shadow.osu"),
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )

    result = evaluate_shadow_run(
        baseline=baseline,
        shadow=shadow,
        reference_timing_semantic_sha256=baseline.timing_semantic_sha256,
        source_quality=_quality(leading_gap_count=1, leading_gap_total_ms=12_000),
        shadow_quality=_quality(),
        diagnostics=_diagnostics((1, 1, 1, 1, 1), first_loss_stage=None),
        minimum_complete_groups=1,
        exit_code=0,
        failure_class=None,
        failure_context=None,
        invocation_count=1,
    )

    assert result == ShadowEvaluation(
        status="CONTRACT_PASS",
        isolation_pass=True,
        failures=(),
        typed_failure_reason=None,
    )


def test_shadow_evaluation_accepts_only_validated_typed_failure_isolation(
    tmp_path: Path,
) -> None:
    baseline = canonical_chart_summary(
        _write_osu(tmp_path / "baseline.osu"),
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )

    result = evaluate_shadow_run(
        baseline=baseline,
        shadow=None,
        reference_timing_semantic_sha256=baseline.timing_semantic_sha256,
        source_quality=_quality(),
        shadow_quality=None,
        diagnostics=None,
        minimum_complete_groups=1,
        exit_code=1,
        failure_class="MANIA_REQUIRED_GAMEPLAY_FAILED",
        failure_context={"reason": "REQUIRED_GAMEPLAY_INTERVAL_NO_LEGAL_GROUP"},
        invocation_count=1,
    )

    assert result.status == "TYPED_FAILURE"
    assert result.isolation_pass is True
    assert result.typed_failure_reason == "REQUIRED_GAMEPLAY_INTERVAL_NO_LEGAL_GROUP"
    assert result.failures == ()


def test_shadow_evaluation_rejects_unstructured_or_suffix_regressing_result(
    tmp_path: Path,
) -> None:
    baseline = canonical_chart_summary(
        _write_osu(tmp_path / "baseline.osu"),
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )
    changed_path = _write_osu(tmp_path / "changed.osu")
    changed_path.write_text(
        changed_path.read_text(encoding="utf-8").replace(
            "320,192,18000,1,0,0:0:0:0:",
            "448,192,18000,1,0,0:0:0:0:",
        ),
        encoding="utf-8",
        newline="\n",
    )
    changed = canonical_chart_summary(
        changed_path,
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )

    result = evaluate_shadow_run(
        baseline=baseline,
        shadow=changed,
        reference_timing_semantic_sha256=baseline.timing_semantic_sha256,
        source_quality=_quality(),
        shadow_quality=_quality(),
        diagnostics=_diagnostics((1, 1, 1, 1, 1), first_loss_stage=None),
        minimum_complete_groups=1,
        exit_code=0,
        failure_class=None,
        failure_context=None,
        invocation_count=1,
    )

    assert result.status == "CONTRACT_FAIL"
    assert result.isolation_pass is False
    assert result.failures == ("SUFFIX_SEMANTIC_DIGEST",)


def test_shadow_evaluation_compares_timing_to_frozen_source_not_off_arm(
    tmp_path: Path,
) -> None:
    source = canonical_chart_summary(
        _write_osu(tmp_path / "source.osu"),
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )
    baseline = canonical_chart_summary(
        _write_osu(
            tmp_path / "baseline.osu",
            timing="0,500,4,2,1,100,1,0\n500,-100,4,2,1,100,0,0",
        ),
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )
    shadow = canonical_chart_summary(
        _write_osu(
            tmp_path / "shadow.osu",
            timing="0,500.000000000000,4,2,1,100,1,0",
        ),
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )

    result = evaluate_shadow_run(
        baseline=baseline,
        shadow=shadow,
        reference_timing_semantic_sha256=source.timing_semantic_sha256,
        source_quality=_quality(leading_gap_count=1, leading_gap_total_ms=12_000),
        shadow_quality=_quality(),
        diagnostics=_diagnostics((1, 1, 1, 1, 1), first_loss_stage=None),
        minimum_complete_groups=1,
        exit_code=0,
        failure_class=None,
        failure_context=None,
        invocation_count=1,
    )

    assert baseline.timing_semantic_sha256 != source.timing_semantic_sha256
    assert shadow.timing_section_sha256 != source.timing_section_sha256
    assert result.status == "CONTRACT_PASS"
    assert result.failures == ()


@pytest.mark.parametrize(
    ("shadow_quality", "expected_failure"),
    (
        (_quality(action="RETRY_MAP", disposition="QUALITY_DEFECT"), "QUALITY_GATE_RETRY"),
        (_quality(leading_gap_count=1, leading_gap_total_ms=8_000), "ACTIVE_LEADING_GAP"),
        (_quality(precision=0.744), "MATCHED_PRECISION_50_REGRESSION"),
        (_quality(f1=0.694), "MATCHED_F1_50_REGRESSION"),
        (_quality(rating=2.99), "PROJECT_RATING_REGRESSION"),
    ),
)
def test_shadow_evaluation_rejects_protected_quality_regressions(
    tmp_path: Path,
    shadow_quality: QualitySummary,
    expected_failure: str,
) -> None:
    baseline = canonical_chart_summary(
        _write_osu(tmp_path / "baseline.osu"),
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )
    shadow = canonical_chart_summary(
        _write_osu(tmp_path / "shadow.osu"),
        interval_start_ms=409,
        interval_end_ms=549,
        partial_end_ms=17_624,
    )

    result = evaluate_shadow_run(
        baseline=baseline,
        shadow=shadow,
        reference_timing_semantic_sha256=baseline.timing_semantic_sha256,
        source_quality=_quality(),
        shadow_quality=shadow_quality,
        diagnostics=_diagnostics((1, 1, 1, 1, 1), first_loss_stage=None),
        minimum_complete_groups=1,
        exit_code=0,
        failure_class=None,
        failure_context=None,
        invocation_count=1,
    )

    assert result.status == "CONTRACT_FAIL"
    assert expected_failure in result.failures
