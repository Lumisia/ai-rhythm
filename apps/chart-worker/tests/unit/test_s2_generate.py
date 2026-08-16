import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chart_worker.analysis.activity import AudioActivity, SongBoundaryContract
from chart_worker.analysis.intro_anchor import IntroAnchorEvidence
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.timing_diagnostics import (
    TimingCoverageGap,
    TimingMetrics,
    TimingSection,
)
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation import family_selection
from chart_worker.generation.candidate_repository import CandidateRepository
from chart_worker.generation.generation_control import (
    MAX_CRASH_ATTEMPTS,
    MAX_TOTAL_ATTEMPTS,
    MAX_VARIANT_ATTEMPTS,
    AttemptBudgetState,
)
from chart_worker.generation.intro_exact_reselection import try_exact_intro_candidate
from chart_worker.generation.intro_family_recovery import intro_phrase_pair_review
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent, parse_osu_file
from chart_worker.generation.osu_writer import timing_to_osu_mania
from chart_worker.generation.params import GenerationRequest
from chart_worker.generation.resnap_diagnostics import (
    ResnapCollision,
    ResnapDiagnostics,
)
from chart_worker.hashing import sha256_file
from chart_worker.schema.note import NoteEvent
from chart_worker.stages import s2_generate
from chart_worker.stages.s2_generate import run_generation
from chart_worker.stages.timing_feedback import RetryTimingSignal
from chart_worker.stages.types import (
    AdditionalInferenceBudget,
    PreparedAudio,
    SongTimingAuthority,
)
from chart_worker.validation.leading_timing_coverage import LeadingTimingCoverage
from chart_worker.validation.quality_gate import (
    GateAction,
    GateAxis,
    evaluate_chart_candidate,
)
from chart_worker.validation.timing_review import TimingAuthorityAction


def _run_generation(*args, **kwargs):
    """run_generation 은 GenerationOutcome 을 돌려준다.

    이 파일의 기존 검증은 발행된 변형 튜플만 보므로 여기서 풀어 준다.
    missing 을 확인하는 테스트는 run_generation 을 직접 호출한다.
    """
    return run_generation(*args, **kwargs).variants


def _prepared(
    tmp_path: Path,
    *,
    duration_ms: int = 2_000,
    boundary_policy_mode: str = "SHADOW",
) -> PreparedAudio:
    audio_path = tmp_path / "audio" / "game.flac"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"audio")
    return PreparedAudio(
        normalized=NormalizedAudio(
            audio_path,
            "audio-profile-v2",
            sha256_file(audio_path),
            duration_ms,
            48_000,
            2,
            duration_ms,
            0,
            0.0,
            -14.0,
            -1.0,
            0.0,
            "LOUDNESS",
        ),
        boundary_policy_mode=boundary_policy_mode,
    )


def _authority(
    prepared: PreparedAudio,
    tmp_path: Path,
    bpm_events: tuple[OsuBpmEvent, ...] = (OsuBpmEvent(0, 120.0),),
) -> SongTimingAuthority:
    reference_path = tmp_path / "audio" / "timing-reference.osu"
    reference_path.write_text(
        timing_to_osu_mania(
            bpm_events,
            audio_filename=prepared.normalized.path.name,
            title="fixture",
        ),
        encoding="utf-8",
    )
    return SongTimingAuthority(
        reference_path=reference_path,
        sha256=sha256_file(reference_path),
        audio_sha256=prepared.normalized.sha256,
        bpm_events=bpm_events,
        generator_name="recording-generator",
        seed=17,
        mode="STANDARD",
        attempt_count=1,
    )


def _analysis() -> OnsetAnalysis:
    rows = tuple(range(125, 2_000, 125))
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=np.zeros(21),
        band_strength=np.zeros((3, 21)),
        onset_ms=rows,
    )


@pytest.mark.parametrize(
    ("duration_ms", "bpm_events"),
    [
        pytest.param(96_523, (OsuBpmEvent(0, 60.0),), id="short-slow"),
        pytest.param(
            178_902,
            (OsuBpmEvent(0, 120.0), OsuBpmEvent(89_451, 180.0)),
            id="median-speedup",
        ),
        pytest.param(
            361_929,
            (
                OsuBpmEvent(0, 240.0),
                OsuBpmEvent(120_000, 90.0),
                OsuBpmEvent(240_000, 160.0),
            ),
            id="long-multi-tempo",
        ),
    ],
)
def test_safe_fallback_hard_contract_across_corpus_duration_and_bpm_extremes(
    monkeypatch,
    tmp_path: Path,
    duration_ms: int,
    bpm_events: tuple[OsuBpmEvent, ...],
):
    case_dir = tmp_path / str(duration_ms)
    prepared = _prepared(case_dir, duration_ms=duration_ms)
    authority = _authority(prepared, case_dir, bpm_events)
    frame_count = duration_ms // 100 + 1
    analysis = OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=np.zeros(frame_count),
        band_strength=np.zeros((3, frame_count)),
        onset_ms=tuple(range(0, duration_ms, 250)),
    )
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        evaluate_chart_candidate,
    )

    candidates = []
    for key_index, key_mode in enumerate((4, 6, 7)):
        for difficulty_index, difficulty in enumerate(
            ("EASY", "NORMAL", "HARD", "EXPERT")
        ):
            state = s2_generate._VariantState(
                key_mode,
                difficulty,
                key_index * 4 + difficulty_index,
            )
            candidates.append(
                s2_generate._safe_fallback_candidate(
                    state,
                    prepared=prepared,
                    authority=authority,
                    onset_analysis=analysis,
                    run_dir=case_dir,
                    base_seed=7,
                    authority_epoch=1,
                )
            )

    assert {(item.request.key_mode, item.request.difficulty) for item in candidates} == {
        (key_mode, difficulty)
        for key_mode in (4, 6, 7)
        for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
    }
    assert all(item.provenance == "SAFE_FALLBACK" for item in candidates)
    assert all(item.generated.bpm_events == bpm_events for item in candidates)
    assert all(
        item.acceptance.decision(axis).action is GateAction.PASS
        for item in candidates
        for axis in (GateAxis.STRUCTURE, GateAxis.TIMING_IDENTITY, GateAxis.SONG_BOUNDS)
    )
    assert all(
        0 <= note.time_ms < duration_ms
        and note.time_ms + (note.duration_ms if note.kind == "HOLD" else 0)
        <= duration_ms
        for item in candidates
        for note in item.generated.notes
    )


def test_song_selection_context_changes_with_boundary_policy(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    boundary = SongBoundaryContract(
        version="song-boundary-contract-v2",
        last_attack_ms=1_000,
        max_note_start_ms=1_070,
        release_end_ms=2_000,
        generation_end_ms=2_000,
        required_coverage_end_ms=1_000,
        quantization_tolerance_ms=70,
        hold_completion_guard_ms=10_000,
        policy_reason="SHORT_ABSOLUTE_SILENCE",
        policy_applied=True,
    )

    base = s2_generate._song_selection_context_id(
        prepared,
        authority,
        boundary,
    )
    changed = s2_generate._song_selection_context_id(
        prepared,
        authority,
        replace(boundary, required_coverage_end_ms=1_001),
    )

    assert base != changed


def _pass_acceptance(authority: SongTimingAuthority):
    rows = tuple(range(125, 2_000, 125))
    chart = GeneratedChart(
        notes=[NoteEvent(row, 0) for row in rows],
        key_mode=4,
        osu_text="",
        generator_name="acceptance-fixture",
        seed=0,
        bpm_events=authority.bpm_events,
    )
    return evaluate_chart_candidate(
        chart,
        authority,
        _analysis(),
        requested_key_mode=4,
        requested_difficulty="EASY",
        duration_ms=2_000,
    )


def _acceptance_with_action(authority: SongTimingAuthority, action: GateAction):
    accepted = _pass_acceptance(authority)
    if action is GateAction.PASS:
        return accepted
    first, *remaining = accepted.decisions
    return replace(
        accepted,
        action=action,
        decisions=(
            replace(first, action=action, reasons=(f"FIXTURE_{action.value}",)),
            *remaining,
        ),
    )


def _coverage_retry_acceptance(authority: SongTimingAuthority):
    accepted = _pass_acceptance(authority)
    gap = TimingCoverageGap(
        start_ms=10_000,
        end_ms=20_000,
        onset_count=20,
        active_onset_count=20,
        active_frame_ratio=1.0,
        position="MIDDLE",
    )
    decisions = tuple(
        replace(
            decision,
            action=GateAction.RETRY_MAP,
            reasons=("ACTIVE_MIDDLE_GAP",),
        )
        if decision.axis is GateAxis.COVERAGE
        else decision
        for decision in accepted.decisions
    )
    return replace(
        accepted,
        action=GateAction.RETRY_MAP,
        decisions=decisions,
        timing=replace(accepted.timing, coverage_gaps=(gap,)),
    )


def _acceptance_for_difficulty(acceptance, difficulty: str):
    ratings = {"EASY": 1.0, "NORMAL": 2.0, "HARD": 3.0, "EXPERT": 4.0}
    assert acceptance.profile is not None
    return replace(
        acceptance,
        profile=replace(
            acceptance.profile,
            difficulty=replace(
                acceptance.profile.difficulty,
                project_rating=ratings[difficulty],
            ),
        ),
    )


def _acceptance_with_rating(acceptance, rating: float):
    assert acceptance.profile is not None
    return replace(
        acceptance,
        profile=replace(
            acceptance.profile,
            difficulty=replace(
                acceptance.profile.difficulty,
                project_rating=rating,
            ),
        ),
    )


@pytest.fixture(autouse=True)
def _accept_candidates_by_default(monkeypatch, tmp_path: Path):
    fixture_dir = tmp_path / "acceptance-fixture"
    prepared = _prepared(fixture_dir)
    authority = _authority(prepared, fixture_dir)
    accepted = _pass_acceptance(authority)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        lambda *args, requested_difficulty, **kwargs: _acceptance_for_difficulty(
            accepted, requested_difficulty
        ),
        raising=False,
    )


class RecordingGenerator:
    def __init__(self):
        self.map_calls = []
        self.map_workdirs = []

    def generate_map(self, request, workdir):
        self.map_calls.append(request)
        self.map_workdirs.append(workdir)
        return GeneratedChart(
            notes=[NoteEvent(500, request.key_mode - 1)],
            key_mode=request.key_mode,
            osu_text="",
            generator_name="recording-fake",
            seed=request.seed,
            bpm_events=(OsuBpmEvent(0, 120.0),),
        )


def test_tail_exhaustion_suppresses_only_the_same_variant_full_seed(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    tail_context = {
        "reason": "END_BOUNDARY_CROSSES_HOLD",
        "signature": "END_BOUNDARY_CROSSES_HOLD:500:11",
        "requestedEndMs": 500,
        "effectiveCutMs": 510,
        "earliestGeneratedSourceWindowId": 11,
        "repairStartWindowId": 9,
        "repairWindowIds": [9, 10, 11],
        "repairAttempts": 2,
    }

    class TailExhaustedEasyGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if (request.key_mode, request.difficulty) == (4, "EASY"):
                raise WorkerError(
                    ErrorCode.MANIA_TAIL_REPAIR_EXHAUSTED,
                    "fixture tail exhausted",
                    context=tail_context,
                )
            return generated

    generator = TailExhaustedEasyGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 12
    assert [
        request.seed
        for request in generator.map_calls
        if (request.key_mode, request.difficulty) == (4, "EASY")
    ] == [0]
    assert len(outcome.variants) == 12
    assert outcome.additional_inference_calls == 0
    assert outcome.missing == ()
    fallback = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert fallback.provenance == "SAFE_FALLBACK"
    assert [json.loads(entry)["code"] for entry in fallback.attempt_errors] == [
        "MANIA_TAIL_REPAIR_EXHAUSTED"
    ]
    assert {
        "reason": "MANIA_TAIL_REPAIR_EXHAUSTED",
        "context": tail_context,
        "outerFullLengthRetrySuppressed": True,
        "totalAttempts": 1,
        "qualityAttempts": 0,
        "crashAttempts": 0,
    } in fallback.attempt_evidence
    journal = [
        json.loads(line)
        for line in (tmp_path / "attempt-journal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failed_variant = [
        entry for entry in journal if (entry["keyMode"], entry["difficulty"]) == (4, "EASY")
    ]
    assert [entry["eventType"] for entry in failed_variant] == [
        "INFERENCE_STARTED",
        "INFERENCE_FAILED",
    ]
    assert failed_variant[-1]["payload"]["error"]["context"] == tail_context


def test_run_generation_creates_exactly_twelve_parseable_variants(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    generator = RecordingGenerator()
    variants = _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=17,
    )

    assert {(variant.key_mode, variant.difficulty) for variant in variants} == {
        (key_mode, difficulty)
        for key_mode in (4, 6, 7)
        for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
    }
    requests = generator.map_calls
    workdirs = generator.map_workdirs
    assert len(requests) == 12
    assert {request.timing_reference_path for request in requests} == {
        authority.reference_path
    }
    assert all(
        variant.timing_authority_sha256 == authority.sha256 for variant in variants
    )
    assert len({request.seed for request in requests}) == 12
    assert [
        (request.key_mode, request.difficulty, request.seed) for request in requests
    ] == [
        (key_mode, difficulty, 17 + index)
        for index, (key_mode, difficulty) in enumerate(
            (key_mode, difficulty)
            for key_mode in (4, 6, 7)
            for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
        )
    ]
    assert len(set(workdirs)) == 12
    assert all(workdir.name == "attempt-1" for workdir in workdirs)
    assert all(workdir.parent.parent.parent.name == "work" for workdir in workdirs)
    assert all("epoch-1" in workdir.parts for workdir in workdirs)
    assert all("candidates" not in workdir.parts for workdir in workdirs)
    assert all(request.duration_ms == 2_000 for request in requests)
    assert all(request.cfg_scale == 1.0 for request in requests)
    assert {
        request.difficulty: request.descriptors
        for request in requests
        if request.key_mode == 4
    } == {
        "EASY": ("expression/simple",),
        "NORMAL": ("style/mixed rice",),
        "HARD": ("style/mixed rice", "streams/bursts"),
        "EXPERT": ("style/mixed rice", "skillset/streams"),
    }
    assert [variant.raw_osu_path.name for variant in variants] == [
        f"{key_mode}k-{difficulty.lower()}.osu"
        for key_mode in (4, 6, 7)
        for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
    ]
    journal = [
        json.loads(line)
        for line in (tmp_path / "attempt-journal.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [entry["eventType"] for entry in journal] == [
        event_type
        for _variant in range(12)
        for event_type in (
            "INFERENCE_STARTED",
            "INFERENCE_COMPLETED",
            "GATE_EVALUATED",
            "CANDIDATE_ADMITTED",
        )
    ]
    assert all(
        entry["payload"]["action"] == "PASS"
        for entry in journal
        if entry["eventType"] == "GATE_EVALUATED"
    )


def test_shadow_selection_adds_no_generator_calls(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    generator = RecordingGenerator()

    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=17,
    )

    assert len(generator.map_calls) == 12
    assert len(outcome.difficulty_selection_shadows) == 3
    assert all(
        comparison.current_assignment == comparison.shadow_assignment
        for comparison in outcome.difficulty_selection_shadows
    )
    assert all(
        parse_osu_file(variant.raw_osu_path).key_mode == variant.key_mode
        for variant in outcome.variants
    )


def test_generation_uses_selector_mode_from_prepared_run_context(
    monkeypatch, tmp_path: Path
):
    prepared = replace(_prepared(tmp_path), difficulty_selector_mode="V2")
    authority = _authority(prepared, tmp_path)
    generator = RecordingGenerator()
    observed_modes: list[str] = []
    real_compare = family_selection.compare_family_candidates

    def record_mode(pools, current_assignment, *, mode):
        observed_modes.append(mode)
        return real_compare(pools, current_assignment, mode=mode)

    monkeypatch.setattr(family_selection, "compare_family_candidates", record_mode)

    run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=17,
    )

    assert observed_modes == ["V2", "V2", "V2"]


def test_song_contract_never_mutates_one_inconsistent_first_row(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    class InconsistentIntroGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            self.map_calls.append(request)
            self.map_workdirs.append(workdir)
            first_row_ms = (
                501
                if (request.key_mode, request.difficulty) == (7, "EXPERT")
                else 500
            )
            return GeneratedChart(
                notes=[NoteEvent(first_row_ms, 0)],
                key_mode=request.key_mode,
                osu_text="",
                generator_name="inconsistent-intro-fixture",
                seed=request.seed,
                bpm_events=authority.bpm_events,
            )

    generator = InconsistentIntroGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=17,
    )

    assert len(generator.map_calls) == 12
    assert {
        min(note.time_ms for note in variant.generated.notes)
        for variant in outcome.variants
    } == {500, 501}
    assert outcome.intro_contract_review is not None
    assert outcome.intro_contract_review.status == "REVIEW"
    assert outcome.intro_contract_review.corrected_count == 0
    unchanged = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (7, "EXPERT")
    )
    assert unchanged.provenance == "PRIMARY"
    assert unchanged.generated.notes == [NoteEvent(501, 0)]
    assert outcome.additional_inference_calls == 0


def test_song_contract_reports_mismatch_without_spending_model_retry(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    authority = replace(
        _authority(prepared, tmp_path, (OsuBpmEvent(1_000, 120.0),)),
        leading_coverage=LeadingTimingCoverage(
            action=TimingAuthorityAction.REVIEW,
            reasons=("CONFIRMED_INTRO_ANCHOR_BEFORE_FIRST_EVENT",),
            first_event_time_ms=1_000,
            leading_duration_ms=1_000,
            onset_count=1,
            active_onset_count=1,
            active_frame_ratio=1.0,
            intro_anchor=IntroAnchorEvidence(
                status="CONFIRMED",
                anchor_ms=500,
                anchor_grid_ms=500,
                grid_distance_ms=0,
                aggregate_percentile_rank=0.99,
                prominent_band_count=3,
                pulse_continuation_matches=3,
                pulse_continuation_opportunities=4,
            ),
        ),
    )

    class LateBlockingMismatchGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            self.map_calls.append(request)
            self.map_workdirs.append(workdir)
            matching_calls = [
                call
                for call in self.map_calls
                if (call.key_mode, call.difficulty) == (7, "HARD")
            ]
            if (request.key_mode, request.difficulty) == (7, "HARD") and len(
                matching_calls
            ) == 2:
                assert (request.key_mode, request.difficulty) == (7, "HARD")
                assert request.add_to_beatmap is False
                return GeneratedChart(
                    notes=[NoteEvent(500, 0), NoteEvent(750, 1)],
                    key_mode=request.key_mode,
                        osu_text="",
                        generator_name="song-contract-recovery-fixture",
                        seed=request.seed,
                        bpm_events=authority.bpm_events,
                    )
            if (request.key_mode, request.difficulty) == (4, "EASY"):
                notes = [NoteEvent(0, 0), NoteEvent(750, 1)]
            elif (request.key_mode, request.difficulty) == (7, "HARD"):
                notes = [NoteEvent(0, 0), NoteEvent(250, 1)]
            else:
                notes = [NoteEvent(500, 0), NoteEvent(750, 1)]
            return GeneratedChart(
                notes=notes,
                key_mode=request.key_mode,
                osu_text="",
                generator_name="song-contract-primary-fixture",
                seed=request.seed,
                bpm_events=authority.bpm_events,
            )

    generator = LateBlockingMismatchGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 12
    assert [
        (request.key_mode, request.difficulty)
        for request in generator.map_calls
    ].count((4, "EASY")) == 1
    assert all(request.add_to_beatmap is False for request in generator.map_calls)
    assert outcome.additional_inference_calls == 0
    assert outcome.intro_contract_review is not None
    assert outcome.intro_contract_review.status == "REVIEW"
    assert outcome.intro_contract_review.corrected_count == 0
    assert {
        min(note.time_ms for note in variant.generated.notes)
        for variant in outcome.variants
    } == {0, 500}
    assert all(variant.provenance != "INTRO_ALIGNED" for variant in outcome.variants)


def test_song_spends_one_shared_recovery_on_a_corroborated_timing_family_outlier(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)

    def metrics(rows: int, precision: float) -> TimingMetrics:
        return replace(
            accepted.timing.overall,
            row_count=rows,
            precision_20=precision,
            precision_50=precision,
            matched_count_50=round(rows * precision),
            matched_precision_50=precision,
            matched_recall_50=precision,
            matched_f1_50=precision,
            onset_reuse_inflation_50=0.0,
        )

    def acceptance(requested_key_mode, requested_difficulty, generated):
        base = _acceptance_for_difficulty(accepted, requested_difficulty)
        bad = (
            requested_key_mode == 4
            and requested_difficulty == "EXPERT"
            and generated.seed == 3
        )
        overall = 0.62 if bad else 0.73
        section_values = (
            ((70, 0.80), (105, 0.42), (110, 0.40))
            if bad
            else ((68, 0.80), (68, 0.72), (67, 0.74))
        )
        sections = tuple(
            TimingSection(
                start_ms=index * 15_000,
                end_ms=(index + 1) * 15_000,
                status="PASS",
                metrics=metrics(rows, precision),
                phase_delta_ms=0.0,
            )
            for index, (rows, precision) in enumerate(section_values)
        )
        return replace(
            base,
            timing=replace(
                base.timing,
                status="REVIEW" if bad else "PASS",
                overall=metrics(sum(rows for rows, _ in section_values), overall),
                sections=sections,
            ),
        )

    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        lambda generated, *args, requested_key_mode, requested_difficulty, **kwargs: (
            acceptance(requested_key_mode, requested_difficulty, generated)
        ),
    )
    generator = RecordingGenerator()

    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 13
    assert [
        request.seed
        for request in generator.map_calls
        if (request.key_mode, request.difficulty) == (4, "EXPERT")
    ] == [3, 15]
    assert outcome.additional_inference_calls == 1
    recovered = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EXPERT")
    )
    assert recovered.selected_seed == 15
    assert recovered.provenance == "RETRY"
    assert recovered.recovery_reason == "TIMING_FAMILY_OUTLIER"
    router_evidence = next(
        evidence
        for evidence in recovered.attempt_evidence
        if evidence["reason"] == "RECOVERY_ROUTER_DECISION"
    )
    assert router_evidence["plan"]["selectedRequestIds"] == [
        "timing:4k:EXPERT"
    ]
    journal = [
        json.loads(line)
        for line in (tmp_path / "attempt-journal.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    recovery_events = [
        entry
        for entry in journal
        if entry["payload"].get("purpose") == "TIMING_FAMILY_RETRY"
    ]
    assert [entry["eventType"] for entry in recovery_events] == [
        "INFERENCE_STARTED",
        "INFERENCE_COMPLETED",
        "GATE_EVALUATED",
        "CANDIDATE_ADMITTED",
    ]
    expert_review = next(
        review
        for review in outcome.timing_family_reviews
        if review.difficulty == "EXPERT"
    )
    assert expert_review.status == "CONSISTENT"


def test_isolated_expert_first_row_uses_one_priority_retry_and_recovers(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    authority = _authority(prepared, tmp_path)

    class RecoveringPhraseGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            self.map_calls.append(request)
            self.map_workdirs.append(workdir)
            matching = [
                call
                for call in self.map_calls
                if (call.key_mode, call.difficulty) == (4, "EXPERT")
            ]
            if (request.key_mode, request.difficulty) == (4, "EXPERT"):
                rows = (0, 12_000) if len(matching) == 1 else (0, 250)
            else:
                rows = (0, 250)
            return GeneratedChart(
                notes=[NoteEvent(row, index % request.key_mode) for index, row in enumerate(rows)],
                key_mode=request.key_mode,
                osu_text="",
                generator_name="phrase-family-recovery-fixture",
                seed=request.seed,
                bpm_events=authority.bpm_events,
            )

    generator = RecoveringPhraseGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 13
    assert outcome.additional_inference_calls == 1
    recovered = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EXPERT")
    )
    assert recovered.selected_seed == 15
    assert recovered.provenance == "RETRY"
    assert recovered.recovery_reason == "INTRO_PHRASE_FAMILY_DEFECT"
    router_evidence = next(
        evidence
        for evidence in recovered.attempt_evidence
        if evidence["reason"] == "RECOVERY_ROUTER_DECISION"
    )
    assert router_evidence["plan"]["selectedRequestIds"] == [
        "intro:4k:EXPERT"
    ]
    review = next(
        review
        for review in outcome.intro_phrase_family_reviews
        if review.hard.key_mode == 4
    )
    assert review.status == "PASS"
    assert review.reason == "CONSISTENT"
    assert any(
        evidence["reason"] == "INTRO_PHRASE_RETRY_SELECTED"
        for evidence in recovered.attempt_evidence
    )


def test_isolated_expert_first_row_reselects_existing_candidate_for_free(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)

    def candidate(difficulty: str, rows: tuple[int, ...], seed: int, attempt: int):
        generated = GeneratedChart(
            notes=[NoteEvent(row, index % 4) for index, row in enumerate(rows)],
            key_mode=4,
            osu_text="",
            generator_name="existing-phrase-candidate-fixture",
            seed=seed,
            bpm_events=authority.bpm_events,
        )
        return s2_generate._Candidate(
            request=GenerationRequest(
                audio_path=prepared.normalized.path,
                timing_reference_path=authority.reference_path,
                key_mode=4,
                difficulty=difficulty,
                seed=seed,
                duration_ms=prepared.normalized.duration_ms,
            ),
            generated=generated,
            acceptance=_acceptance_for_difficulty(accepted, difficulty),
            osu_text="",
            workdir=tmp_path / f"{difficulty.lower()}-{seed}",
            attempt=attempt,
            seed=seed,
            provenance="PRIMARY" if attempt == 1 else "RETRY",
        )

    easy = candidate("EASY", (0, 500), 0, 1)
    normal = candidate("NORMAL", (0, 400), 1, 1)
    hard = candidate("HARD", (0, 250), 2, 1)
    defective = candidate("EXPERT", (0, 12_000), 3, 1)
    replacement = candidate("EXPERT", (0, 125), 15, 2)
    states = {
        difficulty: s2_generate._VariantState(4, difficulty, index)
        for index, difficulty in enumerate(("EASY", "NORMAL", "HARD", "EXPERT"))
    }
    states["EASY"].candidates.admit(easy)
    states["NORMAL"].candidates.admit(normal)
    states["HARD"].candidates.admit(hard)
    states["EXPERT"].candidates.admit(defective)
    states["EXPERT"].candidates.admit(replacement)
    assignment = {
        "EASY": easy,
        "NORMAL": normal,
        "HARD": hard,
        "EXPERT": defective,
    }
    budget = AdditionalInferenceBudget(limit=1)
    context = s2_generate.SongAnalysisContext.build(
        authority,
        _analysis(),
        duration_ms=prepared.normalized.duration_ms,
    )
    generator = RecordingGenerator()

    selections, reviews = s2_generate._apply_intro_phrase_family_recovery(
        [(states, assignment, s2_generate._family_review(assignment))],
        context,
        prepared=prepared,
        authority=authority,
        onset_analysis=_analysis(),
        run_dir=tmp_path,
        generator=generator,
        base_seed=0,
        authority_epoch=1,
        inference_budget=budget,
    )

    assert selections[0][1]["EXPERT"] is replacement
    assert reviews[0].status == "PASS"
    assert budget.used == 0
    assert generator.map_calls == []
    assert any(
        evidence["reason"] == "INTRO_PHRASE_EXISTING_CANDIDATE_RESELECTED"
        for evidence in states["EXPERT"].attempt_evidence
    )


def test_isolated_expert_first_row_accepts_nonblocking_review_replacement(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)

    def candidate(difficulty: str, rows: tuple[int, ...], seed: int, attempt: int):
        generated = GeneratedChart(
            notes=[NoteEvent(row, index % 4) for index, row in enumerate(rows)],
            key_mode=4,
            osu_text="",
            generator_name="review-replacement-fixture",
            seed=seed,
            bpm_events=authority.bpm_events,
        )
        return s2_generate._Candidate(
            request=GenerationRequest(
                audio_path=prepared.normalized.path,
                timing_reference_path=authority.reference_path,
                key_mode=4,
                difficulty=difficulty,
                seed=seed,
                duration_ms=prepared.normalized.duration_ms,
            ),
            generated=generated,
            acceptance=_acceptance_for_difficulty(accepted, difficulty),
            osu_text="",
            workdir=tmp_path / f"{difficulty.lower()}-{seed}",
            attempt=attempt,
            seed=seed,
            provenance="PRIMARY" if attempt == 1 else "RETRY",
        )

    easy = candidate("EASY", (0, 500), 0, 1)
    normal = candidate("NORMAL", (0, 400), 1, 1)
    hard = candidate("HARD", (0, 250), 2, 1)
    defective = candidate("EXPERT", (0, 12_000), 3, 1)
    replacement = candidate("EXPERT", (5_000, 5_250), 15, 2)
    states = {
        difficulty: s2_generate._VariantState(4, difficulty, index)
        for index, difficulty in enumerate(("EASY", "NORMAL", "HARD", "EXPERT"))
    }
    states["EASY"].candidates.admit(easy)
    states["NORMAL"].candidates.admit(normal)
    states["HARD"].candidates.admit(hard)
    states["EXPERT"].candidates.admit(defective)
    states["EXPERT"].candidates.admit(replacement)
    assignment = {
        "EASY": easy,
        "NORMAL": normal,
        "HARD": hard,
        "EXPERT": defective,
    }
    budget = AdditionalInferenceBudget(limit=0)
    context = s2_generate.SongAnalysisContext.build(
        authority,
        _analysis(),
        duration_ms=prepared.normalized.duration_ms,
    )

    selections, reviews = s2_generate._apply_intro_phrase_family_recovery(
        [(states, assignment, s2_generate._family_review(assignment))],
        context,
        prepared=prepared,
        authority=authority,
        onset_analysis=_analysis(),
        run_dir=tmp_path,
        generator=RecordingGenerator(),
        base_seed=0,
        authority_epoch=1,
        inference_budget=budget,
    )

    assert selections[0][1]["EXPERT"] is replacement
    assert reviews[0].status == "REVIEW"
    assert reviews[0].reason == "EXPERT_LATE_START"
    assert reviews[0].should_block_publication is False
    assert budget.used == 0

    after_exact, _contract, _contract_review = s2_generate._apply_intro_start_contract(
        selections,
        context,
    )
    assert after_exact[0][1]["EXPERT"] is replacement
    after_exact_review = intro_phrase_pair_review(
        states,
        after_exact[0][1],
        song_context=context,
        run_dir=tmp_path,
    )
    assert after_exact_review.status == "REVIEW"
    assert after_exact_review.should_block_publication is False


def test_exact_intro_reselection_rejects_new_chart_review_axis(tmp_path: Path):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)

    def candidate(
        difficulty: str,
        rows: tuple[int, ...],
        seed: int,
        *,
        review_pattern: bool = False,
    ):
        acceptance = _acceptance_for_difficulty(accepted, difficulty)
        if review_pattern:
            acceptance = replace(
                acceptance,
                action=GateAction.REVIEW,
                decisions=tuple(
                    replace(
                        decision,
                        action=GateAction.REVIEW,
                        reasons=("FIXTURE_PATTERN_REVIEW",),
                    )
                    if decision.axis is GateAxis.PATTERN
                    else decision
                    for decision in acceptance.decisions
                ),
            )
        generated = GeneratedChart(
            notes=[NoteEvent(row, index % 4) for index, row in enumerate(rows)],
            key_mode=4,
            osu_text="",
            generator_name="exact-intro-quality-regression-fixture",
            seed=seed,
            bpm_events=authority.bpm_events,
        )
        return s2_generate._Candidate(
            request=GenerationRequest(
                audio_path=prepared.normalized.path,
                timing_reference_path=authority.reference_path,
                key_mode=4,
                difficulty=difficulty,
                seed=seed,
                duration_ms=prepared.normalized.duration_ms,
            ),
            generated=generated,
            acceptance=acceptance,
            osu_text="",
            workdir=tmp_path / f"{difficulty.lower()}-{seed}",
            attempt=1,
            seed=seed,
            provenance="PRIMARY",
        )

    easy = candidate("EASY", (0, 500), 0)
    normal = candidate("NORMAL", (0, 400), 1)
    hard = candidate("HARD", (0, 250), 2)
    current = candidate("EXPERT", (125, 250), 3)
    challenger = candidate(
        "EXPERT",
        (0, 125),
        15,
        review_pattern=True,
    )
    states = {
        difficulty: s2_generate._VariantState(4, difficulty, index)
        for index, difficulty in enumerate(("EASY", "NORMAL", "HARD", "EXPERT"))
    }
    states["EXPERT"].candidates.admit(current)
    states["EXPERT"].candidates.admit(challenger)
    assignment = {
        "EASY": easy,
        "NORMAL": normal,
        "HARD": hard,
        "EXPERT": current,
    }

    replacement = try_exact_intro_candidate(
        states["EXPERT"],
        assignment,
        canonical_ms=0,
    )

    assert replacement is None
    assert any(
        evidence["reason"] == "CANDIDATE_REPLACEMENT_POLICY_REJECTED"
        and "NEW_REVIEW_AXIS:PATTERN"
        in evidence["decision"]["reasons"]
        for evidence in states["EXPERT"].attempt_evidence
    )


def test_unresolved_isolated_expert_first_row_is_kept_as_playtest_only(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    authority = _authority(prepared, tmp_path)

    class PersistentPhraseDefectGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            self.map_calls.append(request)
            self.map_workdirs.append(workdir)
            rows = (
                (0, 12_000)
                if (request.key_mode, request.difficulty) == (4, "EXPERT")
                else (0, 250)
            )
            return GeneratedChart(
                notes=[NoteEvent(row, index % request.key_mode) for index, row in enumerate(rows)],
                key_mode=request.key_mode,
                osu_text="",
                generator_name="persistent-phrase-defect-fixture",
                seed=request.seed,
                bpm_events=authority.bpm_events,
            )

    generator = PersistentPhraseDefectGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 13
    assert outcome.additional_inference_calls == 1
    assert len(outcome.variants) == 12
    assert outcome.missing == ()
    expert = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EXPERT")
    )
    assert expert.provenance == "RAW_UNVERIFIED"
    assert expert.recovery_reason == "INTRO_PHRASE_DEFECT_UNRESOLVED"
    assert any(
        evidence["reason"] == "INTRO_PHRASE_DEFECT_PLAYTEST_ONLY"
        for evidence in expert.attempt_evidence
    )


def test_unresolved_intro_keeps_existing_raw_candidate_without_duplicate_identity(
    monkeypatch,
    tmp_path: Path,
):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    soft_retry = replace(
        accepted,
        action=GateAction.RETRY_MAP,
        decisions=tuple(
            replace(
                decision,
                action=GateAction.RETRY_MAP,
                reasons=("FIXTURE_PATTERN_REVIEW",),
            )
            if decision.axis is GateAxis.PATTERN
            else decision
            for decision in accepted.decisions
        ),
    )

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        source = (
            soft_retry
            if generated.key_mode == 4 and requested_difficulty == "EXPERT"
            else accepted
        )
        return _acceptance_for_difficulty(source, requested_difficulty)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)

    class RawPhraseDefectGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            self.map_calls.append(request)
            self.map_workdirs.append(workdir)
            rows = (
                (0, 12_000)
                if (request.key_mode, request.difficulty) == (4, "EXPERT")
                else (0, 250)
            )
            return GeneratedChart(
                notes=[
                    NoteEvent(row, index % request.key_mode)
                    for index, row in enumerate(rows)
                ],
                key_mode=request.key_mode,
                osu_text="",
                generator_name="raw-phrase-defect-fixture",
                seed=request.seed,
                bpm_events=authority.bpm_events,
            )

    generator = RawPhraseDefectGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    expert = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EXPERT")
    )
    assert len(generator.map_calls) == 12
    assert len(outcome.variants) == 12
    assert outcome.missing == ()
    assert expert.provenance == "RAW_UNVERIFIED"
    assert expert.recovery_reason == "QUALITY_GATE_REJECTED"
    assert any(
        evidence.get("reason") == "INTRO_PHRASE_DEFECT_PLAYTEST_ONLY"
        for evidence in expert.attempt_evidence
    )


def test_later_timing_reselection_cannot_reintroduce_intro_phrase_defect(
    monkeypatch,
    tmp_path: Path,
):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    authority = _authority(prepared, tmp_path)

    class StablePhraseGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            self.map_calls.append(request)
            self.map_workdirs.append(workdir)
            return GeneratedChart(
                notes=[NoteEvent(0, 0), NoteEvent(250, 1 % request.key_mode)],
                key_mode=request.key_mode,
                osu_text="",
                generator_name="later-reselection-fixture",
                seed=request.seed,
                bpm_events=authority.bpm_events,
            )

    def timing_reintroduces_defect(selections, **_kwargs):
        updated = list(selections)
        states, original_assignment, _review = updated[0]
        assignment = dict(original_assignment)
        source = assignment["EXPERT"]
        assert source is not None
        defective = replace(
            source,
            generated=replace(
                source.generated,
                notes=[NoteEvent(0, 0), NoteEvent(12_000, 1)],
            ),
            seed=source.seed + 12,
            attempt=source.attempt + 1,
            provenance="RETRY",
        )
        states["EXPERT"].candidates.admit(defective)
        assignment["EXPERT"] = defective
        updated[0] = (states, assignment, s2_generate._family_review(assignment))
        return updated, s2_generate._timing_family_reviews(updated)

    monkeypatch.setattr(
        s2_generate,
        "_apply_timing_family_recovery",
        timing_reintroduces_defect,
    )

    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=StablePhraseGenerator(),
        seed=0,
    )

    expert = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EXPERT")
    )
    assert sorted({note.time_ms for note in expert.generated.notes}) == [0, 250]
    review = next(
        review
        for review in outcome.intro_phrase_family_reviews
        if review.hard.key_mode == 4
    )
    assert review.status == "PASS"
    assert any(
        evidence["reason"] == "INTRO_PHRASE_EXISTING_CANDIDATE_RESELECTED"
        for evidence in expert.attempt_evidence
    )


def test_intro_contract_retry_has_an_independent_one_call_budget(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    source_chart = GeneratedChart(
        notes=[NoteEvent(0, 0), NoteEvent(250, 1)],
        key_mode=7,
        osu_text="",
        generator_name="exhausted-quality-budget-fixture",
        seed=34,
        bpm_events=authority.bpm_events,
    )
    source = s2_generate._Candidate(
        request=GenerationRequest(
            audio_path=prepared.normalized.path,
            timing_reference_path=authority.reference_path,
            key_mode=7,
            difficulty="HARD",
            seed=34,
            duration_ms=prepared.normalized.duration_ms,
        ),
        generated=source_chart,
        acceptance=evaluate_chart_candidate(
            source_chart,
            authority,
            _analysis(),
            requested_key_mode=7,
            requested_difficulty="HARD",
            duration_ms=prepared.normalized.duration_ms,
        ),
        osu_text="",
        workdir=tmp_path / "source",
        attempt=3,
        seed=34,
        provenance="RETRY",
        intro_anchor_covered=False,
    )
    state = s2_generate._VariantState(
        key_mode=7,
        difficulty="HARD",
        flat_index=10,
        journal=s2_generate.AttemptJournal(tmp_path / "attempt-journal.jsonl"),
        budget=AttemptBudgetState(
            max_quality_attempts=MAX_VARIANT_ATTEMPTS,
            max_crash_attempts=MAX_CRASH_ATTEMPTS,
            max_total_attempts=MAX_TOTAL_ATTEMPTS,
            next_attempt=4,
            quality_attempts=MAX_VARIANT_ATTEMPTS,
        ),
        candidates=CandidateRepository(admitted=[source]),
    )
    budget = AdditionalInferenceBudget(limit=1)
    generator = RecordingGenerator()

    candidate = s2_generate._try_intro_contract_retry(
        state,
        source,
        prepared=prepared,
        authority=authority,
        onset_analysis=_analysis(),
        run_dir=tmp_path,
        generator=generator,
        base_seed=0,
        authority_epoch=1,
        inference_budget=budget,
    )

    assert candidate is not None
    assert candidate.seed == 46
    assert budget.used == 1
    assert len(generator.map_calls) == 1
    journal = [
        json.loads(line)
        for line in (tmp_path / "attempt-journal.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [entry["eventType"] for entry in journal] == [
        "INFERENCE_STARTED",
        "INFERENCE_COMPLETED",
        "GATE_EVALUATED",
        "CANDIDATE_ADMITTED",
    ]
    assert {entry["payload"]["purpose"] for entry in journal} == {
        "INTRO_CONTRACT_RETRY"
    }


def test_map_workdirs_are_scoped_to_the_authority_epoch(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    generator = RecordingGenerator()

    _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=17,
        authority_epoch=2,
    )

    assert generator.map_workdirs
    assert all("epoch-2" in workdir.parts for workdir in generator.map_workdirs)
    assert all(
        workdir.relative_to(tmp_path).as_posix().startswith("raw/work/epoch-2/")
        for workdir in generator.map_workdirs
    )
def test_rechecks_pool_after_each_retry_and_skips_seed_31(monkeypatch, tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    ratings = {
        (6, "HARD", 6): 5.44,
        (6, "EXPERT", 19): 5.04,
        (6, "HARD", 18): 5.00,
    }

    def evaluate(generated, *args, requested_key_mode, requested_difficulty, **kwargs):
        del args, kwargs
        rating = ratings.get(
            (requested_key_mode, requested_difficulty, generated.seed),
            {"EASY": 1.0, "NORMAL": 2.0, "HARD": 3.0, "EXPERT": 4.0}[
                requested_difficulty
            ],
        )
        return _acceptance_with_rating(accepted, rating)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)

    class SeedSevenFailsGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            chart = super().generate_map(request, workdir)
            if request.seed == 7:
                return replace(chart, key_mode=4)
            return chart

    generator = SeedSevenFailsGenerator()
    variants = _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    calls_6k = [
        (request.difficulty, request.seed)
        for request in generator.map_calls
        if request.key_mode == 6
    ]
    assert calls_6k == [
        ("EASY", 4),
        ("NORMAL", 5),
        ("HARD", 6),
        ("EXPERT", 7),
        ("EXPERT", 19),
        ("HARD", 18),
    ]
    assert 31 not in [request.seed for request in generator.map_calls]
    selected_6k = {
        variant.difficulty: variant for variant in variants if variant.key_mode == 6
    }
    assert selected_6k["HARD"].selected_seed == 18
    assert selected_6k["EXPERT"].selected_seed == 19


def test_difficulty_inversion_retries_only_pair_and_reuses_earliest_pass_candidate(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    ratings = {
        (4, "EASY", 0): 1.0,
        (4, "NORMAL", 1): 2.0,
        (4, "HARD", 2): 4.1,
        (4, "EXPERT", 3): 3.0,
        (4, "HARD", 14): 4.2,
        (4, "EXPERT", 15): 5.0,
    }

    def evaluate(generated, *args, requested_key_mode, requested_difficulty, **kwargs):
        del args, kwargs
        rating = ratings.get(
            (requested_key_mode, requested_difficulty, generated.seed),
            {"EASY": 1.0, "NORMAL": 2.0, "HARD": 3.0, "EXPERT": 4.0}[
                requested_difficulty
            ],
        )
        return _acceptance_with_rating(accepted, rating)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)

    class NoEarlyPromotionGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            if request.key_mode == 4 and request.seed in (14, 15):
                assert not any(
                    (tmp_path / "raw" / f"4k-{difficulty.lower()}.osu").exists()
                    for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
                )
            return super().generate_map(request, workdir)

    generator = NoEarlyPromotionGenerator()
    variants = _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert [
        (request.key_mode, request.difficulty, request.seed)
        for request in generator.map_calls[:6]
    ] == [
        (4, "EASY", 0),
        (4, "NORMAL", 1),
        (4, "HARD", 2),
        (4, "EXPERT", 3),
        (4, "HARD", 14),
        (4, "EXPERT", 15),
    ]
    selected_4k = {
        variant.difficulty: variant
        for variant in variants
        if variant.key_mode == 4
    }
    assert selected_4k["HARD"].selected_seed == 2
    assert selected_4k["HARD"].attempt == 1
    assert selected_4k["HARD"].candidate_count == 2
    assert selected_4k["HARD"].generation_attempt_count == 2
    assert any(
        evidence.get("reason") == "NOT_SELECTED_BEST_MONOTONIC_FAMILY"
        and evidence["seed"] == 14
        and evidence["attempt"] == 2
        and evidence["serializationValidated"] is True
        and evidence["gateReport"]["qualityProfile"]["difficultyProfile"][
            "projectRating"
        ]
        == 4.2
        for evidence in selected_4k["HARD"].attempt_evidence
    )
    assert selected_4k["EXPERT"].selected_seed == 15
    assert selected_4k["EXPERT"].attempt == 2
    assert selected_4k["EASY"].candidate_count == 1
    assert selected_4k["NORMAL"].candidate_count == 1
    assert all(
        variant.difficulty_order is not None
        and variant.difficulty_order.status == "PASS"
        for variant in selected_4k.values()
    )


def test_equal_difficulty_profiles_publish_without_more_seeds(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _acceptance_with_rating(_pass_acceptance(authority), 2.0)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        lambda *args, **kwargs: accepted,
    )
    generator = RecordingGenerator()

    variants = _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(variants) == 12
    assert [request.seed for request in generator.map_calls] == list(range(12))
    assert variants[0].difficulty_order is not None
    assert variants[0].difficulty_order.ambiguous_pairs == (
        ("EASY", "NORMAL"),
        ("NORMAL", "HARD"),
        ("HARD", "EXPERT"),
    )
    assert len(list((tmp_path / "raw").glob("*k-*.osu"))) == 12


def test_ambiguity_does_not_prevent_retrying_a_separate_inverted_pair(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        if generated.seed in {2, 3}:
            rating = {"HARD": 4.0, "EXPERT": 3.0}[requested_difficulty]
        else:
            rating = {"EASY": 2.0, "NORMAL": 2.0, "HARD": 3.0, "EXPERT": 4.0}[
                requested_difficulty
            ]
        return _acceptance_with_rating(accepted, rating)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)
    generator = RecordingGenerator()

    variants = _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(variants) == 12
    assert [request.seed for request in generator.map_calls] == [
        0,
        1,
        2,
        3,
        14,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ]
    assert variants[0].difficulty_order is not None
    assert variants[0].difficulty_order.ambiguous_pairs == (
        ("EASY", "NORMAL"),
        ("HARD", "EXPERT"),
    )


def test_exhausted_hard_still_tries_available_expert_candidate(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        rating = {
            ("HARD", 26): 4.0,
            ("EXPERT", 3): 3.0,
            ("EXPERT", 15): 5.0,
        }.get(
            (requested_difficulty, generated.seed),
            {"EASY": 1.0, "NORMAL": 2.0, "HARD": 3.0, "EXPERT": 4.0}[
                requested_difficulty
            ],
        )
        return _acceptance_with_rating(accepted, rating)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)

    class HardNeedsThirdAttempt(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if (
                request.key_mode == 4
                and request.difficulty == "HARD"
                and request.seed in {2, 14}
            ):
                return replace(generated, notes=[NoteEvent(500, 4)])
            return generated

    generator = HardNeedsThirdAttempt()
    variants = _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    selected_4k = {
        variant.difficulty: variant for variant in variants if variant.key_mode == 4
    }
    assert selected_4k["HARD"].selected_seed == 26
    assert selected_4k["EXPERT"].selected_seed == 15
    assert [
        (request.difficulty, request.seed)
        for request in generator.map_calls
        if request.key_mode == 4
    ] == [
        ("EASY", 0),
        ("NORMAL", 1),
        ("HARD", 2),
        ("HARD", 14),
        ("HARD", 26),
        ("EXPERT", 3),
        ("EXPERT", 15),
    ]


def test_persistent_inversion_keeps_all_labels_but_marks_order_for_review(
    monkeypatch, tmp_path: Path
):
    """역전이 끝내 안 풀려도 플레이테스트 채보는 삭제하지 않는다."""
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)

    def evaluate(*args, requested_difficulty, **kwargs):
        del args, kwargs
        rating = {"EASY": 1.0, "NORMAL": 2.0, "HARD": 4.0, "EXPERT": 3.0}[
            requested_difficulty
        ]
        return _acceptance_with_rating(accepted, rating)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)
    generator = RecordingGenerator()

    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert [
        (request.difficulty, request.seed)
        for request in generator.map_calls
        if request.key_mode == 4
    ] == [
        ("EASY", 0),
        ("NORMAL", 1),
        ("HARD", 2),
        ("EXPERT", 3),
        ("HARD", 14),
        ("EXPERT", 15),
        ("HARD", 26),
        ("EXPERT", 27),
    ]
    assert outcome.missing == ()
    published_4k = {
        variant.difficulty for variant in outcome.variants if variant.key_mode == 4
    }
    assert published_4k == {"EASY", "NORMAL", "HARD", "EXPERT"}
    assert (tmp_path / "raw" / "4k-hard.osu").is_file()
    assert (tmp_path / "raw" / "4k-expert.osu").is_file()
    assert all(
        variant.difficulty_order is not None
        and variant.difficulty_order.status == "RETRY"
        for variant in outcome.variants
        if variant.key_mode == 4
    )
    assert all(
        variant.provenance in {"PRIMARY", "RETRY"} for variant in outcome.variants
    )


def test_first_retry_candidate_can_reuse_the_existing_harder_candidate(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    ratings = {
        ("EASY", 0): 1.0,
        ("NORMAL", 1): 2.0,
        ("HARD", 2): 4.0,
        ("EXPERT", 3): 3.0,
        ("HARD", 14): 2.5,
        ("EXPERT", 15): 4.0,
    }

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        return _acceptance_with_rating(
            accepted,
            ratings[(requested_difficulty, generated.seed)],
        )

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)
    generator = RecordingGenerator()

    def evaluate_all_modes(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        rating = ratings.get(
            (requested_difficulty, generated.seed),
            {"EASY": 1.0, "NORMAL": 2.0, "HARD": 3.0, "EXPERT": 4.0}[
                requested_difficulty
            ],
        )
        return _acceptance_with_rating(accepted, rating)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate_all_modes)

    variants = _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(variants) == 12
    assert [variant.selected_seed for variant in variants[:4]] == [0, 1, 14, 3]
    assert variants[0].difficulty_order is not None
    assert variants[0].difficulty_order.ambiguous_pairs == ()
    assert [request.seed for request in generator.map_calls] == [
        0,
        1,
        2,
        3,
        14,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ]
    assert len(list((tmp_path / "raw").glob("4k-*.osu"))) == 4


def test_run_generation_preserves_generator_osu_text(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    class OriginalGenerator:
        def __init__(self):
            self.texts = []

        def generate_map(self, request, workdir):
            text = (
                "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\n"
                f"CircleSize:{request.key_mode}\n\n[TimingPoints]\n"
                "0,500,4,2,0,60,1,0\n\n[HitObjects]\n64,192,500,1,0,0:0:0:0:\n"
            )
            self.texts.append(text)
            return GeneratedChart(
                [NoteEvent(500, 0)],
                request.key_mode,
                text,
                "original",
                request.seed,
                (OsuBpmEvent(0, 120.0),),
            )

    generator = OriginalGenerator()
    variants = _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=1,
    )
    assert variants[0].raw_osu_path.read_text(encoding="utf-8") == generator.texts[0]


def test_empty_osu_text_fallback_preserves_every_authority_timing_event(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    bpm_events = (OsuBpmEvent(0, 120.0), OsuBpmEvent(1_000, 150.0))
    authority = _authority(prepared, tmp_path, bpm_events)

    class MultipleTimingGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return GeneratedChart(
                notes=generated.notes,
                key_mode=generated.key_mode,
                osu_text="",
                generator_name=generated.generator_name,
                seed=generated.seed,
                bpm_events=bpm_events,
            )

    variants = _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=MultipleTimingGenerator(),
        seed=0,
    )

    assert parse_osu_file(variants[0].raw_osu_path).bpm_events == bpm_events


def test_partial_stable_promotion_is_cleaned_and_normalized(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    original_read_text = Path.read_text

    def corrupt_second_stable_raw(path: Path, *args, **kwargs):
        if path.name == "4k-normal.osu":
            return "not an osu beatmap"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", corrupt_second_stable_raw)

    with pytest.raises(WorkerError) as captured:
        _run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=RecordingGenerator(),
            seed=0,
        )

    assert captured.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    assert captured.value.context == {
        "key_mode": 4,
        "failure_stage": "PROMOTION",
        "paths": [
            "raw/4k-easy.osu",
            "raw/4k-normal.osu",
            "raw/4k-hard.osu",
            "raw/4k-expert.osu",
        ],
        "selected_seeds": [0, 1, 2, 3],
        "cause_code": "CHART_OSU_PARSE_FAILED",
        "cause": "CHART_OSU_PARSE_FAILED: serialized MAP is not valid osu!mania",
    }
    assert not any((tmp_path / "raw").glob("4k-*.osu"))


def test_failed_key_mode_does_not_discard_the_other_key_modes(
    monkeypatch, tmp_path: Path
):
    """6K 전체가 구조 실패해도 4K·7K 는 그대로 발행한다."""
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    def reject_6k(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        action = (
            GateAction.RETRY_MAP
            if generated.key_mode == 6
            and generated.generator_name != "adaptive-recovery-v1"
            else GateAction.PASS
        )
        return _acceptance_for_difficulty(
            _acceptance_with_action(authority, action),
            requested_difficulty,
        )

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", reject_6k)

    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=RecordingGenerator(),
        seed=0,
    )

    assert {variant.key_mode for variant in outcome.variants} == {4, 6, 7}
    assert len(outcome.variants) == 12
    assert outcome.missing == ()
    assert all(
        variant.provenance == "SAFE_FALLBACK"
        for variant in outcome.variants
        if variant.key_mode == 6
    )
    assert len(list((tmp_path / "raw").glob("6k-*.osu"))) == 4
    assert len(list((tmp_path / "raw").glob("4k-*.osu"))) == 4
    assert len(list((tmp_path / "raw").glob("7k-*.osu"))) == 4


def test_stable_raw_reparse_rejects_text_with_different_timing_identity(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    mismatched_text = (
        "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\n"
        "CircleSize:4\n\n[TimingPoints]\n0,495.867768595041,4,2,0,60,1,0\n\n"
        "[HitObjects]\n64,192,500,1,0,0:0:0:0:\n"
    )

    class MismatchedTextGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return GeneratedChart(
                notes=generated.notes,
                key_mode=generated.key_mode,
                osu_text=mismatched_text,
                generator_name=generated.generator_name,
                seed=generated.seed,
                bpm_events=generated.bpm_events,
            )

    generator = MismatchedTextGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    # 손상된 모델 텍스트는 배포하지 않고 canonical fallback으로 대체한다.
    assert len(outcome.variants) == 12
    assert all(variant.provenance == "SAFE_FALLBACK" for variant in outcome.variants)
    assert len(generator.map_calls) == 12
    assert len(list((tmp_path / "raw").glob("*k-*.osu"))) == 12


@pytest.mark.parametrize(
    ("generated_note", "serialized_key_mode", "serialized_hit_object"),
    [
        pytest.param(
            NoteEvent(500, 0),
            6,
            "64,192,500,1,0,0:0:0:0:",
            id="key-mode",
        ),
        pytest.param(
            NoteEvent(500, 0),
            4,
            "192,192,500,1,0,0:0:0:0:",
            id="lane",
        ),
        pytest.param(
            NoteEvent(500, 0),
            4,
            "64,192,500,128,0,750:0:0:0:0:",
            id="kind",
        ),
        pytest.param(
            NoteEvent(500, 0, kind="HOLD", duration_ms=250),
            4,
            "64,192,500,128,0,800:0:0:0:0:",
            id="hold-duration",
        ),
    ],
)
def test_serialized_note_or_key_mismatch_retries_without_stable_promotion(
    tmp_path: Path,
    generated_note: NoteEvent,
    serialized_key_mode: int,
    serialized_hit_object: str,
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    mismatched_text = (
        "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\n"
        f"CircleSize:{serialized_key_mode}\n\n[TimingPoints]\n"
        "0,500,4,2,0,60,1,0\n\n[HitObjects]\n"
        f"{serialized_hit_object}\n"
    )

    class MismatchedCandidateGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return GeneratedChart(
                notes=[generated_note],
                key_mode=generated.key_mode,
                osu_text=mismatched_text,
                generator_name=generated.generator_name,
                seed=generated.seed,
                bpm_events=generated.bpm_events,
            )

    generator = MismatchedCandidateGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    # 직렬화가 객체와 다르면 모델 결과는 raw로도 쓰지 않고 안전 대체한다.
    fallback = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert fallback.provenance == "SAFE_FALLBACK"
    assert len(generator.map_calls) == 12
    assert (tmp_path / "raw" / "4k-easy.osu").exists()


def test_reference_metadata_mutated_during_map_is_rejected_before_promotion(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    class MutatingReferenceGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            authority.reference_path.write_text(
                authority.reference_path.read_text(encoding="utf-8").replace(
                    "Title:fixture", "Title:mutated"
                ),
                encoding="utf-8",
            )
            return generated

    generator = MutatingReferenceGenerator()
    with pytest.raises(WorkerError) as captured:
        _run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=generator,
            seed=0,
        )

    assert captured.value.code is ErrorCode.ASSET_HASH_MISMATCH
    assert len(generator.map_calls) == 1
    assert not (tmp_path / "raw" / "4k-easy.osu").exists()


def test_canonical_audio_mutated_during_map_is_rejected_before_promotion(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    class MutatingAudioGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            prepared.normalized.path.write_bytes(b"mutated audio")
            return generated

    generator = MutatingAudioGenerator()
    with pytest.raises(WorkerError) as captured:
        _run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=generator,
            seed=0,
        )

    assert captured.value.code is ErrorCode.ASSET_HASH_MISMATCH
    assert len(generator.map_calls) == 1
    assert not (tmp_path / "raw" / "4k-easy.osu").exists()


def test_generated_structure_defects_exhaust_with_gate_evidence_before_stable_raw(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        evaluate_chart_candidate,
    )

    class InvalidLaneGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return GeneratedChart(
                notes=[NoteEvent(500, request.key_mode)],
                key_mode=generated.key_mode,
                osu_text=generated.osu_text,
                generator_name=generated.generator_name,
                seed=generated.seed,
                bpm_events=generated.bpm_events,
            )

    generator = InvalidLaneGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 36
    fallback = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert fallback.provenance == "SAFE_FALLBACK"
    attempts = [
        item for item in fallback.attempt_evidence if "gateReport" in item
    ]
    assert [attempt["seed"] for attempt in attempts] == [0, 12, 24]
    assert [attempt["workdir"] for attempt in attempts] == [
        "raw/work/epoch-1/4k-easy/attempt-1",
        "raw/work/epoch-1/4k-easy/attempt-2",
        "raw/work/epoch-1/4k-easy/attempt-3",
    ]
    assert all(
        attempt["gateReport"]["decisions"]["STRUCTURE"]["reasons"]
        == ["STRUCTURE_INVALID"]
        for attempt in attempts
    )
    # 구조 축 실패 모델 출력은 버리고 검증된 안전 fallback만 발행한다.
    assert (tmp_path / "raw" / "4k-easy.osu").exists()


def test_real_hold_overlap_retries_only_the_failed_variant(tmp_path: Path):
    prepared = _prepared(tmp_path, duration_ms=150_000)
    authority = _authority(prepared, tmp_path)

    class FirstEasyAttemptHasObservedHoldOverlap(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if (
                request.key_mode == 4
                and request.difficulty == "EASY"
                and request.seed == 0
            ):
                return GeneratedChart(
                    notes=[
                        NoteEvent(0, 0, kind="HOLD", duration_ms=134_204),
                        NoteEvent(925, 0),
                    ],
                    key_mode=generated.key_mode,
                    osu_text="",
                    generator_name=generated.generator_name,
                    seed=generated.seed,
                    bpm_events=generated.bpm_events,
                )
            return generated

    generator = FirstEasyAttemptHasObservedHoldOverlap()
    variants = _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    easy_calls = [
        call
        for call in generator.map_calls
        if call.key_mode == 4 and call.difficulty == "EASY"
    ]
    other_calls = [
        call
        for call in generator.map_calls
        if not (call.key_mode == 4 and call.difficulty == "EASY")
    ]
    assert [call.seed for call in easy_calls] == [0, 12]
    assert len(other_calls) == 11
    assert len(variants) == 12
    assert next(
        variant
        for variant in variants
        if variant.key_mode == 4 and variant.difficulty == "EASY"
    ).selected_seed == 12


def test_timing_feedback_two_duplicate_seeds_escalate_before_third_attempt(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        evaluate_chart_candidate,
    )

    class AlwaysDuplicatesAtZero(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return GeneratedChart(
                notes=[NoteEvent(0, 1), NoteEvent(0, 1)],
                key_mode=generated.key_mode,
                osu_text="",
                generator_name=generated.generator_name,
                seed=generated.seed,
                bpm_events=generated.bpm_events,
            )

    generator = AlwaysDuplicatesAtZero()
    with pytest.raises(RetryTimingSignal) as captured:
        _run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=generator,
            seed=0,
        )

    assert [call.seed for call in generator.map_calls] == [0, 12]
    assert captured.value.to_context()["timingSegmentId"] == 0
    assert captured.value.to_context()["failureFamily"] == "DUPLICATE_NOTE"
    assert not (tmp_path / "raw" / "4k-easy.osu").exists()


def test_observed_resnap_duplicate_is_classified_and_escalated_separately(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        evaluate_chart_candidate,
    )

    class AlwaysObservedResnapCollision(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return replace(
                generated,
                notes=[NoteEvent(0, 1), NoteEvent(0, 1)],
                osu_text="",
                resnap_diagnostics=ResnapDiagnostics(
                    status="OBSERVED",
                    collisions=(
                        ResnapCollision(
                            seed=request.seed,
                            lane=1,
                            note_kind="TAP",
                            pre_time_ms=-10,
                            post_time_ms=0,
                            snap_divisor=4,
                            reason="RAW_TIME_COLLISION_PRESERVED",
                        ),
                        ResnapCollision(
                            seed=request.seed,
                            lane=1,
                            note_kind="TAP",
                            pre_time_ms=10,
                            post_time_ms=0,
                            snap_divisor=4,
                            reason="RAW_TIME_COLLISION_PRESERVED",
                        ),
                    ),
                ),
            )

    with pytest.raises(RetryTimingSignal) as captured:
        _run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=AlwaysObservedResnapCollision(),
            seed=0,
        )

    context = captured.value.to_context()
    assert context["failureFamily"] == "RESNAP_COLLISION"
    assert context["seeds"] == [0, 12]
    assert context["signatures"][0]["evidence"]["resnapCollisions"][0] == {
        "seed": 0,
        "lane": 1,
        "noteKind": "TAP",
        "preTimeMs": -10,
        "postTimeMs": 0,
        "snapDivisor": 4,
        "reason": "RAW_TIME_COLLISION_PRESERVED",
    }


def test_timing_feedback_duplicates_in_different_segments_remain_map_retries(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(
        prepared,
        tmp_path,
        (
            OsuBpmEvent(0, 120.0),
            OsuBpmEvent(500, 120.0),
            OsuBpmEvent(1_000, 120.0),
        ),
    )
    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate_chart_candidate)

    class DuplicatesMoveAcrossSegments(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if request.key_mode == 4 and request.difficulty == "EASY":
                time_ms = {0: 0, 12: 500, 24: 1_000}[request.seed]
                return replace(
                    generated,
                    notes=[NoteEvent(time_ms, 1), NoteEvent(time_ms, 1)],
                    osu_text="",
                )
            return generated

    generator = DuplicatesMoveAcrossSegments()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    # 핵심: 중복이 서로 다른 timing 구간에서 나면 timing 층으로 올리지
    # 않는다. RetryTimingSignal 이 아니라 조합 단위 소진으로 끝난다.
    assert [
        call.seed
        for call in generator.map_calls
        if (call.key_mode, call.difficulty) == (4, "EASY")
    ] == [0, 12, 24]
    fallback = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert fallback.provenance == "SAFE_FALLBACK"


def test_timing_feedback_two_active_middle_gaps_escalate(monkeypatch, tmp_path: Path):
    prepared = _prepared(tmp_path, duration_ms=30_000)
    authority = _authority(prepared, tmp_path)
    authority = replace(
        authority,
        local_review=SimpleNamespace(
            segments=(
                SimpleNamespace(
                    metrics=SimpleNamespace(index=0),
                    grid_damage=True,
                ),
            ),
        ),
    )
    calls = 0

    def coverage_then_pass(generated, authority, analysis, **kwargs):
        nonlocal calls
        del generated, analysis, kwargs
        calls += 1
        accepted = _coverage_retry_acceptance(authority)
        return _acceptance_for_difficulty(accepted, "EASY")

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", coverage_then_pass)
    generator = RecordingGenerator()

    with pytest.raises(RetryTimingSignal) as captured:
        _run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=generator,
            seed=0,
        )

    assert calls == 2
    assert [call.seed for call in generator.map_calls] == [0, 12]
    assert captured.value.to_context()["failureFamily"] == "ACTIVE_MIDDLE_GAP"


def test_retries_only_the_failed_variant_with_the_next_seed(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    class FirstEasyAttemptInvalid(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if request.key_mode == 4 and request.difficulty == "EASY" and request.seed == 0:
                return GeneratedChart(
                    notes=[NoteEvent(500, request.key_mode)],
                    key_mode=generated.key_mode,
                    osu_text=generated.osu_text,
                    generator_name=generated.generator_name,
                    seed=generated.seed,
                    bpm_events=generated.bpm_events,
                )
            return generated

    generator = FirstEasyAttemptInvalid()
    variants = _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    calls = [
        (request.key_mode, request.difficulty, request.seed, workdir.name)
        for request, workdir in zip(
            generator.map_calls, generator.map_workdirs, strict=True
        )
    ]
    assert calls[:2] == [
        (4, "EASY", 0, "attempt-1"),
        (4, "EASY", 12, "attempt-2"),
    ]
    assert len(calls) == 13
    assert variants[0].attempt == 2
    assert len(variants[0].attempt_errors) == 1
    assert all(variant.attempt == 1 for variant in variants[1:])
    assert all(not variant.attempt_errors for variant in variants[1:])


def test_retry_map_uses_next_seed_and_promotes_only_the_pass_candidate(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    retry = _acceptance_with_action(authority, GateAction.RETRY_MAP)
    accepted = _acceptance_with_action(authority, GateAction.PASS)
    evaluations = []

    def evaluate(*args, requested_difficulty, **kwargs):
        del args, kwargs
        evaluations.append(len(evaluations) + 1)
        if len(evaluations) == 1:
            return retry
        if len(evaluations) == 2:
            assert not (tmp_path / "raw" / "4k-easy.osu").exists()
        return _acceptance_for_difficulty(accepted, requested_difficulty)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)
    generator = RecordingGenerator()

    variants = _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    first = variants[0]
    assert [request.seed for request in generator.map_calls[:2]] == [0, 12]
    assert first.attempt == 2
    assert first.acceptance == _acceptance_for_difficulty(accepted, "EASY")
    assert first.raw_osu_path == tmp_path / "raw" / "4k-easy.osu"
    assert first.raw_osu_path.is_file()
    assert first.attempt_evidence == (
        {
            "seed": 0,
            "workdir": "raw/work/epoch-1/4k-easy/attempt-1",
            "gateReport": retry.to_report(),
        },
    )


def test_review_candidate_is_published_without_consuming_another_seed(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    review = _acceptance_with_action(authority, GateAction.REVIEW)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        lambda *args, requested_difficulty, **kwargs: _acceptance_for_difficulty(
            review, requested_difficulty
        ),
    )

    class EvidenceGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            (workdir / "candidate.osu").write_text("candidate", encoding="utf-8")
            return generated

    generator = EvidenceGenerator()
    variants = _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(variants) == 12
    assert all(variant.acceptance.action is GateAction.REVIEW for variant in variants)
    assert [request.seed for request in generator.map_calls] == list(range(12))
    assert (
        tmp_path
        / "raw"
        / "work"
        / "epoch-1"
        / "4k-easy"
        / "attempt-1"
        / "candidate.osu"
    ).is_file()
    assert (tmp_path / "raw" / "4k-easy.osu").is_file()


@pytest.mark.parametrize("failure_index", (0, 5, 11))
def test_unknown_completion_never_consumes_a_second_full_length_seed(
    tmp_path: Path,
    failure_index: int,
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    unknown = WorkerError(
        ErrorCode.INFERENCE_COMPLETION_UNKNOWN,
        "accepted invocation lost its terminal record",
        context={"accepted": True},
    )

    class UnknownGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if len(self.map_calls) - 1 == failure_index:
                raise unknown
            return generated

    generator = UnknownGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    slots = tuple(
        (key_mode, difficulty)
        for key_mode in (4, 6, 7)
        for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
    )
    trigger_slot = slots[failure_index]
    assert [request.seed for request in generator.map_calls] == list(
        range(failure_index + 1)
    )
    assert len(outcome.variants) == 12
    assert outcome.missing == ()
    by_slot = {
        (variant.key_mode, variant.difficulty): variant for variant in outcome.variants
    }
    assert all(by_slot[slot].provenance == "PRIMARY" for slot in slots[:failure_index])
    assert all(
        by_slot[slot].provenance == "SAFE_FALLBACK"
        for slot in slots[failure_index:]
    )
    trigger = by_slot[trigger_slot]
    assert any(
        evidence["reason"] == "MODEL_INFERENCE_DISABLED_AFTER_UNKNOWN_COMPLETION"
        and evidence["triggerVariant"]
        == {"keyMode": trigger_slot[0], "difficulty": trigger_slot[1]}
        and evidence["error"]["code"] == "INFERENCE_COMPLETION_UNKNOWN"
        for evidence in trigger.attempt_evidence
    )
    if failure_index + 1 < len(slots):
        following = by_slot[slots[failure_index + 1]]
        assert any(
            evidence["reason"]
            == "MODEL_INFERENCE_SUPPRESSED_AFTER_UNKNOWN_COMPLETION"
            and evidence["triggerVariant"]
            == {"keyMode": trigger_slot[0], "difficulty": trigger_slot[1]}
            for evidence in following.attempt_evidence
        )


def test_three_retry_map_decisions_exhaust_with_structured_attempt_evidence(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    retry = _acceptance_with_action(authority, GateAction.RETRY_MAP)

    def evaluate(generated, *args, **kwargs):
        del args, kwargs
        return accepted if generated.generator_name == "adaptive-recovery-v1" else retry

    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        evaluate,
    )
    generator = RecordingGenerator()

    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    # 기존 모델 예산은 그대로 유한하며, 소진 뒤에는 추가 inference 없이
    # canonical fallback을 발행한다.
    assert [request.seed for request in generator.map_calls] == [
        flat_index + attempt * 12
        for flat_index in range(12)
        for attempt in range(3)
    ]
    assert len(outcome.variants) == 12
    assert all(variant.provenance == "SAFE_FALLBACK" for variant in outcome.variants)
    first = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    gate_attempts = [
        item for item in first.attempt_evidence if "gateReport" in item
    ]
    assert gate_attempts == [
        {
            "seed": attempt_seed,
            "workdir": f"raw/work/epoch-1/4k-easy/attempt-{attempt}",
            "gateReport": retry.to_report(),
        }
        for attempt, attempt_seed in enumerate((0, 12, 24), start=1)
    ]
    assert len(first.attempt_errors) == 3
    assert (tmp_path / "raw" / "4k-easy.osu").exists()


def test_hard_safe_quality_rejection_stops_after_first_full_length_seed(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    soft_retry = replace(
        accepted,
        action=GateAction.RETRY_MAP,
        decisions=tuple(
            replace(
                decision,
                action=GateAction.RETRY_MAP,
                reasons=("FIXTURE_PATTERN_REVIEW",),
            )
            if decision.axis is GateAxis.PATTERN
            else decision
            for decision in accepted.decisions
        ),
    )

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        source = (
            soft_retry
            if generated.generator_name != "adaptive-recovery-v1"
            and generated.key_mode == 4
            and requested_difficulty == "EASY"
            else accepted
        )
        return _acceptance_for_difficulty(source, requested_difficulty)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)
    generator = RecordingGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 12
    assert [
        request.seed
        for request in generator.map_calls
        if (request.key_mode, request.difficulty) == (4, "EASY")
    ] == [0]
    raw = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert raw.provenance == "RAW_UNVERIFIED"
    assert raw.recovery_reason == "QUALITY_GATE_REJECTED"


def test_crashing_variant_gets_safe_fallback_without_regenerating_successes(
    tmp_path: Path,
):
    """모델 출력이 전혀 없어도 canonical timing 기반 안전 채보를 제공한다."""
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    class ExhaustedEasyGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if request.key_mode == 4 and request.difficulty == "EASY":
                raise WorkerError(
                    ErrorCode.CHART_GENERATION_FAILED,
                    "fixture exhausted 4K EASY",
                )
            return generated

    generator = ExhaustedEasyGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(outcome.variants) == 12
    assert outcome.missing == ()
    fallback = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert fallback.provenance == "SAFE_FALLBACK"
    assert fallback.recovery_reason == "NO_STRUCTURALLY_SAFE_MODEL_CANDIDATE"
    assert fallback.generated.generator_name == "adaptive-recovery-v1"
    assert fallback.acceptance.decision(GateAxis.STRUCTURE).action is GateAction.PASS
    assert (
        fallback.acceptance.decision(GateAxis.TIMING_IDENTITY).action
        is GateAction.PASS
    )
    assert fallback.acceptance.decision(GateAxis.SONG_BOUNDS).action is GateAction.PASS
    # 크래시 예산 3회만 쓰고 나머지 11 조합은 한 번씩만 부른다.
    assert len(generator.map_calls) == 14


def test_song_bounds_rejected_model_output_is_not_promoted_as_raw(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    song_bounds_retry = replace(
        accepted,
        action=GateAction.RETRY_MAP,
        decisions=tuple(
            replace(
                decision,
                action=GateAction.RETRY_MAP,
                reasons=("NOTE_START_AFTER_MUSIC_END",),
            )
            if decision.axis is GateAxis.SONG_BOUNDS
            else decision
            for decision in accepted.decisions
        ),
    )

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        source = (
            accepted
            if generated.generator_name == "adaptive-recovery-v1"
            else song_bounds_retry
            if generated.key_mode == 4 and requested_difficulty == "EASY"
            else accepted
        )
        return _acceptance_for_difficulty(source, requested_difficulty)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)
    generator = RecordingGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    fallback = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert fallback.provenance == "SAFE_FALLBACK"
    assert fallback.acceptance.decision(GateAxis.SONG_BOUNDS).action is GateAction.PASS
    assert outcome.diagnostic_raw_candidates == ()
    assert len(generator.map_calls) == 14


def test_localized_coverage_failure_remaps_only_the_exhausted_variant(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path, duration_ms=60_000)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    coverage_retry = _coverage_retry_acceptance(authority)

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        if generated.generator_name == "partial-repair":
            return _acceptance_for_difficulty(accepted, requested_difficulty)
        if requested_difficulty == "EASY" and generated.key_mode == 4:
            return _acceptance_for_difficulty(coverage_retry, requested_difficulty)
        return _acceptance_for_difficulty(accepted, requested_difficulty)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)

    class PartialRepairGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if request.add_to_beatmap:
                return replace(generated, generator_name="partial-repair")
            return generated

    generator = PartialRepairGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )
    variants = outcome.variants

    repaired = next(
        variant
        for variant in variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    partial_requests = [
        request for request in generator.map_calls if request.add_to_beatmap
    ]
    assert len(generator.map_calls) == 13
    assert len(partial_requests) == 1
    assert partial_requests[0].partial_start_ms == 8_000
    assert partial_requests[0].partial_end_ms == 22_000
    assert partial_requests[0].timing_reference_path.is_file()
    assert partial_requests[0].timing_reference_path != authority.reference_path
    assert repaired.provenance == "PARTIAL_REMAP"
    assert repaired.recovery_reason == "ACTIVE_COVERAGE_GAP"
    assert outcome.additional_inference_calls == 1
    assert outcome.additional_inference_work_ms == 14_000
    assert outcome.additional_inference_work_limit_ms == 60_000
    journal = [
        json.loads(line)
        for line in (tmp_path / "attempt-journal.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    recovery_events = [
        entry for entry in journal if entry["payload"].get("purpose") == "PARTIAL_REMAP"
    ]
    assert [entry["eventType"] for entry in recovery_events] == [
        "INFERENCE_STARTED",
        "INFERENCE_COMPLETED",
        "GATE_EVALUATED",
        "CANDIDATE_ADMITTED",
    ]
    assert all(
        variant.provenance == "PRIMARY"
        for variant in variants
        if variant is not repaired
    )


def test_song_work_budget_admits_two_small_partial_repairs(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path, duration_ms=60_000)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    coverage_retry = _coverage_retry_acceptance(authority)

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        if generated.generator_name == "partial-repair":
            return _acceptance_for_difficulty(accepted, requested_difficulty)
        if requested_difficulty == "EASY" and generated.key_mode in {4, 6}:
            return _acceptance_for_difficulty(coverage_retry, requested_difficulty)
        return _acceptance_for_difficulty(accepted, requested_difficulty)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)

    class PartialRepairGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if request.add_to_beatmap:
                return replace(generated, generator_name="partial-repair")
            return generated

    generator = PartialRepairGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    partial_requests = [
        request for request in generator.map_calls if request.add_to_beatmap
    ]
    assert len(generator.map_calls) == 14
    assert [(request.key_mode, request.difficulty) for request in partial_requests] == [
        (4, "EASY"),
        (6, "EASY"),
    ]
    assert outcome.additional_inference_calls == 2
    assert outcome.additional_inference_work_ms == 28_000
    assert outcome.additional_inference_work_limit_ms == 60_000
    assert all(
        next(
            variant
            for variant in outcome.variants
            if (variant.key_mode, variant.difficulty) == slot
        ).provenance
        == "PARTIAL_REMAP"
        for slot in ((4, "EASY"), (6, "EASY"))
    )


def test_partial_tail_exhaustion_blocks_only_that_variant(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path, duration_ms=60_000)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    coverage_retry = _coverage_retry_acceptance(authority)
    tail_context = {
        "reason": "END_BOUNDARY_CROSSES_HOLD",
        "signature": "END_BOUNDARY_CROSSES_HOLD:500:11",
        "requestedEndMs": 500,
        "effectiveCutMs": 510,
        "earliestGeneratedSourceWindowId": 11,
        "repairStartWindowId": 9,
        "repairWindowIds": [9, 10, 11],
        "repairAttempts": 2,
    }

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        if (
            generated.generator_name != "adaptive-recovery-v1"
            and requested_difficulty == "EASY"
            and generated.key_mode == 4
        ):
            return _acceptance_for_difficulty(coverage_retry, requested_difficulty)
        return _acceptance_for_difficulty(accepted, requested_difficulty)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)
    exhausted_context = {}
    real_assemble_publication = s2_generate.assemble_publication

    def capture_exhausted_context(selections, **kwargs):
        failed_state = next(
            states["EASY"]
            for states, _assignment, _review in selections
            if states["EASY"].key_mode == 4
        )
        assert failed_state.exhausted_error is not None
        exhausted_context.update(failed_state.exhausted_error.context)
        return real_assemble_publication(selections, **kwargs)

    monkeypatch.setattr(
        s2_generate,
        "assemble_publication",
        capture_exhausted_context,
    )

    class PartialTailExhaustedGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if request.add_to_beatmap:
                raise WorkerError(
                    ErrorCode.MANIA_TAIL_REPAIR_EXHAUSTED,
                    "fixture partial tail exhausted",
                    context=tail_context,
                )
            return generated

    generator = PartialTailExhaustedGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 13
    assert [
        request.seed
        for request in generator.map_calls
        if (request.key_mode, request.difficulty) == (4, "EASY")
    ] == [0, 36]
    assert generator.map_calls[-1].add_to_beatmap is True
    assert len(outcome.variants) == 12
    assert outcome.additional_inference_calls == 1
    assert outcome.missing == ()
    assert outcome.diagnostic_raw_candidates == ()
    raw = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert raw.provenance == "RAW_UNVERIFIED"
    assert json.loads(raw.attempt_errors[-1]) == {
        "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
        "context": tail_context,
        "message": "MANIA_TAIL_REPAIR_EXHAUSTED: fixture partial tail exhausted",
        "type": "WorkerError",
    }
    assert {
        "reason": "MANIA_TAIL_REPAIR_EXHAUSTED",
        "context": tail_context,
        "outerFullLengthRetrySuppressed": True,
        "partialAttempt": 2,
        "partialSeed": 36,
        "totalAttempts": 2,
        "qualityAttempts": 1,
        "crashAttempts": 0,
    } in raw.attempt_evidence
    assert exhausted_context["totalAttempts"] == 2
    assert exhausted_context["seeds"] == [0, 36]
    assert exhausted_context["partialAttempts"] == [2]
    assert exhausted_context["partialSeeds"] == [36]
    assert exhausted_context["qualityAttempts"] == 1
    assert exhausted_context["crashAttempts"] == 0
    journal = [
        json.loads(line)
        for line in (tmp_path / "attempt-journal.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    partial_events = [
        entry for entry in journal if entry["payload"].get("purpose") == "PARTIAL_REMAP"
    ]
    assert [entry["eventType"] for entry in partial_events] == [
        "INFERENCE_STARTED",
        "INFERENCE_FAILED",
    ]
    assert partial_events[-1]["payload"]["error"]["context"] == tail_context


def test_confirmed_intro_mismatch_is_not_hidden_by_note_mutation(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    authority = replace(
        _authority(prepared, tmp_path, (OsuBpmEvent(405, 140.0),)),
        leading_coverage=LeadingTimingCoverage(
            action=TimingAuthorityAction.REVIEW,
            reasons=("CONFIRMED_INTRO_ANCHOR_BEFORE_FIRST_EVENT",),
            first_event_time_ms=405,
            leading_duration_ms=405,
            onset_count=1,
            active_onset_count=1,
            active_frame_ratio=1.0,
            intro_anchor=IntroAnchorEvidence(
                status="CONFIRMED",
                anchor_ms=21,
                anchor_grid_ms=0,
                grid_distance_ms=21,
                aggregate_percentile_rank=0.99,
                prominent_band_count=3,
                pulse_continuation_matches=3,
                pulse_continuation_opportunities=4,
            ),
        ),
    )

    class IntroRepairGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            self.map_calls.append(request)
            self.map_workdirs.append(workdir)
            if request.add_to_beatmap:
                return GeneratedChart(
                    notes=[NoteEvent(0, 0)],
                    key_mode=request.key_mode,
                    osu_text="",
                    generator_name="intro-preroll-fixture",
                    seed=request.seed,
                    bpm_events=(
                        OsuBpmEvent(21, 140.0),
                        OsuBpmEvent(405, 140.0),
                    ),
                )
            covered = not (
                request.key_mode == 4 and request.difficulty == "EASY"
            )
            return GeneratedChart(
                notes=[NoteEvent(0 if covered else 500, 0)],
                key_mode=request.key_mode,
                osu_text="",
                generator_name="intro-primary-fixture",
                seed=request.seed,
                bpm_events=authority.bpm_events,
            )

    generator = IntroRepairGenerator()
    variants = _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    repaired = next(
        variant
        for variant in variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    repair_requests = [request for request in generator.map_calls if request.add_to_beatmap]
    assert len(generator.map_calls) == 12
    assert repair_requests == []
    assert repaired.provenance == "PRIMARY"
    assert repaired.recovery_reason is None
    assert repaired.generated.notes == [NoteEvent(500, 0)]
    assert repaired.generation_attempt_count == 1
    assert all(
        variant.provenance == "PRIMARY"
        for variant in variants
        if variant is not repaired
    )


def test_unanimous_audio_backed_family_overrides_conflicting_early_anchor(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    authority = replace(
        _authority(prepared, tmp_path, (OsuBpmEvent(405, 140.0),)),
        leading_coverage=LeadingTimingCoverage(
            action=TimingAuthorityAction.REVIEW,
            reasons=("CONFIRMED_INTRO_ANCHOR_BEFORE_FIRST_EVENT",),
            first_event_time_ms=405,
            leading_duration_ms=405,
            onset_count=1,
            active_onset_count=1,
            active_frame_ratio=1.0,
            intro_anchor=IntroAnchorEvidence(
                status="CONFIRMED",
                anchor_ms=21,
                anchor_grid_ms=0,
                grid_distance_ms=21,
                aggregate_percentile_rank=0.99,
                prominent_band_count=3,
                pulse_continuation_matches=3,
                pulse_continuation_opportunities=4,
            ),
        ),
    )

    class AllLateGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            self.map_calls.append(request)
            self.map_workdirs.append(workdir)
            return GeneratedChart(
                notes=[NoteEvent(500, 0), NoteEvent(900, 1)],
                key_mode=request.key_mode,
                osu_text="",
                generator_name="late-intro-fixture",
                seed=request.seed,
                bpm_events=authority.bpm_events,
            )

    generator = AllLateGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 12
    canonical_ms = outcome.intro_start_contract.canonical_first_row_ms
    assert canonical_ms == 500
    assert outcome.intro_start_contract.candidate_support_count == 12
    assert outcome.intro_start_contract.raw_supported is True
    assert outcome.intro_start_contract.audio_supported is True
    assert outcome.intro_contract_review.status == "PASS"
    assert outcome.intro_contract_review.corrected_count == 0
    assert {variant.generated.notes[0].time_ms for variant in outcome.variants} == {500}


def test_rejected_partial_remap_promotes_structurally_safe_raw_for_playtest(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path, duration_ms=60_000)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    coverage_retry = _coverage_retry_acceptance(authority)

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        if generated.generator_name == "adaptive-recovery-v1":
            return _acceptance_for_difficulty(accepted, requested_difficulty)
        if requested_difficulty == "EASY" and generated.key_mode == 4:
            return _acceptance_for_difficulty(coverage_retry, requested_difficulty)
        return _acceptance_for_difficulty(accepted, requested_difficulty)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)
    generator = RecordingGenerator()

    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 13
    assert sum(request.add_to_beatmap for request in generator.map_calls) == 1
    raw = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert len(outcome.variants) == 12
    assert outcome.missing == ()
    assert raw.provenance == "RAW_UNVERIFIED"
    assert raw.recovery_reason == "QUALITY_GATE_REJECTED"
    assert raw.acceptance.action is GateAction.RETRY_MAP
    assert raw.acceptance.decision(GateAxis.STRUCTURE).action is GateAction.PASS
    assert raw.acceptance.decision(GateAxis.TIMING_IDENTITY).action is GateAction.PASS
    assert raw.acceptance.decision(GateAxis.SONG_BOUNDS).action is GateAction.PASS
    assert outcome.diagnostic_raw_candidates == ()


def test_normal_generation_marks_every_variant_primary(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    generator = RecordingGenerator()

    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )
    variants = outcome.variants

    assert len(generator.map_calls) == 12
    assert all(variant.provenance == "PRIMARY" for variant in variants)
    assert all(variant.recovery_reason is None for variant in variants)
    assert outcome.diagnostic_raw_candidates == ()


def test_mixed_exhaustion_retains_gate_evidence_with_all_legacy_errors(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    retry = _acceptance_with_action(authority, GateAction.RETRY_MAP)
    evaluation_calls = 0

    def evaluate(generated, *args, **kwargs):
        nonlocal evaluation_calls
        del args, kwargs
        if generated.generator_name == "adaptive-recovery-v1":
            return accepted
        evaluation_calls += 1
        return retry

    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        evaluate,
    )

    class GateThenLegacyFailureGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if request.seed == 0:
                return generated
            raise WorkerError(
                ErrorCode.CHART_GENERATION_FAILED,
                "fixture legacy generation failure",
            )

    generator = GateThenLegacyFailureGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    fallback = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert fallback.provenance == "SAFE_FALLBACK"
    # 게이트 거절 1건이 품질 예산에서, 크래시 3건이 크래시 예산에서 나간다.
    # 예산이 하나였을 때는 크래시가 품질 재시도 기회까지 먹었다.
    assert len(fallback.attempt_errors) == 4
    assert evaluation_calls == 1
    assert [item for item in fallback.attempt_evidence if "gateReport" in item] == [
        {
            "seed": 0,
            "workdir": "raw/work/epoch-1/4k-easy/attempt-1",
            "gateReport": retry.to_report(),
        }
    ]


def test_real_acceptance_evidence_is_recorded_for_aligned_candidates(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        evaluate_chart_candidate,
    )

    class AlignedGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            chord_size = {
                "EASY": 1,
                "NORMAL": 2,
                "HARD": min(3, request.key_mode),
                "EXPERT": min(4, request.key_mode),
            }[request.difficulty]
            return replace(
                generated,
                notes=[
                    NoteEvent(time_ms, lane)
                    for time_ms in range(125, 2_000, 125)
                    for lane in range(chord_size)
                ],
            )

    variants = _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=AlignedGenerator(),
        seed=0,
    )

    assert {variant.acceptance.action for variant in variants} == {GateAction.PASS}
    assert all(not variant.attempt_evidence for variant in variants)


def test_reports_all_errors_when_one_variant_exhausts_its_attempts(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    class AlwaysInvalid(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return GeneratedChart(
                notes=[NoteEvent(500, request.key_mode)],
                key_mode=generated.key_mode,
                osu_text=generated.osu_text,
                generator_name=generated.generator_name,
                seed=generated.seed,
                bpm_events=generated.bpm_events,
            )

    generator = AlwaysInvalid()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 36
    assert len(outcome.variants) == 12
    assert all(variant.provenance == "SAFE_FALLBACK" for variant in outcome.variants)
    assert all(len(variant.attempt_errors) == 3 for variant in outcome.variants)


def test_generated_timing_identity_defects_exhaust_with_gate_evidence(
    monkeypatch, tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        evaluate_chart_candidate,
    )

    class DifferentTimingGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return GeneratedChart(
                notes=generated.notes,
                key_mode=generated.key_mode,
                osu_text=generated.osu_text,
                generator_name=generated.generator_name,
                seed=generated.seed,
                bpm_events=(OsuBpmEvent(0, 121.0),),
            )

    generator = DifferentTimingGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 36
    fallback = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert fallback.provenance == "SAFE_FALLBACK"
    attempts = [
        item for item in fallback.attempt_evidence if "gateReport" in item
    ]
    assert [attempt["seed"] for attempt in attempts] == [0, 12, 24]
    assert [attempt["workdir"] for attempt in attempts] == [
        "raw/work/epoch-1/4k-easy/attempt-1",
        "raw/work/epoch-1/4k-easy/attempt-2",
        "raw/work/epoch-1/4k-easy/attempt-3",
    ]
    assert all(
        attempt["gateReport"]["decisions"]["TIMING_IDENTITY"]["reasons"]
        == ["TIMING_REFERENCE_MISMATCH"]
        for attempt in attempts
    )
    # 곡 공통 시간축이 다른 모델 결과는 버리고 canonical fallback을 쓴다.
    assert (tmp_path / "raw" / "4k-easy.osu").exists()


def test_crash_attempt_error_preserves_worker_context_and_stderr(tmp_path: Path):
    """크래시 원인을 저장된 산출물로 진단할 수 있어야 한다.

    예전에는 str(error) 만 남겨서 어댑터가 잡아 둔 subprocess stderr 가
    사라졌다. 24곡 배치의 크래시 49건이 전부 `exited with 1` 로만 남은
    원인이고, `.hydra-run/inference.log` 는 크래시 시도에서 비어 있어
    대체 근거도 없었다.
    """
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    class CrashingGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            super().generate_map(request, workdir)
            raise WorkerError(
                ErrorCode.CHART_GENERATION_FAILED,
                "inference failed: exited with 1",
                context={"stderr": "torch.cuda.OutOfMemoryError: CUDA out of memory"},
            )

    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=CrashingGenerator(),
        seed=0,
    )

    fallback = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert fallback.provenance == "SAFE_FALLBACK"
    errors = [json.loads(entry) for entry in fallback.attempt_errors]
    assert len(errors) == 3
    assert all(entry["code"] == "CHART_GENERATION_FAILED" for entry in errors)
    assert all(
        "CUDA out of memory" in entry["context"]["stderr"] for entry in errors
    )
    assert len(errors) == 3
    journal_path = tmp_path / "attempt-journal.jsonl"
    journal = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(journal) == 12 * 3 * 2
    assert [entry["sequence"] for entry in journal] == list(
        range(1, len(journal) + 1)
    )
    assert [entry["eventType"] for entry in journal] == [
        event_type
        for _variant in range(12)
        for _attempt in range(3)
        for event_type in ("INFERENCE_STARTED", "INFERENCE_FAILED")
    ]
    assert {(entry["keyMode"], entry["difficulty"]) for entry in journal} == {
        (key_mode, difficulty)
        for key_mode in (4, 6, 7)
        for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
    }
    assert all(
        entry["payload"]["error"]["code"] == "CHART_GENERATION_FAILED"
        for entry in journal
        if entry["eventType"] == "INFERENCE_FAILED"
    )


@pytest.mark.parametrize(
    ("boundary_policy_mode", "expected_music_end_ms", "expected_last_attack_ms"),
    [
        ("SHADOW", 30_000, 30_000),
        ("EXPERIMENTAL_ENFORCED", 21_000, 20_000),
    ],
)
def test_uncalibrated_music_end_is_shadowed_unless_experimental_mode_is_explicit(
    tmp_path: Path,
    boundary_policy_mode: str,
    expected_music_end_ms: int,
    expected_last_attack_ms: int,
):
    duration_ms = 30_000
    prepared = _prepared(
        tmp_path,
        duration_ms=duration_ms,
        boundary_policy_mode=boundary_policy_mode,
    )
    authority = _authority(prepared, tmp_path)

    frame_ms = 100.0
    frame_count = duration_ms // int(frame_ms)
    rms_db = np.full(frame_count, -10.0)
    rms_db[200:] = -80.0  # 20초 이후는 무음
    analysis = replace(
        _analysis(),
        activity=AudioActivity(
            frame_ms=frame_ms,
            rms_db=rms_db,
            floor_db=-60.0,
            active_onset_ms=tuple(range(0, 20_000, 250)),
        ),
    )

    generator = RecordingGenerator()
    _run_generation(
        prepared,
        authority,
        analysis,
        tmp_path,
        generator=generator,
        seed=0,
    )

    music_ends = {request.music_end_ms for request in generator.map_calls}
    assert music_ends == {expected_music_end_ms}
    assert {request.last_attack_ms for request in generator.map_calls} == {
        expected_last_attack_ms
    }
    assert {request.generation_end_ms for request in generator.map_calls} == {
        30_000
    }
    assert all(request.duration_ms == duration_ms for request in generator.map_calls)


def test_music_end_estimate_keeps_full_audio_when_the_tail_is_short(tmp_path: Path):
    """짧은 꼬리는 자르지 않는다. 정상 아웃트로를 보존한다."""
    duration_ms = 30_000
    prepared = _prepared(tmp_path, duration_ms=duration_ms)
    authority = _authority(prepared, tmp_path)

    frame_ms = 100.0
    frame_count = duration_ms // int(frame_ms)
    rms_db = np.full(frame_count, -10.0)
    rms_db[290:] = -80.0  # 1초짜리 꼬리 무음
    analysis = replace(
        _analysis(),
        activity=AudioActivity(
            frame_ms=frame_ms,
            rms_db=rms_db,
            floor_db=-60.0,
            active_onset_ms=tuple(range(0, 29_000, 250)),
        ),
    )

    generator = RecordingGenerator()
    _run_generation(
        prepared,
        authority,
        analysis,
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert {request.music_end_ms for request in generator.map_calls} == {duration_ms}
    assert {request.last_attack_ms for request in generator.map_calls} == {
        duration_ms
    }
    assert {request.generation_end_ms for request in generator.map_calls} == {
        duration_ms
    }
