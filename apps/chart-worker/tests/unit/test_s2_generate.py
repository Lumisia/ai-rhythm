import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chart_worker.analysis.activity import AudioActivity, SongBoundaryContract
from chart_worker.analysis.coverage_jury import LocalAudioGapEvidence
from chart_worker.analysis.coverage_opportunity import CoverageKind, CoverageOpportunity
from chart_worker.analysis.intro_anchor import IntroAnchorEvidence
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.song_context import SongAnalysisContext
from chart_worker.analysis.terminal_silence import (
    TerminalSilenceObservation,
    TerminalThresholdCandidate,
)
from chart_worker.analysis.timing_diagnostics import (
    TimingCoverageGap,
    TimingMetrics,
    TimingSection,
)
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation import family_selection
from chart_worker.generation.candidate_payload_store import (
    CandidatePayloadIntegrityError,
)
from chart_worker.generation.candidate_repository import CandidateRepository
from chart_worker.generation.difficulty_family_compiler import (
    DifficultyFamilyCompilerDecision,
)
from chart_worker.generation.difficulty_shadow_challenger import (
    DifficultyShadowPartialDecision,
    DifficultyShadowPartialPlan,
    DifficultyShadowTarget,
)
from chart_worker.generation.generation_control import (
    MAX_CRASH_ATTEMPTS,
    MAX_TOTAL_ATTEMPTS,
    MAX_VARIANT_ATTEMPTS,
    AttemptBudgetState,
    RecoveryKind,
)
from chart_worker.generation.intro_exact_reselection import try_exact_intro_candidate
from chart_worker.generation.intro_family_recovery import (
    intro_candidate_view,
    intro_phrase_pair_review,
)
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent, parse_osu_file
from chart_worker.generation.osu_writer import timing_to_osu_mania
from chart_worker.generation.params import GenerationRequest
from chart_worker.generation.partial_remap import PartialRemapWindow
from chart_worker.generation.required_gameplay_interval import (
    RequiredGameplayEvidenceClass,
    RequiredGameplayGroupType,
    RequiredGameplayIntervalMode,
    RequiredGameplayIntervalV1,
)
from chart_worker.generation.resnap_diagnostics import (
    ResnapCollision,
    ResnapDiagnostics,
)
from chart_worker.hashing import sha256_file
from chart_worker.schema.note import NoteEvent
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.stages import s2_generate
from chart_worker.stages.s2_generate import run_generation
from chart_worker.stages.timing_feedback import RetryTimingSignal
from chart_worker.stages.types import (
    AdditionalInferenceBudget,
    PreparedAudio,
    SongTimingAuthority,
)
from chart_worker.validation.intro_start_contract import IntroStartContract
from chart_worker.validation.leading_timing_coverage import LeadingTimingCoverage
from chart_worker.validation.quality_gate import (
    GateAction,
    GateAxis,
    evaluate_chart_candidate,
)
from chart_worker.validation.timing_integrity import (
    TimingIntegrityAssessment,
    TimingIntegrityStatus,
)
from chart_worker.validation.timing_review import TimingAuthorityAction


def _run_generation(*args, **kwargs):
    """run_generation 은 GenerationOutcome 을 돌려준다.

    이 파일의 기존 검증은 발행된 변형 튜플만 보므로 여기서 풀어 준다.
    missing 을 확인하는 테스트는 run_generation 을 직접 호출한다.
    """
    return run_generation(*args, **kwargs).variants


def _observed_required_interval() -> RequiredGameplayIntervalV1:
    return RequiredGameplayIntervalV1(
        start_ms=430,
        end_ms=570,
        minimum_complete_groups=1,
        allowed_group_types=(
            RequiredGameplayGroupType.TAP,
            RequiredGameplayGroupType.HOLD_START,
        ),
        evidence_class=RequiredGameplayEvidenceClass.BROADBAND_ATTACK,
        evidence_digest="a" * 64,
        mode=RequiredGameplayIntervalMode.SHADOW_ENFORCE,
    )


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


def _active_analysis(duration_ms: int, *, onset_step_ms: int = 500) -> OnsetAnalysis:
    frame_ms = 100
    frame_count = duration_ms // frame_ms + 1
    rows = tuple(range(250, duration_ms, onset_step_ms))
    strength = np.zeros(frame_count)
    for time_ms in rows:
        strength[round(time_ms / frame_ms)] = 1.0
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strength,
        band_strength=np.vstack((strength, strength, strength)),
        onset_ms=rows,
        activity=AudioActivity(
            frame_ms=frame_ms,
            rms_db=np.full(frame_count, -12.0),
            floor_db=-40.0,
            active_onset_ms=rows,
        ),
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
        for difficulty_index, difficulty in enumerate(("EASY", "NORMAL", "HARD", "EXPERT")):
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
        and note.time_ms + (note.duration_ms if note.kind == "HOLD" else 0) <= duration_ms
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


def test_final_song_selection_uses_v2_after_recovery_when_configured(
    monkeypatch,
    tmp_path: Path,
):
    prepared = replace(_prepared(tmp_path), difficulty_selector_mode="V2")
    authority = _authority(prepared, tmp_path)
    observed_modes: list[str] = []

    real_compare = family_selection.compare_song_selection

    def record_mode(*args, mode, **kwargs):
        observed_modes.append(mode)
        return real_compare(*args, mode=mode, **kwargs)

    monkeypatch.setattr(s2_generate, "_compare_song_selection", record_mode)

    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=RecordingGenerator(),
        seed=0,
    )

    assert observed_modes == ["V2"]
    assert outcome.song_selection_shadow.mode == "V2"


def test_candidate_payload_corruption_after_selection_fails_before_publication(
    monkeypatch,
    tmp_path: Path,
):
    prepared = replace(_prepared(tmp_path), difficulty_selector_mode="V2")
    authority = _authority(prepared, tmp_path)
    real_compare = family_selection.compare_song_selection

    def corrupt_payload(*args, mode, **kwargs):
        result = real_compare(*args, mode=mode, **kwargs)
        snapshot = result[1].replay_input.candidates[0]
        (tmp_path / snapshot.candidate_payload_ref).write_bytes(b"corrupt")
        return result

    monkeypatch.setattr(s2_generate, "_compare_song_selection", corrupt_payload)

    with pytest.raises(CandidatePayloadIntegrityError, match="hash mismatch"):
        run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=RecordingGenerator(),
            seed=0,
        )

    assert not any((tmp_path / "raw").glob("[467]k-*.osu"))


def test_final_song_selection_refreshes_intro_family_evidence(
    monkeypatch,
    tmp_path: Path,
):
    prepared = replace(_prepared(tmp_path), difficulty_selector_mode="V2")
    authority = _authority(prepared, tmp_path)
    refreshes: list[int] = []

    real_review = s2_generate._intro_phrase_family_reviews

    def record_refresh(selections, **kwargs):
        refreshes.append(sum(len(assignment) for _states, assignment, _review in selections))
        return real_review(selections, **kwargs)

    monkeypatch.setattr(s2_generate, "_intro_phrase_family_reviews", record_refresh)

    run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=RecordingGenerator(),
        seed=0,
    )

    assert refreshes == [12]


def test_safe_family_assignment_runs_after_existing_candidate_recovery(
    monkeypatch,
    tmp_path: Path,
):
    prepared = replace(_prepared(tmp_path), difficulty_selector_mode="V2")
    authority = _authority(prepared, tmp_path)
    observed_key_modes: list[tuple[int, ...]] = []
    observed_final_ordering: list[bool] = []
    real_apply = family_selection.apply_safe_family_assignment

    def record_apply(selections, **kwargs):
        observed_key_modes.append(
            tuple(next(iter(states.values())).key_mode for states, _assignment, _review in selections)
        )
        observed_final_ordering.append(kwargs.get("post_resolution_ordering", False))
        return real_apply(selections, **kwargs)

    monkeypatch.setattr(s2_generate, "_apply_safe_family_assignment", record_apply)

    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=RecordingGenerator(),
        seed=0,
    )

    assert observed_key_modes == [(4, 6, 7)]
    assert observed_final_ordering == [True]
    assert tuple(decision.key_mode for decision in outcome.safe_family_assignments) == (4, 6, 7)
    assert all(
        decision.post_resolution_ordering_status != "NOT_REQUESTED"
        for decision in outcome.safe_family_assignments
    )
    assert all(decision.additional_model_calls == 0 for decision in outcome.safe_family_assignments)


def test_difficulty_family_compiler_shadow_is_opt_in_and_never_published(
    monkeypatch,
    tmp_path: Path,
):
    compile_calls: list[tuple[int, tuple[str, ...]]] = []

    def compile_shadow(slots, **kwargs):
        del kwargs
        key_mode = slots[0].generated.key_mode
        compile_calls.append(
            (key_mode, tuple(slot.difficulty for slot in slots))
        )
        return DifficultyFamilyCompilerDecision(
            key_mode=key_mode,
            status="NOT_NEEDED",
            reason="FAMILY_ALREADY_SEPARATED",
            anchor_candidate_id=None,
            anchor_source_difficulty=None,
            proposals=(),
            proposals_evaluated=0,
        )

    monkeypatch.setattr(
        s2_generate,
        "compile_difficulty_family_shadow",
        compile_shadow,
        raising=False,
    )

    disabled_dir = tmp_path / "disabled"
    disabled_prepared = replace(
        _prepared(disabled_dir),
        difficulty_family_compiler_shadow_enabled=False,
    )
    disabled_generator = RecordingGenerator()
    disabled = run_generation(
        disabled_prepared,
        _authority(disabled_prepared, disabled_dir),
        _analysis(),
        disabled_dir,
        generator=disabled_generator,
        seed=0,
    )
    assert compile_calls == []

    enabled_dir = tmp_path / "enabled"
    enabled_prepared = replace(
        _prepared(enabled_dir),
        difficulty_family_compiler_shadow_enabled=True,
    )
    enabled_generator = RecordingGenerator()
    enabled = run_generation(
        enabled_prepared,
        _authority(enabled_prepared, enabled_dir),
        _analysis(),
        enabled_dir,
        generator=enabled_generator,
        seed=0,
    )

    assert compile_calls == [
        (4, tuple(DIFFICULTIES)),
        (6, tuple(DIFFICULTIES)),
        (7, tuple(DIFFICULTIES)),
    ]
    assert len(enabled_generator.map_calls) == len(disabled_generator.map_calls)
    assert enabled.additional_inference_calls == disabled.additional_inference_calls
    assert tuple(
        (
            variant.key_mode,
            variant.difficulty,
            variant.provenance,
            variant.generated.notes,
        )
        for variant in enabled.variants
    ) == tuple(
        (
            variant.key_mode,
            variant.difficulty,
            variant.provenance,
            variant.generated.notes,
        )
        for variant in disabled.variants
    )
    assert tuple(
        decision.to_report()
        for decision in enabled.difficulty_family_compiler_shadow
    ) == tuple(
        DifficultyFamilyCompilerDecision(
            key_mode=key_mode,
            status="NOT_NEEDED",
            reason="FAMILY_ALREADY_SEPARATED",
            anchor_candidate_id=None,
            anchor_source_difficulty=None,
            proposals=(),
            proposals_evaluated=0,
        ).to_report()
        for key_mode in KEY_MODES
    )


def test_difficulty_family_compiler_shadow_failure_cannot_abort_publication(
    monkeypatch,
    tmp_path: Path,
):
    prepared = replace(
        _prepared(tmp_path),
        difficulty_family_compiler_shadow_enabled=True,
    )

    def fail_compiler(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("compiler fixture failure")

    monkeypatch.setattr(
        s2_generate,
        "compile_difficulty_family_shadow",
        fail_compiler,
    )

    outcome = run_generation(
        prepared,
        _authority(prepared, tmp_path),
        _analysis(),
        tmp_path,
        generator=RecordingGenerator(),
        seed=0,
    )

    assert len(outcome.variants) == 12
    assert tuple(
        (decision.key_mode, decision.status, decision.reason)
        for decision in outcome.difficulty_family_compiler_shadow
    ) == (
        (4, "UNAVAILABLE", "COMPILER_EXECUTION_FAILED"),
        (6, "UNAVAILABLE", "COMPILER_EXECUTION_FAILED"),
        (7, "UNAVAILABLE", "COMPILER_EXECUTION_FAILED"),
    )
    assert all(
        decision.to_report()["failureType"] == "RuntimeError"
        for decision in outcome.difficulty_family_compiler_shadow
    )
    assert all(
        decision.solver_wall_ms >= 0.0
        and decision.candidate_evaluation_wall_ms == 0.0
        and decision.payload_persistence_wall_ms == 0.0
        for decision in outcome.difficulty_family_compiler_shadow
    )


def test_enforced_resolution_relabels_four_unique_candidates_before_compiling(
    monkeypatch,
    tmp_path: Path,
):
    prepared = replace(
        _prepared(tmp_path),
        difficulty_family_resolution_enabled=True,
    )
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    ratings = {"EASY": 4.0, "NORMAL": 3.0, "HARD": 2.0, "EXPERT": 1.0}

    def evaluate(_generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        rating = ratings[requested_difficulty]
        result = _acceptance_with_rating(accepted, rating)
        assert result.profile is not None
        return replace(
            result,
            profile=replace(
                result.profile,
                difficulty_vector_v2=replace(
                    result.profile.difficulty_vector_v2,
                    ordering_score=rating,
                ),
            ),
        )

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)

    class UniqueGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return replace(
                generated,
                notes=[NoteEvent(250 + int(request.seed or 0), request.key_mode - 1)],
            )

    generator = UniqueGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 12
    assert outcome.additional_inference_calls == 0
    assert all(
        decision.status == "NOT_NEEDED"
        for decision in outcome.difficulty_family_compiler_shadow
    )
    assert all(variant.provenance == "PRIMARY" for variant in outcome.variants)
    assert outcome.song_selection_evidence_v3 is not None
    evidence_assignment = dict(outcome.song_selection_evidence_v3.current_assignment)
    evidence_candidates = {
        candidate.candidate_id: candidate
        for candidate in outcome.song_selection_evidence_v3.candidates
    }
    for variant in outcome.variants:
        candidate_id = evidence_assignment[
            f"{variant.key_mode}K:{variant.difficulty}"
        ]
        assert candidate_id is not None
        evidence_candidate = evidence_candidates[candidate_id]
        assert evidence_candidate.difficulty == variant.source_difficulty
        assert sha256_file(tmp_path / evidence_candidate.candidate_payload_ref) == (
            evidence_candidate.candidate_payload_sha256
        )
    for key_mode in KEY_MODES:
        family = tuple(
            variant for variant in outcome.variants if variant.key_mode == key_mode
        )
        project_ratings = tuple(
            variant.acceptance.profile.difficulty.project_rating
            for variant in family
            if variant.acceptance.profile is not None
        )
        assert all(
            harder > easier for easier, harder in pairwise(project_ratings)
        )


def test_difficulty_family_compiler_shadow_persists_only_unpublished_proposals(
    monkeypatch,
    tmp_path: Path,
):
    prepared = replace(
        _prepared(tmp_path),
        difficulty_family_compiler_shadow_enabled=True,
    )
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)

    def with_difficulty_metrics(acceptance, rating: float):
        assert acceptance.profile is not None
        return replace(
            acceptance,
            profile=replace(
                acceptance.profile,
                difficulty=replace(
                    acceptance.profile.difficulty,
                    project_rating=rating,
                ),
                difficulty_vector_v2=replace(
                    acceptance.profile.difficulty_vector_v2,
                    ordering_score=rating,
                ),
            ),
        )

    current_ratings = {
        "EASY": 4.0,
        "NORMAL": 3.9,
        "HARD": 3.8,
        "EXPERT": 3.7,
    }

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        rating = (
            len(generated.notes) / 10.0
            if generated.osu_text
            else current_ratings[requested_difficulty]
        )
        return with_difficulty_metrics(accepted, rating)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)
    monkeypatch.setattr(s2_generate, "_intro_anchor_covered", lambda *_args: True)

    class DenseGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            self.map_calls.append(request)
            self.map_workdirs.append(workdir)
            return GeneratedChart(
                notes=[
                    NoteEvent(100 + index * 40, index % request.key_mode)
                    for index in range(40)
                ],
                key_mode=request.key_mode,
                osu_text="",
                generator_name="dense-fixture",
                seed=request.seed,
                bpm_events=authority.bpm_events,
            )

    generator = DenseGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(outcome.variants) == 12
    assert all(len(variant.generated.notes) == 40 for variant in outcome.variants)
    assert outcome.additional_inference_calls == 0
    assert len(generator.map_calls) == 12
    assert all(
        decision.status == "COMPILED" for decision in outcome.difficulty_family_compiler_shadow
    )
    for decision in outcome.difficulty_family_compiler_shadow:
        assert decision.solver_wall_ms >= 0.0
        assert decision.candidate_evaluation_wall_ms >= 0.0
        assert decision.payload_persistence_wall_ms >= 0.0
        assert [proposal.target_difficulty for proposal in decision.proposals] == list(DIFFICULTIES)
        for proposal in decision.proposals:
            assert proposal.candidate_payload_ref is not None
            payload = tmp_path / proposal.candidate_payload_ref
            assert payload.is_file()
            assert sha256_file(payload) == proposal.candidate_payload_sha256

    active_dir = tmp_path / "active"
    active_prepared = replace(
        _prepared(active_dir),
        difficulty_family_compiler_shadow_enabled=False,
        difficulty_family_resolution_enabled=True,
    )
    authority = _authority(active_prepared, active_dir)
    accepted = _pass_acceptance(authority)
    active_generator = DenseGenerator()
    active = run_generation(
        active_prepared,
        authority,
        _analysis(),
        active_dir,
        generator=active_generator,
        seed=0,
    )

    assert len(active.variants) == 12
    assert len(active_generator.map_calls) == 12
    assert active.additional_inference_calls == 0
    for key_mode in KEY_MODES:
        family = tuple(
            variant for variant in active.variants if variant.key_mode == key_mode
        )
        assert len({variant.generated.osu_text for variant in family}) == 4
        compiled = tuple(
            variant for variant in family if variant.provenance == "SAFE_FALLBACK"
        )
        assert len(compiled) >= 3
        assert all(
            variant.recovery_reason == "DIFFICULTY_FAMILY_COMPILER_V1"
            for variant in compiled
        )
        ratings = tuple(
            variant.acceptance.profile.difficulty.project_rating
            for variant in family
            if variant.acceptance.profile is not None
        )
        assert all(harder > easier for easier, harder in pairwise(ratings))
    assert all(
        decision.status == "COMPILED"
        and decision.publication_mode == "ENFORCED"
        and decision.to_report()["mutatesSelection"] is False
        and decision.to_report()["mutatesPublishedCharts"] is False
        and decision.to_report()["eligibleForFinalSelection"] is True
        for decision in active.difficulty_family_compiler_shadow
    )


def test_unresolved_family_scope_follows_final_selector_not_compiler_status():
    decisions = (
        SimpleNamespace(
            key_mode=4,
            unique_payload_status="SATISFIED",
            family_feasibility_status="SATISFIED",
            selected_score=SimpleNamespace(difficulty_violations=0),
        ),
        SimpleNamespace(
            key_mode=6,
            unique_payload_status="UNAVAILABLE",
            family_feasibility_status="UNAVAILABLE",
            selected_score=SimpleNamespace(difficulty_violations=0),
        ),
        SimpleNamespace(
            key_mode=7,
            unique_payload_status="SATISFIED",
            family_feasibility_status="UNAVAILABLE",
            selected_score=SimpleNamespace(difficulty_violations=2),
        ),
    )

    assert s2_generate._unresolved_family_key_modes(decisions) == {6, 7}


def test_unresolved_key_supplies_a_complete_canonical_fallback_family(
    monkeypatch,
    tmp_path: Path,
):
    states = {
        difficulty: s2_generate._VariantState(4, difficulty, index)
        for index, difficulty in enumerate(DIFFICULTIES)
    }
    selections = [
        (
            states,
            {difficulty: SimpleNamespace() for difficulty in DIFFICULTIES},
            None,
        )
    ]
    calls: list[tuple[str, bool]] = []

    def fake_fallback(state, **kwargs):
        calls.append((state.difficulty, kwargs["register"]))
        return SimpleNamespace(
            provenance="SAFE_FALLBACK",
            recovery_reason=kwargs["recovery_reason"],
            request=SimpleNamespace(difficulty=state.difficulty),
            osu_text=f"fallback-{state.difficulty}",
        )

    monkeypatch.setattr(s2_generate, "_safe_fallback_candidate", fake_fallback)

    supplied = s2_generate._ensure_coherent_safe_fallback_family_candidates(
        selections,
        unresolved_key_modes={4},
        prepared=SimpleNamespace(),
        authority=SimpleNamespace(),
        onset_analysis=SimpleNamespace(),
        run_dir=tmp_path,
        base_seed=17,
        authority_epoch=1,
    )

    assert supplied == {4}
    assert calls == [(difficulty, False) for difficulty in DIFFICULTIES]
    assert all(
        len(states[difficulty].candidates.safe_fallbacks) == 1
        for difficulty in DIFFICULTIES
    )


def test_coherent_fallback_supply_reuses_existing_canonical_slot(
    monkeypatch,
    tmp_path: Path,
):
    states = {
        difficulty: s2_generate._VariantState(4, difficulty, index)
        for index, difficulty in enumerate(DIFFICULTIES)
    }
    existing = SimpleNamespace(
        provenance="SAFE_FALLBACK",
        recovery_reason="NO_STRUCTURALLY_SAFE_MODEL_CANDIDATE",
        request=SimpleNamespace(difficulty="HARD"),
        osu_text="fallback-HARD",
    )
    states["HARD"].candidates.add_safe_fallback(existing)
    calls: list[str] = []

    def fake_fallback(state, **kwargs):
        calls.append(state.difficulty)
        return SimpleNamespace(
            provenance="SAFE_FALLBACK",
            recovery_reason=kwargs["recovery_reason"],
            request=SimpleNamespace(difficulty=state.difficulty),
            osu_text=f"fallback-{state.difficulty}",
        )

    monkeypatch.setattr(s2_generate, "_safe_fallback_candidate", fake_fallback)

    supplied = s2_generate._ensure_coherent_safe_fallback_family_candidates(
        [
            (
                states,
                {difficulty: SimpleNamespace() for difficulty in DIFFICULTIES},
                None,
            )
        ],
        unresolved_key_modes={4},
        prepared=SimpleNamespace(),
        authority=SimpleNamespace(),
        onset_analysis=SimpleNamespace(),
        run_dir=tmp_path,
        base_seed=17,
        authority_epoch=1,
    )

    assert supplied == {4}
    assert calls == ["EASY", "NORMAL", "EXPERT"]
    assert states["HARD"].candidates.safe_fallbacks == (existing,)


def test_final_song_selection_reuses_an_existing_ordered_candidate(
    tmp_path: Path,
):
    prepared = replace(_prepared(tmp_path), difficulty_selector_mode="V2")
    authority = _authority(prepared, tmp_path)
    base_acceptance = _pass_acceptance(authority)
    selections = []
    current_expert = None
    ordered_expert = None

    def make_candidate(
        *,
        key_mode: int,
        difficulty: str,
        seed: int,
        ordering_score: float,
    ):
        acceptance = _acceptance_for_difficulty(base_acceptance, difficulty)
        assert acceptance.profile is not None
        acceptance = replace(
            acceptance,
            profile=replace(
                acceptance.profile,
                difficulty_vector_v2=replace(
                    acceptance.profile.difficulty_vector_v2,
                    ordering_score=ordering_score,
                ),
            ),
        )
        request = GenerationRequest(
            audio_path=prepared.normalized.path,
            timing_reference_path=authority.reference_path,
            key_mode=key_mode,
            difficulty=difficulty,
            seed=seed,
            duration_ms=prepared.normalized.duration_ms,
        )
        return s2_generate._Candidate(
            request=request,
            generated=GeneratedChart(
                notes=[NoteEvent(500, 0)],
                key_mode=key_mode,
                osu_text="",
                generator_name="final-family-fixture",
                seed=seed,
                bpm_events=authority.bpm_events,
            ),
            acceptance=acceptance,
            osu_text="",
            workdir=tmp_path / f"{key_mode}k-{difficulty}-{seed}",
            attempt=1,
            seed=seed,
            provenance="PRIMARY",
            intro_anchor_covered=True,
        )

    for key_index, key_mode in enumerate((4, 6, 7)):
        states = {}
        assignment = {}
        for difficulty_index, difficulty in enumerate(("EASY", "NORMAL", "HARD", "EXPERT")):
            state = s2_generate._VariantState(
                key_mode,
                difficulty,
                key_index * 4 + difficulty_index,
            )
            score = float(difficulty_index + 1)
            if key_mode == 4 and difficulty == "EXPERT":
                score = 2.5
            candidate = make_candidate(
                key_mode=key_mode,
                difficulty=difficulty,
                seed=difficulty_index,
                ordering_score=score,
            )
            state.candidates.admit(candidate)
            states[difficulty] = state
            assignment[difficulty] = candidate
            if key_mode == 4 and difficulty == "EXPERT":
                current_expert = candidate
                ordered_expert = make_candidate(
                    key_mode=key_mode,
                    difficulty=difficulty,
                    seed=99,
                    ordering_score=4.0,
                )
                state.candidates.admit(ordered_expert)
        selections.append((states, assignment, s2_generate._family_review(assignment)))

    intro_contract = IntroStartContract(
        version="intro-start-contract-v2",
        canonical_first_row_ms=500,
        candidate_support_count=12,
        raw_supported=True,
        audio_supported=True,
        grid_distance_ms=0,
        candidates=(),
    )
    selected, comparison = family_selection.compare_song_selection(
        selections,
        prepared=prepared,
        authority=authority,
        run_dir=tmp_path,
        intro_contract=intro_contract,
        boundary=None,
        mode="V2",
    )

    selected_4k = next(
        assignment for states, assignment, _review in selected if states["EASY"].key_mode == 4
    )
    assert selected_4k["EXPERT"] is ordered_expert
    assert selected_4k["EXPERT"] is not current_expert
    assert comparison.current_score.difficulty_violations == 1
    assert comparison.shadow_score.difficulty_violations == 0


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


def test_coverage_repair_timing_guard_rejects_material_metric_regression():
    authority = SongTimingAuthority(
        reference_path=Path("timing-reference.osu"),
        sha256="timing",
        audio_sha256="audio",
        bpm_events=(OsuBpmEvent(0, 120.0),),
        generator_name="test",
        seed=0,
        mode="STANDARD",
        attempt_count=1,
    )
    baseline = _pass_acceptance(authority)
    assert baseline.timing.overall.matched_f1_50 is not None
    assert baseline.timing.overall.matched_precision_50 is not None
    regressed = replace(
        baseline,
        timing=replace(
            baseline.timing,
            overall=replace(
                baseline.timing.overall,
                matched_f1_50=baseline.timing.overall.matched_f1_50 - 0.01,
                matched_precision_50=(
                    baseline.timing.overall.matched_precision_50 - 0.01
                ),
            ),
        ),
    )

    assert s2_generate._coverage_repair_preserves_timing(baseline, baseline)
    assert not s2_generate._coverage_repair_preserves_timing(baseline, regressed)


def test_candidate_stable_id_is_bound_to_exact_serialized_payload(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    request = GenerationRequest(
        audio_path=prepared.normalized.path,
        timing_reference_path=authority.reference_path,
        key_mode=4,
        difficulty="EASY",
        seed=7,
        duration_ms=prepared.normalized.duration_ms,
    )
    candidate = s2_generate._Candidate(
        request=request,
        generated=GeneratedChart(
            notes=[NoteEvent(500, 0)],
            key_mode=4,
            osu_text="payload-a",
            generator_name="payload-identity-fixture",
            seed=7,
            bpm_events=authority.bpm_events,
        ),
        acceptance=_pass_acceptance(authority),
        osu_text="payload-a",
        workdir=tmp_path / "same-payload-ref",
        attempt=1,
        seed=7,
        provenance="PRIMARY",
    )
    changed = replace(candidate, osu_text="payload-b")

    assert family_selection.candidate_stable_id(
        candidate,
        key_mode=4,
        difficulty="EASY",
        run_dir=tmp_path,
    ) != family_selection.candidate_stable_id(
        changed,
        key_mode=4,
        difficulty="EASY",
        run_dir=tmp_path,
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
    opportunity = CoverageOpportunity(
        version="coverage-opportunity-v4",
        start_ms=10_000,
        end_ms=20_000,
        beat_count=20.0,
        strong_attack_count=8,
        active_onset_count=20,
        hold_occupancy_ratio=0.0,
        active_frame_ratio=1.0,
        strong_attack_threshold=0.5,
        evidence_confidence="SUFFICIENT",
        kind=CoverageKind.ATTACK_REQUIRED,
        attack_evidence_scope="GLOBAL",
    )
    gap = TimingCoverageGap(
        start_ms=10_000,
        end_ms=20_000,
        onset_count=20,
        active_onset_count=20,
        active_frame_ratio=1.0,
        position="MIDDLE",
        opportunity=opportunity,
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


def test_raw_playtest_score_prefers_the_shorter_active_gap_when_counts_match(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    base = _coverage_retry_acceptance(authority)

    def candidate(*, seed: int, end_ms: int):
        gap = replace(base.timing.coverage_gaps[0], start_ms=0, end_ms=end_ms)
        acceptance = replace(
            base,
            timing=replace(base.timing, coverage_gaps=(gap,)),
        )
        request = GenerationRequest(
            audio_path=prepared.normalized.path,
            timing_reference_path=authority.reference_path,
            key_mode=4,
            difficulty="EASY",
            seed=seed,
            duration_ms=prepared.normalized.duration_ms,
        )
        return s2_generate._Candidate(
            request=request,
            generated=GeneratedChart(
                notes=[NoteEvent(0, 0)],
                key_mode=4,
                osu_text="",
                generator_name="raw-score-fixture",
                seed=seed,
                bpm_events=authority.bpm_events,
            ),
            acceptance=acceptance,
            osu_text="",
            workdir=tmp_path / f"raw-{seed}",
            attempt=1,
            seed=seed,
            provenance="RAW_UNVERIFIED",
        )

    longer = candidate(seed=0, end_ms=166_000)
    shorter = candidate(seed=1, end_ms=8_000)

    assert min((longer, shorter), key=s2_generate._raw_playtest_score) is shorter


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


def test_intro_contract_reports_reassigned_candidate_as_target_slot(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    song_context = SongAnalysisContext.build(
        authority,
        _analysis(),
        duration_ms=prepared.normalized.duration_ms,
    )
    source = SimpleNamespace(
        request=SimpleNamespace(key_mode=4, difficulty="EASY"),
        generated=SimpleNamespace(notes=[NoteEvent(500, 0)]),
        seed=7,
    )

    view = intro_candidate_view(
        source,
        key_mode=4,
        difficulty="EXPERT",
        song_context=song_context,
    )

    assert (view.key_mode, view.difficulty, view.first_row_ms) == (4, "EXPERT", 500)
    assert source.request.difficulty == "EASY"


@pytest.mark.parametrize(
    ("enabled", "expected_map_calls", "expected_additional_calls", "expected_shadow_count"),
    [
        (False, 14, 0, 0),
        (True, 15, 1, 1),
    ],
)
def test_difficulty_shadow_challenger_is_opt_in_and_never_published(
    monkeypatch,
    tmp_path: Path,
    enabled: bool,
    expected_map_calls: int,
    expected_additional_calls: int,
    expected_shadow_count: int,
):
    prepared = replace(
        _prepared(tmp_path),
        difficulty_shadow_challenger_enabled=enabled,
    )
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    hard_reject = _acceptance_with_action(authority, GateAction.RETRY_MAP)

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        if generated.key_mode == 4 and requested_difficulty == "EXPERT":
            if generated.generator_name == "adaptive-recovery-v1":
                return _acceptance_with_rating(accepted, 1.94)
            if generated.seed == 39:
                return _acceptance_with_rating(accepted, 4.50)
            return _acceptance_with_rating(hard_reject, 1.00)
        result = _acceptance_for_difficulty(accepted, requested_difficulty)
        if generated.key_mode == 4 and requested_difficulty == "HARD":
            result = _acceptance_with_rating(result, 3.68)
        return result

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)
    monkeypatch.setattr(s2_generate, "_intro_anchor_covered", lambda *_args: True)
    planned_source = None

    def plan(target, candidates, _bpm_events, *, duration_ms):
        nonlocal planned_source
        assert duration_ms == prepared.normalized.duration_ms
        # Planner eligibility/provenance is unit-tested separately. This
        # integration fixture supplies the only preserved source so it can
        # isolate partial-request and publication-boundary behavior.
        planned_source = candidates[0]
        return DifficultyShadowPartialDecision(
            reason="PARTIAL_NEAR_SOLUTION_SELECTED",
                plan=DifficultyShadowPartialPlan(
                    target=target,
                    source=planned_source,
                    window=PartialRemapWindow(start_ms=0, end_ms=1_000),
                    required_gameplay_interval=_observed_required_interval(),
                ),
            considered_candidate_count=len(candidates),
        )

    monkeypatch.setattr(
        s2_generate,
        "plan_difficulty_shadow_partial_repair",
        plan,
        raising=False,
    )

    class PartialMergingGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            if request.add_to_beatmap:
                assert not workdir.exists() or not any(workdir.iterdir())
            generated = super().generate_map(request, workdir)
            if not request.add_to_beatmap:
                return generated
            assert planned_source is not None
            return replace(
                generated,
                notes=[NoteEvent(500, request.key_mode - 1), *(
                    note
                    for note in planned_source.generated.notes
                    if note.time_ms > request.partial_end_ms
                )],
            )

    generator = PartialMergingGenerator()

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
    assert expert.provenance == "SAFE_FALLBACK"
    assert expert.family_assignment_kind == "ORIGINAL"
    assert expert.source_difficulty == "EXPERT"
    assert len(generator.map_calls) == expected_map_calls
    assert outcome.additional_inference_calls == expected_additional_calls
    assert outcome.additional_inference_work_ms == (1_000 if enabled else 0)
    assert [
        request.seed
        for request in generator.map_calls
        if (request.key_mode, request.difficulty) == (4, "EXPERT")
    ] == ([3, 15, 27, 39] if enabled else [3, 15, 27])
    assert outcome.song_selection_evidence_v3 is not None
    shadow_records = [
        candidate
        for candidate in outcome.song_selection_evidence_v3.candidates
        if candidate.candidate_role == "SHADOW_CHALLENGER"
    ]
    assert len(shadow_records) == expected_shadow_count
    if enabled:
        assert planned_source is not None
        shadow_request = generator.map_calls[-1]
        assert shadow_request.timing_reference_path != authority.reference_path
        assert shadow_request.timing_reference_path.read_text(encoding="utf-8") == (
            planned_source.osu_text
        )
        assert shadow_request.partial_start_ms == 0
        assert shadow_request.partial_end_ms == 1_000
        assert shadow_request.add_to_beatmap is True
        assert (
            shadow_request.required_gameplay_interval
            == _observed_required_interval()
        )
        assert all(
            request.required_gameplay_interval is None
            for request in generator.map_calls[:-1]
        )
        assert any(
            evidence.get("reason")
            == "DIFFICULTY_SHADOW_FALLBACK_VETO_REPORTED"
            and evidence.get("mutatesSelection") is False
            for evidence in expert.attempt_evidence
        )
    assert dict(outcome.song_selection_evidence_v3.current_assignment)["4K:EXPERT"] not in {
        candidate.candidate_id for candidate in shadow_records
    }
    assert outcome.song_selection_shadow is not None
    assert not any(
        snapshot.candidate_id in {candidate.candidate_id for candidate in shadow_records}
        for snapshot in outcome.song_selection_shadow.replay_input.candidates
    )


def test_generic_full_map_shadow_targets_6k_normal_hard_and_preserves_publication(
    monkeypatch,
    tmp_path: Path,
):
    target = DifficultyShadowTarget(
        6,
        "NORMAL",
        "HARD",
        3.0,
        2.0,
        1.3,
        minimum_rating=3.3,
        maximum_rating=4.7,
    )

    def choose(_slots):
        return target

    def no_partial(_target, candidates, _bpm_events, *, duration_ms):
        assert duration_ms == 2_000
        return DifficultyShadowPartialDecision(
            reason="NO_NEAR_SOLUTION_CANDIDATE",
            plan=None,
            considered_candidate_count=len(candidates),
        )

    def evaluate(generated, authority, *args, requested_difficulty, **kwargs):
        del args, kwargs
        accepted = _pass_acceptance(authority)
        result = _acceptance_for_difficulty(accepted, requested_difficulty)
        if generated.key_mode == 6 and requested_difficulty == "NORMAL":
            return _acceptance_with_rating(result, 3.0)
        if generated.key_mode == 6 and requested_difficulty == "HARD":
            return _acceptance_with_rating(
                result,
                3.5 if generated.seed == 18 else 2.0,
            )
        if generated.key_mode == 6 and requested_difficulty == "EXPERT":
            return _acceptance_with_rating(result, 5.0)
        return result

    monkeypatch.setattr(s2_generate, "choose_difficulty_shadow_target", choose)
    monkeypatch.setattr(
        s2_generate,
        "plan_difficulty_shadow_partial_repair",
        no_partial,
    )
    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)
    monkeypatch.setattr(s2_generate, "_intro_anchor_covered", lambda *_args: True)

    disabled_dir = tmp_path / "disabled"
    disabled = _prepared(disabled_dir)
    disabled_authority = _authority(disabled, disabled_dir)
    disabled_generator = RecordingGenerator()
    disabled_outcome = run_generation(
        disabled,
        disabled_authority,
        _analysis(),
        disabled_dir,
        generator=disabled_generator,
        seed=0,
    )

    enabled_dir = tmp_path / "enabled"
    enabled = replace(
        _prepared(enabled_dir),
        difficulty_shadow_challenger_enabled=True,
    )
    enabled_authority = _authority(enabled, enabled_dir)
    enabled_generator = RecordingGenerator()
    enabled_outcome = run_generation(
        enabled,
        enabled_authority,
        _analysis(),
        enabled_dir,
        generator=enabled_generator,
        seed=0,
    )

    assert len(disabled_generator.map_calls) == 12
    assert len(enabled_generator.map_calls) == 13
    assert enabled_outcome.additional_inference_calls == 1
    assert enabled_outcome.additional_inference_work_ms == 2_000
    shadow_request = enabled_generator.map_calls[-1]
    assert (shadow_request.key_mode, shadow_request.difficulty) == (6, "HARD")
    assert shadow_request.requested_star == pytest.approx(2.5)
    assert shadow_request.timing_reference_path == enabled_authority.reference_path
    assert shadow_request.partial_start_ms is None
    assert shadow_request.partial_end_ms is None
    assert shadow_request.add_to_beatmap is False
    assert tuple(
        (item.key_mode, item.difficulty, item.raw_osu_path.read_bytes())
        for item in enabled_outcome.variants
    ) == tuple(
        (item.key_mode, item.difficulty, item.raw_osu_path.read_bytes())
        for item in disabled_outcome.variants
    )

    def unavailable_compiler(selections, **_kwargs):
        return tuple(
            DifficultyFamilyCompilerDecision(
                key_mode=next(iter(states.values())).key_mode,
                status="UNAVAILABLE",
                reason="NO_SAFE_HARD_PROPOSAL",
                anchor_candidate_id=None,
                anchor_source_difficulty=None,
                proposals=(),
                proposals_evaluated=0,
            )
            for states, _assignment, _review in selections
        )

    monkeypatch.setattr(
        s2_generate,
        "_compile_difficulty_family_shadows",
        unavailable_compiler,
    )

    class BoundedRecoveryGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            # Start with exactly one duplicated 6K pair.  The bounded HARD
            # challenger (seed 18) supplies the fourth unique payload; every
            # other slot is already unique and must not be regenerated.
            time_ms = (
                500
                if request.key_mode == 6
                and request.difficulty in {"NORMAL", "HARD"}
                and request.seed != 18
                else 250 + int(request.seed or 0)
            )
            return replace(
                generated,
                notes=[NoteEvent(time_ms, request.key_mode - 1)],
            )

    active_dir = tmp_path / "active"
    active_prepared = replace(
        _prepared(active_dir),
        difficulty_family_resolution_enabled=True,
    )
    active_authority = _authority(active_prepared, active_dir)
    active_generator = BoundedRecoveryGenerator()
    active_outcome = run_generation(
        active_prepared,
        active_authority,
        _analysis(),
        active_dir,
        generator=active_generator,
        seed=0,
    )

    assert len(active_generator.map_calls) == 13
    assert active_outcome.additional_inference_calls == 1
    assert all(
        len(
            {
                variant.raw_osu_path.read_bytes()
                for variant in active_outcome.variants
                if variant.key_mode == key_mode
            }
        )
        == 4
        for key_mode in KEY_MODES
    )
    selected_hard = next(
        variant
        for variant in active_outcome.variants
        if (variant.key_mode, variant.difficulty) == (6, "HARD")
    )
    assert any(
        evidence.get("reason") == "DIFFICULTY_BOUNDED_RECOVERY_ADMITTED"
        and evidence.get("mutatesSelection") is True
        for evidence in selected_hard.attempt_evidence
    )

    class UnresolvedDuplicateGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            time_ms = (
                500
                if request.key_mode == 6
                and request.difficulty in {"NORMAL", "HARD"}
                else 250 + int(request.seed or 0)
            )
            return replace(
                generated,
                notes=[NoteEvent(time_ms, request.key_mode - 1)],
            )

    unresolved_dir = tmp_path / "unresolved"
    unresolved_prepared = replace(
        _prepared(unresolved_dir),
        difficulty_family_resolution_enabled=True,
    )
    unresolved_generator = UnresolvedDuplicateGenerator()
    unresolved_outcome = run_generation(
        unresolved_prepared,
        _authority(unresolved_prepared, unresolved_dir),
        _analysis(),
        unresolved_dir,
        generator=unresolved_generator,
        seed=0,
    )

    assert len(unresolved_generator.map_calls) == 13
    assert len(unresolved_outcome.variants) == 12
    assert unresolved_outcome.missing == ()
    unresolved_6k = tuple(
        item for item in unresolved_outcome.variants if item.key_mode == 6
    )
    assert len(unresolved_6k) == 4
    assert len({item.raw_osu_path.read_bytes() for item in unresolved_6k}) == 4
    assert any(
        item.recovery_reason == "ATOMIC_DIFFICULTY_FAMILY_FALLBACK"
        for item in unresolved_6k
    )


def test_difficulty_shadow_target_is_planned_after_final_song_selection(
    monkeypatch,
    tmp_path: Path,
):
    prepared = replace(
        _prepared(tmp_path),
        difficulty_shadow_challenger_enabled=True,
    )
    authority = _authority(prepared, tmp_path)
    original_compare = s2_generate._compare_song_selection
    final_selection_ran = False
    planner_saw_final_selection = False

    def compare(*args, **kwargs):
        nonlocal final_selection_ran
        result = original_compare(*args, **kwargs)
        final_selection_ran = True
        return result

    def choose(_slots):
        if final_selection_ran:
            return DifficultyShadowTarget(4, "HARD", "EXPERT", 3.0, 1.0, 2.3)
        return DifficultyShadowTarget(4, "EASY", "NORMAL", 3.0, 1.0, 2.3)

    def plan(_target, candidates, _bpm_events, *, duration_ms):
        nonlocal planner_saw_final_selection
        assert final_selection_ran is True
        assert duration_ms == prepared.normalized.duration_ms
        planner_saw_final_selection = True
        return DifficultyShadowPartialDecision(
            reason="NO_NEAR_SOLUTION_CANDIDATE",
            plan=None,
            considered_candidate_count=len(candidates),
        )

    monkeypatch.setattr(s2_generate, "_compare_song_selection", compare)
    monkeypatch.setattr(s2_generate, "choose_difficulty_shadow_target", choose)
    monkeypatch.setattr(s2_generate, "plan_difficulty_shadow_partial_repair", plan)
    generator = RecordingGenerator()

    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert final_selection_ran is True
    assert planner_saw_final_selection is True
    assert outcome.additional_inference_calls == 1
    assert len(generator.map_calls) == 13


def test_difficulty_shadow_skips_issue_resolved_by_final_song_selection(
    monkeypatch,
    tmp_path: Path,
):
    prepared = replace(
        _prepared(tmp_path),
        difficulty_shadow_challenger_enabled=True,
    )
    authority = _authority(prepared, tmp_path)
    original_compare = s2_generate._compare_song_selection
    final_selection_ran = False

    def compare(*args, **kwargs):
        nonlocal final_selection_ran
        result = original_compare(*args, **kwargs)
        final_selection_ran = True
        return result

    def choose(_slots):
        if final_selection_ran:
            return None
        return SimpleNamespace(
            key_mode=4,
            easier_difficulty="EASY",
            difficulty="NORMAL",
            easier_rating=3.0,
            harder_rating=1.0,
            rating_deficit=2.0,
        )

    monkeypatch.setattr(s2_generate, "_compare_song_selection", compare)
    monkeypatch.setattr(s2_generate, "choose_difficulty_shadow_target", choose)
    generator = RecordingGenerator()

    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert final_selection_ran is True
    assert len(generator.map_calls) == 12
    assert outcome.additional_inference_calls == 0


def test_difficulty_shadow_records_budget_exhaustion_after_higher_priority_work(
    monkeypatch,
    tmp_path: Path,
):
    prepared = replace(
        _prepared(tmp_path),
        difficulty_shadow_challenger_enabled=True,
    )
    authority = _authority(prepared, tmp_path)
    created_states = {}
    original_state = s2_generate._VariantState
    original_plan_recoveries = s2_generate.plan_recoveries

    def make_state(key_mode, difficulty, flat_index, **kwargs):
        state = original_state(key_mode, difficulty, flat_index, **kwargs)
        created_states[(key_mode, difficulty)] = state
        return state

    def choose(_slots):
        return DifficultyShadowTarget(4, "HARD", "EXPERT", 3.0, 1.0, 2.3)

    def partial_plan(target, candidates, _bpm_events, *, duration_ms):
        assert duration_ms == prepared.normalized.duration_ms
        return DifficultyShadowPartialDecision(
            reason="PARTIAL_NEAR_SOLUTION_SELECTED",
            plan=DifficultyShadowPartialPlan(
                target=target,
                source=candidates[0],
                window=PartialRemapWindow(start_ms=0, end_ms=1_000),
                required_gameplay_interval=_observed_required_interval(),
            ),
            considered_candidate_count=len(candidates),
        )

    def route(requests, *, available_generation_ms, available_calls):
        if any(request.kind is RecoveryKind.DIFFICULTY_SHADOW for request in requests):
            available_calls = 0
        return original_plan_recoveries(
            requests,
            available_generation_ms=available_generation_ms,
            available_calls=available_calls,
        )

    monkeypatch.setattr(s2_generate, "_VariantState", make_state)
    monkeypatch.setattr(s2_generate, "choose_difficulty_shadow_target", choose)
    monkeypatch.setattr(
        s2_generate,
        "plan_difficulty_shadow_partial_repair",
        partial_plan,
    )
    monkeypatch.setattr(s2_generate, "plan_recoveries", route)
    generator = RecordingGenerator()

    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert outcome.additional_inference_calls == 0
    assert len(generator.map_calls) == 12
    assert any(
        evidence.get("reason") == "BUDGET_EXHAUSTED_AFTER_HIGHER_PRIORITY_RECOVERY"
        for evidence in created_states[(4, "EXPERT")].attempt_evidence
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
    # No audio activity contract exists in this synthetic fixture.  Do not
    # replace a hard-safe fallback with an uncorroborated sibling payload.
    assert fallback.provenance == "SAFE_FALLBACK"
    assert fallback.family_assignment_kind == "ORIGINAL"
    assert fallback.source_difficulty == "EASY"
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
    assert {request.timing_reference_path for request in requests} == {authority.reference_path}
    assert all(variant.timing_authority_sha256 == authority.sha256 for variant in variants)
    assert len({request.seed for request in requests}) == 12
    assert [(request.key_mode, request.difficulty, request.seed) for request in requests] == [
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
        request.difficulty: request.descriptors for request in requests if request.key_mode == 4
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
        for line in (tmp_path / "attempt-journal.jsonl").read_text(encoding="utf-8").splitlines()
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


def test_generation_uses_selector_mode_from_prepared_run_context(monkeypatch, tmp_path: Path):
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
            first_row_ms = 501 if (request.key_mode, request.difficulty) == (7, "EXPERT") else 500
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
        min(note.time_ms for note in variant.generated.notes) for variant in outcome.variants
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
                call for call in self.map_calls if (call.key_mode, call.difficulty) == (7, "HARD")
            ]
            if (request.key_mode, request.difficulty) == (7, "HARD") and len(matching_calls) == 2:
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
    assert [(request.key_mode, request.difficulty) for request in generator.map_calls].count(
        (4, "EASY")
    ) == 1
    assert all(request.add_to_beatmap is False for request in generator.map_calls)
    assert outcome.additional_inference_calls == 0
    assert outcome.intro_contract_review is not None
    # The supplied anchor omits the corroborating pulse sequence, so the
    # adaptive region contract must remain fail-closed.  We report the exact
    # mismatch but do not rewrite otherwise playable rows or spend inference.
    assert outcome.intro_contract_review.status == "REVIEW"
    assert outcome.intro_contract_review.corrected_count == 0
    assert {
        min(note.time_ms for note in variant.generated.notes) for variant in outcome.variants
    } == {0, 500}
    assert all(not decision.changed for decision in outcome.safe_family_assignments)
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
        bad = requested_key_mode == 4 and requested_difficulty == "EXPERT" and generated.seed == 3
        overall = 0.62 if bad else 0.73
        section_values = (
            ((70, 0.80), (105, 0.42), (110, 0.40)) if bad else ((68, 0.80), (68, 0.72), (67, 0.74))
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
        lambda generated, *args, requested_key_mode, requested_difficulty, **kwargs: acceptance(
            requested_key_mode, requested_difficulty, generated
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
    assert router_evidence["plan"]["selectedRequestIds"] == ["timing:4k:EXPERT"]
    journal = [
        json.loads(line)
        for line in (tmp_path / "attempt-journal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    recovery_events = [
        entry for entry in journal if entry["payload"].get("purpose") == "TIMING_FAMILY_RETRY"
    ]
    assert [entry["eventType"] for entry in recovery_events] == [
        "INFERENCE_STARTED",
        "INFERENCE_COMPLETED",
        "GATE_EVALUATED",
        "CANDIDATE_ADMITTED",
    ]
    expert_review = next(
        review for review in outcome.timing_family_reviews if review.difficulty == "EXPERT"
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
                call for call in self.map_calls if (call.key_mode, call.difficulty) == (4, "EXPERT")
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
        _active_analysis(20_000, onset_step_ms=250),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 13
    assert outcome.additional_inference_calls == 1
    intro_request = generator.map_calls[-1]
    assert (intro_request.key_mode, intro_request.difficulty) == (4, "EXPERT")
    assert intro_request.add_to_beatmap is True
    assert intro_request.partial_start_ms == 0
    assert intro_request.partial_end_ms == 14_000
    assert intro_request.required_gameplay_interval is not None
    assert (
        intro_request.required_gameplay_interval.start_ms,
        intro_request.required_gameplay_interval.end_ms,
    ) == (180, 4_320)
    assert (
        intro_request.required_gameplay_interval.evidence_class
        is RequiredGameplayEvidenceClass.INTRO_REGION_CORROBORATED
    )
    assert outcome.additional_inference_work_ms == 14_000
    recovered = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EXPERT")
    )
    assert recovered.selected_seed == 15
    assert recovered.provenance == "INTRO_RECOVERY"
    assert recovered.recovery_reason == "INTRO_PHRASE_FAMILY_DEFECT"
    router_evidence = next(
        evidence
        for evidence in recovered.attempt_evidence
        if evidence["reason"] == "RECOVERY_ROUTER_DECISION"
    )
    assert router_evidence["plan"]["selectedRequestIds"] == ["intro:4k:EXPERT"]
    review = next(
        review for review in outcome.intro_phrase_family_reviews if review.hard.key_mode == 4
    )
    assert review.status == "PASS"
    assert review.reason == "CONSISTENT"
    assert any(
        evidence["reason"] == "INTRO_PHRASE_RETRY_SELECTED"
        for evidence in recovered.attempt_evidence
    )


@pytest.mark.parametrize(
    ("target_key_mode", "target_difficulty", "expected_seed"),
    [
        (4, "EASY", 12),
        (4, "NORMAL", 13),
        (6, "HARD", 18),
        (7, "EXPERT", 23),
    ],
)
def test_confirmed_intro_region_recovers_any_slot_with_one_bounded_retry(
    tmp_path: Path,
    target_key_mode: int,
    target_difficulty: str,
    expected_seed: int,
):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    authority = _authority(prepared, tmp_path)

    class RecoveringSlotGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            self.map_calls.append(request)
            self.map_workdirs.append(workdir)
            matching = [
                call
                for call in self.map_calls
                if (call.key_mode, call.difficulty)
                == (target_key_mode, target_difficulty)
            ]
            if (request.key_mode, request.difficulty) == (
                target_key_mode,
                target_difficulty,
            ):
                rows = (12_000, 12_250) if len(matching) == 1 else (250, 500)
            else:
                rows = (250, 500)
            return GeneratedChart(
                notes=[
                    NoteEvent(row, index % request.key_mode)
                    for index, row in enumerate(rows)
                ],
                key_mode=request.key_mode,
                osu_text="",
                generator_name="intro-region-recovery-fixture",
                seed=request.seed,
                bpm_events=authority.bpm_events,
            )

    generator = RecoveringSlotGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _active_analysis(20_000, onset_step_ms=250),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 13
    assert outcome.additional_inference_calls == 1
    intro_request = generator.map_calls[-1]
    assert (intro_request.key_mode, intro_request.difficulty) == (
        target_key_mode,
        target_difficulty,
    )
    assert intro_request.add_to_beatmap is True
    assert intro_request.partial_start_ms == 0
    assert intro_request.partial_end_ms == 6_320
    assert intro_request.required_gameplay_interval is not None
    assert (
        intro_request.required_gameplay_interval.evidence_class
        is RequiredGameplayEvidenceClass.INTRO_REGION_CORROBORATED
    )
    recovered = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty)
        == (target_key_mode, target_difficulty)
    )
    assert recovered.selected_seed == expected_seed
    assert recovered.provenance == "INTRO_RECOVERY"
    assert recovered.recovery_reason == "INTRO_REGION_DEFECT"
    router_evidence = next(
        evidence
        for evidence in recovered.attempt_evidence
        if evidence["reason"] == "RECOVERY_ROUTER_DECISION"
    )
    assert router_evidence["plan"]["selectedRequestIds"] == [
        f"intro:{target_key_mode}k:{target_difficulty}"
    ]
    assert any(
        evidence["reason"] == "INTRO_REGION_RETRY_SELECTED"
        for evidence in recovered.attempt_evidence
    )


def test_unaddressed_intro_tempo_preserves_all_slots_without_inventing_bpm(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    authority = replace(
        _authority(
            prepared,
            tmp_path,
            (OsuBpmEvent(6_937, 87.0),),
        ),
        leading_coverage=LeadingTimingCoverage(
            action=TimingAuthorityAction.REVIEW,
            reasons=("CONFIRMED_INTRO_ANCHOR_BEFORE_FIRST_EVENT",),
            first_event_time_ms=6_937,
            leading_duration_ms=6_937,
            onset_count=27,
            active_onset_count=27,
            active_frame_ratio=1.0,
            intro_anchor=IntroAnchorEvidence(
                status="CONFIRMED",
                anchor_ms=250,
                anchor_grid_ms=250,
                grid_distance_ms=0,
                aggregate_percentile_rank=0.99,
                prominent_band_count=3,
                pulse_continuation_matches=16,
                pulse_continuation_opportunities=16,
                supported_pulse_ms=tuple(range(250, 4_251, 250)),
            ),
        ),
    )

    class LateNormalGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            self.map_calls.append(request)
            self.map_workdirs.append(workdir)
            rows = (
                (12_000, 12_250)
                if (request.key_mode, request.difficulty) == (4, "NORMAL")
                else (250, 500)
            )
            return GeneratedChart(
                notes=[
                    NoteEvent(row, index % request.key_mode)
                    for index, row in enumerate(rows)
                ],
                key_mode=request.key_mode,
                osu_text="",
                generator_name="post-zero-tempo-intro-fixture",
                seed=request.seed,
                bpm_events=authority.bpm_events,
            )

    generator = LateNormalGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _active_analysis(20_000, onset_step_ms=250),
        tmp_path,
        generator=generator,
        seed=0,
    )

    intro_retry_workdirs = tuple(
        workdir
        for request, workdir in zip(
            generator.map_calls,
            generator.map_workdirs,
            strict=True,
        )
        if request.add_to_beatmap and "intro-" in str(workdir).lower()
    )
    assert intro_retry_workdirs == ()
    assert len(outcome.variants) == 12
    assert outcome.missing == ()
    selected = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "NORMAL")
    )
    unavailable = next(
        evidence
        for evidence in selected.attempt_evidence
        if evidence.get("reason")
        == "INTRO_REGION_RECOVERY_SKIPPED_UNADDRESSED_TEMPO"
    )
    assert unavailable == {
        "reason": "INTRO_REGION_RECOVERY_SKIPPED_UNADDRESSED_TEMPO",
        "introRegionEndMs": 4_320,
        "firstTimingEventMs": 6_937,
        "modelRetryAttempted": False,
        "preservedSelectedCandidate": True,
    }
    assert any(
        evidence.get("reason") == "INTRO_REGION_DEFECT_PLAYTEST_ONLY"
        for evidence in selected.attempt_evidence
    )


def test_unconfirmed_intro_region_does_not_retry_late_normal(tmp_path: Path):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    authority = _authority(prepared, tmp_path)

    class LateNormalGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            self.map_calls.append(request)
            self.map_workdirs.append(workdir)
            rows = (
                (12_000, 12_250)
                if (request.key_mode, request.difficulty) == (4, "NORMAL")
                else (250, 500)
            )
            return GeneratedChart(
                notes=[
                    NoteEvent(row, index % request.key_mode)
                    for index, row in enumerate(rows)
                ],
                key_mode=request.key_mode,
                osu_text="",
                generator_name="unknown-intro-region-fixture",
                seed=request.seed,
                bpm_events=authority.bpm_events,
            )

    generator = LateNormalGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 12
    assert outcome.additional_inference_calls == 0


def test_failed_confirmed_intro_region_retry_keeps_all_slots_for_playtest(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    authority = _authority(prepared, tmp_path)

    class PersistentlyLateNormalGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            self.map_calls.append(request)
            self.map_workdirs.append(workdir)
            rows = (
                (12_000, 12_250)
                if (request.key_mode, request.difficulty) == (4, "NORMAL")
                else (250, 500)
            )
            return GeneratedChart(
                notes=[
                    NoteEvent(row, index % request.key_mode)
                    for index, row in enumerate(rows)
                ],
                key_mode=request.key_mode,
                osu_text="",
                generator_name="persistent-intro-region-defect-fixture",
                seed=request.seed,
                bpm_events=authority.bpm_events,
            )

    generator = PersistentlyLateNormalGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _active_analysis(20_000, onset_step_ms=250),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(generator.map_calls) == 13
    assert outcome.additional_inference_calls == 1
    assert len(outcome.variants) == 12
    assert outcome.missing == ()
    selected = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "NORMAL")
    )
    assert any(
        evidence.get("reason") == "INTRO_REGION_RETRY_NOT_IMPROVED"
        for evidence in selected.attempt_evidence
    )
    assert any(
        evidence.get("reason") == "INTRO_REGION_DEFECT_PLAYTEST_ONLY"
        for evidence in selected.attempt_evidence
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
        and "NEW_REVIEW_AXIS:PATTERN" in evidence["decision"]["reasons"]
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
                (0, 12_000) if (request.key_mode, request.difficulty) == (4, "EXPERT") else (0, 250)
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

    # Structural phrase evidence still blocks production publication, but
    # absent audio corroboration must not authorize another model call.
    assert len(generator.map_calls) == 12
    assert outcome.additional_inference_calls == 0
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
            soft_retry if generated.key_mode == 4 and requested_difficulty == "EXPERT" else accepted
        )
        return _acceptance_for_difficulty(source, requested_difficulty)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)

    class RawPhraseDefectGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            self.map_calls.append(request)
            self.map_workdirs.append(workdir)
            rows = (
                (0, 12_000) if (request.key_mode, request.difficulty) == (4, "EXPERT") else (0, 250)
            )
            return GeneratedChart(
                notes=[NoteEvent(row, index % request.key_mode) for index, row in enumerate(rows)],
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
        review for review in outcome.intro_phrase_family_reviews if review.hard.key_mode == 4
    )
    assert review.status == "PASS"
    assert any(
        evidence["reason"] == "INTRO_PHRASE_EXISTING_CANDIDATE_RESELECTED"
        for evidence in expert.attempt_evidence
    )


def test_intro_contract_retry_has_an_independent_one_call_budget(tmp_path: Path):
    prepared = _prepared(tmp_path, duration_ms=20_000)
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
    assert generator.map_calls[0].add_to_beatmap is True
    assert generator.map_calls[0].partial_start_ms == 0
    assert generator.map_calls[0].partial_end_ms == 2_250
    journal = [
        json.loads(line)
        for line in (tmp_path / "attempt-journal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["eventType"] for entry in journal] == [
        "INFERENCE_STARTED",
        "INFERENCE_COMPLETED",
        "GATE_EVALUATED",
        "CANDIDATE_ADMITTED",
    ]
    assert {entry["payload"]["purpose"] for entry in journal} == {"INTRO_CONTRACT_RETRY"}


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


def test_structural_retry_remains_but_uncalibrated_order_retry_is_deferred(
    monkeypatch, tmp_path: Path
):
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
            {"EASY": 1.0, "NORMAL": 2.0, "HARD": 3.0, "EXPERT": 4.0}[requested_difficulty],
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
    ]
    assert 31 not in [request.seed for request in generator.map_calls]
    selected_6k = {variant.difficulty: variant for variant in variants if variant.key_mode == 6}
    assert selected_6k["HARD"].selected_seed == 6
    assert selected_6k["EXPERT"].selected_seed == 19
    assert selected_6k["HARD"].difficulty_order is not None
    assert selected_6k["HARD"].difficulty_order.status == "RETRY"


def test_difficulty_inversion_is_reported_without_uncalibrated_seed_retry(
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
            {"EASY": 1.0, "NORMAL": 2.0, "HARD": 3.0, "EXPERT": 4.0}[requested_difficulty],
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
        (request.key_mode, request.difficulty, request.seed) for request in generator.map_calls[:4]
    ] == [
        (4, "EASY", 0),
        (4, "NORMAL", 1),
        (4, "HARD", 2),
        (4, "EXPERT", 3),
    ]
    selected_4k = {variant.difficulty: variant for variant in variants if variant.key_mode == 4}
    assert selected_4k["HARD"].selected_seed == 2
    assert selected_4k["HARD"].attempt == 1
    assert selected_4k["HARD"].candidate_count == 1
    assert selected_4k["HARD"].generation_attempt_count == 1
    assert selected_4k["EXPERT"].selected_seed == 3
    assert selected_4k["EXPERT"].attempt == 1
    assert selected_4k["EASY"].candidate_count == 1
    assert selected_4k["NORMAL"].candidate_count == 1
    assert all(
        variant.difficulty_order is not None and variant.difficulty_order.status == "RETRY"
        for variant in selected_4k.values()
    )


def test_equal_difficulty_profiles_publish_without_more_seeds(monkeypatch, tmp_path: Path):
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


def test_ambiguity_and_inversion_are_reported_without_extra_seed(monkeypatch, tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        if generated.seed in {2, 3}:
            rating = {"HARD": 4.0, "EXPERT": 3.0}[requested_difficulty]
        else:
            rating = {"EASY": 2.0, "NORMAL": 2.0, "HARD": 3.0, "EXPERT": 4.0}[requested_difficulty]
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
    assert [request.seed for request in generator.map_calls] == list(range(12))
    assert variants[0].difficulty_order is not None
    assert variants[0].difficulty_order.ambiguous_pairs == (("EASY", "NORMAL"),)
    assert variants[0].difficulty_order.inverted_pairs == (("HARD", "EXPERT"),)


def test_exhausted_hard_still_tries_available_expert_candidate(monkeypatch, tmp_path: Path):
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
            {"EASY": 1.0, "NORMAL": 2.0, "HARD": 3.0, "EXPERT": 4.0}[requested_difficulty],
        )
        return _acceptance_with_rating(accepted, rating)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)

    class HardNeedsThirdAttempt(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if request.key_mode == 4 and request.difficulty == "HARD" and request.seed in {2, 14}:
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

    selected_4k = {variant.difficulty: variant for variant in variants if variant.key_mode == 4}
    assert selected_4k["HARD"].selected_seed == 26
    assert selected_4k["EXPERT"].selected_seed == 3
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
    ]


def test_uncalibrated_inversion_keeps_all_labels_without_ordinary_full_map_retry(
    monkeypatch, tmp_path: Path
):
    """역전이 끝내 안 풀려도 플레이테스트 채보는 삭제하지 않는다."""
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)

    def evaluate(*args, requested_difficulty, **kwargs):
        del args, kwargs
        rating = {"EASY": 1.0, "NORMAL": 2.0, "HARD": 4.0, "EXPERT": 3.0}[requested_difficulty]
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

    calls_4k = [
        (request.difficulty, request.seed)
        for request in generator.map_calls
        if request.key_mode == 4
    ]
    assert calls_4k == [
        ("EASY", 0),
        ("NORMAL", 1),
        ("HARD", 2),
        ("EXPERT", 3),
    ]
    assert outcome.missing == ()
    published_4k = {variant.difficulty for variant in outcome.variants if variant.key_mode == 4}
    assert published_4k == {"EASY", "NORMAL", "HARD", "EXPERT"}
    assert (tmp_path / "raw" / "4k-hard.osu").is_file()
    assert (tmp_path / "raw" / "4k-expert.osu").is_file()
    assert all(
        variant.difficulty_order is not None and variant.difficulty_order.status == "RETRY"
        for variant in outcome.variants
        if variant.key_mode == 4
    )
    assert all(variant.provenance == "PRIMARY" for variant in outcome.variants)
    assert all(
        any(
            evidence.get("reason") == "DIFFICULTY_ORDER_UNRESOLVED_NO_CALIBRATED_RETRY"
            for evidence in variant.attempt_evidence
        )
        for variant in outcome.variants
        if variant.key_mode == 4
    )


def test_inverted_pair_keeps_existing_candidates_without_uncalibrated_retry(
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
            {"EASY": 1.0, "NORMAL": 2.0, "HARD": 3.0, "EXPERT": 4.0}[requested_difficulty],
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
    assert [variant.selected_seed for variant in variants[:4]] == [0, 1, 2, 3]
    assert variants[0].difficulty_order is not None
    assert variants[0].difficulty_order.status == "RETRY"
    assert variants[0].difficulty_order is not None
    assert variants[0].difficulty_order.ambiguous_pairs == ()
    assert [request.seed for request in generator.map_calls] == list(range(12))
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


def test_partial_stable_promotion_is_cleaned_and_normalized(monkeypatch, tmp_path: Path):
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


def test_failed_key_mode_does_not_discard_the_other_key_modes(monkeypatch, tmp_path: Path):
    """6K 전체가 구조 실패해도 4K·7K 는 그대로 발행한다."""
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    def reject_6k(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        action = (
            GateAction.RETRY_MAP
            if generated.key_mode == 6 and generated.generator_name != "adaptive-recovery-v1"
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
    attempts = [item for item in fallback.attempt_evidence if "gateReport" in item]
    assert [attempt["seed"] for attempt in attempts] == [0, 12, 24]
    assert [attempt["workdir"] for attempt in attempts] == [
        "raw/work/epoch-1/4k-easy/attempt-1",
        "raw/work/epoch-1/4k-easy/attempt-2",
        "raw/work/epoch-1/4k-easy/attempt-3",
    ]
    assert all(
        attempt["gateReport"]["decisions"]["STRUCTURE"]["reasons"] == ["STRUCTURE_INVALID"]
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
            if request.key_mode == 4 and request.difficulty == "EASY" and request.seed == 0:
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
        call for call in generator.map_calls if call.key_mode == 4 and call.difficulty == "EASY"
    ]
    other_calls = [
        call
        for call in generator.map_calls
        if not (call.key_mode == 4 and call.difficulty == "EASY")
    ]
    assert [call.seed for call in easy_calls] == [0, 12]
    assert len(other_calls) == 11
    assert len(variants) == 12
    assert (
        next(
            variant
            for variant in variants
            if variant.key_mode == 4 and variant.difficulty == "EASY"
        ).selected_seed
        == 12
    )


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
        call.seed for call in generator.map_calls if (call.key_mode, call.difficulty) == (4, "EASY")
    ] == [0, 12, 24]
    fallback = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert fallback.provenance == "SAFE_FALLBACK"


def test_single_slot_timing_feedback_does_not_force_a_second_full_map(monkeypatch, tmp_path: Path):
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

    variants = _run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert calls >= 12  # CPU repair/fallback candidates are re-evaluated too.
    assert [call.seed for call in generator.map_calls if not call.add_to_beatmap] == list(range(12))
    assert len(variants) == 12
    assert all(
        any(
            evidence.get("reason") == "QUALITY_DEFECT_ROUTED_TO_RECOVERY"
            for evidence in variant.attempt_evidence
        )
        for variant in variants
    )


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
        for request, workdir in zip(generator.map_calls, generator.map_workdirs, strict=True)
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


def test_retry_map_uses_next_seed_and_promotes_only_the_pass_candidate(monkeypatch, tmp_path: Path):
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
            "reason": "HARD_REJECT_RETRY",
            "candidateDisposition": "HARD_REJECT",
            "repairEligible": False,
            "repairAxes": [],
        },
    )


def test_review_candidate_is_published_without_consuming_another_seed(monkeypatch, tmp_path: Path):
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
        tmp_path / "raw" / "work" / "epoch-1" / "4k-easy" / "attempt-1" / "candidate.osu"
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
    assert [request.seed for request in generator.map_calls] == list(range(failure_index + 1))
    assert len(outcome.variants) == 12
    assert outcome.missing == ()
    by_slot = {(variant.key_mode, variant.difficulty): variant for variant in outcome.variants}
    assert all(by_slot[slot].provenance == "PRIMARY" for slot in slots[:failure_index])
    assert all(
        by_slot[slot].provenance in {"PRIMARY", "SAFE_FALLBACK"}
        for slot in slots[failure_index:]
    )
    assert all(
        variant.family_assignment_kind != "ORIGINAL"
        for slot, variant in by_slot.items()
        if slot in slots[failure_index:] and variant.provenance == "PRIMARY"
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
            evidence["reason"] == "MODEL_INFERENCE_SUPPRESSED_AFTER_UNKNOWN_COMPLETION"
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
        flat_index + attempt * 12 for flat_index in range(12) for attempt in range(3)
    ]
    assert len(outcome.variants) == 12
    assert all(variant.provenance == "SAFE_FALLBACK" for variant in outcome.variants)
    first = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    gate_attempts = [item for item in first.attempt_evidence if "gateReport" in item]
    assert gate_attempts == [
        {
            "seed": attempt_seed,
            "workdir": f"raw/work/epoch-1/4k-easy/attempt-{attempt}",
            "gateReport": retry.to_report(),
            "reason": "HARD_REJECT_RETRY",
            "candidateDisposition": "HARD_REJECT",
            "repairEligible": False,
            "repairAxes": [],
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
    assert fallback.family_assignment_kind == "ORIGINAL"
    assert fallback.source_difficulty == "EASY"
    assert fallback.acceptance.decision(GateAxis.STRUCTURE).action is GateAction.PASS
    assert fallback.acceptance.decision(GateAxis.TIMING_IDENTITY).action is GateAction.PASS
    assert fallback.acceptance.decision(GateAxis.SONG_BOUNDS).action is GateAction.PASS
    # 크래시 예산 3회만 쓰고 나머지 11 조합은 한 번씩만 부른다.
    assert len(generator.map_calls) == 14


def test_song_bounds_rejected_model_output_is_not_promoted_as_raw(monkeypatch, tmp_path: Path):
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
    assert fallback.family_assignment_kind == "ORIGINAL"
    assert fallback.source_difficulty == "EASY"
    assert fallback.acceptance.decision(GateAxis.SONG_BOUNDS).action is GateAction.PASS
    assert outcome.diagnostic_raw_candidates == ()
    assert len(generator.map_calls) == 14
    assert any(
        evidence.get("reason") == "HARD_REJECT_RETRY"
        and evidence.get("candidateDisposition") == "HARD_REJECT"
        for evidence in fallback.attempt_evidence
    )


def test_localized_coverage_failure_remaps_only_the_exhausted_variant(monkeypatch, tmp_path: Path):
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
        variant for variant in variants if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    partial_requests = [request for request in generator.map_calls if request.add_to_beatmap]
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
        for line in (tmp_path / "attempt-journal.jsonl").read_text(encoding="utf-8").splitlines()
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
    assert all(variant.provenance == "PRIMARY" for variant in variants if variant is not repaired)


def test_song_recovery_budget_admits_only_one_small_partial_repair(monkeypatch, tmp_path: Path):
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

    partial_requests = [request for request in generator.map_calls if request.add_to_beatmap]
    assert len(generator.map_calls) == 13
    assert [(request.key_mode, request.difficulty) for request in partial_requests] == [
        (4, "EASY"),
    ]
    assert outcome.additional_inference_calls == 1
    assert outcome.additional_inference_work_ms == 14_000
    assert outcome.additional_inference_work_limit_ms == 60_000
    repaired = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    deferred = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (6, "EASY")
    )
    assert repaired.provenance == "PARTIAL_REMAP"
    assert deferred.provenance != "PARTIAL_REMAP"
    assert any(
        evidence.get("reason") == "RECOVERY_ROUTER_DECISION"
        and "partial:6k:EASY" in evidence["plan"]["deferredRequestIds"]
        for evidence in deferred.attempt_evidence
    )


def test_partial_tail_exhaustion_blocks_only_that_variant(monkeypatch, tmp_path: Path):
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
    selected = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert selected.provenance == "SAFE_FALLBACK"
    assert selected.family_assignment_kind == "ORIGINAL"
    assert selected.source_difficulty == "EASY"
    assert json.loads(selected.attempt_errors[-1]) == {
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
    } in selected.attempt_evidence
    assert exhausted_context["totalAttempts"] == 2
    assert exhausted_context["seeds"] == [0, 36]
    assert exhausted_context["partialAttempts"] == [2]
    assert exhausted_context["partialSeeds"] == [36]
    assert exhausted_context["qualityAttempts"] == 1
    assert exhausted_context["crashAttempts"] == 0
    journal = [
        json.loads(line)
        for line in (tmp_path / "attempt-journal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    partial_events = [
        entry for entry in journal if entry["payload"].get("purpose") == "PARTIAL_REMAP"
    ]
    assert [entry["eventType"] for entry in partial_events] == [
        "INFERENCE_STARTED",
        "INFERENCE_FAILED",
    ]
    assert partial_events[-1]["payload"]["error"]["context"] == tail_context


def test_audio_supported_intro_phase_difference_within_opening_window_is_not_replaced(
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
            covered = not (request.key_mode == 4 and request.difficulty == "EASY")
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
        variant for variant in variants if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    repair_requests = [request for request in generator.map_calls if request.add_to_beatmap]
    assert len(generator.map_calls) == 12
    assert repair_requests == []
    assert repaired.provenance == "PRIMARY"
    assert repaired.recovery_reason is None
    assert repaired.generated.notes == [NoteEvent(500, 0)]
    assert repaired.family_assignment_kind == "ORIGINAL"
    assert repaired.source_difficulty == "EASY"
    assert repaired.generation_attempt_count == 1
    assert all(variant.provenance == "PRIMARY" for variant in variants if variant is not repaired)


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


def test_coverage_repair_outranks_gapped_raw_without_another_full_model_call(
    monkeypatch, tmp_path: Path
):
    duration_ms = 40_000
    prepared = _prepared(tmp_path, duration_ms=duration_ms)
    authority = _authority(prepared, tmp_path)
    analysis = _active_analysis(duration_ms, onset_step_ms=1_000)
    accepted = _pass_acceptance(authority)

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        if generated.key_mode == 4 and requested_difficulty == "EASY":
            return evaluate_chart_candidate(
                generated,
                authority,
                analysis,
                requested_key_mode=4,
                requested_difficulty="EASY",
                duration_ms=duration_ms,
            )
        return _acceptance_for_difficulty(accepted, requested_difficulty)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)

    class OneBoundedGapGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if (request.key_mode, request.difficulty) != (4, "EASY"):
                return generated
            active_rows = analysis.activity.active_onset_ms  # type: ignore[union-attr]
            gap_start_ms = 12_000
            gap_end_ms = 20_000
            rows = sorted(
                {
                    gap_start_ms,
                    gap_end_ms,
                    *(row for row in active_rows if not gap_start_ms < row < gap_end_ms),
                }
            )
            return replace(
                generated,
                notes=[
                    NoteEvent(time_ms=row, lane=index % request.key_mode)
                    for index, row in enumerate(rows)
                ],
            )

    generator = OneBoundedGapGenerator()

    outcome = run_generation(
        prepared,
        authority,
        analysis,
        tmp_path,
        generator=generator,
        seed=0,
    )

    selected = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert selected.provenance == "COVERAGE_REPAIR"
    assert outcome.song_selection_shadow is not None
    assert outcome.song_selection_evidence_v3 is not None
    assert outcome.song_selection_shadow_v3 is not None
    assert outcome.song_selection_shadow_v3.mode == "SHADOW_V3"
    assert outcome.song_selection_shadow_v3.mutates_selection is False
    assert outcome.song_selection_shadow_v3.blockers == ("CALIBRATION_UNAVAILABLE",)
    v3_report = outcome.song_selection_evidence_v3.to_report()
    assert v3_report["mutatesSelection"] is False
    assert v3_report["additionalModelCalls"] == 0
    assert len(v3_report["candidates"]) >= 12
    assert {candidate["candidateId"] for candidate in v3_report["candidates"]} == {
        candidate.candidate_id
        for candidate in outcome.song_selection_shadow.replay_input.candidates
    }
    assert any(candidate["safety"]["activeGaps"] for candidate in v3_report["candidates"])
    selected_v3 = next(
        candidate
        for candidate in v3_report["candidates"]
        if candidate["candidateId"] == v3_report["currentAssignment"]["4K:EASY"]
    )
    assert selected_v3["safety"]["publicationTier"] == "PLAYTEST_ONLY"
    for snapshot in outcome.song_selection_shadow.replay_input.candidates:
        payload_path = tmp_path / snapshot.candidate_payload_ref
        assert payload_path.is_file()
        assert sha256_file(payload_path) == snapshot.candidate_payload_sha256
    assert selected.coverage_repair_gap_count > 0
    assert selected.acceptance.decision(GateAxis.COVERAGE).action is not GateAction.RETRY_MAP
    assert len(generator.map_calls) == 12
    assert all(not request.add_to_beatmap for request in generator.map_calls)
    assert any(
        evidence.get("reason") == "QUALITY_DEFECT_ROUTED_TO_RECOVERY"
        and evidence.get("candidateDisposition") == "QUALITY_DEFECT"
        and evidence.get("repairEligible") is True
        for evidence in selected.attempt_evidence
    )


def test_zero_inference_coverage_repair_precedes_a_viable_partial_model_call(
    monkeypatch, tmp_path: Path
):
    """A validated local repair must avoid the slower partial decoder path."""

    duration_ms = 60_000
    prepared = _prepared(tmp_path, duration_ms=duration_ms)
    authority = _authority(prepared, tmp_path)
    analysis = _active_analysis(duration_ms, onset_step_ms=1_000)
    accepted = _pass_acceptance(authority)
    base_coverage_retry = _coverage_retry_acceptance(authority)
    coverage_retry = replace(
        base_coverage_retry,
        timing=replace(
            base_coverage_retry.timing,
            coverage_gaps=(
                replace(
                    base_coverage_retry.timing.coverage_gaps[0],
                    start_ms=12_000,
                    end_ms=20_000,
                ),
            ),
        ),
    )

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        if generated.generator_name.startswith("coverage-repair-v1:"):
            return _acceptance_for_difficulty(accepted, requested_difficulty)
        if requested_difficulty == "EASY" and generated.key_mode == 4:
            return _acceptance_for_difficulty(coverage_retry, requested_difficulty)
        return _acceptance_for_difficulty(accepted, requested_difficulty)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)

    class PartialMustNotRun(RecordingGenerator):
        def generate_map(self, request, workdir):
            if request.add_to_beatmap:
                raise AssertionError("validated CPU coverage repair must run first")
            generated = super().generate_map(request, workdir)
            if (request.key_mode, request.difficulty) != (4, "EASY"):
                return generated
            active_rows = analysis.activity.active_onset_ms  # type: ignore[union-attr]
            gap_start_ms = 12_000
            gap_end_ms = 20_000
            rows = sorted(
                {
                    gap_start_ms,
                    gap_end_ms,
                    *(row for row in active_rows if not gap_start_ms < row < gap_end_ms),
                }
            )
            return replace(
                generated,
                notes=[
                    NoteEvent(time_ms=row, lane=index % request.key_mode)
                    for index, row in enumerate(rows)
                ],
            )

    generator = PartialMustNotRun()
    outcome = run_generation(
        prepared,
        authority,
        analysis,
        tmp_path,
        generator=generator,
        seed=0,
    )

    selected = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert len(generator.map_calls) == 12
    assert outcome.additional_inference_calls == 0
    assert selected.provenance == "COVERAGE_REPAIR"
    assert selected.coverage_repair_gap_count == 1


def test_still_rejected_coverage_repair_does_not_suppress_partial_model_call(
    monkeypatch, tmp_path: Path
):
    duration_ms = 60_000
    prepared = _prepared(tmp_path, duration_ms=duration_ms)
    authority = _authority(prepared, tmp_path)
    analysis = _active_analysis(duration_ms)
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
        analysis,
        tmp_path,
        generator=generator,
        seed=0,
    )

    selected = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert len(generator.map_calls) == 13
    assert outcome.additional_inference_calls == 1
    assert selected.provenance == "PARTIAL_REMAP"
    missing_cpu_repair = next(
        evidence
        for evidence in selected.attempt_evidence
        if evidence.get("reason") == "COVERAGE_REPAIR_REJECTED"
    )
    assert "inserted-note budget" in str(missing_cpu_repair["message"])


def test_rejected_partial_remap_prefers_coverage_safe_fallback_over_gapped_raw(
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
    selected = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert len(outcome.variants) == 12
    assert outcome.missing == ()
    assert selected.provenance == "SAFE_FALLBACK"
    assert selected.family_assignment_kind == "ORIGINAL"
    assert selected.source_difficulty == "EASY"
    assert selected.acceptance.action is not GateAction.RETRY_MAP
    assert selected.acceptance.decision(GateAxis.STRUCTURE).action is GateAction.PASS
    assert selected.acceptance.decision(GateAxis.TIMING_IDENTITY).action is GateAction.PASS
    assert selected.acceptance.decision(GateAxis.SONG_BOUNDS).action is GateAction.PASS
    assert outcome.diagnostic_raw_candidates == ()
    assert outcome.song_selection_shadow is not None
    fallback_snapshot = next(
        snapshot
        for snapshot in outcome.song_selection_shadow.replay_input.candidates
        if snapshot.key_mode == 4
        and snapshot.difficulty == "EASY"
        and snapshot.provenance == "SAFE_FALLBACK"
    )
    fallback_payload = tmp_path / fallback_snapshot.candidate_payload_ref
    assert fallback_payload.is_file()
    assert sha256_file(fallback_payload) == fallback_snapshot.candidate_payload_sha256


def test_soft_coverage_review_model_repair_outranks_pass_safe_fallback(monkeypatch, tmp_path: Path):
    """A soft coverage label must not discard the model's rhythm and difficulty."""

    prepared = _prepared(tmp_path, duration_ms=60_000)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    base_coverage_retry = _coverage_retry_acceptance(authority)
    coverage_retry = replace(
        base_coverage_retry,
        timing=replace(
            base_coverage_retry.timing,
            coverage_gaps=(
                replace(
                    base_coverage_retry.timing.coverage_gaps[0],
                    start_ms=12_000,
                    end_ms=20_000,
                ),
            ),
        ),
    )
    soft_coverage_review = replace(
        accepted,
        action=GateAction.REVIEW,
        decisions=tuple(
            replace(
                decision,
                action=GateAction.REVIEW,
                reasons=("SUSTAIN_REPRESENTABLE_MIDDLE_GAP",),
            )
            if decision.axis is GateAxis.COVERAGE
            else decision
            for decision in accepted.decisions
        ),
    )

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        if generated.generator_name == "adaptive-recovery-v1":
            return _acceptance_for_difficulty(accepted, requested_difficulty)
        if generated.generator_name.startswith("coverage-repair-v1:"):
            return _acceptance_for_difficulty(soft_coverage_review, requested_difficulty)
        if requested_difficulty == "EASY" and generated.key_mode == 4:
            return _acceptance_for_difficulty(coverage_retry, requested_difficulty)
        return _acceptance_for_difficulty(accepted, requested_difficulty)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)
    analysis = _active_analysis(60_000, onset_step_ms=1_000)

    class OneBoundedGapGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if (request.key_mode, request.difficulty) != (4, "EASY"):
                return generated
            active_rows = analysis.activity.active_onset_ms  # type: ignore[union-attr]
            rows = sorted(
                {
                    12_000,
                    20_000,
                    *(row for row in active_rows if not 12_000 < row < 20_000),
                }
            )
            return replace(
                generated,
                notes=[
                    NoteEvent(time_ms=row, lane=index % request.key_mode)
                    for index, row in enumerate(rows)
                ],
            )

    outcome = run_generation(
        prepared,
        authority,
        analysis,
        tmp_path,
        generator=OneBoundedGapGenerator(),
        seed=0,
    )

    selected = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EASY")
    )
    assert selected.provenance == "COVERAGE_REPAIR"


@pytest.mark.parametrize("selector_mode", ("V2", "SHADOW_V2"))
def test_uncertain_gap_remains_review_without_spending_additional_inference(
    monkeypatch,
    tmp_path: Path,
    selector_mode: str,
):
    duration_ms = 40_000
    prepared = replace(
        _prepared(tmp_path, duration_ms=duration_ms),
        difficulty_selector_mode=selector_mode,
    )
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    opportunity = CoverageOpportunity(
        version="coverage-opportunity-v2",
        start_ms=12_000,
        end_ms=20_000,
        beat_count=16.0,
        strong_attack_count=3,
        active_onset_count=7,
        hold_occupancy_ratio=0.0,
        active_frame_ratio=0.65,
        strong_attack_threshold=0.9,
        evidence_confidence="INSUFFICIENT",
        kind=CoverageKind.INSUFFICIENT_EVIDENCE,
    )
    local_evidence = LocalAudioGapEvidence(
        version="coverage-jury-local-evidence-v1",
        start_ms=12_000,
        end_ms=20_000,
        active_frame_ratio=0.65,
        active_onset_count=7,
        global_strong_attack_count=3,
        local_strong_attack_count=2,
        global_threshold=0.9,
        local_threshold=0.8,
        neighboring_activity_ratio=0.7,
    )
    gap = TimingCoverageGap(
        start_ms=12_000,
        end_ms=20_000,
        onset_count=10,
        active_onset_count=7,
        active_frame_ratio=0.65,
        position="MIDDLE",
        opportunity=opportunity,
        local_audio_evidence=local_evidence,
    )
    uncertain_review = replace(
        accepted,
        action=GateAction.REVIEW,
        decisions=tuple(
            replace(
                decision,
                action=GateAction.REVIEW,
                reasons=("INSUFFICIENT_COVERAGE_EVIDENCE_MIDDLE_GAP",),
            )
            if decision.axis is GateAxis.COVERAGE
            else decision
            for decision in accepted.decisions
        ),
        timing=replace(accepted.timing, coverage_gaps=(gap,)),
    )

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        if (generated.key_mode, requested_difficulty) == (4, "EXPERT"):
            base = uncertain_review
        else:
            base = accepted
        return _acceptance_for_difficulty(base, requested_difficulty)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)

    generator = RecordingGenerator()
    outcome = run_generation(
        prepared,
        authority,
        _active_analysis(duration_ms, onset_step_ms=1_000),
        tmp_path,
        generator=generator,
        seed=0,
    )

    selected = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EXPERT")
    )
    assert len(generator.map_calls) == 12
    assert not any(request.add_to_beatmap for request in generator.map_calls)
    assert outcome.additional_inference_calls == 0
    assert selected.provenance == "PRIMARY"
    assert selected.acceptance.action is GateAction.REVIEW
    assert selected.acceptance.timing.coverage_gaps == (gap,)
    assert outcome.song_selection_shadow is not None
    assert not any(
        candidate.key_mode == 4
        and candidate.difficulty == "EXPERT"
        and candidate.provenance == "PARTIAL_REMAP"
        for candidate in outcome.song_selection_shadow.replay_input.candidates
    )


def test_uncertain_gap_stays_report_only_even_when_family_audio_review_corroborates(
    monkeypatch,
    tmp_path: Path,
):
    duration_ms = 40_000
    prepared = _prepared(tmp_path, duration_ms=duration_ms)
    authority = replace(
        _authority(prepared, tmp_path),
        timing_integrity=TimingIntegrityAssessment(
            status=TimingIntegrityStatus.HEALTHY,
            reasons=(),
            islands=(),
        ),
    )
    accepted = _pass_acceptance(authority)
    opportunity = CoverageOpportunity(
        version="coverage-opportunity-v2",
        start_ms=12_000,
        end_ms=20_000,
        beat_count=16.0,
        strong_attack_count=3,
        active_onset_count=7,
        hold_occupancy_ratio=0.0,
        active_frame_ratio=0.65,
        strong_attack_threshold=0.9,
        evidence_confidence="INSUFFICIENT",
        kind=CoverageKind.INSUFFICIENT_EVIDENCE,
    )
    local_evidence = LocalAudioGapEvidence(
        version="coverage-jury-local-evidence-v1",
        start_ms=12_000,
        end_ms=20_000,
        active_frame_ratio=0.65,
        active_onset_count=7,
        global_strong_attack_count=3,
        local_strong_attack_count=3,
        global_threshold=0.9,
        local_threshold=0.8,
        neighboring_activity_ratio=0.7,
    )
    gap = TimingCoverageGap(
        start_ms=12_000,
        end_ms=20_000,
        onset_count=10,
        active_onset_count=7,
        active_frame_ratio=0.65,
        position="MIDDLE",
        opportunity=opportunity,
        local_audio_evidence=local_evidence,
    )
    corroborated_review = replace(
        accepted,
        action=GateAction.REVIEW,
        decisions=tuple(
            replace(
                decision,
                action=GateAction.REVIEW,
                reasons=("INSUFFICIENT_COVERAGE_EVIDENCE_MIDDLE_GAP",),
            )
            if decision.axis is GateAxis.COVERAGE
            else decision
            for decision in accepted.decisions
        ),
        timing=replace(accepted.timing, coverage_gaps=(gap,)),
    )

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        if generated.generator_name.startswith("coverage-repair-v1:"):
            base = accepted
        elif (generated.key_mode, requested_difficulty) == (4, "EXPERT"):
            base = corroborated_review
        else:
            base = accepted
        return _acceptance_for_difficulty(base, requested_difficulty)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)
    generator = RecordingGenerator()

    outcome = run_generation(
        prepared,
        authority,
        _active_analysis(duration_ms, onset_step_ms=1_000),
        tmp_path,
        generator=generator,
        seed=0,
    )

    selected = next(
        variant
        for variant in outcome.variants
        if (variant.key_mode, variant.difficulty) == (4, "EXPERT")
    )
    assert len(generator.map_calls) == 12
    assert outcome.additional_inference_calls == 0
    assert selected.provenance == "PRIMARY"
    assert selected.coverage_repair_gap_count == 0
    assert selected.acceptance.decision(GateAxis.COVERAGE).action is GateAction.REVIEW
    assert not any(variant.provenance == "COVERAGE_REPAIR" for variant in outcome.variants)


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


def test_mixed_exhaustion_retains_gate_evidence_with_all_legacy_errors(monkeypatch, tmp_path: Path):
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
            "reason": "HARD_REJECT_RETRY",
            "candidateDisposition": "HARD_REJECT",
            "repairEligible": False,
            "repairAxes": [],
        }
    ]


def test_real_acceptance_evidence_is_recorded_for_aligned_candidates(monkeypatch, tmp_path: Path):
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
    monkeypatch,
    tmp_path: Path,
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
    attempts = [item for item in fallback.attempt_evidence if "gateReport" in item]
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
    assert all("CUDA out of memory" in entry["context"]["stderr"] for entry in errors)
    assert len(errors) == 3
    journal_path = tmp_path / "attempt-journal.jsonl"
    journal = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    assert len(journal) == 12 * 3 * 2
    assert [entry["sequence"] for entry in journal] == list(range(1, len(journal) + 1))
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
    assert {request.last_attack_ms for request in generator.map_calls} == {expected_last_attack_ms}
    assert {request.generation_end_ms for request in generator.map_calls} == {30_000}
    assert all(request.duration_ms == duration_ms for request in generator.map_calls)


def test_high_confidence_terminal_consensus_reaches_every_generation_request(
    tmp_path: Path,
):
    duration_ms = 30_000
    prepared = _prepared(
        tmp_path,
        duration_ms=duration_ms,
        boundary_policy_mode="HIGH_CONFIDENCE_ENFORCED",
    )
    authority = _authority(prepared, tmp_path)
    analysis = replace(
        _analysis(),
        activity=AudioActivity(
            frame_ms=100.0,
            rms_db=np.concatenate([np.full(200, -10.0), np.full(100, -80.0)]),
            floor_db=-60.0,
            active_onset_ms=tuple(range(0, 20_000, 250)),
        ),
        terminal_silence=TerminalSilenceObservation(
            version="terminal-silence-observation-v1",
            duration_ms=duration_ms,
            frame_ms=20,
            channel_count=2,
            candidates=tuple(
                TerminalThresholdCandidate(
                    rms_db=rms_db,
                    peak_db=peak_db,
                    suffix_start_ms=20_000,
                    suffix_duration_ms=10_000,
                )
                for rms_db, peak_db in (
                    (-72.0, -60.0),
                    (-66.0, -54.0),
                    (-60.0, -48.0),
                )
            ),
            candidate_spread_ms=0,
            last_onset_ms=19_750,
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

    assert {request.last_attack_ms for request in generator.map_calls} == {20_000}
    assert {request.max_note_start_ms for request in generator.map_calls} == {20_000}
    assert {request.generation_end_ms for request in generator.map_calls} == {20_000}


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
    assert {request.last_attack_ms for request in generator.map_calls} == {duration_ms}
    assert {request.generation_end_ms for request in generator.map_calls} == {duration_ms}
