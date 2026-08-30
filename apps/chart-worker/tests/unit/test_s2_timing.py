from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.beat import BeatGrid
from chart_worker.analysis.intro_anchor import IntroAnchorEvidence
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.mapperatorinator import GeneratedTiming
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.generation.osu_writer import timing_to_osu_mania
from chart_worker.hashing import sha256_file
from chart_worker.stages import s2_timing
from chart_worker.stages.s2_timing import (
    review_intro_timing_addressability,
    run_intro_prefix_timing_recovery,
    run_timing_generation,
)
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.leading_timing_coverage import LeadingTimingCoverage
from chart_worker.validation.timing_authority import TimingAuthorityValidationError
from chart_worker.validation.timing_integrity import (
    TimingIntegrityAssessment,
    TimingIntegrityStatus,
)
from chart_worker.validation.timing_review import (
    TimingAuthorityAction,
    TimingAuthorityReview,
)


def _prepared(tmp_path: Path, *, duration_ms: int = 2_000) -> PreparedAudio:
    audio_path = tmp_path / "audio" / "game.flac"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"canonical-audio")
    return PreparedAudio(
        normalized=NormalizedAudio(
            path=audio_path,
            profile_version="audio-profile-v2",
            sha256=sha256_file(audio_path),
            duration_ms=duration_ms,
            sample_rate_hz=48_000,
            channels=2,
            source_duration_ms=duration_ms,
            trimmed_ms=0,
            gain_db=0.0,
            achieved_lufs=-14.0,
            achieved_true_peak_dbtp=-1.0,
            shortfall_lu=0.0,
            limited_by="LOUDNESS",
        )
    )


def _prepared_sha(tmp_path: Path) -> str:
    return sha256_file(tmp_path / "audio" / "game.flac")


def _intro_authority(
    prepared: PreparedAudio,
    tmp_path: Path,
    *,
    first_event_time_ms: int,
    anchor_status: str = "CONFIRMED",
) -> SongTimingAuthority:
    bpm_events = (OsuBpmEvent(first_event_time_ms, 120.0),)
    reference_path = tmp_path / "audio" / "timing-reference.osu"
    reference_path.write_text(
        timing_to_osu_mania(
            bpm_events,
            audio_filename=prepared.normalized.path.name,
            title="fixture",
        ),
        encoding="utf-8",
    )
    intro_anchor = IntroAnchorEvidence(
        status=anchor_status,
        anchor_ms=250,
        anchor_grid_ms=250,
        grid_distance_ms=0,
        aggregate_percentile_rank=0.99,
        prominent_band_count=3,
        pulse_continuation_matches=4,
        pulse_continuation_opportunities=4,
        supported_pulse_ms=tuple(range(250, 4_251, 250)),
    )
    return SongTimingAuthority(
        reference_path=reference_path,
        sha256=sha256_file(reference_path),
        audio_sha256=prepared.normalized.sha256,
        bpm_events=bpm_events,
        generator_name="intro-authority-fixture",
        seed=17,
        mode="STANDARD",
        attempt_count=1,
        review=TimingAuthorityReview(
            TimingAuthorityAction.REVIEW,
            ("CONFIRMED_INTRO_ANCHOR_BEFORE_FIRST_EVENT",),
        ),
        leading_coverage=LeadingTimingCoverage(
            action=TimingAuthorityAction.REVIEW,
            reasons=("CONFIRMED_INTRO_ANCHOR_BEFORE_FIRST_EVENT",),
            first_event_time_ms=first_event_time_ms,
            leading_duration_ms=first_event_time_ms,
            onset_count=17,
            active_onset_count=17,
            active_frame_ratio=1.0,
            intro_anchor=intro_anchor,
        ),
    )


class RecordingGenerator:
    def __init__(self) -> None:
        self.timing_calls = []
        self.timing_workdirs = []

    def generate_timing(self, request, workdir):
        self.timing_calls.append(request)
        self.timing_workdirs.append(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        return GeneratedTiming(
            osu_text="",
            bpm_events=(OsuBpmEvent(0, 120.0),),
            generator_name="recording-generator",
            seed=request.seed,
            mode="SUPER_TIMING" if request.super_timing else "STANDARD",
        )


@pytest.mark.parametrize(
    ("anchor_status", "first_event_time_ms", "expected_status"),
    [
        ("UNCERTAIN", 6_937, "NOT_APPLICABLE"),
        ("CONFIRMED", 0, "ADDRESSED"),
        ("CONFIRMED", 6_937, "UNADDRESSED"),
    ],
)
def test_intro_timing_addressability_uses_confirmed_region_and_explicit_event_coverage(
    tmp_path: Path,
    anchor_status: str,
    first_event_time_ms: int,
    expected_status: str,
):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    authority = _intro_authority(
        prepared,
        tmp_path,
        first_event_time_ms=first_event_time_ms,
        anchor_status=anchor_status,
    )

    observation = review_intro_timing_addressability(
        authority,
        _active_analysis(20_000),
        duration_ms=20_000,
    )

    assert observation.status == expected_status
    assert observation.first_event_time_ms == first_event_time_ms
    if anchor_status == "CONFIRMED":
        assert observation.allowed_first_row_ms == (180, 4_320)
    else:
        assert observation.allowed_first_row_ms is None


def test_intro_prefix_timing_recovery_promotes_only_an_addressed_full_song_candidate(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    original = _intro_authority(
        prepared,
        tmp_path,
        first_event_time_ms=6_937,
    )
    original_bytes = original.reference_path.read_bytes()
    generator = RecordingGenerator()

    outcome = run_intro_prefix_timing_recovery(
        prepared,
        original,
        _active_analysis(20_000),
        tmp_path,
        generator=generator,
        seed=7,
    )

    assert outcome.status == "SELECTED"
    assert outcome.attempted is True
    assert outcome.authority_epoch == 2
    assert len(generator.timing_calls) == 1
    assert generator.timing_calls[0].super_timing is True
    assert generator.timing_calls[0].seed != 7
    assert outcome.original_authority_sha256 == original.sha256
    assert outcome.authority.sha256 != original.sha256
    assert sha256_file(outcome.authority.reference_path) == outcome.authority.sha256
    assert outcome.authority.reference_path.read_bytes() != original_bytes
    assert outcome.retry_addressability is not None
    assert outcome.retry_addressability.status == "ADDRESSED"


def test_intro_prefix_timing_recovery_does_not_call_model_when_already_addressed(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    original = _intro_authority(
        prepared,
        tmp_path,
        first_event_time_ms=0,
    )

    class UnexpectedGenerator(RecordingGenerator):
        def generate_timing(self, request, workdir):
            raise AssertionError("timing recovery must not run for an addressed intro")

    outcome = run_intro_prefix_timing_recovery(
        prepared,
        original,
        _active_analysis(20_000),
        tmp_path,
        generator=UnexpectedGenerator(),
        seed=7,
    )

    assert outcome.status == "NOT_TRIGGERED"
    assert outcome.attempted is False
    assert outcome.authority is original
    assert outcome.authority_epoch == 1
    assert outcome.original_addressability.status == "ADDRESSED"


def test_intro_prefix_timing_recovery_keeps_original_when_retry_is_unaddressed(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    original = _intro_authority(
        prepared,
        tmp_path,
        first_event_time_ms=6_937,
    )
    original_bytes = original.reference_path.read_bytes()

    class StillLateGenerator(RecordingGenerator):
        def generate_timing(self, request, workdir):
            generated = super().generate_timing(request, workdir)
            return replace(
                generated,
                bpm_events=(OsuBpmEvent(6_937, 120.0),),
            )

    generator = StillLateGenerator()
    outcome = run_intro_prefix_timing_recovery(
        prepared,
        original,
        _active_analysis(20_000),
        tmp_path,
        generator=generator,
        seed=7,
    )

    assert outcome.status == "UNRESOLVED"
    assert outcome.authority is original
    assert outcome.authority_epoch == 1
    assert len(generator.timing_calls) == 1
    assert original.reference_path.read_bytes() == original_bytes
    assert sha256_file(original.reference_path) == original.sha256
    assert outcome.retry_addressability is not None
    assert outcome.retry_addressability.status == "UNADDRESSED"


def test_intro_prefix_timing_recovery_keeps_original_after_known_failure(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    original = _intro_authority(
        prepared,
        tmp_path,
        first_event_time_ms=6_937,
    )
    original_bytes = original.reference_path.read_bytes()
    failure = WorkerError(ErrorCode.CHART_GENERATION_FAILED, "known timing failure")

    class FailingGenerator(RecordingGenerator):
        def generate_timing(self, request, workdir):
            self.timing_calls.append(request)
            self.timing_workdirs.append(workdir)
            raise failure

    generator = FailingGenerator()
    outcome = run_intro_prefix_timing_recovery(
        prepared,
        original,
        _active_analysis(20_000),
        tmp_path,
        generator=generator,
        seed=7,
    )

    assert outcome.status == "FAILED"
    assert outcome.authority is original
    assert outcome.authority_epoch == 1
    assert outcome.error is not None
    assert outcome.error["code"] == "CHART_GENERATION_FAILED"
    assert original.reference_path.read_bytes() == original_bytes
    assert sha256_file(original.reference_path) == original.sha256


def test_intro_prefix_timing_recovery_propagates_unknown_completion_without_mutation(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path, duration_ms=20_000)
    original = _intro_authority(
        prepared,
        tmp_path,
        first_event_time_ms=6_937,
    )
    original_bytes = original.reference_path.read_bytes()
    failure = WorkerError(
        ErrorCode.INFERENCE_COMPLETION_UNKNOWN,
        "terminal state unavailable",
    )

    class UnknownCompletionGenerator(RecordingGenerator):
        def generate_timing(self, request, workdir):
            self.timing_calls.append(request)
            self.timing_workdirs.append(workdir)
            raise failure

    with pytest.raises(WorkerError) as captured:
        run_intro_prefix_timing_recovery(
            prepared,
            original,
            _active_analysis(20_000),
            tmp_path,
            generator=UnknownCompletionGenerator(),
            seed=7,
        )

    assert captured.value is failure
    assert captured.value.context["introPrefixTimingRecovery"]["status"] == "FAILED"
    assert captured.value.context["introPrefixTimingRecovery"]["attempted"] is True
    assert original.reference_path.read_bytes() == original_bytes
    assert sha256_file(original.reference_path) == original.sha256


class InvalidThenSuperGenerator(RecordingGenerator):
    def generate_timing(self, request, workdir):
        self.timing_calls.append(request)
        self.timing_workdirs.append(workdir)
        if not request.super_timing:
            return GeneratedTiming(
                osu_text="",
                bpm_events=(),
                generator_name="recording-generator",
                seed=request.seed,
                mode="STANDARD",
            )
        return GeneratedTiming(
            osu_text="",
            bpm_events=(OsuBpmEvent(0, 120.0),),
            generator_name="recording-generator",
            seed=request.seed,
            mode="SUPER_TIMING",
        )


class LongActiveThenSuperGenerator(RecordingGenerator):
    def generate_timing(self, request, workdir):
        self.timing_calls.append(request)
        self.timing_workdirs.append(workdir)
        return GeneratedTiming(
            osu_text="",
            bpm_events=(OsuBpmEvent(0 if request.super_timing else 95_645, 120.0),),
            generator_name="recording-generator",
            seed=request.seed,
            mode="SUPER_TIMING" if request.super_timing else "STANDARD",
        )


class ShortActiveIntroGenerator(RecordingGenerator):
    def generate_timing(self, request, workdir):
        self.timing_calls.append(request)
        self.timing_workdirs.append(workdir)
        return GeneratedTiming(
            osu_text="",
            bpm_events=(OsuBpmEvent(2_678, 120.0),),
            generator_name="recording-generator",
            seed=request.seed,
            mode="SUPER_TIMING" if request.super_timing else "STANDARD",
        )


class LongThenShortActiveIntroGenerator(RecordingGenerator):
    def generate_timing(self, request, workdir):
        self.timing_calls.append(request)
        self.timing_workdirs.append(workdir)
        return GeneratedTiming(
            osu_text="",
            bpm_events=(
                OsuBpmEvent(2_678 if request.super_timing else 95_645, 120.0),
            ),
            generator_name="recording-generator",
            seed=request.seed,
            mode="SUPER_TIMING" if request.super_timing else "STANDARD",
        )


class AlwaysLongActiveGapGenerator(RecordingGenerator):
    def generate_timing(self, request, workdir):
        self.timing_calls.append(request)
        self.timing_workdirs.append(workdir)
        return GeneratedTiming(
            osu_text="",
            bpm_events=(OsuBpmEvent(95_645, 120.0),),
            generator_name="recording-generator",
            seed=request.seed,
            mode="SUPER_TIMING" if request.super_timing else "STANDARD",
        )


class LocalOutlierThenSuperGenerator(RecordingGenerator):
    def generate_timing(self, request, workdir):
        self.timing_calls.append(request)
        self.timing_workdirs.append(workdir)
        events = (
            (OsuBpmEvent(0, 120.0),)
            if request.super_timing
            else (
                OsuBpmEvent(0, 80.0),
                OsuBpmEvent(10_000, 5.6),
                OsuBpmEvent(21_000, 165.0),
            )
        )
        return GeneratedTiming(
            osu_text="",
            bpm_events=events,
            generator_name="recording-generator",
            seed=request.seed,
            mode="SUPER_TIMING" if request.super_timing else "STANDARD",
        )


class AlwaysLocalOutlierGenerator(LocalOutlierThenSuperGenerator):
    def generate_timing(self, request, workdir):
        generated = super().generate_timing(request, workdir)
        return GeneratedTiming(
            osu_text="",
            bpm_events=(
                OsuBpmEvent(0, 80.0),
                OsuBpmEvent(10_000, 5.6),
                OsuBpmEvent(21_000, 165.0),
            ),
            generator_name=generated.generator_name,
            seed=generated.seed,
            mode=generated.mode,
        )


def _half_tempo_analysis() -> OnsetAnalysis:
    strength = np.zeros(401)
    strength[::10] = 1.0
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strength,
        band_strength=np.zeros((3, strength.size)),
        onset_ms=(),
        n_fft=1,
    )


def _base_tempo_analysis(*, offset_ms: int = 0) -> OnsetAnalysis:
    strength = np.zeros(101)
    strength[offset_ms // 100 :: 5] = 1.0
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strength,
        band_strength=np.zeros((3, strength.size)),
        onset_ms=(),
        n_fft=1,
    )


def _active_analysis(duration_ms: int) -> OnsetAnalysis:
    frame_ms = 100.0
    frame_count = duration_ms // 100 + 1
    strength = np.zeros(frame_count)
    strength[::5] = 1.0
    onsets = tuple(range(0, duration_ms, 250))
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strength,
        band_strength=np.zeros((3, frame_count)),
        onset_ms=onsets,
        n_fft=1,
        activity=AudioActivity(
            frame_ms=frame_ms,
            rms_db=np.full(frame_count, -10.0),
            floor_db=-60.0,
            active_onset_ms=onsets,
        ),
    )


class RetryThenReviewGenerator(RecordingGenerator):
    def generate_timing(self, request, workdir):
        generated = super().generate_timing(request, workdir)
        if not request.super_timing:
            return generated
        return GeneratedTiming(
            osu_text="",
            bpm_events=(OsuBpmEvent(250, 120.0),),
            generator_name=generated.generator_name,
            seed=generated.seed,
            mode=generated.mode,
        )


class IdentityFailureThenSuperGenerator(RecordingGenerator):
    pass


def test_generates_one_standard_timing_and_promotes_it_beside_audio(tmp_path):
    generator = RecordingGenerator()
    authority = run_timing_generation(
        _prepared(tmp_path),
        _base_tempo_analysis(),
        tmp_path,
        generator=generator,
        seed=9,
    )

    assert [call.super_timing for call in generator.timing_calls] == [False]
    assert authority.reference_path == tmp_path / "audio" / "timing-reference.osu"
    assert authority.sha256 == sha256_file(authority.reference_path)
    assert authority.audio_sha256 == _prepared_sha(tmp_path)
    assert authority.mode == "STANDARD"
    assert authority.tempo_metrics is not None
    assert authority.review is not None
    assert authority.candidate_selection is not None
    assert authority.candidate_selection.reason == "ONLY_STRUCTURALLY_VALID_CANDIDATE"
    assert len(authority.candidate_selection.candidates) == 1


def test_structural_standard_failure_uses_super_timing_once(tmp_path):
    generator = InvalidThenSuperGenerator()
    authority = run_timing_generation(
        _prepared(tmp_path),
        _base_tempo_analysis(),
        tmp_path,
        generator=generator,
        seed=9,
    )

    assert [call.super_timing for call in generator.timing_calls] == [False, True]
    assert authority.mode == "SUPER_TIMING"
    assert authority.attempt_count == 2


def test_long_active_standard_gap_uses_super_timing_once(tmp_path):
    generator = LongActiveThenSuperGenerator()

    authority = run_timing_generation(
        _prepared(tmp_path, duration_ms=150_000),
        _active_analysis(150_000),
        tmp_path,
        generator=generator,
        seed=9,
    )

    assert [call.super_timing for call in generator.timing_calls] == [False, True]
    assert authority.mode == "SUPER_TIMING"
    assert authority.attempt_count == 2
    assert authority.bpm_events[0].time_ms == 0
    assert authority.leading_coverage is not None
    assert authority.leading_coverage.action is TimingAuthorityAction.PASS


def test_short_phase_supported_intro_keeps_standard_for_map_recovery(tmp_path):
    """인트로 phase가 맞으면 같은 Timing을 재생성하지 않는다."""
    generator = ShortActiveIntroGenerator()

    authority = run_timing_generation(
        _prepared(tmp_path, duration_ms=150_000),
        _active_analysis(150_000),
        tmp_path,
        generator=generator,
        seed=9,
    )

    assert [call.super_timing for call in generator.timing_calls] == [False]
    assert authority.mode == "STANDARD"
    assert authority.bpm_events[0].time_ms == 2_678
    assert authority.leading_coverage is not None
    assert authority.leading_coverage.action is TimingAuthorityAction.REVIEW
    assert authority.leading_coverage.reasons == (
        "CONFIRMED_INTRO_ANCHOR_BEFORE_FIRST_EVENT",
    )


def test_super_short_active_intro_is_promoted_without_event_mutation(tmp_path):
    generator = LongThenShortActiveIntroGenerator()

    authority = run_timing_generation(
        _prepared(tmp_path, duration_ms=150_000),
        _active_analysis(150_000),
        tmp_path,
        generator=generator,
        seed=9,
    )

    assert [call.super_timing for call in generator.timing_calls] == [False, True]
    assert authority.mode == "SUPER_TIMING"
    assert authority.bpm_events[0].time_ms == 2_678
    assert authority.leading_coverage is not None
    assert authority.leading_coverage.action is TimingAuthorityAction.REVIEW


def test_two_long_active_gaps_fail_with_attempt_evidence(tmp_path):
    generator = AlwaysLongActiveGapGenerator()

    with pytest.raises(WorkerError) as captured:
        run_timing_generation(
            _prepared(tmp_path, duration_ms=150_000),
            _active_analysis(150_000),
            tmp_path,
            generator=generator,
            seed=9,
        )

    assert captured.value.code is ErrorCode.CHART_TIMING_CANDIDATE_FAILED
    assert [call.super_timing for call in generator.timing_calls] == [False, True]
    attempts = captured.value.context["attempts"]
    assert len(attempts) == 2
    assert all(
        attempt["leadingCoverage"]["action"] == "RETRY_TIMING"
        for attempt in attempts
    )


def test_two_long_active_gaps_try_independent_beat_grid_before_final_failure(
    monkeypatch,
    tmp_path,
):
    duration_ms = 150_000
    generator = AlwaysLongActiveGapGenerator()
    prepared = replace(
        _prepared(tmp_path, duration_ms=duration_ms),
        beat_this_enabled=True,
    )
    beat_calls = []

    def analyze_beats(path):
        beat_calls.append(path)
        beat_ms = tuple(range(0, duration_ms, 500))
        return BeatGrid(
            beat_ms=beat_ms,
            downbeat_indices=tuple(range(0, len(beat_ms), 4)),
            bpm=120.0,
            beats_per_bar=4,
            bpm_drift_pct=0.0,
            raw_beat_count=len(beat_ms),
            dropped_beat_count=0,
            residual_rms_ms=0.0,
            residual_max_ms=0.0,
        )

    monkeypatch.setattr(
        s2_timing,
        "review_timing_authority",
        lambda metrics: TimingAuthorityReview(TimingAuthorityAction.PASS, ()),
    )

    authority = run_timing_generation(
        prepared,
        _active_analysis(duration_ms),
        tmp_path,
        generator=generator,
        seed=9,
        beat_analyzer=analyze_beats,
    )

    assert [call.super_timing for call in generator.timing_calls] == [False, True]
    assert beat_calls == [prepared.normalized.path]
    assert authority.mode == "BEAT_THIS_FALLBACK"
    assert authority.attempt_count == 3
    assert authority.leading_coverage is not None
    assert authority.leading_coverage.action is TimingAuthorityAction.PASS
    assert authority.reference_path.is_file()


def test_integrated_tempo_rejection_reaches_independent_fallback(monkeypatch, tmp_path):
    duration_ms = 30_000
    prepared = replace(
        _prepared(tmp_path, duration_ms=duration_ms),
        beat_this_enabled=True,
    )
    reviews = iter(
        (
            TimingAuthorityReview(
                TimingAuthorityAction.RETRY_TIMING,
                ("STRONG_HALF_TEMPO_ALTERNATIVE",),
            ),
            TimingAuthorityReview(
                TimingAuthorityAction.RETRY_TIMING,
                ("STRONG_HALF_TEMPO_ALTERNATIVE",),
            ),
            TimingAuthorityReview(TimingAuthorityAction.PASS, ()),
        )
    )
    monkeypatch.setattr(s2_timing, "review_timing_authority", lambda metrics: next(reviews))
    beat_calls = []

    def analyze_beats(path):
        beat_calls.append(path)
        beat_ms = tuple(range(0, duration_ms, 500))
        return BeatGrid(
            beat_ms=beat_ms,
            downbeat_indices=tuple(range(0, len(beat_ms), 4)),
            bpm=120.0,
            beats_per_bar=4,
            bpm_drift_pct=0.0,
            raw_beat_count=len(beat_ms),
            dropped_beat_count=0,
            residual_rms_ms=0.0,
            residual_max_ms=0.0,
        )

    authority = run_timing_generation(
        prepared,
        _base_tempo_analysis(),
        tmp_path,
        generator=RecordingGenerator(),
        seed=9,
        beat_analyzer=analyze_beats,
    )

    assert beat_calls == [prepared.normalized.path]
    assert authority.mode == "BEAT_THIS_FALLBACK"
    assert authority.attempt_count == 3


def test_integrated_tempo_rejection_without_fallback_still_requires_review(
    monkeypatch, tmp_path
):
    reviews = iter(
        (
            TimingAuthorityReview(
                TimingAuthorityAction.RETRY_TIMING,
                ("STRONG_HALF_TEMPO_ALTERNATIVE",),
            ),
            TimingAuthorityReview(
                TimingAuthorityAction.RETRY_TIMING,
                ("STRONG_HALF_TEMPO_ALTERNATIVE",),
            ),
        )
    )
    monkeypatch.setattr(s2_timing, "review_timing_authority", lambda metrics: next(reviews))

    with pytest.raises(WorkerError) as captured:
        run_timing_generation(
            _prepared(tmp_path, duration_ms=30_000),
            _base_tempo_analysis(),
            tmp_path,
            generator=RecordingGenerator(),
            seed=9,
        )

    assert captured.value.code is ErrorCode.CHART_TIMING_REVIEW_REQUIRED
    assert captured.value.context["attempt_count"] == 2


def test_local_timing_damage_uses_super_once_before_map(monkeypatch, tmp_path):
    generator = LocalOutlierThenSuperGenerator()
    monkeypatch.setattr(
        s2_timing,
        "review_timing_authority",
        lambda metrics: TimingAuthorityReview(TimingAuthorityAction.PASS, ()),
    )

    authority = run_timing_generation(
        _prepared(tmp_path, duration_ms=30_000),
        _active_analysis(30_000),
        tmp_path,
        generator=generator,
        seed=9,
    )

    assert [call.super_timing for call in generator.timing_calls] == [False, True]
    assert authority.mode == "SUPER_TIMING"
    assert authority.local_review is not None
    assert authority.local_review.action is TimingAuthorityAction.PASS
    assert authority.recovery_preflight is not None


def test_two_locally_damaged_timing_candidates_fail_with_segment_evidence(
    monkeypatch, tmp_path
):
    generator = AlwaysLocalOutlierGenerator()
    monkeypatch.setattr(
        s2_timing,
        "review_timing_authority",
        lambda metrics: TimingAuthorityReview(TimingAuthorityAction.PASS, ()),
    )

    with pytest.raises(WorkerError) as captured:
        run_timing_generation(
            _prepared(tmp_path, duration_ms=30_000),
            _active_analysis(30_000),
            tmp_path,
            generator=generator,
            seed=9,
        )

    assert captured.value.code is ErrorCode.CHART_TIMING_CANDIDATE_FAILED
    attempts = captured.value.context["attempts"]
    assert len(attempts) == 2
    assert all(
        attempt["localTimingReview"]["action"] == "RETRY_TIMING"
        for attempt in attempts
    )


def test_healthy_local_timing_keeps_single_standard_inference(monkeypatch, tmp_path):
    generator = RecordingGenerator()
    monkeypatch.setattr(
        s2_timing,
        "review_timing_authority",
        lambda metrics: TimingAuthorityReview(TimingAuthorityAction.PASS, ()),
    )

    authority = run_timing_generation(
        _prepared(tmp_path, duration_ms=30_000),
        _active_analysis(30_000),
        tmp_path,
        generator=generator,
        seed=9,
    )

    assert [call.super_timing for call in generator.timing_calls] == [False]
    assert authority.local_review is not None
    assert authority.local_review.action is TimingAuthorityAction.PASS


def test_damaged_standard_integrity_uses_super_and_never_promotes_the_damaged_candidate(
    monkeypatch, tmp_path
):
    assessments = iter(
        (
            TimingIntegrityAssessment(
                status=TimingIntegrityStatus.DAMAGED,
                reasons=(
                    "ACTIVE_RETURN_TIMING_ISLAND",
                    "RECOVERY_PREFLIGHT_DAMAGED",
                ),
                islands=(),
            ),
            TimingIntegrityAssessment(
                status=TimingIntegrityStatus.HEALTHY,
                reasons=(),
                islands=(),
            ),
        )
    )
    monkeypatch.setattr(
        s2_timing,
        "assess_timing_integrity",
        lambda local_review, recovery_preflight: next(assessments),
    )
    generator = RecordingGenerator()

    authority = run_timing_generation(
        _prepared(tmp_path, duration_ms=30_000),
        _active_analysis(30_000),
        tmp_path,
        generator=generator,
        seed=9,
    )

    assert [call.super_timing for call in generator.timing_calls] == [False, True]
    assert authority.mode == "SUPER_TIMING"
    assert authority.candidate_selection is not None
    assert [
        candidate.mode for candidate in authority.candidate_selection.candidates
    ] == ["SUPER_TIMING"]


def test_standard_needing_corroboration_compares_super_before_selection(
    monkeypatch, tmp_path
):
    assessments = iter(
        (
            TimingIntegrityAssessment(
                status=TimingIntegrityStatus.NEEDS_CORROBORATION,
                reasons=("ACTIVE_RETURN_TIMING_ISLAND",),
                islands=(),
            ),
            TimingIntegrityAssessment(
                status=TimingIntegrityStatus.HEALTHY,
                reasons=(),
                islands=(),
            ),
        )
    )
    monkeypatch.setattr(
        s2_timing,
        "assess_timing_integrity",
        lambda local_review, recovery_preflight: next(assessments),
    )
    generator = RecordingGenerator()

    authority = run_timing_generation(
        _prepared(tmp_path, duration_ms=30_000),
        _active_analysis(30_000),
        tmp_path,
        generator=generator,
        seed=9,
    )

    assert [call.super_timing for call in generator.timing_calls] == [False, True]
    assert authority.mode == "SUPER_TIMING"
    assert authority.candidate_selection is not None
    assert authority.candidate_selection.reason == "BETTER_TIMING_INTEGRITY"


def test_two_uncertain_internal_candidates_use_one_independent_fallback(
    monkeypatch, tmp_path
):
    assessments = iter(
        (
            TimingIntegrityAssessment(
                status=TimingIntegrityStatus.NEEDS_CORROBORATION,
                reasons=("ACTIVE_RETURN_TIMING_ISLAND",),
                islands=(),
            ),
            TimingIntegrityAssessment(
                status=TimingIntegrityStatus.NEEDS_CORROBORATION,
                reasons=("RECOVERY_PREFLIGHT_DAMAGED",),
                islands=(),
            ),
            TimingIntegrityAssessment(
                status=TimingIntegrityStatus.HEALTHY,
                reasons=(),
                islands=(),
            ),
        )
    )
    monkeypatch.setattr(
        s2_timing,
        "assess_timing_integrity",
        lambda local_review, recovery_preflight: next(assessments),
    )
    beat_calls = []

    def analyze_beats(path):
        beat_calls.append(path)
        beats = tuple(range(0, 30_000, 500))
        return BeatGrid(
            beat_ms=beats,
            downbeat_indices=tuple(range(0, len(beats), 4)),
            bpm=120.0,
            beats_per_bar=4,
            bpm_drift_pct=0.0,
            raw_beat_count=len(beats),
            dropped_beat_count=0,
            residual_rms_ms=0.0,
            residual_max_ms=0.0,
        )

    prepared = replace(_prepared(tmp_path, duration_ms=30_000), beat_this_enabled=True)
    generator = RecordingGenerator()
    authority = run_timing_generation(
        prepared,
        _active_analysis(30_000),
        tmp_path,
        generator=generator,
        seed=9,
        beat_analyzer=analyze_beats,
    )

    assert [call.super_timing for call in generator.timing_calls] == [False, True]
    assert beat_calls == [prepared.normalized.path]
    assert authority.mode == "BEAT_THIS_FALLBACK"
    assert authority.candidate_selection is not None
    assert [
        candidate.mode for candidate in authority.candidate_selection.candidates
    ] == ["STANDARD", "SUPER_TIMING", "BEAT_THIS_FALLBACK"]
    assert authority.candidate_selection.reason == "BETTER_TIMING_INTEGRITY"


def test_force_super_timing_skips_standard_and_runs_once(monkeypatch, tmp_path):
    generator = RecordingGenerator()
    monkeypatch.setattr(
        s2_timing,
        "review_timing_authority",
        lambda metrics: TimingAuthorityReview(TimingAuthorityAction.PASS, ()),
    )

    authority = run_timing_generation(
        _prepared(tmp_path, duration_ms=30_000),
        _active_analysis(30_000),
        tmp_path,
        generator=generator,
        seed=9,
        force_super=True,
    )

    assert [call.super_timing for call in generator.timing_calls] == [True]
    assert authority.mode == "SUPER_TIMING"
    assert authority.attempt_count == 1


def test_force_super_timing_failure_does_not_fall_back_to_standard(
    monkeypatch, tmp_path
):
    generator = AlwaysLocalOutlierGenerator()
    monkeypatch.setattr(
        s2_timing,
        "review_timing_authority",
        lambda metrics: TimingAuthorityReview(TimingAuthorityAction.PASS, ()),
    )

    with pytest.raises(WorkerError) as captured:
        run_timing_generation(
            _prepared(tmp_path, duration_ms=30_000),
            _active_analysis(30_000),
            tmp_path,
            generator=generator,
            seed=9,
            force_super=True,
        )

    assert captured.value.code is ErrorCode.CHART_TIMING_CANDIDATE_FAILED
    assert [call.super_timing for call in generator.timing_calls] == [True]
    assert captured.value.context["attempts"][0]["localTimingReview"]["action"] == (
        "RETRY_TIMING"
    )


def test_retry_worthy_standard_accepts_structurally_valid_super_review(tmp_path):
    generator = RetryThenReviewGenerator()

    authority = run_timing_generation(
        _prepared(tmp_path),
        _half_tempo_analysis(),
        tmp_path,
        generator=generator,
        seed=9,
    )

    assert [call.super_timing for call in generator.timing_calls] == [False, True]
    assert authority.mode == "SUPER_TIMING"
    assert authority.attempt_count == 2
    assert authority.review is not None
    assert authority.review.action is TimingAuthorityAction.REVIEW
    assert authority.review.reasons == ("INSUFFICIENT_TEMPO_EVIDENCE",)
    assert authority.reference_path.is_file()


def test_insufficient_standard_review_is_conditionally_accepted_without_super(tmp_path):
    generator = RecordingGenerator()
    silent = OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=np.zeros(101),
        band_strength=np.zeros((3, 101)),
        onset_ms=(),
        n_fft=1,
    )

    authority = run_timing_generation(
        _prepared(tmp_path), silent, tmp_path, generator=generator, seed=9
    )

    assert [call.super_timing for call in generator.timing_calls] == [False]
    assert authority.mode == "STANDARD"
    assert authority.attempt_count == 1
    assert authority.review is not None
    assert authority.review.action is TimingAuthorityAction.REVIEW
    assert authority.review.reasons == ("INSUFFICIENT_TEMPO_EVIDENCE",)
    assert authority.reference_path.is_file()


@pytest.mark.parametrize(
    "reason", ["TEMPO_EVIDENCE_DISAGREES", "WEAK_BASE_TEMPO_SUPPORT"]
)
def test_actionable_standard_review_compares_super_and_keeps_lower_cost_tie(
    monkeypatch, tmp_path, reason
):
    generator = RecordingGenerator()
    reviews = iter(
        (
            TimingAuthorityReview(TimingAuthorityAction.REVIEW, (reason,)),
            TimingAuthorityReview(TimingAuthorityAction.PASS, ()),
        )
    )
    monkeypatch.setattr(s2_timing, "review_timing_authority", lambda metrics: next(reviews))

    authority = run_timing_generation(
        _prepared(tmp_path),
        _base_tempo_analysis(),
        tmp_path,
        generator=generator,
        seed=9,
    )

    assert [call.super_timing for call in generator.timing_calls] == [False, True]
    assert authority.mode == "STANDARD"
    assert authority.attempt_count == 1
    assert authority.candidate_selection is not None
    assert authority.candidate_selection.reason == "LOWER_CANDIDATE_COST"
    assert [
        candidate.mode for candidate in authority.candidate_selection.candidates
    ] == ["STANDARD", "SUPER_TIMING"]


def test_close_standard_super_pair_uses_one_optional_beat_analysis(
    monkeypatch, tmp_path
):
    class ShiftedThenAlignedGenerator(RecordingGenerator):
        def generate_timing(self, request, workdir):
            generated = super().generate_timing(request, workdir)
            return GeneratedTiming(
                osu_text="",
                bpm_events=(OsuBpmEvent(0 if request.super_timing else 200, 120.0),),
                generator_name=generated.generator_name,
                seed=generated.seed,
                mode=generated.mode,
            )

    reviews = iter(
        (
            TimingAuthorityReview(
                TimingAuthorityAction.REVIEW,
                ("TEMPO_EVIDENCE_DISAGREES",),
            ),
            TimingAuthorityReview(TimingAuthorityAction.PASS, ()),
        )
    )
    monkeypatch.setattr(s2_timing, "review_timing_authority", lambda metrics: next(reviews))
    original_build = s2_timing.build_timing_candidate_evidence

    def close_evidence(**kwargs):
        evidence = original_build(**kwargs)
        return replace(
            evidence,
            contradicted_active_ms=0,
            contradicted_ratio=0.0,
            beat_consensus_by_level={"HALF": 0.4, "BASE": 0.4, "DOUBLE": 0.3},
            best_metrical_level="BASE",
        )

    monkeypatch.setattr(s2_timing, "build_timing_candidate_evidence", close_evidence)
    beat_calls = []

    def analyze_beats(path):
        beat_calls.append(path)
        beats = tuple(range(0, 30_000, 1_000))
        return BeatGrid(
            beat_ms=beats,
            downbeat_indices=tuple(range(0, len(beats), 4)),
            bpm=60.0,
            beats_per_bar=4,
            bpm_drift_pct=0.0,
            raw_beat_count=len(beats),
            dropped_beat_count=0,
            residual_rms_ms=0.0,
            residual_max_ms=0.0,
        )

    prepared = replace(_prepared(tmp_path, duration_ms=30_000), beat_this_enabled=True)
    authority = run_timing_generation(
        prepared,
        _active_analysis(30_000),
        tmp_path,
        generator=ShiftedThenAlignedGenerator(),
        seed=9,
        beat_analyzer=analyze_beats,
    )

    assert beat_calls == [prepared.normalized.path]
    assert authority.mode == "SUPER_TIMING"
    assert authority.candidate_selection is not None
    assert authority.candidate_selection.reason == "HIGHER_EXTERNAL_BEAT_F1"
    assert authority.candidate_selection.external_beat_status == "AVAILABLE"


def test_missing_optional_beat_dependency_degrades_to_internal_selection(
    monkeypatch, tmp_path
):
    reviews = iter(
        (
            TimingAuthorityReview(
                TimingAuthorityAction.REVIEW,
                ("TEMPO_EVIDENCE_DISAGREES",),
            ),
            TimingAuthorityReview(TimingAuthorityAction.PASS, ()),
        )
    )
    monkeypatch.setattr(s2_timing, "review_timing_authority", lambda metrics: next(reviews))
    monkeypatch.setattr(
        s2_timing,
        "timing_candidates_need_external_corroboration",
        lambda candidates: True,
    )

    def unavailable(_path):
        raise ModuleNotFoundError("No module named 'beat_this'")

    authority = run_timing_generation(
        replace(_prepared(tmp_path), beat_this_enabled=True),
        _base_tempo_analysis(),
        tmp_path,
        generator=RecordingGenerator(),
        seed=9,
        beat_analyzer=unavailable,
    )

    assert authority.mode == "STANDARD"
    assert authority.candidate_selection is not None
    assert authority.candidate_selection.reason == "LOWER_CANDIDATE_COST"
    assert authority.candidate_selection.external_beat_status == "UNAVAILABLE"
    assert "beat_this" in authority.candidate_selection.external_beat_error


def test_identity_failure_uses_super_timing_and_leaves_no_rejected_reference(tmp_path, monkeypatch):
    generator = IdentityFailureThenSuperGenerator()
    original = s2_timing.validate_timing_identity
    calls = 0

    def fail_standard_identity(actual, expected):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimingAuthorityValidationError("timing authority identity differs")
        original(actual, expected)

    monkeypatch.setattr(s2_timing, "validate_timing_identity", fail_standard_identity)

    authority = run_timing_generation(
        _prepared(tmp_path),
        _base_tempo_analysis(),
        tmp_path,
        generator=generator,
        seed=9,
    )

    assert [call.super_timing for call in generator.timing_calls] == [False, True]
    assert authority.mode == "SUPER_TIMING"
    assert authority.reference_path.is_file()
    assert not list((tmp_path / "audio").glob("timing-reference-*.osu"))


def test_timing_workdir_is_unique_per_authority_epoch(tmp_path):
    """epoch 2 재선택이 epoch 1 의 산출물과 충돌하면 안 된다.

    force_super=True 면 attempt_count 가 항상 1이라, epoch 를 경로에 넣지
    않으면 두 epoch 가 같은 디렉터리를 쓴다. epoch 1 이 남긴 .osu 때문에
    _require_clean_output_dir 이 곡 전체를 실패시켰다 (24곡 배치 song 24).
    """
    prepared = _prepared(tmp_path, duration_ms=150_000)
    analysis = _active_analysis(150_000)

    first = RecordingGenerator()
    run_timing_generation(
        prepared, analysis, tmp_path, generator=first, seed=9, authority_epoch=1
    )
    second = RecordingGenerator()
    run_timing_generation(
        prepared,
        analysis,
        tmp_path,
        generator=second,
        seed=9,
        force_super=True,
        authority_epoch=2,
    )

    used = [*first.timing_workdirs, *second.timing_workdirs]
    assert len(set(used)) == len(used)
    assert all("epoch-1" in workdir.parts for workdir in first.timing_workdirs)
    assert all("epoch-2" in workdir.parts for workdir in second.timing_workdirs)
    # 실제로 만들어진 디렉터리도 겹치지 않는다.
    produced = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in (tmp_path / "timing" / "work").rglob("*")
        if path.is_dir()
    )
    assert "timing/work/epoch-1/standard/attempt-1" in produced
    assert "timing/work/epoch-2/super/attempt-1" in produced
