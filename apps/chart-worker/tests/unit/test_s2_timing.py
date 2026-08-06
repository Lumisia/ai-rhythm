from pathlib import Path

import numpy as np
import pytest

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.mapperatorinator import GeneratedTiming
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.hashing import sha256_file
from chart_worker.stages import s2_timing
from chart_worker.stages.s2_timing import run_timing_generation
from chart_worker.stages.types import PreparedAudio
from chart_worker.validation.timing_authority import TimingAuthorityValidationError
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
            profile_version="audio-profile-v1",
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


class RecordingGenerator:
    def __init__(self) -> None:
        self.timing_calls = []

    def generate_timing(self, request, workdir):
        self.timing_calls.append(request)
        return GeneratedTiming(
            osu_text="",
            bpm_events=(OsuBpmEvent(0, 120.0),),
            generator_name="recording-generator",
            seed=request.seed,
            mode="SUPER_TIMING" if request.super_timing else "STANDARD",
        )


class InvalidThenSuperGenerator(RecordingGenerator):
    def generate_timing(self, request, workdir):
        self.timing_calls.append(request)
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
        return GeneratedTiming(
            osu_text="",
            bpm_events=(OsuBpmEvent(2_678, 120.0),),
            generator_name="recording-generator",
            seed=request.seed,
            mode="SUPER_TIMING" if request.super_timing else "STANDARD",
        )


class AlwaysLongActiveGapGenerator(RecordingGenerator):
    def generate_timing(self, request, workdir):
        self.timing_calls.append(request)
        return GeneratedTiming(
            osu_text="",
            bpm_events=(OsuBpmEvent(95_645, 120.0),),
            generator_name="recording-generator",
            seed=request.seed,
            mode="SUPER_TIMING" if request.super_timing else "STANDARD",
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


def test_short_active_intro_is_promoted_without_event_mutation(tmp_path):
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
def test_actionable_standard_review_uses_super_once(monkeypatch, tmp_path, reason):
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
    assert authority.mode == "SUPER_TIMING"
    assert authority.attempt_count == 2


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
