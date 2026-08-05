from pathlib import Path

import numpy as np
import pytest

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


def _prepared(tmp_path: Path) -> PreparedAudio:
    audio_path = tmp_path / "audio" / "game.flac"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"canonical-audio")
    return PreparedAudio(
        normalized=NormalizedAudio(
            path=audio_path,
            profile_version="audio-profile-v1",
            sha256=sha256_file(audio_path),
            duration_ms=2_000,
            sample_rate_hz=48_000,
            channels=2,
            source_duration_ms=2_000,
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


class PositiveFirstThenSuperGenerator(RecordingGenerator):
    def generate_timing(self, request, workdir):
        self.timing_calls.append(request)
        return GeneratedTiming(
            osu_text="",
            bpm_events=(OsuBpmEvent(250 if request.super_timing else 501, 120.0),),
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


def test_standard_first_event_beyond_one_beat_uses_super_timing_once(tmp_path):
    generator = PositiveFirstThenSuperGenerator()

    authority = run_timing_generation(
        _prepared(tmp_path),
        _base_tempo_analysis(offset_ms=250),
        tmp_path,
        generator=generator,
        seed=9,
    )

    assert [call.super_timing for call in generator.timing_calls] == [False, True]
    assert authority.mode == "SUPER_TIMING"
    assert authority.attempt_count == 2


def test_retry_worthy_standard_runs_super_once_then_blocks_map_generation(tmp_path):
    generator = RetryThenReviewGenerator()

    with pytest.raises(WorkerError) as captured:
        run_timing_generation(
            _prepared(tmp_path),
            _half_tempo_analysis(),
            tmp_path,
            generator=generator,
            seed=9,
        )

    assert captured.value.code is ErrorCode.CHART_TIMING_REVIEW_REQUIRED
    assert [call.super_timing for call in generator.timing_calls] == [False, True]
    assert captured.value.context == {
        "reasons": ("INSUFFICIENT_TEMPO_EVIDENCE",),
        "attempt_count": 2,
        "attempts": [
            {
                "attempt": 1,
                "seed": 9,
                "mode": "STANDARD",
                "workdir": "timing/work/attempt-1",
                "review": {
                    "action": "RETRY_TIMING",
                    "reasons": ["STRONG_HALF_TEMPO_ALTERNATIVE"],
                },
                "tempoMetrics": {
                    "basePulseSupport": 0.5,
                    "halfPulseSupport": 1.0,
                    "doublePulseSupport": 0.25,
                    "baseSupportedPulses": 40,
                    "halfSupportedPulses": 40,
                    "doubleSupportedPulses": 40,
                    "pulseBestAlternative": "HALF",
                    "pulseAlternativeMargin": 0.5,
                    "basePeriodicitySupport": 0.0,
                    "halfPeriodicitySupport": 1.0,
                    "doublePeriodicitySupport": 0.0,
                    "periodicityFrameCount": 400,
                    "periodicityBestAlternative": "HALF",
                    "periodicityMargin": 1.0,
                    "evidenceAgrees": True,
                    "evidenceStatus": "SUFFICIENT",
                },
            },
            {
                "attempt": 2,
                "seed": 9,
                "mode": "SUPER_TIMING",
                "workdir": "timing/work/attempt-2",
                "review": {
                    "action": "REVIEW",
                    "reasons": ["INSUFFICIENT_TEMPO_EVIDENCE"],
                },
                "tempoMetrics": {
                    "basePulseSupport": 0.0,
                    "halfPulseSupport": 0.0,
                    "doublePulseSupport": 0.24528301886792453,
                    "baseSupportedPulses": 0,
                    "halfSupportedPulses": 0,
                    "doubleSupportedPulses": 39,
                    "pulseBestAlternative": "DOUBLE",
                    "pulseAlternativeMargin": 0.24528301886792453,
                    "basePeriodicitySupport": 0.0,
                    "halfPeriodicitySupport": 1.0,
                    "doublePeriodicitySupport": 0.0,
                    "periodicityFrameCount": 397,
                    "periodicityBestAlternative": "HALF",
                    "periodicityMargin": 1.0,
                    "evidenceAgrees": False,
                    "evidenceStatus": "INSUFFICIENT",
                },
            },
        ],
    }
    assert not (tmp_path / "audio" / "timing-reference.osu").exists()


def test_standard_review_blocks_map_generation_without_a_super_retry(tmp_path):
    generator = RecordingGenerator()
    silent = OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=np.zeros(101),
        band_strength=np.zeros((3, 101)),
        onset_ms=(),
        n_fft=1,
    )

    with pytest.raises(WorkerError) as captured:
        run_timing_generation(_prepared(tmp_path), silent, tmp_path, generator=generator, seed=9)

    assert captured.value.code is ErrorCode.CHART_TIMING_REVIEW_REQUIRED
    assert [call.super_timing for call in generator.timing_calls] == [False]
    assert captured.value.context["attempts"] == [
        {
            "attempt": 1,
            "seed": 9,
            "mode": "STANDARD",
            "workdir": "timing/work/attempt-1",
            "review": {
                "action": "REVIEW",
                "reasons": ["INSUFFICIENT_TEMPO_EVIDENCE"],
            },
            "tempoMetrics": {
                "basePulseSupport": 0.0,
                "halfPulseSupport": 0.0,
                "doublePulseSupport": 0.0,
                "baseSupportedPulses": 0,
                "halfSupportedPulses": 0,
                "doubleSupportedPulses": 0,
                "pulseBestAlternative": None,
                "pulseAlternativeMargin": 0.0,
                "basePeriodicitySupport": 0.0,
                "halfPeriodicitySupport": 0.0,
                "doublePeriodicitySupport": 0.0,
                "periodicityFrameCount": 0,
                "periodicityBestAlternative": None,
                "periodicityMargin": 0.0,
                "evidenceAgrees": False,
                "evidenceStatus": "INSUFFICIENT",
            },
        }
    ]
    assert not (tmp_path / "audio" / "timing-reference.osu").exists()


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
