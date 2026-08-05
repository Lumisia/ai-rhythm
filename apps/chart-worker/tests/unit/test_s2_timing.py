from pathlib import Path

from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.generation.mapperatorinator import GeneratedTiming
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.hashing import sha256_file
from chart_worker.stages.s2_timing import run_timing_generation
from chart_worker.stages.types import PreparedAudio


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


def test_generates_one_standard_timing_and_promotes_it_beside_audio(tmp_path):
    generator = RecordingGenerator()
    authority = run_timing_generation(_prepared(tmp_path), tmp_path, generator=generator, seed=9)

    assert [call.super_timing for call in generator.timing_calls] == [False]
    assert authority.reference_path == tmp_path / "audio" / "timing-reference.osu"
    assert authority.sha256 == sha256_file(authority.reference_path)
    assert authority.audio_sha256 == _prepared_sha(tmp_path)
    assert authority.mode == "STANDARD"


def test_structural_standard_failure_uses_super_timing_once(tmp_path):
    generator = InvalidThenSuperGenerator()
    authority = run_timing_generation(_prepared(tmp_path), tmp_path, generator=generator, seed=9)

    assert [call.super_timing for call in generator.timing_calls] == [False, True]
    assert authority.mode == "SUPER_TIMING"
    assert authority.attempt_count == 2


def test_standard_first_event_beyond_one_beat_uses_super_timing_once(tmp_path):
    generator = PositiveFirstThenSuperGenerator()

    authority = run_timing_generation(_prepared(tmp_path), tmp_path, generator=generator, seed=9)

    assert [call.super_timing for call in generator.timing_calls] == [False, True]
    assert authority.mode == "SUPER_TIMING"
    assert authority.attempt_count == 2
