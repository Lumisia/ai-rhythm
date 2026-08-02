from pathlib import Path

import numpy as np

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.analysis.beat import BeatGrid
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.generation.candidate_selection import CandidateParameters
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import parse_osu_file
from chart_worker.schema.note import NoteEvent
from chart_worker.stages.s2_generate import run_generation, run_generation_variant
from chart_worker.stages.types import AnalysisStageResult
from tests.support import timing_candidate

SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _analysis(tmp_path: Path) -> AnalysisStageResult:
    audio_path = tmp_path / "audio" / "game.flac"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"audio")
    timing_path = tmp_path / "analysis" / "timing.osu"
    timing_path.parent.mkdir(parents=True)
    timing_path.write_text("timing", encoding="utf-8")
    return AnalysisStageResult(
        normalized=NormalizedAudio(
            audio_path,
            "audio-profile-v1",
            SHA,
            2_000,
            48_000,
            2,
            2_000,
            0,
            0.0,
            -14.0,
            -1.0,
            0.0,
            "LOUDNESS",
        ),
        signal=AudioSignal(np.zeros((96_000, 2)), 48_000),
        beat_grid=BeatGrid((0, 500, 1_000, 1_500), (0, 2), 120.0, 4, 0.0, 4, 0, 0.0, 0.0),
        onsets=OnsetAnalysis(48_000, 512, np.ones(2), np.ones((3, 2)), (500,)),
        timing_candidate=timing_candidate(),
        timing_osu_path=timing_path,
        timing_quality_report_path=tmp_path / "analysis" / "timing-quality-v1.json",
    )


class RecordingGenerator:
    def __init__(self):
        self.calls = []

    def __call__(self, request, workdir):
        self.calls.append((request, workdir))
        return GeneratedChart(
            notes=[NoteEvent(500, request.key_mode - 1)],
            key_mode=request.key_mode,
            osu_text="",
            generator_name="recording-fake",
            seed=request.seed,
        )


def test_run_generation_creates_exactly_twelve_parseable_variants(tmp_path: Path):
    analysis = _analysis(tmp_path)
    generator = RecordingGenerator()
    variants = run_generation(analysis, tmp_path, generator=generator, seed=17)

    assert {(variant.key_mode, variant.difficulty) for variant in variants} == {
        (key_mode, difficulty)
        for key_mode in (4, 6, 7)
        for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
    }
    requests = [request for request, _ in generator.calls]
    workdirs = [workdir for _, workdir in generator.calls]
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
    assert all("candidates" in workdir.parts for workdir in workdirs)
    assert all("attempt-1" in workdir.parts for workdir in workdirs)
    assert all(request.timing_osu_path == analysis.timing_osu_path for request in requests)
    assert all(request.duration_ms == 2_000 for request in requests)
    assert all(
        parse_osu_file(variant.raw_osu_path).key_mode == variant.key_mode for variant in variants
    )


def test_run_generation_preserves_generator_osu_text(tmp_path: Path):
    analysis = _analysis(tmp_path)

    class OriginalGenerator:
        def __init__(self):
            self.texts = []

        def __call__(self, request, workdir):
            text = (
                "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\n"
                f"CircleSize:{request.key_mode}\n\n[TimingPoints]\n"
                "0,500,4,2,0,60,1,0\n\n[HitObjects]\n64,192,500,1,0,0:0:0:0:\n"
            )
            self.texts.append(text)
            return GeneratedChart(
                [NoteEvent(500, 0)], request.key_mode, text, "original", request.seed
            )

    generator = OriginalGenerator()
    variants = run_generation(analysis, tmp_path, generator=generator, seed=1)
    assert variants[0].raw_osu_path.read_text(encoding="utf-8") == generator.texts[0]


def test_generation_attempts_use_unique_raw_and_work_directories(tmp_path: Path):
    analysis = _analysis(tmp_path)
    generator = RecordingGenerator()

    first = run_generation_variant(
        analysis,
        tmp_path,
        generator=generator,
        key_mode=4,
        difficulty="NORMAL",
        attempt=1,
        parameters=CandidateParameters(seed=19, requested_star=3.0, cfg_scale=1.0),
    )
    second = run_generation_variant(
        analysis,
        tmp_path,
        generator=generator,
        key_mode=4,
        difficulty="NORMAL",
        attempt=2,
        parameters=CandidateParameters(seed=10_019, requested_star=3.0, cfg_scale=1.0),
    )

    assert first.raw_osu_path != second.raw_osu_path
    assert first.raw_osu_path.parent.name == "attempt-1"
    assert second.raw_osu_path.parent.name == "attempt-2"
    assert generator.calls[0][1] != generator.calls[1][1]
    assert first.raw_osu_path.is_file()
    assert second.raw_osu_path.is_file()
