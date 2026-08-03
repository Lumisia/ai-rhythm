from pathlib import Path

import pytest

from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent, parse_osu_file
from chart_worker.schema.note import NoteEvent
from chart_worker.stages.s2_generate import run_generation
from chart_worker.stages.types import PreparedAudio

SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _prepared(tmp_path: Path) -> PreparedAudio:
    audio_path = tmp_path / "audio" / "game.flac"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"audio")
    return PreparedAudio(
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
            bpm_events=(OsuBpmEvent(0, 120.0),),
        )


def test_run_generation_creates_exactly_twelve_parseable_variants(tmp_path: Path):
    prepared = _prepared(tmp_path)
    generator = RecordingGenerator()
    variants = run_generation(prepared, tmp_path, generator=generator, seed=17)

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
    assert all(workdir.name == "attempt-1" for workdir in workdirs)
    assert all(workdir.parent.parent.name == "work" for workdir in workdirs)
    assert all("candidates" not in workdir.parts for workdir in workdirs)
    assert all(request.duration_ms == 2_000 for request in requests)
    assert all(request.cfg_scale == 1.0 for request in requests)
    assert all(len(request.descriptors) == 1 for request in requests)
    assert [variant.raw_osu_path.name for variant in variants] == [
        f"{key_mode}k-{difficulty.lower()}.osu"
        for key_mode in (4, 6, 7)
        for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
    ]
    assert all(
        parse_osu_file(variant.raw_osu_path).key_mode == variant.key_mode for variant in variants
    )


def test_run_generation_preserves_generator_osu_text(tmp_path: Path):
    prepared = _prepared(tmp_path)

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
                [NoteEvent(500, 0)],
                request.key_mode,
                text,
                "original",
                request.seed,
                (OsuBpmEvent(0, 120.0),),
            )

    generator = OriginalGenerator()
    variants = run_generation(prepared, tmp_path, generator=generator, seed=1)
    assert variants[0].raw_osu_path.read_text(encoding="utf-8") == generator.texts[0]


def test_run_generation_never_writes_invalid_output_to_stable_raw(tmp_path: Path):
    prepared = _prepared(tmp_path)

    class InvalidLaneGenerator(RecordingGenerator):
        def __call__(self, request, workdir):
            generated = super().__call__(request, workdir)
            return GeneratedChart(
                notes=[NoteEvent(500, request.key_mode)],
                key_mode=generated.key_mode,
                osu_text=generated.osu_text,
                generator_name=generated.generator_name,
                seed=generated.seed,
                bpm_events=generated.bpm_events,
            )

    with pytest.raises(WorkerError) as captured:
        run_generation(prepared, tmp_path, generator=InvalidLaneGenerator(), seed=0)
    assert captured.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    assert not (tmp_path / "raw" / "4k-easy.osu").exists()


def test_retries_only_the_failed_variant_with_the_next_seed(tmp_path: Path):
    prepared = _prepared(tmp_path)

    class FirstEasyAttemptInvalid(RecordingGenerator):
        def __call__(self, request, workdir):
            generated = super().__call__(request, workdir)
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
    variants = run_generation(prepared, tmp_path, generator=generator, seed=0)

    calls = [
        (request.key_mode, request.difficulty, request.seed, workdir.name)
        for request, workdir in generator.calls
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


def test_reports_all_errors_when_one_variant_exhausts_its_attempts(tmp_path: Path):
    prepared = _prepared(tmp_path)

    class AlwaysInvalid(RecordingGenerator):
        def __call__(self, request, workdir):
            generated = super().__call__(request, workdir)
            return GeneratedChart(
                notes=[NoteEvent(500, request.key_mode)],
                key_mode=generated.key_mode,
                osu_text=generated.osu_text,
                generator_name=generated.generator_name,
                seed=generated.seed,
                bpm_events=generated.bpm_events,
            )

    generator = AlwaysInvalid()
    with pytest.raises(WorkerError) as captured:
        run_generation(prepared, tmp_path, generator=generator, seed=0)

    assert captured.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    assert captured.value.context["key_mode"] == 4
    assert captured.value.context["difficulty"] == "EASY"
    assert captured.value.context["seeds"] == [0, 12, 24]
    assert len(captured.value.context["errors"]) == 3
    assert [request.seed for request, _ in generator.calls] == [0, 12, 24]
