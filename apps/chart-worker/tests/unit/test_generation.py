from pathlib import Path

import pytest

from chart_worker.audio.runner import CommandError, CommandResult
from chart_worker.config import WorkerConfig
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.fake import FakeGenerator, synthesize_chart
from chart_worker.generation.mapperatorinator import (
    MapperatorinatorGenerator,
    build_map_command,
    build_timing_command,
    find_generated_osu,
    inference_env,
)
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.generation.params import (
    DESCRIPTORS,
    REQUESTED_STAR,
    GenerationRequest,
    TimingGenerationRequest,
)

DURATION_MS = 20_000
MINI_OSU = (
    "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:4\n"
    "\n[TimingPoints]\n0,500,4,2,0,60,1,0\n"
    "\n[HitObjects]\n64,192,1000,1,0,0:0:0:0:\n"
    "192,192,1200,1,0,0:0:0:0:\n"
)
TIMING_OSU = (
    "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:4\n"
    "\n[TimingPoints]\n0,500,4,2,0,60,1,0\n\n[HitObjects]\n"
)


@pytest.fixture
def config():
    return WorkerConfig(
        mapperatorinator_python=Path("C:/mapp/.venv/Scripts/python.exe"),
        mapperatorinator_home=Path("C:/mapp"),
    )


def _request(**overrides):
    values = {
        "audio_path": Path("game.flac"),
        "key_mode": 4,
        "difficulty": "NORMAL",
        "duration_ms": DURATION_MS,
    }
    return GenerationRequest(**(values | overrides))


def _pairs(argv):
    return dict(item.split("=", 1) for item in argv if "=" in item)


def _fake_run(osu_text=MINI_OSU, fail=False):
    def run(argv):
        if fail:
            raise CommandError(argv, "exited with 1", returncode=1, stderr="cuda oom")
        output_dir = Path(
            next(item for item in argv if item.startswith("output_path=")).split("=", 1)[1]
            .strip("'")
        )
        (output_dir / "out.osu").write_text(osu_text, encoding="utf-8")
        return CommandResult(argv, 0, "", "")

    return run


def test_requested_stars_and_descriptors_are_explicit():
    assert REQUESTED_STAR == {
        "EASY": 1.0,
        "NORMAL": 1.5,
        "HARD": 2.0,
        "EXPERT": 2.75,
    }
    assert DESCRIPTORS == {
        "EASY": ("expression/simple",),
        "NORMAL": ("style/mixed rice",),
        "HARD": ("style/mixed rice",),
        "EXPERT": ("style/mixed rice",),
    }


def test_direct_request_has_no_hybrid_guidance_fields():
    fields = GenerationRequest.__dataclass_fields__
    assert "timing_osu_path" not in fields
    assert "hold_note_ratio" not in fields
    assert "negative_descriptors" not in fields
    assert _request().cfg_scale == 1.0


def test_direct_request_rejects_classifier_free_guidance():
    with pytest.raises(ValueError, match="cfg_scale=1.0"):
        _request(cfg_scale=1.25)


def test_timing_command_generates_only_timing(config, tmp_path):
    request = TimingGenerationRequest(
        audio_path=Path("game.flac"), duration_ms=20_000, seed=7
    )
    pairs = _pairs(build_timing_command(config, request, tmp_path))
    assert pairs["output_type"] == "[TIMING]"
    assert pairs["super_timing"] == "false"
    assert "beatmap_path" not in pairs
    assert "in_context" not in pairs


def test_map_command_reuses_the_timing_reference(config, tmp_path):
    timing = tmp_path / "audio" / "timing-reference.osu"
    request = _request(timing_reference_path=timing)
    pairs = _pairs(build_map_command(config, request, tmp_path / "map"))
    assert pairs["output_type"] == "[MAP]"
    assert pairs["beatmap_path"] == f"'{timing.resolve()}'"
    assert pairs["in_context"] == "[TIMING]"
    assert "TIMING" not in pairs["output_type"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"key_mode": 5},
        {"difficulty": "NOMAL"},
        {"year": 2030},
        {"duration_ms": 0},
    ],
)
def test_invalid_requests_are_rejected(overrides):
    with pytest.raises(ValueError):
        _request(**overrides)


def test_map_command_requests_only_a_map_generation(config, tmp_path):
    pairs = _pairs(build_map_command(config, _request(difficulty="EXPERT"), tmp_path))
    assert pairs == {
        "hydra.run.dir": f"'{(tmp_path / '.hydra-run').resolve()}'",
        "audio_path": f"'{Path('game.flac').resolve()}'",
        "output_path": f"'{tmp_path.resolve()}'",
        "gamemode": "3",
        "keycount": "4",
        "difficulty": "2.75",
        "year": "2023",
        "end_time": str(DURATION_MS),
        "cfg_scale": "1.0",
        "descriptors": "[style/mixed rice]",
        "precision": "fp16",
        "export_osz": "false",
        "beatmap_path": f"'{Path('timing-reference.osu').resolve()}'",
        "in_context": "[TIMING]",
        "output_type": "[MAP]",
        "super_timing": "false",
        "hitsounded": "false",
        "fast_decoder_loop": "true",
        "use_server": "false",
        "resnap_events": "true",
    }


def test_command_uses_configured_bf16_for_modal_capable_gpu(config, tmp_path):
    bf16 = config.model_copy(update={"mapperatorinator_precision": "bf16"})
    assert _pairs(build_map_command(bf16, _request(), tmp_path))["precision"] == "bf16"


def test_command_requires_a_configured_mapperatorinator(tmp_path):
    with pytest.raises(ValueError, match="must be configured"):
        build_map_command(WorkerConfig(), _request(), tmp_path)


def test_seed_is_optional(config, tmp_path):
    assert "seed" not in _pairs(build_map_command(config, _request(), tmp_path))
    assert _pairs(build_map_command(config, _request(seed=7), tmp_path))["seed"] == "7"


def test_descriptors_with_hydra_list_syntax_are_rejected(config, tmp_path):
    with pytest.raises(ValueError, match="list syntax"):
        build_map_command(config, _request(descriptors=("style/a,b",)), tmp_path)


def test_find_generated_osu_requires_exactly_one_file(tmp_path):
    with pytest.raises(WorkerError) as missing:
        find_generated_osu(tmp_path)
    assert missing.value.code is ErrorCode.CHART_GENERATION_FAILED
    (tmp_path / "a.osu").write_text("a", encoding="utf-8")
    assert find_generated_osu(tmp_path).name == "a.osu"
    (tmp_path / "b.osu").write_text("b", encoding="utf-8")
    with pytest.raises(WorkerError, match="exactly one"):
        find_generated_osu(tmp_path)


def test_generator_generates_map_and_parses_raw_timing(config, tmp_path):
    result = MapperatorinatorGenerator(
        config=config, run=_fake_run(), verify_patch=lambda _home: None
    ).generate_map(_request(seed=3), tmp_path / "work")
    assert [note.time_ms for note in result.notes] == [1000, 1200]
    assert result.key_mode == 4
    assert result.generator_name == "mapperatorinator-v32"
    assert result.seed == 3
    assert [(event.time_ms, event.bpm) for event in result.bpm_events] == [(0, 120.0)]


def test_generator_maps_subprocess_failure(config, tmp_path):
    generator = MapperatorinatorGenerator(
        config=config, run=_fake_run(fail=True), verify_patch=lambda _home: None
    )
    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")
    assert caught.value.code is ErrorCode.CHART_GENERATION_FAILED
    assert "cuda oom" in caught.value.context["stderr"]


def test_generator_maps_parse_failure(config, tmp_path):
    generator = MapperatorinatorGenerator(
        config=config,
        run=_fake_run(osu_text="garbage"),
        verify_patch=lambda _home: None,
    )
    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")
    assert caught.value.code is ErrorCode.CHART_OSU_PARSE_FAILED


def test_generator_rejects_a_mismatched_keycount(config, tmp_path):
    generator = MapperatorinatorGenerator(
        config=config, run=_fake_run(), verify_patch=lambda _home: None
    )
    with pytest.raises(WorkerError, match="asked for 7K"):
        generator.generate_map(_request(key_mode=7), tmp_path / "work")


def test_fake_generator_is_gpu_free_and_deterministic():
    first = synthesize_chart(_request(seed=42))
    second = synthesize_chart(_request(seed=42))
    assert [(note.time_ms, note.lane, note.kind) for note in first] == [
        (note.time_ms, note.lane, note.kind) for note in second
    ]
    result = FakeGenerator().generate_map(_request(seed=42), Path("unused"))
    assert result.bpm_events[0].bpm == 120.0


def test_every_mapperatorinator_path_argument_is_absolute(config, tmp_path):
    argv = build_map_command(
        config,
        _request(audio_path=Path("storage/game.flac")),
        tmp_path / "out",
    )
    pairs = _pairs(argv)
    assert Path(pairs["audio_path"].strip("'")).is_absolute()
    assert Path(pairs["output_path"].strip("'")).is_absolute()
    assert Path(argv[0]).is_absolute()


def test_paths_are_quoted_for_hydra_even_when_they_contain_spaces(config):
    audio_path = Path("C:/Audio Files/Koe no Yukue (Take 2).wav")
    output_path = Path("C:/Output/generated charts")
    pairs = _pairs(
        build_map_command(config, _request(audio_path=audio_path), output_path)
    )
    assert pairs["audio_path"] == f"'{audio_path.resolve()}'"
    assert pairs["output_path"] == f"'{output_path.resolve()}'"
    assert pairs["hydra.run.dir"] == f"'{(output_path / '.hydra-run').resolve()}'"


def test_a_bare_interpreter_name_is_left_for_path_lookup():
    config = WorkerConfig(
        mapperatorinator_python=Path("python"),
        mapperatorinator_home=Path("C:/mapp"),
    )
    assert build_map_command(config, _request(), Path("out"))[0] == "python"


def test_mapperatorinator_child_process_forces_utf8_without_losing_path():
    env = inference_env({"Path": "C:/tools", "pythonutf8": "0"})
    assert env["Path"] == "C:/tools"
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert sum(name.upper() == "PYTHONUTF8" for name in env) == 1


def test_generator_verifies_constraint_patch_before_inference(config, tmp_path):
    calls = []
    generator = MapperatorinatorGenerator(
        config=config,
        run=_fake_run(),
        verify_patch=lambda home: calls.append(home),
    )

    generator.generate_map(_request(), tmp_path / "work")
    assert calls == [config.mapperatorinator_home]


def test_generator_generates_timing_without_hit_objects(config, tmp_path):
    result = MapperatorinatorGenerator(
        config=config,
        run=_fake_run(osu_text=TIMING_OSU),
        verify_patch=lambda _home: None,
    ).generate_timing(
        TimingGenerationRequest(audio_path=Path("game.flac"), duration_ms=DURATION_MS, seed=3),
        tmp_path / "work",
    )
    assert result.bpm_events == (OsuBpmEvent(time_ms=0, bpm=120.0),)
    assert result.mode == "STANDARD"


def test_fake_generator_generates_standard_timing():
    result = FakeGenerator().generate_timing(
        TimingGenerationRequest(audio_path=Path("game.flac"), duration_ms=DURATION_MS),
        Path("unused"),
    )
    assert [(event.time_ms, event.bpm) for event in result.bpm_events] == [(0, 120.0)]
    assert result.mode == "STANDARD"
