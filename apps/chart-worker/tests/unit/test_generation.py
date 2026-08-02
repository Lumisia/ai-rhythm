from pathlib import Path

import numpy as np
import pytest

from chart_worker.analysis.beat import build_beat_grid
from chart_worker.audio.runner import CommandError, CommandResult
from chart_worker.config import WorkerConfig
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.fake import FakeGenerator, synthesize_chart
from chart_worker.generation.mapperatorinator import (
    MapperatorinatorGenerator,
    MapperatorinatorTimingGenerator,
    build_command,
    build_super_timing_command,
    find_generated_osu,
    inference_env,
)
from chart_worker.generation.params import (
    DESCRIPTORS,
    NEGATIVE_DESCRIPTORS,
    PRECISION,
    REQUESTED_STAR,
    GenerationRequest,
)
from chart_worker.generation.timing_osu import beat_grid_to_timing_osu, timing_points_to_osu
from chart_worker.rating.project_rating import TARGET_RATING

BPM = 120.0
DURATION_MS = 20_000


def _grid(beats=32, bpm=BPM):
    times = np.arange(beats) * (60.0 / bpm)
    return build_beat_grid(times, times[::4])


@pytest.fixture
def timing_osu(tmp_path):
    path = tmp_path / "timing.osu"
    path.write_text(beat_grid_to_timing_osu(_grid(), audio_filename="game.flac"), encoding="utf-8")
    return path


@pytest.fixture
def config():
    return WorkerConfig(
        chart_generator="mapperatorinator",
        mapperatorinator_python=Path("C:/mapp/.venv/Scripts/python.exe"),
        mapperatorinator_home=Path("C:/mapp"),
        ffmpeg_shared_bin_dir=None,
    )


def _request(timing_osu=None, **overrides):
    base = {
        "audio_path": Path("game.flac"),
        "key_mode": 4,
        "difficulty": "NORMAL",
        "duration_ms": DURATION_MS,
        "timing_osu_path": timing_osu,
    }
    return GenerationRequest(**(base | overrides))


# --- 파라미터 ---------------------------------------------------------------


def test_requested_star_is_never_below_the_solver_target():
    """solver 는 내리는 방향으로만 동작한다. 재료가 남아 있어야 한다."""
    for difficulty, target in TARGET_RATING.items():
        assert REQUESTED_STAR[difficulty] >= target, difficulty


@pytest.mark.parametrize("difficulty", ["EASY", "NORMAL", "HARD"])
def test_lower_tiers_have_explicit_solver_headroom(difficulty):
    assert REQUESTED_STAR[difficulty] > TARGET_RATING[difficulty]


def test_expert_has_no_headroom_from_the_request_alone():
    """설계 표에서 EXPERT 만 요청값과 목표가 같다(둘 다 5.0).

    여유가 "요청값 > 목표" 가 아니라 "실제 생성 결과가 요청값보다 한 티어
    높게 나온다"는 실측에서만 나온다. 그 관측이 무너지면 EXPERT 는 solver
    가 손댈 재료가 없다. 표본이 2곡뿐이라 곡을 늘리며 확인해야 한다.
    """
    assert REQUESTED_STAR["EXPERT"] == TARGET_RATING["EXPERT"] == 5.0


def test_defaults_are_filled_from_the_difficulty_tables():
    request = _request()
    assert request.requested_star == REQUESTED_STAR["NORMAL"]
    assert request.descriptors == DESCRIPTORS["NORMAL"]
    assert request.negative_descriptors == ()
    assert request.hold_note_ratio > 0


def test_guided_candidate_uses_real_negative_descriptors():
    request = _request(cfg_scale=1.25)
    assert request.cfg_scale == 1.25
    assert request.negative_descriptors == NEGATIVE_DESCRIPTORS["NORMAL"]


def test_no_descriptor_is_the_nonexistent_clean_tag():
    for table in (DESCRIPTORS, NEGATIVE_DESCRIPTORS):
        for tags in table.values():
            assert "style/clean" not in tags


@pytest.mark.parametrize("overrides", [{"key_mode": 5}, {"difficulty": "NOMAL"}, {"year": 2030}])
def test_invalid_requests_are_rejected(overrides):
    with pytest.raises(ValueError):
        _request(**overrides)


def test_guidance_requires_matching_descriptor_counts():
    """개수가 다르면 classifier-free guidance 가 조용히 어긋난다."""
    with pytest.raises(ValueError, match="same number of descriptors"):
        _request(cfg_scale=5.0, descriptors=("a", "b"), negative_descriptors=("c",))


def test_matching_counts_pass_with_guidance():
    request = _request(cfg_scale=5.0, descriptors=("a", "b"), negative_descriptors=("c", "d"))
    assert request.cfg_scale == 5.0


# --- 타이밍 .osu -------------------------------------------------------------


def test_timing_osu_carries_piecewise_uninherited_points(timing_osu):
    from chart_worker.generation.osu_parser import parse_osu_file

    beatmap = parse_osu_file(timing_osu)
    assert beatmap.key_mode == 4
    assert beatmap.notes == []
    assert len(beatmap.bpm_events) == 1
    assert beatmap.bpm_events[0].bpm == pytest.approx(BPM, abs=0.01)


def test_timing_osu_contains_fields_required_by_mapperatorinator_slider(timing_osu):
    text = timing_osu.read_text(encoding="utf-8")
    assert "Creator:ai-rhythm" in text
    assert "HPDrainRate:5" in text


def test_timing_osu_starts_at_the_first_downbeat():
    grid = build_beat_grid(np.arange(16) * 0.5 + 1.0, (np.arange(16) * 0.5 + 1.0)[::4])
    text = beat_grid_to_timing_osu(grid, audio_filename="a.flac")
    offset = text.split("[TimingPoints]\n")[1].split(",")[0]
    assert int(offset) == grid.beat_ms[0]


def test_timing_osu_uses_a_downbeat_when_beats_precede_the_first_downbeat():
    beats = np.arange(20) * 0.5 + 0.12
    grid = build_beat_grid(beats, beats[2::4])

    text = beat_grid_to_timing_osu(grid, audio_filename="a.flac")

    offset = text.split("[TimingPoints]\n")[1].split(",")[0]
    assert int(offset) == grid.beat_ms[2]


def test_timing_points_osu_keeps_each_measured_downbeat_offset():
    from chart_worker.analysis.timing import TimingPoint

    text = timing_points_to_osu(
        (
            TimingPoint(time_ms=120, bpm=120.0, meter=4, start_beat_index=0),
            TimingPoint(time_ms=16_120, bpm=100.0, meter=4, start_beat_index=32),
        ),
        audio_filename="a.flac",
        title="song",
    )

    timing_lines = text.split("[TimingPoints]\n")[1].split("\n\n")[0].splitlines()
    assert timing_lines == ["120,500.000000,4,2,0,60,1,0", "16120,600.000000,4,2,0,60,1,0"]


def test_timing_osu_needs_a_grid():
    from chart_worker.analysis.beat import BeatGrid

    empty = BeatGrid((), (), 0.0, None, 0.0, 0, 0, 0.0, 0.0)
    with pytest.raises(ValueError, match="no beats"):
        beat_grid_to_timing_osu(empty, audio_filename="a.flac")


# --- 명령행 ------------------------------------------------------------------


def _pairs(argv):
    return dict(item.split("=", 1) for item in argv if "=" in item)


def test_command_pins_precision_over_the_config_default(config, timing_osu, tmp_path):
    """v32.yaml 기본값은 bf16 이고 Turing 은 bf16 을 지원하지 않는다."""
    argv = build_command(config, _request(timing_osu), tmp_path)
    assert _pairs(argv)["precision"] == PRECISION


def test_super_timing_command_uses_twenty_pass_mode_without_a_reference_map(config, tmp_path):
    """Super timing must ask V32 for timing only, not reuse Beat This timing."""
    pairs = _pairs(build_super_timing_command(config, Path("game.flac"), tmp_path))

    assert pairs["super_timing"] == "true"
    assert pairs["output_type"] == "[TIMING]"
    assert pairs["in_context"] == "[NONE]"
    assert pairs["gamemode"] == "3"
    assert pairs["precision"] == PRECISION
    assert pairs["export_osz"] == "false"
    assert "beatmap_path" not in pairs


def test_super_timing_command_uses_absolute_audio_and_output_paths(config):
    pairs = _pairs(build_super_timing_command(config, Path("storage/game.flac"), Path("out/timing")))

    assert Path(pairs["audio_path"]).is_absolute()
    assert Path(pairs["output_path"]).is_absolute()


def test_command_selects_mania_and_the_requested_keycount(config, timing_osu, tmp_path):
    argv = build_command(config, _request(timing_osu, key_mode=7), tmp_path)
    pairs = _pairs(argv)
    assert pairs["gamemode"] == "3"
    assert pairs["keycount"] == "7"
    assert pairs["difficulty"] == str(REQUESTED_STAR["NORMAL"])
    assert pairs["year"] == "2023"


def test_timing_context_needs_a_reference_beatmap(config, timing_osu, tmp_path):
    with_timing = build_command(config, _request(timing_osu), tmp_path)
    without = build_command(config, _request(None), tmp_path)
    assert "in_context=[TIMING]" in with_timing
    assert f"beatmap_path={timing_osu}" in with_timing
    assert not [item for item in without if item.startswith("in_context")]


def test_descriptors_are_written_as_hydra_lists(config, timing_osu, tmp_path):
    argv = build_command(
        config,
        _request(timing_osu, difficulty="HARD", cfg_scale=1.25),
        tmp_path,
    )
    pairs = _pairs(argv)
    assert pairs["descriptors"] == "[style/jumpstream,style/handstream,style/LN coordination]"
    assert pairs["negative_descriptors"].startswith("[style/dump,")


def test_unguided_candidate_omits_negative_descriptors_from_hydra(
    config, timing_osu, tmp_path
):
    pairs = _pairs(build_command(config, _request(timing_osu, cfg_scale=1.0), tmp_path))
    assert pairs["negative_descriptors"] == "[]"


def test_seed_is_omitted_when_unset(config, timing_osu, tmp_path):
    assert not [
        item
        for item in build_command(config, _request(timing_osu), tmp_path)
        if item.startswith("seed=")
    ]
    seeded = build_command(config, _request(timing_osu, seed=7), tmp_path)
    assert "seed=7" in seeded


def test_command_requires_a_configured_mapperatorinator(timing_osu, tmp_path):
    bare = WorkerConfig(mapperatorinator_python=None, mapperatorinator_home=None)
    with pytest.raises(ValueError, match="must be configured"):
        build_command(bare, _request(timing_osu), tmp_path)


def test_descriptors_with_list_syntax_are_rejected(config, timing_osu, tmp_path):
    request = _request(timing_osu, descriptors=("style/a,b",))
    with pytest.raises(ValueError, match="list syntax"):
        build_command(config, request, tmp_path)


# --- 산출물 수거 -------------------------------------------------------------


def test_exactly_one_osu_is_expected(tmp_path):
    (tmp_path / "a.osu").write_text("x")
    assert find_generated_osu(tmp_path).name == "a.osu"


def test_missing_output_is_a_generation_failure(tmp_path):
    with pytest.raises(WorkerError) as caught:
        find_generated_osu(tmp_path)
    assert caught.value.code is ErrorCode.CHART_GENERATION_FAILED


def test_several_osu_files_are_a_generation_failure(tmp_path):
    """조용히 하나를 고르면 이전 실행 결과를 채보로 내보내게 된다."""
    (tmp_path / "a.osu").write_text("x")
    (tmp_path / "b.osu").write_text("y")
    with pytest.raises(WorkerError, match="exactly one"):
        find_generated_osu(tmp_path)


def test_generator_rejects_a_preexisting_osu(config, timing_osu, tmp_path):
    """이전 실행 파일 하나를 이번 실행의 결과로 오인하면 안 된다."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "old.osu").write_text(MINI_OSU, encoding="utf-8")

    generator = MapperatorinatorGenerator(
        config=config,
        run=lambda argv: CommandResult(argv, 0, "", ""),
    )
    with pytest.raises(WorkerError, match="already contains") as caught:
        generator(_request(timing_osu), workdir)
    assert caught.value.code is ErrorCode.CHART_GENERATION_FAILED


def test_super_timing_generator_returns_timing_osu_text_without_mania_parsing(config, tmp_path):
    timing_only_osu = "osu file format v14\n\n[TimingPoints]\n120,731.707317,4,2,0,60,1,0\n"

    result = MapperatorinatorTimingGenerator(
        config=config, run=_fake_run(osu_text=timing_only_osu)
    )(Path("game.flac"), tmp_path / "work")

    assert result == timing_only_osu


def test_super_timing_generator_maps_subprocess_failure(config, tmp_path):
    generator = MapperatorinatorTimingGenerator(config=config, run=_fake_run(fail=True))

    with pytest.raises(WorkerError) as caught:
        generator(Path("game.flac"), tmp_path / "work")

    assert caught.value.code is ErrorCode.CHART_GENERATION_FAILED
    assert "cuda oom" in caught.value.context["stderr"]


# --- subprocess 어댑터 -------------------------------------------------------

MINI_OSU = (
    "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:4\n"
    "\n[TimingPoints]\n0,500,4,2,0,60,1,0\n"
    "\n[HitObjects]\n64,192,1000,1,0,0:0:0:0:\n192,192,1200,1,0,0:0:0:0:\n"
)


def _fake_run(osu_text=MINI_OSU, fail=False):
    def run(argv):
        if fail:
            raise CommandError(argv, "exited with 1", returncode=1, stderr="cuda oom")
        output_dir = Path(
            next(item for item in argv if item.startswith("output_path=")).split("=", 1)[1]
        )
        (output_dir / "out.osu").write_text(osu_text, encoding="utf-8")
        return CommandResult(argv, 0, "", "")

    return run


def test_generator_parses_the_produced_beatmap(config, timing_osu, tmp_path):
    generator = MapperatorinatorGenerator(config=config, run=_fake_run())
    result = generator(_request(timing_osu, seed=3), tmp_path / "work")
    assert [note.time_ms for note in result.notes] == [1000, 1200]
    assert result.key_mode == 4
    assert result.generator_name == "mapperatorinator-v32"
    assert result.seed == 3


def test_generator_maps_subprocess_failure(config, timing_osu, tmp_path):
    generator = MapperatorinatorGenerator(config=config, run=_fake_run(fail=True))
    with pytest.raises(WorkerError) as caught:
        generator(_request(timing_osu), tmp_path / "work")
    assert caught.value.code is ErrorCode.CHART_GENERATION_FAILED
    assert "cuda oom" in caught.value.context["stderr"]


def test_generator_maps_parse_failure(config, timing_osu, tmp_path):
    generator = MapperatorinatorGenerator(config=config, run=_fake_run(osu_text="garbage"))
    with pytest.raises(WorkerError) as caught:
        generator(_request(timing_osu), tmp_path / "work")
    assert caught.value.code is ErrorCode.CHART_OSU_PARSE_FAILED


def test_generator_rejects_a_mismatched_keycount(config, timing_osu, tmp_path):
    generator = MapperatorinatorGenerator(config=config, run=_fake_run())
    with pytest.raises(WorkerError, match="asked for 7K"):
        generator(_request(timing_osu, key_mode=7), tmp_path / "work")


# --- fake 생성기 -------------------------------------------------------------


def test_fake_places_notes_on_the_beat_grid(timing_osu, tmp_path):
    notes = synthesize_chart(_request(timing_osu, difficulty="NORMAL", seed=1))
    assert notes
    assert all(0 <= note.time_ms < DURATION_MS for note in notes)
    assert all(0 <= note.lane < 4 for note in notes)


@pytest.mark.parametrize(
    ("difficulty", "harder"), [("EASY", "NORMAL"), ("NORMAL", "HARD"), ("HARD", "EXPERT")]
)
def test_fake_density_grows_with_difficulty(timing_osu, difficulty, harder):
    easy = synthesize_chart(_request(timing_osu, difficulty=difficulty, seed=1))
    hard = synthesize_chart(_request(timing_osu, difficulty=harder, seed=1))
    assert len(hard) > len(easy)


def test_fake_is_deterministic(timing_osu):
    first = synthesize_chart(_request(timing_osu, seed=42))
    second = synthesize_chart(_request(timing_osu, seed=42))
    assert [(n.time_ms, n.lane, n.kind) for n in first] == [
        (n.time_ms, n.lane, n.kind) for n in second
    ]


def test_fake_seeds_change_the_chart(timing_osu):
    first = synthesize_chart(_request(timing_osu, seed=1))
    second = synthesize_chart(_request(timing_osu, seed=2))
    assert [(n.time_ms, n.lane) for n in first] != [(n.time_ms, n.lane) for n in second]


def test_fake_holds_never_overlap_in_a_lane(timing_osu):
    notes = synthesize_chart(_request(timing_osu, difficulty="EXPERT", seed=5))
    end_by_lane: dict[int, int] = {}
    for note in sorted(notes, key=lambda n: n.time_ms):
        assert note.time_ms >= end_by_lane.get(note.lane, -1)
        end_by_lane[note.lane] = note.time_ms + (note.duration_ms or 0)


def test_fake_holds_stay_inside_the_song(timing_osu):
    notes = synthesize_chart(_request(timing_osu, difficulty="EXPERT", seed=5))
    for note in notes:
        assert note.time_ms + (note.duration_ms or 0) < DURATION_MS


def test_fake_needs_timing_and_duration(tmp_path):
    with pytest.raises(WorkerError, match="timing .osu"):
        synthesize_chart(_request(None))


def test_fake_generator_returns_a_chart(timing_osu, tmp_path):
    result = FakeGenerator()(_request(timing_osu, seed=1), tmp_path)
    assert result.generator_name == "fake-synthetic"
    assert result.key_mode == 4
    assert result.notes


def test_fake_generator_can_replay_a_fixture(timing_osu, tmp_path):
    fixture = tmp_path / "fixture.osu"
    fixture.write_text(MINI_OSU, encoding="utf-8")
    result = FakeGenerator(fixture_path=fixture)(_request(timing_osu), tmp_path)
    assert result.generator_name == "fake-fixture"
    assert [note.time_ms for note in result.notes] == [1000, 1200]


def test_fixture_keycount_must_match(timing_osu, tmp_path):
    fixture = tmp_path / "fixture.osu"
    fixture.write_text(MINI_OSU, encoding="utf-8")
    with pytest.raises(WorkerError, match="fixture is 4K"):
        FakeGenerator(fixture_path=fixture)(_request(timing_osu, key_mode=7), tmp_path)


# --- 검수 회귀 --------------------------------------------------------------


def test_the_command_runs_inside_the_mapperatorinator_checkout(config, timing_osu, tmp_path):
    """inference.py 는 상대 경로이고 Hydra 는 실행 위치에서 configs/ 를 찾는다."""
    seen = {}

    def run(argv):
        seen["argv"] = argv
        (tmp_path / "work").mkdir(parents=True, exist_ok=True)
        (tmp_path / "work" / "out.osu").write_text(MINI_OSU, encoding="utf-8")
        return CommandResult(argv, 0, "", "")

    generator = MapperatorinatorGenerator(config=config, run=run)
    generator(_request(timing_osu), tmp_path / "work")
    assert seen["argv"][1] == "inference.py"


def test_fake_is_deterministic_across_processes(timing_osu):
    """문자열이 낀 튜플의 hash() 는 PYTHONHASHSEED 로 소금이 쳐진다."""
    import subprocess
    import sys

    script = (
        "from pathlib import Path;"
        "from chart_worker.generation.params import GenerationRequest;"
        "from chart_worker.generation.fake import synthesize_chart;"
        f"r=GenerationRequest(audio_path=Path('a.flac'),key_mode=4,difficulty='NORMAL',"
        f"duration_ms={DURATION_MS},timing_osu_path=Path(r'{timing_osu}'),seed=42);"
        "print([n.lane for n in synthesize_chart(r)])"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(3)
    }
    assert len(runs) == 1


def test_every_path_argument_is_absolute(config, timing_osu, tmp_path):
    """cwd 가 Mapperatorinator 홈이라 상대 경로는 그 아래로 해석된다."""
    request = _request(Path("work/timing.osu"), audio_path=Path("storage/game.flac"))
    argv = build_command(config, request, Path("out/4-normal"))
    for key in ("audio_path", "output_path", "beatmap_path"):
        value = _pairs(argv)[key]
        assert Path(value).is_absolute(), f"{key}={value}"
    assert Path(argv[0]).is_absolute()


def test_a_bare_interpreter_name_is_left_for_path_lookup():
    bare = WorkerConfig(
        mapperatorinator_python=Path("python"), mapperatorinator_home=Path("C:/mapp")
    )
    argv = build_command(bare, _request(Path("t.osu")), Path("out"))
    assert argv[0] == "python"


def test_mapperatorinator_child_process_forces_utf8_without_losing_path():
    env = inference_env({"Path": "C:/tools", "pythonutf8": "0"})
    assert env["Path"] == "C:/tools"
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert sum(name.upper() == "PYTHONUTF8" for name in env) == 1
