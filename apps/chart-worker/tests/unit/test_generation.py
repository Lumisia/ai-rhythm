import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from chart_worker.audio import profile
from chart_worker.audio.runner import CommandError, CommandResult, CommandRunner
from chart_worker.config import WorkerConfig
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation import mapperatorinator
from chart_worker.generation.fake import FakeGenerator, synthesize_chart
from chart_worker.generation.inference_session import (
    InvocationArtifact,
    InvocationResult,
)
from chart_worker.generation.mapperatorinator import (
    MapperatorinatorGenerator,
    build_map_command,
    build_timing_command,
    find_generated_osu,
    inference_env,
)
from chart_worker.generation.osu_parser import OsuBpmEvent, parse_osu_mania
from chart_worker.generation.params import (
    DESCRIPTORS,
    REQUESTED_STAR,
    GenerationRequest,
    TimingGenerationRequest,
)
from chart_worker.generation.required_gameplay_interval import (
    RequiredGameplayEvidenceClass,
    RequiredGameplayGroupType,
    RequiredGameplayIntervalMode,
    RequiredGameplayIntervalV1,
)
from chart_worker.generation.required_gameplay_invocation import (
    required_gameplay_invocation_digest,
)
from chart_worker.validation.generated_chart import (
    GeneratedChartValidationError,
    validate_generated_chart,
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
INVOCATION_RESULT_PREFIX = "MAPPERATORINATOR_INVOCATION_RESULT="
OVERSIZED_INTEGER_TERMINAL = '{"version":' + "9" * 5_000 + "}"
TAIL_FAILURE_CONTEXT = {
    "reason": "END_BOUNDARY_CROSSES_HOLD",
    "signature": "END_BOUNDARY_CROSSES_HOLD:500:11",
    "requestedEndMs": 500,
    "effectiveCutMs": 500,
    "earliestGeneratedSourceWindowId": 11,
    "repairTriggerReason": "END_BOUNDARY_CROSSES_HOLD",
    "repairTriggerSourceWindowId": 11,
    "repairStartWindowId": 9,
    "repairWindowIds": [9, 10, 11],
    "repairAttempts": 2,
}


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


def _required_interval(
    *,
    start_ms: int = 4_430,
    end_ms: int = 4_570,
    mode: RequiredGameplayIntervalMode = RequiredGameplayIntervalMode.OBSERVE,
) -> RequiredGameplayIntervalV1:
    return RequiredGameplayIntervalV1(
        start_ms=start_ms,
        end_ms=end_ms,
        minimum_complete_groups=1,
        allowed_group_types=(
            RequiredGameplayGroupType.TAP,
            RequiredGameplayGroupType.HOLD_START,
        ),
        evidence_class=RequiredGameplayEvidenceClass.BROADBAND_ATTACK,
        evidence_digest="a" * 64,
        mode=mode,
    )


def _pairs(argv):
    return dict(item.split("=", 1) for item in argv if "=" in item)


def _fake_run(osu_text=MINI_OSU, fail=False, resnap_sidecar=None):
    def run(argv):
        if fail:
            raise CommandError(argv, "exited with 1", returncode=1, stderr="cuda oom")
        output_dir = Path(
            next(item for item in argv if item.startswith("output_path=")).split("=", 1)[1]
            .strip("'")
        )
        osu_path = output_dir / "out.osu"
        osu_path.write_text(osu_text, encoding="utf-8")
        if resnap_sidecar is not None:
            osu_path.with_suffix(".resnap.json").write_text(
                json.dumps(resnap_sidecar),
                encoding="utf-8",
            )
        return CommandResult(argv, 0, "", "")

    return run


def _required_interval_run(
    config: WorkerConfig,
    request: GenerationRequest,
    *,
    write_origin: bool = True,
):
    def run(argv):
        output_dir = Path(
            next(item for item in argv if item.startswith("output_path=")).split("=", 1)[1]
            .strip("'")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        osu_path = output_dir / "out.osu"
        osu_path.write_text(MINI_OSU, encoding="utf-8")
        osu_sha = hashlib.sha256(osu_path.read_bytes()).hexdigest()
        origin = {
            "kind": "GENERATED",
            "sourceWindowId": 0,
            "sourceTokenIndex": 7,
            "referenceEventIndex": None,
        }
        osu_path.with_suffix(".resnap.json").write_text(
            json.dumps(
                {
                    "version": "mania-origin-v2-osu-bound",
                    "seed": 17,
                    "osuSha256": osu_sha,
                    "collisions": [],
                    "maniaObjects": [
                        {
                            "objectId": 0,
                            "lane": 0,
                            "kind": "TAP",
                            "startTimeMs": 1000,
                            "endTimeMs": None,
                            "startGroupId": 0,
                            "endGroupId": None,
                            "startOrigins": [origin],
                            "endOrigins": [],
                        },
                        {
                            "objectId": 1,
                            "lane": 1,
                            "kind": "TAP",
                            "startTimeMs": 1200,
                            "endTimeMs": None,
                            "startGroupId": 1,
                            "endGroupId": None,
                            "startOrigins": [origin],
                            "endOrigins": [],
                        },
                    ],
                    "duplicates": [],
                }
            ),
            encoding="utf-8",
        )
        if write_origin:
            interval = request.required_gameplay_interval
            assert interval is not None
            count = {
                "totalGeneratedCompleteGroups": 2,
                "intervalGeneratedCompleteGroups": 1,
                "tapGroups": 1,
                "holdStartGroups": 0,
            }
            osu_path.with_suffix(".origin.json").write_text(
                json.dumps(
                    {
                        "version": "generation-origin-diagnostics-v1",
                        "output": {
                            "fileName": osu_path.name,
                            "size": osu_path.stat().st_size,
                            "sha256": osu_sha,
                        },
                        "evidenceDigest": interval.evidence_digest,
                        "invocationDigest": required_gameplay_invocation_digest(
                            config, request
                        ),
                        "requiredInterval": {
                            "startMs": interval.start_ms,
                            "endMs": interval.end_ms,
                            "minimumCompleteGroups": interval.minimum_complete_groups,
                            "allowedGroupTypes": [
                                item.value for item in interval.allowed_group_types
                            ],
                            "evidenceClass": interval.evidence_class.value,
                            "mode": interval.mode.value,
                        },
                        "stages": {
                            name: count
                            for name in (
                                "decoder",
                                "windowMerge",
                                "canonical",
                                "resnap",
                                "finalSerialization",
                            )
                        },
                        "firstLossStage": None,
                    }
                ),
                encoding="utf-8",
            )
        return CommandResult(argv, 0, "", "")

    return run


def _terminal_run(record, *, returncode=17, duplicate=False, stderr="diagnostic"):
    def run(argv):
        encoded = record if isinstance(record, str) else json.dumps(record)
        line = f"{INVOCATION_RESULT_PREFIX}{encoded}"
        stdout = f"{line}\n{line}\n" if duplicate else f"{line}\n"
        return CommandResult(argv, returncode, stdout, stderr)

    return run


def _resident_config() -> WorkerConfig:
    return WorkerConfig(
        mapperatorinator_python=Path("C:/mapp/.venv/Scripts/python.exe"),
        mapperatorinator_home=Path("C:/mapp"),
        mapperatorinator_backend="song_session",
        mapperatorinator_hold_state_mode="incremental",
        mapperatorinator_model_root=Path("C:/models/mapperatorinator-snapshot"),
        mapperatorinator_model_revision="a" * 40,
    )


def _resident_session(osu_text: str = MINI_OSU):
    calls: list[tuple[list[str], Path]] = []

    def invoke(argv: list[str], workdir: Path) -> InvocationResult:
        calls.append((list(argv), workdir))
        workdir.mkdir(parents=True, exist_ok=True)
        output = workdir / "resident.osu"
        payload = osu_text.encode("utf-8")
        output.write_bytes(payload)
        return InvocationResult(
            status="SUCCESS",
            command=CommandResult(argv, 0, "", ""),
            accepted=True,
            invocation_id="b" * 64,
            request_hash="c" * 64,
            artifacts=(
                InvocationArtifact(
                    relative_path="resident.osu",
                    size=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                ),
            ),
        )

    return SimpleNamespace(invoke=invoke), calls


def test_song_session_generator_requires_attached_session_before_output_mutation(tmp_path):
    workdir = tmp_path / "must-not-exist"
    generator = MapperatorinatorGenerator(
        config=_resident_config(),
        verify_patch=lambda _home: None,
        require_bound_resnap_sidecar=False,
    )

    with pytest.raises(ValueError, match="attached inference session"):
        generator.generate_map(_request(), workdir)

    assert not workdir.exists()


def test_song_session_generator_uses_only_bound_artifact_and_direct_offline_overrides(tmp_path):
    session, calls = _resident_session()

    result = MapperatorinatorGenerator(
        config=_resident_config(),
        session=session,
        run=lambda _argv: pytest.fail("one-shot runner must not be called"),
        verify_patch=lambda _home: None,
        require_bound_resnap_sidecar=False,
    ).generate_map(_request(seed=3), tmp_path / "resident-work")

    assert result.key_mode == 4
    assert len(calls) == 1
    argv, observed_workdir = calls[0]
    assert observed_workdir == tmp_path / "resident-work"
    pairs = _pairs(argv)
    assert pairs["model_path"] == "'C:\\models\\mapperatorinator-snapshot'"
    assert pairs["use_server"] == "false"
    assert pairs["parallel"] == "false"
    assert pairs["generate_positions"] == "false"
    assert pairs["mania_hold_state_mode"] == "incremental"


def test_requested_stars_and_descriptors_are_explicit():
    assert REQUESTED_STAR == {
        "EASY": 1.0,
        "NORMAL": 1.5,
        "HARD": 2.0,
        "EXPERT": 3.0,
    }
    assert DESCRIPTORS == {
        "EASY": ("expression/simple",),
        "NORMAL": ("style/mixed rice",),
        "HARD": ("style/mixed rice", "streams/bursts"),
        "EXPERT": ("style/mixed rice", "skillset/streams"),
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


@pytest.mark.parametrize("mode", ["full_scan", "incremental_verify", "incremental"])
def test_timing_and_map_commands_each_propagate_one_hold_state_mode(
    config, tmp_path, mode
):
    configured = config.model_copy(update={"mapperatorinator_hold_state_mode": mode})
    commands = (
        build_timing_command(
            configured,
            TimingGenerationRequest(
                audio_path=Path("game.flac"), duration_ms=DURATION_MS
            ),
            tmp_path / "timing",
        ),
        build_map_command(configured, _request(), tmp_path / "map"),
    )

    for argv in commands:
        hold_state_overrides = [
            item for item in argv if item.startswith("mania_hold_state_mode=")
        ]
        assert hold_state_overrides == [f"mania_hold_state_mode={mode}"]


def test_map_command_propagates_opt_in_processor_generation_telemetry(
    config, tmp_path
):
    configured = config.model_copy(
        update={"mapperatorinator_write_generation_telemetry": True}
    )
    argv = build_map_command(configured, _request(), tmp_path / "map")

    assert [
        item for item in argv if item.startswith("write_generation_telemetry=")
    ] == ["write_generation_telemetry=true"]


def test_timing_command_does_not_request_map_processor_telemetry(config, tmp_path):
    configured = config.model_copy(
        update={"mapperatorinator_write_generation_telemetry": True}
    )

    argv = build_timing_command(
        configured,
        TimingGenerationRequest(
            audio_path=Path("game.flac"),
            duration_ms=DURATION_MS,
        ),
        tmp_path / "timing",
    )

    assert not any(
        item.startswith("write_generation_telemetry=") for item in argv
    )


@pytest.mark.parametrize("builder", ["timing", "map"])
def test_generation_commands_do_not_enable_generation_telemetry_by_default(
    config, tmp_path, builder
):
    if builder == "timing":
        argv = build_timing_command(
            config,
            TimingGenerationRequest(
                audio_path=Path("game.flac"),
                duration_ms=DURATION_MS,
            ),
            tmp_path / "timing",
        )
    else:
        argv = build_map_command(config, _request(), tmp_path / "map")

    assert not any(
        item.startswith("write_generation_telemetry=") for item in argv
    )


def test_map_command_reuses_the_timing_reference(config, tmp_path):
    timing = tmp_path / "audio" / "timing-reference.osu"
    request = _request(timing_reference_path=timing)
    pairs = _pairs(build_map_command(config, request, tmp_path / "map"))
    assert pairs["output_type"] == "[MAP]"
    assert pairs["beatmap_path"] == f"'{timing.resolve()}'"
    assert pairs["in_context"] == "[TIMING]"
    assert "TIMING" not in pairs["output_type"]


def test_map_command_separates_generation_and_note_start_horizons(config, tmp_path):
    request = _request(
        music_end_ms=15_000,
        generation_end_ms=18_000,
        last_attack_ms=14_500,
        max_note_start_ms=14_570,
    )

    pairs = _pairs(build_map_command(config, request, tmp_path / "map"))

    assert pairs["end_time"] == "18000"
    assert pairs["last_attack_time"] == "14570"


@pytest.mark.parametrize(
    "overrides",
    [
        {"last_attack_ms": 15_001, "max_note_start_ms": 15_000},
        {"max_note_start_ms": 18_001, "generation_end_ms": 18_000},
        {"max_note_start_ms": DURATION_MS + 1},
    ],
)
def test_map_request_rejects_inconsistent_song_boundaries(overrides):
    with pytest.raises(ValueError, match="last_attack|max_note_start"):
        _request(**overrides)


def test_partial_map_command_replaces_only_the_requested_window(config, tmp_path):
    reference = tmp_path / "candidate.osu"
    request = _request(
        timing_reference_path=reference,
        partial_start_ms=4_000,
        partial_end_ms=12_000,
        add_to_beatmap=True,
    )

    pairs = _pairs(build_map_command(config, request, tmp_path / "partial"))

    assert pairs["beatmap_path"] == f"'{reference.resolve()}'"
    assert pairs["start_time"] == "4000"
    assert pairs["end_time"] == "12000"
    assert pairs["add_to_beatmap"] == "true"


def test_observe_required_gameplay_interval_is_serialized_only_for_partial_map(
    config,
    tmp_path,
):
    request = _request(
        seed=17,
        partial_start_ms=4_000,
        partial_end_ms=12_000,
        add_to_beatmap=True,
        required_gameplay_interval=_required_interval(),
    )

    pairs = _pairs(build_map_command(config, request, tmp_path / "partial"))

    assert pairs["required_gameplay_interval_mode"] == "OBSERVE"
    assert pairs["required_gameplay_interval_start_time"] == "4430"
    assert pairs["required_gameplay_interval_end_time"] == "4570"
    assert pairs["required_gameplay_interval_minimum_complete_groups"] == "1"
    assert pairs["required_gameplay_interval_allowed_group_types"] == "[HOLD_START,TAP]"
    assert (
        pairs["required_gameplay_interval_evidence_class"]
        == "BROADBAND_ATTACK"
    )
    assert pairs["required_gameplay_interval_evidence_digest"] == "a" * 64


def test_full_map_request_rejects_a_required_gameplay_interval():
    with pytest.raises(ValueError, match="partial generation"):
        _request(required_gameplay_interval=_required_interval())


@pytest.mark.parametrize(
    "interval",
    [
        _required_interval(start_ms=3_999),
        _required_interval(end_ms=12_001),
    ],
)
def test_partial_request_rejects_required_interval_outside_its_window(interval):
    with pytest.raises(ValueError, match="required gameplay interval.*partial"):
        _request(
            partial_start_ms=4_000,
            partial_end_ms=12_000,
            add_to_beatmap=True,
            required_gameplay_interval=interval,
        )


def test_v36_adapter_serializes_shadow_enforcement_for_partial_challenger(
    config,
    tmp_path,
):
    request = _request(
        seed=17,
        partial_start_ms=4_000,
        partial_end_ms=12_000,
        add_to_beatmap=True,
        required_gameplay_interval=_required_interval(
            mode=RequiredGameplayIntervalMode.SHADOW_ENFORCE
        ),
    )

    pairs = _pairs(build_map_command(config, request, tmp_path / "partial"))

    assert pairs["required_gameplay_interval_mode"] == "SHADOW_ENFORCE"
    assert pairs["required_gameplay_interval_start_time"] == "4430"
    assert pairs["required_gameplay_interval_end_time"] == "4570"


def test_dataclass_replace_preserves_the_required_interval_identity():
    interval = _required_interval()
    request = _request(
        seed=17,
        partial_start_ms=4_000,
        partial_end_ms=12_000,
        add_to_beatmap=True,
        required_gameplay_interval=interval,
    )

    replaced = replace(request, seed=17)

    assert replaced.required_gameplay_interval is interval


def _required_generation_request(tmp_path: Path) -> GenerationRequest:
    audio = tmp_path / "audio.flac"
    reference = tmp_path / "reference.osu"
    audio.write_bytes(b"audio")
    reference.write_text(TIMING_OSU, encoding="utf-8")
    return _request(
        audio_path=audio,
        timing_reference_path=reference,
        seed=17,
        partial_start_ms=0,
        partial_end_ms=2_000,
        add_to_beatmap=True,
        required_gameplay_interval=_required_interval(start_ms=900, end_ms=1_100),
    )


def test_generator_requires_and_returns_independently_bound_origin_diagnostics(
    config, tmp_path
):
    request = _required_generation_request(tmp_path)
    result = MapperatorinatorGenerator(
        config=config,
        run=_required_interval_run(config, request),
        verify_patch=lambda _home: None,
    ).generate_map(request, tmp_path / "work")

    assert result.origin_diagnostics is not None
    assert result.origin_diagnostics.first_loss_stage is None
    assert (
        result.origin_diagnostics.final_serialization.interval_generated_complete_groups
        == 1
    )


def test_generator_rejects_missing_required_origin_diagnostics(config, tmp_path):
    request = _required_generation_request(tmp_path)
    generator = MapperatorinatorGenerator(
        config=config,
        run=_required_interval_run(config, request, write_origin=False),
        verify_patch=lambda _home: None,
    )

    with pytest.raises(WorkerError, match="origin diagnostics"):
        generator.generate_map(request, tmp_path / "work")


def test_ordinary_map_command_has_no_required_gameplay_overrides(config, tmp_path):
    argv = build_map_command(config, _request(), tmp_path / "ordinary")

    assert not any(item.startswith("required_gameplay_interval_") for item in argv)


@pytest.mark.parametrize(
    "overrides",
    [
        {"partial_start_ms": 4_000},
        {"partial_end_ms": 12_000},
        {"partial_start_ms": 12_000, "partial_end_ms": 4_000},
        {"partial_start_ms": -1, "partial_end_ms": 4_000},
        {"partial_start_ms": 4_000, "partial_end_ms": DURATION_MS + 1},
        {
            "partial_start_ms": 4_000,
            "partial_end_ms": 12_000,
            "add_to_beatmap": False,
        },
    ],
)
def test_partial_map_request_rejects_incomplete_or_unsafe_ranges(overrides):
    with pytest.raises(ValueError, match="partial|add_to_beatmap"):
        _request(**overrides)


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
        "difficulty": "3.0",
        "year": "2023",
        "end_time": str(DURATION_MS),
        "cfg_scale": "1.0",
        "descriptors": "['style/mixed rice','skillset/streams']",
        "precision": "fp16",
        "export_osz": "false",
        "beatmap_path": f"'{Path('timing-reference.osu').resolve()}'",
        "in_context": "[TIMING]",
        "output_type": "[MAP]",
        "super_timing": "false",
        "hitsounded": "false",
        "fast_decoder_loop": "true",
        "resnap_events": "true",
        "mania_hold_state_mode": "incremental",
        "parallel": "false",
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


def test_descriptor_quotes_preserve_trailing_space_for_v32_vocabulary(config, tmp_path):
    pairs = _pairs(
        build_map_command(
            config,
            _request(descriptors=("style/LN mixed ",)),
            tmp_path,
        )
    )

    assert pairs["descriptors"] == "['style/LN mixed ']"


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
        config=config,
        run=_fake_run(),
        verify_patch=lambda _home: None,
        require_bound_resnap_sidecar=False,
    ).generate_map(_request(seed=3), tmp_path / "work")
    assert [note.time_ms for note in result.notes] == [1000, 1200]
    assert result.key_mode == 4
    assert result.generator_name == "mapperatorinator-v32"
    assert result.seed == 3
    assert [(event.time_ms, event.bpm) for event in result.bpm_events] == [(0, 120.0)]


def test_generator_requires_a_bound_resnap_sidecar(config, tmp_path):
    generator = MapperatorinatorGenerator(
        config=config,
        run=_fake_run(),
        verify_patch=lambda _home: None,
    )

    with pytest.raises(WorkerError, match="bound resnap sidecar"):
        generator.generate_map(_request(seed=3), tmp_path / "work")


def test_generator_normalizes_events_within_ten_ms_of_audio_end(config, tmp_path):
    raw = (
        "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:4\n"
        "\n[TimingPoints]\n0,500,4,2,0,60,1,0\n"
        "\n[HitObjects]\n64,192,1000,1,0,0:0:0:0:\n"
        "192,192,19000,128,0,20006:0:0:0:0:\n"
        "320,192,20006,1,0,0:0:0:0:\n"
        "448,192,20006,128,0,20500:0:0:0:0:\n"
        "64,192,20010,1,0,0:0:0:0:\n"
    )
    result = MapperatorinatorGenerator(
        config=config,
        run=_fake_run(osu_text=raw),
        verify_patch=lambda _home: None,
        require_bound_resnap_sidecar=False,
    ).generate_map(_request(), tmp_path / "work")

    assert [
        (note.time_ms, note.lane, note.kind, note.duration_ms)
        for note in result.notes
    ] == [
        (1000, 0, "TAP", None),
        (19000, 1, "HOLD", 1000),
    ]
    reparsed = parse_osu_mania(result.osu_text)
    assert reparsed.notes == result.notes


def test_generator_validates_bound_sidecar_before_local_end_normalization(
    config, tmp_path
):
    raw = (
        "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:4\n"
        "\n[TimingPoints]\n0,500,4,2,0,60,1,0\n"
        "\n[HitObjects]\n64,192,1000,1,0,0:0:0:0:\n"
        "192,192,19000,128,0,20006:0:0:0:0:\n"
        "320,192,20006,1,0,0:0:0:0:\n"
        "448,192,20006,128,0,20500:0:0:0:0:\n"
        "64,192,20010,1,0,0:0:0:0:\n"
    )
    sidecar = {
        "version": "mania-origin-v1-canonical-hold-ir",
        "seed": 11,
        "collisions": [],
        "maniaObjects": [
            {
                "objectId": 0,
                "lane": 0,
                "kind": "TAP",
                "startTimeMs": 1000,
                "endTimeMs": None,
                "startGroupId": 0,
                "endGroupId": None,
                "startOrigins": [],
                "endOrigins": [],
            },
            {
                "objectId": 1,
                "lane": 1,
                "kind": "HOLD",
                "startTimeMs": 19000,
                "endTimeMs": 20006,
                "startGroupId": 1,
                "endGroupId": 2,
                "startOrigins": [],
                "endOrigins": [],
            },
            {
                "objectId": 2,
                "lane": 2,
                "kind": "TAP",
                "startTimeMs": 20006,
                "endTimeMs": None,
                "startGroupId": 3,
                "endGroupId": None,
                "startOrigins": [],
                "endOrigins": [],
            },
            {
                "objectId": 3,
                "lane": 3,
                "kind": "HOLD",
                "startTimeMs": 20006,
                "endTimeMs": 20500,
                "startGroupId": 4,
                "endGroupId": 5,
                "startOrigins": [],
                "endOrigins": [],
            },
            {
                "objectId": 4,
                "lane": 0,
                "kind": "TAP",
                "startTimeMs": 20010,
                "endTimeMs": None,
                "startGroupId": 6,
                "endGroupId": None,
                "startOrigins": [],
                "endOrigins": [],
            },
        ],
        "duplicates": [],
    }

    result = MapperatorinatorGenerator(
        config=config,
        run=_fake_run(osu_text=raw, resnap_sidecar=sidecar),
        verify_patch=lambda _home: None,
        require_bound_resnap_sidecar=False,
    ).generate_map(_request(seed=11), tmp_path / "work")

    assert [(note.time_ms, note.kind, note.duration_ms) for note in result.notes] == [
        (1000, "TAP", None),
        (19000, "HOLD", 1000),
    ]
    assert len(result.resnap_diagnostics.mania_objects) == 5


def test_generator_collapses_only_indistinguishable_model_taps(config, tmp_path):
    raw = MINI_OSU.replace(
        "64,192,1000,1,0,0:0:0:0:\n",
        "64,192,1000,1,0,0:0:0:0:\n"
        "64,192,1000,1,0,0:0:0:0:\n",
    )
    result = MapperatorinatorGenerator(
        config=config,
        run=_fake_run(osu_text=raw),
        verify_patch=lambda _home: None,
        require_bound_resnap_sidecar=False,
    ).generate_map(_request(), tmp_path / "work")

    assert [(note.time_ms, note.lane) for note in result.notes] == [
        (1000, 0),
        (1200, 1),
    ]
    assert parse_osu_mania(result.osu_text).notes == result.notes
    validate_generated_chart(result, key_mode=4, duration_ms=DURATION_MS)


def test_generator_does_not_collapse_tap_and_hold_at_the_same_lane_time(
    config, tmp_path
):
    raw = MINI_OSU.replace(
        "64,192,1000,1,0,0:0:0:0:\n",
        "64,192,1000,1,0,0:0:0:0:\n"
        "64,192,1000,128,0,1500:0:0:0:0:\n",
    )
    result = MapperatorinatorGenerator(
        config=config,
        run=_fake_run(osu_text=raw),
        verify_patch=lambda _home: None,
        require_bound_resnap_sidecar=False,
    ).generate_map(_request(), tmp_path / "work")

    assert len(result.notes) == 3
    with pytest.raises(GeneratedChartValidationError, match="duplicate"):
        validate_generated_chart(result, key_mode=4, duration_ms=DURATION_MS)


def test_generator_does_not_hide_events_beyond_end_tolerance(config, tmp_path):
    raw = MINI_OSU.replace(
        "192,192,1200,1,0,0:0:0:0:",
        "192,192,20011,1,0,0:0:0:0:",
    )
    result = MapperatorinatorGenerator(
        config=config,
        run=_fake_run(osu_text=raw),
        verify_patch=lambda _home: None,
        require_bound_resnap_sidecar=False,
    ).generate_map(_request(), tmp_path / "work")

    with pytest.raises(GeneratedChartValidationError, match="duration"):
        validate_generated_chart(result, key_mode=4, duration_ms=DURATION_MS)


def test_generator_maps_subprocess_failure(config, tmp_path):
    generator = MapperatorinatorGenerator(
        config=config, run=_fake_run(fail=True), verify_patch=lambda _home: None
    )
    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")
    assert caught.value.code is ErrorCode.CHART_GENERATION_FAILED
    assert "cuda oom" in caught.value.context["stderr"]


def test_generator_maps_definitive_tail_exhaustion_with_exact_context(config, tmp_path):
    terminal = {
        "version": 1,
        "status": "FAILURE",
        "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
        "context": TAIL_FAILURE_CONTEXT,
    }
    generator = MapperatorinatorGenerator(
        config=config,
        run=_terminal_run(terminal),
        verify_patch=lambda _home: None,
    )

    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")

    assert caught.value.code is ErrorCode.MANIA_TAIL_REPAIR_EXHAUSTED
    assert caught.value.context == TAIL_FAILURE_CONTEXT
    assert caught.value.context is not TAIL_FAILURE_CONTEXT


def test_generator_accepts_token_budget_exhaustion_as_definitive_tail_failure(
    config, tmp_path
):
    context = {
        **TAIL_FAILURE_CONTEXT,
        "reason": "TOKEN_BUDGET_EXHAUSTED",
        "signature": "TOKEN_BUDGET_EXHAUSTED:500:11",
        "repairTriggerReason": "TOKEN_BUDGET_EXHAUSTED",
    }
    terminal = {
        "version": 1,
        "status": "FAILURE",
        "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
        "context": context,
    }
    generator = MapperatorinatorGenerator(
        config=config,
        run=_terminal_run(terminal),
        verify_patch=lambda _home: None,
    )

    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")

    assert caught.value.code is ErrorCode.MANIA_TAIL_REPAIR_EXHAUSTED
    assert caught.value.context == context


def test_generator_accepts_finalizer_cut_past_requested_end(config, tmp_path):
    context = {
        **TAIL_FAILURE_CONTEXT,
        "requestedEndMs": 500,
        "effectiveCutMs": 510,
    }
    terminal = {
        "version": 1,
        "status": "FAILURE",
        "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
        "context": context,
    }
    generator = MapperatorinatorGenerator(
        config=config,
        run=_terminal_run(terminal),
        verify_patch=lambda _home: None,
    )

    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")

    assert caught.value.code is ErrorCode.MANIA_TAIL_REPAIR_EXHAUSTED
    assert caught.value.context == context


@pytest.mark.parametrize(
    "context",
    [
        {
            **TAIL_FAILURE_CONTEXT,
            "signature": "END_BOUNDARY_CROSSES_HOLD:500:1000000",
            "earliestGeneratedSourceWindowId": 1_000_000,
            "repairTriggerSourceWindowId": 1_000_000,
            "repairStartWindowId": 999_998,
            "repairWindowIds": [999_998, 999_999, 1_000_000],
        },
        {
            **TAIL_FAILURE_CONTEXT,
            "signature": "END_BOUNDARY_CROSSES_HOLD:500:2",
            "earliestGeneratedSourceWindowId": 2,
            "repairTriggerSourceWindowId": 2,
            "repairStartWindowId": 0,
            "repairWindowIds": list(range(4_096)),
        },
        {
            **TAIL_FAILURE_CONTEXT,
            "signature": "END_BOUNDARY_CROSSES_HOLD:86400000:11",
            "requestedEndMs": 86_400_000,
            "effectiveCutMs": 86_400_010,
        },
        {
            **TAIL_FAILURE_CONTEXT,
            "repairWindowIds": [9],
        },
        {
            **TAIL_FAILURE_CONTEXT,
            "reason": "NO_LEGAL_MANIA_GROUP_COMPLETION",
            "signature": "NO_LEGAL_MANIA_GROUP_COMPLETION:500:1",
            "earliestGeneratedSourceWindowId": 1,
            "repairStartWindowId": 1,
            "repairWindowIds": [1],
            "repairTriggerReason": "PRE_TRIM_INVALID",
            "repairTriggerSourceWindowId": 3,
        },
    ],
)
def test_generator_accepts_tail_context_protocol_bounds(config, tmp_path, context):
    terminal = {
        "version": 1,
        "status": "FAILURE",
        "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
        "context": context,
    }
    generator = MapperatorinatorGenerator(
        config=config,
        run=_terminal_run(terminal),
        verify_patch=lambda _home: None,
    )

    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")

    assert caught.value.code is ErrorCode.MANIA_TAIL_REPAIR_EXHAUSTED
    assert caught.value.context == context


@pytest.mark.parametrize(
    ("record", "returncode", "duplicate"),
    [
        ("not-json", 17, False),
        (OVERSIZED_INTEGER_TERMINAL, 17, False),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": TAIL_FAILURE_CONTEXT,
            },
            17,
            True,
        ),
        (
            {
                "version": 2,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": TAIL_FAILURE_CONTEXT,
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "signature": "END_BOUNDARY_CROSSES_HOLD:86400001:11",
                    "requestedEndMs": 86_400_001,
                    "effectiveCutMs": 86_400_011,
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "repairWindowIds": [9, 2**100],
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "signature": f"END_BOUNDARY_CROSSES_HOLD:500:{2**100}",
                    "earliestGeneratedSourceWindowId": 2**100,
                    "repairStartWindowId": 2**100 - 2,
                    "repairWindowIds": [2**100 - 2, 2**100 - 1, 2**100],
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "signature": "END_BOUNDARY_CROSSES_HOLD:500:2",
                    "earliestGeneratedSourceWindowId": 2,
                    "repairTriggerSourceWindowId": 2,
                    "repairStartWindowId": 0,
                    "repairWindowIds": list(range(4_097)),
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "reason": [],
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "reason": "CUDA_OUT_OF_MEMORY",
                    "signature": "CUDA_OUT_OF_MEMORY:500:11",
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "signature": "END_BOUNDARY_CROSSES_HOLD:501:11",
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "effectiveCutMs": 511,
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "repairStartWindowId": 8,
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "repairTriggerReason": "CUDA_OUT_OF_MEMORY",
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "repairTriggerSourceWindowId": True,
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "repairTriggerSourceWindowId": 1_000_001,
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "repairWindowIds": [9, 11],
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "repairWindowIds": [9, 10, 10, 11],
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "repairWindowIds": [10],
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "earliestGeneratedSourceWindowId": None,
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "requestedEndMs": 500,
                    "effectiveCutMs": 490,
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "TAIL_REPAIR_REQUIRED",
                "context": TAIL_FAILURE_CONTEXT,
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "UNRELATED_FAILURE",
                "context": TAIL_FAILURE_CONTEXT,
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    key: value for key, value in TAIL_FAILURE_CONTEXT.items() if key != "signature"
                },
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": TAIL_FAILURE_CONTEXT,
                "unexpected": True,
            },
            17,
            False,
        ),
        (
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": TAIL_FAILURE_CONTEXT,
            },
            0,
            False,
        ),
    ],
)
def test_generator_rejects_untrustworthy_structured_terminal_results(
    config, tmp_path, record, returncode, duplicate
):
    generator = MapperatorinatorGenerator(
        config=config,
        run=_terminal_run(
            record,
            returncode=returncode,
            duplicate=duplicate,
        ),
        verify_patch=lambda _home: None,
    )

    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")

    assert caught.value.code is ErrorCode.INFERENCE_PROTOCOL_FAILED


def test_generator_never_classifies_tail_failure_from_stderr_text(config, tmp_path):
    def run(argv):
        return CommandResult(
            argv,
            17,
            "",
            "MANIA_TAIL_REPAIR_EXHAUSTED at boundary 500",
        )

    generator = MapperatorinatorGenerator(
        config=config,
        run=run,
        verify_patch=lambda _home: None,
    )

    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")

    assert caught.value.code is ErrorCode.CHART_GENERATION_FAILED


def test_generator_normalizes_injected_command_runner_to_capture_terminal_result(
    config, tmp_path, monkeypatch
):
    terminal = {
        "version": 1,
        "status": "FAILURE",
        "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
        "context": TAIL_FAILURE_CONTEXT,
    }
    observed_checks = []

    def fake_run_command(argv, **kwargs):
        observed_checks.append(kwargs["check"])
        return CommandResult(
            argv,
            17,
            f"{INVOCATION_RESULT_PREFIX}{json.dumps(terminal)}\n",
            "",
        )

    monkeypatch.setattr("chart_worker.audio.runner.run_command", fake_run_command)
    generator = MapperatorinatorGenerator(
        config=config,
        run=CommandRunner(check=True),
        verify_patch=lambda _home: None,
    )

    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")

    assert observed_checks == [False]
    assert caught.value.code is ErrorCode.MANIA_TAIL_REPAIR_EXHAUSTED
    assert caught.value.context == TAIL_FAILURE_CONTEXT


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
        config=config,
        run=_fake_run(),
        verify_patch=lambda _home: None,
        require_bound_resnap_sidecar=False,
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
        require_bound_resnap_sidecar=False,
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


def test_generator_retains_optional_resnap_collision_evidence(config, tmp_path):
    generator = MapperatorinatorGenerator(
        config=config,
        run=_fake_run(
            resnap_sidecar={
                "version": "resnap-collisions-v2-preserve-raw",
                "seed": 11,
                "collisions": [
                    {
                        "lane": 1,
                        "noteKind": "TAP",
                        "preTimeMs": 1_190,
                        "postTimeMs": 1_200,
                        "snapDivisor": 4,
                        "reason": "SNAP_TARGET_CONFLICT_PRESERVED",
                    }
                ],
            }
        ),
        verify_patch=lambda _home: None,
        require_bound_resnap_sidecar=False,
    )

    generated = generator.generate_map(_request(seed=11), tmp_path / "work")

    assert generated.resnap_diagnostics.status == "OBSERVED"
    assert generated.resnap_diagnostics.collisions[0].post_time_ms == 1_200
    assert (
        generated.resnap_diagnostics.collisions[0].reason
        == "SNAP_TARGET_CONFLICT_PRESERVED"
    )


def test_generator_rejects_v4_object_sidecar_that_disagrees_with_osu(config, tmp_path):
    sidecar = {
        "version": "mania-origin-v1-canonical-hold-ir",
        "seed": 11,
        "collisions": [],
        "maniaObjects": [
            {
                "objectId": 0,
                "lane": 3,
                "kind": "TAP",
                "startTimeMs": 1000,
                "endTimeMs": None,
                "startGroupId": 0,
                "endGroupId": None,
                "startOrigins": [],
                "endOrigins": [],
            }
        ],
        "duplicates": [],
    }
    generator = MapperatorinatorGenerator(
        config=config,
        run=_fake_run(resnap_sidecar=sidecar),
        verify_patch=lambda _home: None,
        require_bound_resnap_sidecar=False,
    )

    with pytest.raises(WorkerError, match="sidecar.*does not match"):
        generator.generate_map(_request(seed=11), tmp_path / "work")


PARTIAL_REJOIN_CONTEXT = {
    "scope": "REFERENCE",
    "partialStartMs": 150,
    "partialEndMs": 250,
    "validationError": "reference Mania stream is invalid: unclosed HOLD lanes: [0]",
    "earliestGeneratedSourceWindowId": None,
}

REQUIRED_GAMEPLAY_FAILURE_CONTEXT = {
    "version": 1,
    "reason": "REQUIRED_GAMEPLAY_INTERVAL_NO_LEGAL_GROUP",
    "message": "required interval has no legal group completion",
    "lane": None,
    "timeMs": 500,
    "timeTokenId": 50,
    "eventIndex": None,
    "decoderTokenIndex": 21,
    "generatedTokenIndex": 17,
    "rowIndex": None,
    "sourceWindowId": 3,
    "context": {"requiredStartMs": 409, "requiredEndMs": 549},
}


def test_generator_accepts_an_economically_bounded_single_tail_repair(
    config, tmp_path
):
    """Rewinding to a distant fault window may only afford one suffix pass."""
    context = {**TAIL_FAILURE_CONTEXT, "repairAttempts": 1}
    generator = MapperatorinatorGenerator(
        config=config,
        run=_terminal_run(
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": context,
            }
        ),
        verify_patch=lambda _home: None,
    )

    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")

    assert caught.value.code is ErrorCode.MANIA_TAIL_REPAIR_EXHAUSTED
    assert caught.value.context["repairAttempts"] == 1


@pytest.mark.parametrize("repair_attempts", [0, 3, -1])
def test_generator_rejects_tail_repair_counts_outside_the_budget(
    config, tmp_path, repair_attempts
):
    generator = MapperatorinatorGenerator(
        config=config,
        run=_terminal_run(
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_TAIL_REPAIR_EXHAUSTED",
                "context": {
                    **TAIL_FAILURE_CONTEXT,
                    "repairAttempts": repair_attempts,
                },
            }
        ),
        verify_patch=lambda _home: None,
    )

    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")

    assert caught.value.code is ErrorCode.INFERENCE_PROTOCOL_FAILED


def test_generator_maps_typed_partial_rejoin_failure_with_exact_context(
    config, tmp_path
):
    generator = MapperatorinatorGenerator(
        config=config,
        run=_terminal_run(
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_PARTIAL_REJOIN_INVALID",
                "context": PARTIAL_REJOIN_CONTEXT,
            }
        ),
        verify_patch=lambda _home: None,
    )

    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")

    assert caught.value.code is ErrorCode.MANIA_PARTIAL_REJOIN_INVALID
    assert caught.value.context == PARTIAL_REJOIN_CONTEXT
    assert caught.value.context is not PARTIAL_REJOIN_CONTEXT


def test_generator_maps_typed_required_gameplay_failure_with_exact_reason(
    config, tmp_path
):
    generator = MapperatorinatorGenerator(
        config=config,
        run=_terminal_run(
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_REQUIRED_GAMEPLAY_FAILED",
                "context": REQUIRED_GAMEPLAY_FAILURE_CONTEXT,
            }
        ),
        verify_patch=lambda _home: None,
    )

    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")

    assert caught.value.code is ErrorCode.MANIA_REQUIRED_GAMEPLAY_FAILED
    assert caught.value.context == REQUIRED_GAMEPLAY_FAILURE_CONTEXT


@pytest.mark.parametrize(
    "context",
    [
        {**REQUIRED_GAMEPLAY_FAILURE_CONTEXT, "reason": "TOKEN_BUDGET_EXHAUSTED"},
        {**REQUIRED_GAMEPLAY_FAILURE_CONTEXT, "extra": 1},
        {**REQUIRED_GAMEPLAY_FAILURE_CONTEXT, "version": True},
        {**REQUIRED_GAMEPLAY_FAILURE_CONTEXT, "generatedTokenIndex": -1},
        {**REQUIRED_GAMEPLAY_FAILURE_CONTEXT, "message": ""},
    ],
)
def test_generator_rejects_malformed_required_gameplay_failure_context(
    config, tmp_path, context
):
    generator = MapperatorinatorGenerator(
        config=config,
        run=_terminal_run(
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_REQUIRED_GAMEPLAY_FAILED",
                "context": context,
            }
        ),
        verify_patch=lambda _home: None,
    )

    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")

    assert caught.value.code is ErrorCode.INFERENCE_PROTOCOL_FAILED


def test_generator_carries_the_generated_source_window_of_a_rejoin_failure(
    config, tmp_path
):
    context = {
        **PARTIAL_REJOIN_CONTEXT,
        "scope": "REJOINED",
        "earliestGeneratedSourceWindowId": 7,
    }
    generator = MapperatorinatorGenerator(
        config=config,
        run=_terminal_run(
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_PARTIAL_REJOIN_INVALID",
                "context": context,
            }
        ),
        verify_patch=lambda _home: None,
    )

    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")

    assert caught.value.context["earliestGeneratedSourceWindowId"] == 7


@pytest.mark.parametrize(
    "context",
    [
        {**PARTIAL_REJOIN_CONTEXT, "scope": "SOMETHING_ELSE"},
        {**PARTIAL_REJOIN_CONTEXT, "extra": 1},
        {key: value for key, value in PARTIAL_REJOIN_CONTEXT.items() if key != "scope"},
        {**PARTIAL_REJOIN_CONTEXT, "validationError": ""},
        {**PARTIAL_REJOIN_CONTEXT, "partialStartMs": None, "partialEndMs": None},
        {**PARTIAL_REJOIN_CONTEXT, "earliestGeneratedSourceWindowId": -1},
        {**PARTIAL_REJOIN_CONTEXT, "partialStartMs": 300, "partialEndMs": 250},
    ],
)
def test_generator_rejects_malformed_partial_rejoin_contexts(
    config, tmp_path, context
):
    generator = MapperatorinatorGenerator(
        config=config,
        run=_terminal_run(
            {
                "version": 1,
                "status": "FAILURE",
                "code": "MANIA_PARTIAL_REJOIN_INVALID",
                "context": context,
            }
        ),
        verify_patch=lambda _home: None,
    )

    with pytest.raises(WorkerError) as caught:
        generator.generate_map(_request(), tmp_path / "work")

    assert caught.value.code is ErrorCode.INFERENCE_PROTOCOL_FAILED


def test_repair_window_cap_covers_every_window_a_legal_song_can_produce():
    """꼬리 복구가 fault 창까지 되감으면 repairWindowIds 가 곡 전체만큼 길어질 수 있다.

    되감기 폭이 3창으로 고정돼 있을 때는 닿을 수 없던 상한이다. MAX_DURATION_MS
    를 올리면서 이 상한을 같이 올리지 않으면, 터미널 레코드 생성이 구조화되지
    않은 예외로 죽는다.
    """
    # osuT5 configs: src_seq_len 2048, hop_length 128, sample_rate 16000,
    # configs/inference/default.yaml lookback 0.5 + lookahead 0.4.
    ms_per_sequence = (2048 - 1) * 128 * 1000 / 16000
    ms_per_stride = int((2048 - 1) * 128 * 0.1) * 1000 / 16000
    max_windows = math.ceil(profile.MAX_DURATION_MS / ms_per_stride) + 1

    assert ms_per_sequence == 16_376.0
    assert max_windows <= mapperatorinator._MAX_TAIL_REPAIR_WINDOW_COUNT


def test_fake_generator_generates_standard_timing():
    result = FakeGenerator().generate_timing(
        TimingGenerationRequest(audio_path=Path("game.flac"), duration_ms=DURATION_MS),
        Path("unused"),
    )
    assert [(event.time_ms, event.bpm) for event in result.bpm_events] == [(0, 120.0)]
    assert result.mode == "STANDARD"
