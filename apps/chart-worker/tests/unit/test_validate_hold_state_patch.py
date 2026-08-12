from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    script_path = (
        Path(__file__).parents[2] / "scripts" / "validate_hold_state_patch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_hold_state_patch", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker_error(stderr: str) -> str:
    return json.dumps(
        {
            "code": "CHART_GENERATION_FAILED",
            "context": {"stderr": stderr},
            "message": "inference failed",
            "type": "WorkerError",
        }
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        (
            (
                "ValueError: HOLD_END without HOLD_START in lane 3: "
                "end={'timeMs': 1205}"
            ),
            {"kind": "ORPHAN_END", "lane": 3, "timeMs": 1205},
        ),
        (
            (
                "ValueError: overlapping HOLD_START in lane 4: "
                "previous={'timeMs': 94198}, current={'timeMs': 98193}"
            ),
            {"kind": "OVERLAP_START", "lane": 4, "timeMs": 98193},
        ),
        (
            (
                "ValueError: HOLD_START without HOLD_END in lane 1: "
                "start={'timeMs': 174328}"
            ),
            {"kind": "UNCLOSED_START", "lane": 1, "timeMs": 174328},
        ),
        (
            "ValueError: TAP while HOLD is active in lane 5",
            {"kind": "TAP_DURING_HOLD", "lane": 5},
        ),
    ],
)
def test_classify_hold_failure_extracts_the_failing_transition(
    stderr: str, expected: dict[str, int | str]
):
    module = _load_module()

    assert module.classify_hold_failure(_worker_error(stderr)) == expected


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        (
            (
                "ValueError: mania note group 790 has no gameplay column: "
                "type=HOLD_NOTE_END, time=117394"
            ),
            {
                "kind": "MISSING_GAMEPLAY_COLUMN",
                "groupId": 790,
                "eventType": "HOLD_NOTE_END",
                "timeMs": 117394,
            },
        ),
        (
            (
                "ValueError: exact duplicate mania group is not cross-window "
                "idempotent: groups 136 and 137, type=CIRCLE, lane=2, time=78680"
            ),
            {
                "kind": "NON_IDEMPOTENT_DUPLICATE_GROUP",
                "firstGroupId": 136,
                "secondGroupId": 137,
                "eventType": "CIRCLE",
                "lane": 2,
                "timeMs": 78680,
            },
        ),
        (
            "ValueError: incomplete mania group at end of stream: lane=2",
            {
                "kind": "INCOMPLETE_MANIA_GROUP",
                "lane": 2,
            },
        ),
    ],
)
def test_classify_generation_failure_covers_group_atomicity_failures(
    stderr: str, expected: dict[str, int | str]
):
    module = _load_module()

    assert module.classify_generation_failure(stderr) == expected


def test_build_results_document_binds_hashes_and_transition_matrix():
    module = _load_module()
    registration = {
        "version": "registration-v1",
        "runtime": {"fingerprintSha256": "runtime-sha"},
        "cases": [
            {
                "caseId": "a",
                "priorFailure": {"kind": "ORPHAN_END"},
                "keyMode": 4,
                "difficulty": "EASY",
                "seed": 1,
            },
            {
                "caseId": "b",
                "priorFailure": {"kind": "OVERLAP_START"},
                "keyMode": 7,
                "difficulty": "HARD",
                "seed": 10,
            },
        ],
    }
    run_state = {
        "version": "run-v1",
        "status": "COMPLETE_WITH_ISSUES",
        "runtime": {"fingerprintSha256": "runtime-sha"},
        "results": [
            {"caseId": "a", "status": "PASS", "elapsedSec": 2.5},
            {
                "caseId": "b",
                "status": "GENERATION_FAILURE",
                "elapsedSec": 3.5,
                "stderrText": "ValueError: HOLD_START without HOLD_END in lane 2: "
                "start={'timeMs': 900}",
            },
        ],
    }

    result = module.build_results_document(
        registration,
        run_state,
        registration_sha256="registration-sha",
        run_state_sha256="run-state-sha",
    )

    assert result["evidence"] == {
        "registrationSha256": "registration-sha",
        "runStateSha256": "run-state-sha",
        "runtimeFingerprintSha256": "runtime-sha",
    }
    assert result["counts"] == {
        "total": 2,
        "pass": 1,
        "generationFailure": 1,
        "otherFailure": 0,
    }
    assert result["transitionMatrix"] == {
        "ORPHAN_END": {"PASS": 1},
        "OVERLAP_START": {"UNCLOSED_START": 1},
    }


def test_rewrite_generation_overrides_changes_only_the_output_path(tmp_path: Path):
    module = _load_module()
    source = (
        "- audio_path='C:\\evidence\\game.flac'\n"
        "- output_path='C:\\old\\attempt-2'\n"
        "- keycount=7\n"
        "- difficulty=2.0\n"
        "- descriptors=['style/mixed rice','streams/bursts']\n"
        "- beatmap_path='C:\\evidence\\timing-reference.osu'\n"
        "- parallel=false\n"
        "- seed=22"
    )

    rewritten = module.rewrite_generation_overrides(source, tmp_path / "case-01")

    assert rewritten == [
        "audio_path='C:\\evidence\\game.flac'",
        f"output_path='{tmp_path / 'case-01'}'",
        "keycount=7",
        "difficulty=2.0",
        "descriptors=['style/mixed rice','streams/bursts']",
        "beatmap_path='C:\\evidence\\timing-reference.osu'",
        "parallel=false",
        "seed=22",
    ]


def test_bind_attempt_errors_maps_all_crashes_before_filtering_hold_failures():
    module = _load_module()
    attempts = [
        {"attempt": "attempt-1", "seed": 1, "producedOsu": False},
        {"attempt": "attempt-2", "seed": 13, "producedOsu": False},
        {"attempt": "attempt-3", "seed": 25, "producedOsu": True},
    ]
    errors = [
        _worker_error("RuntimeError: CUDA out of memory"),
        _worker_error(
            "ValueError: HOLD_END without HOLD_START in lane 2: "
            "end={'timeMs': 9876}"
        ),
    ]

    bound = module.bind_attempt_errors(attempts, errors)
    hold_cases = [item for item in bound if item["holdFailure"] is not None]

    assert len(hold_cases) == 1
    assert hold_cases[0]["attempt"] == "attempt-2"
    assert hold_cases[0]["seed"] == 13
    assert hold_cases[0]["holdFailure"] == {
        "kind": "ORPHAN_END",
        "lane": 2,
        "timeMs": 9876,
    }


def test_bind_attempt_errors_refuses_ambiguous_count_mismatch():
    module = _load_module()
    attempts = [
        {
            "attempt": "attempt-1",
            "attemptNumber": 1,
            "seed": 1,
            "producedOsu": True,
        },
        {"attempt": "attempt-2", "seed": 13, "producedOsu": False},
        {"attempt": "attempt-3", "seed": 25, "producedOsu": False},
    ]
    errors = [_worker_error("ValueError: HOLD_END without HOLD_START in lane 0")]

    with pytest.raises(ValueError, match="cannot bind HOLD error 1 to one attempt"):
        module.bind_attempt_errors(attempts, errors)


def test_bind_attempt_errors_uses_direct_seed_when_quality_errors_are_mixed_in():
    module = _load_module()
    attempts = [
        {
            "attempt": "attempt-1",
            "attemptNumber": 1,
            "seed": 7,
            "producedOsu": False,
        },
        {
            "attempt": "attempt-2",
            "attemptNumber": 2,
            "seed": 19,
            "producedOsu": True,
        },
        {
            "attempt": "attempt-3",
            "attemptNumber": 3,
            "seed": 31,
            "producedOsu": True,
        },
    ]
    errors = [
        _worker_error(
            "overrides: seed=7\n"
            "ValueError: HOLD_END without HOLD_START in lane 5: "
            "end={'timeMs': 240688}"
        ),
        json.dumps({"action": "RETRY_MAP", "seed": 19}),
        json.dumps({"reason": "RECOVERY_REJECTED"}),
    ]

    bound = module.bind_attempt_errors(attempts, errors)

    assert len(bound) == 1
    assert bound[0]["attempt"] == "attempt-1"
    assert bound[0]["seed"] == 7
    assert bound[0]["holdFailure"]["timeMs"] == 240688


def test_build_registration_binds_the_exact_failed_attempt_and_file_hashes(
    tmp_path: Path,
):
    module = _load_module()
    batch_root = tmp_path / "source-batch"
    song_dir = batch_root / "songs" / "25"
    audio_path = song_dir / "audio" / "game.flac"
    timing_path = song_dir / "audio" / "timing-reference.osu"
    _write_text(audio_path, "audio fixture")
    _write_text(timing_path, "osu file format v14\n")
    errors = [
        _worker_error("RuntimeError: unrelated crash"),
        _worker_error(
            "ValueError: HOLD_END without HOLD_START in lane 3: "
            "end={'timeMs': 1205}"
        ),
    ]
    report_path = song_dir / "generation-report.json"
    _write_text(
        report_path,
        json.dumps(
            {
                "sourceName": "fixture.wav",
                "charts": [
                    {
                        "keyMode": 7,
                        "difficulty": "HARD",
                        "attemptErrors": errors,
                    }
                ],
                "missingCharts": [],
            }
        ),
    )
    _write_text(
        batch_root / "batch-state.json",
        json.dumps(
            {
                "status": "COMPLETE_WITH_ISSUES",
                "songs": [{"originalIndex": 25, "outputPath": str(song_dir)}],
            }
        ),
    )
    for number, seed in ((1, 10), (2, 22), (3, 34)):
        attempt_dir = (
            song_dir
            / "raw"
            / "work"
            / "epoch-1"
            / "7k-hard"
            / f"attempt-{number}"
        )
        _write_text(
            attempt_dir / ".hydra-run" / ".hydra" / "overrides.yaml",
            "\n".join(
                [
                    f"- audio_path='{audio_path}'",
                    f"- output_path='{attempt_dir}'",
                    "- keycount=7",
                    "- difficulty=2.0",
                    "- descriptors=['style/mixed rice','streams/bursts']",
                    f"- beatmap_path='{timing_path}'",
                    "- parallel=false",
                    f"- seed={seed}",
                ]
            ),
        )
    _write_text(
        song_dir
        / "raw"
        / "work"
        / "epoch-1"
        / "7k-hard"
        / "attempt-3"
        / "generated.osu",
        "osu file format v14\n",
    )

    registration = module.build_registration(
        batch_root,
        target_root=tmp_path / "target",
        runtime={"head": "abc", "patchStatus": "APPLIED"},
        created_at="2026-08-11T00:00:00Z",
    )

    assert registration["counts"] == {
        "caseCount": 1,
        "ambiguousCombinationCount": 0,
        "ORPHAN_END": 1,
        "OVERLAP_START": 0,
        "UNCLOSED_START": 0,
    }
    case = registration["cases"][0]
    assert case["caseId"] == "o25-7k-hard-a02-s022"
    assert case["attempt"] == "attempt-2"
    assert case["seed"] == 22
    assert case["priorFailure"] == {
        "kind": "ORPHAN_END",
        "lane": 3,
        "timeMs": 1205,
    }
    assert case["audioSha256"] == _sha256(audio_path)
    assert case["timingReferenceSha256"] == _sha256(timing_path)
    assert case["generationReportSha256"] == _sha256(report_path)
    assert case["request"]["descriptors"] == (
        "['style/mixed rice','streams/bursts']"
    )


def test_build_replay_command_preserves_request_and_isolates_hydra_output(
    tmp_path: Path,
):
    module = _load_module()
    source_output = tmp_path / "old"
    overrides_path = source_output / ".hydra-run" / ".hydra" / "overrides.yaml"
    _write_text(
        overrides_path,
        "\n".join(
            [
                "- audio_path='C:\\evidence\\game.flac'",
                f"- output_path='{source_output}'",
                "- keycount=7",
                "- difficulty=2.0",
                "- parallel=false",
                "- seed=22",
            ]
        ),
    )
    output_path = tmp_path / "runs" / "case-01"

    command = module.build_replay_command(
        {"overridesPath": str(overrides_path), "runOutputPath": str(output_path)},
        mapper_python=Path("C:/mapper/.venv/Scripts/python.exe"),
    )

    assert command == [
        "C:\\mapper\\.venv\\Scripts\\python.exe",
        "inference.py",
        "-cn",
        "v32",
        f"hydra.run.dir='{output_path / '.hydra-run'}'",
        "audio_path='C:\\evidence\\game.flac'",
        f"output_path='{output_path}'",
        "keycount=7",
        "difficulty=2.0",
        "parallel=false",
        "seed=22",
    ]


def test_analyze_generated_output_recomputes_hold_integrity(tmp_path: Path):
    module = _load_module()
    output_path = tmp_path / "case"
    osu_path = output_path / "generated.osu"
    _write_text(
        osu_path,
        """osu file format v14

[General]
Mode:3

[Difficulty]
CircleSize:4

[TimingPoints]
0,500,4,2,1,100,1,0

[HitObjects]
64,192,100,1,0,0:0:0:0:
192,192,200,128,0,400:0:0:0:0:
320,192,500,1,0,0:0:0:0:
""",
    )

    result = module.analyze_generated_output(output_path)

    assert result["noteCount"] == 3
    assert result["tapCount"] == 2
    assert result["holdCount"] == 1
    assert result["firstNoteTimeMs"] == 100
    assert result["lastReleaseEndMs"] == 500
    assert result["maxStartGapMs"] == 300
    assert result["laneState"]["status"] == "PASS"
    assert result["laneState"]["sidecarEvidenceStatus"] == "UNAVAILABLE"
    assert result["osuSha256"] == _sha256(osu_path)


def test_execute_replay_case_runs_inference_and_persists_auditable_logs(
    tmp_path: Path,
):
    module = _load_module()
    mapper_home = tmp_path / "mapper"
    inference_path = mapper_home / "inference.py"
    _write_text(
        inference_path,
        """from pathlib import Path
import sys
value = next(arg for arg in sys.argv if arg.startswith('output_path='))
output = Path(value.split('=', 1)[1].strip(chr(39)))
output.mkdir(parents=True, exist_ok=True)
(output / 'generated.osu').write_text('''osu file format v14
[General]
Mode:3
[Difficulty]
CircleSize:4
[HitObjects]
64,192,100,1,0,0:0:0:0:
''', encoding='utf-8')
print('stub complete')
""",
    )
    source_attempt = tmp_path / "source" / "attempt-1"
    overrides_path = source_attempt / ".hydra-run" / ".hydra" / "overrides.yaml"
    _write_text(
        overrides_path,
        "\n".join(
            [
                "- audio_path='C:\\evidence\\game.flac'",
                f"- output_path='{source_attempt}'",
                "- keycount=4",
                "- difficulty=1.0",
                "- parallel=false",
                "- seed=1",
            ]
        ),
    )
    output_path = tmp_path / "runs" / "case-01"
    case = {
        "caseId": "case-01",
        "overridesPath": str(overrides_path),
        "runOutputPath": str(output_path),
    }

    result = module.execute_replay_case(
        case,
        mapper_home=mapper_home,
        mapper_python=Path(sys.executable),
        timeout_seconds=30,
    )

    assert result["status"] == "PASS"
    assert result["exitCode"] == 0
    assert result["analysis"]["noteCount"] == 1
    assert result["analysis"]["laneState"]["status"] == "PASS"
    assert Path(result["stdoutPath"]).read_text(encoding="utf-8").strip() == (
        "stub complete"
    )
    assert Path(result["stderrPath"]).read_text(encoding="utf-8") == ""
    assert result["commandSha256"] == hashlib.sha256(
        json.dumps(result["command"], ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def test_execute_replay_case_persists_structured_generation_failure(
    tmp_path: Path,
):
    module = _load_module()
    mapper_home = tmp_path / "mapper"
    _write_text(
        mapper_home / "inference.py",
        """import sys
print('ValueError: incomplete mania group at end of stream: lane=2', file=sys.stderr)
raise SystemExit(1)
""",
    )
    source_attempt = tmp_path / "source" / "attempt-1"
    overrides_path = source_attempt / ".hydra-run" / ".hydra" / "overrides.yaml"
    _write_text(
        overrides_path,
        "\n".join(
            [
                "- audio_path='C:\\evidence\\game.flac'",
                f"- output_path='{source_attempt}'",
                "- keycount=4",
                "- difficulty=1.0",
                "- parallel=false",
                "- seed=1",
            ]
        ),
    )
    case = {
        "caseId": "case-failure",
        "overridesPath": str(overrides_path),
        "runOutputPath": str(tmp_path / "runs" / "case-failure"),
    }

    result = module.execute_replay_case(
        case,
        mapper_home=mapper_home,
        mapper_python=Path(sys.executable),
        timeout_seconds=30,
    )

    assert result["status"] == "GENERATION_FAILURE"
    assert result["failureClass"] == "INCOMPLETE_MANIA_GROUP"
    assert result["generationFailure"] == {
        "kind": "INCOMPLETE_MANIA_GROUP",
        "lane": 2,
    }
    assert result["holdFailure"] is None


def test_verify_case_inputs_stops_when_a_registered_input_changes(tmp_path: Path):
    module = _load_module()
    paths = {}
    for name in ("generationReport", "overrides", "audio", "timingReference"):
        path = tmp_path / f"{name}.dat"
        _write_text(path, name)
        paths[name] = path
    case = {
        "generationReportPath": str(paths["generationReport"]),
        "generationReportSha256": _sha256(paths["generationReport"]),
        "overridesPath": str(paths["overrides"]),
        "overridesSha256": _sha256(paths["overrides"]),
        "audioPath": str(paths["audio"]),
        "audioSha256": _sha256(paths["audio"]),
        "timingReferencePath": str(paths["timingReference"]),
        "timingReferenceSha256": _sha256(paths["timingReference"]),
    }
    module.verify_case_inputs(case)
    _write_text(paths["audio"], "mutated")

    with pytest.raises(ValueError, match="audio SHA-256 changed"):
        module.verify_case_inputs(case)
