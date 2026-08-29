import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from chart_worker.validation.mania_star_evidence import (
    PinnedOsuToolsManifestV1,
    parse_osu_tools_mania_difficulty,
    run_pinned_osu_tools_mania_difficulty,
    run_pinned_osu_tools_mania_difficulty_batch,
)


def _output(*, star=4.25, ruleset=3, errors=None, results_extra=None):
    result = {
        "ruleset_id": ruleset,
        "beatmap_id": 0,
        "beatmap": "fixture [Expert]",
        "mods": [],
        "attributes": {
            "star_rating": star,
            "max_combo": 1234,
            "great_hit_window": 34.5,
        },
    }
    if results_extra:
        result.update(results_extra)
    return json.dumps({"errors": errors or [], "results": [result]})


def test_pinned_osu_tools_output_is_preserved_as_unverified_offline_evidence():
    evidence = parse_osu_tools_mania_difficulty(
        _output(),
        input_osu_sha256="a" * 64,
        tool_binary_sha256="b" * 64,
        osu_tools_source_commit="c" * 40,
        osu_source_commit="d" * 40,
        calculator_version=20241007,
    )

    assert evidence.star_rating == 4.25
    assert evidence.verification_state == "PINNED_TOOL_OUTPUT_UNVERIFIED"
    assert evidence.authorizes_calibration_feature is False
    assert evidence.mods == ()
    assert len(evidence.attributes_sha256) == 64
    json.dumps(evidence.to_report(), allow_nan=False)


@pytest.mark.parametrize(
    "stdout",
    [
        "not-json",
        json.dumps({"errors": ["bad map"], "results": []}),
        _output(ruleset=0),
        _output(star=True),
        _output(star=float("nan")),
        json.dumps({"errors": [], "results": []}),
        json.dumps({"errors": [], "results": [{}, {}]}),
    ],
)
def test_osu_tools_parser_fails_closed_on_malformed_or_non_mania_output(stdout: str):
    with pytest.raises((TypeError, ValueError)):
        parse_osu_tools_mania_difficulty(
            stdout,
            input_osu_sha256="a" * 64,
            tool_binary_sha256="b" * 64,
            osu_tools_source_commit="c" * 40,
            osu_source_commit="d" * 40,
            calculator_version=20241007,
        )


def test_unexpected_calculator_version_is_rejected():
    with pytest.raises(ValueError, match="calculator version"):
        parse_osu_tools_mania_difficulty(
            _output(),
            input_osu_sha256="a" * 64,
            tool_binary_sha256="b" * 64,
            osu_tools_source_commit="c" * 40,
            osu_source_commit="d" * 40,
            calculator_version=20250000,
        )


def _manifest(tool: Path, mania_ruleset: Path) -> PinnedOsuToolsManifestV1:
    return PinnedOsuToolsManifestV1(
        tool_binary_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
        mania_ruleset_binary_sha256=hashlib.sha256(mania_ruleset.read_bytes()).hexdigest(),
        osu_tools_source_commit="c" * 40,
        osu_source_commit="d" * 40,
        calculator_version=20241007,
    )


def _workspace_files(case: str) -> tuple[Path, Path, Path]:
    root = Path(".data", "test-tmp", "mania-star-runner", case)
    root.mkdir(parents=True, exist_ok=True)
    return (
        root / "PerformanceCalculator.exe",
        root / "osu.Game.Rulesets.Mania.dll",
        root / "candidate.osu",
    )


def test_pinned_runner_hashes_exact_files_and_is_the_only_authorized_evidence(
    monkeypatch: pytest.MonkeyPatch,
):
    tool, mania_ruleset, osu = _workspace_files("authorized")
    tool.write_bytes(b"pinned-tool")
    mania_ruleset.write_bytes(b"pinned-mania-ruleset")
    osu.write_bytes(b"osu file format v14\n")
    observed = []

    def fake_run(command, **kwargs):
        observed.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=_output(), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    evidence = run_pinned_osu_tools_mania_difficulty(
        osu_path=osu,
        tool_executable=tool,
        mania_ruleset_binary=mania_ruleset,
        manifest=_manifest(tool, mania_ruleset),
    )

    assert observed[0][0] == [
        str(tool.resolve()),
        "difficulty",
        str(osu.resolve()),
        "-r:3",
        "-j",
    ]
    assert observed[0][1]["shell"] is False
    assert evidence.verification_state == "VERIFIED_PINNED_TOOL_EXECUTION"
    assert evidence.authorizes_calibration_feature is True
    assert evidence.input_osu_sha256 == hashlib.sha256(osu.read_bytes()).hexdigest()
    assert evidence.to_report()["authorizesCalibrationFeature"] is True


@pytest.mark.parametrize("tampered", ["tool", "ruleset"])
def test_pinned_runner_rejects_binary_hash_mismatch_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tampered: str,
):
    tool, mania_ruleset, osu = _workspace_files(f"tampered-{tampered}")
    tool.write_bytes(b"pinned-tool")
    mania_ruleset.write_bytes(b"pinned-mania-ruleset")
    osu.write_bytes(b"osu file format v14\n")
    manifest = _manifest(tool, mania_ruleset)
    (tool if tampered == "tool" else mania_ruleset).write_bytes(b"tampered")
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("must fail before execution")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="binary hash mismatch"):
        run_pinned_osu_tools_mania_difficulty(
            osu_path=osu,
            tool_executable=tool,
            mania_ruleset_binary=mania_ruleset,
            manifest=manifest,
        )
    assert called is False


@pytest.mark.parametrize(
    ("returncode", "stderr", "stdout", "message"),
    [
        (1, "failure", "", "nonzero"),
        (0, "warning", _output(), "stderr"),
        (0, "", "x" * (1024 * 1024 + 1), "too large"),
    ],
    ids=("nonzero", "stderr", "oversize"),
)
def test_pinned_runner_rejects_noncanonical_process_completion(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stderr: str,
    stdout: str,
    message: str,
):
    tool, mania_ruleset, osu = _workspace_files(f"process-{message.replace(' ', '-')}")
    tool.write_bytes(b"pinned-tool")
    mania_ruleset.write_bytes(b"pinned-mania-ruleset")
    osu.write_bytes(b"osu file format v14\n")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        ),
    )

    with pytest.raises(ValueError, match=message):
        run_pinned_osu_tools_mania_difficulty(
            osu_path=osu,
            tool_executable=tool,
            mania_ruleset_binary=mania_ruleset,
            manifest=_manifest(tool, mania_ruleset),
        )


def test_pinned_runner_rejects_input_changed_during_execution(
    monkeypatch: pytest.MonkeyPatch,
):
    tool, mania_ruleset, osu = _workspace_files("input-changed")
    tool.write_bytes(b"pinned-tool")
    mania_ruleset.write_bytes(b"pinned-mania-ruleset")
    osu.write_bytes(b"osu file format v14\n")

    def fake_run(*_args, **_kwargs):
        osu.write_bytes(b"changed")
        return SimpleNamespace(returncode=0, stdout=_output(), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="changed during"):
        run_pinned_osu_tools_mania_difficulty(
            osu_path=osu,
            tool_executable=tool,
            mania_ruleset_binary=mania_ruleset,
            manifest=_manifest(tool, mania_ruleset),
        )


def test_pinned_batch_keeps_each_result_bound_to_its_exact_input(
    monkeypatch: pytest.MonkeyPatch,
):
    tool, mania_ruleset, first = _workspace_files("batch-bound")
    second = first.with_name("second.osu")
    tool.write_bytes(b"pinned-tool")
    mania_ruleset.write_bytes(b"pinned-mania-ruleset")
    first.write_bytes(b"first osu")
    second.write_bytes(b"second osu")
    observed = []

    def fake_run(command, **kwargs):
        observed.append((command, kwargs))
        star = 1.25 if Path(command[2]).name == "candidate.osu" else 2.5
        return SimpleNamespace(returncode=0, stdout=_output(star=star), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    items = run_pinned_osu_tools_mania_difficulty_batch(
        osu_paths=(second, first),
        tool_executable=tool,
        mania_ruleset_binary=mania_ruleset,
        manifest=_manifest(tool, mania_ruleset),
        max_workers=2,
    )

    assert [item.osu_path.name for item in items] == ["candidate.osu", "second.osu"]
    assert [item.evidence.star_rating for item in items] == [1.25, 2.5]
    assert items[0].evidence.input_osu_sha256 == hashlib.sha256(first.read_bytes()).hexdigest()
    assert items[1].evidence.input_osu_sha256 == hashlib.sha256(second.read_bytes()).hexdigest()
    assert {Path(command[2]).resolve() for command, _kwargs in observed} == {
        first.resolve(),
        second.resolve(),
    }
    assert all(kwargs["shell"] is False for _command, kwargs in observed)


def test_pinned_batch_rejects_duplicate_resolved_input_before_execution(
    monkeypatch: pytest.MonkeyPatch,
):
    tool, mania_ruleset, osu = _workspace_files("batch-duplicate")
    tool.write_bytes(b"pinned-tool")
    mania_ruleset.write_bytes(b"pinned-mania-ruleset")
    osu.write_bytes(b"osu")
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("duplicate batch must fail before execution")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="unique"):
        run_pinned_osu_tools_mania_difficulty_batch(
            osu_paths=(osu, osu.resolve()),
            tool_executable=tool,
            mania_ruleset_binary=mania_ruleset,
            manifest=_manifest(tool, mania_ruleset),
        )
    assert called is False
