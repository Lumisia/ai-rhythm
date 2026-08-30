from __future__ import annotations

import json
from pathlib import Path

import pytest

from chart_worker.generation.diagnostic_fallback import (
    DiagnosticFallbackIdentity,
    DiagnosticRawCandidate,
    export_diagnostic_fallback,
    select_diagnostic_candidate,
)


def _gate_report(*, attack_gaps: int = 1, structure: str = "PASS") -> dict[str, object]:
    return {
        "action": "RETRY_MAP",
        "decisions": {
            "STRUCTURE": {"action": structure, "reasons": []},
            "TIMING_IDENTITY": {"action": "PASS", "reasons": []},
            "SONG_BOUNDS": {"action": "PASS", "reasons": []},
            "COVERAGE": {
                "action": "RETRY_MAP",
                "reasons": ["ATTACK_REQUIRED_MIDDLE_GAP"] * attack_gaps,
            },
        },
        "timing": {
            "coverageGaps": [
                {
                    "opportunity": {
                        "kind": "ATTACK_REQUIRED",
                        "beatCount": 16.0,
                        "strongAttackCount": 8,
                    }
                }
                for _ in range(attack_gaps)
            ],
            "overall": {"precision50": 0.9},
        },
    }


def _candidate(
    source: Path,
    *,
    key_mode: int = 4,
    difficulty: str = "EASY",
    seed: int = 0,
    attempt: int = 1,
    attack_gaps: int = 1,
    gate_report: dict[str, object] | None = None,
) -> DiagnosticRawCandidate:
    return DiagnosticRawCandidate.create(
        key_mode=key_mode,
        difficulty=difficulty,
        seed=seed,
        attempt=attempt,
        osu_text="osu file format v14\n\n[General]\nMode:3\n",
        source_workdir=source,
        gate_report=gate_report or _gate_report(attack_gaps=attack_gaps),
        attempt_errors=("quality rejected",),
        attempt_evidence=({"seed": seed, "reason": "QUALITY_GATE_RETRY"},),
    )


def _identity() -> DiagnosticFallbackIdentity:
    return DiagnosticFallbackIdentity(
        audio_sha256="a" * 64,
        timing_sha256="b" * 64,
        model_identity="OliBomby/Mapperatorinator-v32@local-snapshot",
        patch_set_id="patch-v27",
        hold_state_mode="incremental",
    )


def test_candidate_freezes_nested_gate_and_attempt_evidence(tmp_path: Path):
    gate = _gate_report()
    evidence = [{"nested": {"value": 1}}]
    candidate = DiagnosticRawCandidate.create(
        key_mode=4,
        difficulty="EASY",
        seed=0,
        attempt=1,
        osu_text="osu file format v14\n",
        source_workdir=tmp_path,
        gate_report=gate,
        attempt_errors=(),
        attempt_evidence=evidence,
    )

    gate["action"] = "PASS"
    evidence[0]["nested"]["value"] = 99

    assert candidate.gate_report()["action"] == "RETRY_MAP"
    assert candidate.attempt_evidence()[0]["nested"]["value"] == 1


def test_structure_or_timing_identity_failure_is_not_exportable(tmp_path: Path):
    with pytest.raises(ValueError, match="STRUCTURE"):
        _candidate(
            tmp_path,
            gate_report=_gate_report(structure="RETRY_MAP"),
        )


def test_song_bounds_failure_is_not_exportable(tmp_path: Path):
    gate = _gate_report()
    gate["decisions"]["SONG_BOUNDS"]["action"] = "RETRY_MAP"

    with pytest.raises(ValueError, match="SONG_BOUNDS"):
        _candidate(tmp_path, gate_report=gate)


def test_selection_prefers_fewer_attack_gaps_then_attempt_and_seed(tmp_path: Path):
    chosen = select_diagnostic_candidate(
        (
            _candidate(tmp_path / "a", seed=12, attempt=2, attack_gaps=2),
            _candidate(tmp_path / "b", seed=24, attempt=3, attack_gaps=1),
            _candidate(tmp_path / "c", seed=0, attempt=1, attack_gaps=1),
        ),
        key_mode=4,
        difficulty="EASY",
    )

    assert chosen.seed == 0
    assert chosen.attempt == 1


def test_selection_rejects_cross_variant_candidates(tmp_path: Path):
    with pytest.raises(ValueError, match="requested variant"):
        select_diagnostic_candidate(
            (_candidate(tmp_path, key_mode=6),),
            key_mode=4,
            difficulty="EASY",
        )


def test_export_is_playtest_only_atomic_and_idempotent(tmp_path: Path):
    run_dir = tmp_path / "run"
    source = run_dir / "raw" / "work" / "attempt-1"
    source.mkdir(parents=True)
    candidate = _candidate(source)
    validated: list[str] = []

    first = export_diagnostic_fallback(
        candidate,
        run_dir=run_dir,
        identity=_identity(),
        validate_osu=lambda text: validated.append(text),
    )
    second = export_diagnostic_fallback(
        candidate,
        run_dir=run_dir,
        identity=_identity(),
        validate_osu=lambda text: validated.append(text),
    )

    assert first == second
    assert first.path == run_dir / "diagnostic-raw-fallback" / "4k-easy" / "map.osu"
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["decision"] == "PLAYTEST_ONLY"
    assert "productionEligible" not in manifest
    assert manifest["sourceWorkdir"] == "raw/work/attempt-1"
    assert manifest["osuSha256"] == first.sha256
    assert manifest["identity"]["holdStateMode"] == "incremental"
    assert validated == [candidate.osu_text, candidate.osu_text]


def test_conflicting_existing_map_fails_without_overwrite(tmp_path: Path):
    run_dir = tmp_path / "run"
    source = run_dir / "raw" / "work" / "attempt-1"
    source.mkdir(parents=True)
    output = run_dir / "diagnostic-raw-fallback" / "4k-easy"
    output.mkdir(parents=True)
    target = output / "map.osu"
    target.write_text("conflict", encoding="utf-8")

    with pytest.raises(FileExistsError, match="conflicting diagnostic"):
        export_diagnostic_fallback(
            _candidate(source),
            run_dir=run_dir,
            identity=_identity(),
            validate_osu=lambda _text: None,
        )

    assert target.read_text(encoding="utf-8") == "conflict"
    assert not (output / "manifest-v1.json").exists()


def test_export_can_use_a_versioned_evidence_namespace(tmp_path: Path):
    run_dir = tmp_path / "run"
    source = run_dir / "raw" / "work" / "attempt-1"
    source.mkdir(parents=True)

    exported = export_diagnostic_fallback(
        _candidate(source),
        run_dir=run_dir,
        identity=_identity(),
        validate_osu=lambda _text: None,
        output_root_name="diagnostic-raw-fallback-v2",
    )

    assert exported.path == (
        run_dir / "diagnostic-raw-fallback-v2" / "4k-easy" / "map.osu"
    )


@pytest.mark.parametrize("name", ["", ".", "..", "nested/path", "nested\\path"])
def test_export_rejects_an_unsafe_evidence_namespace(tmp_path: Path, name: str):
    run_dir = tmp_path / "run"
    source = run_dir / "raw" / "work" / "attempt-1"
    source.mkdir(parents=True)

    with pytest.raises(ValueError, match="output_root_name"):
        export_diagnostic_fallback(
            _candidate(source),
            run_dir=run_dir,
            identity=_identity(),
            validate_osu=lambda _text: None,
            output_root_name=name,
        )


def test_reparse_escape_is_rejected_before_creating_a_variant_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_dir = tmp_path / "run"
    source = run_dir / "raw" / "work" / "attempt-1"
    source.mkdir(parents=True)
    output_root = run_dir / "diagnostic-raw-fallback"
    output_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    real_resolve = Path.resolve

    def simulated_reparse_resolve(path: Path, *args, **kwargs) -> Path:
        if path == output_root or output_root in path.parents:
            suffix = path.relative_to(output_root)
            return outside / suffix
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", simulated_reparse_resolve)

    with pytest.raises(ValueError, match="escapes run_dir"):
        export_diagnostic_fallback(
            _candidate(source),
            run_dir=run_dir,
            identity=_identity(),
            validate_osu=lambda _text: None,
        )

    assert not (output_root / "4k-easy").exists()


def test_source_workdir_must_stay_inside_run_dir(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="source_workdir"):
        export_diagnostic_fallback(
            _candidate(tmp_path / "outside"),
            run_dir=run_dir,
            identity=_identity(),
            validate_osu=lambda _text: None,
        )
