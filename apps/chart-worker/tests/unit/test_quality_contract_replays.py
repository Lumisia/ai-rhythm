"""Optional replays against preserved, ignored campaign evidence.

The unit suite stays portable: when the local campaign is absent, these tests
skip with an explicit reason.  CI or a forensic workstation can opt in with
``CHART_WORKER_QUALITY_REPLAY_CAMPAIGN``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from chart_worker.analysis.gameplay_occupancy import gameplay_intervals
from chart_worker.generation.osu_parser import parse_osu_file


def _campaign() -> Path:
    raw = os.environ.get("CHART_WORKER_QUALITY_REPLAY_CAMPAIGN")
    if not raw:
        pytest.skip("CHART_WORKER_QUALITY_REPLAY_CAMPAIGN is not configured")
    path = Path(raw)
    if not (path / "campaign-state.json").is_file():
        pytest.skip(f"preserved campaign is unavailable: {path}")
    return path


def _song_output(campaign: Path, index: int) -> Path:
    state = json.loads((campaign / "campaign-state.json").read_text(encoding="utf-8"))
    song = next((song for song in state["songs"] if song["index"] == index), None)
    if song is None:
        pytest.skip(f"campaign has no song index {index}")
    output = campaign / "songs" / song["folderName"] / "output"
    if not output.is_dir():
        pytest.skip(f"preserved output is unavailable: {output}")
    return output


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preserved_hold_tail_replay_measures_vacancy_after_hold_release() -> None:
    output = _song_output(_campaign(), 6)
    beatmap = parse_osu_file(output / "raw" / "7k-easy.osu")

    interval = next(
        interval
        for interval in gameplay_intervals(
            beatmap.notes,
            start_ms=181_316,
            end_ms=188_880,
        )
        if interval.row_span_start_ms == 183_997
    )

    assert interval.unoccupied_start_ms == 186_557
    assert interval.end_ms == 188_880
    assert interval.row_span_duration_ms == 4_883
    assert interval.unoccupied_duration_ms == 2_323


def test_preserved_eight_chart_run_had_four_unique_hard_safe_4k_payloads() -> None:
    output = _song_output(_campaign(), 2)
    manifest = json.loads((output / "playtest-run-v2.json").read_text(encoding="utf-8"))
    evidence = json.loads((output / "generation-report.json").read_text(encoding="utf-8"))[
        "songSelectionEvidenceV3"
    ]

    assert len(manifest["charts"]) == 8
    assert {
        (missing["keyMode"], missing["difficulty"])
        for missing in manifest["missingCharts"]
    } == {(4, difficulty) for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")}

    candidates = {candidate["candidateId"]: candidate for candidate in evidence["candidates"]}
    assignment = {
        slot: candidate_id
        for slot, candidate_id in evidence["currentAssignment"].items()
        if slot.startswith("4K:")
    }
    assert len(assignment) == 4
    selected = [candidates[candidate_id] for candidate_id in assignment.values()]
    assert all(candidate["safety"]["hardSafe"] for candidate in selected)
    assert len({candidate["candidatePayloadSha256"] for candidate in selected}) == 4
    for candidate in selected:
        payload = output / candidate["candidatePayloadRef"]
        assert payload.is_file()
        assert _sha256(payload) == candidate["candidatePayloadSha256"]


def test_preserved_quality_guard_keeps_non_regressing_family_unchanged() -> None:
    output = _song_output(_campaign(), 5)
    report = json.loads((output / "generation-report.json").read_text(encoding="utf-8"))

    decisions = report["safeFamilyAssignments"]
    assert len(decisions) == 3
    assert all(decision["candidatesEvaluated"] > 4 for decision in decisions)
    assert all(not decision["changed"] for decision in decisions)
    assert all(
        decision["selectedAssignment"] == decision["currentAssignment"]
        for decision in decisions
    )
