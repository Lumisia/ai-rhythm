"""Recompute cross-key outro-family shadow findings for a stored batch.

This audit is intentionally read-only with respect to generated charts.  It
verifies manifest-bound audio/chart hashes before applying the same pure shadow
review used by the generation pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from chart_worker.validation.outro_family_review import (
    OutroChartView,
    review_outro_family,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_song_path(batch_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (batch_root / path).resolve()


def _latest_note_times(chart: dict[str, Any]) -> tuple[int, int]:
    notes = chart.get("notes")
    if not isinstance(notes, list) or not notes:
        return 0, 0
    starts: list[int] = []
    ends: list[int] = []
    for note in notes:
        if not isinstance(note, dict):
            continue
        time_ms = note.get("timeMs")
        if not isinstance(time_ms, (int, float)):
            continue
        start_ms = round(time_ms)
        duration_ms = note.get("durationMs")
        duration = (
            round(duration_ms)
            if isinstance(duration_ms, (int, float)) and duration_ms > 0
            else 0
        )
        starts.append(start_ms)
        ends.append(start_ms + duration)
    return (max(starts, default=0), max(ends, default=0))


def audit_batch(batch_state_path: Path) -> dict[str, Any]:
    batch_state = _read_json(batch_state_path)
    batch_root = batch_state_path.parent
    results: list[dict[str, Any]] = []
    hash_mismatches: list[dict[str, Any]] = []

    for song in batch_state.get("songs", []):
        if not isinstance(song, dict):
            continue
        song_root = _resolve_song_path(batch_root, str(song["outputPath"]))
        run_paths = [song_root / "playtest-run-v2.json", song_root / "playtest-run-v1.json"]
        run_path = next((path for path in run_paths if path.exists()), None)
        if run_path is None:
            results.append(
                {
                    "index": song.get("index"),
                    "title": Path(str(song.get("sourceName", ""))).stem,
                    "error": "PLAYTEST_RUN_MISSING",
                }
            )
            continue

        run = _read_json(run_path)
        title = str(run.get("title") or Path(str(song.get("sourceName", ""))).stem)
        audio = run.get("audio", {}).get("game", {})
        if isinstance(audio, dict) and audio.get("path") and audio.get("sha256"):
            audio_path = song_root / str(audio["path"])
            actual = _sha256(audio_path) if audio_path.exists() else None
            if actual != audio["sha256"]:
                hash_mismatches.append(
                    {
                        "index": song.get("index"),
                        "title": title,
                        "kind": "audio",
                        "path": str(audio_path),
                        "expected": audio["sha256"],
                        "actual": actual,
                    }
                )

        chart_rows: list[dict[str, Any]] = []
        views: list[OutroChartView] = []
        for chart_ref in run.get("charts", []):
            if not isinstance(chart_ref, dict):
                continue
            chart_path = song_root / str(chart_ref["path"])
            actual = _sha256(chart_path) if chart_path.exists() else None
            if actual != chart_ref.get("sha256"):
                hash_mismatches.append(
                    {
                        "index": song.get("index"),
                        "title": title,
                        "kind": "chart",
                        "path": str(chart_path),
                        "expected": chart_ref.get("sha256"),
                        "actual": actual,
                    }
                )
                continue
            chart = _read_json(chart_path)
            last_start, last_end = _latest_note_times(chart)
            row = {
                "keyMode": int(chart_ref["keyMode"]),
                "difficulty": str(chart_ref["difficulty"]).upper(),
                "lastNoteStartMs": last_start,
                "lastNoteEndMs": last_end,
            }
            chart_rows.append(row)
            views.append(
                OutroChartView(
                    key_mode=row["keyMode"],
                    difficulty=row["difficulty"],
                    last_note_start_ms=last_start,
                    last_note_end_ms=last_end,
                )
            )

        review = review_outro_family(tuple(views)).to_report()
        results.append(
            {
                "index": song.get("index"),
                "title": title,
                "runManifest": run_path.name,
                "charts": chart_rows,
                "review": review,
            }
        )

    findings = [
        {"index": result["index"], "title": result["title"], **finding}
        for result in results
        for finding in result.get("review", {}).get("findings", [])
    ]
    by_slot = Counter(
        f'{finding["keyMode"]}K-{finding["difficulty"]}' for finding in findings
    )
    return {
        "auditVersion": "outro-family-batch-audit-v1",
        "sourceBatchState": str(batch_state_path.resolve()),
        "sourceBatchStatus": batch_state.get("status"),
        "songCount": len(results),
        "chartCount": sum(len(result.get("charts", [])) for result in results),
        "hashMismatchCount": len(hash_mismatches),
        "hashMismatches": hash_mismatches,
        "findingCount": len(findings),
        "findingSongCount": len({finding["index"] for finding in findings}),
        "findingCountsBySlot": dict(sorted(by_slot.items())),
        "findings": findings,
        "songs": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_state", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_batch(args.batch_state)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
