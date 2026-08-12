"""Read-only audit of HARD/EXPERT post-first gaps in a completed batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from chart_worker.analysis.song_context import LocalTempoMap
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.validation.intro_phrase_family import (
    IntroPhraseChartView,
    review_intro_phrase_pair,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    return parser


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_evidence(song_dir: Path, *, report_sha256: str) -> tuple[str, str]:
    manifest_path = next(
        (
            path
            for path in (
                song_dir / "playtest-run-v2.json",
                song_dir / "playtest-run-v1.json",
            )
            if path.is_file()
        ),
        None,
    )
    if manifest_path is None:
        raise ValueError(f"playtest manifest is missing: {song_dir}")

    manifest = _load_json(manifest_path)
    generation_report = manifest.get("generationReport")
    if isinstance(generation_report, dict):
        declared_report_sha = generation_report.get("sha256")
        if declared_report_sha != report_sha256:
            raise ValueError(
                "playtest manifest generation report SHA-256 does not match "
                f"{song_dir / 'generation-report.json'}"
            )
    audio = manifest.get("audio")
    audio_sha256: object | None = None
    if isinstance(audio, dict):
        game = audio.get("game")
        asset = audio.get("asset")
        if isinstance(game, dict):
            audio_sha256 = game.get("sha256")
        elif isinstance(asset, dict):
            audio_sha256 = asset.get("sha256")
        else:
            audio_sha256 = audio.get("sha256")
    if not isinstance(audio_sha256, str) or len(audio_sha256) != 64:
        raise ValueError(f"canonical game audio SHA-256 is missing: {manifest_path}")
    return audio_sha256, _sha256(manifest_path)


def _chart_view(
    song_dir: Path,
    report: dict[str, object],
    *,
    key_mode: int,
    difficulty: str,
) -> tuple[IntroPhraseChartView, LocalTempoMap | None]:
    chart_reports = report.get("charts", [])
    chart_report = next(
        (
            item
            for item in chart_reports
            if item.get("keyMode") == key_mode and item.get("difficulty") == difficulty
        ),
        None,
    )
    if chart_report is None:
        return IntroPhraseChartView(key_mode, difficulty, None, None, None), None

    chart_path = song_dir / chart_report["chartPath"]
    chart = _load_json(chart_path)
    rows = tuple(sorted({int(note["timeMs"]) for note in chart.get("notes", [])}))
    first_row_ms = rows[0] if rows else None
    second_row_ms = rows[1] if len(rows) >= 2 else None
    bpm_events = tuple(
        OsuBpmEvent(time_ms=int(event["timeMs"]), bpm=float(event["bpm"]))
        for event in chart.get("bpmEvents", [])
    )
    tempo_map = LocalTempoMap(bpm_events) if bpm_events else None
    gap_beats = (
        round(tempo_map.beats_between(first_row_ms, second_row_ms), 6)
        if tempo_map is not None
        and first_row_ms is not None
        and second_row_ms is not None
        else None
    )
    contract = report.get("introStartContract") or {}
    intro_candidate = next(
        (
            item
            for item in contract.get("candidates", [])
            if item.get("keyMode") == key_mode and item.get("difficulty") == difficulty
        ),
        None,
    )
    return (
        IntroPhraseChartView(
            key_mode=key_mode,
            difficulty=difficulty,
            first_row_ms=first_row_ms,
            second_row_ms=second_row_ms,
            post_first_gap_beats=gap_beats,
            first_row_audio_supported=(
                intro_candidate.get("audioSupported")
                if intro_candidate is not None
                else None
            ),
            candidate_id=(
                f"{chart_report['chartPath']}@sha256:{_sha256(chart_path)}"
            ),
            seed=chart_report.get("selectedSeed", chart_report.get("seed")),
            attempt=chart_report.get("attemptCount"),
        ),
        tempo_map,
    )


def _runtime_review(
    report: dict[str, object],
    *,
    key_mode: int,
) -> dict[str, object] | None:
    reviews = report.get("introPhraseFamilyReviews", [])
    if not isinstance(reviews, list):
        return None
    for item in reviews:
        if not isinstance(item, dict):
            continue
        hard = item.get("hard")
        expert = item.get("expert")
        if (
            isinstance(hard, dict)
            and isinstance(expert, dict)
            and hard.get("keyMode") == key_mode
            and expert.get("keyMode") == key_mode
        ):
            return item
    return None


def audit_batch(batch_root: Path) -> dict[str, object]:
    batch_root = batch_root.resolve()
    state_path = batch_root / "batch-state.json"
    state = _load_json(state_path)
    batch_state_sha256 = _sha256(state_path)
    rows: list[dict[str, object]] = []
    for song in state.get("songs", []):
        output_path = song.get("outputPath")
        if not output_path:
            continue
        song_dir = Path(output_path)
        if not song_dir.is_absolute():
            song_dir = batch_root / song_dir
        report_path = song_dir / "generation-report.json"
        if not report_path.is_file():
            continue
        report = _load_json(report_path)
        generation_report_sha256 = _sha256(report_path)
        audio_sha256, manifest_sha256 = _manifest_evidence(
            song_dir,
            report_sha256=generation_report_sha256,
        )
        for key_mode in (4, 6, 7):
            hard, hard_tempo = _chart_view(
                song_dir,
                report,
                key_mode=key_mode,
                difficulty="HARD",
            )
            expert, expert_tempo = _chart_view(
                song_dir,
                report,
                key_mode=key_mode,
                difficulty="EXPERT",
            )
            tempo_map = hard_tempo or expert_tempo
            start_delta_beats = (
                round(
                    tempo_map.beats_between(
                        min(hard.first_row_ms, expert.first_row_ms),
                        max(hard.first_row_ms, expert.first_row_ms),
                    ),
                    6,
                )
                if tempo_map is not None
                and hard.first_row_ms is not None
                and expert.first_row_ms is not None
                else None
            )
            review = review_intro_phrase_pair(
                hard,
                expert,
                start_delta_beats=start_delta_beats,
            )
            published_review = review.to_report()
            runtime_review = _runtime_review(report, key_mode=key_mode)
            effective_review = runtime_review or published_review
            rows.append(
                {
                    "batchStateSha256": batch_state_sha256,
                    "songIndex": song.get("index"),
                    "sourceName": song.get("sourceName"),
                    "generationReportSha256": generation_report_sha256,
                    "playtestManifestSha256": manifest_sha256,
                    "audioSha256": audio_sha256,
                    "keyMode": key_mode,
                    "runtimeReview": runtime_review,
                    "publishedReview": published_review,
                    "review": effective_review,
                    "reviewSource": (
                        "GENERATION_RUNTIME"
                        if runtime_review is not None
                        else "PUBLISHED_RECONSTRUCTION"
                    ),
                }
            )

    status_counts = Counter(row["review"]["status"] for row in rows)
    reason_counts = Counter(row["review"]["reason"] for row in rows)
    runtime_status_counts = Counter(
        row["runtimeReview"]["status"]
        for row in rows
        if row["runtimeReview"] is not None
    )
    published_status_counts = Counter(
        row["publishedReview"]["status"] for row in rows
    )
    return {
        "version": "intro-phrase-family-batch-audit-v2",
        "batchRoot": str(batch_root),
        "batchStateSha256": batch_state_sha256,
        "batchStartedAt": state.get("startedAt"),
        "batchFinishedAt": state.get("finishedAt"),
        "songCount": len({row["songIndex"] for row in rows}),
        "pairCount": len(rows),
        "statusCounts": dict(sorted(status_counts.items())),
        "runtimeStatusCounts": dict(sorted(runtime_status_counts.items())),
        "publishedStatusCounts": dict(sorted(published_status_counts.items())),
        "reasonCounts": dict(sorted(reason_counts.items())),
        "defects": [row for row in rows if row["review"]["status"] == "DEFECT"],
        "reviews": [row for row in rows if row["review"]["status"] == "REVIEW"],
        "rows": rows,
    }


def main() -> int:
    args = _parser().parse_args()
    result = audit_batch(args.batch_root)
    printable = (
        {
            key: value
            for key, value in result.items()
            if key not in {"rows"}
        }
        if args.summary_only
        else result
    )
    serialized = json.dumps(printable, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        sys.stdout.buffer.write(serialized.encode("utf-8"))
    else:
        args.output.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
