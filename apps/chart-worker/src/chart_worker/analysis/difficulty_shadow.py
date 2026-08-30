"""Offline DifficultyVector v2 recalculation for archived playtest charts."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from time import perf_counter_ns
from typing import Literal

import numpy as np

from chart_worker.analysis.chart_events import ChartEventIndex
from chart_worker.analysis.difficulty_vector import (
    DifficultyCalibration,
    measure_difficulty_vector,
)
from chart_worker.analysis.song_context import LocalTempoMap
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.hashing import sha256_file
from chart_worker.schema.chart import CamelModel, ChartDocument
from chart_worker.schema.note import NoteEvent
from chart_worker.schema.playtest_run import (
    PlaytestRunManifest,
    PlaytestRunManifestV2,
    PlaytestRunManifestV3,
)
from chart_worker.schema.types import DIFFICULTIES, Difficulty

DIAGNOSTIC_AXES: tuple[tuple[str, str], ...] = (
    ("density_strain", "densityStrain"),
    ("jack_strain", "jackStrain"),
    ("chord_load", "chordLoad"),
    ("ln_strain", "lnStrain"),
    ("coordination", "coordination"),
    ("peak_skill", "peakSkill"),
    ("bounded_stamina", "boundedStamina"),
    ("mean_hold_beats", "meanHoldBeats"),
    ("hold_occupancy_ratio", "holdOccupancyRatio"),
)

# ``peak_skill`` already integrates the official-style overall, individual,
# chord and HOLD terms. Re-fitting those component axes beside the aggregate
# double-counts correlated evidence: the archived 33-song LOSO run produced a
# false 6K NORMAL -> HARD inversion for exactly that reason. Keep the raw axes
# for review/adaptation, but calibrate the ordering aggregate only.
CALIBRATION_AXES: tuple[tuple[str, str], ...] = (
    ("peak_skill", "peakSkill"),
)


class ShadowChartRecord(CamelModel):
    song_id: str
    source_name: str
    audio_sha256: str
    key_mode: int
    difficulty: Difficulty
    current_rating: float
    vector_v2: dict[str, object]
    first_row_ms: int | None
    selected_seed: int | None
    source_report: str


class ShadowBatchReport(CamelModel):
    version: Literal["difficulty-shadow-v2"] = "difficulty-shadow-v2"
    song_count: int
    chart_count: int
    records: list[ShadowChartRecord]
    elapsed_ms: int


class ShadowCalibrationEvaluation(CamelModel):
    key_mode: int
    fold_count: int
    complete_song_count: int
    current_inversions: int
    v2_inversions: int
    current_narrow_pairs: int
    v2_narrow_pairs: int
    accepted: bool


def grouped_song_folds(
    records: tuple[ShadowChartRecord, ...],
    *,
    key_mode: int,
) -> tuple[tuple[tuple[ShadowChartRecord, ...], tuple[ShadowChartRecord, ...]], ...]:
    """Leave one audio SHA out, keeping every seed and batch copy together."""
    matching = tuple(record for record in records if record.key_mode == key_mode)
    by_audio: dict[str, list[ShadowChartRecord]] = defaultdict(list)
    for record in matching:
        by_audio[record.audio_sha256].append(record)
    return tuple(
        (
            tuple(
                record
                for audio_sha256, grouped in sorted(by_audio.items())
                if audio_sha256 != held_audio_sha256
                for record in grouped
            ),
            tuple(by_audio[held_audio_sha256]),
        )
        for held_audio_sha256 in sorted(by_audio)
    )


def _complete_audio_groups(
    records: tuple[ShadowChartRecord, ...],
    *,
    key_mode: int,
    minimum_song_count: int = 2,
) -> dict[str, dict[str, tuple[ShadowChartRecord, ...]]]:
    grouped: dict[str, dict[str, list[ShadowChartRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        if record.key_mode == key_mode:
            grouped[record.audio_sha256][record.difficulty].append(record)
    expected = set(DIFFICULTIES)
    incomplete = [
        audio_sha256
        for audio_sha256, family in grouped.items()
        if set(family) != expected
    ]
    if incomplete:
        raise ValueError(
            "calibration requires a complete family for every audio SHA: "
            + ", ".join(sorted(incomplete))
        )
    if len(grouped) < minimum_song_count:
        raise ValueError(
            f"calibration requires at least {minimum_song_count} complete songs"
        )
    return {
        audio_sha256: {
            difficulty: tuple(family[difficulty])
            for difficulty in DIFFICULTIES
        }
        for audio_sha256, family in grouped.items()
    }


def _median_axis(records: tuple[ShadowChartRecord, ...], report_name: str) -> float:
    values = [records[index].vector_v2.get(report_name) for index in range(len(records))]
    if not all(isinstance(value, (int, float)) for value in values):
        raise ValueError(f"shadow record is missing numeric axis: {report_name}")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _family_axes(
    family: dict[str, tuple[ShadowChartRecord, ...]],
) -> dict[str, dict[str, float]]:
    return {
        difficulty: {
            axis_name: _median_axis(family[difficulty], report_name)
            for axis_name, report_name in CALIBRATION_AXES
        }
        for difficulty in DIFFICULTIES
    }


def fit_calibration(
    records: tuple[ShadowChartRecord, ...],
    *,
    key_mode: int,
) -> DifficultyCalibration:
    """Fit non-negative axis weights from complete, audio-grouped ladders."""
    groups = _complete_audio_groups(records, key_mode=key_mode)
    families = tuple(_family_axes(family) for family in groups.values())
    axis_names = tuple(axis_name for axis_name, _report_name in CALIBRATION_AXES)
    samples = np.asarray(
        [
            [family[difficulty][axis_name] for axis_name in axis_names]
            for family in families
            for difficulty in DIFFICULTIES
        ],
        dtype=np.float64,
    )
    medians = np.median(samples, axis=0)
    iqrs = np.percentile(samples, 75, axis=0) - np.percentile(samples, 25, axis=0)
    variable = iqrs > 1e-9
    if not np.any(variable):
        raise ValueError("calibration has no variable axes")
    selected_names = tuple(
        axis_name for axis_name, keep in zip(axis_names, variable, strict=True) if keep
    )
    selected_medians = medians[variable]
    selected_iqrs = iqrs[variable]
    differences = []
    for family in families:
        for easier, harder in pairwise(DIFFICULTIES):
            easier_values = np.asarray(
                [family[easier][name] for name in selected_names],
                dtype=np.float64,
            )
            harder_values = np.asarray(
                [family[harder][name] for name in selected_names],
                dtype=np.float64,
            )
            differences.append((harder_values - easier_values) / selected_iqrs)
    matrix = np.asarray(differences, dtype=np.float64)
    target = np.ones(matrix.shape[0], dtype=np.float64)
    weights = np.linalg.lstsq(matrix, target, rcond=None)[0]
    weights = np.clip(weights, 0.0, None)
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-12:
        raise ValueError("calibration produced no positive weights")
    weights /= weight_sum
    return DifficultyCalibration(
        key_mode=key_mode,
        axis_names=selected_names,
        medians=tuple(float(value) for value in selected_medians),
        iqrs=tuple(float(value) for value in selected_iqrs),
        weights=tuple(float(value) for value in weights),
        complete_song_count=len(groups),
    )


def _calibrated_score(
    records: tuple[ShadowChartRecord, ...],
    calibration: DifficultyCalibration,
) -> float:
    values = {
        axis_name: _median_axis(records, report_name)
        for axis_name, report_name in CALIBRATION_AXES
    }
    return sum(
        ((values[name] - median) / iqr) * weight
        for name, median, iqr, weight in zip(
            calibration.axis_names,
            calibration.medians,
            calibration.iqrs,
            calibration.weights,
            strict=True,
        )
    )


def _inversion_and_narrow_count(values: tuple[float, ...]) -> tuple[int, int]:
    gaps = tuple(harder - easier for easier, harder in pairwise(values))
    return sum(gap < 0 for gap in gaps), sum(gap < 0.30 for gap in gaps)


def evaluate_calibration_loso(
    records: tuple[ShadowChartRecord, ...],
    *,
    key_mode: int,
) -> ShadowCalibrationEvaluation:
    """Measure held-out ordering; never fit and test on the same audio SHA."""
    groups = _complete_audio_groups(records, key_mode=key_mode)
    current_inversions = 0
    v2_inversions = 0
    current_narrow = 0
    v2_narrow = 0
    folds = grouped_song_folds(records, key_mode=key_mode)
    for train, test in folds:
        calibration = fit_calibration(train, key_mode=key_mode)
        held = _complete_audio_groups(
            test,
            key_mode=key_mode,
            minimum_song_count=1,
        )
        if len(held) != 1:
            raise ValueError("each held-out fold must contain exactly one audio SHA")
        family = next(iter(held.values()))
        current_values = tuple(
            float(np.median([record.current_rating for record in family[difficulty]]))
            for difficulty in DIFFICULTIES
        )
        v2_values = tuple(
            _calibrated_score(family[difficulty], calibration)
            for difficulty in DIFFICULTIES
        )
        current_counts = _inversion_and_narrow_count(current_values)
        v2_counts = _inversion_and_narrow_count(v2_values)
        current_inversions += current_counts[0]
        current_narrow += current_counts[1]
        v2_inversions += v2_counts[0]
        v2_narrow += v2_counts[1]
    return ShadowCalibrationEvaluation(
        key_mode=key_mode,
        fold_count=len(folds),
        complete_song_count=len(groups),
        current_inversions=current_inversions,
        v2_inversions=v2_inversions,
        current_narrow_pairs=current_narrow,
        v2_narrow_pairs=v2_narrow,
        accepted=v2_inversions <= current_inversions,
    )


def _archive_error(message: str) -> ValueError:
    return ValueError(f"ARCHIVE_INCONSISTENT: {message}")


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _archive_error(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise _archive_error(f"{path} must contain a JSON object")
    return value


def _safe_chart_path(song_dir: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise _archive_error("chartPath must be a non-empty string")
    candidate = (song_dir / relative).resolve()
    try:
        candidate.relative_to(song_dir.resolve())
    except ValueError as error:
        raise _archive_error(f"chartPath escapes song directory: {relative}") from error
    if not candidate.is_file():
        raise _archive_error(f"chartPath does not exist: {relative}")
    return candidate


def _manifest_refs(song_dir: Path) -> dict[str, tuple[int, str, str]]:
    candidates = (
        (song_dir / "playtest-run-v3.json", PlaytestRunManifestV3),
        (song_dir / "playtest-run-v2.json", PlaytestRunManifestV2),
        (song_dir / "playtest-run-v1.json", PlaytestRunManifest),
    )
    existing = [(path, model) for path, model in candidates if path.is_file()]
    if not existing:
        return {}
    if len(existing) != 1:
        names = ", ".join(path.name for path, _model in existing)
        raise _archive_error(f"multiple playtest manifests: {names}")
    manifest_path, manifest_model = existing[0]
    try:
        manifest = manifest_model.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise _archive_error(f"invalid playtest manifest: {manifest_path}") from error
    return {
        reference.path: (
            reference.key_mode,
            reference.difficulty,
            reference.sha256,
        )
        for reference in manifest.charts
    }


def _note_events(document: ChartDocument) -> list[NoteEvent]:
    return [
        NoteEvent(
            time_ms=note.time_ms,
            lane=note.lane,
            kind=note.kind,
            duration_ms=note.duration_ms,
        )
        for note in document.notes
    ]


def _record(
    batch_dir: Path,
    report_path: Path,
    chart_entry: object,
    manifest_refs: dict[str, tuple[int, str, str]],
    source_name: str,
) -> ShadowChartRecord:
    if not isinstance(chart_entry, dict):
        raise _archive_error(f"chart entry in {report_path} must be an object")
    song_dir = report_path.parent
    relative_path = chart_entry.get("chartPath")
    chart_path = _safe_chart_path(song_dir, relative_path)
    try:
        document = ChartDocument.model_validate_json(
            chart_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise _archive_error(f"invalid chart document: {chart_path}") from error

    report_key_mode = chart_entry.get("keyMode")
    report_difficulty = chart_entry.get("difficulty")
    if report_key_mode != document.key_mode or report_difficulty != document.difficulty:
        raise _archive_error(
            f"report/chart identity mismatch for {relative_path}: "
            f"{report_key_mode}K {report_difficulty} != "
            f"{document.key_mode}K {document.difficulty}"
        )

    if isinstance(relative_path, str) and manifest_refs:
        manifest_ref = manifest_refs.get(relative_path.replace("\\", "/"))
        if manifest_ref is None:
            raise _archive_error(f"manifest is missing chartPath: {relative_path}")
        key_mode, difficulty, expected_sha256 = manifest_ref
        if (key_mode, difficulty) != (document.key_mode, document.difficulty):
            raise _archive_error(f"manifest/chart identity mismatch: {relative_path}")
        if sha256_file(chart_path) != expected_sha256:
            raise _archive_error(f"manifest SHA mismatch: {relative_path}")

    profile = chart_entry.get("difficultyProfile")
    if isinstance(profile, dict) and isinstance(profile.get("projectRating"), (int, float)):
        current_rating = float(profile["projectRating"])
    else:
        current_rating = document.metrics.project_rating
    if abs(current_rating - document.metrics.project_rating) > 1e-6:
        raise _archive_error(f"rating mismatch: {relative_path}")

    notes = _note_events(document)
    event_index = ChartEventIndex.build(
        notes,
        document.key_mode,
        document.duration_ms,
    )
    tempo_map = LocalTempoMap(
        tuple(OsuBpmEvent(event.time_ms, event.bpm) for event in document.bpm_events)
    )
    vector = measure_difficulty_vector(event_index, tempo_map)
    source_report = report_path.relative_to(batch_dir).as_posix()
    song_id = report_path.parent.relative_to(batch_dir).as_posix()
    selected_seed = chart_entry.get("selectedSeed")
    if selected_seed is not None and not isinstance(selected_seed, int):
        raise _archive_error(f"selectedSeed must be an integer: {relative_path}")
    return ShadowChartRecord(
        song_id=song_id,
        source_name=source_name,
        audio_sha256=document.audio_sha256,
        key_mode=document.key_mode,
        difficulty=document.difficulty,
        current_rating=current_rating,
        vector_v2=vector.to_report(),
        first_row_ms=min((note.time_ms for note in document.notes), default=None),
        selected_seed=selected_seed,
        source_report=source_report,
    )


def recalculate_batch(batch_dir: Path, output_path: Path) -> ShadowBatchReport:
    """Recalculate saved charts only; never invoke a chart generator."""
    started = perf_counter_ns()
    batch_dir = batch_dir.resolve()
    output_path = output_path.resolve()
    if not batch_dir.is_dir():
        raise ValueError(f"batch directory does not exist: {batch_dir}")
    report_paths = tuple(sorted(batch_dir.rglob("generation-report.json")))
    if not report_paths:
        raise ValueError(f"no archived generation reports found in {batch_dir}")
    for report_path in report_paths:
        try:
            output_path.relative_to(report_path.parent.resolve())
        except ValueError:
            continue
        raise ValueError("output path must not be inside a song result directory")

    records: list[ShadowChartRecord] = []
    for report_path in report_paths:
        report = _read_object(report_path)
        charts = report.get("charts")
        if not isinstance(charts, list):
            raise _archive_error(f"charts must be a list: {report_path}")
        manifest_refs = _manifest_refs(report_path.parent)
        source_name = report.get("sourceName")
        if not isinstance(source_name, str) or not source_name:
            source_name = report_path.parent.name
        records.extend(
            _record(
                batch_dir,
                report_path,
                entry,
                manifest_refs,
                source_name,
            )
            for entry in charts
        )

    report = ShadowBatchReport(
        song_count=len(report_paths),
        chart_count=len(records),
        records=records,
        elapsed_ms=round((perf_counter_ns() - started) / 1_000_000),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        report.model_dump_json(by_alias=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
