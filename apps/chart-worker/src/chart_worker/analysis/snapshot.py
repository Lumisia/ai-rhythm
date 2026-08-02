"""S1 분석 결과를 후처리 재실행용으로 저장하고 복원한다."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from chart_worker.analysis.audio_io import load_audio
from chart_worker.analysis.beat import BeatGrid
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.timing import (
    TimingCandidate,
    TimingPoint,
    TimingSource,
    TimingStatus,
    match_times,
    project_beats,
)
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.hashing import sha256_file
from chart_worker.stages.types import AnalysisStageResult

METADATA_NAME = "analysis-v1.json"
ARRAYS_NAME = "onsets-v1.npz"


def _relative(path: Path, run_dir: Path) -> str:
    try:
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        raise ValueError(f"analysis asset is outside the run directory: {path}") from None


def save_analysis_snapshot(analysis: AnalysisStageResult, run_dir: Path) -> tuple[Path, Path]:
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = analysis_dir / ARRAYS_NAME
    np.savez_compressed(
        arrays_path,
        strength=analysis.onsets.strength,
        band_strength=analysis.onsets.band_strength,
    )
    normalized = asdict(analysis.normalized)
    normalized["path"] = _relative(analysis.normalized.path, run_dir)
    if not analysis.timing_osu_path.is_file():
        raise ValueError(f"timing osu is missing: {analysis.timing_osu_path}")
    if not analysis.timing_quality_report_path.is_file():
        raise ValueError(
            f"timing quality report is missing: {analysis.timing_quality_report_path}"
        )
    metadata = {
        "version": 1,
        "normalized": normalized,
        "beatGrid": asdict(analysis.beat_grid),
        "onsets": {
            "sampleRateHz": analysis.onsets.sample_rate_hz,
            "hopLength": analysis.onsets.hop_length,
            "onsetMs": list(analysis.onsets.onset_ms),
            "nFft": analysis.onsets.n_fft,
        },
        "timingOsuPath": _relative(analysis.timing_osu_path, run_dir),
        "timingSelection": {
            "source": analysis.timing_candidate.source.value,
            "points": [
                {
                    "timeMs": point.time_ms,
                    "bpm": point.bpm,
                    "meter": point.meter,
                    "startBeatIndex": point.start_beat_index,
                }
                for point in analysis.timing_candidate.points
            ],
            "qualityReportPath": _relative(analysis.timing_quality_report_path, run_dir),
        },
        "arraysPath": _relative(arrays_path, run_dir),
    }
    metadata_path = analysis_dir / METADATA_NAME
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path, arrays_path


def load_analysis_snapshot(run_dir: Path) -> AnalysisStageResult:
    metadata_path = run_dir / "analysis" / METADATA_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != 1:
        raise ValueError(f"unsupported analysis snapshot version: {metadata.get('version')}")

    normalized_data = dict(metadata["normalized"])
    audio_path = run_dir / normalized_data.pop("path")
    normalized = NormalizedAudio(path=audio_path, **normalized_data)
    actual_sha = sha256_file(audio_path)
    if actual_sha != normalized.sha256:
        raise ValueError(
            f"analysis audio hash mismatch: expected {normalized.sha256}, got {actual_sha}"
        )

    beat_data = dict(metadata["beatGrid"])
    beat_data["beat_ms"] = tuple(beat_data["beat_ms"])
    beat_data["downbeat_indices"] = tuple(beat_data["downbeat_indices"])
    grid = BeatGrid(**beat_data)

    arrays_path = run_dir / metadata["arraysPath"]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        strength = np.array(arrays["strength"], dtype=np.float64, copy=True)
        band_strength = np.array(arrays["band_strength"], dtype=np.float64, copy=True)
    if strength.ndim != 1 or band_strength.shape != (3, strength.size):
        raise ValueError("onset snapshot arrays have invalid shapes")
    onset_data = metadata["onsets"]
    onsets = OnsetAnalysis(
        sample_rate_hz=int(onset_data["sampleRateHz"]),
        hop_length=int(onset_data["hopLength"]),
        strength=strength,
        band_strength=band_strength,
        onset_ms=tuple(int(value) for value in onset_data["onsetMs"]),
        n_fft=int(onset_data["nFft"]),
    )
    timing_path = run_dir / metadata["timingOsuPath"]
    if not timing_path.is_file():
        raise ValueError(f"timing osu is missing: {timing_path}")
    timing_selection = metadata["timingSelection"]
    report_path = run_dir / timing_selection["qualityReportPath"]
    if not report_path.is_file():
        raise ValueError(f"timing quality report is missing: {report_path}")
    points = tuple(
        TimingPoint(
            time_ms=int(point["timeMs"]),
            bpm=float(point["bpm"]),
            meter=int(point["meter"]),
            start_beat_index=(
                int(point["startBeatIndex"])
                if point["startBeatIndex"] is not None
                else None
            ),
        )
        for point in timing_selection["points"]
    )
    if not points:
        raise ValueError("timing selection has no timing points")
    projected = project_beats(points, end_ms=normalized.duration_ms)
    metrics_20 = match_times(projected, grid.beat_ms, window_ms=20)
    metrics_50 = match_times(projected, grid.beat_ms, window_ms=50)
    passed = metrics_50.p95_abs_ms <= 30.0
    candidate = TimingCandidate(
        source=TimingSource(timing_selection["source"]),
        points=points,
        projected_beat_ms=projected,
        f1_20ms=metrics_20.f1,
        f1_50ms=metrics_50.f1,
        p95_abs_ms=metrics_50.p95_abs_ms,
        status=TimingStatus.PASS if passed else TimingStatus.FAIL,
        reasons=() if passed else ("p95 exceeds 30ms",),
    )
    return AnalysisStageResult(
        normalized=normalized,
        signal=load_audio(audio_path),
        beat_grid=grid,
        onsets=onsets,
        timing_candidate=candidate,
        timing_osu_path=timing_path,
        timing_quality_report_path=report_path,
    )
