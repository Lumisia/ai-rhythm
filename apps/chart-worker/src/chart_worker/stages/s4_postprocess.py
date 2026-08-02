"""S4: 분석 주석, 후처리, 검증, chart-v1 기록."""

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from chart_worker.analysis.beat import bpm_events_of
from chart_worker.analysis.onset import annotate_notes
from chart_worker.hashing import sha256_file
from chart_worker.postprocess.difficulty_solver import solve_difficulty
from chart_worker.postprocess.lane_conversion import convert_lanes
from chart_worker.report.alignment import align_notes
from chart_worker.report.chart_metrics import build_chart_metrics
from chart_worker.schema.chart import (
    ChartDocument,
    GeneratorInfo,
    notes_to_chart_notes,
)
from chart_worker.schema.types import lane_semantics
from chart_worker.stages.types import (
    AnalysisStageResult,
    GeneratedVariant,
    PostprocessedVariant,
    PostprocessReports,
    StemStageResult,
)
from chart_worker.validation.playability import validate_and_recover


def _stable_id(audio_sha: str, suffix: str):
    return uuid5(NAMESPACE_URL, f"{audio_sha}:{suffix}")


def run_postprocess(
    analysis: AnalysisStageResult,
    generated_variants: tuple[GeneratedVariant, ...],
    stems: StemStageResult,
    run_dir: Path,
    *,
    worker_version: str,
) -> tuple[PostprocessedVariant, ...]:
    beat_ms = 60_000.0 / analysis.beat_grid.bpm
    audio_sha = analysis.normalized.sha256
    output_dir = run_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for variant in generated_variants:
        annotated = annotate_notes(
            variant.generated.notes,
            onsets=analysis.onsets,
            grid=analysis.beat_grid,
        )
        conversion = convert_lanes(
            annotated,
            key_mode=variant.key_mode,
            difficulty=variant.difficulty,
        )
        difficulty = solve_difficulty(
            conversion.notes,
            duration_ms=analysis.normalized.duration_ms,
            difficulty=variant.difficulty,
            beat_ms=beat_ms,
        )
        playability = validate_and_recover(
            difficulty.notes,
            key_mode=variant.key_mode,
            difficulty=variant.difficulty,
            duration_ms=analysis.normalized.duration_ms,
            beat_ms=beat_ms,
        )
        alignment = align_notes(playability.notes, stems.drum_onsets)
        metrics = build_chart_metrics(
            playability.notes,
            duration_ms=analysis.normalized.duration_ms,
            key_mode=variant.key_mode,
            beat_ms=beat_ms,
            alignment=alignment,
            moved_note_ratio=conversion.moved_note_ratio,
        )
        document = ChartDocument(
            chart_id=_stable_id(
                audio_sha,
                f"chart:{variant.key_mode}:{variant.difficulty}:{worker_version}",
            ),
            song_version_id=_stable_id(audio_sha, "song-version"),
            game_audio_asset_id=_stable_id(audio_sha, "game-audio"),
            audio_sha256=stems.game_ref.sha256,
            key_mode=variant.key_mode,
            difficulty=variant.difficulty,
            lane_semantics=lane_semantics(variant.key_mode),
            offset_ms=analysis.beat_grid.beat_ms[0],
            duration_ms=analysis.normalized.duration_ms,
            bpm_events=bpm_events_of(analysis.beat_grid),
            bpm_source="BEAT_THIS",
            notes=notes_to_chart_notes(playability.notes),
            auto_play_onsets=list(alignment.auto_play_onsets),
            metrics=metrics,
            generator=GeneratorInfo(
                name=variant.generated.generator_name,
                version=variant.generated.generator_name,
                analysis_version="beat-this-final0+librosa",
                postprocess_version=worker_version,
                seed=variant.generated.seed or 0,
            ),
        )
        path = output_dir / f"{variant.key_mode}k-{variant.difficulty.lower()}.chart.json"
        path.write_text(document.to_json(indent=2) + "\n", encoding="utf-8")
        results.append(
            PostprocessedVariant(
                document=document,
                path=path,
                sha256=sha256_file(path),
                reports=PostprocessReports(
                    conversion=conversion,
                    difficulty=difficulty,
                    playability=playability,
                    alignment=alignment,
                ),
            )
        )
    return tuple(results)
