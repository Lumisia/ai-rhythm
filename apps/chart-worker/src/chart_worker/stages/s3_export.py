"""S3: export Mapperatorinator output to chart-v1 without changing notes."""

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from chart_worker.hashing import sha256_file
from chart_worker.report.alignment import align_notes
from chart_worker.report.chart_metrics import build_chart_metrics
from chart_worker.schema.chart import BpmEvent, ChartDocument, GeneratorInfo, notes_to_chart_notes
from chart_worker.schema.types import lane_semantics
from chart_worker.stages.types import ExportedVariant, GeneratedVariant, PreparedAudio


def _stable_id(audio_sha: str, suffix: str):
    return uuid5(NAMESPACE_URL, f"{audio_sha}:{suffix}")


def run_export(
    prepared: PreparedAudio,
    generated_variants: tuple[GeneratedVariant, ...],
    run_dir: Path,
    *,
    worker_version: str,
) -> tuple[ExportedVariant, ...]:
    """Write exact raw notes/timing while calculating read-only metrics."""
    return tuple(
        _export_variant(
            prepared,
            variant,
            run_dir,
            worker_version=worker_version,
        )
        for variant in generated_variants
    )


def _export_variant(
    prepared: PreparedAudio,
    variant: GeneratedVariant,
    run_dir: Path,
    *,
    worker_version: str,
) -> ExportedVariant:
    timing = variant.generated.bpm_events
    if not timing:
        raise ValueError(
            f"generated {variant.key_mode}K {variant.difficulty} map has no timing events"
        )

    notes = variant.generated.notes
    normalized = prepared.normalized
    beat_ms = 60_000.0 / timing[0].bpm
    alignment = align_notes(notes, ())
    metrics = build_chart_metrics(
        notes,
        duration_ms=normalized.duration_ms,
        key_mode=variant.key_mode,
        beat_ms=beat_ms,
        alignment=alignment,
        moved_note_ratio=0.0,
    )
    audio_sha = normalized.sha256
    document = ChartDocument(
        chart_id=_stable_id(
            audio_sha,
            f"chart:{variant.key_mode}:{variant.difficulty}:{worker_version}",
        ),
        song_version_id=_stable_id(audio_sha, "song-version"),
        game_audio_asset_id=_stable_id(audio_sha, "game-audio"),
        audio_sha256=audio_sha,
        key_mode=variant.key_mode,
        difficulty=variant.difficulty,
        lane_semantics=lane_semantics(variant.key_mode),
        offset_ms=0,
        duration_ms=normalized.duration_ms,
        bpm_events=[BpmEvent(time_ms=event.time_ms, bpm=event.bpm) for event in timing],
        bpm_source=(
            "MAPPERATORINATOR"
            if variant.generated.generator_name.startswith("mapperatorinator")
            else "MANUAL"
        ),
        notes=notes_to_chart_notes(notes),
        auto_play_onsets=[],
        metrics=metrics,
        generator=GeneratorInfo(
            name=variant.generated.generator_name,
            version=variant.generated.generator_name,
            analysis_version="direct-map-timing",
            postprocess_version=worker_version,
            seed=variant.generated.seed or 0,
        ),
    )
    output_dir = run_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{variant.key_mode}k-{variant.difficulty.lower()}.chart.json"
    path.write_text(document.to_json(indent=2) + "\n", encoding="utf-8")
    return ExportedVariant(document=document, path=path, sha256=sha256_file(path))
