"""저장된 S1/S2 결과로 S4만 다시 실행한다."""

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from chart_worker.analysis.snapshot import load_analysis_snapshot
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import parse_osu_file
from chart_worker.generation.params import REQUESTED_STAR
from chart_worker.hashing import sha256_file
from chart_worker.schema.chart import ChartDocument
from chart_worker.schema.keysound import KeysoundManifest
from chart_worker.schema.playtest_run import (
    AudioFileRef,
    PlaytestRunManifest,
    RunAudioRefs,
    RunChartRef,
)
from chart_worker.stages.s4_postprocess import run_postprocess
from chart_worker.stages.types import GeneratedVariant, StemStageResult


@dataclass(frozen=True, slots=True)
class PostprocessOptions:
    input_dir: Path
    output_dir: Path
    worker_version: str = "local-reprocess"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_dir", Path(self.input_dir).resolve())
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())
        if self.input_dir == self.output_dir:
            raise ValueError("input and output directories must be different")


@dataclass(frozen=True, slots=True)
class PostprocessResult:
    output_dir: Path
    manifest_path: Path
    chart_paths: tuple[Path, ...]


def _prepare(options: PostprocessOptions) -> PlaytestRunManifest:
    manifest_path = options.input_dir / "playtest-run-v1.json"
    if not manifest_path.is_file():
        raise ValueError(f"input manifest is missing: {manifest_path}")
    if options.output_dir.exists() and not options.output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {options.output_dir}")
    if options.output_dir.exists() and any(options.output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {options.output_dir}")
    options.output_dir.mkdir(parents=True, exist_ok=True)
    return PlaytestRunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def _copy_ref(reference: AudioFileRef, input_dir: Path, output_dir: Path) -> AudioFileRef:
    source = input_dir / reference.path
    if sha256_file(source) != reference.sha256:
        raise ValueError(f"source asset hash mismatch: {source}")
    target = output_dir / reference.path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return AudioFileRef(path=reference.path, sha256=sha256_file(target))


def _variants(manifest: PlaytestRunManifest, input_dir: Path) -> tuple[GeneratedVariant, ...]:
    variants = []
    for reference in manifest.charts:
        chart = ChartDocument.model_validate_json(
            (input_dir / reference.path).read_text(encoding="utf-8")
        )
        raw_path = input_dir / "raw" / f"{reference.key_mode}k-{reference.difficulty.lower()}.osu"
        beatmap = parse_osu_file(raw_path)
        variants.append(
            GeneratedVariant(
                key_mode=reference.key_mode,
                difficulty=reference.difficulty,
                requested_star=REQUESTED_STAR[reference.difficulty],
                raw_osu_path=raw_path,
                generated=GeneratedChart(
                    notes=beatmap.notes,
                    key_mode=beatmap.key_mode,
                    osu_text=raw_path.read_text(encoding="utf-8-sig"),
                    generator_name=chart.generator.name,
                    seed=chart.generator.seed,
                ),
            )
        )
    return tuple(variants)


def run_postprocess_only(options: PostprocessOptions) -> PostprocessResult:
    source_manifest = _prepare(options)
    analysis = load_analysis_snapshot(options.input_dir)
    game_ref = _copy_ref(source_manifest.audio.game, options.input_dir, options.output_dir)
    no_drums_ref = (
        _copy_ref(source_manifest.audio.no_drums, options.input_dir, options.output_dir)
        if source_manifest.audio.no_drums is not None
        else None
    )
    keys_ref = (
        _copy_ref(source_manifest.audio.keys, options.input_dir, options.output_dir)
        if source_manifest.audio.keys is not None
        else None
    )
    keysound_manifest = None
    keysound_path = None
    drum_onsets: tuple[int, ...] = ()
    if source_manifest.keysound_manifest_path is not None:
        source_path = options.input_dir / source_manifest.keysound_manifest_path
        keysound_manifest = KeysoundManifest.model_validate_json(
            source_path.read_text(encoding="utf-8")
        )
        keysound_path = options.output_dir / source_manifest.keysound_manifest_path
        keysound_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, keysound_path)
        drum_onsets = tuple(keysound_manifest.drum_onsets)

    results = run_postprocess(
        analysis,
        _variants(source_manifest, options.input_dir),
        StemStageResult(
            game_ref=game_ref,
            no_drums_ref=no_drums_ref,
            keys_ref=keys_ref,
            drum_onsets=drum_onsets,
            keysound_manifest=keysound_manifest,
            keysound_manifest_path=keysound_path,
        ),
        options.output_dir,
        worker_version=options.worker_version,
    )
    report_path = options.output_dir / "generation-report.json"
    report_path.write_text(
        json.dumps(
            {
                "reprocessedFrom": str(options.input_dir),
                "charts": [
                    {
                        "keyMode": result.document.key_mode,
                        "difficulty": result.document.difficulty,
                        "actualRating": result.document.metrics.project_rating,
                    }
                    for result in results
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = PlaytestRunManifest(
        run_id=uuid4(),
        title=source_manifest.title,
        generated_at=datetime.now(UTC),
        worker_version=options.worker_version,
        audio=RunAudioRefs(game=game_ref, no_drums=no_drums_ref, keys=keys_ref),
        charts=[
            RunChartRef(
                path=result.path.relative_to(options.output_dir).as_posix(),
                sha256=result.sha256,
                key_mode=result.document.key_mode,
                difficulty=result.document.difficulty,
            )
            for result in results
        ],
        keysound_manifest_path=(
            keysound_path.relative_to(options.output_dir).as_posix()
            if keysound_path is not None
            else None
        ),
        generation_report_path=report_path.relative_to(options.output_dir).as_posix(),
    )
    manifest_path = options.output_dir / "playtest-run-v1.json"
    manifest_path.write_text(
        manifest.model_dump_json(by_alias=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return PostprocessResult(
        output_dir=options.output_dir,
        manifest_path=manifest_path,
        chart_paths=tuple(result.path for result in results),
    )
