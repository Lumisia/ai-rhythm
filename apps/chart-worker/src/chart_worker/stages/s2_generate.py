"""S2: 세 키 모드와 네 난이도의 생성기 호출."""

from pathlib import Path

from chart_worker.generation.mapperatorinator import ChartGenerator
from chart_worker.generation.osu_writer import notes_to_osu_mania
from chart_worker.generation.params import GenerationRequest
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.stages.types import AnalysisStageResult, GeneratedVariant


def run_generation(
    analysis: AnalysisStageResult,
    run_dir: Path,
    *,
    generator: ChartGenerator,
    seed: int,
) -> tuple[GeneratedVariant, ...]:
    variants = []
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    offset_ms = analysis.beat_grid.beat_ms[0]

    for index, (key_mode, difficulty) in enumerate(
        (key_mode, difficulty) for key_mode in KEY_MODES for difficulty in DIFFICULTIES
    ):
        request = GenerationRequest(
            audio_path=analysis.normalized.path,
            key_mode=key_mode,
            difficulty=difficulty,
            seed=seed + index,
            timing_osu_path=analysis.timing_osu_path,
            duration_ms=analysis.normalized.duration_ms,
        )
        workdir = raw_dir / "work" / f"{key_mode}k-{difficulty.lower()}"
        generated = generator(request, workdir)
        osu_text = generated.osu_text or notes_to_osu_mania(
            generated.notes,
            key_mode=key_mode,
            bpm=analysis.beat_grid.bpm,
            offset_ms=offset_ms,
            audio_filename=analysis.normalized.path.name,
            title=analysis.normalized.path.stem,
        )
        raw_path = raw_dir / f"{key_mode}k-{difficulty.lower()}.osu"
        raw_path.write_text(osu_text, encoding="utf-8")
        variants.append(
            GeneratedVariant(
                key_mode=key_mode,
                difficulty=difficulty,
                requested_star=request.requested_star,
                raw_osu_path=raw_path,
                generated=generated,
            )
        )
    return tuple(variants)
