"""S2: 세 키 모드와 네 난이도의 생성기 호출."""

from pathlib import Path

from chart_worker.generation.candidate_selection import CandidateParameters
from chart_worker.generation.mapperatorinator import ChartGenerator
from chart_worker.generation.osu_writer import notes_to_osu_mania
from chart_worker.generation.params import REQUESTED_STAR, GenerationRequest
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.stages.types import AnalysisStageResult, GeneratedVariant


def run_generation_variant(
    analysis: AnalysisStageResult,
    run_dir: Path,
    *,
    generator: ChartGenerator,
    key_mode: int,
    difficulty: str,
    attempt: int,
    parameters: CandidateParameters,
) -> GeneratedVariant:
    if attempt < 1:
        raise ValueError("attempt must be at least one")
    offset_ms = analysis.timing_candidate.points[0].time_ms
    bpm = analysis.timing_candidate.points[0].bpm
    candidate_dir = (
        run_dir
        / "raw"
        / "candidates"
        / f"{key_mode}k-{difficulty.lower()}"
        / f"attempt-{attempt}"
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)
    request = GenerationRequest(
        audio_path=analysis.normalized.path,
        key_mode=key_mode,
        difficulty=difficulty,
        seed=parameters.seed,
        timing_osu_path=analysis.timing_osu_path,
        cfg_scale=parameters.cfg_scale,
        requested_star=parameters.requested_star,
        duration_ms=analysis.normalized.duration_ms,
    )
    generated = generator(request, candidate_dir / "work")
    osu_text = generated.osu_text or notes_to_osu_mania(
        generated.notes,
        key_mode=key_mode,
        bpm=bpm,
        offset_ms=offset_ms,
        audio_filename=analysis.normalized.path.name,
        title=analysis.normalized.path.stem,
    )
    raw_path = candidate_dir / f"{key_mode}k-{difficulty.lower()}.osu"
    raw_path.write_text(osu_text, encoding="utf-8")
    return GeneratedVariant(
        key_mode=key_mode,
        difficulty=difficulty,
        requested_star=request.requested_star,
        raw_osu_path=raw_path,
        generated=generated,
        cfg_scale=request.cfg_scale,
        attempt=attempt,
    )


def run_generation(
    analysis: AnalysisStageResult,
    run_dir: Path,
    *,
    generator: ChartGenerator,
    seed: int,
) -> tuple[GeneratedVariant, ...]:
    """Generate the first candidate for all combinations in stable order."""
    variants = []
    for index, (key_mode, difficulty) in enumerate(
        (key_mode, difficulty) for key_mode in KEY_MODES for difficulty in DIFFICULTIES
    ):
        variants.append(
            run_generation_variant(
                analysis,
                run_dir,
                generator=generator,
                key_mode=key_mode,
                difficulty=difficulty,
                attempt=1,
                parameters=CandidateParameters(
                    seed=seed + index,
                    requested_star=REQUESTED_STAR[difficulty],
                    cfg_scale=1.0,
                ),
            )
        )
    return tuple(variants)
