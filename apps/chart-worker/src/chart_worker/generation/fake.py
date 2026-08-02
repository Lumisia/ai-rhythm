"""GPU 없이 도는 가짜 생성기.

파이프라인 조립·후처리·검증·직렬화를 GPU 없이 12회 돌리기 위한 것이다.
채보 품질을 흉내내지 않는다 — 구조만 그럴듯하면 된다.
"""

import random
from dataclasses import dataclass
from pathlib import Path

from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import parse_osu_file
from chart_worker.generation.params import GenerationRequest
from chart_worker.schema.note import Chart, NoteEvent

NOTES_PER_BEAT: dict[str, int] = {
    "EASY": 1,
    "NORMAL": 2,
    "HARD": 4,
    "EXPERT": 6,
}

MIN_HOLD_MS = 60


def _timing_of(request: GenerationRequest) -> tuple[int, float]:
    """참조 .osu 에서 첫 박 시각과 한 박 길이를 읽는다."""
    if request.timing_osu_path is None:
        raise WorkerError(
            ErrorCode.CHART_GENERATION_FAILED,
            "the fake generator needs a timing .osu to place notes on",
        )
    beatmap = parse_osu_file(request.timing_osu_path)
    if not beatmap.bpm_events:
        raise WorkerError(
            ErrorCode.CHART_GENERATION_FAILED,
            "the timing .osu has no uninherited timing point",
        )
    event = beatmap.bpm_events[0]
    return event.time_ms, 60_000.0 / event.bpm


def synthesize_chart(request: GenerationRequest) -> Chart:
    """비트 격자 위에 결정적으로 노트를 흩뿌린다."""
    if request.duration_ms <= 0:
        raise WorkerError(
            ErrorCode.CHART_GENERATION_FAILED,
            "the fake generator needs duration_ms",
        )
    offset_ms, beat_ms = _timing_of(request)
    step_ms = beat_ms / NOTES_PER_BEAT[request.difficulty]
    rng = random.Random((request.seed or 0, request.key_mode, request.difficulty).__hash__())

    notes: list[NoteEvent] = []
    lane_free_at = dict.fromkeys(range(request.key_mode), -1)
    index = 0
    while True:
        time_ms = round(offset_ms + index * step_ms)
        index += 1
        if time_ms >= request.duration_ms:
            break
        if time_ms < 0:
            continue
        lane = rng.randrange(request.key_mode)
        if time_ms < lane_free_at[lane]:
            continue
        if rng.random() < request.hold_note_ratio:
            duration_ms = max(MIN_HOLD_MS, round(step_ms))
            if time_ms + duration_ms >= request.duration_ms:
                continue
            notes.append(
                NoteEvent(time_ms=time_ms, lane=lane, kind="HOLD", duration_ms=duration_ms)
            )
            lane_free_at[lane] = time_ms + duration_ms
        else:
            notes.append(NoteEvent(time_ms=time_ms, lane=lane))
            lane_free_at[lane] = time_ms + 1
    return sorted(notes, key=lambda note: (note.time_ms, note.lane))


@dataclass(frozen=True, slots=True)
class FakeGenerator:
    fixture_path: Path | None = None
    """주면 그 .osu 를 그대로 돌려준다. 골든 테스트용."""

    def __call__(self, request: GenerationRequest, workdir: Path) -> GeneratedChart:
        if self.fixture_path is not None:
            beatmap = parse_osu_file(self.fixture_path)
            if beatmap.key_mode != request.key_mode:
                raise WorkerError(
                    ErrorCode.CHART_GENERATION_FAILED,
                    f"fixture is {beatmap.key_mode}K but {request.key_mode}K was asked for",
                )
            return GeneratedChart(
                notes=beatmap.notes,
                key_mode=beatmap.key_mode,
                osu_text=self.fixture_path.read_text(encoding="utf-8-sig"),
                generator_name="fake-fixture",
                seed=request.seed,
            )

        notes = synthesize_chart(request)
        return GeneratedChart(
            notes=notes,
            key_mode=request.key_mode,
            osu_text="",
            generator_name="fake-synthetic",
            seed=request.seed,
        )
