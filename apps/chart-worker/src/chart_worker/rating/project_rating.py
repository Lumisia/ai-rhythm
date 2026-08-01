from collections import Counter
from dataclasses import dataclass

import numpy as np

from chart_worker.schema.note import Chart, coerce_int

TARGET_RATING: dict[str, float] = {
    "EASY": 1.5,
    "NORMAL": 2.6,
    "HARD": 3.8,
    "EXPERT": 5.0,
}
_TIERS = ((2.0, "EASY"), (3.2, "NORMAL"), (4.5, "HARD"))


@dataclass(frozen=True, slots=True)
class RatingMetrics:
    note_count: int
    hold_count: int
    avg_nps: float
    p95_nps: float
    peak_nps: float
    chord_ratio: float
    max_jack: int
    hold_ratio: float
    rating: float
    tier: str


def tier_of(rating: float) -> str:
    for bound, name in _TIERS:
        if rating < bound:
            return name
    return "EXPERT"


def _windowed_nps(notes: Chart, duration_ms: int) -> np.ndarray:
    times = np.array([note.time_ms for note in notes], dtype=np.int64)
    starts = np.arange(0, max(duration_ms - 1000, 0) + 100, 100, dtype=np.int64)
    left = np.searchsorted(times, starts, side="left")
    right = np.searchsorted(times, starts + 1000, side="left")
    return (right - left).astype(float)


def _max_jack(notes: Chart) -> int:
    best = 0
    run: dict[int, int] = {}
    last_time: int | None = None
    lanes_at_time: set[int] = set()
    for note in notes:
        if last_time is not None and note.time_ms != last_time:
            for lane in list(run):
                if lane not in lanes_at_time:
                    run[lane] = 0
            lanes_at_time = set()
        run[note.lane] = run.get(note.lane, 0) + 1
        lanes_at_time.add(note.lane)
        best = max(best, run[note.lane])
        last_time = note.time_ms
    return best


def measure_rating(notes: Chart, duration_ms: int) -> RatingMetrics:
    """검증을 통과한 채보의 프로젝트 난이도를 계산한다.

    노트가 곡 길이를 벗어나면 예외를 던진다. 범위를 벗어난 노트를
    삭제하거나 잘라내는 일은 플레이 가능성 검증 단계의 책임이며,
    이 함수는 그 뒤에 호출된다.
    """
    if not notes:
        return RatingMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, "EASY")
    duration_ms = coerce_int(duration_ms, "duration_ms")
    if duration_ms <= 0:
        raise ValueError("duration_ms must be a positive integer for a non-empty chart")

    ordered = sorted(notes, key=lambda note: (note.time_ms, note.lane))
    for note in ordered:
        if note.time_ms >= duration_ms:
            raise ValueError("note time_ms must be less than duration_ms")
        if (
            note.kind == "HOLD"
            and note.duration_ms is not None
            and note.time_ms + note.duration_ms > duration_ms
        ):
            raise ValueError("HOLD end time must not exceed duration_ms")
    hold_count = sum(1 for note in ordered if note.kind == "HOLD")
    counts_by_time = Counter(note.time_ms for note in ordered)
    chord_notes = sum(count for count in counts_by_time.values() if count >= 2)
    nps = _windowed_nps(ordered, duration_ms)
    avg_nps = len(ordered) / (duration_ms / 1000.0)
    p95_nps = float(np.percentile(nps, 95))
    peak_nps = float(nps.max())
    chord_ratio = chord_notes / len(ordered)
    hold_ratio = hold_count / len(ordered)
    max_jack = _max_jack(ordered)
    rating = round(
        0.38 * p95_nps
        + 1.10 * chord_ratio
        + 0.15 * max(0, max_jack - 2)
        + 0.60 * hold_ratio
        + 0.10 * avg_nps,
        2,
    )
    return RatingMetrics(
        note_count=len(ordered),
        hold_count=hold_count,
        avg_nps=round(avg_nps, 3),
        p95_nps=round(p95_nps, 3),
        peak_nps=round(peak_nps, 3),
        chord_ratio=round(chord_ratio, 4),
        max_jack=max_jack,
        hold_ratio=round(hold_ratio, 4),
        rating=rating,
        tier=tier_of(rating),
    )
