"""채보 시각과 드럼 onset의 일대일 정렬."""

from dataclasses import dataclass

from chart_worker.schema.note import Chart


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    matched_pairs: tuple[tuple[int, int], ...]
    auto_play_onsets: tuple[int, ...]
    drum_coverage: float
    drum_precision: float
    mean_abs_err_ms: float


def align_notes(
    notes: Chart,
    drum_onsets: tuple[int, ...],
    *,
    snap_window_ms: int = 50,
) -> AlignmentReport:
    """가장 가까운 조합부터 고르는 결정적 일대일 매칭.

    같은 시각의 화음은 하나의 음악 이벤트다. 한 드럼 타격을 화음 노트
    수만큼 중복 매칭하면 coverage와 precision이 부풀기 때문이다.
    """
    if snap_window_ms < 0:
        raise ValueError("snap_window_ms must be non-negative")

    note_times = tuple(sorted({note.time_ms for note in notes}))
    onsets = tuple(sorted(set(drum_onsets)))
    candidates = sorted(
        (abs(note_ms - onset_ms), note_ms, onset_ms)
        for note_ms in note_times
        for onset_ms in onsets
        if abs(note_ms - onset_ms) <= snap_window_ms
    )
    used_notes: set[int] = set()
    used_onsets: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, note_ms, onset_ms in candidates:
        if note_ms in used_notes or onset_ms in used_onsets:
            continue
        used_notes.add(note_ms)
        used_onsets.add(onset_ms)
        pairs.append((note_ms, onset_ms))

    ordered_pairs = tuple(sorted(pairs))
    matched = len(ordered_pairs)
    mean_error = (
        sum(abs(note_ms - onset_ms) for note_ms, onset_ms in ordered_pairs) / matched
        if matched
        else 0.0
    )
    return AlignmentReport(
        matched_pairs=ordered_pairs,
        auto_play_onsets=tuple(onset for onset in onsets if onset not in used_onsets),
        drum_coverage=round(matched / len(onsets), 4) if onsets else 0.0,
        drum_precision=round(matched / len(note_times), 4) if note_times else 0.0,
        mean_abs_err_ms=round(mean_error, 3),
    )
