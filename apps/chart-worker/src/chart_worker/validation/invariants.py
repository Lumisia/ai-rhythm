from collections import Counter, defaultdict

from chart_worker.schema.note import Chart


def check_timing_invariant(before: Chart, after: Chart) -> bool:
    return Counter(note.time_ms for note in after) <= Counter(note.time_ms for note in before)


def check_hold_only_shrinks(before: Chart, after: Chart) -> bool:
    before_durations: dict[tuple[int, int], list[int]] = defaultdict(list)
    for note in before:
        if note.kind == "HOLD" and note.duration_ms is not None:
            key = (note.time_ms, note.origin_lane)
            before_durations[key].append(note.duration_ms)

    after_durations: dict[tuple[int, int], list[int]] = defaultdict(list)
    for note in after:
        if note.kind != "HOLD" or note.duration_ms is None:
            continue
        key = (note.time_ms, note.origin_lane)
        after_durations[key].append(note.duration_ms)

    for key, durations in after_durations.items():
        originals = before_durations.get(key, [])
        if len(durations) > len(originals):
            return False
        matched = zip(sorted(durations, reverse=True), sorted(originals, reverse=True))
        if any(duration > original for duration, original in matched):
            return False
    return True
