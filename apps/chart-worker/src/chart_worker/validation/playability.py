"""최종 플레이 가능성 검사와 제한된 복구."""

import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.postprocess.lane_conversion import convert_lanes
from chart_worker.postprocess.lane_rules import check_lane_rules, hand_of
from chart_worker.postprocess.pattern_policy import Allow, allowance_of, quota_excesses
from chart_worker.postprocess.patterns import PatternInstance, detect_patterns, rows_of
from chart_worker.schema.note import Chart, NoteEvent
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES, lane_semantics
from chart_worker.validation.invariants import check_hold_only_shrinks, check_timing_invariant

MAX_CHORD = {"EASY": 2, "NORMAL": 3, "HARD": 3, "EXPERT": 4}
MAX_JACK_RUN = {"EASY": 1, "NORMAL": 2, "HARD": 3, "EXPERT": 4}
MAX_HAND_SHARE = {"EASY": 0.75, "NORMAL": 0.80, "HARD": 0.85, "EXPERT": 0.90}
HAND_WINDOW_MS = 2_000
MIN_NOTES_FOR_HAND_SHARE = 6
MIN_HOLD_MS = 60
MAX_RECOVERY_PASSES = 3


class ViolationCode(StrEnum):
    LANE_BOUNDS = "LANE_BOUNDS"
    TIME_BOUNDS = "TIME_BOUNDS"
    DUPLICATE = "DUPLICATE"
    HOLD_BOUNDS = "HOLD_BOUNDS"
    HOLD_OVERLAP = "HOLD_OVERLAP"
    CHORD_LIMIT = "CHORD_LIMIT"
    HAND_IMBALANCE = "HAND_IMBALANCE"
    JACK_RUN = "JACK_RUN"
    LANE_RULE = "LANE_RULE"
    PATTERN_FORBIDDEN = "PATTERN_FORBIDDEN"
    PATTERN_QUOTA = "PATTERN_QUOTA"


@dataclass(frozen=True, slots=True)
class PlayabilityViolation:
    code: ViolationCode
    time_ms: int
    end_ms: int
    lanes: tuple[int, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class PlayabilityResult:
    notes: Chart
    recovered_count: int
    deleted_count: int
    passes: int
    violations: tuple[PlayabilityViolation, ...]


def _require_inputs(key_mode: int, difficulty: str, duration_ms: int, beat_ms: float) -> None:
    if key_mode not in KEY_MODES:
        raise ValueError(f"unsupported key_mode: {key_mode}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unsupported difficulty: {difficulty}")
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if beat_ms <= 0:
        raise ValueError("beat_ms must be positive")


def _pattern_violation(
    code: ViolationCode, instance: PatternInstance, detail: str
) -> PlayabilityViolation:
    return PlayabilityViolation(
        code=code,
        time_ms=instance.start_ms,
        end_ms=instance.end_ms,
        lanes=instance.lanes,
        detail=detail,
    )


def _jack_violations(notes: Chart, difficulty: str) -> list[PlayabilityViolation]:
    limit = MAX_JACK_RUN[difficulty]
    active: dict[int, list[int]] = {}
    found = []
    for row in rows_of(notes):
        lanes = set(row.lanes)
        for lane in list(active):
            if lane not in lanes:
                active[lane] = []
        for lane in lanes:
            run = active.setdefault(lane, [])
            run.append(row.time_ms)
            if len(run) == limit + 1:
                found.append(
                    PlayabilityViolation(
                        ViolationCode.JACK_RUN,
                        run[0],
                        run[-1],
                        (lane,),
                        f"jack run exceeds {limit}",
                    )
                )
    return found


def _hand_violations(notes: Chart, key_mode: int, difficulty: str) -> list[PlayabilityViolation]:
    semantics = lane_semantics(key_mode)
    buckets: dict[int, list[NoteEvent]] = defaultdict(list)
    for note in notes:
        buckets[note.time_ms // HAND_WINDOW_MS].append(note)

    found = []
    for bucket, window in buckets.items():
        counts = defaultdict(int)
        for note in window:
            hand = hand_of(semantics[note.lane])
            if hand is not None:
                counts[hand] += 1
        total = sum(counts.values())
        if total < MIN_NOTES_FOR_HAND_SHARE:
            continue
        share = max(counts.values(), default=0) / total
        if share > MAX_HAND_SHARE[difficulty]:
            found.append(
                PlayabilityViolation(
                    ViolationCode.HAND_IMBALANCE,
                    bucket * HAND_WINDOW_MS,
                    (bucket + 1) * HAND_WINDOW_MS - 1,
                    tuple(sorted({note.lane for note in window})),
                    f"one-hand share {share:.3f} exceeds {MAX_HAND_SHARE[difficulty]:.3f}",
                )
            )
    return found


def find_violations(
    notes: Chart,
    *,
    key_mode: int,
    difficulty: str,
    duration_ms: int,
    beat_ms: float,
) -> tuple[PlayabilityViolation, ...]:
    """입력을 수정하지 않고 모든 최종 위반을 찾는다."""
    _require_inputs(key_mode, difficulty, duration_ms, beat_ms)
    found: list[PlayabilityViolation] = []
    seen: set[tuple[int, int]] = set()
    hold_end_by_lane: dict[int, int] = {}
    structurally_valid: Chart = []

    for note in sorted(notes, key=lambda item: (item.time_ms, item.lane)):
        if note.lane >= key_mode:
            found.append(
                PlayabilityViolation(
                    ViolationCode.LANE_BOUNDS,
                    note.time_ms,
                    note.time_ms,
                    (note.lane,),
                    f"lane {note.lane} is outside {key_mode}K",
                )
            )
            continue
        if note.time_ms >= duration_ms:
            found.append(
                PlayabilityViolation(
                    ViolationCode.TIME_BOUNDS,
                    note.time_ms,
                    note.time_ms,
                    (note.lane,),
                    "note starts outside the song",
                )
            )
            continue
        key = (note.time_ms, note.lane)
        if key in seen:
            found.append(
                PlayabilityViolation(
                    ViolationCode.DUPLICATE,
                    note.time_ms,
                    note.time_ms,
                    (note.lane,),
                    "duplicate note at the same time and lane",
                )
            )
            continue
        seen.add(key)
        if note.kind == "HOLD" and note.time_ms + (note.duration_ms or 0) > duration_ms:
            found.append(
                PlayabilityViolation(
                    ViolationCode.HOLD_BOUNDS,
                    note.time_ms,
                    note.time_ms + (note.duration_ms or 0),
                    (note.lane,),
                    "HOLD exceeds the song duration",
                )
            )
        if note.time_ms < hold_end_by_lane.get(note.lane, -1):
            found.append(
                PlayabilityViolation(
                    ViolationCode.HOLD_OVERLAP,
                    note.time_ms,
                    hold_end_by_lane[note.lane],
                    (note.lane,),
                    "note overlaps an active HOLD",
                )
            )
        if note.kind == "HOLD":
            hold_end_by_lane[note.lane] = note.time_ms + (note.duration_ms or 0)
        structurally_valid.append(note)

    for row in rows_of(structurally_valid):
        if row.size > MAX_CHORD[difficulty]:
            found.append(
                PlayabilityViolation(
                    ViolationCode.CHORD_LIMIT,
                    row.time_ms,
                    row.time_ms,
                    row.lanes,
                    f"chord size {row.size} exceeds {MAX_CHORD[difficulty]}",
                )
            )

    found.extend(_hand_violations(structurally_valid, key_mode, difficulty))
    found.extend(_jack_violations(structurally_valid, difficulty))
    found.extend(
        PlayabilityViolation(
            ViolationCode.LANE_RULE,
            violation.time_ms,
            violation.time_ms,
            violation.lanes,
            f"{violation.rule}: {violation.detail}",
        )
        for violation in check_lane_rules(
            structurally_valid, key_mode=key_mode, difficulty=difficulty
        )
    )

    patterns = detect_patterns(structurally_valid, key_mode=key_mode, beat_ms=beat_ms)
    found.extend(
        _pattern_violation(
            ViolationCode.PATTERN_FORBIDDEN,
            instance,
            f"{instance.kind} is forbidden at {difficulty}",
        )
        for instance in patterns
        if allowance_of(instance.kind, difficulty) is Allow.FORBIDDEN
    )
    found.extend(
        _pattern_violation(
            ViolationCode.PATTERN_QUOTA,
            instance,
            f"{instance.kind} exceeds its 8-bar quota",
        )
        for instance in quota_excesses(patterns, difficulty=difficulty, beat_ms=beat_ms)
    )
    return tuple(sorted(found, key=lambda item: (item.time_ms, item.code, item.lanes)))


def _prepare(notes: Chart, *, key_mode: int, duration_ms: int) -> tuple[Chart, int, int]:
    prepared: list[NoteEvent] = []
    deleted = recovered = 0
    for note in notes:
        if note.time_ms >= duration_ms:
            deleted += 1
            continue
        updated = note
        if note.lane >= key_mode:
            updated = dataclasses.replace(note, lane=key_mode - 1)
            recovered += 1
        if updated.kind == "HOLD" and updated.time_ms + (updated.duration_ms or 0) > duration_ms:
            duration = duration_ms - updated.time_ms
            updated = (
                dataclasses.replace(updated, kind="TAP", duration_ms=None)
                if duration < MIN_HOLD_MS
                else dataclasses.replace(updated, duration_ms=duration)
            )
            recovered += 1
        prepared.append(updated)

    strongest: dict[tuple[int, int], NoteEvent] = {}
    for note in prepared:
        key = (note.time_ms, note.lane)
        existing = strongest.get(key)
        if existing is None:
            strongest[key] = note
            continue
        deleted += 1
        if _preservation_key(note) > _preservation_key(existing):
            strongest[key] = note
    return (
        sorted(strongest.values(), key=lambda item: (item.time_ms, item.lane)),
        deleted,
        recovered,
    )


def _preservation_key(note: NoteEvent) -> tuple[float, bool, int]:
    strength = note.onset_strength if note.onset_strength is not None else 0.5
    return strength, note.is_downbeat, -note.time_ms


def _victim(notes: Chart, violation: PlayabilityViolation) -> NoteEvent | None:
    candidates = [
        note
        for note in notes
        if violation.time_ms <= note.time_ms <= violation.end_ms
        and (not violation.lanes or note.lane in violation.lanes)
    ]
    if not candidates:
        return None
    return min(candidates, key=_preservation_key)


def _require_invariants(before: Chart, after: Chart) -> None:
    if not check_timing_invariant(before, after):
        raise WorkerError(
            ErrorCode.CHART_TIMING_INVARIANT_VIOLATED,
            "playability recovery invented note times",
        )
    if not check_hold_only_shrinks(before, after):
        raise WorkerError(
            ErrorCode.CHART_TIMING_INVARIANT_VIOLATED,
            "playability recovery lengthened or invented a HOLD",
        )


def validate_and_recover(
    notes: Chart,
    *,
    key_mode: int,
    difficulty: str,
    duration_ms: int,
    beat_ms: float,
    max_passes: int = MAX_RECOVERY_PASSES,
) -> PlayabilityResult:
    _require_inputs(key_mode, difficulty, duration_ms, beat_ms)
    if max_passes < 0:
        raise ValueError("max_passes must be non-negative")

    before = list(notes)
    current, deleted, recovered = _prepare(before, key_mode=key_mode, duration_ms=duration_ms)
    _require_invariants(before, current)

    for pass_index in range(max_passes):
        converted = convert_lanes(
            current,
            key_mode=key_mode,
            difficulty=difficulty,
            max_passes=1,
        )
        current = converted.notes
        deleted += converted.deleted_count
        recovered += converted.moved_count
        violations = find_violations(
            current,
            key_mode=key_mode,
            difficulty=difficulty,
            duration_ms=duration_ms,
            beat_ms=beat_ms,
        )
        if not violations:
            _require_invariants(before, current)
            return PlayabilityResult(
                current,
                recovered,
                deleted,
                pass_index + 1,
                (),
            )

        victim = _victim(current, violations[0])
        if victim is None:
            break
        current = [note for note in current if note is not victim]
        deleted += 1
        _require_invariants(before, current)

    remaining = find_violations(
        current,
        key_mode=key_mode,
        difficulty=difficulty,
        duration_ms=duration_ms,
        beat_ms=beat_ms,
    )
    if not remaining:
        return PlayabilityResult(current, recovered, deleted, max_passes, ())
    raise WorkerError(
        ErrorCode.CHART_VALIDATION_FAILED,
        f"{len(remaining)} playability violations remain after {max_passes} passes",
        context={
            "violations": [
                {
                    "code": violation.code.value,
                    "time_ms": violation.time_ms,
                    "lanes": violation.lanes,
                }
                for violation in remaining[:20]
            ]
        },
    )
