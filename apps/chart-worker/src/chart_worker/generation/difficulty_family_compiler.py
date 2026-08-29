"""Bounded, report-only compilation of nested mania difficulty families."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Literal, TypeAlias

from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.song_context import LocalTempoMap
from chart_worker.generation.candidate_payload_store import persist_candidate_payload
from chart_worker.generation.candidate_state import Candidate
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.schema.note import NoteEvent
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.stages.types import SongTimingAuthority
from chart_worker.validation.difficulty_order import MIN_ADJACENT_RATING_GAP
from chart_worker.validation.quality_gate import (
    ChartAcceptance,
    GateAction,
    GateAxis,
)

DIFFICULTY_FAMILY_COMPILER_VERSION = "difficulty-family-compiler-shadow-v2-runtime"
BEATS_PER_BUCKET = 8.0
MAX_PROPOSALS_PER_TIER = 15
MIN_TIMING_PRECISION_DELTA = 0.03
_HOLD_BIT = 128
_CIRCLE_BIT = 1
_AUXILIARY_TYPE_BITS = 4 | 16 | 32 | 64
RETENTION_RATIOS: tuple[float, ...] = tuple(round(value / 100.0, 2) for value in range(95, 24, -5))

CompilerStatus: TypeAlias = Literal["NOT_NEEDED", "COMPILED", "UNAVAILABLE"]


@dataclass(frozen=True, slots=True)
class DifficultyFamilyCompilerSlot:
    difficulty: str
    candidate_id: str
    candidate_payload_sha256: str
    generated: GeneratedChart
    acceptance: ChartAcceptance
    osu_text: str
    provenance: str


@dataclass(frozen=True, slots=True)
class DifficultyFamilyCompilerProposal:
    target_difficulty: str
    source_difficulty: str
    source_candidate_id: str
    generated: GeneratedChart
    acceptance: ChartAcceptance
    osu_text: str
    row_retention: float
    note_retention: float
    candidate_payload_ref: str | None = None
    candidate_payload_sha256: str | None = None

    def to_report(self) -> dict[str, object]:
        rows = _group_rows(self.generated.notes)
        rating = _rating(self.acceptance)
        ordering = _ordering_score(self.acceptance)
        return {
            "targetDifficulty": self.target_difficulty,
            "sourceDifficulty": self.source_difficulty,
            "sourceCandidateId": self.source_candidate_id,
            "candidatePayloadRef": self.candidate_payload_ref,
            "candidatePayloadSha256": self.candidate_payload_sha256,
            "rowCount": len(rows),
            "noteCount": len(self.generated.notes),
            "rowRetention": self.row_retention,
            "noteRetention": self.note_retention,
            "firstRowMs": next(iter(rows), None),
            "lastRowMs": next(reversed(rows), None) if rows else None,
            "maxHoldEndMs": max(
                (
                    note.time_ms + (note.duration_ms or 0)
                    for note in self.generated.notes
                    if note.kind == "HOLD"
                ),
                default=None,
            ),
            "projectRating": rating,
            "orderingScore": ordering,
            "timingPrecision50": _precision(self.acceptance),
            "timingF150": self.acceptance.timing.overall.matched_f1_50,
            "acceptanceAction": self.acceptance.action.value,
            "coverageGapCount": len(self.acceptance.timing.coverage_gaps),
        }


@dataclass(frozen=True, slots=True)
class DifficultyFamilyCompilerDecision:
    key_mode: int
    status: CompilerStatus
    reason: str
    anchor_candidate_id: str | None
    anchor_source_difficulty: str | None
    proposals: tuple[DifficultyFamilyCompilerProposal, ...]
    proposals_evaluated: int
    failure_type: str | None = None
    failure_message: str | None = None
    solver_wall_ms: float = 0.0
    candidate_evaluation_wall_ms: float = 0.0
    payload_persistence_wall_ms: float = 0.0
    publication_mode: Literal["SHADOW", "ENFORCED"] = "SHADOW"

    def to_report(self) -> dict[str, object]:
        return {
            "version": DIFFICULTY_FAMILY_COMPILER_VERSION,
            "keyMode": self.key_mode,
            "status": self.status,
            "reason": self.reason,
            "anchorCandidateId": self.anchor_candidate_id,
            "anchorSourceDifficulty": self.anchor_source_difficulty,
            "proposalsEvaluated": self.proposals_evaluated,
            "proposals": [proposal.to_report() for proposal in self.proposals],
            "failureType": self.failure_type,
            "failureMessage": self.failure_message,
            "solverWallMs": self.solver_wall_ms,
            "candidateEvaluationWallMs": self.candidate_evaluation_wall_ms,
            "payloadPersistenceWallMs": self.payload_persistence_wall_ms,
            "publicationMode": self.publication_mode,
            # ENFORCED authorizes these proposals to enter the final selector's
            # candidate pool.  The compiler itself never owns publication.
            "eligibleForFinalSelection": self.publication_mode == "ENFORCED",
            "mutatesSelection": False,
            "mutatesPublishedCharts": False,
            "additionalModelCalls": 0,
        }


EvaluateCandidate: TypeAlias = Callable[[GeneratedChart, str], ChartAcceptance]
SerializeCandidate: TypeAlias = Callable[[GeneratedChart], str]
Clock: TypeAlias = Callable[[], float]


def materialize_compiled_family(
    decision: DifficultyFamilyCompilerDecision,
    *,
    source_candidates: dict[str, Candidate],
    current_assignment: dict[str, Candidate | None],
) -> dict[str, Candidate]:
    """Convert one verified all-or-nothing compiler result into publishable candidates.

    This function does not mutate repositories or the supplied assignment.  A
    caller can therefore validate the complete four-slot result before
    installing it atomically.
    """

    if decision.status != "COMPILED":
        raise ValueError("only a COMPILED decision can be materialized")
    by_difficulty = {
        proposal.target_difficulty: proposal for proposal in decision.proposals
    }
    if len(decision.proposals) != len(DIFFICULTIES) or set(by_difficulty) != set(
        DIFFICULTIES
    ):
        raise ValueError("compiled family must contain four target difficulties")
    payloads = tuple(
        hashlib.sha256(by_difficulty[difficulty].osu_text.encode("utf-8")).hexdigest()
        for difficulty in DIFFICULTIES
    )
    if len(set(payloads)) != len(payloads):
        raise ValueError("compiled family must contain four unique payloads")

    materialized: dict[str, Candidate] = {}
    for difficulty in DIFFICULTIES:
        proposal = by_difficulty[difficulty]
        source = source_candidates.get(proposal.source_candidate_id)
        target_current = current_assignment.get(difficulty)
        if source is None:
            raise ValueError(
                f"compiled family references unknown source candidate: {proposal.source_candidate_id}"
            )
        if target_current is None:
            raise ValueError("compiled family requires a complete current assignment")
        materialized[difficulty] = Candidate(
            request=replace(
                target_current.request,
                difficulty=difficulty,
                seed=source.seed,
                partial_start_ms=None,
                partial_end_ms=None,
                add_to_beatmap=False,
                required_gameplay_interval=None,
            ),
            generated=proposal.generated,
            acceptance=proposal.acceptance,
            osu_text=proposal.osu_text,
            workdir=source.workdir,
            attempt=source.attempt,
            seed=source.seed,
            provenance="SAFE_FALLBACK",
            recovery_reason="DIFFICULTY_FAMILY_COMPILER_V1",
            intro_anchor_covered=source.intro_anchor_covered,
            coverage_repair_gap_count=len(proposal.acceptance.timing.coverage_gaps),
        )
    return materialized


def _elapsed_ms(start: float, end: float) -> float:
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError("compiler clock must be finite and monotonic")
    return round((end - start) * 1_000.0, 6)


def _rating(acceptance: ChartAcceptance) -> float | None:
    profile = acceptance.profile
    return float(profile.difficulty.project_rating) if profile is not None else None


def _ordering_score(acceptance: ChartAcceptance) -> float | None:
    profile = acceptance.profile
    return float(profile.difficulty_vector_v2.ordering_score) if profile is not None else None


def _action_rank(action: GateAction) -> int:
    return {
        GateAction.PASS: 0,
        GateAction.REVIEW: 1,
        GateAction.RETRY_MAP: 2,
    }[action]


def _hard_safe(acceptance: ChartAcceptance) -> bool:
    return all(
        acceptance.decision(axis).action is GateAction.PASS
        for axis in (
            GateAxis.STRUCTURE,
            GateAxis.TIMING_IDENTITY,
            GateAxis.SONG_BOUNDS,
        )
    ) and all(
        acceptance.decision(axis).action is not GateAction.RETRY_MAP
        for axis in (
            GateAxis.TIMING_ALIGNMENT,
            GateAxis.COVERAGE,
        )
    )


def _gap_summary(acceptance: ChartAcceptance) -> tuple[int, int]:
    gaps = acceptance.timing.coverage_gaps
    return (
        len(gaps),
        max((int(gap.end_ms) - int(gap.start_ms) for gap in gaps), default=0),
    )


def _precision(acceptance: ChartAcceptance) -> float | None:
    value = acceptance.timing.overall.matched_precision_50
    return float(value) if value is not None else None


def _family_needs_compilation(
    ordered: tuple[DifficultyFamilyCompilerSlot, ...],
) -> bool:
    if len({slot.candidate_payload_sha256 for slot in ordered}) != len(ordered):
        return True
    for easier, harder in pairwise(ordered):
        easier_rating = _rating(easier.acceptance)
        harder_rating = _rating(harder.acceptance)
        easier_ordering = _ordering_score(easier.acceptance)
        harder_ordering = _ordering_score(harder.acceptance)
        if None in {
            easier_rating,
            harder_rating,
            easier_ordering,
            harder_ordering,
        }:
            return True
        assert easier_rating is not None and harder_rating is not None
        assert easier_ordering is not None and harder_ordering is not None
        if harder_rating <= easier_rating or harder_ordering <= easier_ordering:
            return True
    return False


def _family_has_narrow_positive_gap(
    ordered: tuple[DifficultyFamilyCompilerSlot, ...],
) -> bool:
    for easier, harder in pairwise(ordered):
        easier_rating = _rating(easier.acceptance)
        harder_rating = _rating(harder.acceptance)
        if (
            easier_rating is not None
            and harder_rating is not None
            and 0.0 < harder_rating - easier_rating < MIN_ADJACENT_RATING_GAP
        ):
            return True
    return False


def _group_rows(notes: list[NoteEvent]) -> dict[int, tuple[NoteEvent, ...]]:
    rows: dict[int, list[NoteEvent]] = defaultdict(list)
    for note in notes:
        rows[note.time_ms].append(note)
    return {
        time_ms: tuple(
            sorted(
                row,
                key=lambda note: (
                    note.lane,
                    note.kind,
                    note.duration_ms or 0,
                ),
            )
        )
        for time_ms, row in sorted(rows.items())
    }


def _note_identity(note: NoteEvent) -> tuple[int, int, str, int | None]:
    return (note.time_ms, note.lane, note.kind, note.duration_ms)


def _hit_object_identity(
    line: str,
    *,
    key_mode: int,
) -> tuple[int, int, str, int | None]:
    parts = line.split(",")
    if len(parts) < 5:
        raise ValueError(f"malformed anchor HitObject: {line!r}")
    try:
        x = int(float(parts[0]))
        time_ms = round(float(parts[2]))
        object_type = int(parts[3])
    except ValueError as error:
        raise ValueError(f"malformed anchor HitObject: {line!r}") from error
    if not 0 <= x <= 512:
        raise ValueError(f"anchor mania x coordinate is outside 0..512: {x}")
    lane = min(key_mode - 1, max(0, math.floor(x * key_mode / 512)))
    base_type = object_type & ~_AUXILIARY_TYPE_BITS
    if base_type == _HOLD_BIT:
        if len(parts) < 6 or not parts[5].split(":", 1)[0]:
            raise ValueError(f"anchor HOLD is missing an end time: {line!r}")
        try:
            end_ms = round(float(parts[5].split(":", 1)[0]))
        except ValueError as error:
            raise ValueError(f"malformed anchor HOLD: {line!r}") from error
        duration_ms = end_ms - time_ms
        if duration_ms <= 0:
            raise ValueError(f"anchor HOLD has a non-positive duration: {line!r}")
        return (time_ms, lane, "HOLD", duration_ms)
    if base_type == _CIRCLE_BIT:
        return (time_ms, lane, "TAP", None)
    raise ValueError(f"unsupported anchor HitObject type: {object_type}")


def _delete_anchor_rows_exactly(
    anchor_osu_text: str,
    *,
    key_mode: int,
    kept_notes: list[NoteEvent],
) -> str:
    """Return anchor bytes with only unselected HitObjects removed.

    Surviving object coordinates, hitsounds, samples, timing points, SV points,
    metadata, and formatting stay byte-for-byte identical within the decoded
    text.  Duplicate semantic notes are accounted for with a multiset.
    """

    remaining = Counter(_note_identity(note) for note in kept_notes)
    output: list[str] = []
    in_hit_objects = False
    saw_hit_objects = False
    for raw_line in anchor_osu_text.splitlines(keepends=True):
        stripped = raw_line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_hit_objects = stripped == "[HitObjects]"
            saw_hit_objects = saw_hit_objects or in_hit_objects
            output.append(raw_line)
            continue
        if not in_hit_objects or not stripped or stripped.startswith("//"):
            output.append(raw_line)
            continue
        identity = _hit_object_identity(stripped, key_mode=key_mode)
        if remaining[identity] <= 0:
            continue
        output.append(raw_line)
        remaining[identity] -= 1
    if not saw_hit_objects:
        raise ValueError("anchor osu text has no [HitObjects] section")
    missing = +remaining
    if missing:
        raise ValueError(
            "compiled notes are not an exact semantic subset of anchor HitObjects: "
            f"{sum(missing.values())} missing"
        )
    return "".join(output)


def _row_priority(
    rows: dict[int, tuple[NoteEvent, ...]],
    *,
    tempo_map: LocalTempoMap,
    onset_analysis: OnsetAnalysis,
) -> tuple[int, ...]:
    first_ms = next(iter(rows))
    buckets: dict[int, list[int]] = defaultdict(list)
    for time_ms in rows:
        beats = tempo_map.beats_between(first_ms, time_ms)
        buckets[math.floor(beats / BEATS_PER_BUCKET)].append(time_ms)
    for times in buckets.values():
        times.sort(
            key=lambda time_ms: (
                -onset_analysis.strength_at(time_ms),
                -int(any(note.is_downbeat for note in rows[time_ms])),
                -int(any(note.kind == "HOLD" for note in rows[time_ms])),
                time_ms,
            )
        )
    priority: list[int] = []
    depth = 0
    while True:
        added = False
        for bucket in sorted(buckets):
            times = buckets[bucket]
            if depth < len(times):
                priority.append(times[depth])
                added = True
        if not added:
            return tuple(priority)
        depth += 1


def _simplify(
    source: GeneratedChart,
    *,
    retention: float,
    tempo_map: LocalTempoMap,
    onset_analysis: OnsetAnalysis,
) -> GeneratedChart:
    rows = _group_rows(source.notes)
    if len(rows) < 3:
        return source
    first_ms = next(iter(rows))
    last_ms = next(reversed(rows))
    keep_count = max(2, min(len(rows) - 1, round(len(rows) * retention)))
    priority = _row_priority(
        rows,
        tempo_map=tempo_map,
        onset_analysis=onset_analysis,
    )
    keep_times = {first_ms, last_ms}
    for time_ms in priority:
        if len(keep_times) >= keep_count:
            break
        keep_times.add(time_ms)

    lane_usage = [0] * source.key_mode
    kept: list[NoteEvent] = []
    for time_ms in sorted(keep_times):
        row = rows[time_ms]
        if time_ms in {first_ms, last_ms}:
            chosen = row
        else:
            note_count = max(1, min(len(row), round(len(row) * retention)))
            chosen = tuple(
                sorted(
                    sorted(
                        row,
                        key=lambda note: (
                            lane_usage[note.lane],
                            -int(note.kind == "HOLD"),
                            note.lane,
                        ),
                    )[:note_count],
                    key=lambda note: note.lane,
                )
            )
        kept.extend(chosen)
        for note in chosen:
            lane_usage[note.lane] += 1
    kept.sort(key=lambda note: (note.time_ms, note.lane))
    return replace(source, notes=kept, osu_text="")


def _candidate_is_non_regressing(
    candidate: ChartAcceptance,
    *,
    parent: ChartAcceptance,
    current_target: ChartAcceptance,
) -> bool:
    if not _hard_safe(candidate) or candidate.action is GateAction.RETRY_MAP:
        return False
    if _action_rank(candidate.action) > min(
        _action_rank(parent.action),
        _action_rank(current_target.action),
    ):
        return False
    candidate_gaps = _gap_summary(candidate)
    if candidate_gaps > _gap_summary(parent) or candidate_gaps > _gap_summary(current_target):
        return False
    candidate_precision = _precision(candidate)
    parent_precision = _precision(parent)
    return not (
        candidate_precision is not None
        and parent_precision is not None
        and candidate_precision < parent_precision - MIN_TIMING_PRECISION_DELTA
    )


def _proposal(
    *,
    target: str,
    anchor: DifficultyFamilyCompilerSlot,
    generated: GeneratedChart,
    acceptance: ChartAcceptance,
    osu_text: str,
) -> DifficultyFamilyCompilerProposal:
    anchor_rows = len(_group_rows(anchor.generated.notes))
    rows = len(_group_rows(generated.notes))
    return DifficultyFamilyCompilerProposal(
        target_difficulty=target,
        source_difficulty=anchor.difficulty,
        source_candidate_id=anchor.candidate_id,
        generated=generated,
        acceptance=acceptance,
        osu_text=osu_text,
        row_retention=round(rows / anchor_rows, 6),
        note_retention=round(len(generated.notes) / len(anchor.generated.notes), 6),
        candidate_payload_sha256=hashlib.sha256(osu_text.encode("utf-8")).hexdigest(),
    )


def _unavailable(
    key_mode: int,
    reason: str,
    *,
    anchor: DifficultyFamilyCompilerSlot | None = None,
    evaluated: int = 0,
) -> DifficultyFamilyCompilerDecision:
    return DifficultyFamilyCompilerDecision(
        key_mode=key_mode,
        status="UNAVAILABLE",
        reason=reason,
        anchor_candidate_id=(anchor.candidate_id if anchor is not None else None),
        anchor_source_difficulty=(anchor.difficulty if anchor is not None else None),
        proposals=(),
        proposals_evaluated=evaluated,
    )


def persist_difficulty_family_compiler_payloads(
    decision: DifficultyFamilyCompilerDecision,
    *,
    run_dir: Path,
    clock: Clock = perf_counter,
) -> DifficultyFamilyCompilerDecision:
    """Persist only SHADOW proposal bytes in the immutable candidate store."""

    if decision.status != "COMPILED":
        return decision
    started = clock()
    try:
        proposals = []
        for proposal in decision.proposals:
            artifact = persist_candidate_payload(
                run_dir=run_dir,
                osu_text=proposal.osu_text,
            )
            proposals.append(
                replace(
                    proposal,
                    candidate_payload_ref=artifact.relative_path.as_posix(),
                    candidate_payload_sha256=artifact.sha256,
                )
            )
    except Exception as error:  # noqa: BLE001 - SHADOW persistence is isolated
        elapsed_ms = _elapsed_ms(started, clock())
        return replace(
            decision,
            status="UNAVAILABLE",
            reason="PAYLOAD_PERSISTENCE_FAILED",
            proposals=(),
            failure_type=type(error).__name__,
            failure_message=str(error),
            payload_persistence_wall_ms=elapsed_ms,
        )
    return replace(
        decision,
        proposals=tuple(proposals),
        payload_persistence_wall_ms=_elapsed_ms(started, clock()),
    )


def compile_difficulty_family_shadow(
    slots: tuple[DifficultyFamilyCompilerSlot, ...],
    *,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    duration_ms: int,
    boundary_policy_mode: str,
    evaluate_candidate: EvaluateCandidate,
    serialize_candidate: SerializeCandidate,
    clock: Clock = perf_counter,
) -> DifficultyFamilyCompilerDecision:
    """Compile a nested SHADOW family or abstain without mutating inputs."""
    del serialize_candidate  # proposals preserve exact anchor rows instead
    solver_started = clock()
    candidate_evaluation_wall_ms = 0.0

    def finish(
        decision: DifficultyFamilyCompilerDecision,
    ) -> DifficultyFamilyCompilerDecision:
        return replace(
            decision,
            solver_wall_ms=_elapsed_ms(solver_started, clock()),
            candidate_evaluation_wall_ms=round(
                candidate_evaluation_wall_ms,
                6,
            ),
        )

    if type(slots) is not tuple or len(slots) != len(DIFFICULTIES):
        raise ValueError("compiler requires exactly four difficulty slots")
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if not boundary_policy_mode:
        raise ValueError("boundary_policy_mode must be non-empty")
    key_modes = {slot.generated.key_mode for slot in slots}
    if len(key_modes) != 1 or next(iter(key_modes)) not in KEY_MODES:
        raise ValueError("compiler slots must use one supported key mode")
    key_mode = next(iter(key_modes))
    by_difficulty = {slot.difficulty: slot for slot in slots}
    if set(by_difficulty) != set(DIFFICULTIES):
        raise ValueError("compiler slots must contain every difficulty exactly once")
    ordered = tuple(by_difficulty[difficulty] for difficulty in DIFFICULTIES)
    if not _family_needs_compilation(ordered):
        return finish(
            DifficultyFamilyCompilerDecision(
                key_mode=key_mode,
                status="NOT_NEEDED",
                reason=(
                    "FAMILY_ORDERED_NARROW_REVIEW"
                    if _family_has_narrow_positive_gap(ordered)
                    else "FAMILY_ALREADY_SEPARATED"
                ),
                anchor_candidate_id=None,
                anchor_source_difficulty=None,
                proposals=(),
                proposals_evaluated=0,
            )
        )

    safe_anchors = tuple(
        slot
        for slot in ordered
        if _hard_safe(slot.acceptance)
        and _rating(slot.acceptance) is not None
        and _ordering_score(slot.acceptance) is not None
    )
    if not safe_anchors:
        return finish(_unavailable(key_mode, "NO_SAFE_ANCHOR"))
    expert_target = by_difficulty["EXPERT"]
    safe_anchors = tuple(
        slot
        for slot in safe_anchors
        if _candidate_is_non_regressing(
            slot.acceptance,
            parent=expert_target.acceptance,
            current_target=expert_target.acceptance,
        )
    )
    if not safe_anchors:
        return finish(_unavailable(key_mode, "NO_NON_REGRESSING_EXPERT_ANCHOR"))
    anchor = max(
        safe_anchors,
        key=lambda slot: (
            _rating(slot.acceptance),
            _ordering_score(slot.acceptance),
            DIFFICULTIES.index(slot.difficulty),
            slot.candidate_id,
        ),
    )
    tempo_map = LocalTempoMap(authority.bpm_events)
    expert = _proposal(
        target="EXPERT",
        anchor=anchor,
        generated=anchor.generated,
        acceptance=anchor.acceptance,
        osu_text=anchor.osu_text,
    )
    selected: dict[str, DifficultyFamilyCompilerProposal] = {"EXPERT": expert}
    parent = expert
    evaluated = 0
    for target in reversed(DIFFICULTIES[:-1]):
        current_target = by_difficulty[target]
        parent_rating = _rating(parent.acceptance)
        parent_ordering = _ordering_score(parent.acceptance)
        assert parent_rating is not None and parent_ordering is not None
        feasible: list[DifficultyFamilyCompilerProposal] = []
        seen_payloads: set[str] = set()
        for retention in RETENTION_RATIOS[:MAX_PROPOSALS_PER_TIER]:
            generated = _simplify(
                parent.generated,
                retention=retention,
                tempo_map=tempo_map,
                onset_analysis=onset_analysis,
            )
            osu_text = _delete_anchor_rows_exactly(
                anchor.osu_text,
                key_mode=key_mode,
                kept_notes=generated.notes,
            )
            if osu_text in seen_payloads or osu_text == parent.osu_text:
                continue
            seen_payloads.add(osu_text)
            generated = replace(generated, osu_text=osu_text)
            evaluation_started = clock()
            acceptance = evaluate_candidate(generated, target)
            candidate_evaluation_wall_ms += _elapsed_ms(
                evaluation_started,
                clock(),
            )
            evaluated += 1
            rating = _rating(acceptance)
            ordering = _ordering_score(acceptance)
            if rating is None or ordering is None:
                continue
            if parent_rating - rating < MIN_ADJACENT_RATING_GAP or ordering >= parent_ordering:
                continue
            if not _candidate_is_non_regressing(
                acceptance,
                parent=parent.acceptance,
                current_target=current_target.acceptance,
            ):
                continue
            feasible.append(
                _proposal(
                    target=target,
                    anchor=anchor,
                    generated=generated,
                    acceptance=acceptance,
                    osu_text=osu_text,
                )
            )
        if not feasible:
            return finish(
                _unavailable(
                    key_mode,
                    f"NO_SAFE_{target}_PROPOSAL",
                    anchor=anchor,
                    evaluated=evaluated,
                )
            )
        chosen = max(
            feasible,
            key=lambda proposal: (
                _rating(proposal.acceptance),
                _ordering_score(proposal.acceptance),
                proposal.note_retention,
                proposal.osu_text,
            ),
        )
        selected[target] = chosen
        parent = chosen

    return finish(
        DifficultyFamilyCompilerDecision(
            key_mode=key_mode,
            status="COMPILED",
            reason="NESTED_FAMILY_COMPILED",
            anchor_candidate_id=anchor.candidate_id,
            anchor_source_difficulty=anchor.difficulty,
            proposals=tuple(selected[difficulty] for difficulty in DIFFICULTIES),
            proposals_evaluated=evaluated,
        )
    )
