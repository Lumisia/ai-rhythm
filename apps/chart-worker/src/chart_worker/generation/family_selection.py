"""Pure candidate-family selection and shadow-comparison policy.

Generation and recovery populate candidate repositories.  This module only
projects those candidates into deterministic family choices and diagnostics;
it performs no model calls and writes no artifacts.
"""

import hashlib
import json
from itertools import product
from pathlib import Path

from chart_worker.analysis.activity import (
    OutroObservation,
    SongBoundaryContract,
    evaluate_outro_policy,
    observe_outro,
)
from chart_worker.analysis.intro_anchor import GRID_SUPPORT_WINDOW_MS
from chart_worker.analysis.song_context import SongAnalysisContext
from chart_worker.generation.candidate_payload_store import candidate_payload_artifact
from chart_worker.generation.candidate_state import Candidate, VariantState
from chart_worker.schema.types import DIFFICULTIES
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.difficulty_order import (
    DifficultyOrderReview,
    review_difficulty_order,
)
from chart_worker.validation.difficulty_selector import (
    DifficultyCandidateView,
    DifficultySelectionComparison,
    SelectionMode,
    compare_family_candidates,
)
from chart_worker.validation.family_evidence_v3 import (
    CandidateFamilyEvidenceV3,
    CandidateSafetyEvidenceV3,
    GapIntervalEvidence,
    IntroCandidateVoteV3,
    SongSelectionEvidenceV3,
    build_intro_selection_evidence,
)
from chart_worker.validation.intro_region_contract import review_intro_region_candidate
from chart_worker.validation.intro_start_contract import IntroStartContract
from chart_worker.validation.quality_gate import (
    QUALITY_GATE_VERSION,
    GateAction,
    GateAxis,
)
from chart_worker.validation.safe_family_assignment import (
    SafeFamilyAssignmentDecision,
    SafeFamilyCandidate,
    TerminalEvidenceConfidence,
    select_safe_family_assignment,
)
from chart_worker.validation.song_family_selector import (
    CandidateSnapshot,
    ProtectedMetrics,
    SelectorMode,
    SongSelectionComparison,
    TimingSectionSnapshot,
    compare_song_families,
)

Selection = tuple[
    dict[str, VariantState],
    dict[str, Candidate | None],
    DifficultyOrderReview | None,
]
MAX_CONFIRMED_INTRO_DELAY_BEATS = 8.0


def review_candidates(candidates: tuple[Candidate, ...]) -> DifficultyOrderReview:
    profiles = {}
    for candidate in candidates:
        if candidate.acceptance.profile is None:
            raise ValueError("ladder candidate must carry a chart quality profile")
        profiles[candidate.request.difficulty] = candidate.acceptance.profile.difficulty
    return review_difficulty_order(profiles)


def review_assignment_candidates(
    assignment: dict[str, Candidate | None],
) -> DifficultyOrderReview | None:
    """Review target labels, not the candidates' original request labels."""
    profiles = {}
    for difficulty in DIFFICULTIES:
        candidate = assignment[difficulty]
        if candidate is None:
            continue
        if candidate.acceptance.profile is None:
            raise ValueError("ladder candidate must carry a chart quality profile")
        profiles[difficulty] = candidate.acceptance.profile.difficulty
    return review_difficulty_order(profiles) if profiles else None


def family_score(
    assignment: tuple[Candidate | None, ...],
    review: DifficultyOrderReview | None,
) -> tuple[int, int, int, int, tuple[tuple[int, int], ...]]:
    """Return the deterministic lexicographic score; lower is preferred."""
    chosen = tuple(candidate for candidate in assignment if candidate is not None)
    missing = len(assignment) - len(chosen)
    raw = sum(1 for candidate in chosen if candidate.provenance == "RAW_UNVERIFIED")
    intro_misses = sum(1 for candidate in chosen if candidate.intro_anchor_covered is False)
    crowded = len(review.narrow_pairs) + len(review.ambiguous_pairs) if review is not None else 0
    order = tuple((candidate.attempt, candidate.seed) for candidate in chosen)
    return (missing, raw, intro_misses, crowded, order)


def has_complete_model_family(states: dict[str, VariantState]) -> bool:
    pools = tuple(tuple(states[difficulty].candidates.admitted) for difficulty in DIFFICULTIES)
    if any(not pool for pool in pools):
        return False
    return any(review_candidates(combo).status != "RETRY" for combo in product(*pools))


def select_family(
    states: dict[str, VariantState],
) -> tuple[dict[str, Candidate | None], DifficultyOrderReview | None]:
    options_by_difficulty = []
    for difficulty in DIFFICULTIES:
        state = states[difficulty]
        pool = sorted(
            state.candidates.admitted,
            key=lambda candidate: (candidate.attempt, candidate.seed),
        )
        options_by_difficulty.append((*pool, None))

    best_assignment: dict[str, Candidate | None] | None = None
    best_review: DifficultyOrderReview | None = None
    best_score: tuple | None = None
    for combo in product(*options_by_difficulty):
        chosen = tuple(candidate for candidate in combo if candidate is not None)
        review: DifficultyOrderReview | None = None
        if chosen:
            review = review_candidates(chosen)
            if review.status == "RETRY":
                continue
        score = family_score(combo, review)
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = dict(zip(DIFFICULTIES, combo, strict=True))
            best_review = review
    assert best_assignment is not None
    return best_assignment, best_review


def candidate_stable_id(
    candidate: Candidate,
    *,
    key_mode: int,
    difficulty: str,
    run_dir: Path | None = None,
) -> str:
    if run_dir is not None:
        artifact = candidate_payload_artifact(
            run_dir=run_dir,
            osu_text=candidate.osu_text,
        )
        payload_ref = artifact.relative_path.as_posix()
        payload_sha256 = artifact.sha256
    else:
        payload_ref = candidate.workdir.as_posix()
        payload_sha256 = hashlib.sha256(candidate.osu_text.encode("utf-8")).hexdigest()
    identity = {
        "keyMode": key_mode,
        "difficulty": difficulty,
        "attempt": candidate.attempt,
        "seed": candidate.seed,
        "provenance": candidate.provenance,
        "payloadRef": payload_ref,
        "payloadSha256": payload_sha256,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return (
        f"{key_mode}k:{difficulty}:a{candidate.attempt}:"
        f"s{candidate.seed}:{candidate.provenance}:{digest}"
    )


def compare_difficulty_selection(
    states: dict[str, VariantState],
    assignment: dict[str, Candidate | None],
    *,
    mode: SelectionMode,
) -> tuple[dict[str, Candidate | None], DifficultySelectionComparison]:
    pools: dict[str, tuple[DifficultyCandidateView, ...]] = {}
    current: dict[str, str | None] = {}
    candidate_ids: dict[int, str] = {}
    key_mode = next(iter(states.values())).key_mode
    for difficulty in DIFFICULTIES:
        candidates = tuple(states[difficulty].candidates.admitted)
        views = []
        for candidate in candidates:
            candidate_id = candidate_stable_id(
                candidate,
                key_mode=key_mode,
                difficulty=difficulty,
            )
            candidate_ids[id(candidate)] = candidate_id
            profile = candidate.acceptance.profile
            vector = profile.difficulty_vector_v2 if profile is not None else None
            views.append(
                DifficultyCandidateView(
                    candidate_id=candidate_id,
                    difficulty=difficulty,
                    seed=candidate.seed,
                    attempt=candidate.attempt,
                    provenance=candidate.provenance,
                    intro_anchor_covered=candidate.intro_anchor_covered,
                    current_rating=(
                        profile.difficulty.project_rating if profile is not None else float("inf")
                    ),
                    v2_ordering_score=(vector.ordering_score if vector is not None else None),
                    vector_v2=vector.to_report() if vector is not None else None,
                )
            )
        pools[difficulty] = tuple(views)
        selected = assignment[difficulty]
        current[difficulty] = candidate_ids[id(selected)] if selected is not None else None
    selected_ids, comparison = compare_family_candidates(
        pools,
        current,
        mode=mode,
    )
    if comparison is None:
        raise AssertionError("difficulty comparison is required outside CURRENT mode")
    by_candidate_id = {
        candidate_id: candidate
        for difficulty in DIFFICULTIES
        for candidate in states[difficulty].candidates.admitted
        if (candidate_id := candidate_ids[id(candidate)])
    }
    selected = {
        difficulty: (
            by_candidate_id[candidate_id]
            if (candidate_id := selected_ids[difficulty]) is not None
            else None
        )
        for difficulty in DIFFICULTIES
    }
    return selected, comparison


def song_selection_context_id(
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    boundary: SongBoundaryContract | None,
    intro_contract: IntroStartContract | None = None,
) -> str:
    payload = {
        "version": "song-family-selector-v2-shadow-2-policy-context",
        "audioSha256": authority.audio_sha256,
        "timingAuthoritySha256": authority.sha256,
        "durationMs": prepared.normalized.duration_ms,
        "difficultySelectorMode": prepared.difficulty_selector_mode,
        "qualityGateVersion": QUALITY_GATE_VERSION,
        "songBoundaryContractSha256": (boundary.stable_sha256() if boundary is not None else None),
        "introStartContract": (intro_contract.to_report() if intro_contract is not None else None),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def first_row_ms(candidate: Candidate) -> int | None:
    return min((note.time_ms for note in candidate.generated.notes), default=None)


def _last_gameplay_event_ms(candidate: Candidate) -> int | None:
    return max(
        (
            note.time_ms
            + (
                note.duration_ms
                if note.kind == "HOLD" and note.duration_ms is not None
                else 0
            )
            for note in candidate.generated.notes
        ),
        default=None,
    )


def _intro_distance_ms(
    candidate: Candidate,
    intro_contract: IntroStartContract,
) -> int | None:
    row_ms = first_row_ms(candidate)
    if row_ms is None:
        return None
    region = intro_contract.intro_region
    if region is not None and region.allowed_first_row_ms is not None:
        lower_ms, upper_ms = region.allowed_first_row_ms
        if row_ms < lower_ms:
            return lower_ms - row_ms
        if row_ms > upper_ms:
            return row_ms - upper_ms
        return 0
    canonical = intro_contract.canonical_first_row_ms
    if canonical is None or not intro_contract.audio_supported:
        return None
    return max(0, abs(row_ms - canonical) - GRID_SUPPORT_WINDOW_MS)


def _continuous_tail_risk(
    candidate: Candidate,
    *,
    boundary: SongBoundaryContract | None,
    song_context: SongAnalysisContext | None,
    outro_observation: OutroObservation | None,
) -> tuple[int, int, int, TerminalEvidenceConfidence]:
    last_event_ms = _last_gameplay_event_ms(candidate)
    if last_event_ms is None:
        return (0, 0, 0, "UNKNOWN")
    observation = outro_observation
    if (
        observation is None
        and song_context is not None
        and song_context.onset_analysis.activity is not None
    ):
        observation = observe_outro(
            song_context.onset_analysis.activity,
            song_context.duration_ms,
        )
    last_active_ms = (
        observation.last_active_rms_end_ms if observation is not None else None
    )
    coverage_deficit_ms = (
        max(0, last_active_ms - last_event_ms)
        if last_active_ms is not None
        else 0
    )
    active_onset_count = (
        sum(
            last_event_ms < onset_ms <= last_active_ms
            for onset_ms in song_context.onset_analysis.activity.active_onset_ms
        )
        if (
            song_context is not None
            and song_context.onset_analysis.activity is not None
            and last_active_ms is not None
        )
        else 0
    )
    if (
        boundary is not None
        and boundary.effective_source == "TERMINAL_SILENCE_CONSENSUS"
    ):
        release_end_ms = boundary.release_end_ms
        confidence = "CONFIRMED"
    elif observation is not None:
        release_end_ms = evaluate_outro_policy(observation).release_end_ms
        confidence = "PROVISIONAL"
    else:
        release_end_ms = None
        confidence = "UNKNOWN"
    overflow_ms = (
        max(0, last_event_ms - release_end_ms)
        if release_end_ms is not None
        else 0
    )
    return (
        coverage_deficit_ms,
        active_onset_count,
        overflow_ms,
        confidence,
    )


def _safe_intro_state(
    candidate: Candidate,
    intro_contract: IntroStartContract,
    song_context: SongAnalysisContext | None,
) -> str:
    if intro_contract.intro_region is not None:
        region_review = review_intro_region_candidate(
            intro_contract.intro_region,
            first_row_ms=first_row_ms(candidate),
        )
        if region_review.status == "PASS":
            return "CONFIRMED_SAFE"
        if region_review.status == "DEFECT":
            return "VIOLATION"
        return "UNKNOWN"
    canonical = intro_contract.canonical_first_row_ms
    row_ms = first_row_ms(candidate)
    if canonical is None or not intro_contract.audio_supported:
        return "UNKNOWN"
    if song_context is None:
        return (
            "CONFIRMED_SAFE"
            if row_ms is not None and abs(row_ms - canonical) <= GRID_SUPPORT_WINDOW_MS
            else "VIOLATION"
        )
    if row_ms is None:
        return "VIOLATION"
    anchor = song_context.intro_anchor
    reference_ms = (
        anchor.anchor_grid_ms
        if anchor.status == "CONFIRMED" and anchor.anchor_grid_ms is not None
        else canonical
    )
    audio_supported = any(
        abs(row_ms - onset_ms) <= GRID_SUPPORT_WINDOW_MS
        for onset_ms in song_context.onset_analysis.onset_ms
    ) or (
        anchor.status == "CONFIRMED"
        and anchor.anchor_ms is not None
        and abs(row_ms - anchor.anchor_ms) <= GRID_SUPPORT_WINDOW_MS
    )
    start_ms, end_ms = sorted((reference_ms, row_ms))
    delay_beats = song_context.tempo_map.beats_between(start_ms, end_ms)
    return (
        "CONFIRMED_SAFE"
        if audio_supported and delay_beats <= MAX_CONFIRMED_INTRO_DELAY_BEATS
        else "VIOLATION"
    )


def _safe_boundary_state(
    candidate: Candidate,
    boundary: SongBoundaryContract | None,
) -> str:
    if (
        candidate.acceptance.decision(GateAxis.SONG_BOUNDS).action
        is not GateAction.PASS
    ):
        return "VIOLATION"
    if boundary is None or boundary.effective_source != "TERMINAL_SILENCE_CONSENSUS":
        return "UNKNOWN"
    for note in candidate.generated.notes:
        if note.time_ms > boundary.max_note_start_ms:
            return "VIOLATION"
        end_ms = note.time_ms + (
            note.duration_ms if note.kind == "HOLD" and note.duration_ms is not None else 0
        )
        if end_ms > boundary.release_end_ms:
            return "VIOLATION"
    return "CONFIRMED_SAFE"


def _safe_family_view(
    candidate: Candidate,
    *,
    candidate_id: str,
    source_difficulty: str,
    intro_contract: IntroStartContract,
    boundary: SongBoundaryContract | None,
    song_context: SongAnalysisContext | None,
    outro_observation: OutroObservation | None = None,
) -> SafeFamilyCandidate:
    acceptance = candidate.acceptance
    gaps = tuple(acceptance.timing.coverage_gaps)
    attack_gaps = tuple(
        gap
        for gap in gaps
        if gap.opportunity is None
        or gap.opportunity.kind.value == "ATTACK_REQUIRED"
    )
    profile = acceptance.profile
    difficulty_scores = []
    if profile is not None:
        difficulty_scores.append(
            ("PROJECT_RATING", float(profile.difficulty.project_rating))
        )
        vector = profile.difficulty_vector_v2
        if vector is not None:
            difficulty_scores.append(("ORDERING_SCORE", float(vector.ordering_score)))
    matched_f1 = acceptance.timing.overall.matched_f1_50
    hard_safe = all(
        acceptance.decision(axis).action is GateAction.PASS
        for axis in (
            GateAxis.STRUCTURE,
            GateAxis.TIMING_IDENTITY,
            GateAxis.SONG_BOUNDS,
        )
    )
    (
        tail_coverage_deficit_ms,
        tail_active_onset_count,
        terminal_overflow_ms,
        terminal_overflow_confidence,
    ) = _continuous_tail_risk(
        candidate,
        boundary=boundary,
        song_context=song_context,
        outro_observation=outro_observation,
    )
    return SafeFamilyCandidate(
        candidate_id=candidate_id,
        candidate_payload_sha256=hashlib.sha256(
            candidate.osu_text.encode("utf-8")
        ).hexdigest(),
        key_mode=candidate.request.key_mode,
        source_difficulty=source_difficulty,
        provenance=candidate.provenance,
        hard_safe=hard_safe,
        intro_state=_safe_intro_state(candidate, intro_contract, song_context),
        boundary_state=_safe_boundary_state(candidate, boundary),
        attack_gap_count=len(attack_gaps),
        attack_gap_total_ms=sum(gap.end_ms - gap.start_ms for gap in attack_gaps),
        active_gap_count=len(gaps),
        max_active_gap_ms=max((gap.end_ms - gap.start_ms for gap in gaps), default=0),
        difficulty_scores=tuple(sorted(difficulty_scores)),
        review_rank=(
            0
            if acceptance.action is GateAction.PASS
            else 1
            if acceptance.action is GateAction.REVIEW
            else 2
        ),
        publication_rank={
            "PRODUCTION_CANDIDATE": 0,
            "PLAYTEST_ONLY": 1,
            "DIAGNOSTIC_ONLY": 2,
        }[_publication_tier(candidate)],
        recovery_trust_rank=_recovery_trust_rank(candidate),
        matched_f1_50=(float(matched_f1) if matched_f1 is not None else None),
        attempt=candidate.attempt,
        intro_distance_ms=_intro_distance_ms(candidate, intro_contract),
        tail_coverage_deficit_ms=tail_coverage_deficit_ms,
        tail_active_onset_count=tail_active_onset_count,
        terminal_overflow_ms=terminal_overflow_ms,
        terminal_overflow_confidence=terminal_overflow_confidence,
    )


def apply_safe_family_assignment(
    selections: list[Selection],
    *,
    run_dir: Path,
    intro_contract: IntroStartContract,
    boundary: SongBoundaryContract | None,
    song_context: SongAnalysisContext | None = None,
    post_resolution_ordering: bool = False,
) -> tuple[list[Selection], tuple[SafeFamilyAssignmentDecision, ...]]:
    """Reassign existing playtest candidates within each key; never calls a model."""
    updated: list[Selection] = []
    decisions = []
    outro_observation = (
        observe_outro(
            song_context.onset_analysis.activity,
            song_context.duration_ms,
        )
        if (
            song_context is not None
            and song_context.onset_analysis.activity is not None
        )
        else None
    )
    for states, current_assignment, _review in selections:
        key_mode = next(iter(states.values())).key_mode
        candidates_by_id: dict[str, Candidate] = {}
        ids_by_object: dict[int, str] = {}
        views = []
        for source_difficulty in DIFFICULTIES:
            state = states[source_difficulty]
            for candidate in state.candidates.playtest_candidates:
                if candidate.request.key_mode != key_mode:
                    raise ValueError("candidate key differs from its family")
                if candidate.request.difficulty != source_difficulty:
                    raise ValueError("candidate source difficulty differs from repository slot")
                candidate_id = candidate_stable_id(
                    candidate,
                    key_mode=key_mode,
                    difficulty=source_difficulty,
                    run_dir=run_dir,
                )
                if candidate_id in candidates_by_id:
                    raise ValueError("candidate IDs must be unique inside a key family")
                candidates_by_id[candidate_id] = candidate
                ids_by_object[id(candidate)] = candidate_id
                views.append(
                    _safe_family_view(
                        candidate,
                        candidate_id=candidate_id,
                        source_difficulty=source_difficulty,
                        intro_contract=intro_contract,
                        boundary=boundary,
                        song_context=song_context,
                        outro_observation=outro_observation,
                    )
                )
        current = tuple(
            (difficulty, ids_by_object[id(current_assignment[difficulty])])
            for difficulty in DIFFICULTIES
            if current_assignment[difficulty] is not None
        )
        if len(current) != len(DIFFICULTIES):
            updated.append((states, current_assignment, _review))
            continue
        decision = select_safe_family_assignment(
            tuple(views),
            current_assignment=current,
            post_resolution_ordering=post_resolution_ordering,
        )
        decisions.append(decision)
        assignment = {
            difficulty: candidates_by_id[candidate_id]
            for difficulty, candidate_id in decision.selected_assignment
        }
        if decision.changed:
            sources = dict(decision.source_difficulties)
            duplicates = set(decision.emergency_duplicate_slots)
            for difficulty in DIFFICULTIES:
                states[difficulty].attempt_evidence.append(
                    {
                        "reason": "SAFE_FAMILY_ASSIGNMENT_APPLIED",
                        "targetDifficulty": difficulty,
                        "sourceDifficulty": sources[difficulty],
                        "candidateId": dict(decision.selected_assignment)[difficulty],
                        "assignmentKind": (
                            "EMERGENCY_DUPLICATE"
                            if difficulty in duplicates
                            else "REASSIGNED"
                            if sources[difficulty] != difficulty
                            else "ORIGINAL"
                        ),
                        "decisionReason": decision.reason,
                        "additionalModelCalls": 0,
                    }
                )
                if post_resolution_ordering:
                    states[difficulty].attempt_evidence.append(
                        {
                            "reason": "POST_RESOLUTION_DIFFICULTY_ORDERING_APPLIED",
                            "targetDifficulty": difficulty,
                            "sourceDifficulty": sources[difficulty],
                            "candidateId": dict(decision.selected_assignment)[difficulty],
                            "mutatesPayload": False,
                            "additionalModelCalls": 0,
                        }
                    )
        updated.append(
            (
                states,
                assignment,
                review_assignment_candidates(assignment),
            )
        )
    return updated, tuple(decisions)


def _evidence_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _gap_evidence(candidate: Candidate) -> tuple[GapIntervalEvidence, ...]:
    evidence = []
    for gap in candidate.acceptance.timing.coverage_gaps:
        local_report = (
            gap.local_audio_evidence.to_report() if gap.local_audio_evidence is not None else None
        )
        evidence.append(
            GapIntervalEvidence(
                start_ms=gap.start_ms,
                end_ms=gap.end_ms,
                position=gap.position,
                active_onset_count=gap.active_onset_count,
                active_frame_ratio=float(gap.active_frame_ratio),
                opportunity_kind=(
                    gap.opportunity.kind.value if gap.opportunity is not None else "UNKNOWN"
                ),
                local_audio_evidence_digest=(
                    _evidence_digest(local_report) if local_report is not None else None
                ),
            )
        )
    return tuple(sorted(evidence, key=lambda item: (item.start_ms, item.end_ms)))


def _publication_tier(candidate: Candidate) -> str:
    if (
        candidate.provenance == "RAW_UNVERIFIED"
        or candidate.acceptance.action is GateAction.RETRY_MAP
    ):
        return "DIAGNOSTIC_ONLY"
    if candidate.provenance in {"COVERAGE_REPAIR", "SAFE_FALLBACK"}:
        return "PLAYTEST_ONLY"
    return "PRODUCTION_CANDIDATE"


def _recovery_trust_rank(candidate: Candidate) -> int:
    return {
        "PRIMARY": 0,
        "RETRY": 0,
        "PARTIAL_REMAP": 1,
        "INTRO_RECOVERY": 1,
        "INTRO_ALIGNED": 1,
        "COVERAGE_REPAIR": 2,
        "RAW_UNVERIFIED": 3,
        "SAFE_FALLBACK": 4,
    }[candidate.provenance]


def _candidate_safety_v3(
    candidate: Candidate,
    *,
    candidate_id: str,
) -> CandidateSafetyEvidenceV3:
    acceptance = candidate.acceptance
    return CandidateSafetyEvidenceV3(
        candidate_id=candidate_id,
        structure_safe=(acceptance.decision(GateAxis.STRUCTURE).action is GateAction.PASS),
        timing_identity_safe=(
            acceptance.decision(GateAxis.TIMING_IDENTITY).action is GateAction.PASS
        ),
        song_bounds_safe=(acceptance.decision(GateAxis.SONG_BOUNDS).action is GateAction.PASS),
        serialization_safe=True,
        publication_tier=_publication_tier(candidate),
        model_backed=candidate.provenance != "SAFE_FALLBACK",
        recovery_trust_rank=_recovery_trust_rank(candidate),
        active_gaps=_gap_evidence(candidate),
    )


def _first_row_grid_distance_ms(
    song_context: SongAnalysisContext,
    time_ms: int,
) -> int:
    event = song_context.tempo_map.at(time_ms)
    half_beat_ms = 30_000.0 / event.bpm
    step = round((time_ms - event.time_ms) / half_beat_ms)
    grid_ms = max(0, round(event.time_ms + step * half_beat_ms))
    return abs(time_ms - grid_ms)


def build_song_selection_evidence_v3(
    selections: list[Selection],
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    run_dir: Path,
    song_context: SongAnalysisContext,
    boundary: SongBoundaryContract | None,
) -> SongSelectionEvidenceV3:
    """Project every preserved candidate into immutable, report-only V3 evidence."""
    votes = tuple(
        IntroCandidateVoteV3(
            slot=f"{next(iter(states.values())).key_mode}K:{difficulty}",
            candidate_id=candidate_stable_id(
                candidate,
                key_mode=next(iter(states.values())).key_mode,
                difficulty=difficulty,
                run_dir=run_dir,
            ),
            first_row_ms=first_row_ms(candidate),
        )
        for states, _assignment, _review in selections
        for difficulty in DIFFICULTIES
        for candidate in states[difficulty].candidates.playtest_candidates
    )
    active_onsets = (
        song_context.onset_analysis.activity.active_onset_ms
        if song_context.onset_analysis.activity is not None
        else ()
    )
    intro_selection = build_intro_selection_evidence(
        song_context.intro_anchor,
        active_onset_ms=active_onsets,
        votes=votes,
    )
    context_payload = {
        "version": "song-family-evidence-v3-context",
        "audioSha256": authority.audio_sha256,
        "timingAuthoritySha256": authority.sha256,
        "durationMs": prepared.normalized.duration_ms,
        "qualityGateVersion": QUALITY_GATE_VERSION,
        "songBoundaryContractSha256": (boundary.stable_sha256() if boundary is not None else None),
        "introSelectionSha256": intro_selection.stable_sha256(),
    }
    context_id = _evidence_digest(context_payload)

    records: list[CandidateFamilyEvidenceV3] = []
    assignment: list[tuple[str, str | None]] = []
    for states, selected, _review in selections:
        key_mode = next(iter(states.values())).key_mode
        family_candidate_ids: dict[int, str] = {}
        for difficulty in DIFFICULTIES:
            repository = states[difficulty].candidates
            shadow_candidate_ids = {
                id(candidate) for candidate in repository.shadow_candidates
            }
            for candidate in repository.evidence_candidates:
                artifact = candidate_payload_artifact(
                    run_dir=run_dir,
                    osu_text=candidate.osu_text,
                )
                candidate_id = candidate_stable_id(
                    candidate,
                    key_mode=key_mode,
                    difficulty=difficulty,
                    run_dir=run_dir,
                )
                # Preserve the first repository identity for an object.  Some
                # legacy recovery paths keep the same immutable candidate in
                # more than one repository; target assignment must still refer
                # to one concrete evidence record rather than synthesizing a
                # new identity from the target label.
                family_candidate_ids.setdefault(id(candidate), candidate_id)
                row_ms = first_row_ms(candidate)
                audio_supported = row_ms is not None and (
                    (
                        intro_selection.reference_state == "CONFIRMED_AUDIO"
                        and intro_selection.reference_first_row_ms is not None
                        and abs(row_ms - intro_selection.reference_first_row_ms)
                        <= GRID_SUPPORT_WINDOW_MS
                    )
                    or any(
                        abs(row_ms - onset_ms) <= GRID_SUPPORT_WINDOW_MS
                        for onset_ms in active_onsets
                    )
                )
                records.append(
                    CandidateFamilyEvidenceV3(
                        candidate_id=candidate_id,
                        key_mode=key_mode,
                        difficulty=difficulty,
                        provenance=candidate.provenance,
                        candidate_payload_ref=artifact.relative_path.as_posix(),
                        candidate_payload_sha256=artifact.sha256,
                        safety=_candidate_safety_v3(
                            candidate,
                            candidate_id=candidate_id,
                        ),
                        first_row_ms=row_ms,
                        first_row_audio_supported=audio_supported,
                        first_row_grid_distance_ms=(
                            _first_row_grid_distance_ms(song_context, row_ms)
                            if row_ms is not None
                            else None
                        ),
                        intro_reference_state=intro_selection.reference_state,
                        matched_f1_50=(
                            float(candidate.acceptance.timing.overall.matched_f1_50)
                            if candidate.acceptance.timing.overall.matched_f1_50 is not None
                            else None
                        ),
                        matched_precision_50=(
                            float(candidate.acceptance.timing.overall.matched_precision_50)
                            if candidate.acceptance.timing.overall.matched_precision_50 is not None
                            else None
                        ),
                        review_rank=(
                            0
                            if candidate.acceptance.action is GateAction.PASS
                            else 1
                            if candidate.acceptance.action is GateAction.REVIEW
                            else 2
                        ),
                        candidate_role=(
                            "SHADOW_CHALLENGER"
                            if id(candidate) in shadow_candidate_ids
                            else "PLAYTEST_POOL"
                        ),
                        eligible_target_difficulties=DIFFICULTIES,
                    )
                )
        # Resolve target labels only after all four source repositories have
        # been indexed.  EASY may legitimately select an EXPERT-source payload
        # after relative-difficulty relabelling (and vice versa).
        for difficulty in DIFFICULTIES:
            chosen = selected[difficulty]
            chosen_id = None
            if chosen is not None:
                # A safe-family decision may intentionally assign a candidate
                # from another difficulty repository to this target label.
                # Evidence identity belongs to the immutable source candidate,
                # while the assignment key below records the target label.
                chosen_id = family_candidate_ids.get(id(chosen))
                if chosen_id is None:
                    raise ValueError(
                        "selected family candidate is not preserved in its source repository"
                    )
            assignment.append(
                (
                    f"{key_mode}K:{difficulty}",
                    chosen_id,
                )
            )
    return SongSelectionEvidenceV3(
        context_id=context_id,
        intro_selection=intro_selection,
        candidates=tuple(sorted(records, key=lambda item: item.candidate_id)),
        current_assignment=tuple(sorted(assignment)),
    )


def compare_song_selection(
    selections: list[Selection],
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    run_dir: Path,
    intro_contract: IntroStartContract,
    boundary: SongBoundaryContract | None,
    mode: SelectorMode,
) -> tuple[list[Selection], SongSelectionComparison]:
    context_id = song_selection_context_id(
        prepared,
        authority,
        boundary,
        intro_contract,
    )
    pools: dict[tuple[int, str], tuple[CandidateSnapshot, ...]] = {}
    current: dict[tuple[int, str], str | None] = {}
    candidates_by_id: dict[str, Candidate] = {}
    for states, assignment, _review in selections:
        key_mode = next(iter(states.values())).key_mode
        for difficulty in DIFFICULTIES:
            state = states[difficulty]
            snapshots = []
            ids: dict[int, str] = {}
            for candidate in state.candidates.playtest_candidates:
                artifact = candidate_payload_artifact(
                    run_dir=run_dir,
                    osu_text=candidate.osu_text,
                )
                candidate_id = candidate_stable_id(
                    candidate,
                    key_mode=key_mode,
                    difficulty=difficulty,
                    run_dir=run_dir,
                )
                if candidate_id in ids.values():
                    raise ValueError(f"duplicate deterministic candidate id: {candidate_id}")
                ids[id(candidate)] = candidate_id
                candidates_by_id[candidate_id] = candidate
                acceptance = candidate.acceptance
                profile = acceptance.profile
                vector = profile.difficulty_vector_v2 if profile is not None else None
                snapshots.append(
                    CandidateSnapshot(
                        candidate_id=candidate_id,
                        context_id=context_id,
                        key_mode=key_mode,
                        difficulty=difficulty,
                        attempt=candidate.attempt,
                        seed=candidate.seed,
                        provenance=candidate.provenance,
                        hard_eligible=(
                            acceptance.action is not GateAction.RETRY_MAP
                            and candidate.provenance not in {"RAW_UNVERIFIED", "SAFE_FALLBACK"}
                        ),
                        axis_actions=tuple(
                            (decision.axis.value, decision.action.value)
                            for decision in acceptance.decisions
                        ),
                        protected_metrics=ProtectedMetrics(
                            row_count=acceptance.timing.overall.row_count,
                            onset_count=acceptance.timing.onset_count,
                            matched_count_50=(acceptance.timing.overall.matched_count_50),
                            matched_precision_50=(acceptance.timing.overall.matched_precision_50),
                            matched_recall_50=(acceptance.timing.overall.matched_recall_50),
                            matched_f1_50=(acceptance.timing.overall.matched_f1_50),
                            active_gap_count=len(acceptance.timing.coverage_gaps),
                            hold_integrity_violations=0,
                            review_rank=(
                                0
                                if acceptance.action is GateAction.PASS
                                else 1
                                if acceptance.action is GateAction.REVIEW
                                else 2
                            ),
                        ),
                        difficulty_ordering_score=(
                            vector.ordering_score if vector is not None else None
                        ),
                        first_row_ms=first_row_ms(candidate),
                        timing_sections=tuple(
                            TimingSectionSnapshot(
                                row_count=section.metrics.row_count,
                                matched_precision_50=(section.metrics.matched_precision_50),
                            )
                            for section in acceptance.timing.sections
                        ),
                        candidate_payload_ref=artifact.relative_path.as_posix(),
                        candidate_payload_sha256=artifact.sha256,
                    )
                )
            pools[(key_mode, difficulty)] = tuple(snapshots)
            selected = assignment[difficulty]
            current[(key_mode, difficulty)] = ids[id(selected)] if selected is not None else None
    selected, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=intro_contract.canonical_first_row_ms,
        mode=mode,
    )
    if mode == "SHADOW_V2":
        return selections, comparison

    updated: list[Selection] = []
    for states, _assignment, _review in selections:
        key_mode = next(iter(states.values())).key_mode
        assignment = {
            difficulty: (
                candidates_by_id[candidate_id]
                if (candidate_id := selected[(key_mode, difficulty)]) is not None
                else None
            )
            for difficulty in DIFFICULTIES
        }
        chosen = tuple(
            candidate
            for difficulty in DIFFICULTIES
            if (candidate := assignment[difficulty]) is not None
        )
        updated.append(
            (
                states,
                assignment,
                review_candidates(chosen) if chosen else None,
            )
        )
    return updated, comparison


def compare_song_selection_shadow(
    selections: list[Selection],
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    run_dir: Path,
    intro_contract: IntroStartContract,
    boundary: SongBoundaryContract | None,
) -> SongSelectionComparison:
    """Backward-compatible observation-only wrapper."""
    _unchanged, comparison = compare_song_selection(
        selections,
        prepared=prepared,
        authority=authority,
        run_dir=run_dir,
        intro_contract=intro_contract,
        boundary=boundary,
        mode="SHADOW_V2",
    )
    return comparison
