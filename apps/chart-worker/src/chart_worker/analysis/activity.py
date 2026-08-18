"""Song-relative audio activity used by timing diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from math import ceil, floor
from typing import Literal

import numpy as np

from chart_worker.analysis.terminal_silence import (
    TerminalSilenceObservation,
    consensus_terminal_boundary_ms,
)

SILENCE_DB = -60.0
RMS_FLOOR_PERCENTILE = 20.0
ONSET_FLOOR_PERCENTILE = 25.0

LOCAL_WINDOW_MS = 5_000.0
"""국소 대비를 잴 창 반경. 인트로 한 구절 정도를 덮는다."""

LOCAL_PROMINENCE_DB = 6.0
"""주변 중앙값보다 이만큼 크면 국소적으로 두드러진 타점으로 본다.

곡 전체 RMS 하위 20 퍼센타일 floor 하나로만 판정하면 본편보다 작게
믹스된 인트로의 명확한 타점(onset strength 1.0)까지 quiet 로 분류된다.
24곡 배치에서 22곡 중 21곡의 leadingCoverage 가 이 이유로
QUIET_LEADING_TIMING_GAP PASS 가 됐고, 사용자가 지적한 인트로 미정렬
8곡을 어떤 게이트도 잡지 못했다. 전역 floor 는 유지하되, 국소 대비가
충분한 타점은 조용한 구간 안에서도 active 로 인정한다."""


@dataclass(frozen=True, slots=True)
class AudioActivity:
    frame_ms: float
    rms_db: np.ndarray
    floor_db: float
    active_onset_ms: tuple[int, ...]

    def active_frame_ratio(self, start_ms: int, end_ms: int) -> float:
        """Return the active RMS-frame share after clipping to available audio."""
        if self.frame_ms <= 0 or self.rms_db.size == 0 or end_ms <= start_ms:
            return 0.0
        start = max(0, floor(start_ms / self.frame_ms))
        end = min(self.rms_db.size, ceil(end_ms / self.frame_ms))
        if end <= start:
            return 0.0
        window = self.rms_db[start:end]
        return float(np.count_nonzero(window > self.floor_db) / window.size)


@dataclass(frozen=True, slots=True)
class OutroObservation:
    version: Literal["outro-observation-v2"]
    duration_ms: int
    last_active_rms_end_ms: int | None
    last_detected_onset_ms: int | None
    last_evidence_ms: int | None
    trailing_below_floor_ms: int | None
    tail_rms_percentiles_db: dict[str, float | None]

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "durationMs": self.duration_ms,
            "lastActiveRmsEndMs": self.last_active_rms_end_ms,
            "lastDetectedOnsetMs": self.last_detected_onset_ms,
            "lastEvidenceMs": self.last_evidence_ms,
            "trailingBelowFloorMs": self.trailing_below_floor_ms,
            "tailRmsPercentilesDb": self.tail_rms_percentiles_db,
        }

    def stable_sha256(self) -> str:
        payload = json.dumps(
            self.to_report(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class OutroPolicyDecision:
    version: Literal["outro-policy-v1-provisional"]
    observation_sha256: str
    selected_music_end_ms: int
    last_attack_ms: int
    release_end_ms: int
    generation_end_ms: int
    required_coverage_end_ms: int
    policy_reason: str
    policy_applied: bool
    policy_state: Literal["PROVISIONAL"]
    confidence: Literal["UNKNOWN"]

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "observationSha256": self.observation_sha256,
            "selectedMusicEndMs": self.selected_music_end_ms,
            "lastAttackMs": self.last_attack_ms,
            "releaseEndMs": self.release_end_ms,
            "generationEndMs": self.generation_end_ms,
            "requiredCoverageEndMs": self.required_coverage_end_ms,
            "policyReason": self.policy_reason,
            "policyApplied": self.policy_applied,
            "policyState": self.policy_state,
            "confidence": self.confidence,
        }


BOUNDARY_QUANTIZATION_TOLERANCE_MS = 70
HOLD_COMPLETION_GUARD_MS = 10_000
OUTRO_POLICY_VERSION = "outro-policy-v1-provisional"
SONG_BOUNDARY_CONTRACT_VERSION = "song-boundary-contract-v2"


@dataclass(frozen=True, slots=True)
class SongBoundaryContract:
    version: Literal["song-boundary-contract-v2"]
    last_attack_ms: int
    max_note_start_ms: int
    release_end_ms: int
    generation_end_ms: int
    required_coverage_end_ms: int
    quantization_tolerance_ms: int
    hold_completion_guard_ms: int
    policy_reason: str
    policy_applied: bool
    policy_state: Literal["PROVISIONAL"] = "PROVISIONAL"
    confidence: Literal["UNKNOWN"] = "UNKNOWN"
    enforcement_mode: Literal[
        "SHADOW",
        "EXPERIMENTAL_ENFORCED",
        "HIGH_CONFIDENCE_ENFORCED",
    ] = "SHADOW"
    effective_source: Literal[
        "FULL_DURATION_BASELINE",
        "PROVISIONAL_POLICY",
        "TERMINAL_SILENCE_CONSENSUS",
    ] = "FULL_DURATION_BASELINE"

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "lastAttackMs": self.last_attack_ms,
            "maxNoteStartMs": self.max_note_start_ms,
            "releaseEndMs": self.release_end_ms,
            "generationEndMs": self.generation_end_ms,
            "requiredCoverageEndMs": self.required_coverage_end_ms,
            "quantizationToleranceMs": self.quantization_tolerance_ms,
            "holdCompletionGuardMs": self.hold_completion_guard_ms,
            "policyReason": self.policy_reason,
            "policyApplied": self.policy_applied,
            "policyState": self.policy_state,
            "confidence": self.confidence,
            "enforcementMode": self.enforcement_mode,
            "effectiveSource": self.effective_source,
        }

    def stable_sha256(self) -> str:
        payload = json.dumps(
            self.to_report(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


BoundaryPolicyMode = Literal[
    "SHADOW",
    "EXPERIMENTAL_ENFORCED",
    "HIGH_CONFIDENCE_ENFORCED",
]
BoundaryEffectiveSource = Literal[
    "FULL_DURATION_BASELINE",
    "PROVISIONAL_POLICY",
    "TERMINAL_SILENCE_CONSENSUS",
]


@dataclass(frozen=True, slots=True)
class BoundaryPolicyEvaluation:
    version: Literal["boundary-policy-evaluation-v1"]
    policy_state: Literal["PROVISIONAL"]
    confidence: Literal["UNKNOWN"]
    enforcement_mode: BoundaryPolicyMode
    observation_sha256: str
    provisional_decision: OutroPolicyDecision
    provisional_contract: SongBoundaryContract
    full_duration_contract: SongBoundaryContract
    effective_source: BoundaryEffectiveSource
    effective_contract: SongBoundaryContract

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "policyState": self.policy_state,
            "confidence": self.confidence,
            "enforcementMode": self.enforcement_mode,
            "observationSha256": self.observation_sha256,
            "provisionalDecision": self.provisional_decision.to_report(),
            "provisionalContract": self.provisional_contract.to_report(),
            "fullDurationContract": self.full_duration_contract.to_report(),
            "effectiveSource": self.effective_source,
            "effectiveContract": self.effective_contract.to_report(),
        }


TRAILING_SILENCE_MIN_MS = 5_000.0
"""이보다 짧은 꼬리 무음은 자르지 않는다. 정상 아웃트로를 보존한다."""

RELEASE_MARGIN_MS = 1_000.0
"""마지막 음악 증거 뒤에 남기는 리버브·릴리즈 여유."""
# Require two seconds whose 99th-percentile RMS remains below the established
# absolute silence floor before shortening a sub-five-second tail.
SHORT_ABSOLUTE_SILENCE_MIN_MS = 2_000.0
SHORT_ABSOLUTE_SILENCE_P99_DB = -60.0

MusicEndReason = Literal[
    "LONG_TRAILING_SILENCE",
    "SHORT_ABSOLUTE_SILENCE",
    "FULL_DURATION",
    "NO_EVIDENCE",
    "UNAVAILABLE",
]


def _last_music_evidence_ms(
    activity: AudioActivity,
    duration_ms: int,
) -> tuple[int | None, int | None, int | None]:
    """Return RMS, onset, and combined tail evidence on the canonical timeline."""
    active = np.flatnonzero(activity.rms_db > activity.floor_db)
    last_active_rms_end_ms = (
        min(duration_ms, round((int(active[-1]) + 1) * activity.frame_ms))
        if active.size
        else None
    )
    last_detected_onset_ms = next(
        (
            onset_ms
            for onset_ms in reversed(activity.active_onset_ms)
            if 0 <= onset_ms <= duration_ms
        ),
        None,
    )
    evidence = tuple(
        value
        for value in (last_active_rms_end_ms, last_detected_onset_ms)
        if value is not None
    )
    return (
        last_active_rms_end_ms,
        last_detected_onset_ms,
        max(evidence) if evidence else None,
    )


def _tail_percentile_db(
    activity: AudioActivity,
    duration_ms: int,
    *,
    window_ms: int,
    percentile: float,
) -> float | None:
    if activity.frame_ms <= 0 or activity.rms_db.size == 0 or duration_ms <= 0:
        return None
    end = min(activity.rms_db.size, ceil(duration_ms / activity.frame_ms))
    frame_count = max(1, ceil(window_ms / activity.frame_ms))
    start = max(0, end - frame_count)
    window = activity.rms_db[start:end]
    return float(np.percentile(window, percentile)) if window.size else None


def _legacy_rms_only_music_end_ms(
    activity: AudioActivity,
    duration_ms: int,
) -> int:
    if duration_ms <= 0 or activity.frame_ms <= 0 or activity.rms_db.size == 0:
        return duration_ms
    active = np.flatnonzero(activity.rms_db > activity.floor_db)
    if active.size == 0:
        return duration_ms
    last_active_end_ms = min(
        duration_ms,
        round((int(active[-1]) + 1) * activity.frame_ms),
    )
    if duration_ms - last_active_end_ms < TRAILING_SILENCE_MIN_MS:
        return duration_ms
    return min(duration_ms, round(last_active_end_ms + RELEASE_MARGIN_MS))


def estimate_music_end_ms(activity: AudioActivity, duration_ms: int) -> int:
    """채보 생성을 멈출 음악 종료 시각을 보수적으로 추정한다.

    오디오 파일 길이와 음악 종료는 다르다. 24곡 배치에서 5곡이 5초
    이상의 꼬리 무음을 가졌고, 그 길이가 그대로 모델 `end_time` 으로
    들어가 음악이 끝난 뒤에도 노트가 떨어졌다 (사용자 확인: song 12).

    규칙은 하나다: 마지막 활성 프레임 이후 무음이 충분히 길 때만
    자르고, 여유를 더해 페이드아웃·리버브를 보존한다. 마지막 onset 은
    쓰지 않는다 — onset 이 20초 이상 비는 활성 구간도 실측으로 존재한다.
    """
    return evaluate_outro_policy(
        observe_outro(activity, duration_ms)
    ).selected_music_end_ms


def observe_outro(activity: AudioActivity, duration_ms: int) -> OutroObservation:
    """Record analyzer output only; no threshold policy is selected here."""
    if duration_ms <= 0 or activity.frame_ms <= 0 or activity.rms_db.size == 0:
        return OutroObservation(
            version="outro-observation-v2",
            duration_ms=duration_ms,
            last_active_rms_end_ms=None,
            last_detected_onset_ms=(
                activity.active_onset_ms[-1] if activity.active_onset_ms else None
            ),
            last_evidence_ms=None,
            trailing_below_floor_ms=None,
            tail_rms_percentiles_db={
                "1sP50": None,
                "2sP50": None,
                "2sP99": None,
                "5sP50": None,
            },
        )

    (
        last_active_rms_end_ms,
        last_detected_onset_ms,
        last_evidence_ms,
    ) = _last_music_evidence_ms(
        activity,
        duration_ms,
    )
    trailing_below_floor_ms = (
        max(0, duration_ms - last_active_rms_end_ms)
        if last_active_rms_end_ms is not None
        else duration_ms
    )
    tail_percentiles: dict[str, float | None] = {}
    for window_ms, label in ((1_000, "1sP50"), (2_000, "2sP50"), (5_000, "5sP50")):
        value = _tail_percentile_db(
            activity,
            duration_ms,
            window_ms=window_ms,
            percentile=50,
        )
        tail_percentiles[label] = round(value, 6) if value is not None else None
    two_second_p99 = _tail_percentile_db(
        activity,
        duration_ms,
        window_ms=2_000,
        percentile=99,
    )
    tail_percentiles["2sP99"] = (
        round(two_second_p99, 6) if two_second_p99 is not None else None
    )
    return OutroObservation(
        version="outro-observation-v2",
        duration_ms=duration_ms,
        last_active_rms_end_ms=last_active_rms_end_ms,
        last_detected_onset_ms=last_detected_onset_ms,
        last_evidence_ms=last_evidence_ms,
        trailing_below_floor_ms=trailing_below_floor_ms,
        tail_rms_percentiles_db=tail_percentiles,
    )


def evaluate_outro_policy(observation: OutroObservation) -> OutroPolicyDecision:
    """Apply the provisional policy to a reusable, policy-free observation."""
    duration_ms = observation.duration_ms
    reason: MusicEndReason
    applied = False
    if duration_ms <= 0 or observation.last_evidence_ms is None:
        reason = (
            "UNAVAILABLE"
            if observation.trailing_below_floor_ms is None
            else "NO_EVIDENCE"
        )
    else:
        trailing_ms = duration_ms - observation.last_evidence_ms
        tail_p99_db = observation.tail_rms_percentiles_db.get("2sP99")
        if trailing_ms >= TRAILING_SILENCE_MIN_MS:
            reason = "LONG_TRAILING_SILENCE"
            applied = True
        elif (
            trailing_ms >= SHORT_ABSOLUTE_SILENCE_MIN_MS
            and tail_p99_db is not None
            and tail_p99_db <= SHORT_ABSOLUTE_SILENCE_P99_DB
        ):
            reason = "SHORT_ABSOLUTE_SILENCE"
            applied = True
        else:
            reason = "FULL_DURATION"

    last_attack_ms = (
        observation.last_evidence_ms
        if applied and observation.last_evidence_ms is not None
        else duration_ms
    )
    selected_music_end_ms = (
        min(duration_ms, round(last_attack_ms + RELEASE_MARGIN_MS))
        if applied
        else duration_ms
    )
    completion_end_ms = (
        min(duration_ms, last_attack_ms + HOLD_COMPLETION_GUARD_MS)
        if applied
        else duration_ms
    )
    return OutroPolicyDecision(
        version=OUTRO_POLICY_VERSION,
        observation_sha256=observation.stable_sha256(),
        selected_music_end_ms=selected_music_end_ms,
        last_attack_ms=last_attack_ms,
        release_end_ms=completion_end_ms,
        generation_end_ms=completion_end_ms,
        required_coverage_end_ms=last_attack_ms,
        policy_reason=reason,
        policy_applied=applied,
        policy_state="PROVISIONAL",
        confidence="UNKNOWN",
    )


def _provisional_contract(
    decision: OutroPolicyDecision,
    duration_ms: int,
) -> SongBoundaryContract:
    if decision.policy_applied:
        return SongBoundaryContract(
            version=SONG_BOUNDARY_CONTRACT_VERSION,
            last_attack_ms=decision.last_attack_ms,
            max_note_start_ms=min(
                duration_ms,
                decision.last_attack_ms + BOUNDARY_QUANTIZATION_TOLERANCE_MS,
            ),
            release_end_ms=decision.release_end_ms,
            generation_end_ms=decision.generation_end_ms,
            required_coverage_end_ms=decision.required_coverage_end_ms,
            quantization_tolerance_ms=BOUNDARY_QUANTIZATION_TOLERANCE_MS,
            hold_completion_guard_ms=HOLD_COMPLETION_GUARD_MS,
            policy_reason=decision.policy_reason,
            policy_applied=True,
            enforcement_mode="EXPERIMENTAL_ENFORCED",
            effective_source="PROVISIONAL_POLICY",
        )
    return SongBoundaryContract(
        version=SONG_BOUNDARY_CONTRACT_VERSION,
        last_attack_ms=duration_ms,
        max_note_start_ms=duration_ms,
        release_end_ms=duration_ms,
        generation_end_ms=duration_ms,
        required_coverage_end_ms=duration_ms,
        quantization_tolerance_ms=BOUNDARY_QUANTIZATION_TOLERANCE_MS,
        hold_completion_guard_ms=HOLD_COMPLETION_GUARD_MS,
        policy_reason=decision.policy_reason,
        policy_applied=False,
        enforcement_mode="EXPERIMENTAL_ENFORCED",
        effective_source="PROVISIONAL_POLICY",
    )


def _full_duration_contract(duration_ms: int) -> SongBoundaryContract:
    return SongBoundaryContract(
        version=SONG_BOUNDARY_CONTRACT_VERSION,
        last_attack_ms=duration_ms,
        max_note_start_ms=duration_ms,
        release_end_ms=duration_ms,
        generation_end_ms=duration_ms,
        required_coverage_end_ms=duration_ms,
        quantization_tolerance_ms=BOUNDARY_QUANTIZATION_TOLERANCE_MS,
        hold_completion_guard_ms=HOLD_COMPLETION_GUARD_MS,
        policy_reason="FULL_DURATION_BASELINE",
        policy_applied=False,
        enforcement_mode="SHADOW",
        effective_source="FULL_DURATION_BASELINE",
    )


def _terminal_consensus_contract(
    duration_ms: int,
    boundary_ms: int,
) -> SongBoundaryContract:
    return SongBoundaryContract(
        version=SONG_BOUNDARY_CONTRACT_VERSION,
        last_attack_ms=boundary_ms,
        max_note_start_ms=min(
            duration_ms,
            boundary_ms + BOUNDARY_QUANTIZATION_TOLERANCE_MS,
        ),
        # Keep the decoder horizon long enough to close an already-open HOLD,
        # but do not publish a release deep inside proven terminal silence.
        release_end_ms=boundary_ms,
        generation_end_ms=min(duration_ms, boundary_ms + HOLD_COMPLETION_GUARD_MS),
        required_coverage_end_ms=boundary_ms,
        quantization_tolerance_ms=BOUNDARY_QUANTIZATION_TOLERANCE_MS,
        hold_completion_guard_ms=HOLD_COMPLETION_GUARD_MS,
        policy_reason="TERMINAL_SILENCE_CONSENSUS",
        policy_applied=True,
        enforcement_mode="HIGH_CONFIDENCE_ENFORCED",
        effective_source="TERMINAL_SILENCE_CONSENSUS",
    )


def evaluate_boundary_policy(
    activity: AudioActivity,
    duration_ms: int,
    *,
    enforcement_mode: BoundaryPolicyMode = "SHADOW",
    terminal_silence: TerminalSilenceObservation | None = None,
) -> BoundaryPolicyEvaluation:
    """Dual-evaluate an uncalibrated boundary without hiding enforcement.

    The provisional detector remains observable, but the default effective
    contract is the full-duration baseline until a frozen audit calibrates it.
    """
    if enforcement_mode not in {
        "SHADOW",
        "EXPERIMENTAL_ENFORCED",
        "HIGH_CONFIDENCE_ENFORCED",
    }:
        raise ValueError(f"unsupported boundary enforcement mode: {enforcement_mode}")
    observation = observe_outro(activity, duration_ms)
    decision = evaluate_outro_policy(observation)
    provisional = _provisional_contract(decision, duration_ms)
    baseline = _full_duration_contract(duration_ms)
    if enforcement_mode == "EXPERIMENTAL_ENFORCED":
        source: BoundaryEffectiveSource = "PROVISIONAL_POLICY"
        effective = provisional
    elif enforcement_mode == "HIGH_CONFIDENCE_ENFORCED":
        terminal_boundary_ms = (
            None
            if terminal_silence is None
            else consensus_terminal_boundary_ms(terminal_silence)
        )
        if terminal_boundary_ms is None:
            source = "FULL_DURATION_BASELINE"
            effective = replace(
                baseline,
                enforcement_mode="HIGH_CONFIDENCE_ENFORCED",
            )
        else:
            source = "TERMINAL_SILENCE_CONSENSUS"
            effective = _terminal_consensus_contract(
                duration_ms,
                terminal_boundary_ms,
            )
    else:
        source = "FULL_DURATION_BASELINE"
        effective = baseline
    return BoundaryPolicyEvaluation(
        version="boundary-policy-evaluation-v1",
        policy_state="PROVISIONAL",
        confidence="UNKNOWN",
        enforcement_mode=enforcement_mode,
        observation_sha256=observation.stable_sha256(),
        provisional_decision=decision,
        provisional_contract=provisional,
        full_duration_contract=baseline,
        effective_source=source,
        effective_contract=effective,
    )


def build_song_boundary_contract(
    activity: AudioActivity,
    duration_ms: int,
    *,
    enforcement_mode: BoundaryPolicyMode = "SHADOW",
    terminal_silence: TerminalSilenceObservation | None = None,
) -> SongBoundaryContract:
    """Return the explicitly selected effective note/HOLD horizon contract."""
    return evaluate_boundary_policy(
        activity,
        duration_ms,
        enforcement_mode=enforcement_mode,
        terminal_silence=terminal_silence,
    ).effective_contract


def _window_bounds(
    frame: int,
    frame_count: int,
    *,
    n_fft: int | None,
    hop_length: int | None,
) -> tuple[int, int]:
    if n_fft is None or hop_length is None:
        return frame, frame + 1
    ahead = max(1, ceil(n_fft / hop_length))
    return max(0, frame - 1), min(frame_count, frame + ahead + 1)


def build_audio_activity(
    *,
    rms_db: np.ndarray,
    normalized_strength: np.ndarray,
    onset_frames: np.ndarray,
    frame_ms: float,
    silence_db: float = SILENCE_DB,
    n_fft: int | None = None,
    hop_length: int | None = None,
) -> AudioActivity:
    """Classify detected onsets using song-relative RMS and onset floors."""
    rms = np.asarray(rms_db, dtype=np.float64).reshape(-1)
    strength = np.asarray(normalized_strength, dtype=np.float64).reshape(-1)
    frame_count = min(rms.size, strength.size)
    rms = rms[:frame_count]
    strength = strength[:frame_count]

    non_silent = rms[rms > silence_db]
    floor_db = (
        float(np.percentile(non_silent, RMS_FLOOR_PERCENTILE))
        if non_silent.size
        else float(silence_db)
    )

    candidates: list[tuple[int, float, float]] = []
    for raw_frame in np.asarray(onset_frames, dtype=np.int64).reshape(-1):
        frame = int(raw_frame)
        if frame < 0 or frame >= frame_count:
            continue
        start, end = _window_bounds(
            frame,
            frame_count,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        candidates.append(
            (
                frame,
                float(np.max(rms[start:end])),
                float(np.max(strength[start:end])),
            )
        )

    if not candidates or not non_silent.size:
        active_onset_ms: tuple[int, ...] = ()
    else:
        onset_floor = float(
            np.percentile(
                np.asarray([candidate[2] for candidate in candidates]),
                ONSET_FLOOR_PERCENTILE,
            )
        )
        local_radius = max(1, round(LOCAL_WINDOW_MS / frame_ms)) if frame_ms > 0 else 1

        def _locally_prominent(frame: int, window_rms: float) -> bool:
            start = max(0, frame - local_radius)
            end = min(frame_count, frame + local_radius + 1)
            local = rms[start:end]
            if local.size == 0:
                return False
            return window_rms - float(np.median(local)) >= LOCAL_PROMINENCE_DB

        active_onset_ms = tuple(
            dict.fromkeys(
                round(frame * frame_ms)
                for frame, window_rms, window_strength in candidates
                if window_rms > silence_db
                and window_strength > 0
                and window_strength >= onset_floor
                and (
                    window_rms > floor_db
                    or _locally_prominent(frame, window_rms)
                )
            )
        )

    return AudioActivity(
        frame_ms=float(frame_ms),
        rms_db=rms,
        floor_db=floor_db,
        active_onset_ms=active_onset_ms,
    )
