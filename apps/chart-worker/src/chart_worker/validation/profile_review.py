"""Advisory review of song-relative section profile outliers."""

from math import log, sqrt
from statistics import median

from chart_worker.analysis.chart_profile import ChartQualityProfile
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.validation.quality_gate import (
    GateAction,
    GateAxis,
    GateDecision,
)

MIN_ACTIVE_SECTIONS = 4
ROBUST_Z_SCALE = 0.6745
ROBUST_Z_LIMIT = 3.5
LANE_CONFIDENCE_ALPHA = 0.01

_METRICS = (
    (
        "HOLD_SECTION_OUTLIER",
        lambda profile: profile.hold.section_occupancy_ratios,
        False,
    ),
    (
        "HOLD_RELEASE_SECTION_OUTLIER",
        lambda profile: profile.hold.section_release_counts_250ms,
        False,
    ),
    (
        "LANE_IMBALANCE_SECTION_OUTLIER",
        lambda profile: profile.pattern.section_lane_imbalances,
        True,
    ),
    (
        "ROW_LOOP_SECTION_OUTLIER",
        lambda profile: profile.pattern.section_longest_row_ngram_repeats,
        False,
    ),
)


def _robust_high_outlier_indices(values: tuple[float, ...]) -> tuple[int, ...] | None:
    center = float(median(values))
    mad = float(median(abs(value - center) for value in values))
    if mad == 0:
        return None
    return tuple(
        index
        for index, value in enumerate(values)
        if value > center
        and abs(ROBUST_Z_SCALE * (value - center) / mad) >= ROBUST_Z_LIMIT
    )


def _lane_concentration_supported(
    imbalance: float,
    *,
    note_count: int,
    key_mode: int,
) -> bool:
    """Reject small-sample lane noise with a conservative Hoeffding-style band."""
    if note_count <= 0:
        return False
    single_lane_error = sqrt(
        log(2 * key_mode / LANE_CONFIDENCE_ALPHA) / (2 * note_count)
    )
    return imbalance > 2 * single_lane_error


def review_profile(
    profile: ChartQualityProfile,
    *,
    key_mode: int,
    difficulty: str,
) -> tuple[GateDecision, ...]:
    """Return evidence-only PATTERN review without absolute difficulty targets."""
    if key_mode not in KEY_MODES:
        raise ValueError(f"unsupported key_mode: {key_mode}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unsupported difficulty: {difficulty}")

    active_indices = tuple(
        index for index, is_active in enumerate(profile.active_section_mask) if is_active
    )
    if len(active_indices) < MIN_ACTIVE_SECTIONS:
        return (
            GateDecision(
                GateAxis.PATTERN,
                GateAction.PASS,
                ("INSUFFICIENT_PROFILE_VARIATION",),
            ),
        )

    reasons: list[str] = []
    had_variation = False
    for reason, values_of, needs_lane_support in _METRICS:
        values = values_of(profile)
        active_values = tuple(float(values[index]) for index in active_indices)
        outlier_indices = _robust_high_outlier_indices(active_values)
        if outlier_indices is None:
            continue
        had_variation = True
        if needs_lane_support:
            outlier_indices = tuple(
                active_position
                for active_position in outlier_indices
                if _lane_concentration_supported(
                    active_values[active_position],
                    note_count=profile.pattern.section_note_counts[
                        active_indices[active_position]
                    ],
                    key_mode=key_mode,
                )
            )
        if outlier_indices:
            reasons.append(reason)

    if reasons:
        return (
            GateDecision(GateAxis.PATTERN, GateAction.REVIEW, tuple(reasons)),
        )
    return (
        GateDecision(
            GateAxis.PATTERN,
            GateAction.PASS,
            () if had_variation else ("INSUFFICIENT_PROFILE_VARIATION",),
        ),
    )
