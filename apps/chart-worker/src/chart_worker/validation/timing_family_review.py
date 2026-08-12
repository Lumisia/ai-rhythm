"""Cross-key timing review that treats sibling charts as relative evidence.

The absolute onset score is intentionally not a release gate.  Dense musical
subdivisions can be valid even when the onset detector has no separate peak for
each row.  A chart is an outlier only when it is worse than both sibling key
modes overall and the same local sections are also unusually dense.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Literal

from chart_worker.analysis.timing_diagnostics import TimingDiagnostics

# Measured on the latest complete run for 33 distinct audios (396 family
# members, 2026-08-09).  The 97.5th percentile of the one-to-one sibling gap
# was 0.085431.  This keeps the trigger in the observed tail instead of using
# an arbitrary absolute precision threshold.
OVERALL_SIBLING_GAP_MIN = 0.085

# Local corroboration measured on the same corpus.  Precision alone would
# misclassify valid dense subdivisions; require both a large local deficit and
# substantially higher row density for two consecutive 15-second sections.
LOCAL_SIBLING_GAP_MIN = 0.15
LOCAL_DENSITY_RATIO_MIN = 1.25
MIN_CONSECUTIVE_LOCAL_OUTLIERS = 2

TimingFamilyStatus = Literal["CONSISTENT", "OUTLIER", "INSUFFICIENT"]


@dataclass(frozen=True, slots=True)
class TimingFamilyCandidate:
    key_mode: int
    difficulty: str
    diagnostics: TimingDiagnostics


@dataclass(frozen=True, slots=True)
class TimingFamilyReview:
    status: TimingFamilyStatus
    difficulty: str | None
    target_key_mode: int | None
    overall_sibling_gap: float | None
    longest_local_outlier_run: int
    outlier_section_indices: tuple[int, ...]

    def to_report(self) -> dict[str, object]:
        return {
            "status": self.status,
            "difficulty": self.difficulty,
            "targetKeyMode": self.target_key_mode,
            "overallSiblingGap": self.overall_sibling_gap,
            "longestLocalOutlierRun": self.longest_local_outlier_run,
            "outlierSectionIndices": list(self.outlier_section_indices),
        }


def _longest_true_run(values: tuple[bool, ...]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _candidate_evidence(
    candidate: TimingFamilyCandidate,
    siblings: tuple[TimingFamilyCandidate, TimingFamilyCandidate],
) -> tuple[float, int, tuple[int, ...]] | None:
    precision = candidate.diagnostics.overall.matched_precision_50
    sibling_precisions = tuple(
        sibling.diagnostics.overall.matched_precision_50 for sibling in siblings
    )
    if precision is None or any(value is None for value in sibling_precisions):
        return None
    sibling_gap = float(median(sibling_precisions)) - precision  # type: ignore[arg-type]

    section_count = min(
        len(candidate.diagnostics.sections),
        *(len(sibling.diagnostics.sections) for sibling in siblings),
    )
    flags: list[bool] = []
    for index in range(section_count):
        section = candidate.diagnostics.sections[index].metrics
        sibling_sections = tuple(
            sibling.diagnostics.sections[index].metrics for sibling in siblings
        )
        section_precision = section.matched_precision_50
        sibling_section_precisions = tuple(
            sibling_section.matched_precision_50
            for sibling_section in sibling_sections
        )
        if section_precision is None or any(
            value is None for value in sibling_section_precisions
        ):
            flags.append(False)
            continue
        local_gap = (
            float(median(sibling_section_precisions)) - section_precision  # type: ignore[arg-type]
        )
        sibling_rows = float(
            median(sibling_section.row_count for sibling_section in sibling_sections)
        )
        density_ratio = section.row_count / max(1.0, sibling_rows)
        flags.append(
            section.row_count >= 8
            and local_gap >= LOCAL_SIBLING_GAP_MIN
            and density_ratio >= LOCAL_DENSITY_RATIO_MIN
        )
    flag_tuple = tuple(flags)
    return (
        round(sibling_gap, 6),
        _longest_true_run(flag_tuple),
        tuple(index for index, value in enumerate(flag_tuple) if value),
    )


def review_timing_family(
    candidates: tuple[TimingFamilyCandidate, ...],
) -> TimingFamilyReview:
    if len(candidates) != 3 or {candidate.key_mode for candidate in candidates} != {
        4,
        6,
        7,
    }:
        return TimingFamilyReview("INSUFFICIENT", None, None, None, 0, ())
    difficulties = {candidate.difficulty for candidate in candidates}
    if len(difficulties) != 1:
        return TimingFamilyReview("INSUFFICIENT", None, None, None, 0, ())

    evidence = []
    for candidate in candidates:
        siblings = tuple(other for other in candidates if other is not candidate)
        measured = _candidate_evidence(candidate, siblings)  # type: ignore[arg-type]
        if measured is not None:
            evidence.append((candidate, *measured))
    if not evidence:
        return TimingFamilyReview(
            "INSUFFICIENT",
            next(iter(difficulties)),
            None,
            None,
            0,
            (),
        )

    candidate, gap, longest_run, section_indices = max(
        evidence,
        key=lambda item: (item[1], item[2], -item[0].key_mode),
    )
    is_outlier = (
        gap >= OVERALL_SIBLING_GAP_MIN
        and longest_run >= MIN_CONSECUTIVE_LOCAL_OUTLIERS
    )
    return TimingFamilyReview(
        "OUTLIER" if is_outlier else "CONSISTENT",
        candidate.difficulty,
        candidate.key_mode if is_outlier else None,
        gap,
        longest_run,
        section_indices,
    )
