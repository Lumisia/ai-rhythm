"""Relative difficulty ordering without absolute player labels."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from chart_worker.analysis.chart_profile import DifficultyProfile
from chart_worker.schema.types import DIFFICULTIES


@dataclass(frozen=True, slots=True)
class DifficultyOrderReview:
    status: Literal["PASS", "RETRY"]
    ordered_ratings: tuple[tuple[str, float], ...]
    inverted_pairs: tuple[tuple[str, str], ...]
    ambiguous_pairs: tuple[tuple[str, str], ...]
    retry_difficulties: frozenset[str]

    def to_report(self) -> dict[str, object]:
        return {
            "status": self.status,
            "orderedRatings": dict(self.ordered_ratings),
            "invertedPairs": [list(pair) for pair in self.inverted_pairs],
            "ambiguousPairs": [list(pair) for pair in self.ambiguous_pairs],
            "retryDifficulties": sorted(
                self.retry_difficulties, key=DIFFICULTIES.index
            ),
        }


def review_difficulty_order(
    profiles: Mapping[str, DifficultyProfile],
) -> DifficultyOrderReview:
    """Review only adjacent, within-key project ratings in label order."""
    if set(profiles) != set(DIFFICULTIES):
        raise ValueError("profiles must contain exactly the four supported difficulties")

    ordered_ratings = tuple(
        (difficulty, profiles[difficulty].project_rating)
        for difficulty in DIFFICULTIES
    )
    inverted: list[tuple[str, str]] = []
    ambiguous: list[tuple[str, str]] = []
    for (easier, easier_rating), (harder, harder_rating) in pairwise(ordered_ratings):
        if harder_rating < easier_rating:
            inverted.append((easier, harder))
        elif harder_rating == easier_rating:
            ambiguous.append((easier, harder))

    status: Literal["PASS", "RETRY"]
    if inverted:
        status = "RETRY"
        retry_difficulties = frozenset(
            difficulty for pair in inverted for difficulty in pair
        )
    else:
        status = "PASS"
        retry_difficulties = frozenset()
    return DifficultyOrderReview(
        status=status,
        ordered_ratings=ordered_ratings,
        inverted_pairs=tuple(inverted),
        ambiguous_pairs=tuple(ambiguous),
        retry_difficulties=retry_difficulties,
    )
