"""Relative difficulty ordering without absolute player labels."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from chart_worker.analysis.chart_profile import DifficultyProfile
from chart_worker.schema.types import DIFFICULTIES

MIN_ADJACENT_RATING_GAP = 0.30
"""라벨이 사실상 겹치는 인접 간격. 24곡 배치 실측 분포(중앙 1.12,
p10 0.42)에서 하위 이상치만 걸리도록 잡았다. 차단이 아니라 REVIEW 근거다."""


@dataclass(frozen=True, slots=True)
class DifficultyOrderReview:
    status: Literal["PASS", "RETRY"]
    ordered_ratings: tuple[tuple[str, float], ...]
    inverted_pairs: tuple[tuple[str, str], ...]
    ambiguous_pairs: tuple[tuple[str, str], ...]
    retry_difficulties: frozenset[str]
    narrow_pairs: tuple[tuple[str, str], ...] = ()

    def to_report(self) -> dict[str, object]:
        return {
            "status": self.status,
            "orderedRatings": dict(self.ordered_ratings),
            "invertedPairs": [list(pair) for pair in self.inverted_pairs],
            "ambiguousPairs": [list(pair) for pair in self.ambiguous_pairs],
            "narrowPairs": [list(pair) for pair in self.narrow_pairs],
            "retryDifficulties": sorted(
                self.retry_difficulties, key=DIFFICULTIES.index
            ),
        }


def review_difficulty_order(
    profiles: Mapping[str, DifficultyProfile],
) -> DifficultyOrderReview:
    """Review adjacent, within-key project ratings in label order.

    발행 가능한 후보가 없는 난이도는 빠질 수 있으므로 부분 집합을
    허용한다. 존재하는 라벨들 사이의 인접 쌍만 검사한다.
    """
    unknown = set(profiles).difference(DIFFICULTIES)
    if unknown:
        raise ValueError(f"unknown difficulties: {sorted(unknown)}")
    if not profiles:
        raise ValueError("profiles must contain at least one supported difficulty")

    ordered_ratings = tuple(
        (difficulty, profiles[difficulty].project_rating)
        for difficulty in DIFFICULTIES
        if difficulty in profiles
    )
    inverted: list[tuple[str, str]] = []
    ambiguous: list[tuple[str, str]] = []
    narrow: list[tuple[str, str]] = []
    for (easier, easier_rating), (harder, harder_rating) in pairwise(ordered_ratings):
        if harder_rating < easier_rating:
            inverted.append((easier, harder))
        elif harder_rating == easier_rating:
            ambiguous.append((easier, harder))
        elif harder_rating - easier_rating < MIN_ADJACENT_RATING_GAP:
            narrow.append((easier, harder))

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
        narrow_pairs=tuple(narrow),
    )
