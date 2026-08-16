"""Policy-free storage for generated chart candidates.

Selection and publication rules intentionally live elsewhere. This container
only keeps candidates with different trust levels from being mixed together.
"""

from collections.abc import Iterable
from typing import Generic, TypeVar

CandidateT = TypeVar("CandidateT")


class CandidateRepository(Generic[CandidateT]):
    __slots__ = (
        "_admitted",
        "_partial_sources",
        "_raw_rejected",
        "_safe_fallbacks",
    )

    def __init__(
        self,
        *,
        admitted: Iterable[CandidateT] = (),
        raw_rejected: Iterable[CandidateT] = (),
        safe_fallbacks: Iterable[CandidateT] = (),
        partial_sources: Iterable[CandidateT] = (),
    ) -> None:
        self._admitted = list(admitted)
        self._raw_rejected = list(raw_rejected)
        self._safe_fallbacks = list(safe_fallbacks)
        self._partial_sources = list(partial_sources)

    @property
    def admitted(self) -> tuple[CandidateT, ...]:
        return tuple(self._admitted)

    @property
    def raw_rejected(self) -> tuple[CandidateT, ...]:
        return tuple(self._raw_rejected)

    @property
    def safe_fallbacks(self) -> tuple[CandidateT, ...]:
        return tuple(self._safe_fallbacks)

    @property
    def playtest_candidates(self) -> tuple[CandidateT, ...]:
        """Candidates ordered from strongest to weakest publication trust."""
        return (*self._admitted, *self._raw_rejected, *self._safe_fallbacks)

    @property
    def partial_sources(self) -> tuple[CandidateT, ...]:
        return tuple(self._partial_sources)

    def admit(self, candidate: CandidateT) -> None:
        self._admitted.append(candidate)

    def reject(self, candidate: CandidateT) -> None:
        self._raw_rejected.append(candidate)

    def add_safe_fallback(self, candidate: CandidateT) -> None:
        self._safe_fallbacks.append(candidate)

    def remember_partial_source(self, candidate: CandidateT) -> None:
        self._partial_sources.append(candidate)
