"""Policy-free storage for generated chart candidates.

Selection and publication rules intentionally live elsewhere. This container
only keeps candidates with different trust levels from being mixed together.
"""

from collections.abc import Iterable
from typing import Generic, TypeVar

CandidateT = TypeVar("CandidateT")


class CandidateRepository(Generic[CandidateT]):
    __slots__ = ("_admitted", "_partial_sources", "_raw_rejected")

    def __init__(
        self,
        *,
        admitted: Iterable[CandidateT] = (),
        raw_rejected: Iterable[CandidateT] = (),
        partial_sources: Iterable[CandidateT] = (),
    ) -> None:
        self._admitted = list(admitted)
        self._raw_rejected = list(raw_rejected)
        self._partial_sources = list(partial_sources)

    @property
    def admitted(self) -> tuple[CandidateT, ...]:
        return tuple(self._admitted)

    @property
    def raw_rejected(self) -> tuple[CandidateT, ...]:
        return tuple(self._raw_rejected)

    @property
    def partial_sources(self) -> tuple[CandidateT, ...]:
        return tuple(self._partial_sources)

    def admit(self, candidate: CandidateT) -> None:
        self._admitted.append(candidate)

    def reject(self, candidate: CandidateT) -> None:
        self._raw_rejected.append(candidate)

    def remember_partial_source(self, candidate: CandidateT) -> None:
        self._partial_sources.append(candidate)
