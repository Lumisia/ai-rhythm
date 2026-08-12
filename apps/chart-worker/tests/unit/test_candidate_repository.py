import pytest

from chart_worker.generation.candidate_repository import CandidateRepository


def test_candidate_repository_keeps_admitted_raw_and_partial_sources_separate():
    repository = CandidateRepository[str]()

    repository.admit("verified")
    repository.reject("raw")
    repository.remember_partial_source("partial")

    assert repository.admitted == ("verified",)
    assert repository.raw_rejected == ("raw",)
    assert repository.partial_sources == ("partial",)

    with pytest.raises(AttributeError):
        repository.admitted.append("bypass")


def test_candidate_repository_lists_are_not_shared_between_variants():
    first = CandidateRepository[str]()
    second = CandidateRepository[str]()

    first.admit("first-only")

    assert second.admitted == ()
