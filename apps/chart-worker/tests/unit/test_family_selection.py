from types import SimpleNamespace

from chart_worker.generation.family_selection import family_score


def _candidate(
    *, provenance: str = "PRIMARY", intro_anchor_covered: bool | None = True
) -> SimpleNamespace:
    return SimpleNamespace(
        provenance=provenance,
        intro_anchor_covered=intro_anchor_covered,
        attempt=1,
        seed=7,
    )


def test_family_score_keeps_coverage_ahead_of_raw_candidate_penalty() -> None:
    complete_with_raw = (
        _candidate(),
        _candidate(),
        _candidate(),
        _candidate(provenance="RAW_UNVERIFIED"),
    )
    missing_one = (_candidate(), _candidate(), _candidate(), None)

    assert family_score(complete_with_raw, None) < family_score(missing_one, None)


def test_family_score_penalizes_confirmed_intro_miss_after_provenance() -> None:
    covered = (_candidate(),)
    missed = (_candidate(intro_anchor_covered=False),)

    assert family_score(covered, None) < family_score(missed, None)
