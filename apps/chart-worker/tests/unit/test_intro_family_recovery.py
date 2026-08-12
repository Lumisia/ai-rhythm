from chart_worker.generation.intro_family_recovery import (
    apply_intro_phrase_family_recovery,
    intro_candidate_view,
    intro_phrase_family_reviews,
)


def test_intro_family_policy_is_owned_outside_song_orchestrator():
    assert callable(apply_intro_phrase_family_recovery)
    assert callable(intro_candidate_view)
    assert callable(intro_phrase_family_reviews)
