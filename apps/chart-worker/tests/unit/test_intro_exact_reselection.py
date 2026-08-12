from chart_worker.generation.intro_exact_reselection import apply_intro_start_contract


def test_exact_intro_reselection_is_owned_outside_song_orchestrator():
    assert callable(apply_intro_start_contract)
