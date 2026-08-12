from chart_worker.generation.intro_recovery import execute_intro_retry


def test_intro_retry_execution_is_owned_by_intro_recovery_module():
    assert callable(execute_intro_retry)
