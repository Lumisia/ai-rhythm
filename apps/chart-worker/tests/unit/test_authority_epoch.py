from chart_worker.stages.authority_epoch import AuthorityEpochRecord


def test_authority_epoch_report_preserves_selection_and_escalation_evidence():
    record = AuthorityEpochRecord(
        epoch=1,
        authority_sha256="standard-sha",
        mode="STANDARD",
        status="REJECTED_MAP_TIMING_FEEDBACK",
        escalation={
            "failureFamily": "DUPLICATE_NOTE",
            "seeds": [7, 19],
        },
    )

    assert record.to_report() == {
        "epoch": 1,
        "authoritySha256": "standard-sha",
        "mode": "STANDARD",
        "status": "REJECTED_MAP_TIMING_FEEDBACK",
        "escalation": {
            "failureFamily": "DUPLICATE_NOTE",
            "seeds": [7, 19],
        },
    }
