import json
from pathlib import Path

import pytest

from chart_worker import modal_budget
from chart_worker.modal_budget import (
    MODAL_MAX_INPUT_SECONDS,
    MODAL_MAX_RESERVATION_SECONDS,
    ModalBudgetError,
    estimate_l4_cost_nano_usd,
    reserve_modal_budget,
)


def test_exact_l4_cpu_memory_max_input_cost():
    assert MODAL_MAX_INPUT_SECONDS == 10_800
    assert estimate_l4_cost_nano_usd(MODAL_MAX_INPUT_SECONDS) == 3_347_136_000
    assert MODAL_MAX_RESERVATION_SECONDS == 12_660


def test_soft_cap_allows_six_lifecycle_reservations_and_blocks_seventh(tmp_path: Path):
    ledger = tmp_path / "modal-budget-v1.json"

    reservations = [
        reserve_modal_budget(
            ledger,
            token=f"job-{index}",
            worst_case_seconds=MODAL_MAX_INPUT_SECONDS,
        )
        for index in range(6)
    ]

    assert reservations[-1].projected_reserved_nano_usd == 23_541_523_200
    with pytest.raises(ModalBudgetError, match="soft cap"):
        reserve_modal_budget(
            ledger,
            token="job-6",
            worst_case_seconds=MODAL_MAX_INPUT_SECONDS,
        )


def test_same_token_is_idempotent_but_conflicting_reservation_fails(tmp_path: Path):
    ledger = tmp_path / "modal-budget-v1.json"
    first = reserve_modal_budget(ledger, token="same", worst_case_seconds=600)
    replay = reserve_modal_budget(ledger, token="same", worst_case_seconds=600)

    assert replay.token == first.token
    assert replay.amount_nano_usd == first.amount_nano_usd
    assert replay.projected_reserved_nano_usd == first.projected_reserved_nano_usd
    assert first.replayed is False
    assert replay.replayed is True
    with pytest.raises(ModalBudgetError, match="different amount"):
        reserve_modal_budget(ledger, token="same", worst_case_seconds=601)


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        json.dumps({"version": 2}).encode(),
        json.dumps(
            {
                "version": 1,
                "softCapNanoUsd": 2_400_000_000,
                "reservedNanoUsd": 1,
                "reservations": [],
            }
        ).encode(),
    ],
)
def test_corrupt_or_inconsistent_ledger_fails_closed(tmp_path: Path, payload: bytes):
    ledger = tmp_path / "modal-budget-v1.json"
    ledger.write_bytes(payload)

    with pytest.raises(ModalBudgetError):
        reserve_modal_budget(ledger, token="new", worst_case_seconds=1)


def test_existing_lock_fails_closed_without_changing_ledger(tmp_path: Path):
    ledger = tmp_path / "modal-budget-v1.json"
    lock = ledger.with_suffix(ledger.suffix + ".lock")
    lock.write_text("owner", encoding="utf-8")

    with pytest.raises(ModalBudgetError, match="locked"):
        reserve_modal_budget(ledger, token="new", worst_case_seconds=1)
    assert not ledger.exists()


def test_internal_ledger_shape_check_does_not_depend_on_assert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ledger = tmp_path / "modal-budget-v1.json"
    ledger.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        modal_budget,
        "_decode_ledger",
        lambda _raw: {
            "version": 1,
            "softCapNanoUsd": 24_000_000_000,
            "reservedNanoUsd": 0,
            "reservations": "not-a-list",
        },
    )

    with pytest.raises(ModalBudgetError, match="reservations must be a list"):
        reserve_modal_budget(ledger, token="new", worst_case_seconds=1)
