"""Local fail-closed Modal soft-budget reservation ledger.

The Starter credit is external billing state. This ledger is deliberately
pessimistic: a reservation is never refunded automatically, because a client
cannot prove that a timed-out or disconnected remote input consumed no compute.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODAL_MAX_INPUT_SECONDS = 10_800
MODAL_MAX_STARTUP_SECONDS = 1_800
MODAL_SCALEDOWN_WINDOW_SECONDS = 60
MODAL_MAX_RESERVATION_SECONDS = (
    MODAL_MAX_INPUT_SECONDS + MODAL_MAX_STARTUP_SECONDS + MODAL_SCALEDOWN_WINDOW_SECONDS
)
MODAL_SOFT_CAP_NANO_USD = 24_000_000_000
MODAL_L4_NANO_USD_PER_SECOND = 222_000
MODAL_CPU_4_CORE_NANO_USD_PER_SECOND = 52_400
MODAL_MEMORY_16_GIB_NANO_USD_PER_SECOND = 35_520
MODAL_TOTAL_NANO_USD_PER_SECOND = (
    MODAL_L4_NANO_USD_PER_SECOND
    + MODAL_CPU_4_CORE_NANO_USD_PER_SECOND
    + MODAL_MEMORY_16_GIB_NANO_USD_PER_SECOND
)
_LEDGER_VERSION = 1
_MAX_LEDGER_BYTES = 1_048_576


class ModalBudgetError(RuntimeError):
    """The local reservation cannot be proven safe under the configured cap."""


@dataclass(frozen=True, slots=True)
class ModalBudgetReservation:
    token: str
    amount_nano_usd: int
    projected_reserved_nano_usd: int
    replayed: bool


def estimate_l4_cost_nano_usd(seconds: int) -> int:
    if type(seconds) is not int or not 0 <= seconds <= MODAL_MAX_RESERVATION_SECONDS:
        raise ValueError(
            f"seconds must be an exact int from 0 through {MODAL_MAX_RESERVATION_SECONDS}"
        )
    return seconds * MODAL_TOTAL_NANO_USD_PER_SECOND


def _reservation_cost_nano_usd(worst_case_input_seconds: int) -> int:
    return estimate_l4_cost_nano_usd(
        worst_case_input_seconds
        + MODAL_MAX_STARTUP_SECONDS
        + MODAL_SCALEDOWN_WINDOW_SECONDS
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModalBudgetError(f"duplicate budget ledger key: {key}")
        result[key] = value
    return result


def _require_token(value: object) -> str:
    if type(value) is not str or not value or len(value) > 128:
        raise ModalBudgetError("reservation token must be a non-empty plain string <=128 chars")
    return value


def _empty_ledger() -> dict[str, object]:
    return {
        "version": _LEDGER_VERSION,
        "softCapNanoUsd": MODAL_SOFT_CAP_NANO_USD,
        "reservedNanoUsd": 0,
        "reservations": [],
    }


def _decode_ledger(raw: bytes) -> dict[str, object]:
    if len(raw) > _MAX_LEDGER_BYTES:
        raise ModalBudgetError("budget ledger exceeds the size limit")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except ModalBudgetError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ModalBudgetError("budget ledger is not bounded valid JSON") from error
    if type(value) is not dict or set(value) != {
        "version",
        "softCapNanoUsd",
        "reservedNanoUsd",
        "reservations",
    }:
        raise ModalBudgetError("budget ledger schema is invalid")
    if value["version"] != _LEDGER_VERSION:
        raise ModalBudgetError("budget ledger version is unsupported")
    if value["softCapNanoUsd"] != MODAL_SOFT_CAP_NANO_USD:
        raise ModalBudgetError("budget ledger soft cap differs from the code contract")
    reservations = value["reservations"]
    if type(reservations) is not list:
        raise ModalBudgetError("budget ledger reservations must be a list")
    observed_tokens: set[str] = set()
    computed_total = 0
    for record in reservations:
        if type(record) is not dict or set(record) != {
            "token",
            "worstCaseSeconds",
            "amountNanoUsd",
        }:
            raise ModalBudgetError("budget reservation schema is invalid")
        token = _require_token(record["token"])
        if token in observed_tokens:
            raise ModalBudgetError("budget ledger contains a duplicate reservation token")
        observed_tokens.add(token)
        worst_case_seconds = record["worstCaseSeconds"]
        if type(worst_case_seconds) is not int or not 1 <= worst_case_seconds <= MODAL_MAX_INPUT_SECONDS:
            raise ModalBudgetError("budget reservation duration is invalid")
        expected_amount = _reservation_cost_nano_usd(worst_case_seconds)
        if type(record["amountNanoUsd"]) is not int or record["amountNanoUsd"] != expected_amount:
            raise ModalBudgetError("budget reservation amount does not match its duration")
        computed_total += expected_amount
    reserved = value["reservedNanoUsd"]
    if type(reserved) is not int or reserved != computed_total:
        raise ModalBudgetError("budget ledger total does not match its reservations")
    if not 0 <= reserved <= MODAL_SOFT_CAP_NANO_USD:
        raise ModalBudgetError("budget ledger total exceeds its soft cap")
    return value


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def reserve_modal_budget(
    ledger_path: Path,
    *,
    token: str,
    worst_case_seconds: int,
) -> ModalBudgetReservation:
    if not isinstance(ledger_path, Path) or not ledger_path.is_absolute():
        raise ValueError("ledger_path must be an absolute Path")
    token = _require_token(token)
    if type(worst_case_seconds) is not int or not 1 <= worst_case_seconds <= MODAL_MAX_INPUT_SECONDS:
        raise ValueError(
            f"worst_case_seconds must be an exact int from 1 through {MODAL_MAX_INPUT_SECONDS}"
        )
    parent = ledger_path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ModalBudgetError("budget ledger parent must be an existing non-symlink directory")
    if ledger_path.is_symlink():
        raise ModalBudgetError("budget ledger must not be a symlink")
    lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    try:
        lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ModalBudgetError("budget ledger is locked; manual stale-lock review is required") from error
    try:
        with os.fdopen(lock_descriptor, "w", encoding="utf-8", closefd=True) as stream:
            stream.write(f"pid={os.getpid()} token={token}\n")
            stream.flush()
            os.fsync(stream.fileno())
        ledger = _decode_ledger(ledger_path.read_bytes()) if ledger_path.exists() else _empty_ledger()
        reservations = ledger["reservations"]
        if type(reservations) is not list:
            raise ModalBudgetError("budget ledger reservations must be a list")
        amount = _reservation_cost_nano_usd(worst_case_seconds)
        for record in reservations:
            if type(record) is not dict:
                raise ModalBudgetError("budget reservation schema is invalid")
            if record["token"] != token:
                continue
            if record["amountNanoUsd"] != amount:
                raise ModalBudgetError("reservation token already exists with a different amount")
            return ModalBudgetReservation(
                token=token,
                amount_nano_usd=amount,
                projected_reserved_nano_usd=ledger["reservedNanoUsd"],
                replayed=True,
            )
        projected = ledger["reservedNanoUsd"] + amount
        if projected > MODAL_SOFT_CAP_NANO_USD:
            raise ModalBudgetError("Modal soft cap would be exceeded by this reservation")
        reservations.append(
            {
                "token": token,
                "worstCaseSeconds": worst_case_seconds,
                "amountNanoUsd": amount,
            }
        )
        ledger["reservedNanoUsd"] = projected
        _atomic_write(ledger_path, ledger)
        return ModalBudgetReservation(
            token=token,
            amount_nano_usd=amount,
            projected_reserved_nano_usd=projected,
            replayed=False,
        )
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
