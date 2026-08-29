"""Bind a confirmed intro region to one bounded decoder obligation.

The planner is deliberately identity-free: no song, artist, genre, BPM, key
mode, or difficulty is accepted.  It only promotes an already corroborated
audio/timing contract when the whole interval is inside the partial remap.
"""

from __future__ import annotations

import json
import re
from hashlib import sha256

from chart_worker.generation.partial_remap import PartialRemapWindow
from chart_worker.generation.required_gameplay_interval import (
    RequiredGameplayEvidenceClass,
    RequiredGameplayGroupType,
    RequiredGameplayIntervalDecision,
    RequiredGameplayIntervalMode,
    RequiredGameplayIntervalV1,
)
from chart_worker.validation.intro_region_contract import IntroRegionContract

INTRO_REQUIRED_GAMEPLAY_POLICY_VERSION = "intro-required-gameplay-policy-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _decline(reason: str) -> RequiredGameplayIntervalDecision:
    return RequiredGameplayIntervalDecision(interval=None, reason=reason)


def _evidence_digest(
    contract: IntroRegionContract,
    *,
    partial_window: PartialRemapWindow,
    timing_authority_digest: str,
) -> str:
    payload = {
        "policyVersion": INTRO_REQUIRED_GAMEPLAY_POLICY_VERSION,
        "introRegionContract": contract.to_report(),
        "introRegionContractSha256": contract.stable_sha256(),
        "timingAuthorityDigest": timing_authority_digest,
        "partialWindow": {
            "startMs": partial_window.start_ms,
            "endMs": partial_window.end_ms,
        },
        "minimumCompleteGroups": 1,
        "allowedGroupTypes": [
            RequiredGameplayGroupType.HOLD_START.value,
            RequiredGameplayGroupType.TAP.value,
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def plan_intro_required_gameplay_interval(
    contract: IntroRegionContract,
    *,
    partial_window: PartialRemapWindow,
    timing_authority_digest: str,
    mode: RequiredGameplayIntervalMode,
) -> RequiredGameplayIntervalDecision:
    """Authorize one generated TAP/HOLD start inside a confirmed intro region."""

    if type(contract) is not IntroRegionContract:
        raise TypeError("contract must be an IntroRegionContract")
    if type(partial_window) is not PartialRemapWindow:
        raise TypeError("partial_window must be a PartialRemapWindow")
    if (
        type(partial_window.start_ms) is not int
        or type(partial_window.end_ms) is not int
        or partial_window.start_ms < 0
        or partial_window.start_ms >= partial_window.end_ms
    ):
        raise ValueError("partial_window must satisfy 0 <= start < end")
    if (
        type(timing_authority_digest) is not str
        or _SHA256_PATTERN.fullmatch(timing_authority_digest) is None
    ):
        raise ValueError("timing_authority_digest must be a lowercase SHA-256 digest")
    if type(mode) is not RequiredGameplayIntervalMode:
        raise TypeError("mode must be RequiredGameplayIntervalMode")

    allowed = contract.allowed_first_row_ms
    if contract.status != "CONFIRMED" or allowed is None:
        return _decline("INTRO_REGION_NOT_CONFIRMED")
    start_ms, end_ms = allowed
    if start_ms >= end_ms:
        return _decline("INTRO_REGION_INVALID")
    if partial_window.start_ms > start_ms or partial_window.end_ms < end_ms:
        return _decline("INTRO_REGION_OUTSIDE_PARTIAL_WINDOW")

    allowed_group_types = (
        RequiredGameplayGroupType.TAP,
        RequiredGameplayGroupType.HOLD_START,
    )
    return RequiredGameplayIntervalDecision(
        interval=RequiredGameplayIntervalV1(
            start_ms=start_ms,
            end_ms=end_ms,
            minimum_complete_groups=1,
            allowed_group_types=allowed_group_types,
            evidence_class=RequiredGameplayEvidenceClass.INTRO_REGION_CORROBORATED,
            evidence_digest=_evidence_digest(
                contract,
                partial_window=partial_window,
                timing_authority_digest=timing_authority_digest,
            ),
            mode=mode,
        ),
        reason="INTRO_REGION_CORROBORATED",
    )
