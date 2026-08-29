from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from chart_worker.generation.generation_origin_diagnostics import (
    GENERATION_ORIGIN_DIAGNOSTICS_VERSION,
    GenerationOriginDiagnosticsError,
    read_generation_origin_diagnostics,
)
from chart_worker.generation.required_gameplay_interval import (
    RequiredGameplayEvidenceClass,
    RequiredGameplayGroupType,
    RequiredGameplayIntervalMode,
    RequiredGameplayIntervalV1,
)
from chart_worker.generation.resnap_diagnostics import (
    ManiaEventOrigin,
    ManiaObjectDiagnostic,
    ResnapDiagnostics,
)

_EVIDENCE_SHA = "a" * 64
_INVOCATION_SHA = "b" * 64


def _interval(
    mode: RequiredGameplayIntervalMode = RequiredGameplayIntervalMode.OBSERVE,
) -> RequiredGameplayIntervalV1:
    return RequiredGameplayIntervalV1(
        start_ms=430,
        end_ms=570,
        minimum_complete_groups=1,
        allowed_group_types=(
            RequiredGameplayGroupType.TAP,
            RequiredGameplayGroupType.HOLD_START,
        ),
        evidence_class=RequiredGameplayEvidenceClass.BROADBAND_ATTACK,
        evidence_digest=_EVIDENCE_SHA,
        mode=mode,
    )


def _origin(kind: str) -> ManiaEventOrigin:
    if kind == "GENERATED":
        return ManiaEventOrigin("GENERATED", 0, 7, None)
    return ManiaEventOrigin("REFERENCE", None, None, 3)


def _object(
    object_id: int,
    *,
    time_ms: int,
    origin: str,
) -> ManiaObjectDiagnostic:
    return ManiaObjectDiagnostic(
        object_id=object_id,
        lane=object_id,
        kind="TAP",
        start_time_ms=time_ms,
        end_time_ms=None,
        start_group_id=object_id,
        end_group_id=None,
        start_origins=(_origin(origin),),
        end_origins=(),
    )


def _resnap(*objects: ManiaObjectDiagnostic) -> ResnapDiagnostics:
    return ResnapDiagnostics(status="NO_COLLISIONS", mania_objects=objects)


def _count(total: int, interval: int) -> dict[str, int]:
    return {
        "totalGeneratedCompleteGroups": total,
        "intervalGeneratedCompleteGroups": interval,
        "tapGroups": interval,
        "holdStartGroups": 0,
    }


def _payload(
    osu_path: Path,
    *,
    counts: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1),
    first_loss_stage: str | None = None,
) -> dict[str, object]:
    stage_names = (
        "decoder",
        "windowMerge",
        "canonical",
        "resnap",
        "finalSerialization",
    )
    return {
        "version": GENERATION_ORIGIN_DIAGNOSTICS_VERSION,
        "output": {
            "fileName": osu_path.name,
            "size": osu_path.stat().st_size,
            "sha256": hashlib.sha256(osu_path.read_bytes()).hexdigest(),
        },
        "evidenceDigest": _EVIDENCE_SHA,
        "invocationDigest": _INVOCATION_SHA,
        "requiredInterval": {
            "startMs": 430,
            "endMs": 570,
            "minimumCompleteGroups": 1,
            "allowedGroupTypes": ["HOLD_START", "TAP"],
            "evidenceClass": "BROADBAND_ATTACK",
            "mode": "OBSERVE",
        },
        "stages": {
            name: _count(value, value)
            for name, value in zip(stage_names, counts, strict=True)
        },
        "firstLossStage": first_loss_stage,
    }


def _write(
    tmp_path: Path,
    *,
    counts: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1),
    first_loss_stage: str | None = None,
) -> tuple[Path, dict[str, object]]:
    osu_path = tmp_path / "out.osu"
    osu_path.write_text("osu file format v14\n", encoding="utf-8")
    payload = _payload(
        osu_path,
        counts=counts,
        first_loss_stage=first_loss_stage,
    )
    osu_path.with_suffix(".origin.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return osu_path, payload


def _read(osu_path: Path, resnap: ResnapDiagnostics):
    return read_generation_origin_diagnostics(
        osu_path,
        interval=_interval(),
        expected_invocation_digest=_INVOCATION_SHA,
        resnap_diagnostics=resnap,
    )


def test_reads_a_hash_bound_no_loss_observation(tmp_path: Path):
    osu_path, _ = _write(tmp_path)

    diagnostics = _read(
        osu_path,
        _resnap(_object(0, time_ms=500, origin="GENERATED")),
    )

    assert diagnostics.first_loss_stage is None
    assert diagnostics.decoder.interval_generated_complete_groups == 1
    assert diagnostics.final_serialization.interval_generated_complete_groups == 1


def test_decoder_zero_is_not_mislabeled_as_a_later_loss(tmp_path: Path):
    osu_path, _ = _write(tmp_path, counts=(0, 0, 0, 0, 0))

    diagnostics = _read(
        osu_path,
        _resnap(_object(0, time_ms=500, origin="REFERENCE")),
    )

    assert diagnostics.first_loss_stage is None
    assert diagnostics.decoder.interval_generated_complete_groups == 0


def test_shadow_enforcement_rejects_zero_final_generated_groups(tmp_path: Path):
    osu_path, payload = _write(tmp_path, counts=(0, 0, 0, 0, 0))
    payload["requiredInterval"]["mode"] = "SHADOW_ENFORCE"
    osu_path.with_suffix(".origin.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(GenerationOriginDiagnosticsError, match="minimum"):
        read_generation_origin_diagnostics(
            osu_path,
            interval=_interval(RequiredGameplayIntervalMode.SHADOW_ENFORCE),
            expected_invocation_digest=_INVOCATION_SHA,
            resnap_diagnostics=_resnap(
                _object(0, time_ms=500, origin="REFERENCE")
            ),
        )


@pytest.mark.parametrize(
    ("counts", "first_loss_stage"),
    [
        ((2, 2, 1, 1, 1), "canonical"),
        ((2, 2, 2, 1, 1), "resnap"),
    ],
)
def test_reports_the_first_stage_where_generated_interval_count_drops(
    tmp_path: Path,
    counts: tuple[int, int, int, int, int],
    first_loss_stage: str,
):
    osu_path, _ = _write(
        tmp_path,
        counts=counts,
        first_loss_stage=first_loss_stage,
    )

    diagnostics = _read(
        osu_path,
        _resnap(_object(0, time_ms=500, origin="GENERATED")),
    )

    assert diagnostics.first_loss_stage == first_loss_stage


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(version="unknown"), "version"),
        (
            lambda payload: payload.update(evidenceDigest="c" * 64),
            "evidence digest",
        ),
        (
            lambda payload: payload.update(invocationDigest="c" * 64),
            "invocation digest",
        ),
        (
            lambda payload: payload["output"].update(fileName="../out.osu"),
            "fileName",
        ),
        (
            lambda payload: payload["output"].update(sha256="c" * 64),
            "output.*SHA-256",
        ),
        (
            lambda payload: payload["stages"]["decoder"].update(
                intervalGeneratedCompleteGroups=True
            ),
            "integer",
        ),
        (
            lambda payload: payload["stages"]["decoder"].update(
                totalGeneratedCompleteGroups=-1
            ),
            "non-negative",
        ),
        (
            lambda payload: payload["stages"]["decoder"].update(
                intervalGeneratedCompleteGroups=2
            ),
            "interval.*total",
        ),
        (
            lambda payload: payload["stages"]["decoder"].update(tapGroups=0),
            "group-type counts",
        ),
        (
            lambda payload: payload.update(firstLossStage="unknown"),
            "firstLossStage",
        ),
    ],
)
def test_rejects_malformed_or_unbound_sidecars(tmp_path: Path, mutate, message):
    osu_path, payload = _write(tmp_path)
    mutated = deepcopy(payload)
    mutate(mutated)
    osu_path.with_suffix(".origin.json").write_text(
        json.dumps(mutated),
        encoding="utf-8",
    )

    with pytest.raises(GenerationOriginDiagnosticsError, match=message):
        _read(osu_path, _resnap(_object(0, time_ms=500, origin="GENERATED")))


def test_rejects_a_stage_count_increase(tmp_path: Path):
    osu_path, _ = _write(
        tmp_path,
        counts=(1, 1, 2, 2, 2),
    )

    with pytest.raises(GenerationOriginDiagnosticsError, match="must not increase"):
        _read(
            osu_path,
            _resnap(
                _object(0, time_ms=500, origin="GENERATED"),
                _object(1, time_ms=520, origin="GENERATED"),
            ),
        )


def test_rejects_a_first_loss_stage_that_disagrees_with_counts(tmp_path: Path):
    osu_path, _ = _write(
        tmp_path,
        counts=(2, 2, 1, 1, 1),
        first_loss_stage="resnap",
    )

    with pytest.raises(GenerationOriginDiagnosticsError, match="firstLossStage"):
        _read(osu_path, _resnap(_object(0, time_ms=500, origin="GENERATED")))


def test_rejects_final_generated_count_that_disagrees_with_resnap_origins(
    tmp_path: Path,
):
    osu_path, _ = _write(tmp_path)

    with pytest.raises(GenerationOriginDiagnosticsError, match="resnap origin"):
        _read(osu_path, _resnap(_object(0, time_ms=500, origin="REFERENCE")))


def test_reference_and_out_of_interval_generated_objects_do_not_satisfy_final_count(
    tmp_path: Path,
):
    osu_path, _ = _write(tmp_path, counts=(0, 0, 0, 0, 0))

    diagnostics = _read(
        osu_path,
        _resnap(
            _object(0, time_ms=500, origin="REFERENCE"),
            _object(1, time_ms=1_000, origin="GENERATED"),
        ),
    )

    assert diagnostics.final_serialization.interval_generated_complete_groups == 0
