import hashlib
import json

from chart_worker.generation.resnap_diagnostics import read_resnap_diagnostics


def test_reads_collision_only_sidecar(tmp_path):
    osu_path = tmp_path / "beatmap.osu"
    osu_path.write_text("osu", encoding="utf-8")
    osu_path.with_suffix(".resnap.json").write_text(
        json.dumps(
            {
                "version": "resnap-collisions-v2-preserve-raw",
                "seed": 19,
                "collisions": [
                    {
                        "lane": 2,
                        "noteKind": "HOLD",
                        "preTimeMs": 129_420,
                        "postTimeMs": 128_847,
                        "snapDivisor": 4,
                        "reason": "SNAP_TARGET_CONFLICT_PRESERVED",
                    },
                    {
                        "lane": 2,
                        "noteKind": "TAP",
                        "preTimeMs": 129_010,
                        "postTimeMs": 128_847,
                        "snapDivisor": 4,
                        "reason": "RAW_TIME_COLLISION_PRESERVED",
                    },
                ],
                "maniaObjects": [],
                "duplicates": [],
            }
        ),
        encoding="utf-8",
    )

    diagnostics = read_resnap_diagnostics(osu_path)

    assert diagnostics.status == "OBSERVED"
    assert diagnostics.error is None
    assert diagnostics.collisions[0].to_report() == {
        "seed": 19,
        "lane": 2,
        "noteKind": "HOLD",
        "preTimeMs": 129_420,
        "postTimeMs": 128_847,
        "snapDivisor": 4,
        "reason": "SNAP_TARGET_CONFLICT_PRESERVED",
    }


def test_valid_empty_sidecar_reports_no_collisions(tmp_path):
    osu_path = tmp_path / "beatmap.osu"
    osu_path.with_suffix(".resnap.json").write_text(
        json.dumps(
            {
                "version": "resnap-collisions-v2-preserve-raw",
                "seed": 7,
                "collisions": [],
            }
        ),
        encoding="utf-8",
    )

    diagnostics = read_resnap_diagnostics(osu_path)

    assert diagnostics.status == "NO_COLLISIONS"
    assert diagnostics.collisions == ()


def test_missing_sidecar_is_non_blocking_and_unobserved(tmp_path):
    diagnostics = read_resnap_diagnostics(tmp_path / "beatmap.osu")

    assert diagnostics.status == "UNOBSERVED"
    assert diagnostics.collisions == ()
    assert diagnostics.error is None


def test_malformed_sidecar_is_non_blocking_and_invalid(tmp_path):
    osu_path = tmp_path / "beatmap.osu"
    osu_path.with_suffix(".resnap.json").write_text("not-json", encoding="utf-8")

    diagnostics = read_resnap_diagnostics(osu_path)

    assert diagnostics.status == "INVALID"
    assert diagnostics.collisions == ()
    assert diagnostics.error


def test_unsnapped_member_of_a_collision_allows_zero_divisor(tmp_path):
    osu_path = tmp_path / "beatmap.osu"
    osu_path.with_suffix(".resnap.json").write_text(
        json.dumps(
            {
                "version": "resnap-collisions-v2-preserve-raw",
                "seed": 5,
                "collisions": [
                    {
                        "lane": 0,
                        "noteKind": "TAP",
                        "preTimeMs": 125,
                        "postTimeMs": 125,
                        "snapDivisor": 0,
                        "reason": "RAW_TIME_COLLISION_PRESERVED",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    diagnostics = read_resnap_diagnostics(osu_path)

    assert diagnostics.status == "OBSERVED"
    assert diagnostics.collisions[0].snap_divisor == 0


def test_v2_collision_requires_a_known_preservation_reason(tmp_path):
    osu_path = tmp_path / "beatmap.osu"
    osu_path.with_suffix(".resnap.json").write_text(
        json.dumps(
            {
                "version": "resnap-collisions-v2-preserve-raw",
                "seed": 5,
                "collisions": [
                    {
                        "lane": 0,
                        "noteKind": "TAP",
                        "preTimeMs": 110,
                        "postTimeMs": 125,
                        "snapDivisor": 4,
                        "reason": "UNKNOWN",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    diagnostics = read_resnap_diagnostics(osu_path)

    assert diagnostics.status == "INVALID"
    assert diagnostics.error == "unsupported collision reason"


def test_reads_v3_hold_pair_restoration_sidecar(tmp_path):
    osu_path = tmp_path / "beatmap.osu"
    osu_path.with_suffix(".resnap.json").write_text(
        json.dumps(
            {
                "version": "mania-resnap-v3-hold-pairs",
                "seed": 31,
                "collisions": [
                    {
                        "lane": 0,
                        "noteKind": "HOLD_PAIR",
                        "rawStartMs": 110,
                        "rawEndMs": 260,
                        "proposedStartMs": 125,
                        "proposedEndMs": 375,
                        "reason": "HOLD_PAIR_RAW_RESTORED",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    diagnostics = read_resnap_diagnostics(osu_path)

    assert diagnostics.status == "OBSERVED"
    assert diagnostics.version == "mania-resnap-v3-hold-pairs"
    assert diagnostics.collisions[0].post_time_ms is None
    assert diagnostics.collisions[0].to_report() == {
        "seed": 31,
        "lane": 0,
        "noteKind": "HOLD_PAIR",
        "rawStartMs": 110,
        "rawEndMs": 260,
        "proposedStartMs": 125,
        "proposedEndMs": 375,
        "reason": "HOLD_PAIR_RAW_RESTORED",
    }


def test_reads_start_boundary_restoration_diagnostics(tmp_path):
    osu_path = tmp_path / "beatmap.osu"
    osu_path.write_bytes(b"boundary-restored-osu")
    osu_path.with_suffix(".resnap.json").write_text(
        json.dumps(
            {
                "version": "mania-origin-v2-osu-bound",
                "osuSha256": hashlib.sha256(b"boundary-restored-osu").hexdigest(),
                "seed": 31,
                "collisions": [
                    {
                        "lane": 0,
                        "noteKind": "TAP",
                        "preTimeMs": 110,
                        "postTimeMs": 125,
                        "snapDivisor": 4,
                        "reason": "START_BOUNDARY_RAW_RESTORED",
                    },
                    {
                        "lane": 2,
                        "noteKind": "HOLD_PAIR",
                        "rawStartMs": 110,
                        "rawEndMs": 260,
                        "proposedStartMs": 125,
                        "proposedEndMs": 250,
                        "reason": "HOLD_PAIR_START_BOUNDARY_RAW_RESTORED",
                    },
                ],
                "maniaObjects": [],
                "duplicates": [],
            }
        ),
        encoding="utf-8",
    )

    diagnostics = read_resnap_diagnostics(osu_path)

    assert diagnostics.status == "OBSERVED"
    assert [item.reason for item in diagnostics.collisions] == [
        "START_BOUNDARY_RAW_RESTORED",
        "HOLD_PAIR_START_BOUNDARY_RAW_RESTORED",
    ]


def test_reads_v3_orphan_hold_as_invalid_raw_pair(tmp_path):
    osu_path = tmp_path / "beatmap.osu"
    osu_path.with_suffix(".resnap.json").write_text(
        json.dumps(
            {
                "version": "mania-resnap-v3-hold-pairs",
                "seed": 7,
                "collisions": [
                    {
                        "lane": 2,
                        "noteKind": "HOLD_PAIR",
                        "rawStartMs": 510,
                        "rawEndMs": None,
                        "proposedStartMs": 625,
                        "proposedEndMs": None,
                        "reason": "HOLD_PAIR_INVALID_RAW",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    diagnostics = read_resnap_diagnostics(osu_path)

    assert diagnostics.status == "OBSERVED"
    assert diagnostics.collisions[0].raw_start_ms == 510
    assert diagnostics.collisions[0].raw_end_ms is None


def test_reads_lane_order_restoration_diagnostic_without_discarding_sidecar(
    tmp_path,
):
    osu_path = tmp_path / "chart.osu"
    osu_path.write_bytes(b"current-osu")
    osu_path.with_suffix(".resnap.json").write_text(
        json.dumps(
            {
                "version": "mania-origin-v2-osu-bound",
                "osuSha256": hashlib.sha256(b"current-osu").hexdigest(),
                "seed": 19,
                "collisions": [
                    {
                        "lane": 3,
                        "previousObjectId": 848,
                        "currentObjectId": 851,
                        "proposedPreviousEndMs": 145_138,
                        "proposedCurrentStartMs": 144_950,
                        "rawPreviousEndMs": 145_022,
                        "rawCurrentStartMs": 145_122,
                        "reason": "LANE_ORDER_RAW_RESTORED",
                    }
                ],
                "maniaObjects": [],
                "duplicates": [],
            }
        ),
        encoding="utf-8",
    )

    diagnostics = read_resnap_diagnostics(osu_path)

    assert diagnostics.status == "OBSERVED"
    assert diagnostics.error is None
    assert diagnostics.collisions[0].to_report() == {
        "seed": 19,
        "lane": 3,
        "previousObjectId": 848,
        "currentObjectId": 851,
        "proposedPreviousEndMs": 145_138,
        "proposedCurrentStartMs": 144_950,
        "rawPreviousEndMs": 145_022,
        "rawCurrentStartMs": 145_122,
        "reason": "LANE_ORDER_RAW_RESTORED",
    }


def test_v4_reads_canonical_mania_objects_and_origins(tmp_path):
    osu_path = tmp_path / "chart.osu"
    osu_path.with_suffix(".resnap.json").write_text(
        json.dumps(
            {
                "version": "mania-origin-v1-canonical-hold-ir",
                "seed": 19,
                "collisions": [],
                "maniaObjects": [
                    {
                        "objectId": 0,
                        "lane": 2,
                        "kind": "HOLD",
                        "startTimeMs": 100,
                        "endTimeMs": 500,
                        "startGroupId": 3,
                        "endGroupId": 9,
                        "startOrigins": [
                            {
                                "kind": "GENERATED",
                                "sourceWindowId": 4,
                                "sourceTokenIndex": 7,
                                "referenceEventIndex": None,
                            }
                        ],
                        "endOrigins": [
                            {
                                "kind": "GENERATED",
                                "sourceWindowId": 5,
                                "sourceTokenIndex": 2,
                                "referenceEventIndex": None,
                            }
                        ],
                    }
                ],
                "duplicates": [
                    {
                        "keptGroupId": 3,
                        "droppedGroupId": 4,
                        "reason": "EXACT_CROSS_WINDOW_DUPLICATE",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    diagnostics = read_resnap_diagnostics(osu_path)

    assert diagnostics.status == "NO_COLLISIONS"
    assert diagnostics.version == "mania-origin-v1-canonical-hold-ir"
    assert diagnostics.mania_objects[0].kind == "HOLD"
    assert diagnostics.mania_objects[0].start_origins[0].source_window_id == 4
    assert diagnostics.duplicates[0].reason == "EXACT_CROSS_WINDOW_DUPLICATE"


def test_bound_sidecar_rejects_an_osu_sha_mismatch(tmp_path):
    osu_path = tmp_path / "chart.osu"
    osu_path.write_bytes(b"current-osu")
    osu_path.with_suffix(".resnap.json").write_text(
        json.dumps(
            {
                "version": "mania-origin-v2-osu-bound",
                "osuSha256": hashlib.sha256(b"stale-osu").hexdigest(),
                "seed": 19,
                "collisions": [],
                "maniaObjects": [],
                "duplicates": [],
            }
        ),
        encoding="utf-8",
    )

    diagnostics = read_resnap_diagnostics(osu_path)

    assert diagnostics.status == "INVALID"
    assert "SHA-256" in (diagnostics.error or "")


def test_v4_rejects_non_positive_hold_duration(tmp_path):
    osu_path = tmp_path / "chart.osu"
    osu_path.with_suffix(".resnap.json").write_text(
        json.dumps(
            {
                "version": "mania-origin-v1-canonical-hold-ir",
                "seed": 19,
                "collisions": [],
                "maniaObjects": [
                    {
                        "objectId": 0,
                        "lane": 0,
                        "kind": "HOLD",
                        "startTimeMs": 500,
                        "endTimeMs": 500,
                        "startGroupId": 1,
                        "endGroupId": 2,
                        "startOrigins": [],
                        "endOrigins": [],
                    }
                ],
                "duplicates": [],
            }
        ),
        encoding="utf-8",
    )

    assert read_resnap_diagnostics(osu_path).status == "INVALID"
