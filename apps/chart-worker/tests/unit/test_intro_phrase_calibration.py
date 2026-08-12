from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

from chart_worker.analysis.intro_phrase_calibration import (
    build_intro_phrase_review_queue,
    evaluate_intro_phrase_calibration,
)


def _load_calibration_script():
    script_path = Path(__file__).parents[2] / "scripts" / "calibrate_intro_phrase_family.py"
    spec = importlib.util.spec_from_file_location("calibrate_intro_phrase_family", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(
    *,
    batch: str,
    song_index: int,
    key_mode: int,
    audio: str,
    status: str,
    reason: str,
) -> dict[str, object]:
    suffix = f"{song_index}-{key_mode}"
    return {
        "batchStateSha256": batch,
        "songIndex": song_index,
        "sourceName": f"song-{song_index}.wav",
        "generationReportSha256": (f"{song_index:x}" * 64)[:64],
        "audioSha256": audio,
        "keyMode": key_mode,
        "review": {
            "status": status,
            "reason": reason,
            "hard": {"candidateId": f"hard-{suffix}"},
            "expert": {"candidateId": f"expert-{suffix}"},
        },
    }


def _label(
    row: dict[str, object],
    *,
    label_id: str,
    verdict: str,
    scope: str,
) -> dict[str, object]:
    review = row["review"]
    return {
        "labelId": label_id,
        "reviewerId": "human-reviewer",
        "createdAt": "2026-08-11T00:00:00Z",
        "batchStateSha256": row["batchStateSha256"],
        "songIndex": row["songIndex"],
        "sourceName": row["sourceName"],
        "generationReportSha256": row["generationReportSha256"],
        "audioSha256": row["audioSha256"],
        "keyMode": row["keyMode"],
        "hardCandidateId": review["hard"]["candidateId"],
        "expertCandidateId": review["expert"]["candidateId"],
        "humanVerdict": verdict,
        "confidence": "HIGH",
        "scope": scope,
        "comment": "direct play observation",
    }


def _fixture() -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    old_batch = "a" * 64
    recovery_batch = "b" * 64
    fog_audio = "1" * 64
    koe_audio = "2" * 64
    salt_audio = "3" * 64
    other_audio = "4" * 64
    old_rows = [
        _row(
            batch=old_batch,
            song_index=1,
            key_mode=4,
            audio=fog_audio,
            status="DEFECT",
            reason="ISOLATED_EXPERT_FIRST_ROW",
        ),
        _row(
            batch=old_batch,
            song_index=3,
            key_mode=4,
            audio=koe_audio,
            status="DEFECT",
            reason="ISOLATED_EXPERT_FIRST_ROW",
        ),
        _row(
            batch=old_batch,
            song_index=5,
            key_mode=4,
            audio=salt_audio,
            status="REVIEW",
            reason="EXPERT_EARLY_GHOST",
        ),
        _row(
            batch=old_batch,
            song_index=6,
            key_mode=4,
            audio=other_audio,
            status="PASS",
            reason="CONSISTENT",
        ),
    ]
    recovery_rows = [
        _row(
            batch=recovery_batch,
            song_index=1,
            key_mode=4,
            audio=fog_audio,
            status="REVIEW",
            reason="EXPERT_LATE_START",
        ),
        _row(
            batch=recovery_batch,
            song_index=2,
            key_mode=4,
            audio=koe_audio,
            status="PASS",
            reason="CONSISTENT",
        ),
    ]
    audits = [
        {
            "version": "intro-phrase-family-batch-audit-v1",
            "batchStateSha256": old_batch,
            "batchStartedAt": "2026-08-10T01:00:00Z",
            "songCount": 4,
            "pairCount": 4,
            "rows": old_rows,
        },
        {
            "version": "intro-phrase-family-batch-audit-v1",
            "batchStateSha256": recovery_batch,
            "batchStartedAt": "2026-08-10T03:00:00Z",
            "songCount": 2,
            "pairCount": 2,
            "rows": recovery_rows,
        },
    ]
    cohorts = {
        "version": "intro-phrase-calibration-cohorts-v1",
        "policySnapshot": {
            "version": "intro-phrase-family-v1",
            "evaluationState": "FROZEN_FOR_EXTERNAL_EVALUATION",
            "frozenAt": "2026-08-11T00:00:00Z",
        },
        "cohorts": [
            {
                "cohortId": "historical",
                "batchStateSha256": old_batch,
                "role": "DEVELOPMENT_DISCOVERY",
                "exposedToPolicyDesign": True,
            },
            {
                "cohortId": "recovery",
                "batchStateSha256": recovery_batch,
                "role": "RECOVERY_OUTCOME",
                "exposedToPolicyDesign": True,
            },
        ],
    }
    labels = {
        "version": "intro-phrase-human-label-set-v1",
        "labels": [
            _label(
                old_rows[0],
                label_id="old-fog-defect",
                verdict="DEFECT",
                scope="PAIR_QUALITY",
            ),
            _label(
                old_rows[1],
                label_id="old-koe-defect",
                verdict="DEFECT",
                scope="PAIR_QUALITY",
            ),
            _label(
                old_rows[2],
                label_id="old-salt-acceptable",
                verdict="ACCEPTABLE",
                scope="PAIR_QUALITY",
            ),
            _label(
                recovery_rows[0],
                label_id="new-fog-acceptable",
                verdict="ACCEPTABLE",
                scope="RECOVERY_OUTCOME",
            ),
            _label(
                recovery_rows[1],
                label_id="new-koe-acceptable",
                verdict="ACCEPTABLE",
                scope="RECOVERY_OUTCOME",
            ),
        ],
    }
    return audits, cohorts, labels


def test_separates_discovery_matrix_recovery_outcomes_and_unlabeled_pairs():
    audits, cohorts, labels = _fixture()

    result = evaluate_intro_phrase_calibration(audits, cohorts, labels)
    queue = build_intro_phrase_review_queue(audits, cohorts, labels)

    historical = next(item for item in result["cohorts"] if item["cohortId"] == "historical")
    recovery = next(item for item in result["cohorts"] if item["cohortId"] == "recovery")
    assert historical["matrix"] == {
        "truePositive": 2,
        "trueNegative": 1,
        "falsePositive": 0,
        "falseNegative": 0,
        "eligibleLabelCount": 3,
    }
    assert historical["unlabeledPairCount"] == 1
    assert recovery["labeledPairCount"] == 2
    assert recovery["unlabeledPairCount"] == 0
    assert result["recoveryOutcomes"] == {
        "eligibleLabelCount": 2,
        "accepted": 2,
        "defect": 0,
        "uncertainOrExcluded": 0,
    }
    assert result["promotionEligible"] is False
    assert "DISCOVERY_ONLY_NOT_GENERALIZATION" in result["promotionBlockers"]
    assert "NO_EXTERNAL_UNSEEN_COHORT" in result["promotionBlockers"]
    assert "EXTERNAL_ACCEPTANCE_CRITERIA_NOT_PREREGISTERED" in result["promotionBlockers"]
    assert len(queue["rows"]) == 1
    assert queue["rows"][0]["songIndex"] == 6


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("generationReportSha256", "f" * 64),
        ("audioSha256", "e" * 64),
        ("sourceName", "different-source.wav"),
        ("hardCandidateId", "different-hard-candidate"),
        ("expertCandidateId", "different-expert-candidate"),
    ],
)
def test_rejects_human_labels_not_bound_to_the_exact_audit_evidence(field, bad_value):
    audits, cohorts, labels = _fixture()
    labels["labels"][0][field] = bad_value

    with pytest.raises(ValueError, match=field):
        evaluate_intro_phrase_calibration(audits, cohorts, labels)


def test_rejects_duplicate_label_ids():
    audits, cohorts, labels = _fixture()
    labels["labels"].append(deepcopy(labels["labels"][0]))

    with pytest.raises(ValueError, match="duplicate labelId"):
        evaluate_intro_phrase_calibration(audits, cohorts, labels)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("humanVerdict", "MAYBE"),
        ("confidence", "CERTAIN"),
        ("scope", "THRESHOLD_TUNING"),
    ],
)
def test_rejects_unknown_human_label_enums(field, bad_value):
    audits, cohorts, labels = _fixture()
    labels["labels"][0][field] = bad_value

    with pytest.raises(ValueError, match=field):
        evaluate_intro_phrase_calibration(audits, cohorts, labels)


def test_rejects_unknown_cohort_role():
    audits, cohorts, labels = _fixture()
    cohorts["cohorts"][0]["role"] = "HOLDOUT"

    with pytest.raises(ValueError, match="cohort role"):
        evaluate_intro_phrase_calibration(audits, cohorts, labels)


@pytest.mark.parametrize(
    ("target", "bad_version", "message"),
    [
        ("audit", "intro-phrase-family-batch-audit-v0", "audit version"),
        ("cohorts", "intro-phrase-calibration-cohorts-v0", "cohort manifest version"),
        ("labels", "intro-phrase-human-label-set-v0", "label set version"),
    ],
)
def test_rejects_unknown_evidence_document_versions(target, bad_version, message):
    audits, cohorts, labels = _fixture()
    if target == "audit":
        audits[0]["version"] = bad_version
    elif target == "cohorts":
        cohorts["version"] = bad_version
    else:
        labels["version"] = bad_version

    with pytest.raises(ValueError, match=message):
        evaluate_intro_phrase_calibration(audits, cohorts, labels)


def test_rejects_audit_pair_count_that_does_not_match_rows():
    audits, cohorts, labels = _fixture()
    audits[0]["pairCount"] = 999

    with pytest.raises(ValueError, match="pairCount"):
        evaluate_intro_phrase_calibration(audits, cohorts, labels)


def test_rejects_audit_row_bound_to_a_different_batch():
    audits, cohorts, labels = _fixture()
    audits[0]["rows"][0]["batchStateSha256"] = "f" * 64

    with pytest.raises(ValueError, match="row batchStateSha256"):
        evaluate_intro_phrase_calibration(audits, cohorts, labels)


def test_rejects_missing_duplicate_or_unregistered_cohort_audits():
    audits, cohorts, labels = _fixture()
    cohorts["cohorts"][0]["batchStateSha256"] = "f" * 64
    with pytest.raises(ValueError, match="no matching audit"):
        evaluate_intro_phrase_calibration(audits, cohorts, labels)

    audits, cohorts, labels = _fixture()
    duplicate = deepcopy(cohorts["cohorts"][0])
    duplicate["cohortId"] = "historical-duplicate"
    cohorts["cohorts"].append(duplicate)
    with pytest.raises(ValueError, match="duplicate cohort batch"):
        evaluate_intro_phrase_calibration(audits, cohorts, labels)

    audits, cohorts, labels = _fixture()
    cohorts["cohorts"].pop()
    with pytest.raises(ValueError, match="audit is not registered"):
        evaluate_intro_phrase_calibration(audits, cohorts, labels)


def test_reports_conflicting_high_confidence_human_labels_without_voting():
    audits, cohorts, labels = _fixture()
    conflict = deepcopy(labels["labels"][2])
    conflict["labelId"] = "old-salt-conflicting-defect"
    conflict["humanVerdict"] = "DEFECT"
    labels["labels"].append(conflict)

    result = evaluate_intro_phrase_calibration(audits, cohorts, labels)
    historical = next(item for item in result["cohorts"] if item["cohortId"] == "historical")

    assert historical["matrix"]["eligibleLabelCount"] == 2
    assert result["conflicts"] == [
        {
            "batchStateSha256": "a" * 64,
            "songIndex": 5,
            "keyMode": 4,
            "labelIds": ["old-salt-acceptable", "old-salt-conflicting-defect"],
            "verdicts": ["ACCEPTABLE", "DEFECT"],
        }
    ]
    assert "HUMAN_LABEL_CONFLICT" in result["promotionBlockers"]


def test_rejects_external_unseen_cohort_that_reuses_a_development_audio():
    audits, cohorts, labels = _fixture()
    external_batch = "c" * 64
    external_row = _row(
        batch=external_batch,
        song_index=40,
        key_mode=4,
        audio="1" * 64,
        status="PASS",
        reason="CONSISTENT",
    )
    audits.append(
        {
            "version": "intro-phrase-family-batch-audit-v1",
            "batchStateSha256": external_batch,
            "batchStartedAt": "2026-08-12T00:00:00Z",
            "songCount": 1,
            "pairCount": 1,
            "rows": [external_row],
        }
    )
    cohorts["cohorts"].append(
        {
            "cohortId": "external",
            "batchStateSha256": external_batch,
            "role": "EXTERNAL_UNSEEN",
            "exposedToPolicyDesign": False,
        }
    )

    with pytest.raises(ValueError, match="EXTERNAL_UNSEEN audio leakage"):
        evaluate_intro_phrase_calibration(audits, cohorts, labels)


def test_external_unseen_requires_post_freeze_time_and_confirmed_group_isolation():
    audits, cohorts, labels = _fixture()
    external_batch = "c" * 64
    external_audit = {
        "version": "intro-phrase-family-batch-audit-v1",
        "batchStateSha256": external_batch,
        "batchStartedAt": "2026-08-10T23:59:59Z",
        "songCount": 1,
        "pairCount": 1,
        "rows": [
            _row(
                batch=external_batch,
                song_index=40,
                key_mode=4,
                audio="9" * 64,
                status="PASS",
                reason="CONSISTENT",
            )
        ],
    }
    external_cohort = {
        "cohortId": "external",
        "batchStateSha256": external_batch,
        "role": "EXTERNAL_UNSEEN",
        "exposedToPolicyDesign": False,
        "groupIsolationStatus": "CONFIRMED",
    }
    audits.append(external_audit)
    cohorts["cohorts"].append(external_cohort)

    with pytest.raises(ValueError, match="must start after policy freeze"):
        evaluate_intro_phrase_calibration(audits, cohorts, labels)

    external_audit["batchStartedAt"] = "2026-08-12T00:00:00Z"
    external_cohort.pop("groupIsolationStatus")
    with pytest.raises(ValueError, match="groupIsolationStatus=CONFIRMED"):
        evaluate_intro_phrase_calibration(audits, cohorts, labels)

    external_cohort["groupIsolationStatus"] = "CONFIRMED"
    external_cohort["exposedToPolicyDesign"] = True
    with pytest.raises(ValueError, match="must not be exposed to policy design"):
        evaluate_intro_phrase_calibration(audits, cohorts, labels)


def test_valid_external_unseen_cohort_is_kept_separate_from_discovery():
    audits, cohorts, labels = _fixture()
    external_batch = "c" * 64
    audits.append(
        {
            "version": "intro-phrase-family-batch-audit-v1",
            "batchStateSha256": external_batch,
            "batchStartedAt": "2026-08-12T00:00:00Z",
            "songCount": 1,
            "pairCount": 1,
            "rows": [
                _row(
                    batch=external_batch,
                    song_index=40,
                    key_mode=4,
                    audio="9" * 64,
                    status="PASS",
                    reason="CONSISTENT",
                )
            ],
        }
    )
    cohorts["cohorts"].append(
        {
            "cohortId": "external",
            "batchStateSha256": external_batch,
            "role": "EXTERNAL_UNSEEN",
            "exposedToPolicyDesign": False,
            "groupIsolationStatus": "CONFIRMED",
        }
    )

    result = evaluate_intro_phrase_calibration(audits, cohorts, labels)

    external = next(item for item in result["cohorts"] if item["cohortId"] == "external")
    assert external["role"] == "EXTERNAL_UNSEEN"
    assert external["pairCount"] == 1
    assert "NO_EXTERNAL_UNSEEN_COHORT" not in result["promotionBlockers"]


def test_cli_writes_deterministic_report_and_review_queue(tmp_path: Path):
    audits, cohorts, labels = _fixture()
    audit_paths: list[Path] = []
    for index, audit in enumerate(audits, start=1):
        path = tmp_path / f"audit-{index}.json"
        _write_json(path, audit)
        audit_paths.append(path)
    audits_by_batch = {audit["batchStateSha256"]: path for audit, path in zip(audits, audit_paths)}
    for cohort in cohorts["cohorts"]:
        cohort["auditSha256"] = _sha256(audits_by_batch[cohort["batchStateSha256"]])
    app_root = Path(__file__).parents[2]
    implementation_path = app_root / "src" / "chart_worker" / "analysis" / "intro_phrase_calibration.py"
    cohorts["policySnapshot"]["implementation"] = [
        {
            "path": "src/chart_worker/analysis/intro_phrase_calibration.py",
            "sha256": _sha256(implementation_path),
        }
    ]
    cohorts_path = tmp_path / "cohorts.json"
    labels_path = tmp_path / "labels.json"
    report_path = tmp_path / "report.json"
    report_second_path = tmp_path / "report-second.json"
    queue_path = tmp_path / "queue.json"
    _write_json(cohorts_path, cohorts)
    _write_json(labels_path, labels)
    script = _load_calibration_script()
    common = [
        *(str(path) for path in audit_paths),
        "--cohorts",
        str(cohorts_path),
        "--labels",
        str(labels_path),
    ]

    assert script.main([*common, "--output", str(report_path), "--queue-output", str(queue_path)]) == 0
    assert script.main([*common, "--output", str(report_second_path)]) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    assert report["cohorts"][0]["matrix"]["truePositive"] == 2
    assert report["inputEvidence"]["cohortManifestSha256"] == _sha256(cohorts_path)
    assert report["inputEvidence"]["labelSetSha256"] == _sha256(labels_path)
    assert len(queue["rows"]) == 1
    assert report_path.read_bytes() == report_second_path.read_bytes()


def test_cli_rejects_stale_audit_and_implementation_hash_bindings(tmp_path: Path):
    audits, cohorts, labels = _fixture()
    audit_paths: list[Path] = []
    for index, audit in enumerate(audits, start=1):
        path = tmp_path / f"audit-{index}.json"
        _write_json(path, audit)
        audit_paths.append(path)
    for cohort, audit_path in zip(cohorts["cohorts"], audit_paths):
        cohort["auditSha256"] = _sha256(audit_path)
    app_root = Path(__file__).parents[2]
    implementation_path = app_root / "src" / "chart_worker" / "analysis" / "intro_phrase_calibration.py"
    cohorts["policySnapshot"]["implementation"] = [
        {
            "path": "src/chart_worker/analysis/intro_phrase_calibration.py",
            "sha256": _sha256(implementation_path),
        }
    ]
    cohorts_path = tmp_path / "cohorts.json"
    labels_path = tmp_path / "labels.json"
    report_path = tmp_path / "report.json"
    _write_json(cohorts_path, cohorts)
    _write_json(labels_path, labels)
    script = _load_calibration_script()
    common = [
        *(str(path) for path in audit_paths),
        "--cohorts",
        str(cohorts_path),
        "--labels",
        str(labels_path),
        "--output",
        str(report_path),
    ]

    cohorts["cohorts"][0]["auditSha256"] = "0" * 64
    _write_json(cohorts_path, cohorts)
    with pytest.raises(ValueError, match="audit SHA-256"):
        script.main(common)

    cohorts["cohorts"][0]["auditSha256"] = _sha256(audit_paths[0])
    cohorts["policySnapshot"]["implementation"][0]["sha256"] = "0" * 64
    _write_json(cohorts_path, cohorts)
    with pytest.raises(ValueError, match="implementation SHA-256"):
        script.main(common)


def test_review_queue_prioritizes_defect_review_insufficient_then_pass():
    audits, cohorts, labels = _fixture()
    old_rows = audits[0]["rows"]
    old_rows.append(
        _row(
            batch="a" * 64,
            song_index=2,
            key_mode=6,
            audio="5" * 64,
            status="INSUFFICIENT",
            reason="INSUFFICIENT_ROWS",
        )
    )
    audits[0]["pairCount"] = 5
    labels["labels"] = []

    queue = build_intro_phrase_review_queue(audits, cohorts, labels)

    assert [row["review"]["status"] for row in queue["rows"]] == [
        "DEFECT",
        "DEFECT",
        "REVIEW",
        "INSUFFICIENT",
        "PASS",
    ]
