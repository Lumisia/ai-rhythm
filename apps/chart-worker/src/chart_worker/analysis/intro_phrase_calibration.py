"""Offline evidence evaluation for HARD/EXPERT intro phrase reviews."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

COHORT_ROLES = {
    "DEVELOPMENT_DISCOVERY",
    "RETROSPECTIVE_CHECK",
    "RECOVERY_OUTCOME",
    "EXTERNAL_UNSEEN",
}
HUMAN_VERDICTS = {"DEFECT", "ACCEPTABLE", "UNCERTAIN"}
HUMAN_CONFIDENCES = {"HIGH", "MEDIUM", "LOW"}
LABEL_SCOPES = {"PAIR_QUALITY", "RECOVERY_OUTCOME"}
AUDIT_VERSION = "intro-phrase-family-batch-audit-v1"
COHORT_MANIFEST_VERSION = "intro-phrase-calibration-cohorts-v1"
LABEL_SET_VERSION = "intro-phrase-human-label-set-v1"


def _pair_key(value: dict[str, object]) -> tuple[str, int, int]:
    return (
        str(value["batchStateSha256"]),
        int(value["songIndex"]),
        int(value["keyMode"]),
    )


def _audits_by_batch(
    audits: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for audit in audits:
        batch_sha = str(audit["batchStateSha256"])
        if batch_sha in result:
            raise ValueError(f"duplicate audit batchStateSha256: {batch_sha}")
        result[batch_sha] = audit
    return result


def _validate_document_shapes(
    audits: list[dict[str, object]],
    cohorts: dict[str, object],
    labels: dict[str, object],
) -> None:
    if cohorts.get("version") != COHORT_MANIFEST_VERSION:
        raise ValueError(f"unsupported cohort manifest version: {cohorts.get('version')}")
    if labels.get("version") != LABEL_SET_VERSION:
        raise ValueError(f"unsupported label set version: {labels.get('version')}")
    for audit in audits:
        if audit.get("version") != AUDIT_VERSION:
            raise ValueError(f"unsupported audit version: {audit.get('version')}")
        rows = audit.get("rows")
        if not isinstance(rows, list):
            raise TypeError("audit rows must be a list")
        if audit.get("pairCount") != len(rows):
            raise ValueError("audit pairCount does not match rows")
        batch_sha = audit.get("batchStateSha256")
        for row in rows:
            if row.get("batchStateSha256") != batch_sha:
                raise ValueError("audit row batchStateSha256 does not match audit")


def _validate_cohort_registration(
    audits_by_batch: dict[str, dict[str, object]],
    cohorts: dict[str, object],
) -> None:
    cohort_ids: set[str] = set()
    registered_batches: set[str] = set()
    for cohort in cohorts.get("cohorts", []):
        cohort_id = str(cohort["cohortId"])
        if cohort_id in cohort_ids:
            raise ValueError(f"duplicate cohortId: {cohort_id}")
        cohort_ids.add(cohort_id)
        batch_sha = str(cohort["batchStateSha256"])
        if batch_sha in registered_batches:
            raise ValueError(f"duplicate cohort batch: {batch_sha}")
        registered_batches.add(batch_sha)
        if batch_sha not in audits_by_batch:
            raise ValueError(f"cohort has no matching audit: {batch_sha}")
    unregistered = sorted(set(audits_by_batch) - registered_batches)
    if unregistered:
        raise ValueError(f"audit is not registered by a cohort: {unregistered}")


def _labels_by_pair(
    labels: dict[str, object],
) -> dict[tuple[str, int, int], list[dict[str, object]]]:
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for label in labels.get("labels", []):
        grouped[_pair_key(label)].append(label)
    return grouped


def _rows_by_pair(
    audits: list[dict[str, object]],
) -> dict[tuple[str, int, int], dict[str, object]]:
    result: dict[tuple[str, int, int], dict[str, object]] = {}
    for audit in audits:
        for row in audit.get("rows", []):
            key = _pair_key(row)
            if key in result:
                raise ValueError(f"duplicate audit pair: {key}")
            result[key] = row
    return result


def _validate_labels(
    audits: list[dict[str, object]],
    labels: dict[str, object],
) -> None:
    rows_by_pair = _rows_by_pair(audits)
    label_ids: set[str] = set()
    for label in labels.get("labels", []):
        label_id = str(label["labelId"])
        if label_id in label_ids:
            raise ValueError(f"duplicate labelId: {label_id}")
        label_ids.add(label_id)
        allowed_values = {
            "humanVerdict": HUMAN_VERDICTS,
            "confidence": HUMAN_CONFIDENCES,
            "scope": LABEL_SCOPES,
        }
        for field, allowed in allowed_values.items():
            if label.get(field) not in allowed:
                raise ValueError(f"label {field} is not supported: {label.get(field)}")
        key = _pair_key(label)
        row = rows_by_pair.get(key)
        if row is None:
            raise ValueError(f"label has no matching audit pair: {key}")
        review = row["review"]
        expected = {
            "sourceName": row["sourceName"],
            "generationReportSha256": row["generationReportSha256"],
            "audioSha256": row["audioSha256"],
            "hardCandidateId": review["hard"]["candidateId"],
            "expertCandidateId": review["expert"]["candidateId"],
        }
        for field, expected_value in expected.items():
            if label.get(field) != expected_value:
                raise ValueError(f"label {field} does not match audit evidence: {label_id}")


def _eligible_pair_labels(
    pair_labels: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        label
        for label in pair_labels
        if label.get("scope") == "PAIR_QUALITY"
        and label.get("confidence") in {"HIGH", "MEDIUM"}
        and label.get("humanVerdict") != "UNCERTAIN"
    ]


def _resolved_verdict(pair_labels: list[dict[str, object]]) -> str | None:
    eligible = _eligible_pair_labels(pair_labels)
    verdicts = {str(label["humanVerdict"]) for label in eligible}
    if len(verdicts) != 1:
        return None
    return verdicts.pop()


def _human_label_conflicts(labels_by_pair) -> list[dict[str, object]]:
    conflicts: list[dict[str, object]] = []
    for key, pair_labels in labels_by_pair.items():
        eligible = _eligible_pair_labels(pair_labels)
        verdicts = sorted({str(label["humanVerdict"]) for label in eligible})
        if len(verdicts) <= 1:
            continue
        conflicts.append(
            {
                "batchStateSha256": key[0],
                "songIndex": key[1],
                "keyMode": key[2],
                "labelIds": sorted(str(label["labelId"]) for label in eligible),
                "verdicts": verdicts,
            }
        )
    return sorted(
        conflicts,
        key=lambda item: (
            str(item["batchStateSha256"]),
            int(item["songIndex"]),
            int(item["keyMode"]),
        ),
    )


def _validate_external_audio_is_unseen(
    audits_by_batch: dict[str, dict[str, object]],
    cohorts: dict[str, object],
) -> None:
    external_audio: set[str] = set()
    known_audio: set[str] = set()
    for cohort in cohorts.get("cohorts", []):
        audit = audits_by_batch[str(cohort["batchStateSha256"])]
        audio = {str(row["audioSha256"]) for row in audit.get("rows", [])}
        if cohort["role"] == "EXTERNAL_UNSEEN":
            external_audio.update(audio)
        else:
            known_audio.update(audio)
    overlap = sorted(external_audio & known_audio)
    if overlap:
        raise ValueError(f"EXTERNAL_UNSEEN audio leakage: {overlap}")


def _parse_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be an RFC 3339 timestamp")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an RFC 3339 timestamp") from exc


def _validate_cohort_roles(cohorts: dict[str, object]) -> None:
    for cohort in cohorts.get("cohorts", []):
        if cohort.get("role") not in COHORT_ROLES:
            raise ValueError(f"unsupported cohort role: {cohort.get('role')}")


def _validate_external_metadata(
    audits_by_batch: dict[str, dict[str, object]],
    cohorts: dict[str, object],
) -> None:
    policy_snapshot = cohorts.get("policySnapshot")
    frozen_at = _parse_datetime(
        policy_snapshot.get("frozenAt") if isinstance(policy_snapshot, dict) else None,
        field="policySnapshot.frozenAt",
    )
    for cohort in cohorts.get("cohorts", []):
        if cohort["role"] != "EXTERNAL_UNSEEN":
            continue
        audit = audits_by_batch[str(cohort["batchStateSha256"])]
        started_at = _parse_datetime(
            audit.get("batchStartedAt"),
            field="external audit batchStartedAt",
        )
        if started_at <= frozen_at:
            raise ValueError("EXTERNAL_UNSEEN batch must start after policy freeze")
        if cohort.get("groupIsolationStatus") != "CONFIRMED":
            raise ValueError("EXTERNAL_UNSEEN requires groupIsolationStatus=CONFIRMED")
        if cohort.get("exposedToPolicyDesign") is not False:
            raise ValueError("EXTERNAL_UNSEEN must not be exposed to policy design")


def _is_automatic_defect(row: dict[str, object]) -> bool:
    review = row["review"]
    return (
        review["status"] == "DEFECT"
        and review["reason"] == "ISOLATED_EXPERT_FIRST_ROW"
    )


def _matrix(rows: list[dict[str, object]], labels_by_pair) -> dict[str, int]:
    result = {
        "truePositive": 0,
        "trueNegative": 0,
        "falsePositive": 0,
        "falseNegative": 0,
        "eligibleLabelCount": 0,
    }
    for row in rows:
        verdict = _resolved_verdict(labels_by_pair.get(_pair_key(row), []))
        if verdict is None:
            continue
        result["eligibleLabelCount"] += 1
        automatic = _is_automatic_defect(row)
        human = verdict == "DEFECT"
        if automatic and human:
            result["truePositive"] += 1
        elif not automatic and not human:
            result["trueNegative"] += 1
        elif automatic:
            result["falsePositive"] += 1
        else:
            result["falseNegative"] += 1
    return result


def evaluate_intro_phrase_calibration(
    audits: list[dict[str, object]],
    cohorts: dict[str, object],
    labels: dict[str, object],
) -> dict[str, object]:
    _validate_document_shapes(audits, cohorts, labels)
    audits_by_batch = _audits_by_batch(audits)
    _validate_cohort_roles(cohorts)
    _validate_cohort_registration(audits_by_batch, cohorts)
    _validate_labels(audits, labels)
    _validate_external_audio_is_unseen(audits_by_batch, cohorts)
    _validate_external_metadata(audits_by_batch, cohorts)
    labels_by_pair = _labels_by_pair(labels)
    conflicts = _human_label_conflicts(labels_by_pair)
    cohort_reports: list[dict[str, object]] = []
    recovery_labels: list[dict[str, object]] = []
    roles: set[str] = set()

    for cohort in cohorts.get("cohorts", []):
        role = str(cohort["role"])
        roles.add(role)
        audit = audits_by_batch[str(cohort["batchStateSha256"])]
        rows = list(audit.get("rows", []))
        labeled_keys = {
            _pair_key(label)
            for row in rows
            for label in labels_by_pair.get(_pair_key(row), [])
        }
        cohort_reports.append(
            {
                "cohortId": cohort["cohortId"],
                "role": role,
                "songCount": audit.get("songCount"),
                "pairCount": len(rows),
                "labeledPairCount": len(labeled_keys),
                "unlabeledPairCount": len(rows) - len(labeled_keys),
                "matrix": _matrix(rows, labels_by_pair),
            }
        )
        if role == "RECOVERY_OUTCOME":
            recovery_labels.extend(
                label
                for row in rows
                for label in labels_by_pair.get(_pair_key(row), [])
                if label.get("scope") == "RECOVERY_OUTCOME"
                and label.get("confidence") in {"HIGH", "MEDIUM"}
            )

    recovery = {
        "eligibleLabelCount": len(recovery_labels),
        "accepted": sum(
            label["humanVerdict"] == "ACCEPTABLE" for label in recovery_labels
        ),
        "defect": sum(label["humanVerdict"] == "DEFECT" for label in recovery_labels),
        "uncertainOrExcluded": sum(
            label["humanVerdict"] == "UNCERTAIN" for label in recovery_labels
        ),
    }
    blockers: list[str] = []
    if "DEVELOPMENT_DISCOVERY" in roles:
        blockers.append("DISCOVERY_ONLY_NOT_GENERALIZATION")
    if "EXTERNAL_UNSEEN" not in roles:
        blockers.append("NO_EXTERNAL_UNSEEN_COHORT")
    external_requirements = cohorts.get("externalRegistrationRequirements")
    acceptance_status = (
        external_requirements.get("acceptanceCriteriaStatus")
        if isinstance(external_requirements, dict)
        else None
    )
    if acceptance_status != "PREREGISTERED":
        blockers.append("EXTERNAL_ACCEPTANCE_CRITERIA_NOT_PREREGISTERED")
    if conflicts:
        blockers.append("HUMAN_LABEL_CONFLICT")
    return {
        "version": "intro-phrase-calibration-report-v1",
        "policySnapshot": cohorts.get("policySnapshot"),
        "cohorts": cohort_reports,
        "recoveryOutcomes": recovery,
        "conflicts": conflicts,
        "promotionEligible": False,
        "promotionBlockers": blockers,
    }


def build_intro_phrase_review_queue(
    audits: list[dict[str, object]],
    cohorts: dict[str, object],
    labels: dict[str, object],
) -> dict[str, object]:
    _validate_document_shapes(audits, cohorts, labels)
    audits_by_batch = _audits_by_batch(audits)
    _validate_cohort_roles(cohorts)
    _validate_cohort_registration(audits_by_batch, cohorts)
    _validate_labels(audits, labels)
    labels_by_pair = _labels_by_pair(labels)
    priority = {"DEFECT": 0, "REVIEW": 1, "INSUFFICIENT": 2, "PASS": 3}
    rows: list[dict[str, object]] = []
    for cohort in cohorts.get("cohorts", []):
        if cohort["role"] == "RECOVERY_OUTCOME":
            continue
        audit = audits_by_batch[str(cohort["batchStateSha256"])]
        for row in audit.get("rows", []):
            has_pair_label = any(
                label.get("scope") == "PAIR_QUALITY"
                for label in labels_by_pair.get(_pair_key(row), [])
            )
            if not has_pair_label:
                rows.append({"cohortId": cohort["cohortId"], **row})
    rows.sort(
        key=lambda row: (
            priority.get(str(row["review"]["status"]), 99),
            int(row["songIndex"]),
            int(row["keyMode"]),
        )
    )
    return {"version": "intro-phrase-review-queue-v1", "rows": rows}
