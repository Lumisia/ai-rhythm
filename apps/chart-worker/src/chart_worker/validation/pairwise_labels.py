"""Hash-bound, blinded pairwise difficulty-label contracts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from chart_worker.schema.types import KEY_MODES

PAIRWISE_TASK_VERSION = "difficulty-pairwise-task-v1"
PAIRWISE_LABEL_VERSION = "difficulty-pairwise-label-v1"
PAIRWISE_TASK_BUNDLE_VERSION = "difficulty-pairwise-task-bundle-v1"
PAIRWISE_LABEL_EXPORT_VERSION = "difficulty-pairwise-label-export-v1"

PairwiseAnswer = Literal["LEFT", "RIGHT", "TIE", "UNCERTAIN"]
PairwiseDimension = Literal["harder", "musical_quality"]
CanonicalAnswer = str | Literal["TIE", "UNCERTAIN"]
_ANSWERS = {"LEFT", "RIGHT", "TIE", "UNCERTAIN"}


def _exact_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty exact string")
    return value


def _sha256(value: object, *, name: str) -> str:
    digest = _exact_string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _exact_mapping(
    value: object,
    *,
    name: str,
    keys: frozenset[str],
) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{name} must be an exact string-keyed object")
    actual = frozenset(value)
    if actual != keys:
        raise ValueError(
            f"{name} keys differ: missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _exact_list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an exact list")
    return value


@dataclass(frozen=True, slots=True)
class CandidateLabelBindingV1:
    candidate_id: str
    audio_sha256: str
    key_mode: int
    payload_sha256: str
    feature_sha256: str

    def __post_init__(self) -> None:
        _exact_string(self.candidate_id, name="candidate_id")
        _sha256(self.audio_sha256, name="audio_sha256")
        if type(self.key_mode) is not int or self.key_mode not in KEY_MODES:
            raise ValueError("key_mode is unsupported")
        _sha256(self.payload_sha256, name="payload_sha256")
        _sha256(self.feature_sha256, name="feature_sha256")

    def to_private_report(self) -> dict[str, object]:
        return {
            "candidateId": self.candidate_id,
            "audioSha256": self.audio_sha256,
            "keyMode": self.key_mode,
            "payloadSha256": self.payload_sha256,
            "featureSha256": self.feature_sha256,
        }

    def to_review_report(self) -> dict[str, object]:
        return {"payloadSha256": self.payload_sha256}


@dataclass(frozen=True, slots=True)
class PairwiseTaskV1:
    task_id: str
    left: CandidateLabelBindingV1
    right: CandidateLabelBindingV1
    version: Literal["difficulty-pairwise-task-v1"] = PAIRWISE_TASK_VERSION

    def __post_init__(self) -> None:
        if self.version != PAIRWISE_TASK_VERSION:
            raise ValueError("unsupported pairwise task version")
        _sha256(self.task_id, name="task_id")
        if not isinstance(self.left, CandidateLabelBindingV1) or not isinstance(
            self.right, CandidateLabelBindingV1
        ):
            raise TypeError("left and right must be candidate bindings")
        if self.left.candidate_id == self.right.candidate_id:
            raise ValueError("pairwise candidates must differ")
        if (
            self.left.audio_sha256 != self.right.audio_sha256
            or self.left.key_mode != self.right.key_mode
        ):
            raise ValueError("pairwise candidates must share audio and key mode")

    def to_private_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "taskId": self.task_id,
            "left": self.left.to_private_report(),
            "right": self.right.to_private_report(),
        }

    def to_review_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "taskId": self.task_id,
            "left": self.left.to_review_report(),
            "right": self.right.to_review_report(),
        }

    def stable_sha256(self) -> str:
        return _canonical_sha256(self.to_private_report())


def build_pairwise_task(
    first: CandidateLabelBindingV1,
    second: CandidateLabelBindingV1,
    *,
    presentation_seed: str,
    force_left_candidate_id: str | None = None,
) -> PairwiseTaskV1:
    if not isinstance(first, CandidateLabelBindingV1) or not isinstance(
        second, CandidateLabelBindingV1
    ):
        raise TypeError("pairwise task requires candidate bindings")
    seed = _exact_string(presentation_seed, name="presentation_seed")
    if first.candidate_id == second.candidate_id:
        raise ValueError("pairwise candidates must differ")
    candidates = {first.candidate_id: first, second.candidate_id: second}
    if force_left_candidate_id is not None:
        left_id = _exact_string(force_left_candidate_id, name="force_left_candidate_id")
        if left_id not in candidates:
            raise ValueError("forced left candidate is not in the pair")
    else:
        ordering_digest = hashlib.sha256(
            f"pairwise-side-v1:{seed}:{min(candidates)}:{max(candidates)}".encode()
        ).digest()
        left_id = sorted(candidates)[ordering_digest[0] % 2]
    right_id = next(candidate_id for candidate_id in candidates if candidate_id != left_id)
    task_id = _canonical_sha256(
        {
            "version": PAIRWISE_TASK_VERSION,
            "presentationSeedSha256": hashlib.sha256(seed.encode()).hexdigest(),
            "left": candidates[left_id].to_private_report(),
            "right": candidates[right_id].to_private_report(),
        }
    )
    return PairwiseTaskV1(task_id, candidates[left_id], candidates[right_id])


@dataclass(frozen=True, slots=True)
class PairwiseTaskBundleV1:
    presentation_seed_sha256: str
    include_reversed: bool
    tasks: tuple[PairwiseTaskV1, ...]
    version: Literal["difficulty-pairwise-task-bundle-v1"] = PAIRWISE_TASK_BUNDLE_VERSION

    def __post_init__(self) -> None:
        if self.version != PAIRWISE_TASK_BUNDLE_VERSION:
            raise ValueError("unsupported pairwise task bundle version")
        _sha256(self.presentation_seed_sha256, name="presentation_seed_sha256")
        if type(self.include_reversed) is not bool:
            raise TypeError("include_reversed must be an exact boolean")
        if type(self.tasks) is not tuple or not self.tasks or any(
            not isinstance(task, PairwiseTaskV1) for task in self.tasks
        ):
            raise TypeError("tasks must be a non-empty tuple of PairwiseTaskV1")
        if self.tasks != tuple(sorted(self.tasks, key=lambda task: task.task_id)) or len(
            {task.task_id for task in self.tasks}
        ) != len(self.tasks):
            raise ValueError("tasks must be sorted with unique task identities")

    def to_private_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "presentationSeedSha256": self.presentation_seed_sha256,
            "includeReversed": self.include_reversed,
            "tasks": [task.to_private_report() for task in self.tasks],
        }

    def stable_sha256(self) -> str:
        return _canonical_sha256(self.to_private_report())

    def to_review_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "privateBundleSha256": self.stable_sha256(),
            "tasks": [task.to_review_report() for task in self.tasks],
            "responseContract": {
                "harderAnswer": sorted(_ANSWERS),
                "musicalQualityAnswer": sorted(_ANSWERS),
                "confidence": {"minimum": 1, "maximum": 5},
            },
        }


def build_pairwise_task_bundle(
    bindings: tuple[CandidateLabelBindingV1, ...],
    *,
    pairs: tuple[tuple[str, str], ...],
    presentation_seed: str,
    include_reversed: bool,
) -> PairwiseTaskBundleV1:
    if type(bindings) is not tuple or any(
        not isinstance(binding, CandidateLabelBindingV1) for binding in bindings
    ):
        raise TypeError("bindings must be a tuple of candidate bindings")
    by_id = {binding.candidate_id: binding for binding in bindings}
    if not by_id or len(by_id) != len(bindings):
        raise ValueError("bindings must have unique candidate identities")
    if type(pairs) is not tuple or not pairs or any(
        type(pair) is not tuple
        or len(pair) != 2
        or any(type(candidate_id) is not str or not candidate_id for candidate_id in pair)
        for pair in pairs
    ):
        raise TypeError("pairs must be a non-empty tuple of candidate identity pairs")
    canonical_pairs = tuple(sorted({tuple(sorted(pair)) for pair in pairs}))
    if pairs != canonical_pairs or any(left == right for left, right in pairs):
        raise ValueError("pairs must be canonical, sorted and unique")
    seed = _exact_string(presentation_seed, name="presentation_seed")
    if type(include_reversed) is not bool:
        raise TypeError("include_reversed must be an exact boolean")

    tasks = []
    for index, (first_id, second_id) in enumerate(pairs):
        try:
            first = by_id[first_id]
            second = by_id[second_id]
        except KeyError as error:
            raise ValueError("pair references a missing candidate binding") from error
        primary = build_pairwise_task(
            first,
            second,
            presentation_seed=f"{seed}:pair:{index}:primary",
        )
        tasks.append(primary)
        if include_reversed:
            tasks.append(
                build_pairwise_task(
                    first,
                    second,
                    presentation_seed=f"{seed}:pair:{index}:reversed",
                    force_left_candidate_id=primary.right.candidate_id,
                )
            )
    return PairwiseTaskBundleV1(
        presentation_seed_sha256=hashlib.sha256(seed.encode()).hexdigest(),
        include_reversed=include_reversed,
        tasks=tuple(sorted(tasks, key=lambda task: task.task_id)),
    )


@dataclass(frozen=True, slots=True)
class PairwiseLabelV1:
    task_sha256: str
    rater_sha256: str
    harder_answer: PairwiseAnswer
    musical_quality_answer: PairwiseAnswer
    confidence: int
    version: Literal["difficulty-pairwise-label-v1"] = PAIRWISE_LABEL_VERSION

    def __post_init__(self) -> None:
        if self.version != PAIRWISE_LABEL_VERSION:
            raise ValueError("unsupported pairwise label version")
        _sha256(self.task_sha256, name="task_sha256")
        _sha256(self.rater_sha256, name="rater_sha256")
        for name, answer in (
            ("harder_answer", self.harder_answer),
            ("musical_quality_answer", self.musical_quality_answer),
        ):
            if type(answer) is not str or answer not in _ANSWERS:
                raise ValueError(f"{name} is unsupported")
        if type(self.confidence) is not int:
            raise TypeError("confidence must be an exact integer")
        if not 1 <= self.confidence <= 5:
            raise ValueError("confidence must be between 1 and 5")

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "taskSha256": self.task_sha256,
            "raterSha256": self.rater_sha256,
            "harderAnswer": self.harder_answer,
            "musicalQualityAnswer": self.musical_quality_answer,
            "confidence": self.confidence,
        }


def parse_candidate_label_binding_v1(value: object) -> CandidateLabelBindingV1:
    report = _exact_mapping(
        value,
        name="candidate binding",
        keys=frozenset(
            {
                "candidateId",
                "audioSha256",
                "keyMode",
                "payloadSha256",
                "featureSha256",
            }
        ),
    )
    parsed = CandidateLabelBindingV1(
        candidate_id=report["candidateId"],
        audio_sha256=report["audioSha256"],
        key_mode=report["keyMode"],
        payload_sha256=report["payloadSha256"],
        feature_sha256=report["featureSha256"],
    )
    if parsed.to_private_report() != report:
        raise ValueError("candidate binding projection differs")
    return parsed


def _parse_pairwise_task(value: object) -> PairwiseTaskV1:
    report = _exact_mapping(
        value,
        name="pairwise task",
        keys=frozenset({"version", "taskId", "left", "right"}),
    )
    parsed = PairwiseTaskV1(
        version=report["version"],
        task_id=report["taskId"],
        left=parse_candidate_label_binding_v1(report["left"]),
        right=parse_candidate_label_binding_v1(report["right"]),
    )
    if parsed.to_private_report() != report:
        raise ValueError("pairwise task projection differs")
    return parsed


def parse_pairwise_task_bundle_v1(value: object) -> PairwiseTaskBundleV1:
    report = _exact_mapping(
        value,
        name="pairwise task bundle",
        keys=frozenset(
            {"version", "presentationSeedSha256", "includeReversed", "tasks"}
        ),
    )
    parsed = PairwiseTaskBundleV1(
        version=report["version"],
        presentation_seed_sha256=report["presentationSeedSha256"],
        include_reversed=report["includeReversed"],
        tasks=tuple(
            _parse_pairwise_task(task)
            for task in _exact_list(report["tasks"], name="tasks")
        ),
    )
    if parsed.to_private_report() != report:
        raise ValueError("pairwise task bundle projection differs")
    return parsed


def parse_pairwise_label_v1(value: object) -> PairwiseLabelV1:
    report = _exact_mapping(
        value,
        name="pairwise label",
        keys=frozenset(
            {
                "version",
                "taskSha256",
                "raterSha256",
                "harderAnswer",
                "musicalQualityAnswer",
                "confidence",
            }
        ),
    )
    parsed = PairwiseLabelV1(
        version=report["version"],
        task_sha256=report["taskSha256"],
        rater_sha256=report["raterSha256"],
        harder_answer=report["harderAnswer"],
        musical_quality_answer=report["musicalQualityAnswer"],
        confidence=report["confidence"],
    )
    if parsed.to_report() != report:
        raise ValueError("pairwise label projection differs")
    return parsed


@dataclass(frozen=True, slots=True)
class PairwiseLabelExportV1:
    private_bundle_sha256: str
    packet_sha256: str
    completed_task_count: int
    total_task_count: int
    labels: tuple[PairwiseLabelV1, ...]
    version: Literal["difficulty-pairwise-label-export-v1"] = PAIRWISE_LABEL_EXPORT_VERSION

    def __post_init__(self) -> None:
        if self.version != PAIRWISE_LABEL_EXPORT_VERSION:
            raise ValueError("unsupported pairwise label export version")
        _sha256(self.private_bundle_sha256, name="private_bundle_sha256")
        _sha256(self.packet_sha256, name="packet_sha256")
        for value, name in (
            (self.completed_task_count, "completed_task_count"),
            (self.total_task_count, "total_task_count"),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
        if self.total_task_count < 1 or not 1 <= self.completed_task_count <= (
            self.total_task_count
        ):
            raise ValueError("completed count must be within the total task count")
        if type(self.labels) is not tuple or any(
            not isinstance(label, PairwiseLabelV1) for label in self.labels
        ):
            raise TypeError("labels must be a tuple of PairwiseLabelV1")
        if len(self.labels) != self.completed_task_count:
            raise ValueError("completed count differs from labels")
        if len({label.task_sha256 for label in self.labels}) != len(self.labels):
            raise ValueError("labels must reference a unique task each")

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "privateBundleSha256": self.private_bundle_sha256,
            "packetSha256": self.packet_sha256,
            "completedTaskCount": self.completed_task_count,
            "totalTaskCount": self.total_task_count,
            "labels": [label.to_report() for label in self.labels],
        }


def parse_pairwise_label_export_v1(value: object) -> PairwiseLabelExportV1:
    report = _exact_mapping(
        value,
        name="pairwise label export",
        keys=frozenset(
            {
                "version",
                "privateBundleSha256",
                "packetSha256",
                "completedTaskCount",
                "totalTaskCount",
                "labels",
            }
        ),
    )
    parsed = PairwiseLabelExportV1(
        version=report["version"],
        private_bundle_sha256=report["privateBundleSha256"],
        packet_sha256=report["packetSha256"],
        completed_task_count=report["completedTaskCount"],
        total_task_count=report["totalTaskCount"],
        labels=tuple(
            parse_pairwise_label_v1(label)
            for label in _exact_list(report["labels"], name="labels")
        ),
    )
    if parsed.to_report() != report:
        raise ValueError("pairwise label export projection differs")
    return parsed


def bind_pairwise_label_export_v1(
    exported: PairwiseLabelExportV1,
    *,
    bundle: PairwiseTaskBundleV1,
    expected_packet_sha256: str,
) -> tuple[tuple[PairwiseTaskV1, PairwiseLabelV1], ...]:
    if not isinstance(exported, PairwiseLabelExportV1):
        raise TypeError("exported must be PairwiseLabelExportV1")
    if not isinstance(bundle, PairwiseTaskBundleV1):
        raise TypeError("bundle must be PairwiseTaskBundleV1")
    packet_digest = _sha256(expected_packet_sha256, name="expected_packet_sha256")
    if exported.packet_sha256 != packet_digest:
        raise ValueError("label export packet digest differs")
    if exported.private_bundle_sha256 != bundle.stable_sha256():
        raise ValueError("label export private bundle digest differs")
    if exported.total_task_count != len(bundle.tasks):
        raise ValueError("label export total task count differs")
    by_sha = {task.stable_sha256(): task for task in bundle.tasks}
    entries = []
    for label in exported.labels:
        task = by_sha.get(label.task_sha256)
        if task is None:
            raise ValueError("label export references an unknown task")
        entries.append((task, label))
    return tuple(entries)


def canonical_answer(
    task: PairwiseTaskV1,
    label: PairwiseLabelV1,
    *,
    dimension: PairwiseDimension,
) -> CanonicalAnswer:
    if not isinstance(task, PairwiseTaskV1) or not isinstance(label, PairwiseLabelV1):
        raise TypeError("canonical answer requires a task and label")
    if label.task_sha256 != task.stable_sha256():
        raise ValueError("label task digest does not match the pairwise task")
    if dimension == "harder":
        answer = label.harder_answer
    elif dimension == "musical_quality":
        answer = label.musical_quality_answer
    else:
        raise ValueError("dimension is unsupported")
    if answer == "LEFT":
        return task.left.candidate_id
    if answer == "RIGHT":
        return task.right.candidate_id
    return answer


@dataclass(frozen=True, slots=True)
class PairwiseLabelAuditV1:
    usable_harder_labels: int
    consistent_repeated_pairs: int
    contradictions: tuple[tuple[str, str], ...]


def audit_pairwise_labels(
    entries: tuple[tuple[PairwiseTaskV1, PairwiseLabelV1], ...],
) -> PairwiseLabelAuditV1:
    if type(entries) is not tuple:
        raise TypeError("entries must be a tuple")
    grouped: dict[tuple[str, int, str, str, str], list[CanonicalAnswer]] = defaultdict(list)
    for task, label in entries:
        winner = canonical_answer(task, label, dimension="harder")
        pair = tuple(sorted((task.left.candidate_id, task.right.candidate_id)))
        key = (
            task.left.audio_sha256,
            task.left.key_mode,
            pair[0],
            pair[1],
            label.rater_sha256,
        )
        grouped[key].append(winner)

    contradictions = []
    usable = 0
    consistent_repeated = 0
    for (_audio, _key, left_id, right_id, _rater), answers in grouped.items():
        concrete = {answer for answer in answers if answer not in {"TIE", "UNCERTAIN"}}
        if len(concrete) > 1:
            contradictions.append((left_id, right_id))
            continue
        if len(concrete) == 1:
            usable += 1
            if sum(answer in concrete for answer in answers) >= 2:
                consistent_repeated += 1
    return PairwiseLabelAuditV1(
        usable_harder_labels=usable,
        consistent_repeated_pairs=consistent_repeated,
        contradictions=tuple(sorted(contradictions)),
    )
