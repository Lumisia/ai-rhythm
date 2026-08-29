"""Run one hash-bound OFF/OBSERVE or OFF/OBSERVE/SHADOW canary.

The registration is deliberately identity-free at the semantic layer.  Local
paths are accepted only as transport locators and never enter request/evidence
digests.  This script is research tooling and never changes public selection.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from chart_worker.analysis.onset import OnsetAnalysis, analyze_canonical_audio
from chart_worker.config import WorkerConfig
from chart_worker.errors import WorkerError
from chart_worker.generation.generation_origin_diagnostics import (
    GenerationOriginDiagnostics,
)
from chart_worker.generation.inference_session import (
    SongIdentity,
    inference_song_scope,
)
from chart_worker.generation.mapperatorinator import (
    GeneratedChart,
    MapperatorinatorGenerator,
    find_generated_osu,
)
from chart_worker.generation.mapperatorinator_patch import (
    CONSTRAINT_PATCH_ID,
    EXPECTED_MAPPERATORINATOR_HEAD,
)
from chart_worker.generation.osu_parser import parse_osu_file
from chart_worker.generation.params import GenerationRequest
from chart_worker.generation.partial_remap import PartialRemapWindow
from chart_worker.generation.required_gameplay_interval import (
    RequiredGameplayEvidenceV1,
    RequiredGameplayFamilySlotV1,
    RequiredGameplayIntervalMode,
    RequiredGameplayIntervalV1,
    plan_required_gameplay_interval,
)
from chart_worker.generation.required_gameplay_invocation import (
    required_gameplay_invocation_digest,
    required_gameplay_invocation_payload,
)
from chart_worker.hashing import sha256_file
from chart_worker.pipeline import _open_inference_session, _song_config_digest
from chart_worker.stages.types import SongTimingAuthority
from chart_worker.validation.quality_gate import evaluate_chart_candidate

REGISTRATION_VERSION = "required-gameplay-canary-registration-v1"
REPORT_VERSION = "required-gameplay-canary-report-v1"
SHADOW_REPORT_VERSION = "required-gameplay-shadow-canary-report-v4"
RootCauseClassification = Literal[
    "CONTENT_OBLIGATION_UNSATISFIED",
    "WINDOW_MERGE_DELETION",
    "CANONICAL_DELETION",
    "RESNAP_DELETION",
    "SERIALIZATION_DELETION",
    "NO_OBSERVED_LOSS",
    "ACCOUNTING_INCONCLUSIVE",
    "CANARY_EXECUTION_FAILURE",
]


@dataclass(frozen=True, slots=True)
class ChartSummary:
    osu_sha256: str
    semantic_sha256: str
    suffix_semantic_sha256: str
    timing_section_sha256: str
    timing_semantic_sha256: str
    note_count: int
    hold_count: int
    interval_complete_group_count: int


@dataclass(frozen=True, slots=True)
class CanaryComparison:
    equivalent: bool
    classification: Literal[
        "EQUIVALENT",
        "OBSERVE_INTERFERENCE",
        "CANARY_EXECUTION_FAILURE",
    ]
    mismatches: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QualitySummary:
    action: str
    disposition: str
    first_note_time_ms: int | None
    active_leading_gap_count: int
    active_leading_gap_total_ms: int
    matched_precision_50: float | None
    matched_f1_50: float | None
    project_rating: float | None


@dataclass(frozen=True, slots=True)
class ShadowEvaluation:
    status: Literal["CONTRACT_PASS", "TYPED_FAILURE", "CONTRACT_FAIL"]
    isolation_pass: bool
    failures: tuple[str, ...]
    typed_failure_reason: str | None


@dataclass(frozen=True, slots=True)
class _RunResult:
    label: Literal["OFF", "OBSERVE", "SHADOW"]
    started_at: str
    ended_at: str
    wall_ms: int
    exit_code: int
    failure_class: str | None
    failure_message: str | None
    failure_context: dict[str, object] | None
    invocation_count: int
    chart: ChartSummary | None
    quality: QualitySummary | None
    diagnostics: GenerationOriginDiagnostics | None


def _canonical_sha(payload: object) -> str:
    value = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return sha256(value).hexdigest()


def _report_filename(*, include_shadow: bool) -> str:
    if include_shadow:
        return f"{SHADOW_REPORT_VERSION}.json"
    return f"{REPORT_VERSION}.json"


def _read_cached_report(
    report_path: Path,
    *,
    expected_version: str,
    registration_sha256: str,
) -> dict[str, object] | None:
    if report_path.is_symlink():
        raise ValueError("cached canary report must not be a symlink")
    if not report_path.exists():
        return None
    if not report_path.is_file():
        raise ValueError("cached canary report must be a regular JSON file")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if type(report) is not dict:
        raise ValueError("cached canary report must be a JSON object")
    if report.get("version") != expected_version:
        raise ValueError("cached canary report version differs from this run")
    if report.get("registrationSha256") != registration_sha256:
        raise ValueError("cached canary report belongs to a different registration")
    return report


def _write_report_atomic(report_path: Path, report: dict[str, object]) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=report_path.parent,
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            temporary_path = Path(stream.name)
        if report_path.exists() or report_path.is_symlink():
            raise FileExistsError("canary report appeared before atomic promotion")
        temporary_path.replace(report_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _timing_section_bytes(path: Path) -> bytes:
    lines = Path(path).read_bytes().splitlines(keepends=True)
    collecting = False
    result: list[bytes] = []
    for line in lines:
        stripped = line.strip()
        if stripped == b"[TimingPoints]":
            collecting = True
            result.append(line)
            continue
        if collecting and stripped.startswith(b"[") and stripped.endswith(b"]"):
            break
        if collecting:
            result.append(line)
    if not result:
        raise ValueError(".osu has no raw [TimingPoints] section")
    return b"".join(result)


def _canonical_decimal(value: str) -> str:
    try:
        parsed = Decimal(value.strip())
    except InvalidOperation as error:
        raise ValueError(f"invalid timing decimal: {value!r}") from error
    if not parsed.is_finite():
        raise ValueError(f"non-finite timing decimal: {value!r}")
    if parsed == 0:
        parsed = Decimal(0)
    return format(parsed.normalize(), "f")


def _timing_semantic_payload(path: Path) -> list[list[str]]:
    payload: list[list[str]] = []
    for raw_line in _timing_section_bytes(path).decode("utf-8-sig").splitlines()[1:]:
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 8:
            raise ValueError(f"timing point must have exactly 8 fields: {line!r}")
        try:
            integer_fields = [str(int(item)) for item in parts[2:]]
        except ValueError as error:
            raise ValueError(f"invalid timing integer field: {line!r}") from error
        payload.append(
            [
                _canonical_decimal(parts[0]),
                _canonical_decimal(parts[1]),
                *integer_fields,
            ]
        )
    return payload


def _note_payload(note: Any) -> dict[str, object]:
    return {
        "timeMs": note.time_ms,
        "lane": note.lane,
        "kind": note.kind,
        "durationMs": note.duration_ms,
    }


def canonical_chart_summary(
    path: Path,
    *,
    interval_start_ms: int,
    interval_end_ms: int,
    partial_end_ms: int,
) -> ChartSummary:
    """Independently parse an output without trusting the origin sidecar."""

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("canary output must be an existing regular .osu")
    if not 0 <= interval_start_ms < interval_end_ms <= partial_end_ms:
        raise ValueError("canary interval must fit the partial range")
    beatmap = parse_osu_file(path)
    semantic = {
        "keyMode": beatmap.key_mode,
        "bpmEvents": [
            {"timeMs": event.time_ms, "bpm": event.bpm}
            for event in beatmap.bpm_events
        ],
        "notes": [_note_payload(note) for note in beatmap.notes],
    }
    suffix = [
        _note_payload(note)
        for note in beatmap.notes
        if note.time_ms > partial_end_ms
    ]
    interval_count = sum(
        interval_start_ms <= note.time_ms <= interval_end_ms
        for note in beatmap.notes
    )
    timing_bytes = _timing_section_bytes(path)
    return ChartSummary(
        osu_sha256=sha256_file(path),
        semantic_sha256=_canonical_sha(semantic),
        suffix_semantic_sha256=_canonical_sha(suffix),
        timing_section_sha256=sha256(timing_bytes).hexdigest(),
        timing_semantic_sha256=_canonical_sha(_timing_semantic_payload(path)),
        note_count=len(beatmap.notes),
        hold_count=sum(note.kind == "HOLD" for note in beatmap.notes),
        interval_complete_group_count=interval_count,
    )


def quality_summary(
    path: Path,
    *,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    requested_key_mode: int,
    requested_difficulty: str,
    duration_ms: int,
    boundary_policy_mode: str,
) -> QualitySummary:
    beatmap = parse_osu_file(path)
    generated = GeneratedChart(
        notes=list(beatmap.notes),
        key_mode=beatmap.key_mode,
        osu_text=Path(path).read_text(encoding="utf-8-sig"),
        generator_name="required-gameplay-canary-independent-parser",
        seed=None,
        bpm_events=beatmap.bpm_events,
    )
    acceptance = evaluate_chart_candidate(
        generated,
        authority,
        onset_analysis,
        requested_key_mode=requested_key_mode,
        requested_difficulty=requested_difficulty,
        duration_ms=duration_ms,
        boundary_policy_mode=boundary_policy_mode,  # type: ignore[arg-type]
    )
    leading_gaps = tuple(
        gap
        for gap in acceptance.timing.coverage_gaps
        if gap.position == "LEADING"
    )
    difficulty = (
        acceptance.profile.difficulty
        if acceptance.profile is not None
        else None
    )
    return QualitySummary(
        action=acceptance.action.value,
        disposition=acceptance.disposition.value,
        first_note_time_ms=acceptance.timing.first_note_time_ms,
        active_leading_gap_count=len(leading_gaps),
        active_leading_gap_total_ms=sum(
            gap.end_ms - gap.start_ms for gap in leading_gaps
        ),
        matched_precision_50=acceptance.timing.overall.matched_precision_50,
        matched_f1_50=acceptance.timing.overall.matched_f1_50,
        project_rating=(
            float(difficulty.project_rating) if difficulty is not None else None
        ),
    )


def compare_canary_runs(
    off: ChartSummary | None,
    observe: ChartSummary | None,
    *,
    off_failure_class: str | None,
    observe_failure_class: str | None,
    off_exit_code: int,
    observe_exit_code: int,
    off_invocation_count: int,
    observe_invocation_count: int,
) -> CanaryComparison:
    mismatches: list[str] = []
    for label, chart, failure_class, exit_code, invocation_count in (
        (
            "OFF",
            off,
            off_failure_class,
            off_exit_code,
            off_invocation_count,
        ),
        (
            "OBSERVE",
            observe,
            observe_failure_class,
            observe_exit_code,
            observe_invocation_count,
        ),
    ):
        if exit_code != 0:
            mismatches.append(f"{label}_EXIT_CODE")
        if failure_class is not None:
            mismatches.append(f"{label}_FAILURE_CLASS")
        if invocation_count != 1:
            mismatches.append(f"{label}_INVOCATION_COUNT")
        if chart is None:
            mismatches.append(f"{label}_MISSING_CHART")
    if mismatches:
        return CanaryComparison(
            equivalent=False,
            classification="CANARY_EXECUTION_FAILURE",
            mismatches=tuple(mismatches),
        )

    assert off is not None
    assert observe is not None
    comparisons = (
        (off.osu_sha256 == observe.osu_sha256, "OSU_BYTES"),
        (off.semantic_sha256 == observe.semantic_sha256, "SEMANTIC_DIGEST"),
        (
            off.suffix_semantic_sha256 == observe.suffix_semantic_sha256,
            "SUFFIX_SEMANTIC_DIGEST",
        ),
        (off.note_count == observe.note_count, "NOTE_COUNT"),
        (off.hold_count == observe.hold_count, "HOLD_COUNT"),
        (
            off.timing_section_sha256 == observe.timing_section_sha256,
            "TIMING_SECTION_BYTES",
        ),
    )
    mismatches.extend(name for equal, name in comparisons if not equal)
    return CanaryComparison(
        equivalent=not mismatches,
        classification="EQUIVALENT" if not mismatches else "OBSERVE_INTERFERENCE",
        mismatches=tuple(mismatches),
    )


_REQUIRED_GAMEPLAY_FAILURE_REASONS = frozenset(
    {
        "REQUIRED_GAMEPLAY_INTERVAL_NOT_ADDRESSABLE",
        "REQUIRED_GAMEPLAY_INTERVAL_TOKEN_BUDGET_EXHAUSTED",
        "REQUIRED_GAMEPLAY_INTERVAL_NO_LEGAL_GROUP",
        "REQUIRED_GAMEPLAY_INTERVAL_UNSATISFIED_AT_CUT",
        "REQUIRED_GAMEPLAY_INTERVAL_ACCOUNTING_MISMATCH",
    }
)


def evaluate_shadow_run(
    *,
    baseline: ChartSummary,
    shadow: ChartSummary | None,
    reference_timing_semantic_sha256: str,
    source_quality: QualitySummary,
    shadow_quality: QualitySummary | None,
    diagnostics: GenerationOriginDiagnostics | None,
    minimum_complete_groups: int,
    exit_code: int,
    failure_class: str | None,
    failure_context: dict[str, object] | None,
    invocation_count: int,
) -> ShadowEvaluation:
    typed_reason = (
        failure_context.get("reason")
        if failure_class == "MANIA_REQUIRED_GAMEPLAY_FAILED"
        and type(failure_context) is dict
        else None
    )
    if (
        exit_code != 0
        and invocation_count == 1
        and typed_reason in _REQUIRED_GAMEPLAY_FAILURE_REASONS
    ):
        assert isinstance(typed_reason, str)
        return ShadowEvaluation(
            status="TYPED_FAILURE",
            isolation_pass=True,
            failures=(),
            typed_failure_reason=typed_reason,
        )

    failures: list[str] = []
    if exit_code != 0:
        failures.append("EXIT_CODE")
    if failure_class is not None:
        failures.append("FAILURE_CLASS")
    if invocation_count != 1:
        failures.append("INVOCATION_COUNT")
    if shadow is None:
        failures.append("MISSING_CHART")
    if diagnostics is None:
        failures.append("MISSING_ORIGIN_DIAGNOSTICS")
    if shadow is not None:
        if shadow.interval_complete_group_count < minimum_complete_groups:
            failures.append("INDEPENDENT_INTERVAL_GROUP_COUNT")
        if shadow.suffix_semantic_sha256 != baseline.suffix_semantic_sha256:
            failures.append("SUFFIX_SEMANTIC_DIGEST")
        if shadow.timing_semantic_sha256 != reference_timing_semantic_sha256:
            failures.append("REFERENCE_TIMING_SEMANTICS")
    if shadow_quality is None:
        failures.append("MISSING_SHADOW_QUALITY")
    else:
        action_rank = {"PASS": 0, "REVIEW": 1, "RETRY_MAP": 2}
        if shadow_quality.action == "RETRY_MAP":
            failures.append("QUALITY_GATE_RETRY")
        if action_rank.get(shadow_quality.action, 3) > action_rank.get(
            source_quality.action,
            3,
        ):
            failures.append("QUALITY_ACTION_REGRESSION")
        if shadow_quality.active_leading_gap_count != 0:
            failures.append("ACTIVE_LEADING_GAP")
        for source_value, shadow_value, reason in (
            (
                source_quality.matched_precision_50,
                shadow_quality.matched_precision_50,
                "MATCHED_PRECISION_50_REGRESSION",
            ),
            (
                source_quality.matched_f1_50,
                shadow_quality.matched_f1_50,
                "MATCHED_F1_50_REGRESSION",
            ),
        ):
            if source_value is None or shadow_value is None:
                failures.append(f"{reason}_UNAVAILABLE")
            elif shadow_value < source_value - 0.005:
                failures.append(reason)
        if (
            source_quality.project_rating is None
            or shadow_quality.project_rating is None
        ):
            failures.append("PROJECT_RATING_UNAVAILABLE")
        elif shadow_quality.project_rating + 1e-9 < source_quality.project_rating:
            failures.append("PROJECT_RATING_REGRESSION")
    if (
        diagnostics is not None
        and diagnostics.final_serialization.interval_generated_complete_groups
        < minimum_complete_groups
    ):
        failures.append("GENERATED_ORIGIN_INTERVAL_GROUP_COUNT")
    return ShadowEvaluation(
        status="CONTRACT_PASS" if not failures else "CONTRACT_FAIL",
        isolation_pass=not failures,
        failures=tuple(failures),
        typed_failure_reason=None,
    )


def classify_observation(
    diagnostics: GenerationOriginDiagnostics,
) -> RootCauseClassification:
    counts = (
        diagnostics.decoder.interval_generated_complete_groups,
        diagnostics.window_merge.interval_generated_complete_groups,
        diagnostics.canonical.interval_generated_complete_groups,
        diagnostics.resnap.interval_generated_complete_groups,
        diagnostics.final_serialization.interval_generated_complete_groups,
    )
    if counts[0] == 0 and counts == (0, 0, 0, 0, 0):
        return "CONTENT_OBLIGATION_UNSATISFIED"
    stage_map: dict[str, RootCauseClassification] = {
        "windowMerge": "WINDOW_MERGE_DELETION",
        "canonical": "CANONICAL_DELETION",
        "resnap": "RESNAP_DELETION",
        "finalSerialization": "SERIALIZATION_DELETION",
    }
    if diagnostics.first_loss_stage is not None:
        return stage_map.get(diagnostics.first_loss_stage, "ACCOUNTING_INCONCLUSIVE")
    final_count = diagnostics.final_serialization.interval_generated_complete_groups
    if all(value >= final_count for value in counts):
        return "NO_OBSERVED_LOSS"
    return "ACCOUNTING_INCONCLUSIVE"


def _exact_object(value: object, keys: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{name} has an invalid schema")
    return value


def _read_registration(path: Path) -> dict[str, object]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("registration must be an existing regular JSON file")
    root = _exact_object(
        json.loads(path.read_text(encoding="utf-8")),
        {"version", "input", "runtime", "request", "requiredGameplay"},
        "registration",
    )
    if root["version"] != REGISTRATION_VERSION:
        raise ValueError("unsupported canary registration version")
    return root


def _path(value: object, name: str) -> Path:
    if type(value) is not str:
        raise TypeError(f"{name} must be an explicit absolute path")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.exists():
        raise FileNotFoundError(f"{name} must be an existing absolute non-symlink path")
    return path.resolve()


def _require_hash(path: Path, expected: object, name: str) -> str:
    if type(expected) is not str or len(expected) != 64:
        raise ValueError(f"{name} expected SHA-256 is invalid")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _build_config(registration: dict[str, object]) -> WorkerConfig:
    runtime = _exact_object(
        registration["runtime"],
        {
            "homePath",
            "pythonPath",
            "modelRootPath",
            "modelManifestSha256",
            "modelRevision",
            "upstreamCommit",
            "patchSetId",
        },
        "runtime",
    )
    home = _path(runtime["homePath"], "runtime.homePath")
    python = _path(runtime["pythonPath"], "runtime.pythonPath")
    model_root = _path(runtime["modelRootPath"], "runtime.modelRootPath")
    manifest = model_root / "model-manifest-v1.json"
    _require_hash(manifest, runtime["modelManifestSha256"], "model manifest")
    if runtime["upstreamCommit"] != EXPECTED_MAPPERATORINATOR_HEAD:
        raise ValueError("runtime upstream commit differs from the pinned worker")
    if runtime["patchSetId"] != CONSTRAINT_PATCH_ID:
        raise ValueError("runtime patch set differs from the pinned worker")
    return WorkerConfig(
        chart_generator="mapperatorinator",
        mapperatorinator_home=home,
        mapperatorinator_python=python,
        mapperatorinator_precision="fp16",
        mapperatorinator_hold_state_mode="incremental",
        mapperatorinator_write_generation_telemetry=False,
        mapperatorinator_backend="song_session",
        mapperatorinator_model_root=model_root,
        mapperatorinator_model_revision=runtime["modelRevision"],
        mapperatorinator_tail_repairs=2,
        mapperatorinator_checkpoint_interval_windows=8,
    )


def _build_request_and_interval(
    registration: dict[str, object],
    config: WorkerConfig,
) -> tuple[GenerationRequest, RequiredGameplayIntervalV1, dict[str, str]]:
    inputs = _exact_object(
        registration["input"],
        {
            "audioPath",
            "audioSha256",
            "timingAuthorityPath",
            "timingAuthoritySha256",
            "sourceCandidatePath",
            "sourceCandidateSha256",
        },
        "input",
    )
    audio = _path(inputs["audioPath"], "input.audioPath")
    timing = _path(inputs["timingAuthorityPath"], "input.timingAuthorityPath")
    source = _path(inputs["sourceCandidatePath"], "input.sourceCandidatePath")
    hashes = {
        "audioSha256": _require_hash(audio, inputs["audioSha256"], "audio"),
        "timingAuthoritySha256": _require_hash(
            timing,
            inputs["timingAuthoritySha256"],
            "timing authority",
        ),
        "sourceCandidateSha256": _require_hash(
            source,
            inputs["sourceCandidateSha256"],
            "source candidate",
        ),
    }
    request = _exact_object(
        registration["request"],
        {
            "keyMode",
            "difficulty",
            "requestedStar",
            "cfgScale",
            "descriptors",
            "seed",
            "year",
            "durationMs",
            "musicEndMs",
            "generationEndMs",
            "lastAttackMs",
            "maxNoteStartMs",
            "partialStartMs",
            "partialEndMs",
        },
        "request",
    )
    required = _exact_object(
        registration["requiredGameplay"],
        {"evidence", "expectedInterval"},
        "requiredGameplay",
    )
    evidence_payload = _exact_object(
        required["evidence"],
        {
            "anchorStatus",
            "anchorMs",
            "anchorGridMs",
            "aggregateRank",
            "prominentBandCount",
            "pulseSupportCount",
            "familySlots",
            "localAudioSupported",
            "referenceFirstRowSupported",
            "repeatedHighConfidenceRefusal",
            "timingAuthorityValid",
            "anchorEvidenceDigest",
        },
        "requiredGameplay.evidence",
    )
    family_slots_raw = evidence_payload["familySlots"]
    if type(family_slots_raw) is not list:
        raise TypeError("familySlots must be a list")
    family_slots = tuple(
        RequiredGameplayFamilySlotV1(
            key_mode=item["keyMode"],
            difficulty=item["difficulty"],
            supports_anchor=item["supportsAnchor"],
        )
        for raw in family_slots_raw
        for item in (
            _exact_object(
                raw,
                {"keyMode", "difficulty", "supportsAnchor"},
                "familySlots[]",
            ),
        )
    )
    evidence = RequiredGameplayEvidenceV1(
        anchor_status=evidence_payload["anchorStatus"],
        anchor_ms=evidence_payload["anchorMs"],
        anchor_grid_ms=evidence_payload["anchorGridMs"],
        aggregate_rank=evidence_payload["aggregateRank"],
        prominent_band_count=evidence_payload["prominentBandCount"],
        pulse_support_count=evidence_payload["pulseSupportCount"],
        family_slots=family_slots,
        local_audio_supported=evidence_payload["localAudioSupported"],
        reference_first_row_supported=evidence_payload["referenceFirstRowSupported"],
        repeated_high_confidence_refusal=evidence_payload[
            "repeatedHighConfidenceRefusal"
        ],
        timing_authority_valid=evidence_payload["timingAuthorityValid"],
        timing_authority_digest=hashes["timingAuthoritySha256"],
        anchor_evidence_digest=evidence_payload["anchorEvidenceDigest"],
    )
    source_beatmap = parse_osu_file(source)
    rows = sorted({note.time_ms for note in source_beatmap.notes})
    if len(rows) < 2:
        raise ValueError("source candidate does not contain two distinct rows")
    partial_window = PartialRemapWindow(
        start_ms=request["partialStartMs"],
        end_ms=request["partialEndMs"],
    )
    decision = plan_required_gameplay_interval(
        evidence,
        partial_window=partial_window,
        bpm_events=source_beatmap.bpm_events,
        second_distinct_row_ms=rows[1],
        duration_ms=request["durationMs"],
        mode=RequiredGameplayIntervalMode.OBSERVE,
    )
    if decision.interval is None:
        raise ValueError(f"registered evidence is ineligible: {decision.reason}")
    interval = decision.interval
    expected_interval = _exact_object(
        required["expectedInterval"],
        {"startMs", "endMs", "evidenceClass", "evidenceDigest"},
        "requiredGameplay.expectedInterval",
    )
    actual_interval = {
        "startMs": interval.start_ms,
        "endMs": interval.end_ms,
        "evidenceClass": interval.evidence_class.value,
        "evidenceDigest": interval.evidence_digest,
    }
    if expected_interval != actual_interval:
        raise ValueError("planned required interval differs from frozen registration")
    base = GenerationRequest(
        audio_path=audio,
        timing_reference_path=source,
        key_mode=request["keyMode"],
        difficulty=request["difficulty"],
        requested_star=request["requestedStar"],
        cfg_scale=request["cfgScale"],
        descriptors=tuple(request["descriptors"]),
        seed=request["seed"],
        year=request["year"],
        duration_ms=request["durationMs"],
        music_end_ms=request["musicEndMs"],
        generation_end_ms=request["generationEndMs"],
        last_attack_ms=request["lastAttackMs"],
        max_note_start_ms=request["maxNoteStartMs"],
        partial_start_ms=request["partialStartMs"],
        partial_end_ms=request["partialEndMs"],
        add_to_beatmap=True,
    )
    observed = replace(base, required_gameplay_interval=interval)
    # Force both independent implementations to evaluate before model startup.
    required_gameplay_invocation_digest(config, observed)
    return base, interval, hashes


def _terminal_evidence(workdir: Path) -> tuple[int, int]:
    terminals = sorted(workdir.rglob("resident-terminal-v1.json"))
    if not terminals:
        return 0, 1
    if len(terminals) != 1:
        raise ValueError("one canary run emitted multiple resident terminals")
    payload = json.loads(terminals[0].read_text(encoding="utf-8"))
    value = payload.get("returncode", 1)
    if type(value) is not int:
        raise ValueError("resident terminal returncode is invalid")
    return 1, value


def _stage_registered_inputs(
    registration: dict[str, object],
    request: GenerationRequest,
    output_root: Path,
    hashes: dict[str, str],
) -> GenerationRequest:
    """Place immutable inputs beneath the resident job root without changing identity."""

    inputs = registration["input"]
    assert type(inputs) is dict
    timing_authority = _path(inputs["timingAuthorityPath"], "input.timingAuthorityPath")
    staged_root = output_root / "input"
    staged_root.mkdir(parents=True, exist_ok=False)
    staged_audio = staged_root / "audio.flac"
    staged_source = staged_root / "source-candidate.osu"
    staged_timing = staged_root / "timing-authority.osu"
    for source, destination, expected, name in (
        (request.audio_path, staged_audio, hashes["audioSha256"], "staged audio"),
        (
            request.timing_reference_path,
            staged_source,
            hashes["sourceCandidateSha256"],
            "staged source candidate",
        ),
        (
            timing_authority,
            staged_timing,
            hashes["timingAuthoritySha256"],
            "staged timing authority",
        ),
    ):
        shutil.copyfile(source, destination)
        _require_hash(destination, expected, name)
    return replace(
        request,
        audio_path=staged_audio,
        timing_reference_path=staged_source,
    )


def _run_one(
    *,
    label: Literal["OFF", "OBSERVE", "SHADOW"],
    config: WorkerConfig,
    request: GenerationRequest,
    interval: RequiredGameplayIntervalV1,
    output_root: Path,
    audio_sha256: str,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
) -> _RunResult:
    workdir = output_root / label.lower()
    started = datetime.now(UTC)
    before = time.perf_counter()
    failure_class: str | None = None
    failure_message: str | None = None
    failure_context: dict[str, object] | None = None
    chart: ChartSummary | None = None
    quality: QualitySummary | None = None
    diagnostics: GenerationOriginDiagnostics | None = None
    try:
        session = _open_inference_session(
            config,
            output_root,
            stderr_path=output_root / f"{label.lower()}-resident-stderr.log",
        )
        identity = SongIdentity(
            song_id="required-gameplay-canary",
            audio_sha256=audio_sha256,
            config_digest=_song_config_digest(config),
        )
        with inference_song_scope(session, identity, close_on_exit=True):
            result = MapperatorinatorGenerator(config, session=session).generate_map(
                request,
                workdir,
            )
        osu_path = find_generated_osu(workdir)
        chart = canonical_chart_summary(
            osu_path,
            interval_start_ms=interval.start_ms,
            interval_end_ms=interval.end_ms,
            partial_end_ms=request.partial_end_ms or 0,
        )
        quality = quality_summary(
            osu_path,
            authority=authority,
            onset_analysis=onset_analysis,
            requested_key_mode=request.key_mode,
            requested_difficulty=request.difficulty,
            duration_ms=request.duration_ms,
            boundary_policy_mode=config.boundary_policy_mode,
        )
        diagnostics = result.origin_diagnostics
    except WorkerError as error:
        failure_class = error.code.value
        failure_message = str(error)
        failure_context = dict(error.context)
    except Exception as error:  # noqa: BLE001 - the report must freeze unexpected failures
        failure_class = type(error).__name__
        failure_message = str(error)
    invocation_count, exit_code = _terminal_evidence(workdir)
    ended = datetime.now(UTC)
    return _RunResult(
        label=label,
        started_at=started.isoformat(),
        ended_at=ended.isoformat(),
        wall_ms=round((time.perf_counter() - before) * 1000),
        exit_code=exit_code,
        failure_class=failure_class,
        failure_message=failure_message,
        failure_context=failure_context,
        invocation_count=invocation_count,
        chart=chart,
        quality=quality,
        diagnostics=diagnostics,
    )


def _diagnostics_report(value: GenerationOriginDiagnostics | None) -> object:
    if value is None:
        return None
    return asdict(value)


def run_canary(
    registration_path: Path,
    output_root: Path,
    *,
    include_shadow: bool = False,
    shadow_first: bool = False,
) -> dict[str, object]:
    registration_path = Path(registration_path)
    if registration_path.is_symlink() or not registration_path.is_file():
        raise FileNotFoundError("registration must be an existing regular JSON file")
    registration_sha = sha256_file(registration_path)
    output_root = Path(output_root).resolve()
    report_version = SHADOW_REPORT_VERSION if include_shadow else REPORT_VERSION
    report_path = output_root / _report_filename(include_shadow=include_shadow)
    cached_report = _read_cached_report(
        report_path,
        expected_version=report_version,
        registration_sha256=registration_sha,
    )

    registration = _read_registration(registration_path)
    config = _build_config(registration)
    base_request, interval, hashes = _build_request_and_interval(registration, config)
    runtime_identity = {
        "upstreamCommit": EXPECTED_MAPPERATORINATOR_HEAD,
        "patchSetId": CONSTRAINT_PATCH_ID,
        "modelRevision": config.mapperatorinator_model_revision,
        "modelManifestSha256": sha256_file(
            config.mapperatorinator_model_root / "model-manifest-v1.json"  # type: ignore[operator]
        ),
    }
    if cached_report is not None:
        if cached_report.get("frozenInputs") != hashes:
            raise ValueError("cached canary report frozen inputs differ from registration")
        if cached_report.get("runtime") != runtime_identity:
            raise ValueError("cached canary report runtime identity differs from registration")
        return cached_report
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("output root is non-empty and has no completed canary report")
    output_root.mkdir(parents=True, exist_ok=True)

    base_request = _stage_registered_inputs(
        registration,
        base_request,
        output_root,
        hashes,
    )
    source = canonical_chart_summary(
        base_request.timing_reference_path,
        interval_start_ms=interval.start_ms,
        interval_end_ms=interval.end_ms,
        partial_end_ms=base_request.partial_end_ms or 0,
    )
    timing_authority_path = output_root / "input" / "timing-authority.osu"
    timing_authority_beatmap = parse_osu_file(timing_authority_path)
    authority = SongTimingAuthority(
        reference_path=timing_authority_path,
        sha256=hashes["timingAuthoritySha256"],
        audio_sha256=hashes["audioSha256"],
        bpm_events=timing_authority_beatmap.bpm_events,
        generator_name="frozen-required-gameplay-canary",
        seed=None,
        mode="STANDARD",
        attempt_count=1,
    )
    onset_analysis = analyze_canonical_audio(base_request.audio_path)
    source_quality = quality_summary(
        base_request.timing_reference_path,
        authority=authority,
        onset_analysis=onset_analysis,
        requested_key_mode=base_request.key_mode,
        requested_difficulty=base_request.difficulty,
        duration_ms=base_request.duration_ms,
        boundary_policy_mode=config.boundary_policy_mode,
    )
    observed_request = replace(base_request, required_gameplay_interval=interval)
    semantic_request = required_gameplay_invocation_payload(config, observed_request)
    shadow_interval = replace(
        interval,
        mode=RequiredGameplayIntervalMode.SHADOW_ENFORCE,
    )
    shadow_request = replace(
        base_request,
        required_gameplay_interval=shadow_interval,
    )
    shadow_semantic_request = (
        required_gameplay_invocation_payload(config, shadow_request)
        if include_shadow
        else None
    )
    shadow: _RunResult | None = None
    if include_shadow and shadow_first:
        shadow = _run_one(
            label="SHADOW",
            config=config,
            request=shadow_request,
            interval=shadow_interval,
            output_root=output_root,
            audio_sha256=hashes["audioSha256"],
            authority=authority,
            onset_analysis=onset_analysis,
        )
    off = _run_one(
        label="OFF",
        config=config,
        request=base_request,
        interval=interval,
        output_root=output_root,
        audio_sha256=hashes["audioSha256"],
        authority=authority,
        onset_analysis=onset_analysis,
    )
    observe = _run_one(
        label="OBSERVE",
        config=config,
        request=observed_request,
        interval=interval,
        output_root=output_root,
        audio_sha256=hashes["audioSha256"],
        authority=authority,
        onset_analysis=onset_analysis,
    )
    if include_shadow and shadow is None:
        shadow = _run_one(
            label="SHADOW",
            config=config,
            request=shadow_request,
            interval=shadow_interval,
            output_root=output_root,
            audio_sha256=hashes["audioSha256"],
            authority=authority,
            onset_analysis=onset_analysis,
        )

    comparison = compare_canary_runs(
        off.chart,
        observe.chart,
        off_failure_class=off.failure_class,
        observe_failure_class=observe.failure_class,
        off_exit_code=off.exit_code,
        observe_exit_code=observe.exit_code,
        off_invocation_count=off.invocation_count,
        observe_invocation_count=observe.invocation_count,
    )

    root_cause: str
    independent_final_consistent: bool | None = None
    if comparison.classification == "CANARY_EXECUTION_FAILURE":
        root_cause = "CANARY_EXECUTION_FAILURE"
    elif not comparison.equivalent:
        root_cause = "OBSERVE_INTERFERENCE"
    elif observe.diagnostics is None or observe.chart is None:
        root_cause = "ACCOUNTING_INCONCLUSIVE"
    else:
        final_origin_count = (
            observe.diagnostics.final_serialization.interval_generated_complete_groups
        )
        independent_final_consistent = (
            final_origin_count <= observe.chart.interval_complete_group_count
        )
        root_cause = (
            classify_observation(observe.diagnostics)
            if independent_final_consistent
            else "ACCOUNTING_INCONCLUSIVE"
        )

    report = {
        "version": report_version,
        "registrationSha256": registration_sha,
        "frozenInputs": hashes,
        "runtime": runtime_identity,
        "semanticRequest": semantic_request,
        "semanticRequestSha256": _canonical_sha(semantic_request),
        "shadowSemanticRequest": shadow_semantic_request,
        "shadowSemanticRequestSha256": (
            _canonical_sha(shadow_semantic_request)
            if shadow_semantic_request is not None
            else None
        ),
        "evidenceDigest": interval.evidence_digest,
        "interval": {
            "startMs": interval.start_ms,
            "endMs": interval.end_ms,
            "minimumCompleteGroups": interval.minimum_complete_groups,
            "allowedGroupTypes": [item.value for item in interval.allowed_group_types],
            "evidenceClass": interval.evidence_class.value,
            "mode": interval.mode.value,
        },
        "source": asdict(source),
        "sourceQuality": asdict(source_quality),
        "off": {**asdict(off), "diagnostics": _diagnostics_report(off.diagnostics)},
        "observe": {
            **asdict(observe),
            "diagnostics": _diagnostics_report(observe.diagnostics),
        },
        "shadow": (
            {**asdict(shadow), "diagnostics": _diagnostics_report(shadow.diagnostics)}
            if shadow is not None
            else None
        ),
        "shadowEvaluation": (
            asdict(
                evaluate_shadow_run(
                    baseline=off.chart,
                    shadow=shadow.chart,
                    reference_timing_semantic_sha256=(
                        source.timing_semantic_sha256
                    ),
                    source_quality=source_quality,
                    shadow_quality=shadow.quality,
                    diagnostics=shadow.diagnostics,
                    minimum_complete_groups=shadow_interval.minimum_complete_groups,
                    exit_code=shadow.exit_code,
                    failure_class=shadow.failure_class,
                    failure_context=shadow.failure_context,
                    invocation_count=shadow.invocation_count,
                )
            )
            if shadow is not None and off.chart is not None
            else None
        ),
        "comparison": asdict(comparison),
        "independentFinalCountConsistent": independent_final_consistent,
        "rootCauseClassification": root_cause,
        "authorizedNextPlan": {
            "CONTENT_OBLIGATION_UNSATISFIED": "PHASE_B_D_DECODER_FSM_SHADOW_ONLY",
            "WINDOW_MERGE_DELETION": "PHASE_B_M_WINDOW_MERGE_ONLY",
            "CANONICAL_DELETION": "PHASE_B_C_CANONICAL_ONLY",
            "RESNAP_DELETION": "PHASE_B_R_RESNAP_IDENTITY_ONLY",
            "SERIALIZATION_DELETION": "PHASE_B_S_SERIALIZATION_ONLY",
            "NO_OBSERVED_LOSS": "NO_REPAIR_REQUIRED_FROM_THIS_CANARY",
            "ACCOUNTING_INCONCLUSIVE": "EXPAND_TELEMETRY_NO_ENFORCEMENT",
            "OBSERVE_INTERFERENCE": "DISABLE_V35_NO_ENFORCEMENT",
            "CANARY_EXECUTION_FAILURE": "FIX_EXECUTION_ENVIRONMENT_NO_INFERENCE",
        }[root_cause],
    }
    _write_report_atomic(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--include-shadow", action="store_true")
    parser.add_argument("--shadow-first", action="store_true")
    args = parser.parse_args()
    if args.shadow_first and not args.include_shadow:
        parser.error("--shadow-first requires --include-shadow")
    report = run_canary(
        args.registration,
        args.output_root,
        include_shadow=args.include_shadow,
        shadow_first=args.shadow_first,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["rootCauseClassification"] == "CANARY_EXECUTION_FAILURE":
        return 4
    if report["rootCauseClassification"] == "OBSERVE_INTERFERENCE":
        return 2
    shadow_evaluation = report.get("shadowEvaluation")
    if isinstance(shadow_evaluation, dict) and (
        shadow_evaluation.get("status") != "CONTRACT_PASS"
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
