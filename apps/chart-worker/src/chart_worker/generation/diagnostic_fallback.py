"""Isolated PLAYTEST_ONLY export of hard-safe, quality-rejected model output."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from chart_worker.schema.types import DIFFICULTIES, KEY_MODES

DIAGNOSTIC_FALLBACK_VERSION = "diagnostic-raw-fallback-v1"


def _canonical_json(value: object, field_name: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError(f"{field_name} must be finite JSON data") from error


def _exact_positive_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an exact integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _exact_non_negative_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _require_hard_pass(gate_report: Mapping[str, object]) -> None:
    decisions = gate_report.get("decisions")
    if not isinstance(decisions, Mapping):
        raise TypeError("gate report decisions must be a mapping")
    try:
        structure = decisions["STRUCTURE"]
        timing_identity = decisions["TIMING_IDENTITY"]
        song_bounds = decisions["SONG_BOUNDS"]
    except KeyError as error:
        raise ValueError("gate report is missing a hard decision") from error
    if (
        not isinstance(structure, Mapping)
        or not isinstance(timing_identity, Mapping)
        or not isinstance(song_bounds, Mapping)
    ):
        raise TypeError("hard gate decisions must be mappings")
    if (
        structure.get("action") != "PASS"
        or timing_identity.get("action") != "PASS"
        or song_bounds.get("action") != "PASS"
    ):
        raise ValueError("STRUCTURE, TIMING_IDENTITY, and SONG_BOUNDS must all PASS")


@dataclass(frozen=True, slots=True)
class DiagnosticRawCandidate:
    key_mode: int
    difficulty: str
    seed: int
    attempt: int
    osu_text: str
    source_workdir: Path
    attempt_errors: tuple[str, ...]
    _gate_report_json: str = field(repr=False)
    _attempt_evidence_json: str = field(repr=False)

    @classmethod
    def create(
        cls,
        *,
        key_mode: int,
        difficulty: str,
        seed: int,
        attempt: int,
        osu_text: str,
        source_workdir: Path,
        gate_report: Mapping[str, object],
        attempt_errors: Iterable[str],
        attempt_evidence: Iterable[Mapping[str, object]],
    ) -> DiagnosticRawCandidate:
        key_mode = _exact_positive_int(key_mode, "key_mode")
        if key_mode not in KEY_MODES:
            raise ValueError(f"unsupported key_mode: {key_mode}")
        if type(difficulty) is not str or difficulty not in DIFFICULTIES:
            raise ValueError(f"unsupported difficulty: {difficulty!r}")
        seed = _exact_non_negative_int(seed, "seed")
        attempt = _exact_positive_int(attempt, "attempt")
        if type(osu_text) is not str or not osu_text:
            raise ValueError("osu_text must be a non-empty exact string")
        if type(source_workdir) is not Path:
            source_workdir = Path(source_workdir)
        errors = tuple(attempt_errors)
        if any(type(error) is not str for error in errors):
            raise TypeError("attempt_errors must contain exact strings")
        gate_json = _canonical_json(gate_report, "gate_report")
        gate_projection = json.loads(gate_json)
        _require_hard_pass(gate_projection)
        evidence_json = _canonical_json(list(attempt_evidence), "attempt_evidence")
        return cls(
            key_mode=key_mode,
            difficulty=difficulty,
            seed=seed,
            attempt=attempt,
            osu_text=osu_text,
            source_workdir=source_workdir,
            attempt_errors=errors,
            _gate_report_json=gate_json,
            _attempt_evidence_json=evidence_json,
        )

    def gate_report(self) -> dict[str, object]:
        return json.loads(self._gate_report_json)

    def attempt_evidence(self) -> list[dict[str, object]]:
        return json.loads(self._attempt_evidence_json)

    def selection_score(self) -> tuple[int, int, float, int, int]:
        report = self.gate_report()
        decisions = report["decisions"]
        if not isinstance(decisions, dict):
            raise TypeError("stored gate report decisions must be a mapping")
        noncoverage_retry_count = sum(
            decision.get("action") == "RETRY_MAP"
            for axis, decision in decisions.items()
            if axis not in {"STRUCTURE", "TIMING_IDENTITY", "COVERAGE"}
            and isinstance(decision, dict)
        )
        timing = report.get("timing")
        timing = timing if isinstance(timing, dict) else {}
        gaps = timing.get("coverageGaps")
        gaps = gaps if isinstance(gaps, list) else []
        attack_gap_count = sum(
            isinstance(gap, dict)
            and isinstance(gap.get("opportunity"), dict)
            and gap["opportunity"].get("kind") == "ATTACK_REQUIRED"
            for gap in gaps
        )
        overall = timing.get("overall")
        precision = overall.get("precision50") if isinstance(overall, dict) else None
        precision_score = -float(precision) if type(precision) in {int, float} else 0.0
        return (
            noncoverage_retry_count,
            attack_gap_count,
            precision_score,
            self.attempt,
            self.seed,
        )


@dataclass(frozen=True, slots=True)
class DiagnosticFallbackIdentity:
    audio_sha256: str
    timing_sha256: str
    model_identity: str
    patch_set_id: str
    hold_state_mode: str

    def __post_init__(self) -> None:
        for field_name in ("audio_sha256", "timing_sha256"):
            value = getattr(self, field_name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA-256")
        for field_name in ("model_identity", "patch_set_id"):
            value = getattr(self, field_name)
            if type(value) is not str or not value:
                raise ValueError(f"{field_name} must be a non-empty exact string")
        if self.hold_state_mode != "incremental":
            raise ValueError("diagnostic fallback requires incremental hold-state mode")

    def to_report(self) -> dict[str, str]:
        return {
            "audioSha256": self.audio_sha256,
            "timingSha256": self.timing_sha256,
            "modelIdentity": self.model_identity,
            "patchSetId": self.patch_set_id,
            "holdStateMode": self.hold_state_mode,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticFallbackExport:
    key_mode: int
    difficulty: str
    path: Path
    sha256: str
    manifest_path: Path

    def to_report(self, *, relative_to: Path) -> dict[str, object]:
        return {
            "keyMode": self.key_mode,
            "difficulty": self.difficulty,
            "path": self.path.relative_to(relative_to).as_posix(),
            "sha256": self.sha256,
            "manifestPath": self.manifest_path.relative_to(relative_to).as_posix(),
        }


def select_diagnostic_candidate(
    candidates: Iterable[DiagnosticRawCandidate],
    *,
    key_mode: int,
    difficulty: str,
) -> DiagnosticRawCandidate:
    materialized = tuple(candidates)
    if not materialized:
        raise ValueError("at least one diagnostic candidate is required")
    if any(
        candidate.key_mode != key_mode or candidate.difficulty != difficulty
        for candidate in materialized
    ):
        raise ValueError("all diagnostic candidates must match the requested variant")
    return min(materialized, key=DiagnosticRawCandidate.selection_score)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"conflicting diagnostic temporary file: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_output_directory(path: Path, *, run_dir: Path) -> None:
    """Validate an existing parent before any child creation.

    Resolving only after ``mkdir`` is too late for a junction/reparse point:
    creating the child may already have mutated a directory outside the run.
    """
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"diagnostic output path is not a directory: {path}")
        if not _is_within(path.resolve(), run_dir):
            raise ValueError("diagnostic output directory escapes run_dir")
        return
    parent = path.parent
    if not parent.is_dir() or not _is_within(parent.resolve(), run_dir):
        raise ValueError("diagnostic output parent escapes run_dir")
    path.mkdir()
    if not _is_within(path.resolve(), run_dir):
        raise ValueError("diagnostic output directory escapes run_dir")


def export_diagnostic_fallback(
    candidate: DiagnosticRawCandidate,
    *,
    run_dir: Path,
    identity: DiagnosticFallbackIdentity,
    validate_osu: Callable[[str], None],
    output_root_name: str = "diagnostic-raw-fallback",
) -> DiagnosticFallbackExport:
    if type(output_root_name) is not str:
        raise TypeError("output_root_name must be an exact string")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", output_root_name) is None:
        raise ValueError("output_root_name must be a safe lowercase directory name")
    run_dir = run_dir.resolve()
    source_workdir = candidate.source_workdir.resolve()
    if not _is_within(source_workdir, run_dir):
        raise ValueError("source_workdir must be inside run_dir")
    validate_osu(candidate.osu_text)

    output_root = run_dir / output_root_name
    _prepare_output_directory(output_root, run_dir=run_dir)
    variant_dir = output_root / f"{candidate.key_mode}k-{candidate.difficulty.lower()}"
    _prepare_output_directory(variant_dir, run_dir=run_dir)
    map_path = variant_dir / "map.osu"
    manifest_path = variant_dir / "manifest-v1.json"
    map_bytes = candidate.osu_text.encode("utf-8")
    map_sha256 = hashlib.sha256(map_bytes).hexdigest()
    manifest = {
        "version": DIAGNOSTIC_FALLBACK_VERSION,
        "decision": "PLAYTEST_ONLY",
        "keyMode": candidate.key_mode,
        "difficulty": candidate.difficulty,
        "seed": candidate.seed,
        "attempt": candidate.attempt,
        "sourceWorkdir": source_workdir.relative_to(run_dir).as_posix(),
        "osuPath": map_path.relative_to(run_dir).as_posix(),
        "osuSha256": map_sha256,
        "identity": identity.to_report(),
        "selectionScore": list(candidate.selection_score()),
        "gateReport": candidate.gate_report(),
        "attemptErrors": list(candidate.attempt_errors),
        "attemptEvidence": candidate.attempt_evidence(),
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")

    if map_path.exists() and map_path.read_bytes() != map_bytes:
        raise FileExistsError(f"conflicting diagnostic map already exists: {map_path}")
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise FileExistsError(
            f"conflicting diagnostic manifest already exists: {manifest_path}"
        )
    if not map_path.exists():
        _atomic_write(map_path, map_bytes)
    if not manifest_path.exists():
        _atomic_write(manifest_path, manifest_bytes)
    return DiagnosticFallbackExport(
        key_mode=candidate.key_mode,
        difficulty=candidate.difficulty,
        path=map_path,
        sha256=map_sha256,
        manifest_path=manifest_path,
    )
