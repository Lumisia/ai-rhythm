"""Modal resident chart-worker entrypoint.

This module defines the reviewed resource/lifecycle contract only. Deployment
still requires an immutable OCI image containing chart-worker plus the verified
v27 Mapperatorinator source, two pre-created Volumes, an immutable model
revision, and an external Starter-workspace spend alarm.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from chart_worker.config import WorkerConfig
from chart_worker.generation.inference_session import InferenceSession
from chart_worker.modal_budget import MODAL_MAX_INPUT_SECONDS, reserve_modal_budget
from chart_worker.modal_request import request_output_dir
from chart_worker.pipeline import (
    PipelineDependencies,
    PipelineOptions,
    _open_inference_session,
    run_pipeline,
)

APP_NAME = "ai-rhythm-resident-chart-worker"
MAPPERATORINATOR_HOME = Path("/opt/mapperatorinator")
MODEL_ROOT = Path("/models/snapshot")
JOBS_ROOT = Path("/jobs")
RUNS_ROOT = JOBS_ROOT / "runs"
RESIDENT_STDERR = Path("/tmp/ai-rhythm-resident-stderr.log")
SOFT_BUDGET_USD = 24
WORKSPACE_CREDIT_USD = 30


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if type(value) is not str or not value:
        raise RuntimeError(f"{name} must be configured before Modal app import")
    return value


def _immutable_image_reference() -> str:
    value = _required_environment("AI_RHYTHM_MODAL_IMAGE")
    marker = "@sha256:"
    if value.count(marker) != 1:
        raise RuntimeError("AI_RHYTHM_MODAL_IMAGE must be an OCI reference pinned by sha256")
    digest = value.rsplit(marker, 1)[1]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError("AI_RHYTHM_MODAL_IMAGE has an invalid sha256 digest")
    return value


def _validate_model_revision(value: object) -> str:
    if type(value) is not str or len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError("model revision must be an exact lowercase 40-character Git SHA")
    return value


def _immutable_model_revision() -> str:
    return _validate_model_revision(
        _required_environment("AI_RHYTHM_MAPPERATORINATOR_MODEL_REVISION")
    )


image = modal.Image.from_registry(_immutable_image_reference())
model_volume = modal.Volume.from_name(
    os.environ.get("AI_RHYTHM_MODAL_MODEL_VOLUME", "ai-rhythm-models"),
    create_if_missing=False,
)
jobs_volume = modal.Volume.from_name(
    os.environ.get("AI_RHYTHM_MODAL_JOBS_VOLUME", "ai-rhythm-jobs"),
    create_if_missing=False,
)
app = modal.App(APP_NAME)


def _strict_request(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "requestId",
        "source",
        "title",
        "seed",
        "workerVersion",
    }:
        raise ValueError("Modal song request has an invalid schema")
    request_id = value["requestId"]
    source = value["source"]
    title = value["title"]
    seed = value["seed"]
    worker_version = value["workerVersion"]
    if type(request_id) is not str or not request_id or len(request_id) > 128:
        raise ValueError("requestId must be a non-empty plain string <=128 chars")
    if type(source) is not str or not source or "\\" in source:
        raise ValueError("source must be a non-empty POSIX relative path")
    relative = PurePosixPath(source)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError("source must stay beneath the jobs Volume")
    if type(title) is not str or not title.strip():
        raise ValueError("title must be a non-empty plain string")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an exact uint32")
    if type(worker_version) is not str or not worker_version:
        raise ValueError("workerVersion must be a non-empty plain string")
    return value


def _source_path(relative: str) -> Path:
    root = JOBS_ROOT.resolve(strict=True)
    candidate = (root / relative).resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file() or candidate.is_symlink():
        raise ValueError("source must resolve to a regular non-symlink file beneath /jobs")
    return candidate


def _relative_artifact(path: Path) -> str:
    resolved = path.resolve(strict=True)
    root = JOBS_ROOT.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise RuntimeError("pipeline artifact escaped the jobs Volume")
    return resolved.relative_to(root).as_posix()


@app.cls(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16_384,
    volumes={
        str(MODEL_ROOT.parent): model_volume.with_mount_options(read_only=True),
        str(JOBS_ROOT): jobs_volume,
    },
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=10_800,
    startup_timeout=1_800,
    retries=0,
    block_network=True,
)
@modal.concurrent(max_inputs=1, target_inputs=1)
class ResidentChartWorker:
    model_revision: str = modal.parameter()
    session: InferenceSession
    config: WorkerConfig

    @modal.enter()
    def start_container(self) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if not MAPPERATORINATOR_HOME.is_dir():
            raise RuntimeError("immutable image is missing /opt/mapperatorinator")
        if not MODEL_ROOT.is_dir():
            raise RuntimeError("model Volume is missing /models/snapshot")
        JOBS_ROOT.mkdir(parents=True, exist_ok=True)
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        self.config = WorkerConfig(
            mapperatorinator_backend="song_session",
            mapperatorinator_hold_state_mode="incremental",
            mapperatorinator_home=MAPPERATORINATOR_HOME,
            mapperatorinator_python=Path(sys.executable),
            mapperatorinator_model_root=MODEL_ROOT,
            mapperatorinator_model_revision=_validate_model_revision(self.model_revision),
            mapperatorinator_tail_repairs=2,
            mapperatorinator_checkpoint_interval_windows=8,
            mapperatorinator_resident_startup_timeout_sec=1_800,
            mapperatorinator_resident_invocation_timeout_sec=10_800,
            mapperatorinator_resident_close_timeout_sec=5,
            storage_local_root=JOBS_ROOT / "storage",
        )
        self.session = _open_inference_session(
            self.config,
            JOBS_ROOT,
            stderr_path=RESIDENT_STDERR,
        )

    @modal.method()
    def run_song(self, request: dict[str, Any]) -> dict[str, object]:
        jobs_volume.reload()
        validated = _strict_request(request)
        source = _source_path(validated["source"])
        dependencies = PipelineDependencies(
            config=self.config,
            attached_inference_session=self.session,
        )
        result = run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=request_output_dir(RUNS_ROOT, validated["requestId"]),
                title=validated["title"],
                generator="mapperatorinator",
                seed=validated["seed"],
                worker_version=validated["workerVersion"],
            ),
            dependencies=dependencies,
        )
        jobs_volume.commit()
        manifest = result.manifest_path.read_bytes()
        return {
            "requestId": validated["requestId"],
            "runId": str(result.run_id),
            "manifest": _relative_artifact(result.manifest_path),
            "manifestSha256": hashlib.sha256(manifest).hexdigest(),
            "charts": [_relative_artifact(path) for path in result.chart_paths],
        }

    @modal.exit()
    def stop_container(self) -> None:
        session = getattr(self, "session", None)
        if session is not None:
            session.close()


@app.local_entrypoint()
def main(
    request_json: str,
    budget_ledger: str,
    worst_case_seconds: int = MODAL_MAX_INPUT_SECONDS,
) -> None:
    request = json.loads(request_json)
    validated = _strict_request(request)
    reserve_modal_budget(
        Path(budget_ledger).resolve(),
        token=validated["requestId"],
        worst_case_seconds=worst_case_seconds,
    )
    result = ResidentChartWorker(model_revision=_immutable_model_revision()).run_song.remote(
        validated
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
