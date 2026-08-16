import ast
from pathlib import Path

import pytest

from chart_worker.modal_request import request_output_dir

APP_PATH = Path(__file__).resolve().parents[2] / "modal_app.py"


def _call_keyword(call: ast.Call, name: str):
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def test_modal_class_has_fail_closed_single_l4_container_contract():
    module = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    worker = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ResidentChartWorker"
    )
    cls_call = next(
        decorator
        for decorator in worker.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "cls"
    )
    concurrent_call = next(
        decorator
        for decorator in worker.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "concurrent"
    )

    assert ast.literal_eval(_call_keyword(cls_call, "gpu")) == "L4"
    assert ast.literal_eval(_call_keyword(cls_call, "cpu")) == 4
    assert ast.literal_eval(_call_keyword(cls_call, "memory")) == 16_384
    assert ast.literal_eval(_call_keyword(cls_call, "min_containers")) == 0
    assert ast.literal_eval(_call_keyword(cls_call, "max_containers")) == 1
    assert ast.literal_eval(_call_keyword(cls_call, "scaledown_window")) == 60
    assert ast.literal_eval(_call_keyword(cls_call, "timeout")) == 10_800
    assert ast.literal_eval(_call_keyword(cls_call, "startup_timeout")) == 1_800
    assert ast.literal_eval(_call_keyword(cls_call, "retries")) == 0
    assert ast.literal_eval(_call_keyword(concurrent_call, "max_inputs")) == 1
    assert ast.literal_eval(_call_keyword(concurrent_call, "target_inputs")) == 1

    volumes = _call_keyword(cls_call, "volumes")
    assert isinstance(volumes, ast.Dict)
    model_mount = volumes.values[0]
    assert isinstance(model_mount, ast.Call)
    assert isinstance(model_mount.func, ast.Attribute)
    assert model_mount.func.attr == "with_mount_options"
    assert ast.literal_eval(_call_keyword(model_mount, "read_only")) is True


def test_modal_lifecycle_reuses_attached_session_and_commits_only_after_success():
    source = APP_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    worker = next(node for node in module.body if isinstance(node, ast.ClassDef))
    methods = {node.name: node for node in worker.body if isinstance(node, ast.FunctionDef)}

    assert {"start_container", "run_song", "stop_container"} <= methods.keys()
    assert "mapperatorinator_backend=\"song_session\"" in source
    assert "mapperatorinator_hold_state_mode=\"incremental\"" in source
    assert "attached_inference_session=self.session" in source
    run_source = ast.get_source_segment(source, methods["run_song"])
    assert run_source is not None
    assert run_source.index("run_pipeline(") < run_source.index("jobs_volume.commit()")


def test_modal_requests_use_distinct_deterministic_run_directories(tmp_path: Path):
    first = request_output_dir(tmp_path, "request-A")
    repeated = request_output_dir(tmp_path, "request-A")
    second = request_output_dir(tmp_path, "request-B")

    assert first == repeated
    assert first != second
    assert first.parent == tmp_path.resolve()
    assert len(first.name) == 64
    assert set(first.name) <= set("0123456789abcdef")
    assert not first.exists()


@pytest.mark.parametrize("request_id", ["", True, 1, "x" * 129])
def test_modal_request_directory_rejects_invalid_request_ids(
    tmp_path: Path,
    request_id: object,
):
    with pytest.raises((TypeError, ValueError)):
        request_output_dir(tmp_path, request_id)  # type: ignore[arg-type]


def test_modal_run_song_passes_the_request_scoped_directory_to_pipeline():
    source = APP_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    worker = next(node for node in module.body if isinstance(node, ast.ClassDef))
    run_song = next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_song"
    )
    run_source = ast.get_source_segment(source, run_song)

    assert run_source is not None
    assert 'request_output_dir(RUNS_ROOT, validated["requestId"])' in run_source
    assert "output_dir=RUNS_ROOT," not in run_source


def test_local_entrypoint_reserves_budget_before_remote_call():
    source = APP_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    entrypoint = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    entry_source = ast.get_source_segment(source, entrypoint)
    assert entry_source is not None
    assert entry_source.index("reserve_modal_budget(") < entry_source.index(".run_song.remote(")


def test_model_revision_is_an_explicit_modal_parameter_not_an_unforwarded_env_read():
    source = APP_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    worker = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ResidentChartWorker"
    )
    parameter = next(
        node
        for node in worker.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "model_revision"
    )
    assert isinstance(parameter.value, ast.Call)
    assert isinstance(parameter.value.func, ast.Attribute)
    assert parameter.value.func.attr == "parameter"
    start = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "start_container")
    start_source = ast.get_source_segment(source, start)
    assert start_source is not None
    assert "self.model_revision" in start_source
    main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    main_source = ast.get_source_segment(source, main)
    assert main_source is not None
    assert "ResidentChartWorker(model_revision=_immutable_model_revision())" in main_source
