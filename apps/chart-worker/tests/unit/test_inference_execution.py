import json

from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.inference_execution import error_report_json


def test_error_report_preserves_machine_code_and_subprocess_context() -> None:
    error = WorkerError(
        ErrorCode.CHART_GENERATION_FAILED,
        "generator exited",
        context={"stderr": "CUDA failure", "exitCode": 1},
    )

    report = json.loads(error_report_json(error))

    assert report["code"] == ErrorCode.CHART_GENERATION_FAILED.value
    assert report["context"] == {"stderr": "CUDA failure", "exitCode": 1}
    assert "generator exited" in report["message"]
