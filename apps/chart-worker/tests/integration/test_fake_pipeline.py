import hashlib
import wave
from pathlib import Path

import numpy as np

from chart_worker.pipeline import PipelineOptions, run_pipeline
from chart_worker.schema.chart import ChartDocument
from chart_worker.schema.playtest_run import PlaytestRunManifestV2
from tests.support import contract_fixture_dependencies


def _write_tone_wav(path: Path) -> None:
    sample_rate_hz = 48_000
    frames = np.arange(sample_rate_hz * 8, dtype=np.float64)
    tone = (0.08 * np.sin(2.0 * np.pi * 220.0 * frames / sample_rate_hz) * 32_767).astype("<i2")
    stereo = np.column_stack((tone, tone))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate_hz)
        output.writeframes(stereo.tobytes())


def test_fake_pipeline_writes_twelve_hash_verified_charts(tmp_path: Path):
    source = tmp_path / "contract.wav"
    _write_tone_wav(source)
    output_dir = tmp_path / "run"
    result = run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=output_dir,
            title="contract fixture",
            generator="fake",
            seed=7,
            worker_version="fixture",
        ),
        dependencies=contract_fixture_dependencies(),
    )

    manifest = PlaytestRunManifestV2.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )
    assert len(manifest.charts) == 12
    for reference in manifest.charts:
        path = output_dir / reference.path
        ChartDocument.model_validate_json(path.read_text(encoding="utf-8"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference.sha256
    assert (output_dir / manifest.audio.game.path).is_file()
    report_path = output_dir / manifest.generation_report.path
    assert report_path.is_file()
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == manifest.generation_report.sha256
    assert manifest.strict_blockers == ["BOUNDARY_POLICY_UNCALIBRATED"]
    assert manifest.publication.decision == "PLAYTEST_ONLY"
    assert manifest.publication.reason_codes == ["BOUNDARY_POLICY_UNCALIBRATED"]
    assert not (output_dir / "analysis").exists()
    assert set(result.elapsed_ms_by_stage) == {
        "prepare",
        "analysis",
        "timing",
        "generation",
        "export",
    }
