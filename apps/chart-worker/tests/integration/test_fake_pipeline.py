import hashlib
import wave
from pathlib import Path

import numpy as np

from chart_worker.pipeline import PipelineOptions, run_pipeline
from chart_worker.schema.chart import ChartDocument
from chart_worker.schema.playtest_run import PlaytestRunManifest
from tests.support import contract_fixture_dependencies

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[4]
    / "apps"
    / "frontend"
    / "src"
    / "test"
    / "fixtures"
    / "playtest-run"
)


def _write_tone_wav(path: Path) -> None:
    sample_rate_hz = 48_000
    frames = np.arange(sample_rate_hz * 8, dtype=np.float64)
    tone = (0.08 * np.sin(2.0 * np.pi * 220.0 * frames / sample_rate_hz) * 32_767).astype(
        "<i2"
    )
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
            seed=7,
            worker_version="fixture",
        ),
        dependencies=contract_fixture_dependencies(),
    )

    manifest = PlaytestRunManifest.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )
    assert len(manifest.charts) == 12
    for reference in manifest.charts:
        path = output_dir / reference.path
        ChartDocument.model_validate_json(path.read_text(encoding="utf-8"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference.sha256
        assert path.read_bytes() == (FIXTURE_ROOT / reference.path).read_bytes()
    assert result.manifest_path.read_bytes() == (FIXTURE_ROOT / "playtest-run-v1.json").read_bytes()
    assert (output_dir / manifest.audio.game.path).read_bytes() == (
        FIXTURE_ROOT / manifest.audio.game.path
    ).read_bytes()
    assert (output_dir / "analysis" / "analysis-v1.json").is_file()
    assert (output_dir / "analysis" / "onsets-v1.npz").is_file()
