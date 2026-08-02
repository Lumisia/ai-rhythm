import hashlib
import wave
from pathlib import Path

from chart_worker.pipeline import PipelineOptions, run_pipeline
from chart_worker.schema.chart import ChartDocument
from chart_worker.schema.playtest_run import PlaytestRunManifest
from tests.support import fake_dependencies


def _write_silent_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes(b"\0\0\0\0" * 4_800)


def test_fake_pipeline_writes_twelve_hash_verified_charts(tmp_path: Path):
    source = tmp_path / "short.wav"
    _write_silent_wav(source)
    output_dir = tmp_path / "run"
    result = run_pipeline(
        PipelineOptions(source=source, output_dir=output_dir, title="short", seed=3),
        dependencies=fake_dependencies(),
    )

    manifest = PlaytestRunManifest.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )
    assert len(manifest.charts) == 12
    for reference in manifest.charts:
        path = output_dir / reference.path
        ChartDocument.model_validate_json(path.read_text(encoding="utf-8"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference.sha256
    assert (output_dir / "analysis" / "analysis-v1.json").is_file()
    assert (output_dir / "analysis" / "onsets-v1.npz").is_file()
