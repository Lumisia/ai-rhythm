from pathlib import Path

import pytest

from chart_worker.pipeline import PipelineOptions, run_pipeline
from chart_worker.reprocess import PostprocessOptions, run_postprocess_only
from chart_worker.schema.playtest_run import PlaytestRunManifest
from tests.support import fake_dependencies


def test_postprocess_reuses_snapshot_without_modifying_source_run(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    input_dir = tmp_path / "input"
    original = run_pipeline(
        PipelineOptions(source=source, output_dir=input_dir, title="fixture", seed=7),
        dependencies=fake_dependencies(),
    )
    before = original.manifest_path.read_bytes()

    result = run_postprocess_only(
        PostprocessOptions(input_dir=input_dir, output_dir=tmp_path / "output")
    )

    manifest = PlaytestRunManifest.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )
    assert len(manifest.charts) == 12
    assert all((result.output_dir / chart.path).is_file() for chart in manifest.charts)
    assert original.manifest_path.read_bytes() == before


def test_postprocess_rejects_the_input_directory_as_output(tmp_path: Path):
    with pytest.raises(ValueError, match="different"):
        PostprocessOptions(input_dir=tmp_path, output_dir=tmp_path)
