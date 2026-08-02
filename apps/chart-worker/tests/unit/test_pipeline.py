from pathlib import Path

import pytest

from chart_worker.pipeline import PipelineOptions, run_pipeline
from chart_worker.schema.playtest_run import PlaytestRunManifest
from tests.support import fake_dependencies


def test_fake_pipeline_writes_playtest_manifest(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"

    result = run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=output_dir,
            title="fixture",
            generator="fake",
            keysounds=False,
            seed=7,
        ),
        dependencies=fake_dependencies(),
    )

    manifest = PlaytestRunManifest.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )
    assert len(manifest.charts) == 12
    assert len({(chart.key_mode, chart.difficulty) for chart in manifest.charts}) == 12
    assert all((output_dir / chart.path).is_file() for chart in manifest.charts)
    assert len(result.raw_osu_paths) == 12
    assert set(result.elapsed_ms_by_stage) == {"analysis", "generation", "stems", "postprocess"}
    assert (output_dir / manifest.generation_report_path).is_file()


def test_pipeline_refuses_a_nonempty_output_without_removing_it(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        run_pipeline(
            PipelineOptions(source=source, output_dir=output_dir, title="fixture"),
            dependencies=fake_dependencies(),
        )

    assert marker.read_text(encoding="utf-8") == "keep"
