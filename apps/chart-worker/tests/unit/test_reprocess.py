from pathlib import Path

import pytest

from chart_worker.pipeline import PipelineOptions, run_pipeline
from chart_worker.reprocess import PostprocessOptions, run_postprocess_only
from chart_worker.schema.playtest_run import PlaytestRunManifest
from tests.support import fake_dependencies, fake_stem_stage


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


def _run_with_keysounds(tmp_path: Path, *, keysounds: bool):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    input_dir = tmp_path / "input"
    run_pipeline(
        PipelineOptions(source=source, output_dir=input_dir, title="fixture", seed=7),
        dependencies=fake_dependencies(),
    )
    seen: list[bool] = []

    def stems_stage(analysis, run_dir: Path, enabled: bool):
        seen.append(enabled)
        # 실행 폴더 안을 가리켜야 S3 가 자산 경로를 받아들인다.
        assert analysis.normalized.path.is_relative_to(run_dir)
        return fake_stem_stage(analysis, run_dir, enabled)

    result = run_postprocess_only(
        PostprocessOptions(
            input_dir=input_dir, output_dir=tmp_path / "output", keysounds=keysounds
        ),
        stems_stage=stems_stage,
    )
    manifest = PlaytestRunManifest.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )
    return manifest, seen, result


def test_postprocess_can_add_keysounds_to_a_run_that_has_none(tmp_path: Path):
    """스템을 얻자고 Mapperatorinator 를 다시 돌릴 이유가 없다."""
    manifest, seen, result = _run_with_keysounds(tmp_path, keysounds=True)
    assert seen == [True]
    assert manifest.audio.no_drums is not None
    assert manifest.audio.keys is not None
    assert manifest.keysound_manifest_path is not None
    assert (result.output_dir / manifest.audio.no_drums.path).is_file()
    assert (result.output_dir / manifest.keysound_manifest_path).is_file()


def test_postprocess_leaves_keysounds_alone_unless_asked(tmp_path: Path):
    manifest, seen, _ = _run_with_keysounds(tmp_path, keysounds=False)
    assert seen == []
    assert manifest.audio.no_drums is None
    assert manifest.keysound_manifest_path is None


@pytest.mark.parametrize(
    "missing_relative_path, expected_message",
    (
        ("analysis/timing.osu", "timing osu is missing"),
        ("analysis/timing-quality-v1.json", "timing quality report is missing"),
    ),
)
def test_postprocess_rejects_an_incomplete_selected_timing_snapshot(
    tmp_path: Path, missing_relative_path: str, expected_message: str
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    input_dir = tmp_path / "input"
    run_pipeline(
        PipelineOptions(source=source, output_dir=input_dir, title="fixture", seed=7),
        dependencies=fake_dependencies(),
    )
    (input_dir / missing_relative_path).unlink()

    with pytest.raises(ValueError, match=expected_message):
        run_postprocess_only(
            PostprocessOptions(input_dir=input_dir, output_dir=tmp_path / "output")
        )
