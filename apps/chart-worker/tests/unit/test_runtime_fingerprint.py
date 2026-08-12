from chart_worker.analysis.runtime_fingerprint import build_runtime_fingerprint
from chart_worker.config import WorkerConfig
from tests.support import fake_dependencies


def test_runtime_fingerprint_is_stable_and_does_not_expose_absolute_paths(tmp_path):
    dependencies = fake_dependencies()
    prepared = dependencies.prepare(
        tmp_path / "source.wav",
        tmp_path / "run",
        dependencies.config,
    )
    analysis = dependencies.analyze(prepared.normalized.path)
    authority = dependencies.timing(
        prepared,
        analysis,
        tmp_path / "run",
        dependencies.select_generator("fake", dependencies.config),
        7,
    )
    config = WorkerConfig(
        chart_generator="fake",
        storage_local_root=tmp_path / "private-storage",
    )

    first = build_runtime_fingerprint(
        config=config,
        prepared=prepared,
        analysis=analysis,
        authority=authority,
        generator="fake",
        worker_version="test-version",
    )
    second = build_runtime_fingerprint(
        config=config.model_copy(),
        prepared=prepared,
        analysis=analysis,
        authority=authority,
        generator="fake",
        worker_version="test-version",
    )

    assert first == second
    assert first["id"].startswith("sha256:")
    assert first["canonicalAudioSha256"] == prepared.normalized.sha256
    assert first["timingAuthoritySha256"] == authority.sha256
    assert first["analyzer"]["sampleRateHz"] == analysis.sample_rate_hz
    assert first["analyzer"]["hopLength"] == analysis.hop_length
    assert first["analyzer"]["nFft"] == analysis.n_fft
    assert first["qualityGateVersion"] == "quality-gate-v3-outro-review"
    assert first["outroPolicyVersion"] == "outro-policy-v1-provisional"
    assert first["songBoundaryContractVersion"] == "song-boundary-contract-v2"
    assert first["songBoundaryContractSha256"] is None
    assert first["evidenceGrade"] == "VERIFIED_CODE"
    assert str(tmp_path) not in str(first)


def test_runtime_fingerprint_changes_when_evaluation_context_changes(tmp_path):
    dependencies = fake_dependencies()
    prepared = dependencies.prepare(
        tmp_path / "source.wav",
        tmp_path / "run",
        dependencies.config,
    )
    analysis = dependencies.analyze(prepared.normalized.path)
    authority = dependencies.timing(
        prepared,
        analysis,
        tmp_path / "run",
        dependencies.select_generator("fake", dependencies.config),
        7,
    )

    base = build_runtime_fingerprint(
        config=dependencies.config,
        prepared=prepared,
        analysis=analysis,
        authority=authority,
        generator="fake",
        worker_version="a",
    )
    changed = build_runtime_fingerprint(
        config=dependencies.config,
        prepared=prepared,
        analysis=analysis,
        authority=authority,
        generator="fake",
        worker_version="b",
    )

    assert base["id"] != changed["id"]
