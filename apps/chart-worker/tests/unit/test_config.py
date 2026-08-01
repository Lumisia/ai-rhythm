from pathlib import Path

from chart_worker.config import load_config


def test_defaults_to_fake_generator(monkeypatch):
    monkeypatch.delenv("CHART_GENERATOR", raising=False)
    assert load_config().chart_generator == "fake"


def test_reads_mapperatorinator_paths(monkeypatch):
    monkeypatch.setenv("CHART_GENERATOR", "mapperatorinator")
    monkeypatch.setenv("MAPPERATORINATOR_HOME", r"C:\Users\PC\Mapperatorinator")
    cfg = load_config()
    assert cfg.mapperatorinator_home == Path(r"C:\Users\PC\Mapperatorinator")


def test_precision_defaults_to_fp16(monkeypatch):
    monkeypatch.delenv("MAPPERATORINATOR_PRECISION", raising=False)
    assert load_config().mapperatorinator_precision == "fp16"
