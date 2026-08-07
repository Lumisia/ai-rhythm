import json

from chart_worker.generation.resnap_diagnostics import read_resnap_diagnostics


def test_reads_collision_only_sidecar(tmp_path):
    osu_path = tmp_path / "beatmap.osu"
    osu_path.write_text("osu", encoding="utf-8")
    osu_path.with_suffix(".resnap.json").write_text(
        json.dumps(
            {
                "version": "resnap-collisions-v1",
                "seed": 19,
                "collisions": [
                    {
                        "lane": 2,
                        "noteKind": "HOLD",
                        "preTimeMs": 129_420,
                        "postTimeMs": 128_847,
                        "snapDivisor": 4,
                    },
                    {
                        "lane": 2,
                        "noteKind": "TAP",
                        "preTimeMs": 129_010,
                        "postTimeMs": 128_847,
                        "snapDivisor": 4,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    diagnostics = read_resnap_diagnostics(osu_path)

    assert diagnostics.status == "OBSERVED"
    assert diagnostics.error is None
    assert diagnostics.collisions[0].to_report() == {
        "seed": 19,
        "lane": 2,
        "noteKind": "HOLD",
        "preTimeMs": 129_420,
        "postTimeMs": 128_847,
        "snapDivisor": 4,
    }


def test_valid_empty_sidecar_reports_no_collisions(tmp_path):
    osu_path = tmp_path / "beatmap.osu"
    osu_path.with_suffix(".resnap.json").write_text(
        json.dumps(
            {
                "version": "resnap-collisions-v1",
                "seed": 7,
                "collisions": [],
            }
        ),
        encoding="utf-8",
    )

    diagnostics = read_resnap_diagnostics(osu_path)

    assert diagnostics.status == "NO_COLLISIONS"
    assert diagnostics.collisions == ()


def test_missing_sidecar_is_non_blocking_and_unobserved(tmp_path):
    diagnostics = read_resnap_diagnostics(tmp_path / "beatmap.osu")

    assert diagnostics.status == "UNOBSERVED"
    assert diagnostics.collisions == ()
    assert diagnostics.error is None


def test_malformed_sidecar_is_non_blocking_and_invalid(tmp_path):
    osu_path = tmp_path / "beatmap.osu"
    osu_path.with_suffix(".resnap.json").write_text("not-json", encoding="utf-8")

    diagnostics = read_resnap_diagnostics(osu_path)

    assert diagnostics.status == "INVALID"
    assert diagnostics.collisions == ()
    assert diagnostics.error
