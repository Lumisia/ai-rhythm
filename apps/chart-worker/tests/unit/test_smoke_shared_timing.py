import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.generation.osu_writer import notes_to_osu_mania, timing_to_osu_mania
from chart_worker.hashing import sha256_file
from chart_worker.schema.note import NoteEvent
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority


def test_cli_requires_smoke_inputs_and_defaults_seed():
    import smoke_shared_timing as smoke

    args = smoke.parse_args(
        [
            "--source",
            "source.wav",
            "--output",
            "out",
            "--key-mode",
            "4",
            "--difficulty",
            "NORMAL",
        ]
    )

    assert (args.source, args.output, args.key_mode, args.difficulty, args.seed) == (
        Path("source.wav"),
        Path("out"),
        4,
        "NORMAL",
        0,
    )
    with pytest.raises(SystemExit):
        smoke.parse_args([])


def _prepared(tmp_path: Path) -> PreparedAudio:
    audio = tmp_path / "audio" / "game.flac"
    audio.parent.mkdir()
    audio.write_bytes(b"canonical audio")
    return PreparedAudio(
        normalized=NormalizedAudio(
            path=audio,
            profile_version="audio-profile-v1",
            sha256=sha256_file(audio),
            duration_ms=10_000,
            sample_rate_hz=48_000,
            channels=2,
            source_duration_ms=10_000,
            trimmed_ms=0,
            gain_db=0.0,
            achieved_lufs=-14.0,
            achieved_true_peak_dbtp=-1.0,
            shortfall_lu=0.0,
            limited_by="none",
        )
    )


def _map_command(request, workdir):
    return [
        f"output_path='{workdir.resolve()}'",
        f"keycount={request.key_mode}",
        f"beatmap_path='{request.timing_reference_path.resolve()}'",
        "in_context=[TIMING]",
        "output_type=[MAP]",
    ]


def test_smoke_orchestrates_one_shared_timing_and_one_map_without_mutation(
    tmp_path: Path,
):
    import smoke_shared_timing as smoke

    calls: list[str] = []
    prepared = _prepared(tmp_path)
    timing_events = (OsuBpmEvent(0, 120.0), OsuBpmEvent(5_000, 150.0))

    def prepare(source, output, *, config):
        calls.append("prepare")
        assert source == Path("source.wav")
        assert output == tmp_path
        return prepared

    def analyze(path):
        calls.append("analyze")
        assert path == prepared.normalized.path
        return object()

    def timing(prepared_audio, output, *, generator, seed):
        calls.append("timing")
        assert prepared_audio is prepared
        assert seed == 0
        reference_path = output / "audio" / "timing-reference.osu"
        reference_path.write_text(
            timing_to_osu_mania(
                timing_events, audio_filename="game.flac", title="fixture"
            ),
            encoding="utf-8",
        )
        return SongTimingAuthority(
            reference_path=reference_path,
            sha256=sha256_file(reference_path),
            audio_sha256=prepared.normalized.sha256,
            bpm_events=timing_events,
            generator_name="fake",
            seed=seed,
            mode="STANDARD",
            attempt_count=1,
        )

    class Generator:
        def __init__(self):
            self.requests = []

        def generate_map(self, request, workdir):
            calls.append("map")
            self.requests.append((request, workdir))
            notes = [
                NoteEvent(1_000, 0),
                NoteEvent(2_000, 1, "HOLD", 500),
                NoteEvent(4_000, 3),
            ]
            return GeneratedChart(
                notes=notes,
                key_mode=4,
                osu_text=notes_to_osu_mania(
                    notes,
                    key_mode=4,
                    bpm=120.0,
                    offset_ms=0,
                    audio_filename="game.flac",
                    title="fixture",
                    bpm_events=timing_events,
                ),
                generator_name="fake",
                seed=request.seed,
                bpm_events=timing_events,
            )

    generator = Generator()
    summary = smoke.run_smoke(
        source=Path("source.wav"),
        output=tmp_path,
        key_mode=4,
        difficulty="NORMAL",
        seed=0,
        dependencies=smoke.SmokeDependencies(
            config=lambda: object(),
            prepare=prepare,
            analyze=analyze,
            timing=timing,
            generator=lambda _config: generator,
            map_command=lambda _config, request, workdir: _map_command(request, workdir),
        ),
    )

    assert calls == ["prepare", "analyze", "timing", "map"]
    request, workdir = generator.requests[0]
    assert (request.key_mode, request.difficulty, request.seed) == (4, "NORMAL", 0)
    assert request.timing_reference_path == tmp_path / "audio" / "timing-reference.osu"
    assert workdir == tmp_path / "map" / "work" / "attempt-1"
    assert summary["referenceBpmEvents"] == [[0, 120.0], [5_000, 150.0]]
    assert summary["mapBpmEvents"] == [[0, 120.0], [5_000, 150.0]]
    assert summary["noteCount"] == 3
    assert summary["holdCount"] == 1
    assert summary["firstNote"] == {
        "timeMs": 1_000,
        "lane": 0,
        "kind": "TAP",
        "durationMs": None,
    }
    assert summary["maximumGapMs"] == 2_000
    assert summary["mapCommandContract"] == {
        "keycount": "4",
        "beatmapPath": str((tmp_path / "audio" / "timing-reference.osu").resolve()),
        "outputType": "[MAP]",
        "inContext": "[TIMING]",
    }
    assert json.loads((tmp_path / "shared-timing-smoke.json").read_text("utf-8")) == summary
    assert (tmp_path / "raw" / "4k-normal.osu").is_file()


def test_smoke_rejects_raw_note_mutation_before_accepting_chart(tmp_path: Path):
    import smoke_shared_timing as smoke

    prepared = _prepared(tmp_path)
    events = (OsuBpmEvent(0, 120.0),)
    reference_path = tmp_path / "audio" / "timing-reference.osu"
    reference_path.write_text(
        timing_to_osu_mania(events, audio_filename="game.flac", title="fixture"),
        encoding="utf-8",
    )
    authority = SongTimingAuthority(
        reference_path=reference_path,
        sha256=sha256_file(reference_path),
        audio_sha256=prepared.normalized.sha256,
        bpm_events=events,
        generator_name="fake",
        seed=0,
        mode="STANDARD",
        attempt_count=1,
    )

    class MutatingGenerator:
        def generate_map(self, request, workdir):
            return GeneratedChart(
                notes=[NoteEvent(1_000, 0)],
                key_mode=4,
                osu_text=notes_to_osu_mania(
                    [NoteEvent(1_000, 1)],
                    key_mode=4,
                    bpm=120.0,
                    offset_ms=0,
                    audio_filename="game.flac",
                    title="fixture",
                    bpm_events=events,
                ),
                generator_name="fake",
                seed=0,
                bpm_events=events,
            )

    with pytest.raises(ValueError, match="note fields changed"):
        smoke.run_smoke(
            source=Path("source.wav"),
            output=tmp_path,
            key_mode=4,
            difficulty="NORMAL",
            seed=0,
            dependencies=smoke.SmokeDependencies(
                config=lambda: object(),
                prepare=lambda *_args, **_kwargs: prepared,
                analyze=lambda _path: object(),
                timing=lambda *_args, **_kwargs: authority,
                generator=lambda _config: MutatingGenerator(),
                map_command=lambda _config, request, workdir: _map_command(
                    request, workdir
                ),
            ),
        )
    assert not (tmp_path / "raw" / "4k-normal.osu").exists()
    assert not (tmp_path / "shared-timing-smoke.json").exists()


def test_smoke_rejects_a_non_shared_map_command_before_generation(tmp_path: Path):
    import smoke_shared_timing as smoke

    prepared = _prepared(tmp_path)
    events = (OsuBpmEvent(0, 120.0),)
    reference_path = tmp_path / "audio" / "timing-reference.osu"
    reference_path.write_text(
        timing_to_osu_mania(events, audio_filename="game.flac", title="fixture"),
        encoding="utf-8",
    )
    authority = SongTimingAuthority(
        reference_path=reference_path,
        sha256=sha256_file(reference_path),
        audio_sha256=prepared.normalized.sha256,
        bpm_events=events,
        generator_name="fake",
        seed=0,
        mode="STANDARD",
        attempt_count=1,
    )

    class RecordingGenerator:
        calls = 0

        def generate_map(self, request, workdir):
            self.calls += 1
            raise AssertionError("invalid command must fail before MAP generation")

    generator = RecordingGenerator()
    with pytest.raises(ValueError, match="MAP command contract"):
        smoke.run_smoke(
            source=Path("source.wav"),
            output=tmp_path,
            key_mode=4,
            difficulty="NORMAL",
            dependencies=smoke.SmokeDependencies(
                config=lambda: object(),
                prepare=lambda *_args, **_kwargs: prepared,
                analyze=lambda _path: object(),
                timing=lambda *_args, **_kwargs: authority,
                generator=lambda _config: generator,
                map_command=lambda _config, request, workdir: [
                    "keycount=4",
                    f"beatmap_path='{request.timing_reference_path.resolve()}'",
                    "in_context=[]",
                    "output_type=[MAP]",
                ],
            ),
        )
    assert generator.calls == 0
    assert not (tmp_path / "raw" / "4k-normal.osu").exists()
    assert not (tmp_path / "shared-timing-smoke.json").exists()


def test_smoke_rejects_a_reference_hash_change_after_map_generation(tmp_path: Path):
    import smoke_shared_timing as smoke

    prepared = _prepared(tmp_path)
    events = (OsuBpmEvent(0, 120.0),)
    reference_path = tmp_path / "audio" / "timing-reference.osu"
    reference_path.write_text(
        timing_to_osu_mania(events, audio_filename="game.flac", title="fixture"),
        encoding="utf-8",
    )
    authority = SongTimingAuthority(
        reference_path=reference_path,
        sha256=sha256_file(reference_path),
        audio_sha256=prepared.normalized.sha256,
        bpm_events=events,
        generator_name="fake",
        seed=0,
        mode="STANDARD",
        attempt_count=1,
    )

    class ChangingGenerator:
        def generate_map(self, request, workdir):
            reference_path.write_text(
                reference_path.read_text(encoding="utf-8") + "// altered\n",
                encoding="utf-8",
            )
            notes = [NoteEvent(1_000, 0)]
            return GeneratedChart(
                notes=notes,
                key_mode=4,
                osu_text=notes_to_osu_mania(
                    notes,
                    key_mode=4,
                    bpm=120.0,
                    offset_ms=0,
                    audio_filename="game.flac",
                    title="fixture",
                    bpm_events=events,
                ),
                generator_name="fake",
                seed=0,
                bpm_events=events,
            )

    with pytest.raises(ValueError, match="reference hash changed"):
        smoke.run_smoke(
            source=Path("source.wav"),
            output=tmp_path,
            key_mode=4,
            difficulty="NORMAL",
            dependencies=smoke.SmokeDependencies(
                config=lambda: object(),
                prepare=lambda *_args, **_kwargs: prepared,
                analyze=lambda _path: object(),
                timing=lambda *_args, **_kwargs: authority,
                generator=lambda _config: ChangingGenerator(),
                map_command=lambda _config, request, workdir: _map_command(
                    request, workdir
                ),
            ),
        )
    assert not (tmp_path / "raw" / "4k-normal.osu").exists()
    assert not (tmp_path / "shared-timing-smoke.json").exists()
