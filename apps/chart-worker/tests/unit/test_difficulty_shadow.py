import json
from pathlib import Path
from uuid import UUID

import pytest

from chart_worker.analysis.difficulty_shadow import (
    ShadowChartRecord,
    evaluate_calibration_loso,
    fit_calibration,
    grouped_song_folds,
    recalculate_batch,
)
from chart_worker.generation import mapperatorinator
from chart_worker.schema.chart import (
    BpmEvent,
    ChartDocument,
    ChartMetrics,
    GeneratorInfo,
    notes_to_chart_notes,
)
from chart_worker.schema.note import NoteEvent
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES, lane_semantics

UUID_A = UUID("11111111-1111-4111-8111-111111111111")
SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _metrics(difficulty: str, note_count: int) -> ChartMetrics:
    return ChartMetrics(
        note_count=note_count,
        hold_count=0,
        avg_nps=1.0,
        p95_nps=2.0,
        peak_nps=2.0,
        chord_ratio=0.0,
        max_jack=1,
        project_rating={"EASY": 1.0, "NORMAL": 2.0, "HARD": 3.0, "EXPERT": 4.0}[
            difficulty
        ],
        project_tier=difficulty,
        pattern_entropy=0.0,
        drum_coverage=1.0,
        drum_precision=1.0,
        mean_abs_err_ms=0.0,
        side_note_ratio=0.0,
        side_hold_ratio=0.0,
        moved_note_ratio=0.0,
    )


@pytest.fixture
def archived_batch(tmp_path: Path) -> Path:
    batch_dir = tmp_path / "batch"
    song_dir = batch_dir / "songs" / "01"
    chart_dir = song_dir / "charts"
    chart_dir.mkdir(parents=True)
    charts = []
    for key_mode in KEY_MODES:
        for difficulty_index, difficulty in enumerate(DIFFICULTIES):
            notes = [
                NoteEvent(100 + difficulty_index * 10, 0),
                NoteEvent(500, min(1, key_mode - 1)),
            ]
            filename = f"{key_mode}k-{difficulty.lower()}.chart.json"
            document = ChartDocument(
                chart_id=UUID_A,
                song_version_id=UUID_A,
                game_audio_asset_id=UUID_A,
                audio_sha256=SHA,
                key_mode=key_mode,
                difficulty=difficulty,
                lane_semantics=lane_semantics(key_mode),
                offset_ms=0,
                duration_ms=2_000,
                bpm_events=[
                    BpmEvent(time_ms=0, bpm=120.0),
                    BpmEvent(time_ms=1_000, bpm=180.0),
                ],
                bpm_source="MAPPERATORINATOR",
                notes=notes_to_chart_notes(notes),
                auto_play_onsets=[],
                metrics=_metrics(difficulty, len(notes)),
                generator=GeneratorInfo(
                    name="mapperatorinator-v32",
                    version="test",
                    analysis_version="test",
                    postprocess_version="test",
                    seed=19,
                ),
            )
            (chart_dir / filename).write_text(document.to_json(indent=2), encoding="utf-8")
            charts.append(
                {
                    "keyMode": key_mode,
                    "difficulty": difficulty,
                    "chartPath": f"charts/{filename}",
                    "selectedSeed": 19,
                    "difficultyProfile": {
                        "projectRating": document.metrics.project_rating
                    },
                }
            )
    (song_dir / "generation-report.json").write_text(
        json.dumps(
            {"runId": "song-one", "sourceName": "song.wav", "charts": charts},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return batch_dir


def test_recalculation_never_constructs_a_generator(
    monkeypatch: pytest.MonkeyPatch,
    archived_batch: Path,
    tmp_path: Path,
):
    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("generator must not be constructed")

    monkeypatch.setattr(
        mapperatorinator,
        "MapperatorinatorGenerator",
        fail_if_called,
    )
    output_path = tmp_path / "difficulty-shadow-v2.json"

    report = recalculate_batch(archived_batch, output_path)

    assert report.song_count == 1
    assert report.chart_count == 12
    assert len(report.records) == 12
    assert output_path.exists()
    assert report.records[0].vector_v2["version"] == "difficulty-vector-v2"
    assert report.records[0].source_name == "song.wav"
    assert report.records[0].audio_sha256 == SHA


def test_recalculation_rejects_report_chart_identity_mismatch(
    archived_batch: Path,
    tmp_path: Path,
):
    report_path = archived_batch / "songs" / "01" / "generation-report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["charts"][0]["keyMode"] = 7
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="ARCHIVE_INCONSISTENT"):
        recalculate_batch(archived_batch, tmp_path / "shadow.json")


def test_recalculation_never_writes_inside_a_song_result_directory(
    archived_batch: Path,
):
    output = archived_batch / "songs" / "01" / "shadow.json"

    with pytest.raises(ValueError, match="song result directory"):
        recalculate_batch(archived_batch, output)


def test_recalculation_supports_legacy_flat_batch_layout(
    archived_batch: Path,
    tmp_path: Path,
):
    legacy_song = archived_batch / "legacy-song"
    (archived_batch / "songs" / "01").rename(legacy_song)
    (archived_batch / "songs").rmdir()

    report = recalculate_batch(archived_batch, tmp_path / "legacy-shadow.json")

    assert report.song_count == 1
    assert report.chart_count == 12
    assert {record.song_id for record in report.records} == {"legacy-song"}


def _shadow_record(
    audio_sha256: str,
    difficulty: str,
    rank: float,
    *,
    key_mode: int = 4,
) -> ShadowChartRecord:
    vector = {
        "densityStrain": rank * 2.0,
        "jackStrain": rank * 1.2,
        "chordLoad": rank * 0.5,
        "lnStrain": rank * 0.3,
        "coordination": rank * 1.5,
        "peakSkill": rank * 3.0,
        "boundedStamina": rank * 0.2,
        "orderingScore": rank * 3.0,
        "meanHoldBeats": rank * 0.1,
        "p95HoldBeats": rank * 0.2,
        "holdOccupancyRatio": rank * 0.02,
        "overlapInputLoad": 0.0,
        "releaseLoad": 0.0,
        "version": "difficulty-vector-v2",
        "sectionPeaks": [],
        "weightedPeakSum": rank * 3.0,
        "maxSectionPeak": rank,
    }
    return ShadowChartRecord(
        song_id=audio_sha256,
        source_name=f"{audio_sha256}.wav",
        audio_sha256=(audio_sha256 * 64)[:64],
        key_mode=key_mode,
        difficulty=difficulty,
        current_rating=rank,
        vector_v2=vector,
        first_row_ms=0,
        selected_seed=1,
        source_report="generation-report.json",
    )


def _complete_records(song_count: int = 4) -> tuple[ShadowChartRecord, ...]:
    return tuple(
        _shadow_record(str(song), difficulty, float(index + 1))
        for song in range(song_count)
        for index, difficulty in enumerate(DIFFICULTIES)
    )


def test_all_records_of_one_audio_stay_in_the_same_fold():
    records = _complete_records()

    folds = grouped_song_folds(records, key_mode=4)

    assert len(folds) == 4
    assert all(
        not ({record.audio_sha256 for record in train} & {record.audio_sha256 for record in test})
        for train, test in folds
    )


def test_calibration_rejects_an_audio_group_with_missing_difficulties():
    records = tuple(
        record
        for record in _complete_records()
        if not (record.song_id == "0" and record.difficulty == "EXPERT")
    )

    with pytest.raises(ValueError, match="complete family"):
        fit_calibration(records, key_mode=4)


def test_calibration_weights_are_non_negative_and_loso_is_accepted():
    records = _complete_records()

    calibration = fit_calibration(records, key_mode=4)
    evaluation = evaluate_calibration_loso(records, key_mode=4)

    assert all(weight >= 0 for weight in calibration.weights)
    assert sum(calibration.weights) == pytest.approx(1.0)
    assert evaluation.fold_count == 4
    assert evaluation.accepted is True
    assert evaluation.v2_inversions <= evaluation.current_inversions


def test_ordering_calibration_uses_integrated_peak_without_double_counting_axes():
    records = _complete_records()

    calibration = fit_calibration(records, key_mode=4)

    assert calibration.axis_names == ("peak_skill",)
