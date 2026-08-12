from chart_worker.analysis.beat import BeatGrid
from chart_worker.analysis.beat_corroboration import measure_beat_corroboration
from chart_worker.generation.osu_parser import OsuBpmEvent


def _grid(*, start_ms: int = 0) -> BeatGrid:
    beats = tuple(range(start_ms, 30_000, 1_000))
    return BeatGrid(
        beat_ms=beats,
        downbeat_indices=tuple(range(0, len(beats), 4)),
        bpm=60.0,
        beats_per_bar=4,
        bpm_drift_pct=0.0,
        raw_beat_count=len(beats),
        dropped_beat_count=0,
        residual_rms_ms=0.0,
        residual_max_ms=0.0,
    )


def test_corroboration_accepts_the_equivalent_half_tempo_level():
    result = measure_beat_corroboration(
        (OsuBpmEvent(0, 120.0),),
        _grid(),
        duration_ms=30_000,
    )

    assert result.best_metrical_level == "HALF"
    assert result.f1_by_level["HALF"] == 1.0
    assert result.precision_by_level["HALF"] == 1.0
    assert result.recall_by_level["HALF"] == 1.0


def test_corroboration_penalizes_a_phase_shift_outside_the_official_window():
    aligned = measure_beat_corroboration(
        (OsuBpmEvent(0, 120.0),),
        _grid(),
        duration_ms=30_000,
    )
    shifted = measure_beat_corroboration(
        (OsuBpmEvent(200, 120.0),),
        _grid(),
        duration_ms=30_000,
    )

    assert shifted.best_f1 < aligned.best_f1
