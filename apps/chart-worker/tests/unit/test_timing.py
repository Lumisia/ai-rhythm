import numpy as np
import pytest

from chart_worker.analysis.beat import build_beat_grid
from chart_worker.analysis.timing import fit_piecewise_timing, match_times


def test_piecewise_fit_splits_a_real_tempo_change_at_a_downbeat():
    first = np.arange(32) * 0.5
    second = first[-1] + np.arange(1, 33) * 0.6
    beats = np.concatenate([first, second])
    grid = build_beat_grid(beats, beats[::4])

    points = fit_piecewise_timing(grid)

    assert len(points) == 2
    assert points[1].start_beat_index == 32
    assert points[0].bpm == pytest.approx(120.0, abs=0.05)
    assert points[1].bpm == pytest.approx(100.0, abs=0.05)


def test_match_times_is_one_to_one_and_reports_f1_at_the_window():
    result = match_times((100, 101, 300), (100, 300), window_ms=20)

    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == 1.0
    assert result.f1 == pytest.approx(0.8)
