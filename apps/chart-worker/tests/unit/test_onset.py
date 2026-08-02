import numpy as np
import pytest

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.analysis.beat import build_beat_grid
from chart_worker.analysis.onset import (
    BAND_NAMES,
    OnsetAnalysis,
    analyze_onsets,
    annotate_notes,
    normalize_envelope,
)
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.schema.note import NoteEvent

SAMPLE_RATE = 48_000
HOP = 512
FRAME_MS = HOP * 1000 / SAMPLE_RATE  # 10.6667


BAND_PEAK_FRAMES = (0, 12, 24)
"""창 폭(-1 ~ +4 프레임)보다 멀리 떼어 놓아야 서로 섞이지 않는다."""


def _analysis(strength=None, bands=None, onset_ms=(), n_fft=2048, frames=36):
    if strength is None:
        strength = np.zeros(frames)
        strength[BAND_PEAK_FRAMES[1]] = 1.0
    if bands is None:
        bands = np.zeros((3, frames))
        for row, frame in enumerate(BAND_PEAK_FRAMES):
            bands[row, frame] = 1.0
    return OnsetAnalysis(
        sample_rate_hz=SAMPLE_RATE,
        hop_length=HOP,
        strength=np.asarray(strength),
        band_strength=np.asarray(bands),
        onset_ms=tuple(onset_ms),
        n_fft=n_fft,
    )


def test_normalization_divides_by_the_percentile_and_clips():
    values = np.array([0.0, 1.0, 2.0, 100.0])
    normalized = normalize_envelope(values, percentile=75)
    assert normalized.max() == 1.0
    assert normalized[0] == 0.0


def test_normalization_uses_a_percentile_so_one_hit_cannot_flatten_the_rest():
    """최대값으로 나누면 한 번의 큰 타격이 나머지를 전부 눌러버린다."""
    values = np.concatenate([np.full(999, 1.0), np.array([1000.0])])
    by_max = values / values.max()
    by_p99 = normalize_envelope(values)
    assert by_max[0] == pytest.approx(0.001)
    assert by_p99[0] == pytest.approx(1.0)


def test_normalization_of_a_silent_envelope_is_zero():
    assert normalize_envelope(np.zeros(8)).tolist() == [0.0] * 8


def test_normalization_of_an_empty_envelope_is_empty():
    assert normalize_envelope(np.array([])).size == 0


def test_frame_lookup_rounds_to_the_nearest_frame():
    analysis = _analysis()
    assert analysis.frame_of(0) == 0
    assert analysis.frame_of(round(FRAME_MS)) == 1
    assert analysis.frame_of(round(FRAME_MS * 2)) == 2


def test_frame_lookup_clamps_outside_the_signal():
    analysis = _analysis()
    assert analysis.frame_of(-500) == 0
    assert analysis.frame_of(10_000_000) == analysis.frame_count - 1


def test_frame_lookup_on_an_empty_analysis_is_an_error():
    empty = _analysis(strength=np.array([]), bands=np.zeros((3, 0)))
    with pytest.raises(ValueError, match="no frames"):
        empty.frame_of(0)


def test_strength_is_sampled_at_the_note_time():
    analysis = _analysis()
    assert analysis.strength_at(round(FRAME_MS * BAND_PEAK_FRAMES[1])) == 1.0


@pytest.mark.parametrize(("frame", "expected"), list(zip(BAND_PEAK_FRAMES, ("LOW", "MID", "HIGH"))))
def test_band_is_the_strongest_row(frame, expected):
    assert _analysis().band_at(round(FRAME_MS * frame)) == expected


def test_strength_window_leans_forward_not_backward():
    """spectral flux 는 어택 직후에 뜬다. 절대 앞서지 않는다."""
    analysis = _analysis()
    window = analysis.window_of(round(FRAME_MS * 10))
    assert window.start == 9
    assert window.stop == 15  # 10 + ceil(2048/512) + 1


def test_a_peak_one_frame_late_is_still_found():
    """단일 프레임을 읽으면 가장 강한 타격이 강도 0 으로 나온다."""
    strength = np.zeros(36)
    strength[11] = 0.9
    analysis = _analysis(strength=strength)
    assert analysis.strength[10] == 0.0
    assert analysis.strength_at(round(FRAME_MS * 10)) == 0.9


def test_a_peak_beyond_the_smear_is_not_borrowed():
    strength = np.zeros(36)
    strength[16] = 0.9
    assert _analysis(strength=strength).strength_at(round(FRAME_MS * 10)) == 0.0


def test_window_shrinks_with_a_smaller_transform():
    analysis = _analysis(n_fft=HOP)
    window = analysis.window_of(round(FRAME_MS * 10))
    assert (window.start, window.stop) == (9, 12)


def test_band_names_match_the_row_order():
    assert BAND_NAMES == ("LOW", "MID", "HIGH")


def test_analyze_onsets_hands_mono_audio_to_the_backend():
    seen = {}

    def backend(mono, sample_rate_hz):
        seen["ndim"] = mono.ndim
        seen["rate"] = sample_rate_hz
        return _analysis()

    analyze_onsets(AudioSignal(np.zeros((100, 2)), SAMPLE_RATE), backend=backend)
    assert seen == {"ndim": 1, "rate": SAMPLE_RATE}


def test_backend_failure_becomes_a_worker_error():
    def backend(mono, sample_rate_hz):
        raise RuntimeError("numba exploded")

    with pytest.raises(WorkerError) as caught:
        analyze_onsets(AudioSignal(np.zeros((100, 2)), SAMPLE_RATE), backend=backend)
    assert caught.value.code is ErrorCode.CHART_ANALYSIS_FAILED


# --- annotate_notes -----------------------------------------------------------

BPM = 120.0
BEAT_SEC = 60.0 / BPM


def _grid(beats=16):
    times = np.arange(beats) * BEAT_SEC
    return build_beat_grid(times, times[::4])


def _wide_analysis(frames=400):
    strength = np.linspace(0.0, 1.0, frames)
    bands = np.zeros((3, frames))
    bands[0, :] = 1.0
    return _analysis(strength=strength, bands=bands)


def test_annotation_never_moves_a_note():
    """타이밍 불변 원칙은 분석 주석에도 적용된다."""
    notes = [NoteEvent(time_ms=t, lane=t % 4) for t in (0, 250, 500, 1234)]
    annotated = annotate_notes(notes, onsets=_wide_analysis(), grid=_grid())
    assert [note.time_ms for note in annotated] == [0, 250, 500, 1234]
    assert [note.lane for note in annotated] == [note.lane for note in notes]


def test_annotation_fills_strength_and_band():
    annotated = annotate_notes(
        [NoteEvent(time_ms=1000, lane=0)], onsets=_wide_analysis(), grid=_grid()
    )
    assert 0.0 <= annotated[0].onset_strength <= 1.0
    assert annotated[0].band == "LOW"


def test_downbeats_are_flagged_within_the_window():
    notes = [NoteEvent(time_ms=t, lane=0) for t in (0, 40, 60, 2000)]
    annotated = annotate_notes(notes, onsets=_wide_analysis(), grid=_grid())
    assert [note.is_downbeat for note in annotated] == [True, True, False, True]


def test_beat_fraction_is_zero_on_the_beat_and_half_offbeat():
    beat_ms = round(BEAT_SEC * 1000)
    notes = [NoteEvent(time_ms=t, lane=0) for t in (0, beat_ms // 2, beat_ms, beat_ms + 125)]
    annotated = annotate_notes(notes, onsets=_wide_analysis(), grid=_grid())
    fractions = [note.beat_fraction for note in annotated]
    assert fractions[0] == pytest.approx(0.0)
    assert fractions[1] == pytest.approx(0.5)
    assert fractions[2] == pytest.approx(0.0)
    assert fractions[3] == pytest.approx(0.25)


def test_notes_beyond_the_last_beat_still_get_a_fraction():
    grid = _grid(beats=4)
    late = grid.beat_ms[-1] + 5_000
    [note] = annotate_notes([NoteEvent(time_ms=late, lane=0)], onsets=_wide_analysis(), grid=grid)
    assert note.beat_fraction is not None


def test_annotation_keeps_hold_notes_intact():
    notes = [NoteEvent(time_ms=500, lane=1, kind="HOLD", duration_ms=250)]
    [note] = annotate_notes(notes, onsets=_wide_analysis(), grid=_grid())
    assert (note.kind, note.duration_ms) == ("HOLD", 250)


def test_annotation_preserves_origin_lane():
    notes = [NoteEvent(time_ms=0, lane=2, origin_lane=0)]
    [note] = annotate_notes(notes, onsets=_wide_analysis(), grid=_grid())
    assert (note.lane, note.origin_lane) == (2, 0)


def test_annotating_an_empty_chart_is_empty():
    assert annotate_notes([], onsets=_wide_analysis(), grid=_grid()) == []
