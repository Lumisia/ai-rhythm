"""실제 librosa 로 도는 onset 분석 검증.

타격 시각을 아는 신호를 numpy 로 만든다. 저장소에 바이너리도,
ffmpeg 도 필요 없다.
"""

import numpy as np
import pytest

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.analysis.beat import build_beat_grid
from chart_worker.analysis.onset import (
    BAND_EDGES_HZ,
    analyze_onsets,
    annotate_notes,
    librosa_backend,
    mel_band_channels,
)
from chart_worker.schema.note import NoteEvent

pytestmark = pytest.mark.beat_this

pytest.importorskip("librosa", reason="analysis extra is not installed")

SAMPLE_RATE = 48_000
HIT_SEC = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)


def _hit(frequency: float, seconds: float, decay: float = 0.03) -> np.ndarray:
    span = np.arange(int(seconds * SAMPLE_RATE))
    envelope = np.exp(-span / (decay * SAMPLE_RATE))
    return np.sin(2 * np.pi * frequency * span / SAMPLE_RATE) * envelope


@pytest.fixture(scope="module")
def backend():
    return librosa_backend()


@pytest.fixture(scope="module")
def struck_signal():
    """0.5초 간격 타격. 저역·중역·고역을 번갈아 친다."""
    total = int(4.0 * SAMPLE_RATE)
    track = np.zeros(total)
    for index, at in enumerate(HIT_SEC):
        frequency = (60.0, 700.0, 6000.0)[index % 3]
        hit = _hit(frequency, 0.2)
        start = int(at * SAMPLE_RATE)
        end = min(start + hit.size, total)
        track[start:end] += hit[: end - start] * 0.8
    return AudioSignal(samples=np.column_stack([track, track]), sample_rate_hz=SAMPLE_RATE)


def test_band_edges_map_to_distinct_mel_bins():
    channels = mel_band_channels(SAMPLE_RATE)
    assert channels[0] == 0
    assert channels[-1] == 128
    assert channels == sorted(set(channels))


def test_band_edges_that_collapse_are_rejected():
    with pytest.raises(ValueError, match="same mel bin"):
        mel_band_channels(SAMPLE_RATE, edges_hz=(250.0, 251.0))


def test_envelopes_are_normalized(struck_signal, backend):
    analysis = analyze_onsets(struck_signal, backend=backend)
    assert 0.0 <= analysis.strength.min()
    assert analysis.strength.max() == pytest.approx(1.0)
    assert analysis.band_strength.shape[0] == 3
    assert analysis.band_strength.shape[1] == analysis.frame_count


def test_frame_spacing_beats_the_judgment_window(struck_signal, backend):
    analysis = analyze_onsets(struck_signal, backend=backend)
    assert analysis.frame_ms == pytest.approx(10.667, abs=0.01)
    assert analysis.frame_ms < 22, "판정 창보다 촘촘해야 한다"


def test_detected_onsets_land_on_the_hits(struck_signal, backend):
    analysis = analyze_onsets(struck_signal, backend=backend)
    detected = np.asarray(analysis.onset_ms)
    for at in HIT_SEC:
        nearest = np.abs(detected - at * 1000).min()
        assert nearest <= 30, f"{at}s 타격 근처에 onset 이 없다"


def test_strength_is_higher_on_hits_than_between_them(struck_signal, backend):
    """solver 순위 함수로 쓰려면 타격과 공백이 갈려야 한다."""
    analysis = analyze_onsets(struck_signal, backend=backend)
    on_hits = [analysis.strength_at(round(at * 1000)) for at in HIT_SEC]
    between = [analysis.strength_at(round((at + 0.25) * 1000)) for at in HIT_SEC[:-1]]
    assert min(on_hits) > max(between)


def test_band_follows_the_struck_register(struck_signal, backend):
    analysis = analyze_onsets(struck_signal, backend=backend)
    labels = [analysis.band_at(round(at * 1000)) for at in HIT_SEC]
    assert labels[0] == "LOW", f"60 Hz 타격이 {labels[0]} 로 분류됐다"
    assert labels[2] == "HIGH", f"6 kHz 타격이 {labels[2]} 로 분류됐다"
    assert BAND_EDGES_HZ == (250.0, 2000.0)


def test_annotation_produces_a_usable_chart(struck_signal, backend):
    analysis = analyze_onsets(struck_signal, backend=backend)
    beats = np.arange(8) * 0.5
    grid = build_beat_grid(beats, beats[::4])
    notes = [NoteEvent(time_ms=round(at * 1000), lane=index % 4) for index, at in enumerate(HIT_SEC)]

    annotated = annotate_notes(notes, onsets=analysis, grid=grid)
    assert [note.time_ms for note in annotated] == [note.time_ms for note in notes]
    assert all(note.onset_strength is not None for note in annotated)
    assert all(note.band in ("LOW", "MID", "HIGH") for note in annotated)
    # 다운비트는 0.0s 와 2.0s 다. 타격은 0.5s 부터 시작한다.
    assert [note.is_downbeat for note in annotated] == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]


def test_detected_onsets_are_backtracked_to_the_attack(struck_signal, backend):
    """flux 피크는 어택보다 늦게 뜬다. 되돌리지 않으면 키음이 밀린다."""
    detected = np.asarray(analyze_onsets(struck_signal, backend=backend).onset_ms)
    errors = [abs(detected - at * 1000).min() for at in HIT_SEC]
    assert max(errors) <= 10
    assert float(np.mean(errors)) <= 6
