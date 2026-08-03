import numpy as np
import pytest

from chart_worker.analysis.activity import AudioActivity, build_audio_activity


def test_builds_song_relative_active_onsets():
    activity = build_audio_activity(
        rms_db=np.array([-70.0, -30.0, -20.0, -10.0]),
        normalized_strength=np.array([0.0, 0.2, 0.5, 1.0]),
        onset_frames=np.array([1, 2, 3]),
        frame_ms=10.0,
        silence_db=-60.0,
    )

    assert activity.floor_db == pytest.approx(-26.0)
    assert activity.active_onset_ms == (20, 30)


def test_active_frame_ratio_clips_to_available_frames():
    activity = AudioActivity(
        frame_ms=10.0,
        rms_db=np.array([-30.0, -10.0, -10.0]),
        floor_db=-20.0,
        active_onset_ms=(),
    )

    assert activity.active_frame_ratio(0, 30) == pytest.approx(2 / 3)
    assert activity.active_frame_ratio(-100, 10_000) == pytest.approx(2 / 3)


def test_silent_audio_has_no_active_onsets_or_frames():
    activity = build_audio_activity(
        rms_db=np.full(4, -80.0),
        normalized_strength=np.zeros(4),
        onset_frames=np.array([], dtype=np.int64),
        frame_ms=10.0,
    )

    assert activity.active_onset_ms == ()
    assert activity.active_frame_ratio(0, 40) == 0.0


def test_onset_window_uses_the_same_forward_smear_as_note_sampling():
    activity = build_audio_activity(
        rms_db=np.array([-70.0, -30.0, -10.0, -70.0]),
        normalized_strength=np.array([0.0, 0.1, 1.0, 0.0]),
        onset_frames=np.array([1]),
        frame_ms=10.0,
        n_fft=2,
        hop_length=1,
    )

    assert activity.active_onset_ms == (10,)
