from types import SimpleNamespace

import numpy as np
import pytest

from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.song_context import LocalTempoMap, SongAnalysisContext
from chart_worker.generation.osu_parser import OsuBpmEvent


def test_beats_between_integrates_each_tempo_segment():
    tempo = LocalTempoMap(
        (OsuBpmEvent(0, 120.0), OsuBpmEvent(1_000, 240.0))
    )

    assert tempo.beats_between(500, 1_500) == pytest.approx(3.0)


def test_same_timestamp_uses_last_input_timing_point():
    tempo = LocalTempoMap(
        (OsuBpmEvent(0, 120.0), OsuBpmEvent(0, 150.0))
    )

    assert tempo.events == (OsuBpmEvent(0, 150.0),)
    assert tempo.at(0).bpm == 150.0


def test_tempo_lookup_uses_the_active_event_at_a_boundary():
    tempo = LocalTempoMap(
        (OsuBpmEvent(-500, 100.0), OsuBpmEvent(1_000, 200.0))
    )

    assert tempo.at(999).bpm == 100.0
    assert tempo.at(1_000).bpm == 200.0


def test_song_context_builds_tempo_and_intro_evidence_once():
    analysis = OnsetAnalysis(
        sample_rate_hz=48_000,
        hop_length=512,
        strength=np.zeros(200),
        band_strength=np.zeros((3, 200)),
        onset_ms=(),
    )
    authority = SimpleNamespace(
        bpm_events=(OsuBpmEvent(0, 120.0),)
    )

    context = SongAnalysisContext.build(
        authority,
        analysis,
        duration_ms=2_000,
    )

    assert context.tempo_map.at(1_500).bpm == 120.0
    assert context.intro_anchor.status == "NON_RHYTHMIC"


def test_tempo_map_rejects_invalid_ranges_and_bpms():
    with pytest.raises(ValueError, match="at least one"):
        LocalTempoMap(())
    with pytest.raises(ValueError, match="bpm"):
        LocalTempoMap((OsuBpmEvent(0, 0.0),))
    tempo = LocalTempoMap((OsuBpmEvent(0, 120.0),))
    with pytest.raises(ValueError, match="end_ms"):
        tempo.beats_between(100, 99)
