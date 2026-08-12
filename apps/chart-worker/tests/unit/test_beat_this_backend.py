from pathlib import Path

import numpy as np

from chart_worker.analysis.beat_this_backend import BeatThisFileAnalyzer


def test_file_analyzer_reuses_one_loaded_model_for_multiple_songs():
    loads = []
    calls = []

    class FakeModel:
        def __call__(self, path):
            calls.append(path)
            beats = np.arange(0.0, 8.0, 0.5)
            return beats, beats[::4]

    def factory(checkpoint, device, float16):
        loads.append((checkpoint, device, float16))
        return FakeModel()

    analyzer = BeatThisFileAnalyzer(
        checkpoint="final0",
        device="cpu",
        float16=False,
        model_factory=factory,
    )

    first = analyzer(Path("first.flac"))
    second = analyzer(Path("second.flac"))

    assert loads == [("final0", "cpu", False)]
    assert calls == [Path("first.flac"), Path("second.flac")]
    assert len(first.beat_ms) == len(second.beat_ms) == 16
