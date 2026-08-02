import dataclasses

import numpy as np
import pytest

from chart_worker.schema.note import NoteEvent
from chart_worker.schema.types import LaneSemantic, lane_semantics


def test_note_is_frozen():
    note = NoteEvent(time_ms=1000, lane=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        note.time_ms = 2000  # type: ignore[misc]


def test_origin_lane_survives_lane_move():
    note = NoteEvent(time_ms=0, lane=0)
    moved = dataclasses.replace(note, lane=2)
    assert (moved.origin_lane, moved.lane) == (0, 2)


def test_lane_layouts():
    assert lane_semantics(4) == list(LaneSemantic)[0:4]
    assert lane_semantics(6)[0] is LaneSemantic.SIDE_LEFT
    assert lane_semantics(7)[3] is LaneSemantic.CENTER


def test_rejects_unsupported_key_mode():
    with pytest.raises(ValueError, match="unsupported key_mode"):
        lane_semantics(5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"time_ms": -1, "lane": 0}, "time_ms"),
        ({"time_ms": 0, "lane": -1}, "lane"),
        ({"time_ms": 0, "lane": 0, "kind": "HOLD"}, "duration_ms"),
        ({"time_ms": 0, "lane": 0, "kind": "HOLD", "duration_ms": 0}, "duration_ms"),
        ({"time_ms": 0, "lane": 0, "duration_ms": 100}, "duration_ms"),
        ({"time_ms": 0, "lane": 0, "kind": "SLIDE"}, "kind"),
        ({"time_ms": 0.5, "lane": 0}, "time_ms"),
        ({"time_ms": True, "lane": 0}, "time_ms"),
        ({"time_ms": 0, "lane": 1.5}, "lane"),
        ({"time_ms": 0, "lane": False}, "lane"),
        ({"time_ms": 0, "lane": 0, "kind": "HOLD", "duration_ms": 10.5}, "duration_ms"),
        ({"time_ms": 0, "lane": 0, "origin_lane": 1.5}, "origin_lane"),
        ({"time_ms": 0, "lane": 0, "origin_lane": -2}, "origin_lane"),
    ],
)
def test_rejects_invalid_note_state(kwargs, message):
    with pytest.raises(ValueError, match=message):
        NoteEvent(**kwargs)


def test_accepts_numpy_integers_and_normalizes_them():
    """분석 단계가 넘기는 numpy 스칼라를 받아 파이썬 int 로 정규화한다."""
    note = NoteEvent(
        time_ms=np.int64(1200),
        lane=np.int32(2),
        kind="HOLD",
        duration_ms=np.int64(400),
        origin_lane=np.int16(0),
    )
    values = (note.time_ms, note.lane, note.duration_ms, note.origin_lane)
    assert values == (1200, 2, 400, 0)
    assert all(type(value) is int for value in values)


def test_numpy_origin_lane_sentinel_still_defaults_to_lane():
    assert NoteEvent(time_ms=0, lane=np.int64(3)).origin_lane == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"time_ms": np.float64(1200.0), "lane": 0}, "time_ms"),
        ({"time_ms": 0, "lane": np.float32(1.0)}, "lane"),
        (
            {"time_ms": 0, "lane": 0, "kind": "HOLD", "duration_ms": np.float64(400.0)},
            "duration_ms",
        ),
        ({"time_ms": np.bool_(True), "lane": 0}, "time_ms"),
    ],
)
def test_rejects_numpy_non_integers(kwargs, message):
    with pytest.raises(ValueError, match=message):
        NoteEvent(**kwargs)
