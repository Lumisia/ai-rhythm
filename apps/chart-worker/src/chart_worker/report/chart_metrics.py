"""후처리 결과를 chart-v1 metrics로 조립한다."""

from chart_worker.postprocess.patterns import (
    detect_patterns,
    pattern_entropy,
    pattern_histogram,
)
from chart_worker.rating.project_rating import measure_rating
from chart_worker.report.alignment import AlignmentReport
from chart_worker.schema.chart import ChartMetrics
from chart_worker.schema.note import Chart
from chart_worker.schema.types import LaneSemantic, lane_semantics


def build_chart_metrics(
    notes: Chart,
    *,
    duration_ms: int,
    key_mode: int,
    beat_ms: float,
    alignment: AlignmentReport,
    moved_note_ratio: float,
) -> ChartMetrics:
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if beat_ms <= 0:
        raise ValueError("beat_ms must be positive")
    if not 0.0 <= moved_note_ratio <= 1.0:
        raise ValueError("moved_note_ratio must be a ratio between 0 and 1")

    semantics = lane_semantics(key_mode)
    rating = measure_rating(notes, duration_ms)
    instances = detect_patterns(notes, key_mode=key_mode, beat_ms=beat_ms)
    side_lanes = {
        index
        for index, semantic in enumerate(semantics)
        if semantic in (LaneSemantic.SIDE_LEFT, LaneSemantic.SIDE_RIGHT)
    }
    side_notes = [note for note in notes if note.lane in side_lanes]
    side_holds = sum(1 for note in side_notes if note.kind == "HOLD")

    return ChartMetrics(
        note_count=rating.note_count,
        hold_count=rating.hold_count,
        avg_nps=rating.avg_nps,
        p95_nps=rating.p95_nps,
        peak_nps=rating.peak_nps,
        chord_ratio=rating.chord_ratio,
        max_jack=rating.max_jack,
        project_rating=rating.rating,
        project_tier=rating.tier,
        pattern_entropy=pattern_entropy(pattern_histogram(instances)),
        drum_coverage=alignment.drum_coverage,
        drum_precision=alignment.drum_precision,
        mean_abs_err_ms=alignment.mean_abs_err_ms,
        side_note_ratio=round(len(side_notes) / len(notes), 4) if notes else 0.0,
        side_hold_ratio=round(side_holds / len(side_notes), 4) if side_notes else 0.0,
        moved_note_ratio=moved_note_ratio,
    )
