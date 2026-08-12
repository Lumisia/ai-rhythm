from chart_worker.analysis.hold_lane_state import analyze_hold_lane_state
from chart_worker.generation.resnap_diagnostics import ResnapDiagnostics
from chart_worker.schema.note import NoteEvent


def test_hold_lane_state_accepts_release_then_note_at_the_same_time():
    trace = analyze_hold_lane_state(
        [
            NoteEvent(100, 0, "HOLD", 200),
            NoteEvent(300, 0),
            NoteEvent(400, 1, "HOLD", 100),
        ],
        ResnapDiagnostics.unobserved(),
    )

    assert trace.status == "PASS"
    assert trace.hold_count == 2
    assert trace.tap_count == 1
    assert trace.transition_count == 5
    assert trace.violations == ()
    assert trace.sidecar_evidence_status == "UNAVAILABLE"


def test_hold_lane_state_reports_same_lane_activity_before_release():
    trace = analyze_hold_lane_state(
        [
            NoteEvent(100, 2, "HOLD", 300),
            NoteEvent(250, 2),
        ],
        ResnapDiagnostics.unobserved(),
    )

    assert trace.status == "VIOLATION"
    assert trace.violations[0].code == "NOTE_WHILE_HOLD_ACTIVE"
    assert trace.violations[0].lane == 2
    assert trace.violations[0].time_ms == 250
    assert trace.violations[0].active_hold_end_ms == 400


def test_hold_lane_state_report_is_explicitly_shadow_only():
    report = analyze_hold_lane_state(
        [NoteEvent(0, 0)],
        ResnapDiagnostics.unobserved(),
    ).to_report()

    assert report["version"] == "hold-lane-state-shadow-v1"
    assert report["enforcement"] == "SHADOW"
    assert report["status"] == "PASS"
