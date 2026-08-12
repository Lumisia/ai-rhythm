import numpy as np
import pytest

from chart_worker.analysis.activity import (
    AudioActivity,
    build_audio_activity,
    build_song_boundary_contract,
    estimate_music_end_ms,
    evaluate_boundary_policy,
    evaluate_outro_policy,
    observe_outro,
)


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


def _intro_fixture(*, prominent: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """조용한 인트로 10초 + 본편 90초. 프레임 100 ms."""
    intro = np.full(100, -45.0)
    main = np.linspace(-20.0, -10.0, 900)
    rms_db = np.concatenate([intro, main])
    if prominent:
        rms_db[30] = -30.0  # 인트로 안에서 확실히 두드러지는 타점
    strength = np.zeros(1_000)
    strength[30] = 1.0
    strength[400] = 1.0
    return rms_db, strength, np.array([30, 400])


def test_locally_prominent_intro_onset_stays_active_below_the_song_floor():
    """인트로가 본편보다 작게 믹스돼도 명확한 타점은 active 여야 한다.

    곡 전체 RMS 하위 20 퍼센타일 floor 하나로 판정하면, onset strength 가
    1.0 인 인트로 타점까지 quiet 로 분류된다. 24곡 배치에서 22곡 중
    21곡의 leadingCoverage 가 이 이유로 QUIET_LEADING_TIMING_GAP PASS 가
    됐고, 사용자가 지적한 인트로 미정렬 8곡을 어떤 게이트도 못 잡았다.
    """
    rms_db, strength, onset_frames = _intro_fixture(prominent=True)

    activity = build_audio_activity(
        rms_db=rms_db,
        normalized_strength=strength,
        onset_frames=onset_frames,
        frame_ms=100.0,
    )

    # 전역 floor 는 본편 기준이라 -45 dB 인트로를 통째로 걸러내지만,
    # 국소 대비가 큰 타점 하나는 살아남는다.
    assert activity.floor_db > -45.0
    assert 3_000 in activity.active_onset_ms
    assert 40_000 in activity.active_onset_ms


def test_quiet_ambient_intro_without_local_contrast_stays_inactive():
    """대비 없는 앰비언트 도입부는 그대로 조용한 인트로로 둔다."""
    rms_db, strength, onset_frames = _intro_fixture(prominent=False)

    activity = build_audio_activity(
        rms_db=rms_db,
        normalized_strength=strength,
        onset_frames=onset_frames,
        frame_ms=100.0,
    )

    assert 3_000 not in activity.active_onset_ms
    assert 40_000 in activity.active_onset_ms


def test_music_end_trims_only_a_long_trailing_silence():
    frame_ms = 100.0
    rms_db = np.concatenate([np.full(200, -10.0), np.full(100, -80.0)])
    activity = AudioActivity(
        frame_ms=frame_ms,
        rms_db=rms_db,
        floor_db=-60.0,
        active_onset_ms=(),
    )

    assert estimate_music_end_ms(activity, 30_000) == 21_000


def test_music_end_keeps_full_audio_for_a_short_tail():
    frame_ms = 100.0
    rms_db = np.concatenate([np.full(290, -10.0), np.full(10, -80.0)])
    activity = AudioActivity(
        frame_ms=frame_ms,
        rms_db=rms_db,
        floor_db=-60.0,
        active_onset_ms=(),
    )

    assert estimate_music_end_ms(activity, 30_000) == 30_000


def test_music_end_trims_a_short_tail_only_with_absolute_silence():
    activity = AudioActivity(
        frame_ms=100.0,
        rms_db=np.concatenate([np.full(260, -10.0), np.full(40, -80.0)]),
        floor_db=-60.0,
        active_onset_ms=(1_000, 25_000),
    )

    assert estimate_music_end_ms(activity, 30_000) == 27_000


def test_music_end_keeps_a_short_tail_with_audible_residual_energy():
    activity = AudioActivity(
        frame_ms=100.0,
        rms_db=np.concatenate([np.full(260, -10.0), np.full(40, -50.0)]),
        floor_db=-40.0,
        active_onset_ms=(1_000, 25_000),
    )

    assert estimate_music_end_ms(activity, 30_000) == 30_000


def test_music_end_never_precedes_the_last_active_onset():
    """A locally prominent onset is music evidence even below the global RMS floor."""
    activity = AudioActivity(
        frame_ms=100.0,
        rms_db=np.concatenate([np.full(200, -10.0), np.full(100, -80.0)]),
        floor_db=-60.0,
        active_onset_ms=(1_000, 24_000),
    )

    assert estimate_music_end_ms(activity, 30_000) == 25_000


def test_music_end_releases_after_last_onset_when_short_tail_is_silent():
    activity = AudioActivity(
        frame_ms=100.0,
        rms_db=np.concatenate([np.full(200, -10.0), np.full(100, -80.0)]),
        floor_db=-60.0,
        active_onset_ms=(1_000, 26_000),
    )

    assert estimate_music_end_ms(activity, 30_000) == 27_000


def test_music_end_keeps_full_audio_when_activity_is_unavailable():
    activity = AudioActivity(
        frame_ms=0.0,
        rms_db=np.array([]),
        floor_db=-60.0,
        active_onset_ms=(),
    )

    assert estimate_music_end_ms(activity, 30_000) == 30_000


def test_outro_observation_records_selected_and_legacy_tail_policies():
    activity = AudioActivity(
        frame_ms=1_000,
        rms_db=np.asarray([-20.0] * 26 + [-80.0] * 4),
        floor_db=-40.0,
        active_onset_ms=(1_000, 25_000),
    )

    observation = observe_outro(activity, 30_000)
    decision = evaluate_outro_policy(observation)

    assert observation.last_active_rms_end_ms == 26_000
    assert observation.last_detected_onset_ms == 25_000
    assert observation.last_evidence_ms == 26_000
    assert observation.trailing_below_floor_ms == 4_000
    assert decision.selected_music_end_ms == 27_000
    assert decision.last_attack_ms == 26_000
    assert decision.release_end_ms == 30_000
    assert decision.generation_end_ms == 30_000
    assert decision.policy_reason == "SHORT_ABSOLUTE_SILENCE"
    assert decision.policy_applied is True
    assert "selectedMusicEndMs" not in observation.to_report()
    assert "policyReason" not in observation.to_report()


def test_outro_observation_release_candidate_includes_late_onset_evidence():
    activity = AudioActivity(
        frame_ms=1_000,
        rms_db=np.asarray([-20.0] * 20 + [-80.0] * 10),
        floor_db=-40.0,
        active_onset_ms=(1_000, 24_000),
    )

    observation = observe_outro(activity, 30_000)
    decision = evaluate_outro_policy(observation)

    assert observation.last_active_rms_end_ms == 20_000
    assert observation.last_detected_onset_ms == 24_000
    assert observation.last_evidence_ms == 24_000
    assert observation.trailing_below_floor_ms == 10_000
    assert decision.selected_music_end_ms == 25_000
    assert decision.last_attack_ms == 24_000
    assert decision.policy_reason == "LONG_TRAILING_SILENCE"
    assert decision.observation_sha256 == observation.stable_sha256()


def test_song_boundary_separates_note_start_from_hold_completion_horizon():
    activity = AudioActivity(
        frame_ms=100.0,
        rms_db=np.concatenate([np.full(840, -10.0), np.full(45, -80.0)]),
        floor_db=-60.0,
        active_onset_ms=(1_000, 84_100),
    )

    contract = build_song_boundary_contract(
        activity,
        88_500,
        enforcement_mode="EXPERIMENTAL_ENFORCED",
    )

    assert contract.last_attack_ms == 84_100
    assert contract.required_coverage_end_ms == 84_100
    assert contract.release_end_ms == 88_500
    assert contract.generation_end_ms == 88_500
    assert contract.max_note_start_ms == 84_170
    assert contract.policy_applied is True


def test_uncalibrated_boundary_defaults_to_shadow_full_duration():
    activity = AudioActivity(
        frame_ms=100.0,
        rms_db=np.concatenate([np.full(200, -10.0), np.full(100, -80.0)]),
        floor_db=-60.0,
        active_onset_ms=(1_000, 19_900),
    )

    evaluation = evaluate_boundary_policy(activity, 30_000)

    assert evaluation.policy_state == "PROVISIONAL"
    assert evaluation.confidence == "UNKNOWN"
    assert evaluation.enforcement_mode == "SHADOW"
    # RMS evidence covers the frame ending at 20,000ms, then adds 70ms.
    assert evaluation.provisional_contract.max_note_start_ms == 20_070
    assert evaluation.full_duration_contract.max_note_start_ms == 30_000
    assert evaluation.effective_source == "FULL_DURATION_BASELINE"
    assert evaluation.effective_contract == evaluation.full_duration_contract
    assert evaluation.effective_contract.policy_applied is False
    assert evaluation.provisional_decision.policy_applied is True
    assert evaluation.to_report()["observationSha256"] == (
        evaluation.provisional_decision.observation_sha256
    )


def test_experimental_boundary_enforces_the_same_provisional_candidate():
    activity = AudioActivity(
        frame_ms=100.0,
        rms_db=np.concatenate([np.full(200, -10.0), np.full(100, -80.0)]),
        floor_db=-60.0,
        active_onset_ms=(1_000, 19_900),
    )

    evaluation = evaluate_boundary_policy(
        activity,
        30_000,
        enforcement_mode="EXPERIMENTAL_ENFORCED",
    )

    assert evaluation.effective_source == "PROVISIONAL_POLICY"
    assert evaluation.effective_contract == evaluation.provisional_contract
    assert evaluation.effective_contract.max_note_start_ms == 20_070
    assert evaluation.effective_contract.policy_applied is True


def test_boundary_report_separates_policy_state_confidence_and_enforcement():
    activity = AudioActivity(
        frame_ms=1_000.0,
        rms_db=np.asarray([-20.0] * 20 + [-80.0] * 10),
        floor_db=-40.0,
        active_onset_ms=(1_000, 19_000),
    )

    report = evaluate_boundary_policy(activity, 30_000).to_report()

    assert report["policyState"] == "PROVISIONAL"
    assert report["confidence"] == "UNKNOWN"
    assert report["enforcementMode"] == "SHADOW"
    assert report["provisionalDecision"]["policyState"] == "PROVISIONAL"
    assert report["provisionalDecision"]["confidence"] == "UNKNOWN"
    assert report["effectiveContract"]["maxNoteStartMs"] == 30_000


def test_song_boundary_does_not_enforce_an_uncertain_short_tail():
    activity = AudioActivity(
        frame_ms=100.0,
        rms_db=np.concatenate([np.full(290, -10.0), np.full(10, -80.0)]),
        floor_db=-60.0,
        active_onset_ms=(1_000, 28_000),
    )

    contract = build_song_boundary_contract(activity, 30_000)

    assert contract.last_attack_ms == 30_000
    assert contract.release_end_ms == 30_000
    assert contract.generation_end_ms == 30_000
    assert contract.max_note_start_ms == 30_000
    assert contract.policy_applied is False
