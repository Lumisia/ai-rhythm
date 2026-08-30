import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chart_worker import pipeline
from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.intro_anchor import IntroAnchorEvidence
from chart_worker.analysis.terminal_silence import (
    TerminalSilenceObservation,
    TerminalThresholdCandidate,
)
from chart_worker.config import WorkerConfig
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.attempt_journal import AttemptJournal
from chart_worker.generation.diagnostic_fallback import DiagnosticRawCandidate
from chart_worker.generation.fake import FakeGenerator
from chart_worker.generation.inference_session import SessionState, SongIdentity
from chart_worker.generation.mapperatorinator_patch import CONSTRAINT_PATCH_ID
from chart_worker.generation.resnap_diagnostics import RESNAP_DIAGNOSTICS_VERSION
from chart_worker.hashing import sha256_file
from chart_worker.pipeline import PipelineOptions, run_pipeline
from chart_worker.schema.chart import ChartDocument
from chart_worker.schema.playtest_run import PlaytestRunManifestV3
from chart_worker.stages.s2_timing import run_timing_generation
from chart_worker.stages.timing_feedback import (
    MapTimingFailureSignature,
    RetryTimingSignal,
)
from chart_worker.stages.types import MissingVariant
from chart_worker.validation.leading_timing_coverage import LeadingTimingCoverage
from chart_worker.validation.quality_gate import QUALITY_GATE_VERSION, GateAction, GateAxis
from chart_worker.validation.timing_review import (
    TimingAuthorityAction,
    TimingAuthorityReview,
)
from tests.support import fake_dependencies

_FAKE_UNCORROBORATED_INTRO_ANCHOR_REPORT = {
    "status": "UNCERTAIN",
    "anchorMs": 0,
    "anchorGridMs": 0,
    "gridDistanceMs": 0,
    "aggregatePercentileRank": 0.625,
    "prominentBandCount": 0,
    "pulseContinuationMatches": 4,
    "pulseContinuationOpportunities": 4,
    "supportedPulseMs": list(range(0, 4_001, 250)),
}


def test_family_reassigned_primary_is_playtest_only_recovery() -> None:
    variant = SimpleNamespace(
        provenance="PRIMARY",
        family_assignment_kind="REASSIGNED",
        acceptance=SimpleNamespace(action=GateAction.PASS),
    )

    assert pipeline._variant_is_playtest_only(variant) is True
    assert pipeline._playability_tier(variant) == "RECOVERY_PLAYABLE"


class _RecordingInferenceSession:
    def __init__(self) -> None:
        self.state = SessionState.IDLE
        self.events: list[object] = []

    def begin_song(self, identity: SongIdentity) -> None:
        assert self.state is SessionState.IDLE
        self.events.append(("begin", identity))
        self.state = SessionState.SONG_ACTIVE

    def invoke(self, argv, workdir):
        raise AssertionError("pipeline lifecycle test does not invoke inference directly")

    def end_song(self) -> None:
        assert self.state is SessionState.SONG_ACTIVE
        self.events.append("end")
        self.state = SessionState.IDLE

    def close(self) -> None:
        self.events.append("close")
        self.state = SessionState.CLOSED


def _song_session_config() -> WorkerConfig:
    return WorkerConfig(
        mapperatorinator_backend="song_session",
        mapperatorinator_hold_state_mode="incremental",
        mapperatorinator_home=Path(r"C:\Mapperatorinator"),
        mapperatorinator_python=Path(r"C:\Mapperatorinator\.venv\Scripts\python.exe"),
        mapperatorinator_model_root=Path(r"C:\models\mapperatorinator-v32"),
        mapperatorinator_model_revision="a" * 40,
    )


def test_pipeline_propagates_difficulty_shadow_challenger_opt_in(
    monkeypatch,
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    base = dependencies.prepare(
        source,
        tmp_path / "base",
        WorkerConfig(),
    )
    monkeypatch.setattr(pipeline, "run_prepare", lambda *_args, **_kwargs: base)

    prepared = pipeline._prepare_stage(
        source,
        tmp_path / "run",
        WorkerConfig(difficulty_shadow_challenger_enabled=True),
    )

    assert prepared.difficulty_shadow_challenger_enabled is True


def test_pipeline_propagates_difficulty_family_compiler_shadow_opt_in(
    monkeypatch,
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    base = dependencies.prepare(
        source,
        tmp_path / "base",
        WorkerConfig(),
    )
    monkeypatch.setattr(pipeline, "run_prepare", lambda *_args, **_kwargs: base)

    prepared = pipeline._prepare_stage(
        source,
        tmp_path / "run",
        WorkerConfig(difficulty_family_compiler_shadow_enabled=True),
    )

    assert prepared.difficulty_family_compiler_shadow_enabled is True


def test_pipeline_propagates_enforced_unique_family_resolution(
    monkeypatch,
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    base = dependencies.prepare(source, tmp_path / "base", WorkerConfig())
    monkeypatch.setattr(pipeline, "run_prepare", lambda *_args, **_kwargs: base)

    prepared = pipeline._prepare_stage(
        source,
        tmp_path / "run",
        WorkerConfig(difficulty_family_resolution_enabled=False),
    )

    assert prepared.difficulty_family_resolution_enabled is False


def test_pipeline_owns_one_song_session_across_timing_and_maps(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    session = _RecordingInferenceSession()
    opened: list[Path] = []
    bound: list[object] = []
    dependencies = fake_dependencies()

    result = run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="mapperatorinator",
        ),
        dependencies=replace(
            dependencies,
            config=_song_session_config(),
            open_inference_session=lambda _config, run_dir: opened.append(run_dir) or session,
            bind_inference_session=lambda generator, observed: (
                bound.append((generator, observed)) or generator
            ),
        ),
    )

    assert result.output_dir == tmp_path / "run"
    assert opened == [tmp_path / "run"]
    assert len(bound) == 1 and bound[0][1] is session
    assert session.events[0][0] == "begin"
    identity = session.events[0][1]
    assert identity.audio_sha256 == sha256_file(tmp_path / "run" / "audio" / "game.flac")
    assert session.events[1:] == ["end", "close"]


def test_pipeline_does_not_close_attached_container_session(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    session = _RecordingInferenceSession()
    dependencies = fake_dependencies()

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="mapperatorinator",
        ),
        dependencies=replace(
            dependencies,
            config=_song_session_config(),
            attached_inference_session=session,
            open_inference_session=lambda _config, _run_dir: pytest.fail(
                "attached sessions must not open a child"
            ),
            bind_inference_session=lambda generator, _session: generator,
        ),
    )

    assert session.events[-1] == "end"
    assert "close" not in session.events
    assert session.state is SessionState.IDLE


def test_pipeline_closes_owned_session_on_base_exception(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    session = _RecordingInferenceSession()
    dependencies = fake_dependencies()

    def interrupted_timing(*_args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=tmp_path / "run",
                title="fixture",
                generator="mapperatorinator",
            ),
            dependencies=replace(
                dependencies,
                config=_song_session_config(),
                timing=interrupted_timing,
                open_inference_session=lambda _config, _run_dir: session,
                bind_inference_session=lambda generator, _session: generator,
            ),
        )

    assert session.events[-2:] == ["end", "close"]
    assert session.state is SessionState.CLOSED


def test_unknown_resident_completion_is_reported_as_infra_without_another_generation(
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    calls = 0
    unknown = WorkerError(
        ErrorCode.INFERENCE_COMPLETION_UNKNOWN,
        "accepted invocation lost its terminal record",
        context={
            "accepted": True,
            "invocationId": "a" * 64,
            "requestHash": "b" * 64,
        },
    )

    def generation(*_args):
        nonlocal calls
        calls += 1
        raise unknown

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=tmp_path / "run",
                title="fixture",
                generator="mapperatorinator",
            ),
            dependencies=replace(dependencies, generation=generation),
        )

    assert captured.value is unknown
    assert calls == 1
    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    assert report["status"] == "FAILED"
    assert report["failureStage"] == "GENERATION"
    assert report["outcomeStatusV2"]["failureCategory"] == "INFRA"
    assert report["error"] == {
        "code": "INFERENCE_COMPLETION_UNKNOWN",
        "context": unknown.context,
    }


def test_map_unknown_after_timing_returns_twelve_safe_playtest_charts(
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    delegate = FakeGenerator()

    class UnknownMapGenerator:
        def __init__(self) -> None:
            self.map_calls = 0

        def generate_timing(self, request, workdir):
            return delegate.generate_timing(request, workdir)

        def generate_map(self, request, workdir):
            del request, workdir
            self.map_calls += 1
            raise WorkerError(
                ErrorCode.INFERENCE_COMPLETION_UNKNOWN,
                "accepted MAP invocation lost its terminal record",
                context={
                    "accepted": True,
                    "invocationId": "a" * 64,
                    "requestHash": "b" * 64,
                },
            )

    generator = UnknownMapGenerator()
    result = run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
            seed=7,
        ),
        dependencies=replace(
            dependencies,
            select_generator=lambda _name, _config: generator,
        ),
    )

    manifest = PlaytestRunManifestV3.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )
    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    assert generator.map_calls == 1
    assert len(manifest.charts) == 12
    assert manifest.missing_charts == []
    assert {(chart.key_mode, chart.difficulty) for chart in manifest.charts} == {
        (key_mode, difficulty)
        for key_mode in (4, 6, 7)
        for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
    }
    assert all(chart.provenance == "SAFE_FALLBACK" for chart in manifest.charts)
    assert all(
        "productionEligible" not in chart.model_dump(by_alias=True)
        and "distributionTier" not in chart.model_dump(by_alias=True)
        for chart in manifest.charts
    )
    assert report["status"] == "REVIEW"


def test_direct_pipeline_writes_twelve_unmodified_charts(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"

    result = run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=output_dir,
            title="fixture",
            generator="fake",
            seed=7,
        ),
        dependencies=fake_dependencies(),
    )

    assert result.manifest_path == output_dir / "playtest-run-v3.json"
    assert not (output_dir / "playtest-run-v2.json").exists()
    manifest = PlaytestRunManifestV3.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )
    assert len(manifest.charts) == 12
    assert len(result.raw_osu_paths) == 12
    assert set(result.elapsed_ms_by_stage) == {
        "prepare",
        "analysis",
        "timing",
        "generation",
        "export",
    }
    assert manifest.audio.no_drums is None
    assert manifest.audio.keys is None
    assert manifest.keysound_manifest_path is None
    assert all(chart.provenance == "PRIMARY" for chart in manifest.charts)
    assert all(chart.family_resolution_state == "RESOLVED" for chart in manifest.charts)
    assert all(
        "productionEligible" not in chart.model_dump(by_alias=True)
        and "distributionTier" not in chart.model_dump(by_alias=True)
        for chart in manifest.charts
    )
    assert not (output_dir / "analysis").exists()

    report = json.loads((output_dir / "generation-report.json").read_text())
    assert manifest.generation_report.path == "generation-report.json"
    assert manifest.generation_report.sha256 == sha256_file(output_dir / "generation-report.json")
    assert manifest.outcome.model_dump(by_alias=True) == report["outcomeStatusV2"]
    assert manifest.publication.model_dump(by_alias=True) == report["publicationDecision"]
    assert manifest.strict_blockers == ["BOUNDARY_POLICY_UNCALIBRATED"]
    assert report["boundaryPublicationAssessment"] == {
        "version": "boundary-publication-assessment-v1",
        "evidenceStatus": "UNAVAILABLE",
        "policyState": None,
        "confidence": None,
        "strictBlockers": ["BOUNDARY_POLICY_UNCALIBRATED"],
    }
    assert manifest.publication.decision == "PLAYTEST_ONLY"
    assert report["strategy"] == "MAPPERATORINATOR_SHARED_TIMING"
    assert report["additionalInferenceWorkMs"] == 0
    assert report["additionalInferenceWorkLimitMs"] == 10_000
    assert report["difficultyFamilyCompilerShadow"] == []
    assert report["difficultyFamilyResolution"] == []
    assert report["safeFamilyAssignmentPolicy"] == {
        "version": "safe-family-assignment-v3",
        "payloadUniqueness": "HARD_PUBLICATION_INVARIANT",
        "familyFeasibility": "HARD_BEFORE_RANKING",
        "relativeDifficultyMode": "RELABEL_UNIQUE_THEN_BOUNDED_RESOLUTION",
        "resolutionOrder": [
            "EXISTING_UNIQUE_RELABEL",
            "DETERMINISTIC_FAMILY_COMPILER",
            "ONE_CALL_TARGETED_GENERATION",
            "COHERENT_CANONICAL_FAMILY_FALLBACK",
            "FAIL_CLOSED",
        ],
        "maxConfirmedIntroDelayBeats": 8.0,
        "maxSafeSubstitutesPerKey": 20,
        "additionalModelCallsMax": 1,
    }
    assert report["timingAuthority"] == "audio/timing-reference.osu"
    assert report["timingAuthoritySha256"] == sha256_file(
        output_dir / "audio" / "timing-reference.osu"
    )
    assert report["timingGenerationMode"] == "STANDARD"
    assert report["timingAttemptCount"] == 1
    assert report["timingAuthorityTempoMetrics"] == {
        "basePulseSupport": 1.0,
        "halfPulseSupport": 1.0,
        "doublePulseSupport": 0.5,
        "baseSupportedPulses": 20,
        "halfSupportedPulses": 10,
        "doubleSupportedPulses": 20,
        "pulseBestAlternative": None,
        "pulseAlternativeMargin": 0.0,
        "basePeriodicitySupport": 1.0,
        "halfPeriodicitySupport": 1.0,
        "doublePeriodicitySupport": 0.0,
        "periodicityFrameCount": 100,
        "periodicityBestAlternative": None,
        "periodicityMargin": 0.0,
        "evidenceAgrees": False,
        "evidenceStatus": "SUFFICIENT",
    }
    assert report["timingAuthorityReview"] == {
        "action": "PASS",
        "reasons": ["TEMPO_CANDIDATE_AMBIGUOUS"],
    }
    assert report["timingAuthorityLeadingCoverage"] == {
        "action": "PASS",
        "reasons": [],
        "firstEventTimeMs": 0,
        "leadingDurationMs": 0,
        "onsetCount": 0,
        "activeOnsetCount": 0,
        "activeFrameRatio": 0.0,
        "introAnchor": _FAKE_UNCORROBORATED_INTRO_ANCHOR_REPORT,
    }
    assert report["timingAuthorityLocalReview"]["version"] == (
        "local-timing-review-v2-duration-weighted"
    )
    assert report["timingAuthoritySelection"]["selectedMode"] == "STANDARD"
    assert report["timingAuthoritySelection"]["reason"] == ("ONLY_STRUCTURALLY_VALID_CANDIDATE")
    assert report["timingAuthorityLocalReview"]["action"] == "PASS"
    assert report["timingAuthorityRecoveryPreflight"]["version"] == ("recovery-preflight-v1")
    assert report["timingAuthorityIntegrity"] == {
        "version": "timing-integrity-v1",
        "status": "HEALTHY",
        "reasons": [],
        "islands": [],
    }
    assert report["noteMutationEnabled"] is False
    assert report["mapperatorinatorConstraintPatch"] is None
    assert report["mapperatorinatorHoldStateMode"] is None
    assert report["attemptsPerChartMax"] == 6
    assert report["qualityAttemptsPerChartMax"] == 3
    assert report["crashAttemptsPerChartMax"] == 3
    assert report["canonicalAudioSha256"] == manifest.audio.game.sha256
    assert report["runtimeFingerprint"]["canonicalAudioSha256"] == (manifest.audio.game.sha256)
    assert (
        report["runtimeFingerprint"]["timingAuthoritySha256"] == (report["timingAuthoritySha256"])
    )
    assert report["runtimeFingerprint"]["id"].startswith("sha256:")
    assert report["runtimeFingerprint"]["evidenceGrade"] == "VERIFIED_CODE"
    assert report["qualityGateVersion"] == QUALITY_GATE_VERSION
    assert report["publishable"] is False
    assert report["publicationDecision"] == {
        "policyVersion": "PUBLICATION_POLICY_V2",
        "decision": "PLAYTEST_ONLY",
        "reasonCodes": ["BOUNDARY_POLICY_UNCALIBRATED"],
    }
    assert report["status"] == "PASS"
    assert report["timingReviewRequired"] is False
    assert report["selectedAuthorityEpoch"] == 1
    assert report["timingCandidates"] == [
        {
            "epoch": 1,
            "authoritySha256": report["timingAuthoritySha256"],
            "mode": "STANDARD",
            "status": "SELECTED",
            "escalation": None,
        }
    ]
    assert report["mapTimingEscalations"] == []
    assert report["resnapCollisions"] == []
    assert report["elapsedMsByStage"] == result.elapsed_ms_by_stage
    assert len(report["charts"]) == 12
    assert set(report["difficultyOrder"]) == {"4K", "6K", "7K"}
    for key_mode in (4, 6, 7):
        family_observation = report["difficultyOrder"][f"{key_mode}K"]["finalFamilyObservation"]
        assert family_observation["keyMode"] == key_mode
        assert family_observation["calibrationState"] == "PILOT_ONLY"
        assert family_observation["contractStatus"] == "UNCALIBRATED"
        assert family_observation["productionCalibrationEnforced"] is False
        assert family_observation["provisionalConcern"] == "NONE"
        assert family_observation["resolutionStatus"] == "NO_OBSERVED_CONCERN"
        assert family_observation["policyState"] == "REPORTING_ENFORCED"
        assert family_observation["mutatesSelection"] is False
        assert family_observation["mutatesCharts"] is False
        assert family_observation["mutatesQualityStatus"] is True
        assert family_observation["mutatesCharts"] is False
        assert [entry["difficulty"] for entry in family_observation["entries"]] == [
            "EASY",
            "NORMAL",
            "HARD",
            "EXPERT",
        ]
    assert len(report["difficultySelectionShadow"]) == 3
    assert all(
        comparison["mode"] == "SHADOW_V2" for comparison in report["difficultySelectionShadow"]
    )
    assert report["songSelectionShadow"]["mode"] == "SHADOW_V2"
    assert report["songSelectionShadow"]["contextId"] != "EMPTY"
    assert report["songSelectionShadow"]["songFamiliesEvaluated"] > 0
    assert (
        report["songSelectionShadow"]["currentAssignment"]
        == (report["songSelectionShadow"]["shadowAssignment"])
    )
    replay_input = report["songSelectionShadow"]["replayInput"]
    assert replay_input["version"] == "song-selection-replay-v2"
    protected_metrics = replay_input["candidates"][0]["protectedMetrics"]
    assert {
        "rowCount",
        "onsetCount",
        "matchedCount50",
        "matchedPrecision50",
        "matchedRecall50",
        "matchedF150",
    } <= set(protected_metrics)
    v3_evidence = report["songSelectionEvidenceV3"]
    assert v3_evidence["version"] == "song-selection-evidence-v3"
    assert v3_evidence["mutatesSelection"] is False
    assert v3_evidence["additionalModelCalls"] == 0
    assert len(v3_evidence["candidates"]) == len(replay_input["candidates"])
    assert len(report["songSelectionEvidenceV3Sha256"]) == 64
    assert report["songSelectionShadowV3"]["mode"] == "SHADOW_V3"
    assert report["songSelectionShadowV3"]["mutatesSelection"] is False
    assert report["songSelectionShadowV3"]["blockers"] == ["CALIBRATION_UNAVAILABLE"]
    assert report["introStartContract"]["version"] == "intro-start-contract-v3"
    assert report["introContractReview"]["status"] == "PASS"
    assert len(report["introPhraseFamilyReviews"]) == 3
    assert all(review["status"] == "PASS" for review in report["introPhraseFamilyReviews"])
    assert report["outroFamilyReview"]["version"] == "outro-family-review-v3-tiered-start-shadow"
    assert report["outroFamilyReview"]["mode"] == "SHADOW"
    assert report["outroFamilyReview"]["mutatesCharts"] is False
    assert report["outroFamilyReview"]["additionalInferenceCalls"] == 0
    assert report["additionalInferenceCalls"] <= 1
    assert all(review["status"] == "PASS" for review in report["difficultyOrder"].values())

    for chart_report, chart_ref, raw_path in zip(
        report["charts"], manifest.charts, result.raw_osu_paths, strict=True
    ):
        document = ChartDocument.model_validate_json(
            (output_dir / chart_ref.path).read_text(encoding="utf-8")
        )
        assert chart_report["rawNoteCount"] == len(document.notes)
        assert chart_report["finalNoteCount"] == len(document.notes)
        assert chart_report["attemptCount"] == 1
        assert chart_report["candidateCount"] == 1
        assert chart_report["generationAttemptCount"] == 1
        assert chart_report["provenance"] == "PRIMARY"
        assert chart_report["recoveryReason"] is None
        assert "recoveryPlan" not in chart_report
        assert chart_report["selectedSeed"] == chart_report["seed"]
        assert chart_report["attemptErrors"] == []
        assert chart_report["timingDiagnostics"]["status"] == "PASS"
        assert chart_report["acceptanceStatus"] == "PASS"
        assert chart_report["acceptanceReasons"] == ["INSUFFICIENT_PROFILE_VARIATION"]
        assert set(chart_report["noteGrid"]) == {
            "uniqueRowCount",
            "cleanRowCount",
            "cleanRate",
            "absoluteP95Beats",
        }
        assert chart_report["noteGrid"]["uniqueRowCount"] == chart_report["rawNoteCount"]
        assert chart_report["noteGrid"]["cleanRowCount"] == chart_report["rawNoteCount"]
        assert chart_report["noteGrid"]["cleanRate"] == 1.0
        assert 0.0 <= chart_report["noteGrid"]["absoluteP95Beats"] <= 0.025
        assert set(chart_report["acceptanceDecisions"]) == {
            "STRUCTURE",
            "SONG_BOUNDS",
            "TIMING_IDENTITY",
            "TIMING_ALIGNMENT",
            "COVERAGE",
            "PATTERN",
        }
        assert set(chart_report) >= {
            "difficultyProfile",
            "difficultyVectorV2",
            "holdProfile",
            "patternProfile",
        }
        assert chart_report["difficultyProfile"]["projectRating"] > 0
        assert chart_report["difficultyVectorV2"]["orderingScore"] > 0
        assert "sectionOccupancyRatios" in chart_report["holdProfile"]
        assert "sectionLaneImbalances" in chart_report["patternProfile"]
        assert "activeOnsetCount" in chart_report["timingDiagnostics"]
        assert "quietCoverageGaps" in chart_report["timingDiagnostics"]
        assert chart_report["cfgScale"] == 1.0
        assert chart_report["resnapDiagnostics"] == {
            "version": RESNAP_DIAGNOSTICS_VERSION,
            "status": "UNOBSERVED",
            "osuSha256": None,
            "collisions": [],
            "maniaObjects": [],
            "duplicates": [],
            "error": None,
        }
        assert chart_report["holdLaneStateTrace"] == {
            "version": "hold-lane-state-shadow-v1",
            "enforcement": "SHADOW",
            "status": "PASS",
            "laneCount": chart_report["keyMode"],
            "holdCount": chart_report["holdCount"],
            "tapCount": chart_report["rawNoteCount"] - chart_report["holdCount"],
            "transitionCount": chart_report["rawNoteCount"] + chart_report["holdCount"],
            "violations": [],
            "sidecarEvidenceStatus": "UNAVAILABLE",
            "sidecarObjectCount": 0,
            "sidecarHoldCount": 0,
            "originBackedHoldCount": 0,
            "sidecarMismatch": None,
        }
        assert (
            chart_report["descriptors"]
            == {
                "EASY": ["expression/simple"],
                "NORMAL": ["style/mixed rice"],
                "HARD": ["style/mixed rice", "streams/bursts"],
                "EXPERT": [
                    "style/mixed rice",
                    "skillset/streams",
                ],
            }[chart_report["difficulty"]]
        )
        assert chart_report["chartPath"] == chart_ref.path
        assert chart_report["rawOsuPath"] == raw_path.relative_to(output_dir).as_posix()
        assert "candidates" not in chart_report
        assert raw_path.parent == output_dir / "raw"


def test_success_report_uses_null_for_unavailable_optional_timing_evidence(
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def timing(prepared, analysis, run_dir, generator, seed):
        authority = dependencies.timing(prepared, analysis, run_dir, generator, seed)
        return replace(authority, tempo_metrics=None, review=None, leading_coverage=None)

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
        ),
        dependencies=replace(dependencies, timing=timing),
    )

    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    assert report["timingAuthorityTempoMetrics"] is None
    assert report["timingAuthorityReview"] is None
    assert report["timingAuthorityLeadingCoverage"] is None


def test_authority_review_sets_review_required_when_all_maps_pass(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def timing(prepared, analysis, run_dir, generator, seed):
        authority = dependencies.timing(prepared, analysis, run_dir, generator, seed)
        return replace(
            authority,
            review=TimingAuthorityReview(
                TimingAuthorityAction.REVIEW,
                ("SHORT_ACTIVE_LEADING_TIMING_GAP",),
            ),
            leading_coverage=LeadingTimingCoverage(
                action=TimingAuthorityAction.REVIEW,
                reasons=("SHORT_ACTIVE_LEADING_TIMING_GAP",),
                first_event_time_ms=2_678,
                leading_duration_ms=2_678,
                onset_count=11,
                active_onset_count=11,
                active_frame_ratio=1.0,
                intro_anchor=IntroAnchorEvidence(
                    status="CONFIRMED",
                    anchor_ms=100,
                    anchor_grid_ms=0,
                    grid_distance_ms=100,
                    aggregate_percentile_rank=0.95,
                    prominent_band_count=3,
                    pulse_continuation_matches=2,
                    pulse_continuation_opportunities=4,
                ),
            ),
        )

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
        ),
        dependencies=replace(dependencies, timing=timing),
    )

    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    assert all(chart["acceptanceStatus"] == "PASS" for chart in report["charts"])
    assert report["timingAuthorityReview"]["action"] == "REVIEW"
    assert report["timingReviewRequired"] is True


@pytest.mark.parametrize(
    ("code", "status", "generator", "hold_state_mode"),
    [
        (ErrorCode.CHART_TIMING_REVIEW_REQUIRED, "REVIEW", "fake", None),
        (ErrorCode.CHART_TIMING_REVIEW_REQUIRED, "REVIEW", "mapperatorinator", "incremental"),
        (ErrorCode.CHART_CANDIDATES_EXHAUSTED, "EXHAUSTED", "fake", None),
        (ErrorCode.CHART_CANDIDATES_EXHAUSTED, "EXHAUSTED", "mapperatorinator", "incremental"),
        (ErrorCode.CHART_VALIDATION_FAILED, "FAILED", "fake", None),
        (ErrorCode.CHART_VALIDATION_FAILED, "FAILED", "mapperatorinator", "incremental"),
    ],
)
def test_withheld_generation_writes_failure_report_without_publishable_artifacts(
    code: ErrorCode, status: str, generator: str, hold_state_mode: str | None, tmp_path: Path
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    dependencies = fake_dependencies()
    if generator == "mapperatorinator":
        dependencies = replace(
            dependencies,
            config=dependencies.config.model_copy(
                update={"mapperatorinator_hold_state_mode": "incremental"}
            ),
        )
    export_calls = []
    failure = WorkerError(
        code,
        "withheld fixture",
        context={"seed": 7, "attempts": [{"seed": 7, "gateReport": {"action": status}}]},
    )

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        del prepared, authority, analysis, run_dir, generator, seed, authority_epoch
        raise failure

    def export(prepared, generated, run_dir, worker_version):
        export_calls.append(run_dir)
        return dependencies.export(prepared, generated, run_dir, worker_version)

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=output_dir,
                title="fixture",
                generator=generator,
                seed=7,
            ),
            dependencies=replace(
                dependencies,
                generation=generation,
                export=export,
            ),
        )

    assert captured.value is failure
    report = json.loads((output_dir / "generation-report.json").read_text())
    assert report["version"] == 1
    assert report["qualityGateVersion"] == QUALITY_GATE_VERSION
    assert report["runId"] == "00000000-0000-0000-0000-000000000007"
    assert report["publishable"] is False
    assert report["publicationDecision"] == {
        "policyVersion": "PUBLICATION_POLICY_V2",
        "decision": "REJECTED",
        "reasonCodes": [
            "BOUNDARY_POLICY_UNCALIBRATED",
            "EXECUTION_FAILED",
            "INCOMPLETE_CHART_SET",
            "QUALITY_UNKNOWN",
            "STRICT_OUTCOME_FALSE",
        ],
    }
    assert report["status"] == status
    assert report["mapperatorinatorHoldStateMode"] == hold_state_mode
    assert report["error"] == {
        "code": code.value,
        "context": failure.context,
    }
    assert report["canonicalAudioSha256"] == sha256_file(output_dir / "audio" / "game.flac")
    assert report["timingAuthority"] == "audio/timing-reference.osu"
    assert report["timingAuthoritySha256"] == sha256_file(
        output_dir / "audio" / "timing-reference.osu"
    )
    assert report["timingAuthorityTempoMetrics"]["basePulseSupport"] == 1.0
    assert report["timingAuthorityReview"] == {
        "action": "PASS",
        "reasons": ["TEMPO_CANDIDATE_AMBIGUOUS"],
    }
    assert report["timingAuthorityLeadingCoverage"] == {
        "action": "PASS",
        "reasons": [],
        "firstEventTimeMs": 0,
        "leadingDurationMs": 0,
        "onsetCount": 0,
        "activeOnsetCount": 0,
        "activeFrameRatio": 0.0,
        "introAnchor": _FAKE_UNCORROBORATED_INTRO_ANCHOR_REPORT,
    }
    assert report["timingAuthorityLocalReview"]["version"] == (
        "local-timing-review-v2-duration-weighted"
    )
    assert report["timingAuthorityRecoveryPreflight"]["version"] == ("recovery-preflight-v1")
    assert export_calls == []
    assert not (output_dir / "charts").exists()
    assert not (output_dir / "playtest-run-v2.json").exists()
    assert not (output_dir / "playtest-run-v3.json").exists()


def test_v2_only_difficulty_inversion_forces_review_instead_of_false_pass(
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    dependencies = fake_dependencies()

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        outcome = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )
        changed = []
        for variant in outcome.variants:
            if (variant.key_mode, variant.difficulty) != (4, "EXPERT"):
                changed.append(variant)
                continue
            profile = variant.acceptance.profile
            assert profile is not None
            changed.append(
                replace(
                    variant,
                    acceptance=replace(
                        variant.acceptance,
                        profile=replace(
                            profile,
                            difficulty_vector_v2=replace(
                                profile.difficulty_vector_v2,
                                ordering_score=-1.0,
                            ),
                        ),
                    ),
                )
            )
        return replace(outcome, variants=tuple(changed))

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=output_dir,
            title="fixture",
            generator="fake",
            seed=7,
        ),
        dependencies=replace(dependencies, generation=generation),
    )

    report = json.loads((output_dir / "generation-report.json").read_text())
    observation = report["difficultyOrder"]["4K"]["finalFamilyObservation"]
    assert observation["provisionalConcern"] == "METRIC_DISAGREEMENT"
    assert observation["resolutionStatus"] == "UNRESOLVED"
    assert observation["requiresReview"] is True
    assert report["difficultyReviewRequired"] is True
    assert report["timingReviewRequired"] is False
    assert report["status"] == "REVIEW"
    assert report["outcomeStatusV2"]["quality"] == "REVIEW"


def test_generation_failure_report_projects_durable_attempt_journal(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    dependencies = fake_dependencies()

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        del prepared, authority, analysis, generator, seed
        journal = AttemptJournal(run_dir / "attempt-journal.jsonl")
        journal.append(
            event_type="INFERENCE_STARTED",
            authority_epoch=authority_epoch,
            key_mode=4,
            difficulty="EASY",
            attempt=1,
            seed=7,
        )
        journal.append(
            event_type="INFERENCE_FAILED",
            authority_epoch=authority_epoch,
            key_mode=4,
            difficulty="EASY",
            attempt=1,
            seed=7,
            payload={"code": "CHART_GENERATION_FAILED"},
        )
        raise WorkerError(
            ErrorCode.CHART_CANDIDATES_EXHAUSTED,
            "fixture exhausted",
        )

    with pytest.raises(WorkerError):
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=output_dir,
                title="fixture",
                generator="fake",
                seed=7,
            ),
            dependencies=replace(dependencies, generation=generation),
        )

    report = json.loads((output_dir / "generation-report.json").read_text())
    projection = report["attemptJournal"]
    assert projection["status"] == "AVAILABLE"
    assert projection["recordCount"] == 2
    assert projection["eventCounts"] == {
        "INFERENCE_FAILED": 1,
        "INFERENCE_STARTED": 1,
    }
    assert [record["sequence"] for record in projection["records"]] == [1, 2]


def test_pipeline_rejects_missing_difficulty_order_before_export(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    dependencies = fake_dependencies()
    export_calls = []

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        outcome = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )
        variants = outcome.variants
        return replace(
            outcome,
            variants=(replace(variants[0], difficulty_order=None), *variants[1:]),
        )

    def export(prepared, generated, run_dir, worker_version):
        export_calls.append(run_dir)
        return dependencies.export(prepared, generated, run_dir, worker_version)

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=output_dir,
                title="fixture",
                generator="fake",
            ),
            dependencies=replace(
                dependencies,
                generation=generation,
                export=export,
            ),
        )

    assert captured.value.code is ErrorCode.CHART_VALIDATION_FAILED
    assert captured.value.context == {
        "missingDifficultyOrder": [{"keyMode": 4, "difficulty": "EASY"}]
    }
    report = json.loads((output_dir / "generation-report.json").read_text())
    assert report["publishable"] is False
    assert report["status"] == "FAILED"
    assert report["error"] == {
        "code": "CHART_VALIDATION_FAILED",
        "context": captured.value.context,
    }
    assert export_calls == []
    assert not (output_dir / "charts").exists()
    assert not (output_dir / "playtest-run-v2.json").exists()
    assert not (output_dir / "playtest-run-v3.json").exists()


@pytest.mark.parametrize(
    ("generator", "hold_state_mode"),
    [("fake", None), ("mapperatorinator", "incremental_verify")],
)
def test_timing_review_writes_pre_authority_failure_report_and_stops_pipeline(
    generator: str,
    hold_state_mode: str | None,
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    dependencies = fake_dependencies()
    if generator == "mapperatorinator":
        dependencies = replace(
            dependencies,
            config=dependencies.config.model_copy(
                update={"mapperatorinator_hold_state_mode": "incremental_verify"}
            ),
        )
    generation_calls = []
    export_calls = []
    failure = WorkerError(
        ErrorCode.CHART_TIMING_REVIEW_REQUIRED,
        "timing review fixture",
        context={
            "reasons": ("INSUFFICIENT_TEMPO_EVIDENCE",),
            "attempt_count": 1,
            "attempts": [{"attempt": 1, "seed": 7, "mode": "STANDARD"}],
        },
    )

    def timing(prepared, analysis, run_dir, generator, seed):
        del prepared, analysis, run_dir, generator, seed
        raise failure

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        generation_calls.append(run_dir)
        return dependencies.generation(
            prepared, authority, analysis, run_dir, generator, seed, authority_epoch
        )

    def export(prepared, generated, run_dir, worker_version):
        export_calls.append(run_dir)
        return dependencies.export(prepared, generated, run_dir, worker_version)

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=output_dir,
                title="fixture",
                generator=generator,
                seed=7,
            ),
            dependencies=replace(
                dependencies,
                timing=timing,
                generation=generation,
                export=export,
            ),
        )

    assert captured.value is failure
    report = json.loads((output_dir / "generation-report.json").read_text())
    assert report["publishable"] is False
    assert report["status"] == "REVIEW"
    assert report["failureStage"] == "TIMING"
    assert report["mapperatorinatorHoldStateMode"] == hold_state_mode
    assert report["error"] == {
        "code": "CHART_TIMING_REVIEW_REQUIRED",
        "context": {
            "reasons": ["INSUFFICIENT_TEMPO_EVIDENCE"],
            "attempt_count": 1,
            "attempts": [{"attempt": 1, "seed": 7, "mode": "STANDARD"}],
        },
    }
    assert report["canonicalAudioSha256"] == sha256_file(output_dir / "audio" / "game.flac")
    assert report["timingAuthority"] is None
    assert report["timingAuthoritySha256"] is None
    assert generation_calls == []
    assert export_calls == []
    assert not (output_dir / "audio" / "timing-reference.osu").exists()
    assert not (output_dir / "raw").exists()
    assert not (output_dir / "charts").exists()
    assert not (output_dir / "playtest-run-v2.json").exists()
    assert not (output_dir / "playtest-run-v3.json").exists()


def test_timing_candidates_exhausted_writes_attempt_evidence_before_reraising(
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    dependencies = fake_dependencies()
    failure = WorkerError(
        ErrorCode.CHART_TIMING_CANDIDATE_FAILED,
        "standard and Super Timing candidates missed active leading audio",
        context={
            "seeds": [7, 8],
            "attempts": [
                {
                    "mode": "STANDARD",
                    "leadingCoverage": {
                        "action": "RETRY_TIMING",
                        "reasons": ["ACTIVE_LEADING_TIMING_GAP"],
                    },
                },
                {
                    "mode": "SUPER_TIMING",
                    "leadingCoverage": {
                        "action": "RETRY_TIMING",
                        "reasons": ["ACTIVE_LEADING_TIMING_GAP"],
                    },
                },
            ],
        },
    )

    def timing(prepared, analysis, run_dir, generator, seed):
        del prepared, analysis, run_dir, generator, seed
        raise failure

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=output_dir,
                title="fixture",
                generator="fake",
                seed=7,
            ),
            dependencies=replace(dependencies, timing=timing),
        )

    assert captured.value is failure
    report = json.loads((output_dir / "generation-report.json").read_text())
    assert report["publishable"] is False
    assert report["status"] == "EXHAUSTED"
    assert report["failureStage"] == "TIMING"
    assert report["error"] == {
        "code": "CHART_TIMING_CANDIDATE_FAILED",
        "context": failure.context,
    }
    assert report["timingAuthority"] is None
    assert not (output_dir / "playtest-run-v2.json").exists()
    assert not (output_dir / "playtest-run-v3.json").exists()


@pytest.mark.parametrize(
    "provenance",
    ("PRIMARY", "RAW_UNVERIFIED", "SAFE_FALLBACK"),
)
def test_pipeline_rejects_retry_map_variant_returned_by_generation_stage(
    tmp_path: Path,
    provenance: str,
):
    action = GateAction.RETRY_MAP
    code = ErrorCode.CHART_CANDIDATES_EXHAUSTED
    status = "EXHAUSTED"
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    dependencies = fake_dependencies()
    export_calls = []
    rejected_acceptance = None

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        nonlocal rejected_acceptance
        outcome = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )
        variants = outcome.variants
        first, *remaining = variants[0].acceptance.decisions
        rejected_acceptance = replace(
            variants[0].acceptance,
            action=action,
            decisions=(
                replace(first, action=action, reasons=(f"FIXTURE_{action.value}",)),
                *remaining,
            ),
        )
        return replace(
            outcome,
            variants=(
                replace(
                    variants[0],
                    acceptance=rejected_acceptance,
                    provenance=provenance,
                ),
                *variants[1:],
            ),
        )

    def export(prepared, generated, run_dir, worker_version):
        export_calls.append(run_dir)
        return dependencies.export(prepared, generated, run_dir, worker_version)

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=output_dir,
                title="fixture",
                generator="fake",
            ),
            dependencies=replace(
                dependencies,
                generation=generation,
                export=export,
            ),
        )

    assert rejected_acceptance is not None
    assert captured.value.code is code
    assert captured.value.context == {
        "variants": [
            {
                "key_mode": 4,
                "difficulty": "EASY",
                "provenance": provenance,
                "gate_report": rejected_acceptance.to_report(),
            }
        ]
    }
    report = json.loads((output_dir / "generation-report.json").read_text())
    assert report["publishable"] is False
    assert report["status"] == status
    assert report["error"] == {
        "code": code.value,
        "context": captured.value.context,
    }
    assert export_calls == []
    assert not (output_dir / "charts").exists()
    assert not (output_dir / "playtest-run-v2.json").exists()
    assert not (output_dir / "playtest-run-v3.json").exists()


@pytest.mark.parametrize(
    ("provenance", "recovery_reason"),
    (
        ("RAW_UNVERIFIED", "QUALITY_GATE_REJECTED"),
        ("SAFE_FALLBACK", "NO_STRUCTURALLY_SAFE_MODEL_CANDIDATE"),
    ),
)
def test_pipeline_allows_hard_safe_fallback_only_as_playtest_output(
    tmp_path: Path,
    provenance: str,
    recovery_reason: str,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    dependencies = fake_dependencies()

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        outcome = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )
        first = outcome.variants[0]
        decisions = tuple(
            replace(
                decision,
                action=GateAction.RETRY_MAP,
                reasons=("FIXTURE_SOFT_QUALITY_REJECTION",),
            )
            if decision.axis is GateAxis.COVERAGE
            else decision
            for decision in first.acceptance.decisions
        )
        acceptance = replace(
            first.acceptance,
            action=GateAction.RETRY_MAP,
            decisions=decisions,
        )
        fallback = replace(
            first,
            acceptance=acceptance,
            provenance=provenance,
            recovery_reason=recovery_reason,
        )
        return replace(outcome, variants=(fallback, *outcome.variants[1:]))

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=output_dir,
            title="fixture",
            generator="fake",
            seed=7,
        ),
        dependencies=replace(dependencies, generation=generation),
    )

    report = json.loads((output_dir / "generation-report.json").read_text())
    assert len(report["charts"]) == 12
    assert report["charts"][0]["provenance"] == provenance
    assert report["charts"][0]["acceptanceStatus"] == "RETRY_MAP"
    assert "productionEligible" not in report["charts"][0]
    assert "distributionTier" not in report["charts"][0]
    assert report["status"] == "REVIEW"
    assert report["publishable"] is False
    assert report["publicationDecision"]["decision"] == "PLAYTEST_ONLY"
    manifest = PlaytestRunManifestV3.model_validate_json(
        (output_dir / "playtest-run-v3.json").read_text()
    )
    chart = next(
        item for item in manifest.charts if (item.key_mode, item.difficulty) == (4, "EASY")
    )
    assert chart.provenance == provenance
    assert "productionEligible" not in chart.model_dump(by_alias=True)
    assert "distributionTier" not in chart.model_dump(by_alias=True)


def test_pipeline_publishes_review_variant_with_warning(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    dependencies = fake_dependencies()
    export_calls = []

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        outcome = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )
        variants = outcome.variants
        first, *remaining = variants[0].acceptance.decisions
        review = replace(
            variants[0].acceptance,
            action=GateAction.REVIEW,
            decisions=(
                replace(first, action=GateAction.REVIEW, reasons=("FIXTURE_REVIEW",)),
                *remaining,
            ),
        )
        return replace(
            outcome,
            variants=(replace(variants[0], acceptance=review), *variants[1:]),
        )

    def export(prepared, generated, run_dir, worker_version):
        export_calls.append(run_dir)
        return dependencies.export(prepared, generated, run_dir, worker_version)

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=output_dir,
            title="fixture",
            generator="fake",
        ),
        dependencies=replace(
            dependencies,
            generation=generation,
            export=export,
        ),
    )

    report = json.loads((output_dir / "generation-report.json").read_text())
    assert report["publishable"] is False
    assert report["publicationDecision"] == {
        "policyVersion": "PUBLICATION_POLICY_V2",
        "decision": "PLAYTEST_ONLY",
        "reasonCodes": [
            "BOUNDARY_POLICY_UNCALIBRATED",
            "QUALITY_REVIEW_REQUIRED",
            "STRICT_OUTCOME_FALSE",
        ],
    }
    assert report["status"] == "PASS"
    assert report["timingReviewRequired"] is True
    assert report["charts"][0]["acceptanceStatus"] == "REVIEW"
    assert "FIXTURE_REVIEW" in report["charts"][0]["acceptanceReasons"]
    assert export_calls == [output_dir]
    assert (output_dir / "playtest-run-v3.json").is_file()
    assert not (output_dir / "playtest-run-v2.json").exists()


def test_generation_report_uses_recorded_acceptance_timing(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")

    assert not hasattr(pipeline, "diagnose_chart_timing")

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
        ),
        dependencies=fake_dependencies(),
    )


def test_pipeline_runs_shared_timing_before_twelve_maps_with_one_generator(
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    calls = []
    select_calls = []

    class RecordingGenerator:
        def __init__(self):
            self.delegate = FakeGenerator()
            self.timing_calls = 0
            self.map_calls = 0

        def generate_timing(self, request, workdir):
            self.timing_calls += 1
            return self.delegate.generate_timing(request, workdir)

        def generate_map(self, request, workdir):
            self.map_calls += 1
            return self.delegate.generate_map(request, workdir)

    generator = RecordingGenerator()

    def prepare(source_path, run_dir, config):
        calls.append("prepare")
        return dependencies.prepare(source_path, run_dir, config)

    def analyze(path):
        calls.append("analyze")
        return dependencies.analyze(path)

    def select_generator(name, config):
        select_calls.append((name, config))
        return generator

    def timing(prepared, analysis, run_dir, selected_generator, seed):
        calls.append("timing")
        assert selected_generator is generator
        return dependencies.timing(prepared, analysis, run_dir, selected_generator, seed)

    def generation(
        prepared,
        authority,
        analysis,
        run_dir,
        selected_generator,
        seed,
        authority_epoch,
    ):
        calls.append("generation")
        assert selected_generator is generator
        return dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            selected_generator,
            seed,
            authority_epoch,
        )

    def export(prepared, generated, run_dir, worker_version):
        calls.append("export")
        return dependencies.export(prepared, generated, run_dir, worker_version)

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
        ),
        dependencies=replace(
            dependencies,
            prepare=prepare,
            analyze=analyze,
            select_generator=select_generator,
            timing=timing,
            generation=generation,
            export=export,
        ),
    )

    assert calls == ["prepare", "analyze", "timing", "generation", "export"]
    assert len(select_calls) == 1
    assert generator.timing_calls == 1
    assert generator.map_calls == 12


def test_generation_report_records_the_selected_retry_attempt(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        outcome = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )
        variants = outcome.variants
        retried = replace(
            variants[0],
            generated=replace(variants[0].generated, seed=19),
            attempt=2,
            attempt_errors=("lane 4 is outside requested 4K",),
        )
        return replace(outcome, variants=(retried, *variants[1:]))

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
            seed=7,
        ),
        dependencies=replace(dependencies, generation=generation),
    )

    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    first = report["charts"][0]
    assert first["attemptCount"] == 2
    assert first["seed"] == 19
    assert first["attemptErrors"] == ["lane 4 is outside requested 4K"]


def test_generation_report_records_raw_unverified_provenance(tmp_path: Path):
    """품질 축을 우회한 raw 모델 출력은 provenance 와 곡 상태로 드러난다."""
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        outcome = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )
        variants = outcome.variants
        recovered = replace(
            variants[0],
            provenance="RAW_UNVERIFIED",
            recovery_reason="QUALITY_GATE_REJECTED",
        )
        return replace(outcome, variants=(recovered, *variants[1:]))

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
            seed=7,
        ),
        dependencies=replace(dependencies, generation=generation),
    )

    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    first = report["charts"][0]
    assert first["provenance"] == "RAW_UNVERIFIED"
    assert first["recoveryReason"] == "QUALITY_GATE_REJECTED"
    assert "productionEligible" not in first
    assert "distributionTier" not in first
    assert "recoveryPlan" not in first
    # raw 출력이 섞이면 곡을 PASS 로 올리지 않는다.
    assert report["status"] == "REVIEW"
    assert report["publishable"] is False
    assert report["publicationDecision"]["decision"] == "PLAYTEST_ONLY"
    manifest = PlaytestRunManifestV3.model_validate_json(
        (tmp_path / "run" / "playtest-run-v3.json").read_text(encoding="utf-8")
    )
    assert manifest.charts[0].playability_tier == "DIAGNOSTIC_ONLY"
    assert manifest.charts[0].coverage_summary is not None


def test_manifest_marks_coverage_repair_as_recovery_playable(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        outcome = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )
        first, *remaining = outcome.variants
        repaired = replace(
            first,
            provenance="COVERAGE_REPAIR",
            recovery_reason="ACTIVE_COVERAGE_GAP_REPAIRED",
            coverage_repair_gap_count=2,
        )
        return replace(outcome, variants=(repaired, *remaining))

    result = run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
            seed=7,
        ),
        dependencies=replace(dependencies, generation=generation),
    )

    manifest = PlaytestRunManifestV3.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )
    first = manifest.charts[0]
    assert first.provenance == "COVERAGE_REPAIR"
    assert "productionEligible" not in first.model_dump(by_alias=True)
    assert "distributionTier" not in first.model_dump(by_alias=True)
    assert first.playability_tier == "RECOVERY_PLAYABLE"
    assert first.coverage_summary is not None
    assert first.coverage_summary.repaired_gap_count == 2


def test_manifest_does_not_call_a_still_rejected_coverage_repair_playable(
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        outcome = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )
        first, *remaining = outcome.variants
        rejected = replace(
            first.acceptance,
            action=GateAction.RETRY_MAP,
            decisions=tuple(
                replace(
                    decision,
                    action=GateAction.RETRY_MAP,
                    reasons=("FIXTURE_TIMING_ALIGNMENT_RETRY",),
                )
                if decision.axis is GateAxis.TIMING_ALIGNMENT
                else decision
                for decision in first.acceptance.decisions
            ),
        )
        repaired = replace(
            first,
            acceptance=rejected,
            provenance="COVERAGE_REPAIR",
            recovery_reason="ACTIVE_COVERAGE_GAP_REPAIRED",
            coverage_repair_gap_count=2,
        )
        return replace(outcome, variants=(repaired, *remaining))

    result = run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
            seed=7,
        ),
        dependencies=replace(dependencies, generation=generation),
    )

    manifest = PlaytestRunManifestV3.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )
    assert manifest.charts[0].playability_tier == "DIAGNOSTIC_ONLY"


def test_generation_report_records_missing_charts_as_partial(tmp_path: Path):
    """조합 하나가 빠지면 나머지는 발행하고 곡 상태만 PARTIAL 로 낮춘다."""
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        outcome = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )
        dropped, *kept = outcome.variants
        return replace(
            outcome,
            variants=tuple(kept),
            missing=(
                MissingVariant(
                    key_mode=dropped.key_mode,
                    difficulty=dropped.difficulty,
                    reason="NO_PUBLISHABLE_CANDIDATE",
                ),
            ),
        )

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
            seed=7,
        ),
        dependencies=replace(dependencies, generation=generation),
    )

    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    assert report["status"] == "PARTIAL"
    assert report["publishable"] is False
    assert report["publicationDecision"] == {
        "policyVersion": "PUBLICATION_POLICY_V2",
        "decision": "PLAYTEST_ONLY",
        "reasonCodes": [
            "BOUNDARY_POLICY_UNCALIBRATED",
            "INCOMPLETE_CHART_SET",
            "QUALITY_REVIEW_REQUIRED",
            "STRICT_OUTCOME_FALSE",
        ],
    }
    assert report["availableCharts"] == 11
    assert len(report["charts"]) == 11
    assert report["missingCharts"] == [
        {
            "keyMode": 4,
            "difficulty": "EASY",
            "reason": "NO_PUBLISHABLE_CANDIDATE",
            "attemptErrors": [],
            "attemptEvidence": [],
        }
    ]


def test_generation_report_preserves_blocked_intro_phrase_defect(tmp_path: Path):
    from chart_worker.validation.intro_phrase_family import (
        IntroPhraseChartView,
        review_intro_phrase_pair,
    )

    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        outcome = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )
        dropped = next(
            variant
            for variant in outcome.variants
            if (variant.key_mode, variant.difficulty) == (4, "EXPERT")
        )
        review = review_intro_phrase_pair(
            IntroPhraseChartView(4, "HARD", 100, 300, 0.4),
            IntroPhraseChartView(4, "EXPERT", 0, 12_000, 24.0),
            start_delta_beats=0.2,
        )
        return replace(
            outcome,
            variants=tuple(variant for variant in outcome.variants if variant is not dropped),
            missing=(
                MissingVariant(
                    4,
                    "EXPERT",
                    "INTRO_PHRASE_DEFECT_UNRESOLVED",
                    attempt_evidence=({"reason": "INTRO_PHRASE_DEFECT_PUBLICATION_BLOCKED"},),
                ),
            ),
            intro_phrase_family_reviews=(review,),
        )

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
            seed=7,
        ),
        dependencies=replace(dependencies, generation=generation),
    )

    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    assert report["status"] == "PARTIAL"
    assert report["publishable"] is False
    assert report["timingReviewRequired"] is True
    assert report["introPhraseFamilyReviews"][0]["status"] == "DEFECT"
    assert report["introPhraseFamilyReviews"][0]["reason"] == ("ISOLATED_EXPERT_FIRST_ROW")
    assert report["missingCharts"][0]["reason"] == ("INTRO_PHRASE_DEFECT_UNRESOLVED")


def test_intro_phrase_review_observation_does_not_downgrade_or_block_song(
    tmp_path: Path,
):
    from chart_worker.validation.intro_phrase_family import (
        IntroPhraseChartView,
        review_intro_phrase_pair,
    )

    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        outcome = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )
        review = review_intro_phrase_pair(
            IntroPhraseChartView(4, "HARD", 15_790, 15_915, 0.25),
            IntroPhraseChartView(4, "EXPERT", 0, 11_848, 23.696),
            start_delta_beats=31.58,
        )
        assert review.status == "REVIEW"
        assert review.should_block_publication is False
        return replace(outcome, intro_phrase_family_reviews=(review,))

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
        ),
        dependencies=replace(dependencies, generation=generation),
    )

    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    assert report["status"] == "PASS"
    assert report["publishable"] is False
    assert report["publicationDecision"]["reasonCodes"] == ["BOUNDARY_POLICY_UNCALIBRATED"]
    assert report["timingReviewRequired"] is False
    assert report["introPhraseFamilyReviews"][0]["status"] == "REVIEW"
    assert report["introPhraseFamilyReviews"][0]["reason"] == "EXPERT_EARLY_GHOST"


def test_pipeline_analyzes_only_the_canonical_game_audio_once(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    calls = []

    def analyze(path):
        calls.append(path)
        return dependencies.analyze(path)

    result = run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
        ),
        dependencies=replace(dependencies, analyze=analyze),
    )

    assert calls == [result.output_dir / "audio" / "game.flac"]


def test_pipeline_passes_one_shared_onset_analysis_to_timing_and_map_generation(
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    analyses = []
    timing_analyses = []
    generation_analyses = []

    def analyze(path):
        analysis = dependencies.analyze(path)
        analyses.append(analysis)
        return analysis

    def timing(prepared, analysis, run_dir, generator, seed):
        timing_analyses.append(analysis)
        return dependencies.timing(prepared, analysis, run_dir, generator, seed)

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        generation_analyses.append(analysis)
        return dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
        ),
        dependencies=replace(
            dependencies,
            analyze=analyze,
            timing=timing,
            generation=generation,
        ),
    )

    assert timing_analyses == analyses
    assert generation_analyses == analyses


def test_pipeline_passes_shared_activity_to_every_chart_diagnostic(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def analyze(path):
        analysis = dependencies.analyze(path)
        return replace(
            analysis,
            activity=AudioActivity(
                frame_ms=10.0,
                rms_db=np.full(200, -80.0),
                floor_db=-60.0,
                active_onset_ms=(),
            ),
            terminal_silence=TerminalSilenceObservation(
                version="terminal-silence-observation-v1",
                duration_ms=2_000,
                frame_ms=20,
                channel_count=2,
                candidates=(
                    TerminalThresholdCandidate(
                        rms_db=-66.0,
                        peak_db=-54.0,
                        suffix_start_ms=1_500,
                        suffix_duration_ms=500,
                    ),
                ),
                candidate_spread_ms=0,
                last_onset_ms=1_400,
            ),
        )

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
        ),
        dependencies=replace(dependencies, analyze=analyze),
    )

    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    outro_profile = report["musicBounds"]["outroEvidenceProfile"]
    terminal = report["musicBounds"]["terminalSilenceObservation"]
    assert terminal["version"] == "terminal-silence-observation-v1"
    assert terminal["policyState"] == "OBSERVATION_ONLY"
    assert terminal["mutatesGeneration"] is False
    assert terminal["candidates"][0]["suffixStartMs"] == 1_500
    assert outro_profile["version"] == "outro-evidence-profile-v1"
    assert outro_profile["policyState"] == "UNCALIBRATED"
    assert outro_profile["semanticClassification"] == "UNAVAILABLE"
    assert [window["windowMs"] for window in outro_profile["windows"]] == [
        1_000,
        2_000,
        5_000,
        10_000,
    ]
    boundary = report["musicBounds"]["boundaryPolicyEvaluation"]
    assert boundary["policyState"] == "PROVISIONAL"
    assert boundary["confidence"] == "UNKNOWN"
    assert boundary["enforcementMode"] == "HIGH_CONFIDENCE_ENFORCED"
    assert boundary["effectiveSource"] == "FULL_DURATION_BASELINE"
    assert boundary["effectiveContract"] == report["musicBounds"]["songBoundaryContract"]
    assert boundary["observationSha256"] == (boundary["provisionalDecision"]["observationSha256"])
    assert report["boundaryPublicationAssessment"] == {
        "version": "boundary-publication-assessment-v1",
        "evidenceStatus": "AVAILABLE",
        "policyState": "PROVISIONAL",
        "confidence": "UNKNOWN",
        "strictBlockers": ["BOUNDARY_POLICY_UNCALIBRATED"],
    }
    assert report["publishable"] is False
    assert report["publicationDecision"]["decision"] == "PLAYTEST_ONLY"
    assert report["status"] == "PASS"
    assert report["timingReviewRequired"] is True
    assert len(report["charts"]) == 12
    assert all(chart["timingDiagnostics"]["activeOnsetCount"] == 0 for chart in report["charts"])
    assert all(
        "LOW_ACTIVE_ONSET_SUPPORT" in chart["acceptanceDecisions"]["TIMING_ALIGNMENT"]["reasons"]
        for chart in report["charts"]
    )
    assert len(list((tmp_path / "run" / "charts").glob("*.json"))) == 12


def test_hard_safe_quality_rejection_enters_normal_playtest_as_diagnostic_only(
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        outcome = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )
        dropped, *kept = outcome.variants
        source_workdir = run_dir / "raw" / "work" / "diagnostic-source"
        source_workdir.mkdir(parents=True)
        diagnostic = DiagnosticRawCandidate.create(
            key_mode=dropped.key_mode,
            difficulty=dropped.difficulty,
            seed=dropped.selected_seed or 0,
            attempt=dropped.attempt,
            osu_text=dropped.raw_osu_path.read_text(encoding="utf-8"),
            source_workdir=source_workdir,
            gate_report=dropped.acceptance.to_report(),
            attempt_errors=("fixture quality rejection",),
            attempt_evidence=({"reason": "QUALITY_GATE_RETRY"},),
        )
        playtest_fallback = replace(
            dropped,
            provenance="RAW_UNVERIFIED",
            production_eligible=False,
            family_resolution_state="UNRESOLVED",
            family_resolution_reasons=("QUALITY_REJECTED_HARD_SAFE_PLAYTEST_RETURN",),
            recovery_reason="QUALITY_REJECTED_HARD_SAFE_PLAYTEST_RETURN",
        )
        return replace(
            outcome,
            variants=(playtest_fallback, *kept),
            missing=(),
            diagnostic_raw_candidates=(diagnostic,),
        )

    config = WorkerConfig(
        mapperatorinator_hold_state_mode="incremental",
        mapperatorinator_model_root=Path(r"C:\models\mapperatorinator-v32"),
        mapperatorinator_model_revision="a" * 40,
    )
    run_dir = tmp_path / "run"
    result = run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=run_dir,
            title="fixture",
            generator="mapperatorinator",
            seed=7,
        ),
        dependencies=replace(
            dependencies,
            config=config,
            generation=generation,
        ),
    )

    diagnostic_path = run_dir / "diagnostic-raw-fallback" / "4k-easy" / "map.osu"
    assert diagnostic_path.is_file()
    assert diagnostic_path not in result.chart_paths
    assert diagnostic_path not in result.raw_osu_paths
    report = json.loads((run_dir / "generation-report.json").read_text(encoding="utf-8"))
    assert len(report["charts"]) == 12
    assert report["missingCharts"] == []
    fallback = next(
        chart
        for chart in report["charts"]
        if chart["keyMode"] == 4 and chart["difficulty"] == "EASY"
    )
    assert fallback["provenance"] == "RAW_UNVERIFIED"
    assert "productionEligible" not in fallback
    assert "distributionTier" not in fallback
    assert fallback["playabilityTier"] == "DIAGNOSTIC_ONLY"
    assert fallback["familyResolutionState"] == "UNRESOLVED"
    assert fallback["familyResolutionReasons"] == [
        "QUALITY_REJECTED_HARD_SAFE_PLAYTEST_RETURN"
    ]
    assert report["diagnosticRawFallbacks"][0]["path"].endswith("/map.osu")
    assert "productionEligible" not in report["diagnosticRawFallbacks"][0]
    manifest = json.loads((diagnostic_path.parent / "manifest-v1.json").read_text(encoding="utf-8"))
    assert manifest["decision"] == "PLAYTEST_ONLY"
    assert "productionEligible" not in manifest
    assert manifest["identity"]["patchSetId"] == CONSTRAINT_PATCH_ID


def test_outro_family_shadow_finding_marks_review_without_spending_generation_budget(
    tmp_path: Path,
):
    from chart_worker.validation.outro_family_review import (
        OutroChartView,
        review_outro_family,
    )

    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def generation(*args, **kwargs):
        outcome = dependencies.generation(*args, **kwargs)
        review = review_outro_family(
            (
                OutroChartView(4, "HARD", 99_000, 99_500),
                OutroChartView(6, "HARD", 99_200, 99_700),
                OutroChartView(7, "HARD", 89_000, 89_500),
            )
        )
        return replace(outcome, outro_family_review=review)

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
        ),
        dependencies=replace(dependencies, generation=generation),
    )

    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    assert report["outroFamilyReview"]["status"] == "REVIEW"
    assert report["outroFamilyReview"]["findings"][0]["reason"] == ("OUTRO_FAMILY_EARLY_START")
    assert report["status"] == "REVIEW"
    assert report["timingReviewRequired"] is True
    assert report["additionalInferenceCalls"] == 0


def test_generation_report_records_mapperatorinator_constraint_patch(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    dependencies = replace(
        dependencies,
        config=dependencies.config.model_copy(
            update={"mapperatorinator_hold_state_mode": "incremental_verify"}
        ),
    )

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="mapperatorinator",
        ),
        dependencies=dependencies,
    )

    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    assert report["mapperatorinatorConstraintPatch"] == CONSTRAINT_PATCH_ID
    assert report["mapperatorinatorHoldStateMode"] == "incremental_verify"


def test_pipeline_rejects_canonical_audio_changed_after_prepare(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    analyze_calls = []

    def prepare(source_path, run_dir, config):
        prepared = dependencies.prepare(source_path, run_dir, config)
        prepared.normalized.path.write_bytes(b"tampered after hashing")
        return prepared

    def analyze(path):
        analyze_calls.append(path)
        return dependencies.analyze(path)

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=tmp_path / "run",
                title="fixture",
                generator="fake",
            ),
            dependencies=replace(
                dependencies,
                prepare=prepare,
                analyze=analyze,
            ),
        )

    assert captured.value.code is ErrorCode.ASSET_HASH_MISMATCH
    assert analyze_calls == []


def test_pipeline_rejects_canonical_audio_changed_during_generation(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    export_calls = []

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        variants = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )
        prepared.normalized.path.write_bytes(b"tampered during generation")
        return variants

    def export(prepared, generated, run_dir, worker_version):
        export_calls.append(run_dir)
        return dependencies.export(prepared, generated, run_dir, worker_version)

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=tmp_path / "run",
                title="fixture",
                generator="fake",
            ),
            dependencies=replace(
                dependencies,
                generation=generation,
                export=export,
            ),
        )

    assert captured.value.code is ErrorCode.ASSET_HASH_MISMATCH
    assert export_calls == []


def test_pipeline_rejects_canonical_audio_changed_during_timing(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    generation_calls = []

    def timing(prepared, analysis, run_dir, generator, seed):
        authority = dependencies.timing(prepared, analysis, run_dir, generator, seed)
        prepared.normalized.path.write_bytes(b"tampered during timing")
        return authority

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        generation_calls.append(run_dir)
        return dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=tmp_path / "run",
                title="fixture",
                generator="fake",
            ),
            dependencies=replace(
                dependencies,
                timing=timing,
                generation=generation,
            ),
        )

    assert captured.value.code is ErrorCode.ASSET_HASH_MISMATCH
    assert generation_calls == []


def test_pipeline_defaults_to_mapperatorinator():
    options = PipelineOptions(
        source=Path("song.wav"),
        output_dir=Path("run"),
        title="song",
    )
    assert options.generator == "mapperatorinator"


def test_pipeline_rejects_an_empty_title():
    with pytest.raises(ValueError, match="title"):
        PipelineOptions(
            source=Path("song.wav"),
            output_dir=Path("run"),
            title="  ",
        )


def test_pipeline_rejects_a_nonempty_output_without_deleting_existing_files(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        run_pipeline(
            PipelineOptions(source=source, output_dir=output_dir, title="fixture"),
            dependencies=fake_dependencies(),
        )

    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "keep"


def _retry_timing_signal(authority_sha256: str) -> RetryTimingSignal:
    return RetryTimingSignal(
        tuple(
            MapTimingFailureSignature(
                authority_sha256=authority_sha256,
                key_mode=4,
                difficulty="NORMAL",
                seed=seed,
                timing_segment_id=3,
                failure_family="DUPLICATE_NOTE",
                time_ms=4_000,
                grid_aligned=True,
            )
            for seed in (7, 19)
        )
    )


def test_map_timing_feedback_replaces_authority_and_regenerates_all_variants(
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    generation_modes: list[str] = []
    generation_epochs: list[int] = []
    stable_raw_counts: list[int] = []
    super_calls: list[int] = []

    def super_timing(prepared, analysis, run_dir, generator, seed, authority_epoch=2):
        super_calls.append(seed)
        return run_timing_generation(
            prepared,
            analysis,
            run_dir,
            generator=generator,
            seed=seed,
            force_super=True,
        )

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        generation_modes.append(authority.mode)
        generation_epochs.append(authority_epoch)
        stable_raw_counts.append(len(list((run_dir / "raw").glob("*.osu"))))
        if len(generation_modes) == 1:
            raise _retry_timing_signal(authority.sha256)
        outcome = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
            authority_epoch,
        )
        assert len(outcome.variants) == 12
        assert not outcome.missing
        assert all(
            variant.timing_authority_sha256 == authority.sha256 for variant in outcome.variants
        )
        return outcome

    result = run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
            seed=7,
        ),
        dependencies=replace(
            dependencies,
            super_timing=super_timing,
            generation=generation,
        ),
    )

    assert generation_modes == ["STANDARD", "SUPER_TIMING"]
    assert generation_epochs == [1, 2]
    assert stable_raw_counts == [0, 0]
    assert super_calls == [7]
    assert len(result.raw_osu_paths) == 12
    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    assert report["selectedAuthorityEpoch"] == 2
    assert [candidate["status"] for candidate in report["timingCandidates"]] == [
        "REJECTED_MAP_TIMING_FEEDBACK",
        "SELECTED",
    ]
    assert report["timingGenerationMode"] == "SUPER_TIMING"
    assert len(report["mapTimingEscalations"]) == 1
    assert report["mapTimingEscalations"][0]["failureFamily"] == "DUPLICATE_NOTE"


@pytest.mark.parametrize(
    ("generator", "hold_state_mode"),
    [("fake", None), ("mapperatorinator", "incremental")],
)
def test_second_map_timing_feedback_is_bounded_and_reported(
    generator: str, hold_state_mode: str | None, tmp_path: Path
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    if generator == "mapperatorinator":
        dependencies = replace(
            dependencies,
            config=dependencies.config.model_copy(
                update={"mapperatorinator_hold_state_mode": "incremental"}
            ),
        )
    generation_modes: list[str] = []

    def super_timing(prepared, analysis, run_dir, generator, seed, authority_epoch=2):
        return run_timing_generation(
            prepared,
            analysis,
            run_dir,
            generator=generator,
            seed=seed,
            force_super=True,
        )

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        del prepared, analysis, run_dir, generator, seed, authority_epoch
        generation_modes.append(authority.mode)
        raise _retry_timing_signal(authority.sha256)

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=tmp_path / "run",
                title="fixture",
                generator=generator,
                seed=7,
            ),
            dependencies=replace(
                dependencies,
                super_timing=super_timing,
                generation=generation,
            ),
        )

    assert captured.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    assert generation_modes == ["STANDARD", "SUPER_TIMING"]
    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    assert report["selectedAuthorityEpoch"] is None
    assert [candidate["status"] for candidate in report["timingCandidates"]] == [
        "REJECTED_MAP_TIMING_FEEDBACK",
        "FAILED",
    ]
    assert len(report["mapTimingEscalations"]) == 2
    assert report["failureStage"] == "GENERATION"
    assert report["mapperatorinatorHoldStateMode"] == hold_state_mode


def test_super_timing_failure_writes_mapperatorinator_hold_state_mode(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    dependencies = replace(
        dependencies,
        config=dependencies.config.model_copy(
            update={"mapperatorinator_hold_state_mode": "incremental_verify"}
        ),
    )
    failure = WorkerError(
        ErrorCode.CHART_TIMING_CANDIDATE_FAILED,
        "super timing fixture",
    )

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        del prepared, analysis, run_dir, generator, seed, authority_epoch
        raise _retry_timing_signal(authority.sha256)

    def super_timing(prepared, analysis, run_dir, generator, seed, authority_epoch=2):
        del prepared, analysis, run_dir, generator, seed, authority_epoch
        raise failure

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=tmp_path / "run",
                title="fixture",
                generator="mapperatorinator",
                seed=7,
            ),
            dependencies=replace(
                dependencies,
                generation=generation,
                super_timing=super_timing,
            ),
        )

    assert captured.value is failure
    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    assert report["failureStage"] == "TIMING"
    assert report["mapperatorinatorHoldStateMode"] == "incremental_verify"


def test_independent_fallback_does_not_reenter_failed_super_timing_path(
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    generation_modes: list[str] = []
    super_calls: list[int] = []

    def timing(prepared, analysis, run_dir, generator, seed):
        authority = dependencies.timing(prepared, analysis, run_dir, generator, seed)
        return replace(authority, mode="BEAT_THIS_FALLBACK")

    def super_timing(prepared, analysis, run_dir, generator, seed, authority_epoch=2):
        del prepared, analysis, run_dir, generator, authority_epoch
        super_calls.append(seed)
        raise AssertionError("independent fallback must not re-enter Super Timing")

    def generation(prepared, authority, analysis, run_dir, generator, seed, authority_epoch):
        del prepared, analysis, run_dir, generator, seed, authority_epoch
        generation_modes.append(authority.mode)
        raise _retry_timing_signal(authority.sha256)

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=tmp_path / "run",
                title="fixture",
                generator="fake",
                seed=7,
            ),
            dependencies=replace(
                dependencies,
                timing=timing,
                super_timing=super_timing,
                generation=generation,
            ),
        )

    assert captured.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    assert generation_modes == ["BEAT_THIS_FALLBACK"]
    assert super_calls == []
    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    assert report["failureStage"] == "GENERATION"
    assert report["timingGenerationMode"] == "BEAT_THIS_FALLBACK"
    assert report["timingCandidates"][-1]["mode"] == "BEAT_THIS_FALLBACK"
    assert report["timingCandidates"][-1]["status"] == "FAILED"
