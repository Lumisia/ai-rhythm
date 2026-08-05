from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent, parse_osu_file
from chart_worker.generation.osu_writer import timing_to_osu_mania
from chart_worker.hashing import sha256_file
from chart_worker.schema.note import NoteEvent
from chart_worker.stages import s2_generate
from chart_worker.stages.s2_generate import run_generation
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.quality_gate import GateAction, evaluate_chart_candidate


def _prepared(tmp_path: Path) -> PreparedAudio:
    audio_path = tmp_path / "audio" / "game.flac"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"audio")
    return PreparedAudio(
        normalized=NormalizedAudio(
            audio_path,
            "audio-profile-v1",
            sha256_file(audio_path),
            2_000,
            48_000,
            2,
            2_000,
            0,
            0.0,
            -14.0,
            -1.0,
            0.0,
            "LOUDNESS",
        ),
    )


def _authority(
    prepared: PreparedAudio,
    tmp_path: Path,
    bpm_events: tuple[OsuBpmEvent, ...] = (OsuBpmEvent(0, 120.0),),
) -> SongTimingAuthority:
    reference_path = tmp_path / "audio" / "timing-reference.osu"
    reference_path.write_text(
        timing_to_osu_mania(
            bpm_events,
            audio_filename=prepared.normalized.path.name,
            title="fixture",
        ),
        encoding="utf-8",
    )
    return SongTimingAuthority(
        reference_path=reference_path,
        sha256=sha256_file(reference_path),
        audio_sha256=prepared.normalized.sha256,
        bpm_events=bpm_events,
        generator_name="recording-generator",
        seed=17,
        mode="STANDARD",
        attempt_count=1,
    )


def _analysis() -> OnsetAnalysis:
    rows = tuple(range(125, 2_000, 125))
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=np.zeros(21),
        band_strength=np.zeros((3, 21)),
        onset_ms=rows,
    )


def _pass_acceptance(authority: SongTimingAuthority):
    rows = tuple(range(125, 2_000, 125))
    chart = GeneratedChart(
        notes=[NoteEvent(row, 0) for row in rows],
        key_mode=4,
        osu_text="",
        generator_name="acceptance-fixture",
        seed=0,
        bpm_events=authority.bpm_events,
    )
    return evaluate_chart_candidate(
        chart,
        authority,
        _analysis(),
        requested_key_mode=4,
        requested_difficulty="EASY",
        duration_ms=2_000,
    )


def _acceptance_with_action(authority: SongTimingAuthority, action: GateAction):
    accepted = _pass_acceptance(authority)
    if action is GateAction.PASS:
        return accepted
    first, *remaining = accepted.decisions
    return replace(
        accepted,
        action=action,
        decisions=(
            replace(first, action=action, reasons=(f"FIXTURE_{action.value}",)),
            *remaining,
        ),
    )


@pytest.fixture(autouse=True)
def _accept_candidates_by_default(monkeypatch, tmp_path: Path):
    fixture_dir = tmp_path / "acceptance-fixture"
    prepared = _prepared(fixture_dir)
    authority = _authority(prepared, fixture_dir)
    accepted = _pass_acceptance(authority)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        lambda *args, **kwargs: accepted,
        raising=False,
    )


class RecordingGenerator:
    def __init__(self):
        self.map_calls = []
        self.map_workdirs = []

    def generate_map(self, request, workdir):
        self.map_calls.append(request)
        self.map_workdirs.append(workdir)
        return GeneratedChart(
            notes=[NoteEvent(500, request.key_mode - 1)],
            key_mode=request.key_mode,
            osu_text="",
            generator_name="recording-fake",
            seed=request.seed,
            bpm_events=(OsuBpmEvent(0, 120.0),),
        )


def test_run_generation_creates_exactly_twelve_parseable_variants(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    generator = RecordingGenerator()
    variants = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=17,
    )

    assert {(variant.key_mode, variant.difficulty) for variant in variants} == {
        (key_mode, difficulty)
        for key_mode in (4, 6, 7)
        for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
    }
    requests = generator.map_calls
    workdirs = generator.map_workdirs
    assert len(requests) == 12
    assert {request.timing_reference_path for request in requests} == {
        authority.reference_path
    }
    assert all(
        variant.timing_authority_sha256 == authority.sha256 for variant in variants
    )
    assert len({request.seed for request in requests}) == 12
    assert [
        (request.key_mode, request.difficulty, request.seed) for request in requests
    ] == [
        (key_mode, difficulty, 17 + index)
        for index, (key_mode, difficulty) in enumerate(
            (key_mode, difficulty)
            for key_mode in (4, 6, 7)
            for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
        )
    ]
    assert len(set(workdirs)) == 12
    assert all(workdir.name == "attempt-1" for workdir in workdirs)
    assert all(workdir.parent.parent.name == "work" for workdir in workdirs)
    assert all("candidates" not in workdir.parts for workdir in workdirs)
    assert all(request.duration_ms == 2_000 for request in requests)
    assert all(request.cfg_scale == 1.0 for request in requests)
    assert all(len(request.descriptors) == 1 for request in requests)
    assert [variant.raw_osu_path.name for variant in variants] == [
        f"{key_mode}k-{difficulty.lower()}.osu"
        for key_mode in (4, 6, 7)
        for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
    ]
    assert all(
        parse_osu_file(variant.raw_osu_path).key_mode == variant.key_mode for variant in variants
    )


def test_run_generation_preserves_generator_osu_text(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    class OriginalGenerator:
        def __init__(self):
            self.texts = []

        def generate_map(self, request, workdir):
            text = (
                "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\n"
                f"CircleSize:{request.key_mode}\n\n[TimingPoints]\n"
                "0,500,4,2,0,60,1,0\n\n[HitObjects]\n64,192,500,1,0,0:0:0:0:\n"
            )
            self.texts.append(text)
            return GeneratedChart(
                [NoteEvent(500, 0)],
                request.key_mode,
                text,
                "original",
                request.seed,
                (OsuBpmEvent(0, 120.0),),
            )

    generator = OriginalGenerator()
    variants = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=1,
    )
    assert variants[0].raw_osu_path.read_text(encoding="utf-8") == generator.texts[0]


def test_empty_osu_text_fallback_preserves_every_authority_timing_event(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    bpm_events = (OsuBpmEvent(0, 120.0), OsuBpmEvent(1_000, 150.0))
    authority = _authority(prepared, tmp_path, bpm_events)

    class MultipleTimingGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return GeneratedChart(
                notes=generated.notes,
                key_mode=generated.key_mode,
                osu_text="",
                generator_name=generated.generator_name,
                seed=generated.seed,
                bpm_events=bpm_events,
            )

    variants = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=MultipleTimingGenerator(),
        seed=0,
    )

    assert parse_osu_file(variants[0].raw_osu_path).bpm_events == bpm_events


def test_stable_raw_reparse_rejects_text_with_different_timing_identity(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    mismatched_text = (
        "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\n"
        "CircleSize:4\n\n[TimingPoints]\n0,495.867768595041,4,2,0,60,1,0\n\n"
        "[HitObjects]\n64,192,500,1,0,0:0:0:0:\n"
    )

    class MismatchedTextGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return GeneratedChart(
                notes=generated.notes,
                key_mode=generated.key_mode,
                osu_text=mismatched_text,
                generator_name=generated.generator_name,
                seed=generated.seed,
                bpm_events=generated.bpm_events,
            )

    generator = MismatchedTextGenerator()
    with pytest.raises(WorkerError) as captured:
        run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=generator,
            seed=0,
        )

    assert captured.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    assert len(generator.map_calls) == 3
    assert not (tmp_path / "raw" / "4k-easy.osu").exists()


@pytest.mark.parametrize(
    ("generated_note", "serialized_key_mode", "serialized_hit_object"),
    [
        pytest.param(
            NoteEvent(500, 0),
            6,
            "64,192,500,1,0,0:0:0:0:",
            id="key-mode",
        ),
        pytest.param(
            NoteEvent(500, 0),
            4,
            "192,192,500,1,0,0:0:0:0:",
            id="lane",
        ),
        pytest.param(
            NoteEvent(500, 0),
            4,
            "64,192,500,128,0,750:0:0:0:0:",
            id="kind",
        ),
        pytest.param(
            NoteEvent(500, 0, kind="HOLD", duration_ms=250),
            4,
            "64,192,500,128,0,800:0:0:0:0:",
            id="hold-duration",
        ),
    ],
)
def test_serialized_note_or_key_mismatch_retries_without_stable_promotion(
    tmp_path: Path,
    generated_note: NoteEvent,
    serialized_key_mode: int,
    serialized_hit_object: str,
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    mismatched_text = (
        "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\n"
        f"CircleSize:{serialized_key_mode}\n\n[TimingPoints]\n"
        "0,500,4,2,0,60,1,0\n\n[HitObjects]\n"
        f"{serialized_hit_object}\n"
    )

    class MismatchedCandidateGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return GeneratedChart(
                notes=[generated_note],
                key_mode=generated.key_mode,
                osu_text=mismatched_text,
                generator_name=generated.generator_name,
                seed=generated.seed,
                bpm_events=generated.bpm_events,
            )

    generator = MismatchedCandidateGenerator()
    with pytest.raises(WorkerError) as captured:
        run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=generator,
            seed=0,
        )

    assert captured.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    assert len(generator.map_calls) == 3
    assert not (tmp_path / "raw" / "4k-easy.osu").exists()


def test_reference_metadata_mutated_during_map_is_rejected_before_promotion(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    class MutatingReferenceGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            authority.reference_path.write_text(
                authority.reference_path.read_text(encoding="utf-8").replace(
                    "Title:fixture", "Title:mutated"
                ),
                encoding="utf-8",
            )
            return generated

    generator = MutatingReferenceGenerator()
    with pytest.raises(WorkerError) as captured:
        run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=generator,
            seed=0,
        )

    assert captured.value.code is ErrorCode.ASSET_HASH_MISMATCH
    assert len(generator.map_calls) == 1
    assert not (tmp_path / "raw" / "4k-easy.osu").exists()


def test_canonical_audio_mutated_during_map_is_rejected_before_promotion(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    class MutatingAudioGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            prepared.normalized.path.write_bytes(b"mutated audio")
            return generated

    generator = MutatingAudioGenerator()
    with pytest.raises(WorkerError) as captured:
        run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=generator,
            seed=0,
        )

    assert captured.value.code is ErrorCode.ASSET_HASH_MISMATCH
    assert len(generator.map_calls) == 1
    assert not (tmp_path / "raw" / "4k-easy.osu").exists()


def test_run_generation_never_writes_invalid_output_to_stable_raw(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    class InvalidLaneGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return GeneratedChart(
                notes=[NoteEvent(500, request.key_mode)],
                key_mode=generated.key_mode,
                osu_text=generated.osu_text,
                generator_name=generated.generator_name,
                seed=generated.seed,
                bpm_events=generated.bpm_events,
            )

    with pytest.raises(WorkerError) as captured:
        run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=InvalidLaneGenerator(),
            seed=0,
        )
    assert captured.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    assert not (tmp_path / "raw" / "4k-easy.osu").exists()


def test_retries_only_the_failed_variant_with_the_next_seed(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    class FirstEasyAttemptInvalid(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if request.key_mode == 4 and request.difficulty == "EASY" and request.seed == 0:
                return GeneratedChart(
                    notes=[NoteEvent(500, request.key_mode)],
                    key_mode=generated.key_mode,
                    osu_text=generated.osu_text,
                    generator_name=generated.generator_name,
                    seed=generated.seed,
                    bpm_events=generated.bpm_events,
                )
            return generated

    generator = FirstEasyAttemptInvalid()
    variants = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    calls = [
        (request.key_mode, request.difficulty, request.seed, workdir.name)
        for request, workdir in zip(
            generator.map_calls, generator.map_workdirs, strict=True
        )
    ]
    assert calls[:2] == [
        (4, "EASY", 0, "attempt-1"),
        (4, "EASY", 12, "attempt-2"),
    ]
    assert len(calls) == 13
    assert variants[0].attempt == 2
    assert len(variants[0].attempt_errors) == 1
    assert all(variant.attempt == 1 for variant in variants[1:])
    assert all(not variant.attempt_errors for variant in variants[1:])


def test_retry_map_uses_next_seed_and_promotes_only_the_pass_candidate(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    retry = _acceptance_with_action(authority, GateAction.RETRY_MAP)
    accepted = _acceptance_with_action(authority, GateAction.PASS)
    evaluations = []

    def evaluate(*args, **kwargs):
        del args, kwargs
        evaluations.append(len(evaluations) + 1)
        if len(evaluations) == 1:
            return retry
        if len(evaluations) == 2:
            assert not (tmp_path / "raw" / "4k-easy.osu").exists()
        return accepted

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)
    generator = RecordingGenerator()

    variants = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    first = variants[0]
    assert [request.seed for request in generator.map_calls[:2]] == [0, 12]
    assert first.attempt == 2
    assert first.acceptance == accepted
    assert first.raw_osu_path == tmp_path / "raw" / "4k-easy.osu"
    assert first.raw_osu_path.is_file()
    assert first.attempt_evidence == (
        {
            "seed": 0,
            "workdir": "raw/work/4k-easy/attempt-1",
            "gateReport": retry.to_report(),
        },
    )


def test_review_quarantines_candidate_without_consuming_another_seed(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    review = _acceptance_with_action(authority, GateAction.REVIEW)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        lambda *args, **kwargs: review,
    )

    class EvidenceGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            (workdir / "candidate.osu").write_text("candidate", encoding="utf-8")
            return generated

    generator = EvidenceGenerator()
    with pytest.raises(WorkerError) as captured:
        run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=generator,
            seed=0,
        )

    assert captured.value.code is ErrorCode.CHART_TIMING_REVIEW_REQUIRED
    assert [request.seed for request in generator.map_calls] == [0]
    assert captured.value.context == {
        "seed": 0,
        "workdir": "raw/work/4k-easy/attempt-1",
        "key_mode": 4,
        "difficulty": "EASY",
        "gate_report": review.to_report(),
    }
    assert (
        tmp_path / "raw" / "work" / "4k-easy" / "attempt-1" / "candidate.osu"
    ).is_file()
    assert not (tmp_path / "raw" / "4k-easy.osu").exists()


def test_three_retry_map_decisions_exhaust_with_structured_attempt_evidence(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    retry = _acceptance_with_action(authority, GateAction.RETRY_MAP)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        lambda *args, **kwargs: retry,
    )
    generator = RecordingGenerator()

    with pytest.raises(WorkerError) as captured:
        run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=generator,
            seed=0,
        )

    assert captured.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    assert [request.seed for request in generator.map_calls] == [0, 12, 24]
    assert captured.value.context["attempts"] == [
        {
            "seed": attempt_seed,
            "workdir": f"raw/work/4k-easy/attempt-{attempt}",
            "gateReport": retry.to_report(),
        }
        for attempt, attempt_seed in enumerate((0, 12, 24), start=1)
    ]
    assert len(captured.value.context["errors"]) == 3
    assert not (tmp_path / "raw" / "4k-easy.osu").exists()


def test_mixed_exhaustion_retains_gate_evidence_with_all_legacy_errors(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    retry = _acceptance_with_action(authority, GateAction.RETRY_MAP)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        lambda *args, **kwargs: retry,
    )

    class GateThenInvalidGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if request.seed == 0:
                return generated
            return replace(
                generated,
                notes=[NoteEvent(500, request.key_mode)],
            )

    generator = GateThenInvalidGenerator()
    with pytest.raises(WorkerError) as captured:
        run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=generator,
            seed=0,
        )

    assert captured.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    assert captured.value.context["seeds"] == [0, 12, 24]
    assert len(captured.value.context["errors"]) == 3
    assert captured.value.context["attempts"] == [
        {
            "seed": 0,
            "workdir": "raw/work/4k-easy/attempt-1",
            "gateReport": retry.to_report(),
        }
    ]


def test_real_acceptance_evidence_is_recorded_for_aligned_candidates(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        evaluate_chart_candidate,
    )

    class AlignedGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return replace(
                generated,
                notes=[
                    NoteEvent(time_ms, index % request.key_mode)
                    for index, time_ms in enumerate(range(125, 2_000, 125))
                ],
            )

    variants = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=AlignedGenerator(),
        seed=0,
    )

    assert {variant.acceptance.action for variant in variants} == {GateAction.PASS}
    assert all(not variant.attempt_evidence for variant in variants)


def test_reports_all_errors_when_one_variant_exhausts_its_attempts(tmp_path: Path):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    class AlwaysInvalid(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return GeneratedChart(
                notes=[NoteEvent(500, request.key_mode)],
                key_mode=generated.key_mode,
                osu_text=generated.osu_text,
                generator_name=generated.generator_name,
                seed=generated.seed,
                bpm_events=generated.bpm_events,
            )

    generator = AlwaysInvalid()
    with pytest.raises(WorkerError) as captured:
        run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=generator,
            seed=0,
        )

    assert captured.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    assert captured.value.context["key_mode"] == 4
    assert captured.value.context["difficulty"] == "EASY"
    assert captured.value.context["seeds"] == [0, 12, 24]
    assert len(captured.value.context["errors"]) == 3
    assert [request.seed for request in generator.map_calls] == [0, 12, 24]


def test_different_timing_identity_retries_only_failed_map_without_promotion(
    tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)

    class DifferentTimingGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return GeneratedChart(
                notes=generated.notes,
                key_mode=generated.key_mode,
                osu_text=generated.osu_text,
                generator_name=generated.generator_name,
                seed=generated.seed,
                bpm_events=(OsuBpmEvent(0, 121.0),),
            )

    generator = DifferentTimingGenerator()
    with pytest.raises(WorkerError) as captured:
        run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=generator,
            seed=0,
        )

    assert captured.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    assert [
        (request.key_mode, request.difficulty, request.seed)
        for request in generator.map_calls
    ] == [
        (4, "EASY", 0),
        (4, "EASY", 12),
        (4, "EASY", 24),
    ]
    assert not (tmp_path / "raw" / "4k-easy.osu").exists()
