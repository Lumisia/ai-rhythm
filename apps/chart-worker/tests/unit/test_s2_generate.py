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


def _prepared(tmp_path: Path, *, duration_ms: int = 2_000) -> PreparedAudio:
    audio_path = tmp_path / "audio" / "game.flac"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"audio")
    return PreparedAudio(
        normalized=NormalizedAudio(
            audio_path,
            "audio-profile-v1",
            sha256_file(audio_path),
            duration_ms,
            48_000,
            2,
            duration_ms,
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


def _acceptance_for_difficulty(acceptance, difficulty: str):
    ratings = {"EASY": 1.0, "NORMAL": 2.0, "HARD": 3.0, "EXPERT": 4.0}
    assert acceptance.profile is not None
    return replace(
        acceptance,
        profile=replace(
            acceptance.profile,
            difficulty=replace(
                acceptance.profile.difficulty,
                project_rating=ratings[difficulty],
            ),
        ),
    )


def _acceptance_with_rating(acceptance, rating: float):
    assert acceptance.profile is not None
    return replace(
        acceptance,
        profile=replace(
            acceptance.profile,
            difficulty=replace(
                acceptance.profile.difficulty,
                project_rating=rating,
            ),
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
        lambda *args, requested_difficulty, **kwargs: _acceptance_for_difficulty(
            accepted, requested_difficulty
        ),
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


def test_difficulty_inversion_retries_only_pair_and_reuses_earliest_pass_candidate(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    ratings = {
        (4, "EASY", 0): 1.0,
        (4, "NORMAL", 1): 2.0,
        (4, "HARD", 2): 4.1,
        (4, "EXPERT", 3): 3.0,
        (4, "HARD", 14): 4.2,
        (4, "EXPERT", 15): 5.0,
    }

    def evaluate(generated, *args, requested_key_mode, requested_difficulty, **kwargs):
        del args, kwargs
        rating = ratings.get(
            (requested_key_mode, requested_difficulty, generated.seed),
            {"EASY": 1.0, "NORMAL": 2.0, "HARD": 3.0, "EXPERT": 4.0}[
                requested_difficulty
            ],
        )
        return _acceptance_with_rating(accepted, rating)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)

    class NoEarlyPromotionGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            if request.key_mode == 4 and request.seed in (14, 15):
                assert not any(
                    (tmp_path / "raw" / f"4k-{difficulty.lower()}.osu").exists()
                    for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
                )
            return super().generate_map(request, workdir)

    generator = NoEarlyPromotionGenerator()
    variants = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert [
        (request.key_mode, request.difficulty, request.seed)
        for request in generator.map_calls[:6]
    ] == [
        (4, "EASY", 0),
        (4, "NORMAL", 1),
        (4, "HARD", 2),
        (4, "EXPERT", 3),
        (4, "HARD", 14),
        (4, "EXPERT", 15),
    ]
    selected_4k = {
        variant.difficulty: variant
        for variant in variants
        if variant.key_mode == 4
    }
    assert selected_4k["HARD"].selected_seed == 2
    assert selected_4k["HARD"].attempt == 1
    assert selected_4k["HARD"].candidate_count == 2
    assert selected_4k["HARD"].generation_attempt_count == 2
    assert any(
        evidence.get("reason")
        == "NOT_SELECTED_EARLIEST_MONOTONIC_COMBINATION"
        and evidence["seed"] == 14
        and evidence["attempt"] == 2
        and evidence["serializationValidated"] is True
        and evidence["gateReport"]["qualityProfile"]["difficultyProfile"][
            "projectRating"
        ]
        == 4.2
        for evidence in selected_4k["HARD"].attempt_evidence
    )
    assert selected_4k["EXPERT"].selected_seed == 15
    assert selected_4k["EXPERT"].attempt == 2
    assert selected_4k["EASY"].candidate_count == 1
    assert selected_4k["NORMAL"].candidate_count == 1
    assert all(
        variant.difficulty_order is not None
        and variant.difficulty_order.status == "PASS"
        for variant in selected_4k.values()
    )


def test_equal_difficulty_profiles_publish_without_more_seeds(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _acceptance_with_rating(_pass_acceptance(authority), 2.0)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        lambda *args, **kwargs: accepted,
    )
    generator = RecordingGenerator()

    variants = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(variants) == 12
    assert [request.seed for request in generator.map_calls] == list(range(12))
    assert variants[0].difficulty_order is not None
    assert variants[0].difficulty_order.ambiguous_pairs == (
        ("EASY", "NORMAL"),
        ("NORMAL", "HARD"),
        ("HARD", "EXPERT"),
    )
    assert len(list((tmp_path / "raw").glob("*k-*.osu"))) == 12


def test_ambiguity_does_not_prevent_retrying_a_separate_inverted_pair(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        if generated.seed in {2, 3}:
            rating = {"HARD": 4.0, "EXPERT": 3.0}[requested_difficulty]
        else:
            rating = {"EASY": 2.0, "NORMAL": 2.0, "HARD": 3.0, "EXPERT": 4.0}[
                requested_difficulty
            ]
        return _acceptance_with_rating(accepted, rating)

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

    assert len(variants) == 12
    assert [request.seed for request in generator.map_calls] == [
        0,
        1,
        2,
        3,
        14,
        15,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ]
    assert variants[0].difficulty_order is not None
    assert variants[0].difficulty_order.ambiguous_pairs == (
        ("EASY", "NORMAL"),
        ("HARD", "EXPERT"),
    )


def test_persistent_inversion_exhausts_only_the_affected_label_budgets(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)

    def evaluate(*args, requested_difficulty, **kwargs):
        del args, kwargs
        rating = {"EASY": 1.0, "NORMAL": 2.0, "HARD": 4.0, "EXPERT": 3.0}[
            requested_difficulty
        ]
        return _acceptance_with_rating(accepted, rating)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)
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
    assert captured.value.context["difficulty"] == "HARD"
    assert captured.value.context["seeds"] == [2, 14, 26]
    assert [
        (request.difficulty, request.seed) for request in generator.map_calls
    ] == [
        ("EASY", 0),
        ("NORMAL", 1),
        ("HARD", 2),
        ("EXPERT", 3),
        ("HARD", 14),
        ("EXPERT", 15),
        ("HARD", 26),
        ("EXPERT", 27),
    ]
    assert not any((tmp_path / "raw").glob("4k-*.osu"))


def test_earliest_ambiguous_pool_combination_is_published(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    accepted = _pass_acceptance(authority)
    ratings = {
        ("EASY", 0): 1.0,
        ("NORMAL", 1): 2.0,
        ("HARD", 2): 4.0,
        ("EXPERT", 3): 3.0,
        ("HARD", 14): 2.5,
        ("EXPERT", 15): 4.0,
    }

    def evaluate(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        return _acceptance_with_rating(
            accepted,
            ratings[(requested_difficulty, generated.seed)],
        )

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate)
    generator = RecordingGenerator()

    def evaluate_all_modes(generated, *args, requested_difficulty, **kwargs):
        del args, kwargs
        rating = ratings.get(
            (requested_difficulty, generated.seed),
            {"EASY": 1.0, "NORMAL": 2.0, "HARD": 3.0, "EXPERT": 4.0}[
                requested_difficulty
            ],
        )
        return _acceptance_with_rating(accepted, rating)

    monkeypatch.setattr(s2_generate, "evaluate_chart_candidate", evaluate_all_modes)

    variants = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(variants) == 12
    assert [variant.selected_seed for variant in variants[:4]] == [0, 1, 2, 15]
    assert variants[0].difficulty_order is not None
    assert variants[0].difficulty_order.ambiguous_pairs == (("HARD", "EXPERT"),)
    assert [request.seed for request in generator.map_calls] == [
        0,
        1,
        2,
        3,
        14,
        15,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ]
    assert len(list((tmp_path / "raw").glob("4k-*.osu"))) == 4


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


def test_partial_stable_promotion_is_cleaned_and_normalized(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    original_read_text = Path.read_text

    def corrupt_second_stable_raw(path: Path, *args, **kwargs):
        if path.name == "4k-normal.osu":
            return "not an osu beatmap"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", corrupt_second_stable_raw)

    with pytest.raises(WorkerError) as captured:
        run_generation(
            prepared,
            authority,
            _analysis(),
            tmp_path,
            generator=RecordingGenerator(),
            seed=0,
        )

    assert captured.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    assert captured.value.context == {
        "key_mode": 4,
        "failure_stage": "PROMOTION",
        "paths": [
            "raw/4k-easy.osu",
            "raw/4k-normal.osu",
            "raw/4k-hard.osu",
            "raw/4k-expert.osu",
        ],
        "selected_seeds": [0, 1, 2, 3],
        "cause_code": "CHART_OSU_PARSE_FAILED",
        "cause": "CHART_OSU_PARSE_FAILED: serialized MAP is not valid osu!mania",
    }
    assert not any((tmp_path / "raw").glob("4k-*.osu"))


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


def test_generated_structure_defects_exhaust_with_gate_evidence_before_stable_raw(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        evaluate_chart_candidate,
    )

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

    generator = InvalidLaneGenerator()
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
    assert [attempt["seed"] for attempt in captured.value.context["attempts"]] == [
        0,
        12,
        24,
    ]
    assert [attempt["workdir"] for attempt in captured.value.context["attempts"]] == [
        "raw/work/4k-easy/attempt-1",
        "raw/work/4k-easy/attempt-2",
        "raw/work/4k-easy/attempt-3",
    ]
    assert all(
        attempt["gateReport"]["decisions"]["STRUCTURE"]["reasons"]
        == ["STRUCTURE_INVALID"]
        for attempt in captured.value.context["attempts"]
    )
    assert not (tmp_path / "raw" / "4k-easy.osu").exists()


def test_real_hold_overlap_retries_only_the_failed_variant(tmp_path: Path):
    prepared = _prepared(tmp_path, duration_ms=150_000)
    authority = _authority(prepared, tmp_path)

    class FirstEasyAttemptHasObservedHoldOverlap(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if (
                request.key_mode == 4
                and request.difficulty == "EASY"
                and request.seed == 0
            ):
                return GeneratedChart(
                    notes=[
                        NoteEvent(0, 0, kind="HOLD", duration_ms=134_204),
                        NoteEvent(925, 0),
                    ],
                    key_mode=generated.key_mode,
                    osu_text="",
                    generator_name=generated.generator_name,
                    seed=generated.seed,
                    bpm_events=generated.bpm_events,
                )
            return generated

    generator = FirstEasyAttemptHasObservedHoldOverlap()
    variants = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    easy_calls = [
        call
        for call in generator.map_calls
        if call.key_mode == 4 and call.difficulty == "EASY"
    ]
    other_calls = [
        call
        for call in generator.map_calls
        if not (call.key_mode == 4 and call.difficulty == "EASY")
    ]
    assert [call.seed for call in easy_calls] == [0, 12]
    assert len(other_calls) == 11
    assert len(variants) == 12
    assert next(
        variant
        for variant in variants
        if variant.key_mode == 4 and variant.difficulty == "EASY"
    ).selected_seed == 12


def test_observed_zero_ms_duplicates_exhaust_three_raw_attempts(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        evaluate_chart_candidate,
    )

    class AlwaysDuplicatesAtZero(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            return GeneratedChart(
                notes=[NoteEvent(0, 1), NoteEvent(0, 1)],
                key_mode=generated.key_mode,
                osu_text="",
                generator_name=generated.generator_name,
                seed=generated.seed,
                bpm_events=generated.bpm_events,
            )

    generator = AlwaysDuplicatesAtZero()
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
    assert [call.seed for call in generator.map_calls] == [0, 12, 24]
    assert [attempt["workdir"] for attempt in captured.value.context["attempts"]] == [
        "raw/work/4k-easy/attempt-1",
        "raw/work/4k-easy/attempt-2",
        "raw/work/4k-easy/attempt-3",
    ]
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

    def evaluate(*args, requested_difficulty, **kwargs):
        del args, kwargs
        evaluations.append(len(evaluations) + 1)
        if len(evaluations) == 1:
            return retry
        if len(evaluations) == 2:
            assert not (tmp_path / "raw" / "4k-easy.osu").exists()
        return _acceptance_for_difficulty(accepted, requested_difficulty)

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
    assert first.acceptance == _acceptance_for_difficulty(accepted, "EASY")
    assert first.raw_osu_path == tmp_path / "raw" / "4k-easy.osu"
    assert first.raw_osu_path.is_file()
    assert first.attempt_evidence == (
        {
            "seed": 0,
            "workdir": "raw/work/4k-easy/attempt-1",
            "gateReport": retry.to_report(),
        },
    )


def test_review_candidate_is_published_without_consuming_another_seed(
    monkeypatch, tmp_path: Path
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    review = _acceptance_with_action(authority, GateAction.REVIEW)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        lambda *args, requested_difficulty, **kwargs: _acceptance_for_difficulty(
            review, requested_difficulty
        ),
    )

    class EvidenceGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            (workdir / "candidate.osu").write_text("candidate", encoding="utf-8")
            return generated

    generator = EvidenceGenerator()
    variants = run_generation(
        prepared,
        authority,
        _analysis(),
        tmp_path,
        generator=generator,
        seed=0,
    )

    assert len(variants) == 12
    assert all(variant.acceptance.action is GateAction.REVIEW for variant in variants)
    assert [request.seed for request in generator.map_calls] == list(range(12))
    assert (
        tmp_path / "raw" / "work" / "4k-easy" / "attempt-1" / "candidate.osu"
    ).is_file()
    assert (tmp_path / "raw" / "4k-easy.osu").is_file()


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
    evaluation_calls = 0

    def evaluate(*args, **kwargs):
        nonlocal evaluation_calls
        del args, kwargs
        evaluation_calls += 1
        return retry

    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        evaluate,
    )

    class GateThenLegacyFailureGenerator(RecordingGenerator):
        def generate_map(self, request, workdir):
            generated = super().generate_map(request, workdir)
            if request.seed == 0:
                return generated
            raise WorkerError(
                ErrorCode.CHART_GENERATION_FAILED,
                "fixture legacy generation failure",
            )

    generator = GateThenLegacyFailureGenerator()
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
    assert evaluation_calls == 1
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
            chord_size = {
                "EASY": 1,
                "NORMAL": 2,
                "HARD": min(3, request.key_mode),
                "EXPERT": min(4, request.key_mode),
            }[request.difficulty]
            return replace(
                generated,
                notes=[
                    NoteEvent(time_ms, lane)
                    for time_ms in range(125, 2_000, 125)
                    for lane in range(chord_size)
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


def test_generated_timing_identity_defects_exhaust_with_gate_evidence(
    monkeypatch, tmp_path: Path,
):
    prepared = _prepared(tmp_path)
    authority = _authority(prepared, tmp_path)
    monkeypatch.setattr(
        s2_generate,
        "evaluate_chart_candidate",
        evaluate_chart_candidate,
    )

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
    assert [attempt["seed"] for attempt in captured.value.context["attempts"]] == [
        0,
        12,
        24,
    ]
    assert [attempt["workdir"] for attempt in captured.value.context["attempts"]] == [
        "raw/work/4k-easy/attempt-1",
        "raw/work/4k-easy/attempt-2",
        "raw/work/4k-easy/attempt-3",
    ]
    assert all(
        attempt["gateReport"]["decisions"]["TIMING_IDENTITY"]["reasons"]
        == ["TIMING_REFERENCE_MISMATCH"]
        for attempt in captured.value.context["attempts"]
    )
    assert not (tmp_path / "raw" / "4k-easy.osu").exists()
