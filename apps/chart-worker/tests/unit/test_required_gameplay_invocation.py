from dataclasses import replace
from pathlib import Path

import pytest

from chart_worker.config import WorkerConfig
from chart_worker.generation.params import GenerationRequest
from chart_worker.generation.required_gameplay_interval import (
    RequiredGameplayEvidenceClass,
    RequiredGameplayGroupType,
    RequiredGameplayIntervalMode,
    RequiredGameplayIntervalV1,
)
from chart_worker.generation.required_gameplay_invocation import (
    required_gameplay_invocation_digest,
    required_gameplay_invocation_payload,
)

_SHA_A = "a" * 64


@pytest.fixture
def interval() -> RequiredGameplayIntervalV1:
    return RequiredGameplayIntervalV1(
        start_ms=5_000,
        end_ms=6_000,
        minimum_complete_groups=1,
        allowed_group_types=(
            RequiredGameplayGroupType.HOLD_START,
            RequiredGameplayGroupType.TAP,
        ),
        evidence_class=RequiredGameplayEvidenceClass.BROADBAND_ATTACK,
        evidence_digest=_SHA_A,
        mode=RequiredGameplayIntervalMode.OBSERVE,
    )


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    audio = tmp_path / "first-title.flac"
    reference = tmp_path / "first-reference.osu"
    audio.write_bytes(b"canonical-audio")
    reference.write_bytes(b"canonical-reference")
    return audio, reference


def _request(
    inputs: tuple[Path, Path], interval: RequiredGameplayIntervalV1
) -> GenerationRequest:
    audio, reference = inputs
    return GenerationRequest(
        audio_path=audio,
        timing_reference_path=reference,
        key_mode=4,
        difficulty="HARD",
        year=2023,
        seed=17,
        cfg_scale=1.0,
        descriptors=("style/mixed rice", "streams/bursts"),
        requested_star=2.0,
        duration_ms=20_000,
        music_end_ms=18_000,
        generation_end_ms=18_100,
        last_attack_ms=17_900,
        max_note_start_ms=18_000,
        partial_start_ms=4_000,
        partial_end_ms=8_000,
        add_to_beatmap=True,
        required_gameplay_interval=interval,
    )


def test_payload_contains_only_runtime_semantics_and_content_identities(
    inputs: tuple[Path, Path], interval: RequiredGameplayIntervalV1
):
    payload = required_gameplay_invocation_payload(
        WorkerConfig(), _request(inputs, interval)
    )

    assert payload == {
        "version": "required-gameplay-invocation-v1",
        "input": {
            "audioSha256": "3eefd3e3eb3fdf888300551242398f848640d23ce4369e2e4db0778997375d6c",
            "referenceSha256": "7daa016e5b57bd9e0dcef15f629965f70336af3950c1c1b6e1ed0329318125d3",
        },
        "runtime": {
            "configName": "v32",
            "fastDecoderLoop": True,
            "maniaHoldStateMode": "incremental",
            "precision": "fp16",
            "resnapEvents": True,
        },
        "generation": {
            "addToBeatmap": True,
            "cfgScale": 1.0,
            "descriptors": ["style/mixed rice", "streams/bursts"],
            "difficulty": 2.0,
            "endTimeMs": 8_000,
            "gamemode": 3,
            "inContext": ["TIMING"],
            "keycount": 4,
            "lastAttackTimeMs": 18_000,
            "outputType": ["MAP"],
            "parallel": False,
            "seed": 17,
            "startTimeMs": 4_000,
            "superTiming": False,
            "year": 2023,
        },
        "requiredGameplayInterval": {
            "allowedGroupTypes": ["HOLD_START", "TAP"],
            "endTimeMs": 6_000,
            "evidenceClass": "BROADBAND_ATTACK",
            "evidenceDigest": _SHA_A,
            "minimumCompleteGroups": 1,
            "mode": "OBSERVE",
            "startTimeMs": 5_000,
        },
    }
    serialized = repr(payload).lower()
    assert "first-title" not in serialized
    assert "first-reference" not in serialized
    assert "hard" not in serialized
    assert "song_session" not in serialized
    assert required_gameplay_invocation_digest(
        WorkerConfig(), _request(inputs, interval)
    ) == "84bdae33e91b705cad0eb52bcc5478a29153fc9ceb911d29b1006aaae398bbcc"


def test_digest_ignores_paths_when_input_bytes_and_semantics_match(
    tmp_path: Path, inputs: tuple[Path, Path], interval: RequiredGameplayIntervalV1
):
    request = _request(inputs, interval)
    other_dir = tmp_path / "renamed"
    other_dir.mkdir()
    audio = other_dir / "unrelated.flac"
    reference = other_dir / "unrelated.osu"
    audio.write_bytes(inputs[0].read_bytes())
    reference.write_bytes(inputs[1].read_bytes())

    assert required_gameplay_invocation_digest(
        WorkerConfig(), request
    ) == required_gameplay_invocation_digest(
        WorkerConfig(),
        replace(request, audio_path=audio, timing_reference_path=reference),
    )


def test_shadow_enforcement_mode_is_hash_bound_without_changing_other_semantics(
    inputs: tuple[Path, Path], interval: RequiredGameplayIntervalV1
):
    observed = _request(inputs, interval)
    shadow = replace(
        observed,
        required_gameplay_interval=replace(
            interval,
            mode=RequiredGameplayIntervalMode.SHADOW_ENFORCE,
        ),
    )

    payload = required_gameplay_invocation_payload(WorkerConfig(), shadow)

    assert payload["requiredGameplayInterval"]["mode"] == "SHADOW_ENFORCE"
    assert required_gameplay_invocation_digest(
        WorkerConfig(), observed
    ) != required_gameplay_invocation_digest(WorkerConfig(), shadow)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda request, tmp_path: replace(request, key_mode=6),
        lambda request, tmp_path: replace(request, requested_star=2.1),
        lambda request, tmp_path: replace(request, seed=18),
        lambda request, tmp_path: replace(request, partial_start_ms=3_999),
        lambda request, tmp_path: replace(request, partial_end_ms=8_001),
        lambda request, tmp_path: replace(request, max_note_start_ms=17_999),
        lambda request, tmp_path: replace(
            request, descriptors=tuple(reversed(request.descriptors))
        ),
        lambda request, tmp_path: replace(
            request,
            required_gameplay_interval=replace(
                request.required_gameplay_interval,
                minimum_complete_groups=2,
            ),
        ),
    ],
)
def test_digest_changes_for_every_generation_semantic(
    mutate,
    tmp_path: Path,
    inputs: tuple[Path, Path],
    interval: RequiredGameplayIntervalV1,
):
    request = _request(inputs, interval)
    changed = mutate(request, tmp_path)

    assert required_gameplay_invocation_digest(
        WorkerConfig(), request
    ) != required_gameplay_invocation_digest(WorkerConfig(), changed)


def test_digest_changes_when_input_content_changes(
    inputs: tuple[Path, Path], interval: RequiredGameplayIntervalV1
):
    request = _request(inputs, interval)
    before = required_gameplay_invocation_digest(WorkerConfig(), request)
    inputs[0].write_bytes(b"different-audio")

    assert required_gameplay_invocation_digest(WorkerConfig(), request) != before


@pytest.mark.parametrize("seed", [None, True, -1, 2**32])
def test_required_interval_request_requires_an_exact_uint32_seed(
    seed: object,
    inputs: tuple[Path, Path],
    interval: RequiredGameplayIntervalV1,
):
    with pytest.raises((TypeError, ValueError), match="seed"):
        replace(_request(inputs, interval), seed=seed)


def test_payload_rejects_missing_or_symlink_inputs(
    tmp_path: Path, inputs: tuple[Path, Path], interval: RequiredGameplayIntervalV1
):
    request = _request(inputs, interval)
    missing = replace(request, audio_path=tmp_path / "missing.flac")
    with pytest.raises(FileNotFoundError, match="audio_path"):
        required_gameplay_invocation_payload(WorkerConfig(), missing)

    symlink = tmp_path / "linked.flac"
    try:
        symlink.symlink_to(inputs[0])
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(FileNotFoundError, match="audio_path"):
        required_gameplay_invocation_payload(
            WorkerConfig(), replace(request, audio_path=symlink)
        )
