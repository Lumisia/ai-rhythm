import hashlib
import json
from dataclasses import replace

import pytest

from chart_worker.validation.pairwise_labels import (
    CandidateLabelBindingV1,
    PairwiseLabelExportV1,
    PairwiseLabelV1,
    audit_pairwise_labels,
    bind_pairwise_label_export_v1,
    build_pairwise_task,
    build_pairwise_task_bundle,
    canonical_answer,
    parse_pairwise_label_export_v1,
    parse_pairwise_label_v1,
    parse_pairwise_task_bundle_v1,
)


def _binding(candidate_id: str, *, audio: str = "a" * 64):
    return CandidateLabelBindingV1(
        candidate_id=candidate_id,
        audio_sha256=audio,
        key_mode=4,
        payload_sha256=hashlib.sha256(f"payload:{candidate_id}".encode()).hexdigest(),
        feature_sha256=hashlib.sha256(f"feature:{candidate_id}".encode()).hexdigest(),
    )


def test_review_task_is_hash_bound_but_hides_requested_difficulty_and_provenance():
    task = build_pairwise_task(
        _binding("alpha-1"),
        _binding("beta-2"),
        presentation_seed="blind-seed-1",
    )

    review = task.to_review_report()

    assert set(review) == {"version", "taskId", "left", "right"}
    assert set(review["left"]) == {"payloadSha256"}
    assert "difficulty" not in review["left"]
    assert "provenance" not in review["left"]
    assert len(task.stable_sha256()) == 64


def test_left_right_answer_normalizes_to_the_same_candidate_after_reversal():
    first = build_pairwise_task(
        _binding("alpha-1"),
        _binding("beta-2"),
        presentation_seed="seed-a",
        force_left_candidate_id="alpha-1",
    )
    reversed_task = build_pairwise_task(
        _binding("alpha-1"),
        _binding("beta-2"),
        presentation_seed="seed-b",
        force_left_candidate_id="beta-2",
    )
    left_label = PairwiseLabelV1(
        task_sha256=first.stable_sha256(),
        rater_sha256="f" * 64,
        harder_answer="LEFT",
        musical_quality_answer="TIE",
        confidence=4,
    )
    right_label = replace(
        left_label,
        task_sha256=reversed_task.stable_sha256(),
        harder_answer="RIGHT",
    )

    assert canonical_answer(first, left_label, dimension="harder") == "alpha-1"
    assert canonical_answer(reversed_task, right_label, dimension="harder") == "alpha-1"
    audit = audit_pairwise_labels(((first, left_label), (reversed_task, right_label)))
    assert audit.contradictions == ()
    assert audit.consistent_repeated_pairs == 1


def test_reversed_duplicate_contradiction_is_reported_not_silently_averaged():
    first = build_pairwise_task(
        _binding("alpha-1"),
        _binding("beta-2"),
        presentation_seed="seed-a",
        force_left_candidate_id="alpha-1",
    )
    reversed_task = build_pairwise_task(
        _binding("alpha-1"),
        _binding("beta-2"),
        presentation_seed="seed-b",
        force_left_candidate_id="beta-2",
    )
    labels = (
        (
            first,
            PairwiseLabelV1(
                task_sha256=first.stable_sha256(),
                rater_sha256="f" * 64,
                harder_answer="LEFT",
                musical_quality_answer="TIE",
                confidence=4,
            ),
        ),
        (
            reversed_task,
            PairwiseLabelV1(
                task_sha256=reversed_task.stable_sha256(),
                rater_sha256="f" * 64,
                harder_answer="LEFT",
                musical_quality_answer="TIE",
                confidence=4,
            ),
        ),
    )

    audit = audit_pairwise_labels(labels)

    assert audit.contradictions == (("alpha-1", "beta-2"),)
    assert audit.usable_harder_labels == 0


@pytest.mark.parametrize("answer", ["TIE", "UNCERTAIN"])
def test_tie_and_uncertain_are_preserved(answer: str):
    task = build_pairwise_task(
        _binding("alpha-1"),
        _binding("beta-2"),
        presentation_seed="seed",
    )
    label = PairwiseLabelV1(
        task_sha256=task.stable_sha256(),
        rater_sha256="f" * 64,
        harder_answer=answer,
        musical_quality_answer="UNCERTAIN",
        confidence=2,
    )

    assert canonical_answer(task, label, dimension="harder") == answer


def test_label_rejects_wrong_task_hash_and_non_exact_confidence():
    task = build_pairwise_task(
        _binding("alpha-1"),
        _binding("beta-2"),
        presentation_seed="seed",
    )
    with pytest.raises((TypeError, ValueError)):
        PairwiseLabelV1(
            task_sha256="bad",
            rater_sha256="f" * 64,
            harder_answer="LEFT",
            musical_quality_answer="RIGHT",
            confidence=True,
        )
    label = PairwiseLabelV1(
        task_sha256="0" * 64,
        rater_sha256="f" * 64,
        harder_answer="LEFT",
        musical_quality_answer="RIGHT",
        confidence=3,
    )
    with pytest.raises(ValueError, match="task digest"):
        canonical_answer(task, label, dimension="harder")


def test_bundle_is_deterministic_blinded_and_roundtrips_private_binding():
    bindings = (_binding("alpha-1"), _binding("beta-2"), _binding("gamma-3"))
    pairs = (("alpha-1", "beta-2"), ("alpha-1", "gamma-3"))

    bundle = build_pairwise_task_bundle(
        bindings,
        pairs=pairs,
        presentation_seed="review-bundle-seed",
        include_reversed=True,
    )
    repeated = build_pairwise_task_bundle(
        tuple(reversed(bindings)),
        pairs=pairs,
        presentation_seed="review-bundle-seed",
        include_reversed=True,
    )

    assert bundle == repeated
    assert len(bundle.tasks) == 4
    assert len({task.task_id for task in bundle.tasks}) == 4
    review_text = json.dumps(bundle.to_review_report(), sort_keys=True)
    for secret in (
        "candidateId",
        "audioSha256",
        "keyMode",
        "featureSha256",
        "difficulty",
        "provenance",
    ):
        assert f'"{secret}":' not in review_text
    assert "payloadSha256" in review_text
    assert parse_pairwise_task_bundle_v1(bundle.to_private_report()) == bundle


def test_bundle_rejects_duplicate_noncanonical_or_cross_song_pairs():
    alpha = _binding("alpha-1")
    beta = _binding("beta-2")
    other_song = _binding("gamma-3", audio="b" * 64)

    with pytest.raises(ValueError, match="sorted and unique"):
        build_pairwise_task_bundle(
            (alpha, beta),
            pairs=(("beta-2", "alpha-1"),),
            presentation_seed="seed",
            include_reversed=False,
        )
    with pytest.raises(ValueError, match="share audio"):
        build_pairwise_task_bundle(
            (alpha, other_song),
            pairs=(("alpha-1", "gamma-3"),),
            presentation_seed="seed",
            include_reversed=False,
        )


def test_strict_bundle_and_label_parser_reject_extra_keys_and_preserve_answers():
    bundle = build_pairwise_task_bundle(
        (_binding("alpha-1"), _binding("beta-2")),
        pairs=(("alpha-1", "beta-2"),),
        presentation_seed="seed",
        include_reversed=False,
    )
    report = bundle.to_private_report()
    with pytest.raises(ValueError, match="keys differ"):
        parse_pairwise_task_bundle_v1({**report, "unexpected": True})

    task = bundle.tasks[0]
    label = PairwiseLabelV1(
        task_sha256=task.stable_sha256(),
        rater_sha256="f" * 64,
        harder_answer="LEFT",
        musical_quality_answer="TIE",
        confidence=5,
    )
    assert parse_pairwise_label_v1(label.to_report()) == label
    with pytest.raises(ValueError, match="keys differ"):
        parse_pairwise_label_v1({**label.to_report(), "candidateId": "leak"})


def test_label_export_roundtrips_and_binds_only_to_its_private_packet_contract():
    bundle = build_pairwise_task_bundle(
        (_binding("alpha-1"), _binding("beta-2")),
        pairs=(("alpha-1", "beta-2"),),
        presentation_seed="seed",
        include_reversed=False,
    )
    task = bundle.tasks[0]
    label = PairwiseLabelV1(
        task_sha256=task.stable_sha256(),
        rater_sha256="f" * 64,
        harder_answer="LEFT",
        musical_quality_answer="TIE",
        confidence=5,
    )
    exported = PairwiseLabelExportV1(
        private_bundle_sha256=bundle.stable_sha256(),
        packet_sha256="e" * 64,
        completed_task_count=1,
        total_task_count=1,
        labels=(label,),
    )

    assert parse_pairwise_label_export_v1(exported.to_report()) == exported
    assert bind_pairwise_label_export_v1(
        exported,
        bundle=bundle,
        expected_packet_sha256="e" * 64,
    ) == ((task, label),)

    with pytest.raises(ValueError, match="packet digest"):
        bind_pairwise_label_export_v1(
            exported,
            bundle=bundle,
            expected_packet_sha256="d" * 64,
        )


def test_label_export_rejects_count_mismatch_duplicate_or_unknown_tasks():
    bundle = build_pairwise_task_bundle(
        (_binding("alpha-1"), _binding("beta-2")),
        pairs=(("alpha-1", "beta-2"),),
        presentation_seed="seed",
        include_reversed=False,
    )
    task = bundle.tasks[0]
    label = PairwiseLabelV1(
        task_sha256=task.stable_sha256(),
        rater_sha256="f" * 64,
        harder_answer="LEFT",
        musical_quality_answer="TIE",
        confidence=5,
    )
    with pytest.raises(ValueError, match="completed count"):
        PairwiseLabelExportV1(
            private_bundle_sha256=bundle.stable_sha256(),
            packet_sha256="e" * 64,
            completed_task_count=2,
            total_task_count=2,
            labels=(label,),
        )
    with pytest.raises(ValueError, match="unique task"):
        PairwiseLabelExportV1(
            private_bundle_sha256=bundle.stable_sha256(),
            packet_sha256="e" * 64,
            completed_task_count=2,
            total_task_count=2,
            labels=(label, label),
        )

    unknown = PairwiseLabelExportV1(
        private_bundle_sha256=bundle.stable_sha256(),
        packet_sha256="e" * 64,
        completed_task_count=1,
        total_task_count=1,
        labels=(replace(label, task_sha256="0" * 64),),
    )
    with pytest.raises(ValueError, match="unknown task"):
        bind_pairwise_label_export_v1(
            unknown,
            bundle=bundle,
            expected_packet_sha256="e" * 64,
        )
