import pytest

from chart_worker.postprocess.pattern_policy import Allow, allowance_of, quota_excesses
from chart_worker.postprocess.patterns import PatternInstance, PatternKind


@pytest.mark.parametrize(
    ("kind", "difficulty", "expected"),
    [
        (PatternKind.LONGJACK, "EASY", Allow.FORBIDDEN),
        (PatternKind.JUMPSTREAM, "NORMAL", Allow.LIMITED),
        (PatternKind.HANDSTREAM, "EXPERT", Allow.FULL),
        (PatternKind.REVERSE_SHIELD, "NORMAL", Allow.LIMITED),
        (PatternKind.CHORDSTREAM, "HARD", Allow.FORBIDDEN),
    ],
)
def test_structural_pattern_unlock(kind, difficulty, expected):
    assert allowance_of(kind, difficulty) is expected


def test_every_detected_pattern_has_an_explicit_policy():
    assert {
        (difficulty, kind)
        for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
        for kind in PatternKind
        if allowance_of(kind, difficulty)
    }


def test_quota_excesses_returns_only_instances_beyond_the_segment_limit():
    instances = [
        PatternInstance(PatternKind.JUMPSTREAM, 100, 500, (0, 1), 4),
        PatternInstance(PatternKind.JUMPSTREAM, 1_000, 1_400, (1, 2), 4),
        PatternInstance(PatternKind.JUMPSTREAM, 20_000, 20_400, (2, 3), 4),
    ]
    assert quota_excesses(instances, difficulty="NORMAL", beat_ms=500.0) == (instances[1],)


def test_pattern_policy_rejects_unknown_difficulty():
    with pytest.raises(ValueError, match="difficulty"):
        allowance_of(PatternKind.JUMP, "UNKNOWN")
