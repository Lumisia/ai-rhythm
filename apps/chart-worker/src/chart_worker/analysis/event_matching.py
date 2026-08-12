"""Small ordered-event matching primitives shared by timing diagnostics."""

import numpy as np


def maximum_ordered_match_count(
    estimates: np.ndarray,
    references: np.ndarray,
    *,
    window_ms: int,
) -> int:
    """Return maximum cardinality for sorted one-to-one window matches."""
    estimate_index = 0
    reference_index = 0
    matched = 0
    while estimate_index < estimates.size and reference_index < references.size:
        estimate = int(estimates[estimate_index])
        reference = int(references[reference_index])
        if estimate < reference - window_ms:
            estimate_index += 1
        elif reference < estimate - window_ms:
            reference_index += 1
        else:
            matched += 1
            estimate_index += 1
            reference_index += 1
    return matched
