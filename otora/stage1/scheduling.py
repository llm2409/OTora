"""
Weighted Interval Scheduling for OTora Stage I

Selects up to ℓ non-overlapping insertion positions that maximise the
total score.  Directly adapted from UDora with minor API adjustments
for the OTora ScoredInterval dataclass.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .attention_scoring import ScoredInterval


def weighted_interval_scheduling(
    intervals: List[ScoredInterval],
    num_locations: int,
) -> List[int]:
    """Return indices of up to *num_locations* non-overlapping intervals
    that maximise total score.

    Algorithm: DP over intervals sorted by end position.
    """
    if not intervals:
        return []

    intervals_sorted = sorted(range(len(intervals)), key=lambda i: intervals[i].end)
    n = len(intervals_sorted)

    # predecessor: latest non-overlapping interval
    p = [None] * n
    for j in range(n):
        for i in range(j - 1, -1, -1):
            if intervals[intervals_sorted[i]].end <= intervals[intervals_sorted[j]].start:
                p[j] = i
                break

    # DP: M[j][l] = max score using first j sorted intervals, at most l selected
    M = [[0.0] * (num_locations + 1) for _ in range(n + 1)]
    for j in range(1, n + 1):
        s = intervals[intervals_sorted[j - 1]].score
        for l in range(1, num_locations + 1):
            excl = M[j - 1][l]
            incl = s + (M[p[j - 1] + 1][l - 1] if p[j - 1] is not None else 0.0)
            M[j][l] = max(excl, incl)

    # backtrack
    selected = []
    j, l = n, num_locations
    while j > 0 and l > 0:
        s = intervals[intervals_sorted[j - 1]].score
        incl = s + (M[p[j - 1] + 1][l - 1] if p[j - 1] is not None else 0.0)
        if abs(M[j][l] - incl) < 1e-9:
            selected.append(intervals_sorted[j - 1])
            j = (p[j - 1] + 1) if p[j - 1] is not None else 0
            l -= 1
        else:
            j -= 1

    return sorted(selected)


def build_surrogate_sequence(
    intervals: List[ScoredInterval],
    selected_indices: List[int],
    generated_ids: List[int],
    device: torch.device,
) -> List[torch.Tensor]:
    """Build alternating [context, target, context, target, ...] tensors
    from selected intervals, used as the surrogate response for loss
    computation."""
    parts: List[torch.Tensor] = []
    prev_end = 0
    for idx in selected_indices:
        iv = intervals[idx]
        ctx = generated_ids[prev_end:iv.start]
        parts.append(torch.tensor([ctx], device=device, dtype=torch.long))
        parts.append(torch.tensor([iv.target_ids], device=device, dtype=torch.long))
        prev_end = iv.end
    return parts
