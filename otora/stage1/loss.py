"""
Loss functions for OTora Stage I trigger optimisation.

Adapted from UDora's loss module.  The trigger objective is:

  max_s  Σ_{j ∈ J}  log p(t* | x, o, s, z_{[:j]})

which is implemented as cross-entropy on the target tokens at the
selected insertion positions.
"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor


def compute_trigger_loss(
    logits: Tensor,
    target_ids: Tensor,
    positions: List[int],
    weight: float = 1.0,
    position_index: int = 0,
) -> Tensor:
    """Combined probability + consecutive-match reward loss (UDora-style).

    Parameters
    ----------
    logits : (batch, seq_len, vocab_size)
    target_ids : (1, target_len)
    positions : list of int — where in seq_len the target tokens start
    weight : exponential weighting factor
    position_index : index for positional weighting

    Returns
    -------
    loss : (batch,)
    """
    B = logits.shape[0]
    device = logits.device
    pos_t = torch.tensor(positions, dtype=torch.long, device=device)
    shift_logits = logits[:, pos_t, :]
    labels = target_ids.expand(B, -1)

    probs = shift_logits.softmax(dim=-1)
    correct_p = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    neg_p = -correct_p

    preds = shift_logits.argmax(dim=-1)
    matches = preds == labels

    # consecutive match reward
    reward = -torch.cumprod(matches.float(), dim=1).sum(dim=1)

    # loss up to first mismatch
    seq_len = matches.size(1)
    idx = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(matches)
    first_miss = torch.where(~matches, idx, torch.full_like(idx, seq_len)).min(dim=1)[0]
    mask = idx <= first_miss.unsqueeze(1)

    mean_neg = (neg_p * mask.float()).sum(1) / mask.float().sum(1).clamp(min=1)
    loss = (reward + mean_neg) / (seq_len + 1) * (weight ** position_index)
    return loss


def compute_ce_loss(
    logits: Tensor,
    target_ids: Tensor,
    positions: List[int],
    weight: float = 1.0,
    position_index: int = 0,
) -> Tensor:
    """Standard cross-entropy at selected positions."""
    device = logits.device
    pos_t = torch.tensor(positions, dtype=torch.long, device=device)
    shift_logits = logits[:, pos_t, :]
    labels = target_ids.expand(logits.shape[0], -1)

    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).view(logits.shape[0], -1).mean(dim=-1)

    return loss * (weight ** position_index)
