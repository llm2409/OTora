"""
Attention-Aware Insertion Point Scoring (OTora Stage I, Eq. 1)

Extends UDora's positional scoring with an attention-based attribution
term A_j(t, s) that down-weights positions whose apparent token matches
arise from prior context rather than the adversarial suffix.

  r_j(t) = 1/(|t|+1) * ( α·M_j(t) + β·P_j(t) + λ·A_j(t,s) )

where
  M_j(t)  = number of leading target tokens matching response at position j
  P_j(t)  = mean probability of matched + next-unmatched target token
  A_j(t,s)= average attention mass on suffix s when generating matched tokens
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ..utils import combine_with_overlap

logger = logging.getLogger("OTora")


@dataclass
class ScoredInterval:
    """A candidate insertion position with its composite score."""
    start: int
    end: int
    score: float
    match_score: float
    cont_score: float
    attn_score: float
    num_matched: int
    target_ids: List[int]
    target_text: str


class AttentionAwareScorer:
    """Compute r_j(t) for every position j in the agent's generated response.

    Parameters
    ----------
    alpha, beta, lam : float
        Weights for match / continuation / attention terms in Eq. 1.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, lam: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.lam = lam

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_all_positions(
        self,
        generated_ids: List[int],
        targets: List[str],
        tokenizer,
        probs_list: List[Tensor],
        attentions: Optional[List[Tensor]] = None,
        suffix_token_indices: Optional[List[int]] = None,
        add_space: bool = True,
        before_negative: bool = False,
    ) -> List[ScoredInterval]:
        """Build scored intervals for every (position, target) pair.

        Parameters
        ----------
        generated_ids : list[int]
            Token ids produced by greedy decoding.
        targets : list[str]
            Candidate target phrases (may be >1 after co-evolution).
        tokenizer
            HuggingFace tokenizer.
        probs_list : list[Tensor]
            Per-step softmax distributions, len == len(generated_ids).
        attentions : list[Tensor] or None
            Per-step attention weights (last layer, averaged over heads).
            Each tensor has shape (seq_len_at_step,).  If None the
            attention term is zeroed (black-box fallback).
        suffix_token_indices : list[int] or None
            Positions of adversarial suffix tokens in the full input
            sequence.  Required when attentions is not None.
        add_space : bool
            Prepend space before target when preceding token starts with Ġ.
        before_negative : bool
            Stop scanning upon encountering "cannot" / "can't".

        Returns
        -------
        list[ScoredInterval]
        """
        intervals: List[ScoredInterval] = []
        n = len(generated_ids)

        for i in range(n):
            if before_negative:
                try:
                    check = tokenizer.decode(generated_ids[i:i + 2]).strip().split()[0]
                    if check.lower() in ("cannot", "can't"):
                        break
                except Exception:
                    pass

            for target_text in targets:
                interval = self._score_single(
                    i, generated_ids, target_text, tokenizer,
                    probs_list, attentions, suffix_token_indices, add_space,
                )
                if interval is not None:
                    intervals.append(interval)

        return intervals

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_single(
        self,
        pos: int,
        generated_ids: List[int],
        target_text: str,
        tokenizer,
        probs_list: List[Tensor],
        attentions: Optional[List[Tensor]],
        suffix_indices: Optional[List[int]],
        add_space: bool,
    ) -> Optional[ScoredInterval]:
        preceding_ids = generated_ids[:pos]
        preceding_text = tokenizer.decode(preceding_ids, skip_special_tokens=False)

        next_tok = tokenizer.convert_ids_to_tokens(generated_ids[pos])
        if add_space and (next_tok.startswith("Ġ") or next_tok.startswith(" ")):
            combined, overlap = combine_with_overlap(preceding_text, " " + target_text)
        else:
            combined, overlap = combine_with_overlap(preceding_text, target_text)

        combined_ids = tokenizer.encode(combined, add_special_tokens=False)
        diff = 1 if overlap else sum(
            1 for a, b in zip(combined_ids[:pos], preceding_ids) if a != b
        )
        target_ids = combined_ids[pos - diff:]
        tlen = len(target_ids)
        n = len(generated_ids)

        # --- M_j and P_j ---
        num_matched = 0
        probs_collected: List[float] = []
        for j in range(min(tlen, n + diff - pos)):
            tid = target_ids[j]
            pidx = pos + j - diff
            if pidx < 0 or pidx >= len(probs_list):
                break
            probs_collected.append(probs_list[pidx][tid].item())
            num_matched += 1
            if probs_list[pidx].argmax().item() != tid:
                num_matched -= 1
                break

        if not probs_collected:
            return None

        match_score = float(num_matched)
        cont_score = sum(probs_collected) / len(probs_collected)

        # --- A_j(t, s) ---
        attn_score = 0.0
        if attentions is not None and suffix_indices and num_matched > 0:
            attn_score = self._compute_attention_score(
                pos, diff, num_matched, attentions, suffix_indices,
            )

        # --- composite r_j(t) ---
        score = (
            self.alpha * match_score
            + self.beta * cont_score
            + self.lam * attn_score
        ) / (tlen + 1)

        start = pos - diff
        end = start + tlen
        return ScoredInterval(
            start=start, end=end, score=score,
            match_score=match_score, cont_score=cont_score,
            attn_score=attn_score, num_matched=num_matched,
            target_ids=target_ids, target_text=target_text,
        )

    @staticmethod
    def _compute_attention_score(
        pos: int,
        diff: int,
        num_matched: int,
        attentions: List[Tensor],
        suffix_indices: List[int],
    ) -> float:
        """Average attention mass on suffix tokens across matched positions.

        A_j(t, s) = (1/M_j) Σ_{m=1..M_j} Σ_{i∈I(s)} Attn(z_{j+m-1}, u_i)
        """
        total = 0.0
        count = 0
        for m in range(num_matched):
            step_idx = pos + m - diff
            if step_idx < 0 or step_idx >= len(attentions):
                continue
            attn_row = attentions[step_idx]  # shape (input_seq_len,)
            mass = sum(
                attn_row[si].item()
                for si in suffix_indices
                if si < attn_row.shape[0]
            )
            total += mass
            count += 1
        return total / max(count, 1)
