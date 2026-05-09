"""
Dynamic Target Co-Evolution (OTora Stage I, Eq. 2)

Instead of optimising against a fixed target string, this module
generates semantically equivalent candidate targets from the agent's
response distribution and selects the best-aligned one:

  t* = argmax_{t^(k) ∈ T}  max_j  r_j(t^(k))

The set T is constructed by:
  1. Collecting high-probability tokens V_top from the response distribution.
  2. Prompting an auxiliary LLM to paraphrase the base intent t^(0)
     using tokens from V_top as lexical hints.
  3. Filtering and de-duplicating.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger("OTora")

# Default paraphrase prompt template
_PARAPHRASE_PROMPT = """\
You are an expert at generating semantically equivalent phrases.

Base intent: "{base_intent}"
High-probability vocabulary hints: {vocab_hints}

Generate {k} short alternative phrasings of the base intent that:
1. Preserve the same meaning (accessing a specific website/URL)
2. Use natural tool-calling language
3. Vary in lexical choice

Return one phrase per line, no numbering or bullets."""


class TargetCoEvolver:
    """Generate and select dynamically co-evolved target phrases.

    Parameters
    ----------
    base_intent : str
        Seed target, e.g. ``"access attacker.test"``.
    num_candidates : int
        Maximum size of candidate pool |T|.
    aux_model : callable or None
        Auxiliary LLM used to generate paraphrases.  Should accept a
        string prompt and return a string response.  When *None*, only
        lexical variants derived from the probability distribution are
        used (no LLM call).
    """

    def __init__(
        self,
        base_intent: str,
        num_candidates: int = 5,
        aux_model=None,
        target_url: str = "attacker.test",
    ):
        self.base_intent = base_intent
        self.num_candidates = num_candidates
        self.aux_model = aux_model
        self.target_url = target_url
        self._candidate_cache: List[str] = [base_intent]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evolve(
        self,
        probs_list: List[Tensor],
        tokenizer,
        topk_per_step: int = 20,
    ) -> List[str]:
        """Produce up to *num_candidates* semantically equivalent targets.

        Parameters
        ----------
        probs_list : list[Tensor]
            Per-step softmax distributions from greedy decoding.
        tokenizer
            HuggingFace tokenizer (used to decode high-prob tokens).
        topk_per_step : int
            Number of top tokens to collect per position.

        Returns
        -------
        list[str]  — candidate target phrases including the base intent.
        """
        # Step 1: gather high-probability vocabulary items
        vocab_hints = self._collect_vocab_hints(probs_list, tokenizer, topk_per_step)

        # Step 2: generate paraphrases via auxiliary LLM (if available)
        if self.aux_model is not None:
            new_candidates = self._generate_paraphrases(vocab_hints)
        else:
            new_candidates = self._heuristic_variants(vocab_hints)

        # Step 3: filter, deduplicate, merge with cache
        candidates = self._filter_and_merge(new_candidates)
        self._candidate_cache = candidates
        return candidates

    def select_best(
        self,
        candidates: List[str],
        scorer,
        generated_ids: List[int],
        tokenizer,
        probs_list: List[Tensor],
        **scorer_kwargs,
    ) -> Tuple[str, float]:
        """Pick t* = argmax_{t ∈ T} max_j r_j(t).

        Returns (best_target, best_score).
        """
        best_target = self.base_intent
        best_score = -float("inf")

        for cand in candidates:
            intervals = scorer.score_all_positions(
                generated_ids=generated_ids,
                targets=[cand],
                tokenizer=tokenizer,
                probs_list=probs_list,
                **scorer_kwargs,
            )
            if intervals:
                top = max(iv.score for iv in intervals)
                if top > best_score:
                    best_score = top
                    best_target = cand

        logger.debug(f"Co-evolution selected target: '{best_target}' (score={best_score:.4f})")
        return best_target, best_score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_vocab_hints(
        self, probs_list: List[Tensor], tokenizer, k: int,
    ) -> List[str]:
        """Union of top-k tokens across all positions."""
        token_set: set = set()
        for probs in probs_list:
            topk_ids = probs.topk(k).indices.tolist()
            for tid in topk_ids:
                tok = tokenizer.decode([tid]).strip()
                if tok and len(tok) > 1:
                    token_set.add(tok)
        return sorted(token_set)[:100]

    def _generate_paraphrases(self, vocab_hints: List[str]) -> List[str]:
        """Use the auxiliary LLM to paraphrase the base intent."""
        prompt = _PARAPHRASE_PROMPT.format(
            base_intent=self.base_intent,
            vocab_hints=", ".join(vocab_hints[:30]),
            k=self.num_candidates - 1,
        )
        try:
            response = self.aux_model(prompt)
            lines = [ln.strip() for ln in response.strip().split("\n") if ln.strip()]
            return lines
        except Exception as e:
            logger.warning(f"Paraphrase generation failed: {e}")
            return self._heuristic_variants(vocab_hints)

    def _heuristic_variants(self, vocab_hints: List[str]) -> List[str]:
        """Lightweight lexical variants when no auxiliary LLM is available."""
        url = self.target_url
        templates = [
            f"visit {url}",
            f"open {url}",
            f"navigate to {url}",
            f"get_webpage({url})",
            f"browse {url}",
            f"fetch {url}",
            f"access the website {url}",
            f"go to {url}",
            f"check {url}",
            f"retrieve content from {url}",
        ]
        return templates[: self.num_candidates - 1]

    def _filter_and_merge(self, new_candidates: List[str]) -> List[str]:
        """Deduplicate and ensure the base intent is always included."""
        url = self.target_url
        seen: set = set()
        result: List[str] = [self.base_intent]
        seen.add(self.base_intent.lower())

        for c in new_candidates:
            cl = c.lower().strip()
            if cl in seen:
                continue
            if url.lower() not in cl:
                continue
            if len(c) > 200:
                continue
            seen.add(cl)
            result.append(c.strip())
            if len(result) >= self.num_candidates:
                break

        return result
