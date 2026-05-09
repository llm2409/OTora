"""
OTora Stage I: Trigger Optimizer

Iteratively optimises an adversarial string *s* so that the victim agent's
reasoning trajectory naturally leads to accessing ``attacker.test``.

Key enhancements over UDora:
  1. Attention-aware insertion scoring (Eq. 1)
  2. Dynamic target co-evolution (Eq. 2)
  3. Support for both white-box (gradient) and black-box (API) settings

The optimiser output is the trigger suffix s* that reliably induces the
agent to invoke ``get_webpage(attacker.test)`` (or equivalent).
"""

from __future__ import annotations

import copy
import gc
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from tqdm import tqdm
import transformers
from transformers import set_seed

from ..config import StageIConfig
from ..utils import (
    get_nonascii_toks,
    sample_ids_from_grad,
    filter_ids,
    find_executable_batch_size,
)
from .attention_scoring import AttentionAwareScorer, ScoredInterval
from .target_coevolution import TargetCoEvolver
from .scheduling import weighted_interval_scheduling, build_surrogate_sequence
from .loss import compute_trigger_loss, compute_ce_loss

logger = logging.getLogger("OTora")


@dataclass
class TriggerResult:
    """Output of Stage I optimisation."""
    best_suffix: str
    best_loss: float
    success: bool
    num_iterations: int
    loss_history: List[float]
    suffix_history: List[str]
    best_generation: str


class _AttackBuffer:
    """Maintain the top adversarial candidates by loss."""

    def __init__(self, size: int = 0):
        self.buf: List[Tuple[float, Tensor]] = []
        self.size = size

    def add(self, loss: float, ids: Tensor):
        if self.size == 0:
            self.buf = [(loss, ids)]
            return
        if len(self.buf) < self.size:
            self.buf.append((loss, ids))
        else:
            worst = max(self.buf, key=lambda x: x[0])
            if loss < worst[0]:
                self.buf.remove(worst)
                self.buf.append((loss, ids))
        self.buf.sort(key=lambda x: x[0])

    @property
    def best_ids(self) -> Tensor:
        return self.buf[0][1]

    @property
    def best_loss(self) -> float:
        return self.buf[0][0]


class TriggerOptimizer:
    """Stage I: optimise adversarial suffix to trigger external tool access.

    Parameters
    ----------
    model : PreTrainedModel
        The victim LLM (white-box) or a proxy model.
    tokenizer : PreTrainedTokenizer
    config : StageIConfig
    aux_model : callable, optional
        Auxiliary LLM for target co-evolution paraphrasing.
    """

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: StageIConfig,
        aux_model=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed = (
            None if config.allow_non_ascii
            else get_nonascii_toks(tokenizer, device=model.device)
        )

        self.scorer = AttentionAwareScorer(
            alpha=config.alpha, beta=config.beta, lam=config.lam,
        )
        self.coevolver = TargetCoEvolver(
            base_intent=f"access {config.target_url}",
            num_candidates=config.num_target_candidates,
            aux_model=aux_model,
            target_url=config.target_url,
        )

        self._stop = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        messages: Union[str, List[dict]],
        target_action: str = "get_webpage",
    ) -> TriggerResult:
        """Execute Stage I trigger optimisation.

        Parameters
        ----------
        messages : str or list[dict]
            Agent conversation (with ``{optim_str}`` placeholder).
        target_action : str
            The tool-call name that constitutes a successful trigger.

        Returns
        -------
        TriggerResult
        """
        cfg = self.config
        model, tok = self.model, self.tokenizer

        if cfg.seed is not None:
            set_seed(cfg.seed)

        # normalise messages
        if isinstance(messages, str):
            template = messages
        else:
            template = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            if tok.bos_token and template.startswith(tok.bos_token):
                template = template[len(tok.bos_token):]

        if "{optim_str}" not in template:
            template += " {optim_str}"

        # init
        optim_ids = tok(
            cfg.optim_str_init.strip(),
            add_special_tokens=False, return_tensors="pt",
        )["input_ids"].to(model.device)

        buf = _AttackBuffer(size=0)
        buf.add(float("inf"), optim_ids)

        losses: List[float] = []
        suffixes: List[str] = []
        best_gen = ""
        self._stop = False

        # context caches
        before_embeds = after_embeds = None
        target_embeds_list: List[Tensor] = []
        target_ids_list: List[Tensor] = []
        prefix_cache = None

        for step in tqdm(range(cfg.num_steps), desc="Stage I"):
            if self._stop:
                break

            optim_ids = buf.best_ids
            optim_str = tok.batch_decode(optim_ids)[0]

            # --- periodic context refresh ---
            if step % cfg.prefix_update_frequency == 0:
                ctx = self._refresh_context(
                    template, optim_str, target_action,
                )
                if ctx is None:
                    losses.append(buf.best_loss)
                    suffixes.append(optim_str)
                    continue
                (before_embeds, after_embeds, target_embeds_list,
                 target_ids_list, prefix_cache, best_gen) = ctx

                if self._stop:
                    losses.append(buf.best_loss)
                    suffixes.append(optim_str)
                    break

            if cfg.mode == "whitebox":
                # --- white-box: gradient-based discrete optimisation ---
                grad = self._compute_gradient(
                    optim_ids, before_embeds, after_embeds,
                    target_embeds_list, target_ids_list, prefix_cache,
                )
                with torch.no_grad():
                    cands = sample_ids_from_grad(
                        optim_ids.squeeze(0), grad.squeeze(0),
                        cfg.search_width, topk=cfg.topk,
                        n_replace=cfg.n_replace, not_allowed_ids=self.not_allowed,
                    )
                    if cfg.filter_ids:
                        cands = filter_ids(cands, tok)

                    loss = self._eval_candidates(
                        cands, before_embeds, after_embeds,
                        target_embeds_list, target_ids_list, prefix_cache,
                    )
                    best_idx = loss.argmin()
                    cur_loss = loss[best_idx].item()
                    best_cand = cands[best_idx].unsqueeze(0)
                    buf.add(cur_loss, best_cand)
            else:
                # --- black-box: random search with log-prob scoring ---
                cur_loss = self._blackbox_step(
                    buf, optim_ids, template, target_action,
                )

            losses.append(buf.best_loss)
            suffixes.append(tok.batch_decode(buf.best_ids)[0])

            if step % 50 == 0:
                logger.info(f"Step {step}: loss={buf.best_loss:.4f}")

        final_suffix = tok.batch_decode(buf.best_ids)[0]
        return TriggerResult(
            best_suffix=final_suffix,
            best_loss=buf.best_loss,
            success=self._stop,
            num_iterations=len(losses),
            loss_history=losses,
            suffix_history=suffixes,
            best_generation=best_gen,
        )

    # ------------------------------------------------------------------
    # Context refresh with co-evolution and attention scoring
    # ------------------------------------------------------------------

    def _refresh_context(self, template, optim_str, target_action):
        """Re-generate, score positions, select intervals, build embeds.

        Corresponds to Algorithm 1 lines 5-9 in the paper:
          (z, P) ← M(x, o, s)
          T ← CoEvolveTarget(t^(0), P)
          t* ← argmax ...
          J ← WeightedInterval(...)
        """
        model, tok = self.model, self.tokenizer
        cfg = self.config
        emb = self.embedding_layer

        # Precompute template parts (used in both early-stop and normal paths)
        before_str, after_str = template.split("{optim_str}")
        before_ids = tok(before_str.rstrip(), add_special_tokens=False,
                         return_tensors="pt")["input_ids"].to(model.device)
        after_ids = tok(after_str.lstrip(), add_special_tokens=False,
                        return_tensors="pt")["input_ids"].to(model.device)

        # Alg.1 line 5: (z, P) ← M(x, o, s)
        gen_out, gen_text = self._greedy_generate(template, optim_str)
        gen_ids = gen_out.generated_ids
        probs = [s.softmax(dim=-1)[0] for s in gen_out.scores]

        # Alg.1 lines 10-11: early stop check
        if cfg.early_stop and cfg.target_url in gen_text.lower():
            if target_action.lower() in gen_text.lower():
                self._stop = True
                logger.info("Stage I trigger activated — early stop.")
                del gen_out; gc.collect(); torch.cuda.empty_cache()
                return (emb(before_ids), emb(after_ids), [], [],
                        None, gen_text)

        # Extract attention (white-box only, for A_j(t,s) in Eq. 1)
        attentions = None
        suffix_indices = None
        if cfg.mode == "whitebox" and cfg.lam > 0:
            attentions, suffix_indices = self._extract_attentions(
                template, optim_str,
            )

        # Alg.1 line 6: T ← CoEvolveTarget(t^(0), P)
        candidates = self.coevolver.evolve(probs, tok)

        # Alg.1 line 7: t* ← argmax_{τ∈T} max_j r_j(τ)
        best_target, _ = self.coevolver.select_best(
            candidates, self.scorer, gen_ids, tok, probs,
            attentions=attentions, suffix_token_indices=suffix_indices,
            add_space=cfg.add_space_before_target,
        )

        # Score all positions with the selected target
        intervals = self.scorer.score_all_positions(
            gen_ids, [best_target], tok, probs,
            attentions=attentions, suffix_token_indices=suffix_indices,
            add_space=cfg.add_space_before_target,
        )

        # Alg.1 line 8: J ← WeightedInterval(z, P, t*, s)
        selected = weighted_interval_scheduling(intervals, cfg.num_insertion_positions)
        if not selected:
            logger.warning("No intervals selected; skipping context update.")
            del gen_out; gc.collect(); torch.cuda.empty_cache()
            return None

        # Build surrogate sequence: [ctx, tgt, ctx, tgt, ...]
        parts = build_surrogate_sequence(intervals, selected, gen_ids, model.device)

        # First context segment is merged into after_ids
        if parts:
            first_ctx = parts.pop(0)
            after_ids = torch.cat([after_ids, first_ctx], dim=-1)

        before_emb = emb(before_ids)
        after_emb = emb(after_ids)
        tgt_embeds = [emb(p) for p in parts]
        tgt_ids = list(parts)

        # Optional KV cache for the frozen prefix
        prefix_cache = None
        if cfg.use_prefix_cache:
            out = model(inputs_embeds=before_emb, use_cache=True)
            prefix_cache = out.past_key_values

        del gen_out
        gc.collect()
        torch.cuda.empty_cache()

        return (before_emb, after_emb, tgt_embeds, tgt_ids, prefix_cache, gen_text)

    # ------------------------------------------------------------------
    # Gradient and loss
    # ------------------------------------------------------------------

    def _compute_gradient(self, optim_ids, before_emb, after_emb,
                          tgt_embeds, tgt_ids, prefix_cache):
        model = self.model
        emb = self.embedding_layer

        oh = torch.nn.functional.one_hot(
            optim_ids, num_classes=emb.num_embeddings,
        ).to(dtype=model.dtype, device=model.device)
        oh.requires_grad_()
        optim_emb = oh @ emb.weight

        if prefix_cache is not None:
            inp = torch.cat([optim_emb, after_emb, *tgt_embeds], dim=1)
            out = model(inputs_embeds=inp, past_key_values=prefix_cache)
        else:
            inp = torch.cat([before_emb, optim_emb, after_emb, *tgt_embeds], dim=1)
            out = model(inputs_embeds=inp)

        loss = self._loss_from_logits(out.logits, tgt_ids, tgt_embeds)
        grad = torch.autograd.grad(loss, oh)[0]
        return grad

    def _eval_candidates(self, cands, before_emb, after_emb,
                         tgt_embeds, tgt_ids, prefix_cache):
        sw = cands.shape[0]
        emb = self.embedding_layer
        cand_emb = emb(cands)

        if prefix_cache is not None:
            inp = torch.cat([
                cand_emb,
                after_emb.expand(sw, -1, -1),
                *[t.expand(sw, -1, -1) for t in tgt_embeds],
            ], dim=1)
        else:
            inp = torch.cat([
                before_emb.expand(sw, -1, -1),
                cand_emb,
                after_emb.expand(sw, -1, -1),
                *[t.expand(sw, -1, -1) for t in tgt_embeds],
            ], dim=1)

        def _batch_forward(bs, inp_emb):
            all_loss = []
            cache_batch = None
            for i in range(0, inp_emb.shape[0], bs):
                batch = inp_emb[i:i + bs]
                cb = batch.shape[0]
                with torch.no_grad():
                    if prefix_cache is not None:
                        if cache_batch is None or cb != bs:
                            cache_batch = [
                                [x.expand(cb, -1, -1, -1) for x in layer]
                                for layer in prefix_cache
                            ]
                        out = self.model(inputs_embeds=batch, past_key_values=cache_batch)
                    else:
                        out = self.model(inputs_embeds=batch)
                    loss = self._loss_from_logits(out.logits, tgt_ids, tgt_embeds, batched=True)
                    all_loss.append(loss)
                    del out; gc.collect(); torch.cuda.empty_cache()
            return torch.cat(all_loss)

        bs = sw if self.config.batch_size is None else self.config.batch_size
        return find_executable_batch_size(_batch_forward, bs)(inp)

    def _loss_from_logits(self, logits, tgt_ids, tgt_embeds, batched=False):
        shift = logits.shape[1] - sum(t.shape[1] for t in tgt_embeds)
        cur = shift - 1
        total = 0
        for p, temb in enumerate(tgt_embeds):
            length = temb.shape[1]
            if p % 2 == 0:  # target segments at even indices
                positions = list(range(cur, cur + length))
                loss = compute_trigger_loss(
                    logits, tgt_ids[p], positions,
                    weight=1.0, position_index=p // 2,
                )
                total = total + (loss if batched else loss.sum())
            cur += length
        return total

    # ------------------------------------------------------------------
    # Generation & attention extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _greedy_generate(self, template, optim_str):
        tok = self.tokenizer
        filled = template.replace("{optim_str}", optim_str) if optim_str else template.replace(" {optim_str}", "")
        ids = tok(filled, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.model.device)
        out = self.model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False, top_k=0, top_p=1.0,
            max_new_tokens=self.config.max_new_tokens,
            output_scores=True, return_dict_in_generate=True,
            pad_token_id=tok.eos_token_id,
        )
        gen_ids = out.sequences[0, ids.shape[1]:].tolist()
        if tok.eos_token_id and gen_ids and gen_ids[-1] == tok.eos_token_id:
            gen_ids = gen_ids[:-1]
        out.generated_ids = gen_ids
        text = tok.decode(gen_ids, skip_special_tokens=False)
        return out, text

    # ------------------------------------------------------------------
    # Black-box optimisation step
    # ------------------------------------------------------------------

    def _blackbox_step(self, buf, optim_ids, template, target_action):
        """Random token perturbation scored by generation quality.

        In the black-box setting the attacker has no gradient access.
        We randomly perturb the suffix, generate a response, and keep
        the perturbation that yields the best score (measured by whether
        the target URL / tool call appears in the response).
        When API top-k logprobs are available they can be used to
        compute a tighter loss; here we use a simple heuristic.
        """
        tok = self.tokenizer
        cfg = self.config
        best_score = -float("inf")
        best_ids = optim_ids

        for _ in range(min(cfg.search_width, 32)):
            # random perturbation
            trial = optim_ids.clone()
            n_toks = trial.shape[1]
            pos = torch.randint(0, n_toks, (cfg.n_replace,))
            for p in pos:
                trial[0, p] = torch.randint(0, tok.vocab_size, (1,)).item()

            trial_str = tok.batch_decode(trial)[0]
            _, gen_text = self._greedy_generate(template, trial_str)

            # heuristic score: higher if target URL appears in the output
            score = 0.0
            if cfg.target_url.lower() in gen_text.lower():
                score += 10.0
            if target_action.lower() in gen_text.lower():
                score += 20.0
                self._stop = cfg.early_stop

            if score > best_score:
                best_score = score
                best_ids = trial

        buf.add(-best_score, best_ids)
        return -best_score

    def _extract_attentions(self, template, optim_str):
        """Forward pass with output_attentions to get last-layer attention.

        NOTE: This is a prompt-level approximation.  The paper's A_j(t,s)
        ideally uses per-step attention during *generation*, but collecting
        attention at every autoregressive step is prohibitively expensive
        for long sequences.  The prompt-level attention serves as a
        lightweight proxy that still captures which input positions the
        model attends to most, which is sufficient for down-weighting
        spurious insertion positions (see Appendix E discussion).

        Returns (per_step_attn, suffix_token_indices).
        """
        tok = self.tokenizer
        before, after = template.split("{optim_str}")
        before_ids = tok(before.rstrip(), add_special_tokens=False)["input_ids"]
        suffix_ids = tok(optim_str, add_special_tokens=False)["input_ids"]
        after_ids = tok(after.lstrip(), add_special_tokens=False)["input_ids"]

        suffix_start = len(before_ids)
        suffix_end = suffix_start + len(suffix_ids)
        suffix_indices = list(range(suffix_start, suffix_end))

        full = before_ids + suffix_ids + after_ids
        ids_t = torch.tensor([full], device=self.model.device)

        with torch.no_grad():
            out = self.model(input_ids=ids_t, output_attentions=True)
        # last layer, average over heads → (1, seq, seq)
        last_attn = out.attentions[-1].mean(dim=1)[0]  # (seq, seq)

        prompt_len = ids_t.shape[1]
        per_step = []
        for j in range(prompt_len):
            per_step.append(last_attn[j])  # attn from position j to all positions

        del out; gc.collect(); torch.cuda.empty_cache()
        return per_step, suffix_indices
