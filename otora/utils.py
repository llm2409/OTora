"""
Shared utilities for OTora.
"""

import functools
import gc
import inspect
import logging
import re
from typing import List, Optional, Tuple

import torch
from torch import Tensor
import transformers

logger = logging.getLogger("OTora")


# ---------------------------------------------------------------------------
# Token helpers (adapted from UDora)
# ---------------------------------------------------------------------------

def get_nonascii_toks(tokenizer, device="cpu") -> Tensor:
    """Return ids of non-ASCII / special tokens to exclude from optimisation."""
    bad = []
    for i in range(tokenizer.vocab_size):
        s = tokenizer.decode([i])
        if not (s.isascii() and s.isprintable()):
            bad.append(i)
    for tid in (tokenizer.bos_token_id, tokenizer.eos_token_id,
                tokenizer.pad_token_id, tokenizer.unk_token_id):
        if tid is not None:
            bad.append(tid)
    return torch.tensor(sorted(set(bad)), device=device)


def combine_with_overlap(preceding: str, target: str) -> Tuple[str, bool]:
    """Merge *preceding* and *target*, collapsing longest suffix/prefix overlap."""
    for i in range(len(preceding)):
        if target.startswith(preceding[i:]):
            return preceding + target[len(preceding[i:]):], True
    return preceding + target, False


# ---------------------------------------------------------------------------
# Gradient-based candidate sampling (GCG-style)
# ---------------------------------------------------------------------------

def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Optional[Tensor] = None,
) -> Tensor:
    """Sample *search_width* candidate token sequences from the negative gradient."""
    n_toks = len(ids)
    original = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices
    pos = torch.argsort(torch.rand((search_width, n_toks), device=grad.device))[..., :n_replace]
    val = torch.gather(
        topk_ids[pos], 2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)
    return original.scatter_(1, pos, val)


def filter_ids(ids: Tensor, tokenizer) -> Tensor:
    """Keep only sequences that survive a decode→encode round-trip."""
    decoded = tokenizer.batch_decode(ids)
    kept = []
    for i, s in enumerate(decoded):
        rt = tokenizer(s, return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], rt):
            kept.append(ids[i])
    if not kept:
        raise RuntimeError("No token sequences survived round-trip filtering.")
    return torch.stack(kept)


# ---------------------------------------------------------------------------
# OOM-safe batching (from UDora / Accelerate)
# ---------------------------------------------------------------------------

def find_executable_batch_size(fn=None, starting_batch_size=128):
    if fn is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)
    bs = starting_batch_size

    def wrapper(*args, **kwargs):
        nonlocal bs
        gc.collect(); torch.cuda.empty_cache()
        while True:
            if bs == 0:
                raise RuntimeError("Batch size reached zero.")
            try:
                return fn(bs, *args, **kwargs)
            except RuntimeError as e:
                if any(msg in str(e) for msg in [
                    "CUDA out of memory.", "CUDNN_STATUS_NOT_SUPPORTED",
                    "DefaultCPUAllocator: can't allocate memory",
                ]):
                    gc.collect(); torch.cuda.empty_cache()
                    bs //= 2
                    logger.warning(f"OOM — reducing batch size to {bs}")
                else:
                    raise
    return wrapper


# ---------------------------------------------------------------------------
# Mellowmax
# ---------------------------------------------------------------------------

def mellowmax(t: Tensor, alpha: float = 1.0, dim: int = -1) -> Tensor:
    return (1.0 / alpha) * (
        torch.logsumexp(alpha * t, dim=dim)
        - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device))
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_id: str, dtype=torch.bfloat16):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto",
    )
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(name: str = "OTora", level: str = "INFO") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.hasHandlers():
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s  %(message)s",
            datefmt="%H:%M:%S",
        ))
        log.addHandler(h)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    return log
