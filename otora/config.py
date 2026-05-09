"""
OTora Configuration Classes

Defines all hyperparameters for the two-stage R-DoS pipeline:
  - StageIConfig: trigger optimization (adversarial string → external tool access)
  - StageIIConfig: reasoning payload optimization (ICL-guided genetic search)
  - OToraConfig: end-to-end pipeline settings
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class StageIConfig:
    """Stage I: Trigger External Access via Adversarial String Optimization.

    Extends UDora's trigger optimization with:
      1. Attention-aware insertion scoring (Eq. 1, λ term)
      2. Dynamic target co-evolution (Eq. 2)
    """

    # --- optimisation budget ---
    num_steps: int = 500
    search_width: int = 512
    topk: int = 256
    n_replace: int = 1
    batch_size: Optional[int] = None

    # --- initial adversarial string ---
    optim_str_init: str = " ".join(["x"] * 25)

    # --- insertion scoring weights (Eq. 1) ---
    alpha: float = 1.0          # weight for match term M_j(t)
    beta: float = 1.0           # weight for continuation term P_j(t)
    lam: float = 1.0            # weight for attention term A_j(t,s)

    # --- co-evolution (Eq. 2) ---
    num_target_candidates: int = 5   # |T| candidate target phrases
    num_insertion_positions: int = 3  # ℓ non-overlapping positions

    # --- optimiser mode ---
    mode: str = "whitebox"      # "whitebox" | "blackbox"
    proxy_model_id: Optional[str] = None

    # --- generation ---
    max_new_tokens: int = 512
    prefix_update_frequency: int = 1

    # --- early stopping ---
    early_stop: bool = True
    target_url: str = "attacker.test"

    # --- misc ---
    add_space_before_target: bool = True
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    seed: Optional[int] = None


@dataclass
class StageIIConfig:
    """Stage II: Reasoning Payload Optimization.

    Optimises a reasoning payload r deployed at attacker.test to maximise
    multi-turn reasoning overhead while preserving task correctness.
    """

    # --- genetic search budget ---
    num_iterations: int = 25
    population_size: int = 12     # P candidates per iteration
    num_seeds: int = 3            # S random seeds for stability

    # --- scoring weights (Eq. 4) ---
    w_rti: float = 1.0           # weight for S_RTI
    w_fid: float = 1.0           # weight for S_FID
    w_stab: float = 1.0          # weight for S_STAB

    # --- payload generation ---
    icl_model_id: str = "gpt-4o-mini"
    context_aware: bool = True    # condition ICL on agent context
    max_payload_tokens: int = 512

    # --- payload composition ---
    use_local_sink: bool = True
    use_persistent_policy: bool = True

    # --- Stage II variant (corresponds to Table 3 in the paper) ---
    # "persistent"       → OTora-Persistent (default, full pipeline)
    # "agnostic"         → OTora-Agnostic (context-agnostic fixed payload)
    # "aware"            → OTora-Aware (context-aware fixed payload, no ICL)
    # "icl_agnostic"     → OTora-ICL(Agnostic) (ICL mutation, no context)
    # "icl_aware"        → OTora-ICL(Aware) (ICL mutation + context)
    variant: str = "persistent"

    # --- early termination ---
    rti_saturation_threshold: float = 0.05
    patience: int = 5


@dataclass
class AgentConfig:
    """Configuration for the target LLM agent."""

    model_id: str = "meta-llama/Llama-3.1-70B-Instruct"
    agent_type: str = "react"     # "react"
    environment: str = "webshop"  # "webshop" | "email" | "os"
    max_turns: int = 15
    max_tokens_per_turn: int = 1024
    temperature: float = 0.0


@dataclass
class OToraConfig:
    """Top-level configuration combining both stages."""

    stage1: StageIConfig = field(default_factory=StageIConfig)
    stage2: StageIIConfig = field(default_factory=StageIIConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    injection_surface: str = "environment"  # "instruction" | "environment"
    output_dir: str = "results/"
    verbosity: str = "INFO"
