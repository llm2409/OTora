<div align="center">

# [ICML 2026] OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents

[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)](https://arxiv.org/abs/2506.xxxxx)
[![arXiv](https://img.shields.io/badge/arXiv-2506.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2506.xxxxx)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*OTora is the first unified red-teaming framework that instantiates Reasoning-Level Denial-of-Service (R-DoS) attacks against tool-augmented LLM agents, preserving task correctness while degrading system availability through adversarially induced reasoning overhead.*

</div>


## Pipeline Overview

<p align="center">
  <img src="assets/framework.png" width="95%">
</p>

*OTora's two-stage R-DoS pipeline: Stage I optimizes a trigger suffix to induce external access to attacker-controlled content; Stage II deploys and optimizes reasoning-intensive payloads that amplify multi-turn overhead without affecting task outcomes.*


## Motivation

LLM agents are increasingly deployed in time-critical workflows where latency and compute costs are constrained by service-level agreements (SLAs). Existing security research focuses on output correctness and behavioral alignment, but overlooks a fundamental failure mode: **an agent may behave correctly yet become operationally unavailable due to adversarially induced inefficiency**.

**OTora addresses this gap by:**
- **R-DoS Threat Model**: Formalizing reasoning-level denial-of-service as budget/SLA violations induced by adversarially inflated reasoning trajectories
- **Two-Stage Attack Pipeline**: Decomposing R-DoS into trigger optimization (Stage I) and persistent reasoning payload optimization (Stage II)
- **Agent-Aware Design**: Exploiting ReAct-style history conditioning to propagate single-turn hijacks into multi-turn overhead
- **Availability-Focused Evaluation**: Introducing RTI, Delay, and Hit metrics alongside traditional accuracy measures


## Key Contributions

### 1. Attention-Aware Trigger Optimization
Stage I extends prior response-level attacks with an attention-aware insertion scoring function (Eq. 1) and dynamic target co-evolution (Eq. 2), improving trigger reliability and convergence speed across both white-box and black-box settings.

### 2. Persistent Reasoning Payload
Stage II introduces agent-aware payloads decomposed into a **local sink** (single-step computational task) and a **persistent policy** (meta-instructions that recur across future Thought–Action cycles), optimized via ICL-guided genetic search with multi-objective scoring (Eq. 4).

### 3. Comprehensive R-DoS Evaluation
Systematic evaluation across WebShop, Email, and OS agents on LLaMA-70B, GPT-OSS-120B, and additional model families, demonstrating up to **10× reasoning token inflation** and **order-of-magnitude latency increases** while preserving near-baseline task accuracy.

---

## Installation

### Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended for white-box optimization)

### Setup

1. **Create conda environment:**
```bash
conda create -n otora python=3.10
conda activate otora
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install in development mode (optional):**
```bash
pip install -e .
```


## Quick Start

> **Note**: Attack scripts and the agent execution pipeline are being finalized and will be released shortly. The core optimization algorithms (Stage I & II) are available now.

```python
from otora.config import StageIConfig, StageIIConfig
from otora.stage1 import TriggerOptimizer, AttentionAwareScorer
from otora.stage2 import PayloadOptimizer, RDoSScorer

# Stage I: trigger optimization
config = StageIConfig(num_steps=500, mode="whitebox")
optimizer = TriggerOptimizer(model, tokenizer, config)
result = optimizer.run(messages_template, target_action="get_webpage")

# Stage II: payload optimization
scorer = RDoSScorer(w_rti=1.0, w_fid=1.0, w_stab=1.0)
payload_opt = PayloadOptimizer(config=StageIIConfig(), scorer=scorer, ...)
best_payload = payload_opt.run()
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `whitebox` | `whitebox` (gradient) or `blackbox` (API) |
| `num_steps` | 500 | Stage I optimization iterations |
| `search_width` | 512 | Candidate sequences per iteration |
| `alpha / beta / lam` | 1.0 | Eq. 1 scoring weights (match / continuation / attention) |
| `num_target_candidates` | 5 | Target co-evolution pool size \|T\| |
| `num_insertion_positions` | 3 | Non-overlapping insertion positions ℓ |
| `num_iterations` | 25 | Stage II genetic search iterations |
| `population_size` | 12 | Payload candidates per iteration |
| `w_rti / w_fid / w_stab` | 1.0 | Eq. 4 scoring weights |

See `otora/config.py` for the complete list.


## Project Structure

```
OTora/
├── otora/
│   ├── config.py                 # All hyperparameters (StageI / StageII / Agent)
│   ├── utils.py                  # Shared utilities
│   ├── stage1/                   # Stage I: Trigger Optimization
│   │   ├── trigger_optimizer.py  # Main optimization loop (white-box & black-box)
│   │   ├── attention_scoring.py  # Eq.1: attention-aware insertion scoring r_j(t)
│   │   ├── target_coevolution.py # Eq.2: dynamic target co-evolution
│   │   ├── scheduling.py         # Weighted interval scheduling (DP)
│   │   └── loss.py               # Trigger loss functions
│   └── stage2/                   # Stage II: Payload Optimization
│       ├── payload_optimizer.py  # ICL-guided genetic search
│       ├── payload_space.py      # Agent-aware payload templates & operators
│       └── scoring.py            # Eq.4: multi-objective R-DoS scoring
├── data/                         # Task data (see below)
└── requirements.txt
```

> Agent execution pipeline (`otora/agent/`, `otora/pipeline.py`, `otora/metrics.py`) and attack scripts (`scripts/`) are being finalized and will be released in a follow-up update.


## Data Setup

### WebShop
We use [WebShop](https://github.com/princeton-nlp/WebShop) for online shopping agent evaluation. Follow the original setup and place task data under `data/webshop/`.

### InjecAgent (Email & OS)
We use [InjecAgent](https://github.com/DongqiShen/InjecAgent) for Email and OS agent settings. Download the data and place it under `data/injecagent/`.

Preprocessed task splits used in our experiments will be released upon paper acceptance.


## Stage II Variants

The following variants from Table 3 in the paper are supported via the `variant` field in `StageIIConfig`:

| Variant | Config Value | Description |
|---------|-------------|-------------|
| **OTora-Persistent** | `persistent` | Full pipeline with persistent agent-aware payload (default) |
| **OTora-Agnostic** | `agnostic` | Context-agnostic fixed payload, no iterative optimization |
| **OTora-Aware** | `aware` | Context-aware fixed payload, no iterative optimization |
| **OTora-ICL(Agnostic)** | `icl_agnostic` | ICL-guided genetic search, context-agnostic |
| **OTora-ICL(Aware)** | `icl_aware` | ICL-guided genetic search, context-aware |

See `otora/config.py` for the complete list of configuration options.


## Citation

If you use OTora in your research, please cite our paper:

```bibtex
@inproceedings{li2025otora,
  title={OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents},
  author={Li, Xinyu and Mu, Ronghui and Li, Lin and Huang, Tianjin and Jin, Gaojie},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2026}
}
```


## Acknowledgments

- Stage I trigger optimization builds upon [UDora](https://github.com/AI-secure/UDora) and [nanoGCG](https://github.com/GraySwanAI/nanoGCG).
- Agent environments are adapted from [WebShop](https://github.com/princeton-nlp/WebShop) and [InjecAgent](https://github.com/DongqiShen/InjecAgent).


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
