"""
OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents

Two-stage pipeline:
  Stage I  – Trigger external tool access via adversarial string optimization
  Stage II – Optimize reasoning payloads to induce persistent multi-turn overhead
"""

__version__ = "0.1.0"

from .config import OToraConfig, StageIConfig, StageIIConfig
