#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Typed configuration objects for the speculative-decoding pipeline.

These dataclasses are the single source of truth for the Python API. YAML
recipes (see ``recipes/``) are loaded into the very same structures by
:mod:`quark.experimental.speculative_decoding.run`, so a recipe and a hand-built
config are interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EagleArchitectureConfig:
    """Architecture of the EAGLE-3 draft head.

    The draft is a small (usually single-layer) transformer that consumes the
    target model's hidden states (from ``aux_hidden_layers``) plus token
    embeddings, and predicts the target's next-token distribution.
    """

    num_hidden_layers: int = 1
    hidden_size: int | None = None  # defaults to the target hidden size
    intermediate_size: int | None = None  # defaults to the target intermediate size
    num_attention_heads: int | None = None  # defaults to the target
    num_key_value_heads: int | None = None  # defaults to the target
    # EAGLE-3 concatenates ``len(aux_hidden_layers)`` target hidden states and
    # reduces them with an ``fc`` layer; ``fc_norm``/``norm_output`` add RMSNorms
    # that materially stabilize cold-start training (EAGLE-3.1 behavior).
    fc_norm: bool = True
    norm_output: bool = True
    # Draft vocabulary. When smaller than the target vocab, a ``d2t``/``t2d``
    # mapping (see ``data/vocab.py``) compresses the lm_head for speed.
    draft_vocab_size: int | None = None  # None -> share the full target vocab
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int | None = None

    def merged_with_target(self, target_cfg: Any) -> EagleArchitectureConfig:
        """Fill unset fields from the target model's HF config."""
        out = EagleArchitectureConfig(**self.__dict__)
        out.hidden_size = self.hidden_size or getattr(target_cfg, "hidden_size", None)
        out.intermediate_size = self.intermediate_size or getattr(target_cfg, "intermediate_size", None)
        out.num_attention_heads = self.num_attention_heads or getattr(target_cfg, "num_attention_heads", None)
        out.num_key_value_heads = (
            self.num_key_value_heads or getattr(target_cfg, "num_key_value_heads", None) or out.num_attention_heads
        )
        out.max_position_embeddings = self.max_position_embeddings or getattr(
            target_cfg, "max_position_embeddings", 4096
        )
        return out


@dataclass
class SpecConfig:
    """Top-level speculative-decoding recipe passed to :func:`convert`."""

    method: str = "eagle3"
    eagle_architecture_config: dict[str, Any] = field(default_factory=dict)
    # Target layer indices whose hidden states feed the draft. EAGLE-3 typically
    # uses a low/mid/high triple, e.g. [2, mid, last-2].
    aux_hidden_layers: list[int] = field(default_factory=lambda: [2, -3, -1])
    # Test-time-training unroll depth (how many future tokens the draft is
    # trained to roll out autoregressively during training).
    ttt_length: int = 7

    def eagle_config(self) -> EagleArchitectureConfig:
        return EagleArchitectureConfig(**self.eagle_architecture_config)


@dataclass
class DataConfig:
    """Where the training/eval data live and how sequences are shaped."""

    train: str = ""
    eval: str = ""
    draft_vocab_cache: str | None = None  # path to d2t.pt from calibrate_draft_vocab
    max_seq_length: int = 4096
    chat_template: str | None = None  # MUST equal the serving template
    num_workers: int = 4


@dataclass
class TrainConfig:
    """Cold-start training hyper-parameters and orchestration knobs."""

    # Hidden-state extraction regime. All three produce identical training
    # semantics; they differ only in *how* target hidden states reach the trainer.
    #   online    - target co-located with the trainer (feasible for <=~8B targets)
    #   offline   - dump hidden states to disk, then train the draft alone
    #   streaming - a live vLLM serve streams hidden states to the trainer (big MoE)
    extraction: str = "online"

    cold_start: bool = True  # no warm-start from an existing draft
    learning_rate: float = 1e-4
    lr_schedule: str = "cosine"  # "cosine" | "constant" | "linear"
    warmup_ratio: float = 0.02
    num_epochs: int = 1
    max_steps: int | None = None  # overrides epoch-derived horizon
    micro_batch_size: int = 1
    grad_accum: int = 8
    # Topology of the run. The bundled trainer is single-GPU (no FSDP/DDP wrapper
    # is wired up yet), so it rejects num_gpus > 1 instead of silently using one
    # device; these fields exist to record the reference topology and for the
    # TorchSpec trainer in examples/.
    num_gpus: int = 1
    backend: str = "fsdp2"  # "fsdp2" | "ddp" | "single"

    output_dir: str = "ckpts/eagle3"
    save_interval: int = 1000
    serve_eval_interval: int = 0  # 0 disables in-loop serve-eval
    select_best_by: str = "serve_al"  # "serve_al" | "eval_loss"
    watchdog: bool = True  # auto-resume on a dead extraction engine

    # Loss shaping
    position_decay: float = 0.9  # per-position weight decay across the TTT unroll
    seed: int = 0
    bf16: bool = True
    log_interval: int = 10

    # Extraction/serving engine settings (used by streaming/offline + serve-eval).
    engine: str = "vllm"
    tp_size: int = 1
    target_endpoint: str | None = None  # for streaming / serve-eval
    served_model_name: str = "target"


@dataclass
class TorchSpecRunConfig:
    """Orchestration settings for the validated multi-GPU TorchSpec backend.

    The native :class:`TrainConfig` remains the configuration for ``qsd.train``.
    This object describes the reproducible end-to-end path used by the CLI:
    target-generated on-policy data, streaming 8-GPU training, conversion, and
    a matched baseline/speculative benchmark.
    """

    profile: str = "full"
    cache_dir: str = "~/.cache/amd-quark/eagle3"
    image: str = "quark-specdec-rocm:latest"
    prompt_dataset: str = "allenai/tulu-3-sft-mixture"
    num_prompts: int = 150_000
    eval_size: int = 256
    generation_max_tokens: int = 4096
    benchmark_prompts: int = 40
    benchmark_rounds: int = 3
    min_speedup: float = 1.25
    min_served_al: float = 2.35


@dataclass
class InferenceConfig:
    """ROCm/vLLM serving knobs (deployment + serve-eval)."""

    engine: str = "vllm"
    tp_size: int = 1
    moe_backend: str | None = None  # e.g. "aiter" for MXFP4 MoE
    attention_backend: str | None = None  # e.g. "TRITON_ATTN"
    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class QuantConfig:
    """Quark quantization hooks for target (verifier) and optional draft quant."""

    target: str | None = None  # e.g. "mxfp4" | "fp8" (already-quantized verifier)
    draft: str | None = None  # None | "fp8" | "mxfp4" (PTQ/QAT via quark.torch)
