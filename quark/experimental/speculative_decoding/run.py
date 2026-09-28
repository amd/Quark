#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Config-driven EAGLE-3 entry point: YAML recipe + CLI dotlist overrides.

Simplest form — pick a target with ``--base_model`` (defaults to the packaged
``recipes/eagle3_default.yaml``)::

    python3 -m quark.experimental.speculative_decoding.run --base_model Qwen/Qwen3-8B

which runs the validated full TorchSpec pipeline (on-policy data, 8-GPU
training, export, and measured speedup). It is shorthand for::

    python3 -m quark.experimental.speculative_decoding.run \
        --config recipes/eagle3_default.yaml \
        model.target_model_path=Qwen/Qwen3-8B training.output_dir=ckpts/qwen3-8b-eagle3

``--base_model`` is an alias for ``model.target_model_path=``; any explicit
dotlist override still wins. Set ``execution.backend=native`` to run the
single-GPU Quark reference trainer instead.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from quark.experimental.speculative_decoding.config import DataConfig, SpecConfig, TrainConfig
from quark.experimental.speculative_decoding.utils.logging import get_logger

logger = get_logger("qsd.run")

# Packaged recipe used when the caller does not pass --config.
_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "recipes", "eagle3_default.yaml")

# Every recipe key :func:`run_from_config` actually reads. A key with no consumer
# is silently inert, which lets a recipe look like it configures serving or
# evaluation when it does not, so we name them at load time instead.
_CONSUMED: dict[str, set[str]] = {
    "execution": {
        "backend",
        "profile",
        "cache_dir",
        "image",
        "runner_profile",
        "asset_profile",
        "target_tp_size",
        "world_size",
        "inference_num_gpus",
        "training_num_gpus",
        "target_max_model_len",
    },
    "model": {
        "target_model_path",
        "trust_remote_code",
        "embedding_key",
        "lm_head_key",
        "norm_key",
    },
    "eagle": {"eagle_architecture_config", "aux_hidden_layers", "ttt_length"},
    "data": {
        "auto_generate",
        "prompt_dataset",
        "num_prompts",
        "eval_size",
        "generation_max_tokens",
        "quick_num_prompts",
        "quick_eval_size",
        "train_data_path",
        "train",
        "eval_data_path",
        "eval",
        "draft_vocab_cache",
        "max_seq_length",
        "chat_template",
        "domain_manifest",
    },
    "training": {
        "extraction",
        "cold_start",
        "learning_rate",
        "lr_schedule",
        "warmup_ratio",
        "num_epochs",
        "quick_num_epochs",
        "max_steps",
        "micro_batch_size",
        "draft_accumulation_steps",
        "grad_accum",
        "num_gpus_per_node",
        "num_gpus",
        "train_backend",
        "backend",
        "output_dir",
        "save_interval",
        "serve_eval_interval",
        "select_best_by",
        "watchdog",
        "target_endpoint",
        "world_size",
        "training_num_gpus",
        "training_num_gpus_per_node",
    },
    "quant": {"target"},
    "inference": {
        "target_endpoint",
        "target_tp_size",
        "tp_size",
        "inference_num_gpus",
        "max_model_len",
        "env",
    },
    "benchmark": {
        "enabled",
        "num_prompts",
        "rounds",
        "quick_num_prompts",
        "quick_rounds",
        "min_speedup",
        "min_served_al",
        "num_speculative_tokens",
    },
}


def _warn_unconsumed(cfg: dict[str, Any]) -> None:
    inert = [k for k in cfg if k != "method" and k not in _CONSUMED]
    for section, known in _CONSUMED.items():
        block = cfg.get(section)
        if isinstance(block, dict):
            inert += [f"{section}.{k}" for k in block if k not in known]
    if inert:
        logger.warning("recipe keys with no consumer, ignored: %s", ", ".join(sorted(inert)))


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` dotlist overrides (OmegaConf-style)."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"bad override (expected key=value): {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(value)
    return cfg


def _resolve_config_path(path: str) -> str:
    requested = Path(path).expanduser()
    if requested.is_file():
        return str(requested)
    # ``--config recipes/foo.yaml`` is a public, CWD-independent alias for an
    # installed package recipe. Do not silently reinterpret arbitrary missing
    # paths: only the recipes namespace gets this fallback.
    if requested.parent == Path("recipes"):
        packaged = Path(__file__).resolve().parent / "recipes" / requested.name
        if packaged.is_file():
            return str(packaged)
    raise FileNotFoundError(f"recipe not found: {path!r}. Packaged recipes may be addressed as 'recipes/<name>.yaml'.")


def load_config(path: str, overrides: list[str]) -> dict[str, Any]:
    resolved = _resolve_config_path(path)
    with open(resolved, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _apply_overrides(cfg, overrides)


def _run_native_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute the single-GPU Quark reference trainer."""
    # Keep heavyweight torch/transformers imports off the validated Docker
    # orchestration path. This also makes CLI config/help usable in a lightweight
    # host environment while all GPU dependencies stay in the prepared image.
    from quark.experimental.speculative_decoding.convert import convert
    from quark.experimental.speculative_decoding.export.convert_to_vllm import convert_to_vllm
    from quark.experimental.speculative_decoding.export.export_hf import export_hf
    from quark.experimental.speculative_decoding.quant.integrate import load_target_verifier
    from quark.experimental.speculative_decoding.training.trainer import train
    from quark.experimental.speculative_decoding.utils.watchdog import run_with_watchdog

    model_cfg = cfg.get("model", {})
    target_path = model_cfg["target_model_path"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    embedding_key = model_cfg.get("embedding_key")
    quant_cfg = cfg.get("quant", {})

    spec_cfg = SpecConfig(
        method=cfg.get("method", "eagle3"),
        eagle_architecture_config=cfg.get("eagle", {}).get("eagle_architecture_config", {}),
        aux_hidden_layers=cfg.get("eagle", {}).get("aux_hidden_layers", [2, -3, -1]),
        ttt_length=cfg.get("eagle", {}).get("ttt_length", 7),
    )

    data = cfg.get("data", {})
    data_cfg = DataConfig(
        train=data.get("train_data_path", data.get("train", "")),
        eval=data.get("eval_data_path", data.get("eval", "")),
        draft_vocab_cache=data.get("draft_vocab_cache"),
        max_seq_length=data.get("max_seq_length", 4096),
        chat_template=data.get("chat_template"),
    )

    tr = cfg.get("training", {})
    train_cfg = TrainConfig(
        extraction=tr.get("extraction", "online"),
        cold_start=tr.get("cold_start", True),
        learning_rate=float(tr.get("learning_rate", 1e-4)),
        lr_schedule=tr.get("lr_schedule", "cosine"),
        warmup_ratio=float(tr.get("warmup_ratio", 0.02)),
        num_epochs=int(tr.get("num_epochs", 1)),
        max_steps=tr.get("max_steps"),
        micro_batch_size=int(tr.get("micro_batch_size", 1)),
        grad_accum=int(tr.get("draft_accumulation_steps", tr.get("grad_accum", 8))),
        num_gpus=int(tr.get("num_gpus_per_node", tr.get("num_gpus", 1))),
        backend=tr.get("train_backend", tr.get("backend", "fsdp2")),
        output_dir=tr.get("output_dir", "ckpts/eagle3"),
        save_interval=int(tr.get("save_interval", 1000)),
        serve_eval_interval=int(tr.get("serve_eval_interval", 0)),
        select_best_by=tr.get("select_best_by", "serve_al"),
        watchdog=bool(tr.get("watchdog", True)),
        target_endpoint=tr.get("target_endpoint") or cfg.get("inference", {}).get("target_endpoint"),
    )

    logger.info("loading target verifier: %s (quant=%s)", target_path, quant_cfg.get("target"))
    target = load_target_verifier(target_path, quant=quant_cfg.get("target"), trust_remote_code=trust_remote_code)

    spec_model = convert(target, spec_cfg, trust_remote_code=trust_remote_code, embedding_key=embedding_key)
    spec_model.target_model_path = target_path  # so the trainer can load the tokenizer

    # The trainer resumes from the newest checkpoint on entry when watchdog=True, so
    # relaunching it is what turns a dead extraction engine into a resumed run rather
    # than a lost one. Without this wrapper the flag only ever resumes a manual restart.
    if train_cfg.watchdog:
        result = run_with_watchdog(lambda: train(spec_model, data_cfg, train_cfg))
    else:
        result = train(spec_model, data_cfg, train_cfg)

    out_dir = os.path.join(train_cfg.output_dir, "release")
    draft_hf = export_hf(spec_model, os.path.join(out_dir, "draft_hf"), best_ckpt=result.get("best_ckpt"))
    convert_to_vllm(draft_hf, verifier=target_path, out=os.path.join(out_dir, "draft_vllm"))
    result["draft_hf"] = draft_hf
    result["draft_vllm"] = os.path.join(out_dir, "draft_vllm")
    logger.info("pipeline complete: %s", result)
    return result


def run_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a recipe to the validated or native execution backend."""
    _warn_unconsumed(cfg)
    backend = str(cfg.get("execution", {}).get("backend", "native")).lower()
    if backend == "torchspec":
        from quark.experimental.speculative_decoding.torchspec_runner import run_torchspec_from_config

        return run_torchspec_from_config(cfg)
    if backend == "native":
        return _run_native_from_config(cfg)
    raise ValueError(f"unknown execution.backend={backend!r}; expected 'torchspec' or 'native'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quark speculative decoding (EAGLE-3) pipeline")
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help="path to a YAML recipe (default: packaged recipes/eagle3_default.yaml)",
    )
    parser.add_argument("--base_model", help="target model repo id or local dir; alias for model.target_model_path=")
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. training.num_epochs=1")
    args = parser.parse_args()

    # --base_model is a friendly shortcut applied first; explicit dotlists still win.
    overrides = list(args.overrides)
    if args.base_model:
        overrides = [f"model.target_model_path={args.base_model}"] + overrides
    cfg = load_config(args.config, overrides)
    run_from_config(cfg)


if __name__ == "__main__":
    main()
