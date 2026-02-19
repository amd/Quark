#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from datasets import load_dataset

from quark.contrib.llm_eval import ppl_eval
from quark.shares.utils.log import ScreenLogger
from quark.torch import LLMTemplate, ModelQuantizer, export_safetensors
from quark.torch.quantization.config.config import AWQConfig
from quark.torch.utils.llm import (
    get_calib_dataloader,
    get_model,
    get_tokenizer,
    prepare_for_moe_quant,
    revert_model_patching,
)

try:
    # Needed only when the model is loaded with accelerate offload (meta tensors).
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module  # type: ignore
    from accelerate.utils import PrefixedDataset  # type: ignore

    _ACCELERATE_AVAILABLE = True
except Exception:
    AlignDevicesHook = None  # type: ignore[assignment]
    add_hook_to_module = None  # type: ignore[assignment]
    PrefixedDataset = None  # type: ignore[assignment]
    _ACCELERATE_AVAILABLE = False

try:
    # Transformers>=5.0 provides GLM-4 MoE Lite implementation.
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (  # type: ignore[attr-defined]
        Glm4MoeLiteNaiveMoe,
    )
except Exception:
    Glm4MoeLiteNaiveMoe = None  # type: ignore[assignment]


DEFAULT_INPUT_MODEL_PATH = "zai-org/GLM-4.7-Flash"
DEFAULT_OUTPUT_MODEL_PATH = "amd/GLM-4.7-Flash-MXFP4"

logger = ScreenLogger(__name__)

# Presets aligned with:
#   Quark/examples/torch/language_modeling/llm_ptq/internal_scripts/glm.sh
PRESETS: dict[str, dict[str, Any]] = {
    # FP8 (weight per-channel static, act per-token dynamic / PTPC): attn + MoE, no kv-cache
    "fp8_ptpc_attn_moe_no_kvcache": {
        "quant_scheme": "ptpc_fp8",
        "exclude_layers": ["lm_head", "*mlp.gate*"],
    },
    # FP8 (weight+act per-tensor static): attn + MoE, no kv-cache
    "fp8_pertensor_attn_moe_no_kvcache": {
        "quant_scheme": "fp8",
        "exclude_layers": ["lm_head", "*mlp.gate*"],
    },
    # MXFP4 (weight static, act dynamic): MoE only, no kv-cache
    "mxfp4_moe_only_no_kvcache": {
        "quant_scheme": "mxfp4",
        "exclude_layers": ["lm_head", "*self_attn*", "*mlp.gate*"],
    },
    # INT4 weight-only: attn + MoE + lm_head, no kv-cache
    "int4_wo_128_attn_moe_lm_head_no_kvcache": {
        "quant_scheme": "int4_wo_128",
        "exclude_layers": ["*mlp.gate*"],
    },
    # INT4 weight-only + AWQ: attn + MoE + lm_head, no kv-cache
    # (This is not in glm.sh, but useful when you want INT4+AWQ explicitly.)
    "int4_wo_128_awq_attn_moe_lm_head_no_kvcache": {
        "quant_scheme": "int4_wo_128",
        "quant_algo": "awq",
        "exclude_layers": ["*mlp.gate*"],
    },
}


def _copy_non_weight_files(src_dir: str, dst_dir: str) -> None:
    """
    Copy non-weight files from an HF model directory (json/jinja/tokenizer, etc.),
    while skipping *.safetensors and model.safetensors.index.json.

    Note: `export_safetensors` exports the essential HF weights and config, but the
    original model directory may contain extra assets (e.g. chat_template.jinja).
    We do a conservative copy here so offline inference keeps those auxiliary files.
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for p in src.iterdir():
        if p.is_dir():
            continue
        name = p.name
        if name.endswith(".safetensors"):
            continue
        if name == "model.safetensors.index.json":
            continue
        # Export will (re-)write config / generation_config; copying them here is harmless
        # (later writes will overwrite).
        shutil.copy2(p, dst / name)


def _register_glm47_flash_template() -> None:
    """
    Register a Quark LLMTemplate for GLM-4.7-Flash (config.model_type = glm4_moe_lite).

    We exclude attention/router/shared experts/embedding/lm_head/norm and the layer-0 dense MLP,
    so that we effectively "quantize only MoE experts' MLP weights".
    """
    model_type = "glm4_moe_lite"
    if model_type in LLMTemplate.list_available():
        return

    def _build_glm4_moe_lite_awq_config(num_experts: int = 64) -> AWQConfig:
        """
        AWQ config for GLM-4 MoE Lite, inlined here to avoid modifying Quark.

        Parameter names are based on HF weights like:
          - model.layers.N.self_attn.q_a_proj.weight
          - model.layers.N.self_attn.kv_a_proj_with_mqa.weight
          - model.layers.N.mlp.experts.E.(gate_proj|up_proj|down_proj).weight
        """
        # MoE experts (routed experts). Shared experts are excluded in our template by default.
        scaling_layers = [
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ]

        return AWQConfig(scaling_layers=scaling_layers, model_decoder_layers="model.layers")

    glm47_flash_template = LLMTemplate(
        model_type=model_type,
        # GLM-4.7-Flash attention projections are not standard k_proj/v_proj names.
        # This is only for kv-cache related configuration (this script does NOT quantize attention/KV by default).
        kv_layers_name=["*kv_a_proj_with_mqa", "*kv_b_proj"],
        q_layer_name="*q_a_proj",
        exclude_layers_name=[
            # embeddings / lm head / norms
            "model.embed_tokens*",
            "*embed_tokens*",
            "*lm_head*",
            "*layernorm*",
            "*norm*",
            # attention blocks
            "*self_attn*",
            # router gate & shared experts (typically keep in BF16)
            "*mlp.gate*",
            "*mlp.shared_experts*",
            # layer-0 is a dense MLP (first_k_dense_replace=1)
            "model.layers.0.mlp.*",
        ],
        awq_config=_build_glm4_moe_lite_awq_config(),
    )
    LLMTemplate.register_template(glm47_flash_template)
    logger.info("Registered LLMTemplate: %s", model_type)


@torch.no_grad()
def replace_glm4moelite_experts_with_linear(experts_module: Any) -> None:
    """
    Convert fused experts in HF `Glm4MoeLiteNaiveMoe` into three separate Linear layers per expert:
    `gate_proj`, `up_proj`, and `down_proj`.

    This helper is adapted from Quark's internal `replacement_utils copy.py`, but inlined here
    because it is not present on the main branch.
    """
    if getattr(experts_module, "_glm4moelite_replaced", False):
        return

    logger.info("Converting Glm4MoeLiteNaiveMoe experts to separate gate/up/down Linear layers...")

    num_experts: int = int(experts_module.num_experts)
    hidden_size: int = getattr(experts_module, "hidden_size", experts_module.hidden_dim)
    expert_dim: int = getattr(experts_module, "expert_dim", experts_module.intermediate_dim)
    original_device = experts_module.gate_up_proj.device
    original_dtype = experts_module.gate_up_proj.dtype

    # Expose common attribute names used by the synced forward helper.
    experts_module.hidden_size = hidden_size
    experts_module.expert_dim = expert_dim

    is_meta: bool = getattr(experts_module.gate_up_proj, "is_meta", False) or original_device == torch.device("meta")
    target_device_for_new = original_device if not is_meta else torch.device("meta")

    for expert_index in range(num_experts):
        expert_module = nn.Module()
        expert_module.gate_proj = nn.Linear(
            hidden_size, expert_dim, bias=False, device=target_device_for_new, dtype=original_dtype
        )
        expert_module.up_proj = nn.Linear(
            hidden_size, expert_dim, bias=False, device=target_device_for_new, dtype=original_dtype
        )
        expert_module.down_proj = nn.Linear(
            expert_dim, hidden_size, bias=False, device=target_device_for_new, dtype=original_dtype
        )
        setattr(experts_module, str(expert_index), expert_module)

    weights_synced = _glm4moelite_sync_weights_to_linear(experts_module)
    experts_module.forward = MethodType(_glm4moelite_forward, experts_module)

    if weights_synced:
        _glm4moelite_cleanup_fused(experts_module)

    experts_module._glm4moelite_replaced = True


@torch.no_grad()
def _glm4moelite_sync_weights_to_linear(module: Any) -> bool:
    """
    Split fused weights and copy into per-expert Linear layers.
    Returns True if synced; returns False if fused weights are still on 'meta' (not materialized).

    Fused tensors in HF `Glm4MoeLiteNaiveMoe` are expected to be:
      - gate_up_proj: [num_experts, 2*expert_dim, hidden_size]  (out, in)
      - down_proj:    [num_experts, hidden_size, expert_dim]    (out, in)
    """
    if getattr(module, "_weights_synced", False):
        return True

    W_gate_up = getattr(module, "gate_up_proj", None)
    W_down = getattr(module, "down_proj", None)
    if W_gate_up is None or W_down is None:
        return False

    is_offload = getattr(W_gate_up, "is_meta", False) or W_gate_up.device == torch.device("meta")
    if is_offload:
        # Loaded with accelerate offload: tensors live in module._hf_hook.weights_map on CPU.
        if not _ACCELERATE_AVAILABLE:
            raise RuntimeError(
                "Model appears to be loaded with accelerate offload (meta tensors), but accelerate is not available."
            )
        if not hasattr(module, "_hf_hook"):
            return False
        W_gate_up = module._hf_hook.weights_map["gate_up_proj"]
        W_down = module._hf_hook.weights_map["down_proj"]

    try:
        for expert_index in range(int(module.num_experts)):
            expert_module = getattr(module, str(expert_index))

            W_gate_up_current = W_gate_up[expert_index]  # [2*expert_dim, hidden_size]
            W_gate_current = W_gate_up_current[: int(module.expert_dim), :]
            W_up_current = W_gate_up_current[int(module.expert_dim) :, :]
            W_down_current = W_down[expert_index]  # [hidden_size, expert_dim]

            if is_offload:
                hook = module._hf_hook
                dataset = hook.weights_map.dataset
                layer_value = [W_gate_current, W_up_current, W_down_current]
                for idx, layer_name in enumerate(["gate_proj", "up_proj", "down_proj"]):
                    prefix = f"{hook.weights_map.prefix}{expert_index}.{layer_name}."
                    prefixed_weights_map = PrefixedDataset(dataset, prefix)
                    full_name = f"{prefix}weight"
                    dataset.all_keys.append(full_name)
                    dataset.state_dict[full_name] = layer_value[idx]

                    quark_hook = AlignDevicesHook(
                        execution_device=hook.execution_device,
                        offload=hook.offload,
                        io_same_device=hook.io_same_device,
                        weights_map=prefixed_weights_map,
                        offload_buffers=hook.offload_buffers,
                        place_submodules=hook.place_submodules,
                        skip_keys=hook.skip_keys,
                        tied_params_map=hook.tied_params_map,
                    )
                    linear_module = getattr(expert_module, layer_name)
                    add_hook_to_module(linear_module, quark_hook)
            else:
                # No transpose needed: nn.Linear expects [out_features, in_features], which matches fused tensors.
                expert_module.gate_proj.weight.data.copy_(W_gate_current.to(module.gate_up_proj.device))
                expert_module.up_proj.weight.data.copy_(W_up_current.to(module.gate_up_proj.device))
                expert_module.down_proj.weight.data.copy_(W_down_current.to(module.down_proj.device))

        if is_offload:
            # Remove original fused tensors from the CPU weights map to avoid duplication.
            prefix = module._hf_hook.weights_map.prefix
            del module._hf_hook.weights_map.dataset.state_dict[f"{prefix}gate_up_proj"]
            del module._hf_hook.weights_map.dataset.state_dict[f"{prefix}down_proj"]
            module._hf_hook.weights_map.dataset.all_keys.remove(f"{prefix}gate_up_proj")
            module._hf_hook.weights_map.dataset.all_keys.remove(f"{prefix}down_proj")

        module._weights_synced = True
        return True
    except Exception as e:
        logger.warning("Failed to sync Glm4MoeLite weights: %s", e)
        return False


def _glm4moelite_forward(
    self: Any, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
) -> torch.Tensor:
    """
    Forward using per-expert `gate_proj`, `up_proj`, `down_proj` (nn.Linear),
    matching the original `Glm4MoeLiteNaiveMoe.forward` semantics but without `nn.functional.linear`.
    """
    synced = _glm4moelite_sync_weights_to_linear(self)
    if not synced:
        raise RuntimeError(
            "Glm4MoeLiteNaiveMoe weights are on 'meta' (not materialized). "
            "Move fused parameters to a real device first, then call forward."
        )

    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx_int = int(expert_idx[0].item())
        if expert_idx_int == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx_int])
        if token_idx.numel() == 0:
            continue

        current_state = hidden_states[token_idx]
        expert_module = getattr(self, str(expert_idx_int))

        gate = expert_module.gate_proj(current_state)
        up = expert_module.up_proj(current_state)
        current_hidden_states = self.act_fn(gate) * up
        current_hidden_states = expert_module.down_proj(current_hidden_states)

        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


@torch.no_grad()
def _glm4moelite_cleanup_fused(module: Any) -> None:
    """Optionally remove fused params from the module after the replacement."""
    for name in ["gate_up_proj", "down_proj"]:
        if hasattr(module, name):
            logger.debug("Removing %s attribute from %s", name, type(module))
            delattr(module, name)
            torch.cuda.empty_cache()


@torch.no_grad()
def patch_glm4moelite_moe(model: nn.Module) -> int:
    """
    Apply GLM-4 MoE Lite expert replacement to all `Glm4MoeLiteNaiveMoe` modules in the model.
    Returns the number of patched modules.
    """
    patched = 0
    for _, module in model.named_modules(remove_duplicate=False):
        is_target = False
        if (
            Glm4MoeLiteNaiveMoe is not None
            and isinstance(module, Glm4MoeLiteNaiveMoe)
            or module.__class__.__name__ == "Glm4MoeLiteNaiveMoe"
        ):
            is_target = True

        if is_target:
            replace_glm4moelite_experts_with_linear(module)
            patched += 1

    if patched > 0:
        logger.info("Patched %d Glm4MoeLiteNaiveMoe module(s) for quantization.", patched)
    return patched


def _resolve_calib_device(device: str, model: nn.Module) -> str:
    """
    Resolve a torch-compatible device string for calibration inputs.

    HF/accelerate accepts `device_map="auto"` (we expose it as --device auto), but `torch.Tensor.to("auto")`
    is invalid. For calibration inputs we pick a concrete device:
    - Prefer the lowest-index CUDA device present in `model.hf_device_map`
    - Otherwise fall back to `cuda:0` if CUDA is available, else `cpu`
    """
    if device != "auto":
        return str(device)

    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        cuda_ids: list[int] = []
        for v in hf_map.values():
            m = re.match(r"^cuda:(\d+)$", str(v))
            if m:
                cuda_ids.append(int(m.group(1)))
        if cuda_ids:
            return f"cuda:{min(cuda_ids)}"

    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_quantized_hf_path, exist_ok=True)

    _register_glm47_flash_template()

    # Apply preset defaults (if requested). User-provided flags always win.
    if getattr(args, "preset", None):
        preset_cfg = PRESETS[args.preset]
        args.quant_scheme = preset_cfg["quant_scheme"]
        if getattr(args, "quant_algo", None) is None and "quant_algo" in preset_cfg:
            args.quant_algo = preset_cfg["quant_algo"]
        if args.exclude_layers is None:
            args.exclude_layers = list(preset_cfg["exclude_layers"])
        logger.info("Using preset: %s", args.preset)

    logger.info("Input model: %s", args.input_model_path)
    logger.info("Output dir: %s", args.output_quantized_hf_path)

    logger.info("Step 1/4: Loading model and tokenizer ...")
    model, _ = get_model(
        args.input_model_path,
        data_type=args.data_type,
        device=args.device,
        multi_gpu=args.multi_gpu,
        multi_device=args.multi_device,
        attn_implementation=args.model_attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    prepare_for_moe_quant(model)
    # GLM-4.7-Flash (glm4_moe_lite) needs an extra MoE expert replacement pass before quantization.
    patch_glm4moelite_moe(model)
    model_type = model.config.model_type if hasattr(model.config, "model_type") else model.config.architectures[0]
    tokenizer = get_tokenizer(
        args.input_model_path, max_seq_len=args.seq_len, model_type=model_type, trust_remote_code=args.trust_remote_code
    )

    logger.info("Step 2/4: Building calibration dataloader ...")
    base_device = str(model.device) if (args.multi_gpu or args.multi_device) else str(args.device)
    main_device = _resolve_calib_device(base_device, model)
    # Same flow as Quark's quantize_quark.py. Note that some dataset names may trigger downloads
    # via `datasets.load_dataset` unless they are already cached locally.
    logger.info("Calibration dataset: %s (Quark get_calib_dataloader)", args.dataset)
    calib_dataloader = get_calib_dataloader(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_calib_data=args.num_calib_data,
        seqlen=args.seq_len,
        device=main_device,
    )

    logger.info("Step 3/4: Quantizing (following quantize_quark.py: LLMTemplate -> ModelQuantizer) ...")
    template = LLMTemplate.get(model_type)
    exclude_layers = args.exclude_layers
    if exclude_layers is not None:
        logger.info("Exclude layers (override): %s", exclude_layers)
    if getattr(args, "quant_algo", None):
        logger.info("Quantization algorithm(s): %s", args.quant_algo)
    quant_config = template.get_config(
        scheme=args.quant_scheme,
        algorithm=args.quant_algo,
        kv_cache_scheme=None,
        min_kv_scale=0.0,
        layer_config={},
        attention_scheme=None,
        exclude_layers=exclude_layers,
        algo_configs=None,
    )
    quantizer = ModelQuantizer(quant_config, args.multi_device)
    model = quantizer.quantize_model(model, calib_dataloader)
    model = quantizer.freeze(model)
    revert_model_patching(model)

    logger.info("Step 4/4: Exporting HF safetensors (following quantize_quark.py: export_safetensors) ...")
    _copy_non_weight_files(args.input_model_path, args.output_quantized_hf_path)
    with torch.no_grad():
        export_safetensors(
            model=model,
            output_dir=args.output_quantized_hf_path,
            custom_mode="quark",
            weight_format=args.export_weight_format,
            pack_method=args.pack_method,
        )
        tokenizer.save_pretrained(args.output_quantized_hf_path)

    if args.do_evaluation:
        logger.info("Evaluating PPL...")
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        if args.num_eval_data != -1:
            text = "\n\n".join(testdata["text"][: args.num_eval_data])
        else:
            text = "\n\n".join(testdata["text"])
        testenc = tokenizer(text, return_tensors="pt")
        ppl = ppl_eval(model, testenc, main_device)
        logger.info("Perplexity: %s", ppl.item())

    logger.info("Export completed.")
    logger.info("========== Quantization Completed Successfully ==========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline quantization for zai-org/GLM-4.7-Flash (load from a local HF directory), "
        "export hf_format safetensors following Quark's quantize_quark.py flow."
    )
    parser.add_argument("--input-model-path", dest="input_model_path", type=str, default=DEFAULT_INPUT_MODEL_PATH)
    parser.add_argument(
        "--output-quantized-hf-path", dest="output_quantized_hf_path", type=str, default=DEFAULT_OUTPUT_MODEL_PATH
    )

    # Model loading
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--multi-gpu", dest="multi_gpu", action="store_true")
    parser.add_argument("--multi-device", dest="multi_device", action="store_true")
    parser.add_argument(
        "--model-attn-implementation",
        dest="model_attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--data-type",
        dest="data_type",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=True)

    # Calibration
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        default="pileval",
        help="Calibration dataset name (same as quantize_quark.py). Default is 'pileval'.",
    )
    parser.add_argument("--seq-len", dest="seq_len", type=int, default=512)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1)
    parser.add_argument("--num-calib-data", dest="num_calib_data", type=int, default=128)

    # Quantization
    parser.add_argument(
        "--preset",
        dest="preset",
        type=str,
        choices=sorted(PRESETS.keys()),
        default="mxfp4_moe_only_no_kvcache",
        help="Convenience preset aligned with Quark's internal glm.sh recipes. "
        "This sets --quant-scheme and (unless you provide --exclude_layers) also sets exclude patterns.",
    )
    parser.add_argument("--quant-scheme", dest="quant_scheme", type=str, default="mxfp4")
    parser.add_argument(
        "--quant-algo",
        dest="quant_algo",
        type=str,
        default=None,
        help="Optional quantization algorithm(s) to apply. Example: --quant-algo awq. "
        "You may also pass a comma-separated list (e.g. awq,rotation) if supported by your Quark build.",
    )
    parser.add_argument(
        "--exclude_layers",
        type=str,
        nargs="*",
        default=None,
        help="Explicit layer wildcard patterns to exclude from quantization. "
        "If not provided, the selected --preset will provide defaults; otherwise the template defaults are used. "
        'Example: --exclude_layers "*self_attn*" "*mlp.gate*" "lm_head"',
    )

    # Export
    parser.add_argument("--pack-method", dest="pack_method", type=str, default="reorder", choices=["order", "reorder"])
    parser.add_argument(
        "--export-weight-format",
        dest="export_weight_format",
        type=str,
        default="real_quantized",
        choices=["fake_quantized", "real_quantized"],
    )

    # Evaluation (PPL)
    parser.add_argument("--do_evaluation", action="store_true")
    parser.add_argument(
        "--num_eval_data",
        help="Number of samples for PPL evaluation. Default is -1 (use all data).",
        type=int,
        default=-1,
    )

    main(parser.parse_args())
