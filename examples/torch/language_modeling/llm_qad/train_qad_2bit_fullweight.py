#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Full-weight QAD (no LoRA): student weights stay in BF16/FP16 but every forward
uses GPTQ-style grouped 2-bit fake-quant on linear weights (STE). Teacher is
frozen BF16; loss = KD (KL/JSD) + CE like QADTrainer.

Memory: loads *both* student and teacher — plan for ~2× model weights in VRAM
plus optimizer states on all student parameters. Use multi-GPU (device_map),
optional gradient checkpointing (off by default for speed on large GPUs), and/or
DeepSpeed (pass a json via HF TrainingArguments).

Speed: pass ``--attn_implementation sdpa`` (default) for faster attention; use
``--gradient_checkpointing`` only if you hit OOM.

Dataset: uses TaskMixDataset / WikiText from
``quark.torch.algorithm.qad_trainer.datasets`` (task_include, SlimPajama caps,
teacher reasoning traces, etc.).
"""

from __future__ import annotations

import argparse
import inspect
import math
import os

import torch
from qad_weight_fakequant import (
    Linear2BitGroupSTE,
    bake_experts_2bit_ste,
    replace_gptoss_experts_with_2bit_ste,
    replace_linears_with_group_2bit_ste,
    strip_group_2bit_ste_to_linear,
)
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from quark.common.utils.log import ScreenLogger
from quark.torch.algorithm.blockwise_tuning.blockwise_utils import block_forward
from quark.torch.algorithm.qad_trainer import QADTrainer
from quark.torch.algorithm.utils.prepare import (
    get_model_layers,
    init_blockwise_algo,
    init_device_map,
)
from quark.torch.algorithm.utils.utils import clear_memory
from quark.torch.utils.llm.model_preparation import get_model

logger = ScreenLogger(__name__)


class _PPLEvalCallback(TrainerCallback):
    """Evaluate WikiText PPL every ``eval_every`` optimizer steps."""

    def __init__(self, eval_dataset, collate_fn, eval_every: int, batch_size: int = 4) -> None:
        self.eval_dataset = eval_dataset
        self.collate_fn = collate_fn
        self.eval_every = eval_every
        self.batch_size = batch_size

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if self.eval_every <= 0 or state.global_step % self.eval_every != 0:
            return
        if model is None:
            return
        loader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=False,
        )
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(model.device)
                labels = batch["labels"].to(model.device)
                loss = model(input_ids=input_ids, labels=labels).loss
                losses.append(loss.item())
        ppl = math.exp(sum(losses) / max(1, len(losses)))
        logger.info(f"[QAD-2bit] step {state.global_step} eval PPL: {ppl:.4f}")
        model.train()


class _RoundingAnnealCallback(TrainerCallback):
    """Anneal rounding temperature from temp_start → temp_end over anneal_steps.

    Uses cosine schedule: temp = end + 0.5*(start-end)*(1 + cos(pi * t))
    where t = min(step / anneal_steps, 1). This gives a smooth transition from
    soft to hard rounding that stabilizes training.
    """

    def __init__(
        self, model: torch.nn.Module, anneal_steps: int, temp_start: float = 1.0, temp_end: float = 0.01
    ) -> None:
        self.model = model
        self.anneal_steps = max(anneal_steps, 1)
        self.temp_start = temp_start
        self.temp_end = temp_end

    def _set_temp(self, temp: float) -> None:
        for m in self.model.modules():
            if isinstance(m, Linear2BitGroupSTE) and m.learned_rounding:
                m.set_rounding_temperature(temp)

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        t = min(state.global_step / self.anneal_steps, 1.0)
        temp = self.temp_end + 0.5 * (self.temp_start - self.temp_end) * (1.0 + math.cos(math.pi * t))
        self._set_temp(temp)
        if state.global_step % max(args.logging_steps, 1) == 0:
            logger.info(f"[QAD-2bit] step {state.global_step} rounding_temperature={temp:.4f}")


def _block_ap_train_one_block(
    layer: torch.nn.Module,
    module_kwargs: dict,
    layer_inputs: list[torch.Tensor],
    fp_layer_outputs: list[torch.Tensor],
    device: torch.device,
    epochs: int,
    weight_lr: float,
    quant_lr: float,
    layer_index: int,
) -> None:
    """Train a single block to minimize MSE vs teacher output.

    Supports separate learning rates for model weights vs quantization params
    (log_scale, rounding_logit).
    """
    import time

    from quark.torch.algorithm.blockwise_tuning.blockwise_utils import (
        block_batch_forward,
        blockwise_eval,
    )
    from quark.torch.algorithm.utils.utils import TensorData

    criterion = torch.nn.MSELoss()
    num_update_steps_per_epoch = max(len(layer_inputs), 1)
    max_steps = int(epochs * num_update_steps_per_epoch)

    tensordata = TensorData(layer_inputs, fp_layer_outputs, device)
    tensordata_loader = DataLoader(tensordata, batch_size=None, shuffle=True)

    before_loss = blockwise_eval(layer, module_kwargs, tensordata_loader, criterion, device)

    weight_params, quant_params = [], []
    quant_keywords = {"log_scale", "rounding_logit"}
    for n, p in layer.named_parameters():
        p.requires_grad = True
        if any(kw in n for kw in quant_keywords):
            quant_params.append(p)
        else:
            weight_params.append(p)

    logger.info(
        f"[Block-AP] Block {layer_index}: {len(weight_params)} weight params (lr={weight_lr:.2e}), "
        f"{len(quant_params)} quant params (lr={quant_lr:.2e})"
    )

    param_groups = [{"params": weight_params, "lr": weight_lr, "weight_decay": 0.0}]
    if quant_params:
        param_groups.append({"params": quant_params, "lr": quant_lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps,
        eta_min=weight_lr / 10.0,
    )

    for epoch in range(epochs):
        start_time = time.time()
        layer.train()
        for inp, fp_out in tensordata_loader:
            outputs = block_batch_forward(layer, module_kwargs, inp, device)
            loss = criterion(outputs, fp_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(weight_params + quant_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            layer.zero_grad()

        after_loss = blockwise_eval(layer, module_kwargs, tensordata_loader, criterion, device)
        lr_now = scheduler.get_lr()[0]
        elapsed = time.time() - start_time
        logger.info(
            f"[Block-AP] Block {layer_index}, Epoch {epoch}: "
            f"MSE {before_loss:.8f} -> {after_loss:.8f}, "
            f"LR {lr_now:.6f}, "
            f"MaxMem {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB, "
            f"Time {elapsed:.1f}s"
        )

    clear_memory()


def run_block_ap_warmup(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    tokenizer,
    args: argparse.Namespace,
) -> None:
    """EfficientQAT-style Block-AP warmup: block-wise MSE reconstruction.

    For each transformer block, train all parameters (weights + quantization
    params) to minimize the MSE between the FP16 teacher block output and the
    quantized student block output.  This provides a much better initialization
    for the subsequent E2E KD phase.

    Uses Quark's existing infrastructure for block input capture and forward
    passes, with a custom training loop supporting per-group learning rates.
    """
    from datasets import load_dataset as hf_load_dataset
    from tqdm import tqdm

    logger.info(
        f"[Block-AP] Starting block-wise warmup: "
        f"epochs={args.block_ap_epochs}, weight_lr={args.block_ap_weight_lr}, "
        f"quant_lr={args.block_ap_quant_lr}, samples={args.block_ap_samples}, "
        f"seq_len={args.block_ap_seq_len}"
    )

    # -- Collect calibration data as a simple DataLoader of input_ids tensors --
    logger.info("[Block-AP] Loading calibration data from SlimPajama...")
    raw_ds = hf_load_dataset("DKYoon/SlimPajama-6B", split="train", streaming=True)
    cal_ids: list[torch.Tensor] = []
    for ex in raw_ds:
        toks = tokenizer(
            ex["text"],
            truncation=True,
            max_length=args.block_ap_seq_len,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        if toks.numel() >= args.block_ap_seq_len:
            cal_ids.append(toks[: args.block_ap_seq_len])
        if len(cal_ids) >= args.block_ap_samples:
            break
    logger.info(f"[Block-AP] Collected {len(cal_ids)} calibration samples (seq_len={args.block_ap_seq_len})")

    cal_loader = DataLoader(
        cal_ids,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: torch.stack(batch),
    )

    # -- Capture block inputs by running calibration data through the student --
    student.eval()
    decoder_layers_path = args.block_ap_decoder_layers
    logger.info(f"[Block-AP] Capturing block inputs via {decoder_layers_path}...")
    modules, module_kwargs, inps = init_blockwise_algo(student, decoder_layers_path, cal_loader)
    modules_fp = get_model_layers(teacher, decoder_layers_path)
    device_map = init_device_map(student)

    num_blocks = len(modules)
    num_batches = len(inps)
    logger.info(f"[Block-AP] {num_blocks} blocks, {num_batches} batches captured")

    # -- Block-wise loop (same structure as BlockwiseTuningProcessor.apply) --
    cache_on_gpu = False
    layer_inputs = [inp.detach().requires_grad_(False) for inp in inps]
    layer_outputs: list[torch.Tensor] = []
    fp_layer_inputs = list(layer_inputs)
    fp_layer_outputs: list[torch.Tensor] = []

    fwd_use_cache = student.config.use_cache
    student.config.use_cache = False
    teacher.config.use_cache = False

    from quark.torch.algorithm.utils.module import get_device, move_to_device

    for i in range(num_blocks):
        modules[i] = modules[i].to("cpu")
    clear_memory()

    for i in tqdm(range(num_blocks), desc="Block-AP"):
        logger.info(f"[Block-AP] Block {i + 1}/{num_blocks}")
        layer = modules[i]
        layer_fp = modules_fp[i]

        force_cpu = False
        if get_device(layer) == torch.device("cpu"):
            move_to_device(layer, device_map.get(f"{decoder_layers_path}.{i}", torch.device("cuda:0")))
            force_cpu = True
        cur_device = get_device(layer)

        # FP16 teacher block forward (no grad)
        fp_layer_outputs = block_forward(
            layer_fp,
            module_kwargs,
            num_batches,
            cur_device,
            fp_layer_inputs,
            fp_layer_outputs,
            cache_on_gpu,
        )
        layer_fp = move_to_device(layer_fp, torch.device("cpu") if force_cpu else cur_device)

        # Train quantized student block to match teacher block output
        _block_ap_train_one_block(
            layer,
            module_kwargs,
            layer_inputs,
            fp_layer_outputs,
            cur_device,
            args.block_ap_epochs,
            args.block_ap_weight_lr,
            args.block_ap_quant_lr,
            layer_index=i,
        )

        # Student block forward to get outputs for next block
        layer_outputs = block_forward(
            layer,
            module_kwargs,
            num_batches,
            cur_device,
            layer_inputs,
            layer_outputs,
            cache_on_gpu,
        )

        layer = move_to_device(layer, torch.device("cpu") if force_cpu else cur_device)

        del layer, layer_fp, layer_inputs, fp_layer_inputs
        # These feed the next block iteration (read at the top of the loop).
        layer_inputs, layer_outputs = layer_outputs, []  # noqa: F841
        fp_layer_inputs, fp_layer_outputs = fp_layer_outputs, []  # noqa: F841
        clear_memory()

    student.config.use_cache = fwd_use_cache
    teacher.config.use_cache = fwd_use_cache
    student.train()
    logger.info("[Block-AP] Block-wise warmup complete")


# Datasets (general-LM + task-mix) live in the quark qad_trainer library.
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def _load_student_ignore_quant(ckpt_path: str, args: argparse.Namespace) -> torch.nn.Module:
    """Load a fake-quantized 2-bit student as plain bf16, ignoring any (informational)
    quark ``quantization_config`` in its ``config.json``.

    The 2-bit error is already baked into the bf16 weights, so there is nothing to
    re-quantize; loading through the HF quark quantizer would try to rebuild the
    quantizer and fail for some export schemes (e.g. bfp16: "Serialization of bfp16
    models is not yet supported in Quark"). This keeps the QAD recipe loadable on any
    quark version.
    """
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.bfloat16)
    cfg = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True, attn_implementation=args.attn_implementation)
    if hasattr(cfg, "quantization_config"):
        del cfg.quantization_config
    return AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        config=cfg,
        torch_dtype=dtype,
        device_map=("auto" if args.multi_gpu else args.device),
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )


from quark.torch.algorithm.qad_trainer.datasets import (  # noqa: E402
    PileLMDataset,
    TaskMixDataset,
    WikiTextLMDataset,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-weight 2-bit-fake-quant QAD (no LoRA)")
    p.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="HF checkpoint for the student (and teacher if --teacher_model_dir is not set)",
    )
    p.add_argument(
        "--teacher_model_dir",
        type=str,
        default=None,
        help="Separate HF checkpoint for the BF16 teacher. Defaults to --model_dir.",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--group_size", type=int, default=64, help="GPTQ-style group size along in_features")
    p.add_argument(
        "--linear_include_substrings",
        type=str,
        default="proj",
        help="Comma-separated substrings; linear must match one to be wrapped (default: proj). "
        "Use empty string to wrap all non-excluded Linears.",
    )
    p.add_argument(
        "--exclude_globs",
        type=str,
        default="*embed_tokens*,*lm_head*",
        help="Comma-separated fnmatch globs for module names to skip",
    )
    p.add_argument(
        "--quantize_moe_experts",
        action="store_true",
        default=False,
        help="Also fake-quant gpt-oss MoE expert weights (gate_up_proj/down_proj, "
        "packed 3D nn.Parameter tensors that the Linear replacement cannot reach). "
        "Required for meaningful QAD on gpt-oss (experts are ~90%% of params). "
        "No-op on dense models. Omit this flag to keep the dense Linear-only behavior.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--multi_gpu", action="store_true", help="device_map=auto for student and teacher")
    p.add_argument("--dataset", type=str, default="taskmix", choices=["taskmix", "pile", "wikitext"])
    p.add_argument("--task_include", type=str, default="all")
    p.add_argument("--use_slimpajama", action="store_true")
    p.add_argument(
        "--max_raw_slimpajama",
        type=int,
        default=100000,
        help="More SlimPajama rows (v14 LoRA used 50000; QAD often benefits from more data).",
    )
    p.add_argument("--pile_ratio", type=float, default=0.15)
    p.add_argument("--max_raw_pile", type=int, default=30000)
    p.add_argument("--max_per_task", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--max_train_samples", type=int, default=120000)
    p.add_argument("--max_eval_samples", type=int, default=256)
    p.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Micro-batch size (raise for throughput if VRAM allows; OOM → 1).",
    )
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Default 4 with batch 2 keeps ~8k tokens/optimizer step (2×4×1024).",
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=8000,
        help="Match LoRA+KD scale unless you intentionally extend (Phi-4 LoRA uses 8000; QwQ uses 5000).",
    )
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument(
        "--lr", type=float, default=5e-6, help="Lower LR than LoRA; full weights + STE are easy to destabilize."
    )
    p.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="HF Trainer log interval; also controls QAD loss breakdown print frequency.",
    )
    p.add_argument(
        "--log_loss_breakdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print total / logit_kd / kd_total / ce every ~logging_steps optimizer steps (default: on).",
    )
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--kd_temperature", type=float, default=1.2)
    p.add_argument("--kd_alpha", type=float, default=0.85)
    p.add_argument("--kd_loss_type", type=str, default="jsd", choices=["kl", "jsd"])
    p.add_argument("--kd_mode", type=str, default="output", choices=["output", "layer", "attention"])
    p.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bf16 (default: on; pass --no-bf16 to disable)",
    )
    p.add_argument("--fp16", action="store_true")
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save VRAM (slower). Default off for throughput on H100/H200-class GPUs.",
    )
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="sdpa: fast PyTorch attention (default). flash_attention_2 needs flash-attn. "
        "eager: slowest, most compatible.",
    )
    p.add_argument(
        "--eval_every",
        type=int,
        default=500,
        help="Evaluate WikiText PPL every N optimizer steps (0=disabled). Matches LoRA+KD v14 default.",
    )
    p.add_argument("--dataloader_num_workers", type=int, default=4, help=">0 prefetches batches to GPU.")
    p.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed json config")
    p.add_argument(
        "--learned_scales",
        action="store_true",
        default=False,
        help="Use trainable per-group scales (LSQ-style) instead of max-abs RTN. "
        "Omit this flag to keep v1/v2 behavior (RTN scales). "
        "Automatically enabled when --learned_rounding is set.",
    )
    p.add_argument(
        "--scale_lr_mult",
        type=float,
        default=10.0,
        help="Learning-rate multiplier for log_scale params relative to --lr (default: 10).",
    )
    # --- Learned rounding (FlexRound / TesseraQ-style) ---
    p.add_argument(
        "--learned_rounding",
        action="store_true",
        default=False,
        help="Learn per-element rounding decisions (FlexRound-style). Each weight "
        "gets a sigmoid-parameterised logit that controls whether to round up or "
        "down to the nearest of {-1,-1/3,1/3,1}. Combined with learned scales "
        "for joint optimisation. At save time, decisions are hardened to produce "
        "exact 4-level weights for NPU deployment. "
        "Omit this flag to keep v1/v2 RTN behavior.",
    )
    p.add_argument(
        "--rounding_lr_mult",
        type=float,
        default=100.0,
        help="Learning-rate multiplier for rounding_logit params relative to --lr "
        "(default: 100 → 5e-4 when lr=5e-6). Rounding logits are auxiliary "
        "parameters that converge faster than pretrained weights.",
    )
    p.add_argument(
        "--rounding_anneal_steps",
        type=int,
        default=1500,
        help="Anneal rounding temperature from 1.0 to 0.01 over this many steps "
        "(default: 1500). Set to 0 to disable annealing (hard rounding from start).",
    )
    p.add_argument(
        "--rounding_temp_start",
        type=float,
        default=1.0,
        help="Starting rounding temperature (default: 1.0 = soft rounding).",
    )
    p.add_argument(
        "--rounding_temp_end",
        type=float,
        default=0.01,
        help="Final rounding temperature (default: 0.01 ≈ hard rounding).",
    )
    # --- Block-AP warmup (EfficientQAT-style) ---
    p.add_argument(
        "--block_ap_warmup",
        action="store_true",
        default=False,
        help="Run EfficientQAT-style block-wise reconstruction warmup before E2E KD. "
        "Each transformer block is trained independently to minimize MSE between "
        "the FP16 teacher block output and quantized student block output. This "
        "provides a much better initialization for the E2E KD phase. "
        "Omit this flag to keep v1/v2/v3/v4 behavior.",
    )
    p.add_argument(
        "--block_ap_epochs",
        type=int,
        default=2,
        help="Epochs per block during Block-AP warmup (default: 2, per EfficientQAT).",
    )
    p.add_argument(
        "--block_ap_weight_lr",
        type=float,
        default=2e-5,
        help="Learning rate for model weights during Block-AP (default: 2e-5, per EfficientQAT 2-bit).",
    )
    p.add_argument(
        "--block_ap_quant_lr",
        type=float,
        default=1e-4,
        help="Learning rate for quantization params (log_scale, rounding_logit) during Block-AP.",
    )
    p.add_argument(
        "--block_ap_samples",
        type=int,
        default=4096,
        help="Number of calibration samples for Block-AP (default: 4096, per EfficientQAT).",
    )
    p.add_argument(
        "--block_ap_seq_len",
        type=int,
        default=2048,
        help="Context length for Block-AP calibration data (default: 2048).",
    )
    p.add_argument(
        "--block_ap_decoder_layers",
        type=str,
        default="model.layers",
        help="Dot-path to the decoder layer ModuleList (default: model.layers for Phi-4).",
    )
    p.add_argument(
        "--block_ap_resume",
        type=str,
        default=None,
        help="Path to a saved Block-AP checkpoint to resume from. Skips the Block-AP "
        "warmup phase and loads the student directly from this checkpoint. "
        "Useful when Block-AP completed but E2E training failed.",
    )
    # --- On-policy KD ---
    p.add_argument(
        "--on_policy_kd",
        action="store_true",
        default=False,
        help="Enable on-policy KD: student generates sequences, teacher supervises. "
        "Addresses exposure bias by training on student's own distribution.",
    )
    p.add_argument(
        "--on_policy_every", type=int, default=5, help="Run on-policy KD every N optimizer steps (default: 5)."
    )
    p.add_argument(
        "--on_policy_max_gen_len",
        type=int,
        default=256,
        help="Max new tokens to generate during on-policy step (default: 256).",
    )
    p.add_argument(
        "--on_policy_prompt_len",
        type=int,
        default=64,
        help="Prompt prefix length for on-policy generation (default: 64).",
    )
    p.add_argument(
        "--on_policy_temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for on-policy generation (default: 0.7).",
    )
    p.add_argument(
        "--on_policy_alpha",
        type=float,
        default=0.3,
        help="Weight of on-policy loss added to regular loss (default: 0.3). "
        "Linearly ramped from 0 over first 30%% of training.",
    )
    return p.parse_args()


def _collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("[QAD-2bit] TF32 matmul/cudnn enabled; cudnn.benchmark=True")

    if args.fp16:
        args.bf16 = False

    include_subs: tuple[str, ...] | None
    if args.linear_include_substrings.strip() == "":
        include_subs = None
    else:
        include_subs = tuple(s.strip() for s in args.linear_include_substrings.split(",") if s.strip())

    exclude_globs = tuple(p.strip() for p in args.exclude_globs.split(",") if p.strip())

    teacher_dir = args.teacher_model_dir or args.model_dir

    logger.info(f"[QAD-2bit] Loading TEACHER from {teacher_dir}")
    teacher, _ = get_model(
        ckpt_path=teacher_dir,
        data_type="bfloat16" if args.bf16 else ("float16" if args.fp16 else "bfloat16"),
        device=args.device,
        multi_gpu=args.multi_gpu,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    logger.info(f"[QAD-2bit] Loading STUDENT from {args.model_dir}")
    student = _load_student_ignore_quant(args.model_dir, args)

    use_learned_rounding = args.learned_rounding
    use_learned_scales = args.learned_scales or use_learned_rounding

    replaced = replace_linears_with_group_2bit_ste(
        student,
        group_size=args.group_size,
        exclude_globs=exclude_globs,
        include_substrings=include_subs,
        learned_scales=use_learned_scales,
        learned_rounding=use_learned_rounding,
    )
    if use_learned_rounding:
        mode_str = "learned-rounding+learned-scale"
    elif use_learned_scales:
        mode_str = "learned-scale"
    else:
        mode_str = "RTN"
    logger.info(
        f"[QAD-2bit] Replaced {len(replaced)} Linear layers with group-2bit STE (g={args.group_size}, {mode_str})"
    )
    for n in replaced[:8]:
        logger.info(f"  {n}")
    if len(replaced) > 8:
        logger.info("  ...")
    if args.quantize_moe_experts:
        exp_replaced = replace_gptoss_experts_with_2bit_ste(
            student,
            group_size=args.group_size,
            learned_scales=use_learned_scales,
        )
        logger.info(
            f"[QAD-2bit] Replaced {len(exp_replaced)} MoE expert blocks with group-2bit STE (g={args.group_size})"
        )
        if not exp_replaced:
            logger.warning("[QAD-2bit] --quantize_moe_experts set but no gpt-oss experts found (dense model?)")

    if args.gradient_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        logger.info("[QAD-2bit] Gradient checkpointing enabled on student")

    tokenizer = AutoTokenizer.from_pretrained(teacher_dir, trust_remote_code=True)

    # --- Optional Block-AP warmup (EfficientQAT-style) ---
    if args.block_ap_resume:
        logger.info(f"[QAD-2bit] Resuming from Block-AP checkpoint: {args.block_ap_resume}")
        del student
        clear_memory()
        student = _load_student_ignore_quant(args.block_ap_resume, args)
        replaced = replace_linears_with_group_2bit_ste(
            student,
            group_size=args.group_size,
            exclude_globs=exclude_globs,
            include_substrings=include_subs,
            learned_scales=use_learned_scales,
            learned_rounding=use_learned_rounding,
        )
        logger.info(f"[QAD-2bit] Re-wrapped {len(replaced)} Linear layers from Block-AP checkpoint")
        if args.quantize_moe_experts:
            exp_replaced = replace_gptoss_experts_with_2bit_ste(
                student,
                group_size=args.group_size,
                learned_scales=use_learned_scales,
            )
            logger.info(f"[QAD-2bit] Re-wrapped {len(exp_replaced)} MoE expert blocks from Block-AP checkpoint")
        if args.gradient_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
            student.gradient_checkpointing_enable()
    elif args.block_ap_warmup:
        run_block_ap_warmup(student, teacher, tokenizer, args)
        block_ap_ckpt = os.path.join(args.output_dir, "block_ap_checkpoint")
        logger.info(f"[QAD-2bit] Saving Block-AP checkpoint to {block_ap_ckpt}")
        student.save_pretrained(block_ap_ckpt, safe_serialization=True)
        tokenizer.save_pretrained(block_ap_ckpt)
        logger.info("[QAD-2bit] Block-AP checkpoint saved")

        # Block-AP leaves decoder layers on CPU for BOTH student and teacher.
        # Reload both from disk so they end up on their intended devices (via
        # HuggingFace's device_map="auto") with clean fake-quant wrappers.
        del student
        del teacher
        clear_memory()
        logger.info(f"[QAD-2bit] Reloading student from Block-AP checkpoint: {block_ap_ckpt}")
        student = _load_student_ignore_quant(block_ap_ckpt, args)
        logger.info(f"[QAD-2bit] Reloading teacher from {args.teacher_model_dir or args.model_dir}")
        teacher, _ = get_model(
            ckpt_path=(args.teacher_model_dir or args.model_dir),
            data_type="bfloat16" if args.bf16 else ("float16" if args.fp16 else "bfloat16"),
            device=args.device,
            multi_gpu=args.multi_gpu,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        replaced = replace_linears_with_group_2bit_ste(
            student,
            group_size=args.group_size,
            exclude_globs=exclude_globs,
            include_substrings=include_subs,
            learned_scales=use_learned_scales,
            learned_rounding=use_learned_rounding,
        )
        logger.info(f"[QAD-2bit] Re-wrapped {len(replaced)} Linear layers after Block-AP reload")
        if args.quantize_moe_experts:
            exp_replaced = replace_gptoss_experts_with_2bit_ste(
                student,
                group_size=args.group_size,
                learned_scales=use_learned_scales,
            )
            logger.info(f"[QAD-2bit] Re-wrapped {len(exp_replaced)} MoE expert blocks after Block-AP reload")
        if args.gradient_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
            student.gradient_checkpointing_enable()

    student.train()
    # QADTrainer expects model.teacher
    student.teacher = teacher  # type: ignore[attr-defined]

    if args.dataset == "taskmix":
        train_ds = TaskMixDataset(
            tokenizer,
            seq_len=args.seq_len,
            max_samples=args.max_train_samples,
            pile_ratio=args.pile_ratio,
            task_include=args.task_include,
            use_slimpajama=args.use_slimpajama,
            max_raw_slimpajama=args.max_raw_slimpajama,
            max_per_task=args.max_per_task,
            max_raw_pile=args.max_raw_pile,
        )
    elif args.dataset == "pile":
        train_ds = PileLMDataset(tokenizer, seq_len=args.seq_len, max_samples=args.max_train_samples)
    else:
        train_ds = WikiTextLMDataset(tokenizer, split="train", seq_len=args.seq_len, max_samples=args.max_train_samples)

    eval_ds = WikiTextLMDataset(tokenizer, split="validation", seq_len=args.seq_len, max_samples=args.max_eval_samples)

    dl_workers = args.dataloader_num_workers
    train_kw: dict = dict(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch_fused",
        dataloader_num_workers=dl_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=dl_workers > 0,
        report_to="none",
        remove_unused_columns=False,
        dataloader_drop_last=True,
        deepspeed=args.deepspeed,
        eval_strategy="no",
        logging_strategy="steps",
    )
    if dl_workers > 0:
        train_kw["dataloader_prefetch_factor"] = 4
    try:
        targs = TrainingArguments(**train_kw)
    except (TypeError, ValueError) as e:
        logger.warning("[QAD-2bit] TrainingArguments with adamw_torch_fused failed (%s); retrying default optim.", e)
        train_kw.pop("optim", None)
        targs = TrainingArguments(**train_kw)

    # Transformers >= ~4.46 uses ``processing_class``; older versions used ``tokenizer``.
    _tok_kw: dict = (
        {"processing_class": tokenizer}
        if "processing_class" in inspect.signature(Trainer.__init__).parameters
        else {"tokenizer": tokenizer}
    )
    callbacks = []
    if args.eval_every > 0:
        callbacks.append(_PPLEvalCallback(eval_ds, _collate, args.eval_every))
    if use_learned_rounding and args.rounding_anneal_steps > 0:
        callbacks.append(
            _RoundingAnnealCallback(
                student,
                anneal_steps=args.rounding_anneal_steps,
                temp_start=args.rounding_temp_start,
                temp_end=args.rounding_temp_end,
            )
        )
        logger.info(
            f"[QAD-2bit] Rounding anneal: temp {args.rounding_temp_start} → {args.rounding_temp_end} "
            f"over {args.rounding_anneal_steps} steps (cosine schedule)"
        )

    # Build custom optimizer when learned_scales or learned_rounding is active
    # so auxiliary params (log_scale, rounding_logit) get no weight decay and
    # separate (typically higher) learning rates.
    custom_optimizers: tuple = (None, None)
    if use_learned_scales or use_learned_rounding:
        scale_params, rounding_params, other_params = [], [], []
        for name, param in student.named_parameters():
            if not param.requires_grad:
                continue
            if "rounding_logit" in name:
                rounding_params.append(param)
            elif "log_scale" in name:
                scale_params.append(param)
            else:
                other_params.append(param)

        scale_lr = args.lr * args.scale_lr_mult
        rounding_lr = args.lr * args.rounding_lr_mult

        param_groups = [
            {"params": other_params, "lr": args.lr, "weight_decay": targs.weight_decay},
        ]
        if scale_params:
            param_groups.append({"params": scale_params, "lr": scale_lr, "weight_decay": 0.0})
        if rounding_params:
            param_groups.append({"params": rounding_params, "lr": rounding_lr, "weight_decay": 0.0})

        logger.info(
            f"[QAD-2bit] Custom optimizer groups: "
            f"{len(other_params)} weight params (lr={args.lr:.2e}), "
            f"{len(scale_params)} log_scale params (lr={scale_lr:.2e}), "
            f"{len(rounding_params)} rounding_logit params (lr={rounding_lr:.2e})"
        )
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(targs.adam_beta1, targs.adam_beta2),
            eps=targs.adam_epsilon,
            fused=True,
        )
        custom_optimizers = (optimizer, None)

    trainer = QADTrainer(
        model=student,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=_collate,
        **_tok_kw,
        temperature=args.kd_temperature,
        kd_mode=args.kd_mode,
        kd_loss_type=args.kd_loss_type,
        kd_alpha=args.kd_alpha,
        log_loss_breakdown=args.log_loss_breakdown,
        on_policy_kd=args.on_policy_kd,
        on_policy_every=args.on_policy_every,
        on_policy_max_gen_len=args.on_policy_max_gen_len,
        on_policy_prompt_len=args.on_policy_prompt_len,
        on_policy_temperature=args.on_policy_temperature,
        on_policy_alpha=args.on_policy_alpha,
        kd_tokenizer=tokenizer,
        callbacks=callbacks,
        optimizers=custom_optimizers,
    )

    trainer.train()
    logger.info(f"[QAD-2bit] Saving student to {args.output_dir}")
    # Teacher was attached as a submodule for QADTrainer; do not serialize it.
    if hasattr(student, "teacher"):
        delattr(student, "teacher")
    if args.quantize_moe_experts:
        bake_experts_2bit_ste(student)
        logger.info("[QAD-2bit] Baked MoE expert 2-bit fake-quant into expert weights")
    strip_group_2bit_ste_to_linear(student)
    student.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    # Clean up intermediate checkpoints to free disk space
    import glob
    import shutil

    for ckpt_dir in glob.glob(os.path.join(args.output_dir, "checkpoint-*")):
        logger.info(f"[QAD-2bit] Removing intermediate checkpoint: {ckpt_dir}")
        shutil.rmtree(ckpt_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
