#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from quark.common.utils.log import ScreenLogger
from quark.experimental.torch.autoround.wrapper import _WrapperLinearBase, make_wrapper
from quark.torch.algorithm.blockwise_tuning.blockwise_utils import block_batch_forward, block_forward
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.utils.module import get_device, get_nested_attr_from_module, move_to_device
from quark.torch.algorithm.utils.prepare import (
    get_model_layers,
    init_blockwise_algo,
    init_device_map,
    reset_model_kv_cache,
)
from quark.torch.algorithm.utils.utils import clear_memory

if TYPE_CHECKING:
    from quark.torch.quantization.config.config import AutoRoundConfig

logger = ScreenLogger(__name__)

__all__ = ["AutoRoundProcessor", "optimize_wrappers_signed_sgd"]

CPU = torch.device("cpu")


def _set_nested_module(root: nn.Module, attr_path: str, value: nn.Module) -> None:
    """Set a (possibly dotted) submodule attribute on ``root``."""
    parts = attr_path.split(".")
    parent = root if len(parts) == 1 else get_nested_attr_from_module(root, ".".join(parts[:-1]))
    setattr(parent, parts[-1], value)


class _EpochIndexSampler:
    """Cycles through ``nsamples`` indices in shuffled order without replacement per epoch,
    reshuffling once exhausted -- matches the official repo's minibatch sampler, guaranteeing
    every calibration sample is drawn exactly once per ``nsamples // batch_size`` steps (unlike
    i.i.d. ``random.sample`` per step, which sign-SGD can overfit to uneven sample coverage)."""

    def __init__(self, nsamples: int, batch_size: int) -> None:
        self.nsamples = nsamples
        self.batch_size = max(1, min(batch_size, nsamples))
        self.indices = list(range(nsamples))
        random.shuffle(self.indices)
        self.pos = 0

    def next_batch(self) -> list[int]:
        if self.pos + self.batch_size > self.nsamples:
            random.shuffle(self.indices)
            self.pos = 0
        batch = self.indices[self.pos : self.pos + self.batch_size]
        self.pos += self.batch_size
        return batch


def optimize_wrappers_signed_sgd(
    block: nn.Module,
    wrappers: dict[str, _WrapperLinearBase],
    module_kwargs: dict[str, Any],
    layer_inputs: list[torch.Tensor],
    target_outputs: list[torch.Tensor],
    device: torch.device,
    iters: int,
    lr: float,
    batch_size: int = 8,
    minmax_lr: float | None = None,
) -> float:
    """AutoRound per-block optimization: learn the rounding offset V (and optionally each
    wrapper's clip-tuning parameter(s)) by signed gradient descent.

    Runs ``iters`` total minibatch steps per block (not ``iters`` epochs, matching the paper).
    Each step draws up to ``batch_size`` inputs via ``_EpochIndexSampler``, sums their
    reconstruction MSE, and takes one signed-gradient step: V at ``lr``, each wrapper's
    ``clip_params()`` (empty unless minmax tuning is on) in a second SGD param group at
    ``minmax_lr`` (defaults to ``lr``). All gradients are replaced by their sign.

    Returns the best-loss-step MSE (best-loss params, including clip params, are restored at the
    end -- signed-gradient descent with a fixed step size hovers around the optimum rather than
    converging to it, so the last step's params can be worse than an earlier one's).
    """
    v_params = [w.value for w in wrappers.values()]
    clip_params: list[torch.Tensor] = []
    for w in wrappers.values():
        clip_params.extend(w.clip_params())

    params = v_params + clip_params

    # Only the learnable params are trainable; freeze everything else in the block.
    for p in block.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)

    # lr/minmax_lr as torch.tensor, not a plain float -- matches the official repo's
    # quantizer.py (`lr = torch.tensor(self.lr)`); kept purely for fidelity, numerically
    # equivalent to a float lr under LinearLR to ~1e-6.
    lr_t = torch.tensor(lr)
    minmax_lr_t = lr_t if minmax_lr is None else torch.tensor(minmax_lr)
    param_groups = [{"params": v_params, "lr": lr_t}]
    if clip_params:
        param_groups.append({"params": clip_params, "lr": minmax_lr_t})
    opt = torch.optim.SGD(param_groups)

    # Linear LR decay over the tuning steps (paper §4.1), applied to both param groups.
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=max(1, iters))

    # bf16 autocast during tuning on CUDA (paper enables AMP); disabled on CPU so the fast CPU
    # unit test is unaffected.
    use_amp = device.type == "cuda"

    # V/min_scale/max_scale live in the model's (possibly fp16/bf16) weight dtype, so .grad can
    # underflow to exactly 0 before sign() sees it. Scaling the loss up before backward (safe --
    # sign-SGD only reads sign()) matches the official repo's `_scale_loss_and_backward`.
    LOSS_SCALE = 1000.0

    num_batches = len(layer_inputs)
    sampler = _EpochIndexSampler(num_batches, batch_size)
    mb = sampler.batch_size

    best_loss = float("inf")
    best_params = [p.detach().clone() for p in params]
    for _ in range(iters):
        chosen = sampler.next_batch()
        opt.zero_grad()
        loss = torch.zeros((), device=device)
        for j in chosen:
            x = move_to_device(layer_inputs[j], device)
            t = move_to_device(target_outputs[j], device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = block_batch_forward(block, module_kwargs, x, device)
            # fp32 upcast before the loss (matches the official repo) -- sign-SGD is sensitive
            # to sign flips from bf16 rounding noise in the reconstruction loss itself.
            loss = loss + F.mse_loss(out.float(), t.float())
        loss = loss / mb
        (loss * LOSS_SCALE).backward()
        for p in params:
            if p.grad is not None:
                p.grad = torch.sign(p.grad)
        # Snapshot BEFORE opt.step()/scheduler.step(): `loss` was computed with the current
        # (pre-update) params, so the best-loss snapshot must be taken here to actually
        # correspond to `best_loss` -- taking it after the step would restore the wrong state.
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = [p.detach().clone() for p in params]
        opt.step()
        scheduler.step()

    with torch.no_grad():
        for p, b in zip(params, best_params, strict=False):
            p.copy_(b)

    return best_loss


class AutoRoundProcessor(BaseAlgoProcessor):
    """AutoRound (arXiv 2309.05516) block-reconstruction processor.

    Mirrors :class:`BlockwiseTuningProcessor`: uses ``init_blockwise_algo`` to capture real
    per-block inputs (with attention_mask/position_embeddings in ``module_kwargs``), runs the
    full-precision block as the reconstruction target via ``block_forward``, optimizes the
    learnable rounding offset V of each quantized linear (reading its real Quark
    ``weight_quantizer`` through ``make_wrapper`` — ``WrapperLinearInt`` for INT4,
    ``WrapperLinearMXFP4`` for OCP microscaling FP4), then propagates the quantized block
    outputs to the next block.
    """

    def __init__(
        self,
        fp_model: nn.Module,
        model: nn.Module,
        algo_config: AutoRoundConfig,
        data_loader: DataLoader[torch.Tensor],
    ) -> None:
        self.fp_model = fp_model
        self.model = model
        self.config = algo_config
        self.model_decoder_layers = algo_config.model_decoder_layers
        self.data_loader = data_loader

        self.last_block_mse: float | None = None

        self.device_map = init_device_map(self.model)
        self.modules, self.module_kwargs, self.inps = init_blockwise_algo(
            self.model, self.model_decoder_layers, self.data_loader
        )
        self.modules_fp = get_model_layers(self.fp_model, self.model_decoder_layers)

    def _wrap_block(self, block: nn.Module) -> dict[str, _WrapperLinearBase]:
        """Wrap each configured linear with a wrapper (dispatched by ``make_wrapper`` on the
        linear's weight_quantizer dtype) that reads its real weight_quantizer.

        ``inside_layer_modules`` may list module names that only exist on some blocks
        (e.g. hybrid attention architectures where blocks alternate between linear-attention
        and full-attention layers with disjoint submodule names) -- entries absent from a
        given block are skipped rather than raising.
        """
        wrappers: dict[str, _WrapperLinearBase] = {}
        for name in self.config.inside_layer_modules:
            try:
                linear = get_nested_attr_from_module(block, name)
            except AttributeError:
                continue
            wrapper = make_wrapper(linear, enable_minmax_tuning=self.config.enable_minmax_tuning)
            _set_nested_module(block, name, wrapper)
            wrappers[name] = wrapper
        return wrappers

    @torch.no_grad()
    def _freeze_block(self, block: nn.Module, wrappers: dict[str, _WrapperLinearBase]) -> None:
        """Bake the learned quantized weight into each linear and restore the plain linear.

        Idempotent: the baked weight equals its own dequantization, so a subsequent forward
        through the layer's weight_quantizer reproduces it exactly.
        """
        for name, wrapper in wrappers.items():
            # When minmax tuning is on, write the tuned scale/zero_point back into the real
            # weight_quantizer BEFORE baking, so that inference re-quantization of the baked
            # weight uses the tuned qparams and reproduces the tuned output exactly. The baked
            # w_deq already lies on the tuned quant grid, so weight_quantizer(baked) == baked.
            if getattr(wrapper, "enable_minmax_tuning", False):
                weight_quantizer = getattr(wrapper.linear, "weight_quantizer", None)
                if weight_quantizer is not None:
                    scale, zero_point = wrapper.get_scale_zero_point()  # 2D [out, n_groups]
                    tgt_scale = weight_quantizer.scale
                    tgt_zp = weight_quantizer.zero_point
                    scale = scale.reshape(tgt_scale.shape).to(dtype=tgt_scale.dtype, device=tgt_scale.device)
                    zero_point = zero_point.reshape(tgt_zp.shape).to(dtype=tgt_zp.dtype, device=tgt_zp.device)
                    tgt_scale.copy_(scale)
                    tgt_zp.copy_(zero_point)
            wrapper.linear.weight.data.copy_(wrapper._quantize_weight())
            _set_nested_module(block, name, wrapper.linear)

    def _autoround_block(
        self,
        block: nn.Module,
        device: torch.device,
        layer_inputs: list[torch.Tensor],
        target_outputs: list[torch.Tensor],
    ) -> None:
        wrappers = self._wrap_block(block)
        self.last_block_mse = optimize_wrappers_signed_sgd(
            block,
            wrappers,
            self.module_kwargs,
            layer_inputs,
            target_outputs,
            device,
            self.config.iters,
            self.config.lr,
            self.config.batch_size,
            self.config.minmax_lr if self.config.enable_minmax_tuning else None,
        )
        self._freeze_block(block, wrappers)

    def apply(self) -> None:
        # Disable TF32 for the duration of tuning only (matches the official repo -- reduced
        # matmul precision would perturb the reconstruction-error signal signed-SGD tunes
        # against), then restore both flags to their prior values afterward. `cudnn.flags(...)`
        # is a context manager and has no effect unless entered with `with`.
        prior_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
                self._apply()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prior_allow_tf32

    def _apply(self) -> None:
        # keep block outputs on cpu
        cache_examples_on_gpu = False

        num_batches = len(self.inps)

        layer_inputs = [inp.detach().requires_grad_(False) for inp in self.inps]
        layer_outputs: list[torch.Tensor] = []

        fp_layer_inputs = list(layer_inputs)
        fp_layer_outputs: list[torch.Tensor] = []

        forward_pass_use_cache = reset_model_kv_cache(self.model, use_cache=False)
        fp_forward_pass_use_cache = reset_model_kv_cache(self.fp_model, use_cache=False)

        # tuning on gpu, other blocks in cpu
        for i in range(len(self.modules)):
            self.modules[i] = self.modules[i].to("cpu")
        clear_memory()

        for i in tqdm(range(len(self.modules)), desc="AutoRound"):
            logger.info(f"Start AutoRound layer {i + 1}/{len(self.modules)}")
            layer = self.modules[i]
            layer_fp = self.modules_fp[i]

            force_layer_back_to_cpu = False
            if get_device(layer) == CPU:
                move_to_device(layer, self.device_map[f"{self.model_decoder_layers}.{i}"])
                force_layer_back_to_cpu = True
            cur_layer_device = get_device(layer)

            # full-precision block output = reconstruction target
            fp_layer_outputs = block_forward(
                layer_fp,
                self.module_kwargs,
                num_batches,
                cur_layer_device,
                fp_layer_inputs,
                fp_layer_outputs,
                cache_examples_on_gpu,
            )

            layer_fp = move_to_device(layer_fp, CPU if force_layer_back_to_cpu else cur_layer_device)

            # optimize V (signed-SGD) then bake the quantized weights
            self._autoround_block(layer, cur_layer_device, layer_inputs, fp_layer_outputs)

            # quantized block output to propagate to the next block
            layer_outputs = block_forward(
                layer,
                self.module_kwargs,
                num_batches,
                cur_layer_device,
                layer_inputs,
                layer_outputs,
                cache_examples_on_gpu,
            )

            layer = move_to_device(layer, CPU if force_layer_back_to_cpu else cur_layer_device)

            del layer
            del layer_fp
            del layer_inputs
            del fp_layer_inputs
            layer_inputs, layer_outputs = layer_outputs, []  # noqa
            fp_layer_inputs, fp_layer_outputs = fp_layer_outputs, []  # noqa
            clear_memory()

        reset_model_kv_cache(self.model, use_cache=forward_pass_use_cache)
        reset_model_kv_cache(self.fp_model, use_cache=fp_forward_pass_use_cache)

        del self.fp_model
        clear_memory()
