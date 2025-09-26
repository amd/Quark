#
# Modifications copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2023 MIT HAN Lab
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import functools
import inspect
import math
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from quark.shares.utils.log import ScreenLogger
from quark.torch.algorithm.awq.scale import apply_clip, apply_scale
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.utils.module import append_str_prefix, get_moe_layers, get_named_quant_linears
from quark.torch.algorithm.utils.prepare import (
    cache_model_inps,
    get_layers_for_scaling,
    get_model_layers,
    init_device_map,
    reset_model_kv_cache,
)
from quark.torch.algorithm.utils.utils import clear_memory, get_num_attn_heads_from_model, is_attention_module
from quark.torch.quantization.debug import QUARK_ALGO_DEBUG
from quark.torch.quantization.tensor_quantize import NonScaledFakeQuantize, ScaledFakeQuantize
from quark.torch.quantization.utils import assert_no_nan
from quark.torch.utils.accelerate_helper import OffloadParameter, update_offload_parameter
from quark.torch.utils.exceptions import LossError
from quark.torch.utils.torch_utils import get_op_name

__all__ = ["AwqProcessor"]
logger = ScreenLogger(__name__)


class AwqProcessor(BaseAlgoProcessor):
    def __init__(self, model: nn.Module, quant_algo_config: Any, data_loader: DataLoader[torch.Tensor]) -> None:
        # assert isinstance(quant_algo_config, AWQConfig)
        self.model = model
        # If accelerate is used, the model will have the attribute _hf_hook
        self.using_accelerate = hasattr(self.model, "_hf_hook")
        self.recover_attn_implementation = self.model.config._attn_implementation
        # The `QUARK_AWQ_MEMORY_OPTIMIZATION` flag is intended for use in memory-constrained environments.
        # When enabled, it can significantly reduce GPU memory usage.
        # By default, this optimization is disabled.
        QUARK_AWQ_MEMORY_OPTIMIZATION = os.environ.get("QUARK_AWQ_MEMORY_OPTIMIZATION", None) == "1"
        if QUARK_AWQ_MEMORY_OPTIMIZATION:
            self.model.config._attn_implementation = "sdpa"
        self.device = model.device
        self.data_loader = data_loader
        self.model_decoder_layers = quant_algo_config.model_decoder_layers
        self.scaling_layers = quant_algo_config.scaling_layers
        self.device_map = init_device_map(self.model)
        self.modules, self.module_kwargs, self.inps = self.init_quant()
        self.global_scales_list: list[torch.Tensor] = []
        self.num_attention_heads, self.num_key_value_heads = get_num_attn_heads_from_model(model)

    def apply(self) -> None:
        # prevent OOM.
        # The forward of awq requires extra memory, and the simultaneous input of n batches, as opposed to batch by batch,
        # allows for multiple speedups at the expense of device transfer time (which is small enough compared to batch by batch),
        # as well as better OOM prevention.
        if not self.using_accelerate:
            for i in range(len(self.modules)):
                self.modules[i] = self.modules[i].to("cpu")
        clear_memory()
        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            # Move module and inputs to correct device
            common_device = next(self.modules[i].parameters()).device
            if common_device is None or str(common_device) == "cpu":
                common_device = self.device_map[f"{self.model_decoder_layers}.{i}"]
                self.modules[i] = self.modules[i].to(common_device)

            # [STEP 1]: Get layer, extract linear modules, extract input features
            named_linears = get_named_quant_linears(self.modules[i])
            moe_input_layers = get_moe_layers(self.modules[i])
            named_input_layers = {**named_linears, **moe_input_layers}
            input_feat = self._get_input_feat(self.modules[i], named_input_layers)
            clear_memory()

            # [STEP 2]: Compute and apply scale list
            module_config: list[dict[str, Any]] = get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs, self.scaling_layers
            )
            scales_list = []
            for layer in module_config:
                scales = self._search_best_scale(
                    self.modules[i], **layer
                )  # scales: (pre_layer, layer, best_scales, best_ratio)
                if scales is not None:
                    if QUARK_ALGO_DEBUG:
                        logger.info(
                            f"AWQ for layer {i}: {scales[1]}, best_ratio={scales[3]}, scales_max={scales[2].max().item()}, scales_min={scales[2].min().item()}"
                        )
                    scales_list.append(scales[:-1])

            apply_scale(
                self.modules[i],
                scales_list,
                input_feat_dict=input_feat,
                device=common_device,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
            )
            scales_list = append_str_prefix(scales_list, get_op_name(self.model, self.modules[i]) + ".")
            if os.environ.get("QUARK_SAVE_ACTIVATION_SCALES", "None") == "true":
                self.global_scales_list.extend(scales_list)

            # [STEP 3]: Compute and apply clipping list
            clip_list = self._search_best_clip(named_linears, input_feat)
            apply_clip(self.modules[i], clip_list, common_device)
            clip_list = append_str_prefix(clip_list, get_op_name(self.model, self.modules[i]) + ".")

            # [STEP 4]: Quantize weights
            self._apply_quant(named_linears)
            self.modules[i] = self.modules[i].to("cpu")
            clear_memory()

        # recover model attention config
        self.model.config._attn_implementation = self.recover_attn_implementation

        if os.environ.get("QUARK_SAVE_ACTIVATION_SCALES", "None") == "true":
            filename = os.environ.get("QUARK_ACTIVATION_SCALES_FILENAME", "activation_scales_awq.pt")
            torch.save(self.global_scales_list, filename)
            logger.info(f"AWQ Activation Scales Successfully Saved to {filename}")

    @torch.no_grad()
    def _search_best_scale(
        self,
        module: nn.Module,
        prev_op: nn.Module,
        layers: list[nn.Linear],
        inp: torch.Tensor,
        module2inspect: nn.Module | None = None,
        kwargs: dict[str, Any] = {},
    ) -> tuple[str, tuple[str, ...], torch.Tensor, float]:
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        if "past_key_value" in kwargs:
            kwargs.pop("past_key_value")

        # Put x on the right device
        inp_device = next(module2inspect.parameters()).device
        if inp_device.type == "meta":
            inp_device = torch.device("cuda")
        inp = inp.to(inp_device)

        # [STEP 1]: Compute maximum of weight
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        group_size = layers[0]._weight_quantizer.group_size
        for _m in layers:
            assert _m._weight_quantizer.group_size == group_size
        if group_size is not None and group_size > 0:
            weight = weight.view(-1, group_size)
        weight = weight.abs_()
        weight.div_(weight.amax(dim=1, keepdim=True))
        w_scale = weight
        clear_memory(weight)
        w_scale = w_scale.view(org_shape)
        w_max = w_scale.mean(0)
        clear_memory(weight)

        # [STEP 2]: Compute maximum of x
        x_max = inp.abs().view(-1, inp.shape[-1]).mean(0)

        # [STEP 3]: Compute output of module
        with torch.no_grad():
            if is_attention_module(module2inspect):
                with self._capture_layer_output(module2inspect) as hook_outputs:
                    _ = module(self.inps[0], **self.module_kwargs)
                fp16_output = hook_outputs["output"]
            else:
                forward_params = inspect.signature(module2inspect.forward).parameters
                filtered_kwargs = {
                    k: v for k, v in kwargs.items() if k in forward_params
                }  # the parameters of module2inspect may or may not be the same as the decoder, so need it
                fp16_output = module2inspect(inp, **filtered_kwargs)

            assert fp16_output is not None
            if isinstance(fp16_output, tuple):
                fp16_output = fp16_output[0]

        # [STEP 4]: Compute loss
        best_scales, best_ratio = self._compute_best_scale(
            module, inp, w_max, x_max, module2inspect, layers, fp16_output, kwargs
        )

        return (get_op_name(module, prev_op), tuple([get_op_name(module, m) for m in layers]), best_scales, best_ratio)

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_output: torch.Tensor,
        int_w_output: torch.Tensor,
        device: torch.device,
    ) -> float:
        loss = 0.0
        fp16_output_flat = fp16_output.to(device).view(-1)
        int_w_output_flat = int_w_output.to(device).view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()
        max_chunk_memory = (1024 * 1024 * 1024) // 2  # max memory 0.5G
        chunk_size = max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)
        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks, strict=False):
            chunk_loss = (fp16_chunk - int_w_chunk).float().pow(2).sum().item()
            loss += chunk_loss
        return loss / num_elements

    def _compute_best_scale(
        self,
        module: nn.Module,
        x: torch.Tensor,
        w_max: torch.Tensor,
        x_max: torch.Tensor,
        module2inspect: nn.Module,
        linears2scale: list[nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: dict[str, Any] = {},
    ) -> tuple[torch.Tensor, float]:
        n_grid = 20
        BEST_RATIO_INIT = -1.0
        best_ratio = BEST_RATIO_INIT
        best_scales = None
        best_error = float("inf")

        # The `.clone()` is necessary to bypass this bug: https://github.com/pytorch/pytorch/issues/137710

        device = x.device
        x_max = x_max.view(-1).to(device)
        if not self.using_accelerate:
            w_max = w_max.view(-1).to(device)

        scales_view = None
        hook_list = []

        def pre_hook(fc: torch.nn.Module, input: Any) -> Any:
            with OffloadParameter(fc):
                weight = fc.weight * scales_view
            assert isinstance(fc, torch.nn.Linear)
            quantized = self.pseudo_quantize_tensor(weight, fc) / scales_view

            # Temporarily turn off accelerate offload. If it is not turned off, the update of weight by pytorch hook will not take effect.
            if hasattr(fc, "_hf_hook") and fc._hf_hook.offload:
                fc._hf_hook_offload = fc._hf_hook.offload
                fc._hf_hook.offload = False
            fc.weight_bck = fc.weight.cpu().clone()
            fc.weight.copy_(quantized)
            return input

        def post_hook(fc: torch.nn.Module, input: Any, output: Any) -> Any:
            # Reset accelerate offload to its default state
            if hasattr(fc, "_hf_hook") and hasattr(fc, "_hf_hook_offload") and not fc._hf_hook.offload:
                fc._hf_hook.offload = fc._hf_hook_offload
            fc.weight.copy_(fc.weight_bck.to(fc.weight.device))
            del fc.weight_bck
            return output

        # The AWQ algorithm requires modifying the weights to find the optimal ones. We use hooks to change the weights only during usage, and then restore the original weights afterward.
        for fc in linears2scale:
            hook_list.append(fc.register_forward_pre_hook(pre_hook))
            hook_list.append(fc.register_forward_hook(post_hook))

        for i in range(n_grid):
            # create new scales
            ratio = i / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)

            # W * X
            if is_attention_module(module2inspect):
                with self._capture_layer_output(module2inspect) as hook_outputs:
                    _ = module(self.inps[0], **self.module_kwargs)
                int_w_output = hook_outputs["output"]
            else:
                forward_params = inspect.signature(
                    module2inspect.forward
                ).parameters  # the parameters of module2inspect may or may not be the same as the decoder, so need it
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in forward_params}
                int_w_output = module2inspect(x, **filtered_kwargs)

            assert int_w_output is not None
            if isinstance(int_w_output, tuple):
                int_w_output = int_w_output[0]

            # compute mean squared error (L2 norm)
            loss = self._compute_loss(fp16_output, int_w_output, device)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()

            clear_memory()

        for hook in hook_list:
            hook.remove()

        if best_ratio == BEST_RATIO_INIT or best_scales is None:
            raise LossError(
                "The best_ratio and best_scales were not updated. Please check whether the loss computation contains any NaN or Inf values."
            )

        assert_no_nan(best_scales, message="best_scales contains NaN!")
        return best_scales.detach().cpu(), best_ratio

    @torch.no_grad()
    def _search_best_clip(
        self, named_linears: dict[str, nn.Linear], input_feat: dict[str, Any]
    ) -> list[tuple[str, torch.Tensor]]:
        clip_list: list[tuple[str, torch.Tensor]] = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            if name not in input_feat:  # For MoeBlock in Moe Models
                continue
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue
            if not self.using_accelerate:
                named_linears[name].to(self.device)
            max_val = self._compute_best_clip(named_linears[name], input_feat[name])
            clip_list.append((name, max_val))
            if not self.using_accelerate:
                named_linears[name].cpu()

        return clip_list

    @torch.no_grad()
    def _compute_best_clip(
        self,
        named_linears: torch.nn.Linear,
        input_feat: torch.Tensor,
        n_grid: int = 20,
        max_shrink: float = 0.5,
        n_sample_token: int = 512,
    ) -> torch.Tensor:
        with OffloadParameter(named_linears):
            w = named_linears.weight
        assert w.dim() == 2
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        tmp_group_size = named_linears._weight_quantizer.group_size
        group_size = tmp_group_size if tmp_group_size is not None and tmp_group_size > 0 else w.shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
        n_sample_token = min(input_feat.shape[1], n_sample_token)
        input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
        w = w.reshape(w.shape[0], 1, -1, group_size)

        oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
        w_all = w
        best_max_val_all = []

        for i_b in range(math.ceil(w.shape[0] / oc_batch_size)):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = cur_w
                q_w = self.pseudo_quantize_tensor(cur_w, named_linears)
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs = torch.where(cur_best_idx, err, min_errs)
                best_max_val = torch.where(cur_best_idx, max_val, best_max_val)
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)

    def _get_input_feat(self, layer: nn.Module, named_linears: dict[str, nn.Linear]) -> dict[str, torch.Tensor]:
        # firstly, get input features of all linear layers
        def cache_input_hook(
            m: nn.Module, x: tuple[torch.Tensor], y: torch.Tensor, name: str, feat_dict: dict[str, list[torch.Tensor]]
        ) -> None:
            x = x[0]
            x = x.detach().cpu()
            if x.numel() > 0:  # for moe layer
                feat_dict[name].append(x)

        input_feat: dict[str, list[torch.Tensor]] = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        model_device = next(layer.parameters()).device
        if "meta" in str(model_device):
            model_device = torch.device("cuda")
        for i, inp in enumerate(self.inps):
            if inp is not None:
                inp = inp.to(next(layer.parameters()).device)
        # get output as next layer's input

        if "kwargs" in self.module_kwargs and self.module_kwargs["kwargs"] is None:
            self.module_kwargs.pop("kwargs")
        self.layer_inps = self.inps[0]
        output = layer(self.inps[0], **self.module_kwargs)
        self.inps = [output[0]] if isinstance(output, tuple) else [output]

        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        return input_feat

    @torch.no_grad()
    def pseudo_quantize_tensor(
        self, w: torch.Tensor, linear_layer: nn.Linear, get_scale_zp: bool = False
    ) -> torch.Tensor:
        for module in linear_layer.modules():
            if isinstance(module, ScaledFakeQuantize) or isinstance(module, NonScaledFakeQuantize):
                module.enable_observer()
                module.enable_fake_quant()

        if not get_scale_zp:
            org_w_shape = w.shape
            group_size = linear_layer._weight_quantizer.group_size
            if group_size is not None and group_size > 0:
                assert org_w_shape[-1] % group_size == 0
                w = w.reshape(-1, group_size)
            else:
                w = w.reshape(-1, w.shape[-1])
            assert w.dim() == 2
            w_q = linear_layer._weight_quantizer(w)
            w_q = w_q.reshape(org_w_shape)
        else:
            w_q = linear_layer._weight_quantizer(w)

        linear_layer._weight_quantizer.observer.reset_state()
        if not self.using_accelerate:
            linear_layer._weight_quantizer.observer.to(self.device)
        for module in linear_layer.modules():
            if isinstance(module, ScaledFakeQuantize) or isinstance(module, NonScaledFakeQuantize):
                module.disable_observer()
                module.disable_fake_quant()

        if get_scale_zp:
            linear_layer.weight.data = w_q
        return cast(torch.Tensor, w_q)

    def init_quant(self) -> tuple[nn.ModuleList, dict[str, Any], list[Any]]:
        modules = get_model_layers(self.model, self.model_decoder_layers)
        forward_pass_use_cache = reset_model_kv_cache(self.model, use_cache=False)
        modules, layer_kwargs, inputs = cache_model_inps(self.model, modules, self.data_loader)
        reset_model_kv_cache(self.model, use_cache=forward_pass_use_cache)
        return modules, layer_kwargs, inputs

    def _apply_quant(self, named_linears: dict[str, nn.Linear]) -> None:
        for name, linear_layer in named_linears.items():
            with OffloadParameter(linear_layer):
                # NOTE: small regression in perplexity if linear layer uses .cpu().float()
                if not self.using_accelerate:
                    linear_layer = linear_layer.to(self.device)
                self.pseudo_quantize_tensor(linear_layer.weight.data, linear_layer, get_scale_zp=True)
                update_offload_parameter(linear_layer, "weight", linear_layer.weight)

    @contextmanager
    def _capture_layer_output(self, module: nn.Module) -> Generator[dict[str, Any], None, None]:
        class _StopForward(Exception):
            pass

        outputs: dict[str, Any] = {}

        def create_hook() -> Callable[[nn.Module, tuple[Any, ...], Any], Any | None]:
            def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> Any | None:
                outputs["output"] = output
                raise _StopForward(output)

            return hook

        handle = module.register_forward_hook(create_hook())
        try:
            yield outputs
        except _StopForward:
            pass
        finally:
            handle.remove()
