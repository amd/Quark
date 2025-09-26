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
import typing
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from quark.shares.utils.log import ScreenLogger
from quark.torch.algorithm.awq.scale import apply_scale
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.utils.module import append_str_prefix, get_moe_layers, get_named_quant_linears
from quark.torch.algorithm.utils.prepare import (
    cache_model_inps,
    get_layers_for_scaling,
    get_model_layers,
    init_device_map,
    reset_model_kv_cache,
)
from quark.torch.algorithm.utils.utils import clear_memory, get_num_attn_heads_from_model
from quark.torch.kernel import mx
from quark.torch.quantization import OCP_MXFP4Spec
from quark.torch.quantization.debug import QUARK_ALGO_DEBUG
from quark.torch.quantization.tensor_quantize import NonScaledFakeQuantize, ScaledFakeQuantize
from quark.torch.quantization.utils import assert_no_nan
from quark.torch.utils.torch_utils import get_op_name

if typing.TYPE_CHECKING:
    from quark.torch.quantization.config.config import QuantizationSpec

logger = ScreenLogger(__name__)

CPU = torch.device("cpu")
CUDA = torch.device("cuda")


def is_mxfp4_spec(qspec: QuantizationSpec) -> bool:
    """
    Check if the given quantization spec is an OCP_MXFP4Spec.
    """
    # TODO: Only used to dispatch to more efficient mxfp4 qdq kernel.
    # The efficiency of observer + fake quant should be improved.
    if qspec.scale_calculation_mode is None:
        return False
    mxfp4_spec = OCP_MXFP4Spec(
        ch_axis=-1, is_dynamic=True, scale_calculation_mode=qspec.scale_calculation_mode
    ).to_quantization_spec()
    dynamic_mxfp4_spec = OCP_MXFP4Spec(
        ch_axis=-1, is_dynamic=False, scale_calculation_mode=qspec.scale_calculation_mode
    ).to_quantization_spec()
    return qspec in (mxfp4_spec, dynamic_mxfp4_spec)


class AutoSmoothQuantProcessor(BaseAlgoProcessor):
    def __init__(self, model: nn.Module, quant_algo_config: Any, data_loader: DataLoader[torch.Tensor]) -> None:
        self.model = model
        self.device = model.device
        self.data_loader = data_loader
        self.model_decoder_layers = quant_algo_config.model_decoder_layers
        self.scaling_layers = quant_algo_config.scaling_layers
        self.compute_scale_loss = quant_algo_config.compute_scale_loss
        self.device_map = init_device_map(self.model)
        self.modules, self.module_kwargs, self.inps = self.init_quant()
        self.num_attention_heads, self.num_key_value_heads = get_num_attn_heads_from_model(model)

    def apply(self) -> None:
        # prevent OOM.
        # The forward of awq requires extra memory, and the simultaneous input of n batches, as opposed to batch by batch,
        # allows for multiple speedups at the expense of device transfer time (which is small enough compared to batch by batch),
        # as well as better OOM prevention.
        for i in range(len(self.modules)):
            self.modules[i] = self.modules[i].to("cpu")
        clear_memory()
        for i in tqdm(range(len(self.modules)), desc="Auto-SmoothQuant"):
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
                if QUARK_ALGO_DEBUG:
                    logger.info(
                        f"AutoSmoothQuant for layer {i}: {scales[1]}, best_ratio={scales[3]}, scales_max={scales[2].max().item()}, scales_min={scales[2].min().item()}"
                    )
                scales_list.append(scales[:-1])

            apply_scale(
                self.modules[i],
                scales_list,
                input_feat_dict=None,
                device=common_device,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
            )
            scales_list = append_str_prefix(scales_list, get_op_name(self.model, self.modules[i]) + ".")

            self.modules[i] = self.modules[i].to("cpu")
            clear_memory()

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
        for inp in self.inps:
            if inp is not None:
                inp = inp.to(next(layer.parameters()).device)
        # get output as next layer's input

        output = layer(self.inps[0], **self.module_kwargs)
        self.inps = [output[0]] if isinstance(output, tuple) else [output]

        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        return input_feat

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
        inp = inp.to(next(module2inspect.parameters()).device)

        # [STEP 1]: Compute maximum of x
        x_max = inp.abs().view(-1, inp.shape[-1]).max(0)[0]

        # [STEP 2]: Compute output of module
        with torch.no_grad():
            forward_params = inspect.signature(module2inspect.forward).parameters
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in forward_params
            }  # the parameters of module2inspect may or may not be the same as the decoder, so need it
            fp16_output = module2inspect(inp, **filtered_kwargs)
            if isinstance(fp16_output, tuple):
                fp16_output = fp16_output[0]

        # [STEP 3]: Compute loss
        best_scales, best_ratio = self._compute_best_scale(inp, x_max, module2inspect, layers, fp16_output, kwargs)

        return (get_op_name(module, prev_op), tuple([get_op_name(module, m) for m in layers]), best_scales, best_ratio)

    def _compute_best_scale(
        self,
        x: torch.Tensor,
        x_max: torch.Tensor,
        module2inspect: nn.Module,
        linears2scale: list[nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: dict[str, Any] = {},
    ) -> tuple[torch.Tensor, float]:
        n_grid = 10

        org_sd = {k: v.clone() for k, v in module2inspect.state_dict().items()}

        device = x.device
        x_max = x_max.view(-1).to(device)

        losses = []
        candidate_scales = []
        candidate_ratios = []

        for i in range(n_grid):
            ratio = i / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)

            # Q(W * s)
            for fc in linears2scale:
                fc.weight.mul_(scales_view)
                fc.weight.data = self.pseudo_quantize_tensor(fc.weight.data, fc)

            x_scale = x.div(scales_view)
            x_q = self.pseudo_quantize_tensor(x_scale, fc, False, False)

            # W * X
            forward_params = inspect.signature(
                module2inspect.forward
            ).parameters  # the parameters of module2inspect may or may not be the same as the decoder, so need it
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in forward_params}
            int_w_output = module2inspect(x_q, **filtered_kwargs)
            if isinstance(int_w_output, tuple):
                int_w_output = int_w_output[0]

            if self.compute_scale_loss == "MSE":
                pow_num = 2.0
            elif self.compute_scale_loss == "MAE":
                pow_num = 1.0
            elif self.compute_scale_loss == "RMSE":
                pow_num = 0.5
            else:
                raise ValueError(
                    f"Invalid value for compute_scale_loss: {self.compute_scale_loss}. Expected 'MAE', 'MSE' or 'RMSE'."
                )

            loss = (fp16_output - int_w_output).float().abs().pow(pow_num).mean()  # NOTE: float prevents overflow
            losses.append(loss)
            candidate_scales.append(scales)
            candidate_ratios.append(ratio)

            module2inspect.load_state_dict(org_sd)

        best_idx = torch.stack(losses).argmin().item()
        best_scales = candidate_scales[best_idx]  # type: ignore[index]
        best_ratio = candidate_ratios[best_idx]  # type: ignore[index]

        assert_no_nan(best_scales, "best_scales should not contain NaN values")

        return best_scales.detach(), best_ratio

    @torch.no_grad()
    def pseudo_quantize_tensor(
        self,
        w: torch.Tensor,
        linear_layer: nn.Linear,
        get_scale_zp: bool = False,  # TODO: unused
        is_weight: bool = True,
    ) -> torch.Tensor:
        if is_weight:
            quantizer = linear_layer._weight_quantizer
        else:
            quantizer = linear_layer._input_quantizer

        if quantizer is None:
            return w

        qspec: QuantizationSpec = quantizer.quant_spec
        if is_mxfp4_spec(qspec) and not get_scale_zp:
            # TODO: Make observer + fake as efficient as this.
            assert qspec.scale_calculation_mode is not None
            w_q = mx.qdq_mxfp4(w, qspec.scale_calculation_mode)

        else:
            for module in linear_layer.modules():
                if isinstance(module, ScaledFakeQuantize) or isinstance(module, NonScaledFakeQuantize):
                    module.enable_observer()
                    module.enable_fake_quant()

            if not get_scale_zp:
                org_w_shape = w.shape
                group_size = quantizer.group_size
                if group_size is not None and group_size > 0:
                    assert org_w_shape[-1] % group_size == 0
                    w = w.reshape(-1, group_size)
                else:
                    w = w.reshape(-1, w.shape[-1])
                assert w.dim() == 2
                w_q = quantizer(w)
                w_q = w_q.reshape(org_w_shape)
            else:
                w_q = quantizer(w)

            quantizer.observer.reset_state()
            quantizer.observer.to(linear_layer.weight.device)
            for module in linear_layer.modules():
                if isinstance(module, ScaledFakeQuantize) or isinstance(module, NonScaledFakeQuantize):
                    module.disable_observer()
                    module.disable_fake_quant()

        if get_scale_zp:
            linear_layer.weight.data = w_q
        return w_q

    def init_quant(self) -> tuple[nn.ModuleList, dict[str, Any], list[Any]]:
        modules = get_model_layers(self.model, self.model_decoder_layers)
        forward_pass_use_cache = reset_model_kv_cache(self.model, use_cache=False)
        for i in range(len(modules)):
            if i > 0:
                modules[i] = modules[i].to("cpu")
        clear_memory()
        modules, layer_kwargs, inputs = cache_model_inps(self.model, modules, self.data_loader)
        reset_model_kv_cache(self.model, use_cache=forward_pass_use_cache)
        return modules, layer_kwargs, inputs
