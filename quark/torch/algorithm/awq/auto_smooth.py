#
# Modifications copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2023 MIT HAN Lab
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import contextlib
import fnmatch
import functools
import typing
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from quark.common.utils.log import ScreenLogger
from quark.torch.algorithm.awq.scale import apply_scale
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.utils.module import (
    append_str_prefix,
    get_moe_layers,
    get_named_quant_linears,
    resolve_per_layer_kwargs,
)
from quark.torch.algorithm.utils.prepare import (
    cache_model_inps,
    get_layers_for_scaling,
    get_model_layers,
    init_device_map,
    reset_model_kv_cache,
)
from quark.torch.algorithm.utils.utils import clear_memory, get_num_attn_heads_from_model
from quark.torch.kernel import mx
from quark.torch.quantization.tensor_quantize import NonScaledFakeQuantize, ScaledFakeQuantize
from quark.torch.utils import QUARK_ALGO_DEBUG, assert_no_nan, get_op_name

from .utils import align_attention_mask_with_input

if typing.TYPE_CHECKING:
    from quark.torch.quantization.config.config import QTensorConfig

logger = ScreenLogger(__name__)

CPU = torch.device("cpu")
CUDA = torch.device("cuda")


class AutoSmoothQuantProcessor(BaseAlgoProcessor):
    def __init__(self, model: nn.Module, quant_algo_config: Any, data_loader: DataLoader[torch.Tensor]) -> None:
        self.model = model
        # If accelerate is used, the model will have the attribute _hf_hook
        self.using_accelerate = hasattr(self.model, "_hf_hook")
        self.device = model.device
        self.data_loader = data_loader
        self.model_decoder_layers = quant_algo_config.model_decoder_layers
        self.scaling_layers = quant_algo_config.scaling_layers
        self.compute_scale_loss = quant_algo_config.compute_scale_loss
        self.device_map = init_device_map(self.model)
        self.modules, self.module_kwargs, self.inps = self.init_quant()
        self.num_attention_heads, self.num_key_value_heads = get_num_attn_heads_from_model(model)

    def apply(self) -> None:
        self._park_layers_on_cpu()
        for i in tqdm(range(len(self.modules)), desc="Auto-SmoothQuant"):
            with self._layer_on_device(i) as common_device:
                # [STEP 1]: Get layer, extract linear modules, extract input features
                named_linears = get_named_quant_linears(self.modules[i])
                moe_input_layers = get_moe_layers(self.modules[i])
                named_input_layers = self._select_input_hook_targets({**named_linears, **moe_input_layers})
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
            clear_memory()

    def _park_layers_on_cpu(self) -> None:
        """Park every decoder layer on CPU, so each step pulls back only the layer it needs.

        This prevents OOM: the AWQ forward requires extra memory, and takes n batches at once
        rather than batch by batch, which buys several speedups at the expense of device transfer
        time (small enough by comparison) as well as better OOM prevention.

        Does nothing when accelerate manages placement: each layer's hook already owns its device
        and the model already fits, so parking fights that placement -- on a large MoE it can push
        the whole model through host RAM and stall with the GPUs idle.
        """
        if not self.using_accelerate:
            for i in range(len(self.modules)):
                self.modules[i] = self.modules[i].to("cpu")
        clear_memory()

    @contextlib.contextmanager
    def _layer_on_device(self, index: int) -> Iterator[torch.device]:
        """Put layer ``index`` on the device it runs on, and park it back afterwards.

        Does nothing when accelerate manages placement, beyond reporting the device to run on:
        relocating a module accelerate is tracking desyncs its dispatch state. Otherwise the layer
        was parked on CPU by ``_park_layers_on_cpu``, so it is pulled back for the step and parked
        again after it.

        :param int index: Index of the decoder layer in ``self.modules``.

        :return: A context manager yielding the device the layer runs on.
        :rtype: Iterator[torch.device]
        """
        device, pull_on_device = self._layer_device(index)
        if pull_on_device:
            self.modules[index] = self.modules[index].to(device)
        try:
            yield device
        finally:
            if pull_on_device and not self.using_accelerate:
                self.modules[index] = self.modules[index].to("cpu")

    def _layer_device(self, index: int) -> tuple[torch.device, bool]:
        """The device layer ``index`` runs on, and whether the caller must move it there.

        A layer already on a device runs there. One sitting on CPU runs on its ``device_map``
        device -- that device is reported either way, since ``apply_scale`` needs it, but the layer
        is only relocated when accelerate is not the one placing it.

        :param int index: Index of the decoder layer in ``self.modules``.

        :return: The device to run the layer on, and whether the caller must move it there.
        :rtype: tuple[torch.device, bool]
        :raises NotImplementedError: If the layer is offloaded (parameters on ``meta``).
        """
        device = next(self.modules[index].parameters()).device

        # Offloaded layers are `meta` between forwards, and scales are applied outside a forward,
        # so they would be written to placeholder tensors and lost.
        if device.type == "meta":
            raise NotImplementedError(
                f"Auto-SmoothQuant does not support offloaded layers: layer {index} of "
                f"'{self.model_decoder_layers}' has its parameters on the 'meta' device, so the smoothing "
                "scales would be applied to placeholder tensors and silently discarded. Load the model with "
                "a `device_map` that keeps every decoder layer materialized (no CPU/disk offload), or open "
                "an issue."
            )

        if str(device) == "cpu":
            return self.device_map[f"{self.model_decoder_layers}.{index}"], not self.using_accelerate
        return device, False

    def _select_input_hook_targets(self, named_input_layers: dict[str, nn.Linear]) -> dict[str, nn.Linear]:
        """Hook only the modules whose inputs are read back.

        ``get_layers_for_scaling`` only reads ``input_feat`` entries matching ``group["inp"]``;
        hooking every linear copies discarded activations to host, which dominates runtime on a
        large MoE. MoE configs name ``inp`` by suffix (e.g. ``"w2"`` for ``experts.0.w2``), so
        match with the same ``"*" + inp`` rule the consumer uses.
        """
        patterns = ["*" + group["inp"] for group in self.scaling_layers if group.get("inp")]
        if not patterns:
            return named_input_layers
        wanted_inps: set[str] = set()
        for pattern in patterns:
            wanted_inps.update(fnmatch.filter(named_input_layers.keys(), pattern))
        return {name: module for name, module in named_input_layers.items() if name in wanted_inps}

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

        layer_kwargs = resolve_per_layer_kwargs(layer, self.module_kwargs)
        outputs = []
        for in_data in self.inps:
            outputs.append(layer(in_data, **layer_kwargs))
        self.inps = [output[0] if isinstance(output, tuple) else output for output in outputs]

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
            filtered_kwargs = align_attention_mask_with_input(module2inspect, kwargs, inp)
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
            filtered_kwargs = align_attention_mask_with_input(module2inspect, kwargs, x_q)
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

        qspec: QTensorConfig = quantizer.quant_spec
        if qspec.is_ocp_mxfp4() and not get_scale_zp:
            # TODO: `observer` + `scaled_fake_quantize` should be made roughly
            # as fast as this, however there is currently no interface to fuse
            # the two at the moment as `mx.qdq_mxfp4` is doing.
            assert qspec.scale_calculation_mode is not None
            w_q = mx.qdq_mxfp4(w, qspec.scale_calculation_mode)
        else:
            for module in linear_layer.modules():
                if isinstance(module, ScaledFakeQuantize | NonScaledFakeQuantize):
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
                if isinstance(module, ScaledFakeQuantize | NonScaledFakeQuantize):
                    module.disable_observer()
                    module.disable_fake_quant()

        if get_scale_zp:
            linear_layer.weight.data = w_q
        return w_q

    def init_quant(self) -> tuple[nn.ModuleList, dict[str, Any], list[Any]]:
        modules = get_model_layers(self.model, self.model_decoder_layers)
        forward_pass_use_cache = reset_model_kv_cache(self.model, use_cache=False)
        # Parking is only safe because ``cache_model_inps`` pulls layers back on-device, which it
        # refuses to do for accelerate-managed modules. Accelerate has already placed the model.
        if not self.using_accelerate:
            for i in range(len(modules)):
                if i > 0:
                    modules[i] = modules[i].to("cpu")
        clear_memory()
        modules, layer_kwargs, inputs = cache_model_inps(self.model, modules, self.data_loader)
        reset_model_kv_cache(self.model, use_cache=forward_pass_use_cache)
        return modules, layer_kwargs, inputs
