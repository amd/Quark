#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#


from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from tqdm import tqdm

from quark.common.utils.log import ScreenLogger
from quark.torch.export.config.config import JsonExporterConfig

if TYPE_CHECKING:
    from quark.torch.quantization.config.config import QConfig
from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
from quark.torch.export.prequantized_layer_handler import (
    dequantize_prequantized_linears,
    preserve_prequantized_layers,
)
from quark.torch.export.utils import find_patterns_groups
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.tensor_quantize import SequentialQuantize
from quark.torch.utils import getattr_recursive, setattr_recursive

logger = ScreenLogger(__name__)


class ModelPostProcessor:
    def __init__(
        self,
        model: nn.Module,
        export_config: JsonExporterConfig,
        custom_mode: str,
        output_quant: bool,
        quantization_config: QConfig | None = None,
    ) -> None:
        self._model = model
        self._config = export_config
        self.custom_mode = custom_mode
        self.output_quant = output_quant
        self._quantization_config = quantization_config

    @property
    def model(self) -> nn.Module:
        return self._model

    def merge_scale(self) -> None:
        """
        call _virtual_merge_weight_matrix and _merge_kv_scale to merge scale and build k_scale, v_scale if export autofp8 format.
        """
        if self._config.weight_merge_groups and len(self._config.weight_merge_groups) > 0:
            logger.info(
                f"Merging scale for weight matrices in the groups weight_merge_groups={self._config.weight_merge_groups}."
            )
            self._virtual_merge_weight_matrix()
        if (self._config.kv_cache_group is not None) and (len(self._config.kv_cache_group) > 0):
            logger.info(f"Merging scales for the kv cache, kv_cache_group={self._config.kv_cache_group}.")
            self._merge_kv_scale()

    def get_processed_model(self) -> nn.Module:
        """
        Converts the model's module (e.g. `QuantLinear`) according to the weight_format.

        If `weight_format=real_quantized"` is used, relevant modules will be replaced by modules handling low-precision data types, as `QParamsLinear`.

        If `weight_format=fake_quantized"` high precision parameters and original modules are kept.

        Pre-quantized linears (FP8Linear, compressed-tensors quantized linear) that were NOT quantized by Quark
        (e.g., in ignore list) will be dequantized to nn.Linear for export compatibility.
        """
        logger.info("Model post process start.")
        logger.info("Simplifying quantized operators...")

        if getattr(self._quantization_config, "keep_prequantized_layers", False):
            preserve_prequantized_layers(
                model=self._model,
                custom_mode=self.custom_mode,
                pack_method=self._config.pack_method,
                quantization_config=self._quantization_config,
            )
        else:
            dequantize_prequantized_linears(self._model)

        def _release_module_tensors(module: QuantLinear) -> None:
            """Release heavyweight tensor references from an exported QuantLinear."""
            # Weight tensors
            if hasattr(module, "weight") and module.weight is not None:
                module.weight = None
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = None
            # Quantizers (hold scale, zero_point tensors)
            if hasattr(module, "_weight_quantizer_inv"):
                module._weight_quantizer_inv = None
            if hasattr(module, "_weight_quantizer"):
                module._weight_quantizer = None
            if hasattr(module, "_input_quantizer"):
                module._input_quantizer = None
            if hasattr(module, "_output_quantizer"):
                module._output_quantizer = None
            if hasattr(module, "_bias_quantizer"):
                module._bias_quantizer = None

        if self._config.weight_format == "real_quantized":
            logger.info("Real_quantized: Doing real quantization for operators...")
            # Keep only module names so replaced QuantLinear instances are not
            # pinned in memory by a snapshot of `named_modules`.
            quantlinear_names = [
                name
                for name, module in self._model.named_modules(remove_duplicate=False)
                if isinstance(module, QuantLinear)
            ]
            processed_count = 0
            for name in tqdm(quantlinear_names, desc="Converting QuantLinear for export", total=len(quantlinear_names)):
                module = getattr_recursive(self._model, name)
                if not isinstance(module, QuantLinear):
                    continue

                # In export flow, we need to modify the state_dict format, so we add the "export_enabled" flag to control the flow.
                module.register_buffer("export_enabled", torch.tensor([1], dtype=torch.uint8), persistent=False)
                # Perform real quantization and packing on the original device.
                export_linear = QParamsLinear.from_module(
                    module,
                    self.custom_mode,
                    self._config.pack_method,
                )
                setattr_recursive(self._model, name, export_linear)
                _release_module_tensors(module)
                processed_count += 1

                # Drop local references so the old QuantLinear can be reclaimed
                # immediately after replacement.
                del module
                del export_linear

                # Periodically collect Python cycles and return freed blocks to
                # the CUDA allocator.
                if processed_count % 10 == 0 and torch.cuda.is_available():
                    # gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            if torch.cuda.is_available():
                # gc.collect()
                torch.cuda.empty_cache()
        elif self._config.weight_format == "fake_quantized":
            logger.info("Fake_quantized: save float_w, scale and zero_point for operators...")
            named_modules = dict(self._model.named_modules(remove_duplicate=False))
            for name, module in tqdm(named_modules.items()):
                if isinstance(module, QuantLinear):
                    # In export flow, we need to modify the state_dict format, so we add the "export_enabled" flag to control the flow.
                    module.register_buffer("export_enabled", torch.tensor([1], dtype=torch.uint8), persistent=False)
        named_modules = dict(self._model.named_modules(remove_duplicate=False))
        has_dbrx_experts = any(module.__class__.__name__ == "DbrxExperts_" for module in named_modules.values())
        if has_dbrx_experts:
            self._merge_params_for_DbrxExperts()

        logger.info("Model post process end")
        return self._model

    def _virtual_merge_weight_matrix(self) -> None:
        """
        Select the maximum scale value in weight_group as the scale for each module in the group.
        """
        named_modules = dict(self._model.named_modules(remove_duplicate=False))
        names = list(named_modules.keys())
        merge_groups = find_patterns_groups(self._config.weight_merge_groups, names)

        for merge_group in merge_groups:
            module_merge_group = [getattr_recursive(self._model, name) for name in merge_group]
            self._merge_scaling_factor(module_merge_group)

    def _merge_scaling_factor(self, module_group: list[nn.Module]) -> None:
        weight_quant_or_not = all(getattr(module, "_weight_quantizer", None) is not None for module in module_group)
        if not weight_quant_or_not:
            return

        weight_sequential_quant = any(
            isinstance(module._weight_quantizer, SequentialQuantize) for module in module_group
        )
        if weight_sequential_quant:
            logger.warning(
                "SequentialQuantize is not supported for weight merge, please raise an issue if you need this feature."
            )
            return

        static_quant_or_not = all(module._weight_quantizer.quant_spec.is_dynamic is False for module in module_group)
        if not static_quant_or_not:
            return
        per_tensor_or_not = all(
            module._weight_quantizer.quant_spec.qscheme.name == "per_tensor" for module in module_group
        )
        if not per_tensor_or_not:
            return
        zero_point_or_not = all(
            (hasattr(module._weight_quantizer, "zero_point") and module._weight_quantizer.zero_point is not None)
            for module in module_group
        )
        if zero_point_or_not:
            symmetric_quant_or_not = all(torch.all(module._weight_quantizer.zero_point == 0) for module in module_group)
            if not symmetric_quant_or_not:
                return

        weight_scale_list = [module._weight_quantizer.scale for module in module_group]

        group_weight_scale = (
            torch.stack(
                weight_scale_list,
            )
            .max(dim=0)
            .values
        )

        for module in module_group:
            module._weight_quantizer.scale.data = group_weight_scale.clone()

        output_quant_or_not = all(getattr(module, "output_quantizer", None) is not None for module in module_group)
        if not output_quant_or_not:
            return

        output_sequential_quant = any(
            isinstance(module.output_quantizer, SequentialQuantize) for module in module_group
        )
        if output_sequential_quant:
            logger.warning(
                "SequentialQuantize is not supported for weight merge, please raise an issue if you need this feature."
            )
            return
        static_quant_or_not = all(module.output_quantizer.quant_spec.is_dynamic is False for module in module_group)
        if not static_quant_or_not:
            return
        per_tensor_or_not = all(
            module.output_quantizer.quant_spec.qscheme.name == "per_tensor" for module in module_group
        )
        if not per_tensor_or_not:
            return
        zero_point_or_not = all(
            (hasattr(module.output_quantizer, "zero_point") and module.output_quantizer.zero_point is not None)
            for module in module_group
        )
        if zero_point_or_not:
            symmetric_quant_or_not = all(torch.all(module.output_quantizer.zero_point == 0) for module in module_group)
            # that is all now
            if not symmetric_quant_or_not:
                return

        output_scale_list = [module.output_quantizer.scale for module in module_group]

        group_output_scale = (
            torch.stack(
                output_scale_list,
            )
            .max(dim=0)
            .values
        )

        for module in module_group:
            module.output_quantizer.scale.data = group_output_scale.clone()

    def _merge_kv_scale(self) -> None:
        """
        Select the maximum kv_scale value in kv_group as the scale for kv, and replace `attn.k_proj.out_scale` by `attn.k_scale` and `attn.v_proj.out_scale` by `attn.v_scale`.
        """
        named_modules = dict(self._model.named_modules(remove_duplicate=False))
        names = list(named_modules.keys())
        kv_groups = find_patterns_groups([self._config.kv_cache_group], names)
        for kv_group in kv_groups:
            kv_modules = [getattr_recursive(self._model, name) for name in kv_group]
            self._build_kv_scale(kv_group, kv_modules, self._config.min_kv_scale)

    def _build_kv_scale(
        self, module_names: list[str], module_group: list[nn.Module], min_kv_scale: float = 0.0
    ) -> None:
        output_quant_or_not = all(getattr(module, "output_quantizer", None) is not None for module in module_group)
        if not output_quant_or_not:
            return

        # currently, sequential quantizer is not supported for kv cache
        output_sequential_quant = any(
            isinstance(module.output_quantizer, SequentialQuantize) for module in module_group
        )
        if output_sequential_quant:
            logger.warning(
                "SequentialQuantize is not supported for kv cache, please raise an issue if you need this feature."
            )
            return

        static_quant_or_not = all(module.output_quantizer.quant_spec.is_dynamic is False for module in module_group)
        if not static_quant_or_not:
            return
        per_tensor_or_not = all(
            module.output_quantizer.quant_spec.qscheme.name == "per_tensor" for module in module_group
        )
        if not per_tensor_or_not:
            return
        zero_point_or_not = all(
            (hasattr(module.output_quantizer, "zero_point") and module.output_quantizer.zero_point is not None)
            for module in module_group
        )
        if zero_point_or_not:
            symmetric_quant_or_not = all(torch.all(module.output_quantizer.zero_point == 0) for module in module_group)
            if not symmetric_quant_or_not:
                return

        output_scale_list = [module.output_quantizer.scale for module in module_group]
        kv_scale = (
            torch.stack(
                output_scale_list,
            )
            .max(dim=0)
            .values
        )

        parent_module_name = ".".join(module_names[0].split(".")[:-1])
        # The custom_mode used along kv_cache_group must be "quark" or "fp8".
        if self.custom_mode == "fp8":
            if not self.output_quant:
                for module in module_group:
                    module._output_quantizer = None  # type: ignore
            parent_module = getattr_recursive(self._model, parent_module_name)
            parent_module.k_scale = torch.nn.Parameter(
                torch.max(kv_scale.clone(), torch.tensor(min_kv_scale, dtype=kv_scale.dtype, device=kv_scale.device))
            )
            parent_module.v_scale = torch.nn.Parameter(
                torch.max(kv_scale.clone(), torch.tensor(min_kv_scale, dtype=kv_scale.dtype, device=kv_scale.device))
            )
        # for quark format, we keep output_scale, even if there is kv_cache
        else:
            if not self.output_quant:
                for module in module_group:
                    if isinstance(module, QuantLinear):
                        if module.output_quantizer is not None and hasattr(module.output_quantizer, "scale"):
                            module.output_quantizer.scale.data = torch.max(
                                kv_scale.clone(),
                                torch.tensor(min_kv_scale, dtype=kv_scale.dtype, device=kv_scale.device),
                            )

        if min_kv_scale > kv_scale:
            logger.warning(f"Increase {parent_module_name} kv cache scaling factor to a minimum of {min_kv_scale}.")

    def _merge_params_for_DbrxExperts(self) -> None:
        named_modules = dict(self._model.named_modules(remove_duplicate=False))
        for name, module in tqdm(named_modules.items()):
            if module.__class__.__name__ == "DbrxExperts_":
                export_experts = torch.nn.Module()
                export_experts.mlp = torch.nn.Module()

                w1_weight_tensors = [expert.w1.weight for expert in module.mlp]
                w1_weight_concat = torch.cat(w1_weight_tensors)
                export_experts.mlp.w1_weight = torch.nn.Parameter(w1_weight_concat, requires_grad=False)

                if module.mlp[0].w1.weight_quantizer is not None:
                    w1_weight_scale_tensors = [expert.w1.weight_quantizer.scale for expert in module.mlp]
                    w1_weight_scale_concat = torch.stack(w1_weight_scale_tensors)
                    export_experts.mlp.w1_weight_scale = torch.nn.Parameter(w1_weight_scale_concat, requires_grad=False)

                if module.mlp[0].w1.input_quantizer is not None:
                    w1_input_scale_tensors = [expert.w1.input_quantizer.scale for expert in module.mlp]
                    w1_input_scale_concat = torch.stack(w1_input_scale_tensors)
                    export_experts.mlp.w1_input_scale = torch.nn.Parameter(w1_input_scale_concat, requires_grad=False)

                if module.mlp[0].w1.output_quantizer is not None:
                    w1_output_scale_tensors = [expert.w1.output_quantizer.scale for expert in module.mlp]
                    w1_output_scale_concat = torch.stack(w1_output_scale_tensors)
                    export_experts.mlp.w1_output_scale = torch.nn.Parameter(w1_output_scale_concat, requires_grad=False)

                v1_weight_tensors = [expert.v1.weight for expert in module.mlp]
                v1_weight_concat = torch.cat(v1_weight_tensors)
                export_experts.mlp.v1_weight = torch.nn.Parameter(v1_weight_concat, requires_grad=False)

                if module.mlp[0].v1.weight_quantizer is not None:
                    v1_weight_scale_tensors = [expert.v1.weight_quantizer.scale for expert in module.mlp]
                    v1_weight_scale_concat = torch.stack(v1_weight_scale_tensors)
                    export_experts.mlp.v1_weight_scale = torch.nn.Parameter(v1_weight_scale_concat, requires_grad=False)

                if module.mlp[0].v1.input_quantizer is not None:
                    v1_input_scale_tensors = [expert.v1.input_quantizer.scale for expert in module.mlp]
                    v1_input_scale_concat = torch.stack(v1_input_scale_tensors)
                    export_experts.mlp.v1_input_scale = torch.nn.Parameter(v1_input_scale_concat, requires_grad=False)

                if module.mlp[0].v1.output_quantizer is not None:
                    v1_output_scale_tensors = [expert.v1.output_quantizer.scale for expert in module.mlp]
                    v1_output_scale_concat = torch.stack(v1_output_scale_tensors)
                    export_experts.mlp.v1_output_scale = torch.nn.Parameter(v1_output_scale_concat, requires_grad=False)

                # transpose w2.weight back when exporting dbrx model
                w2_weight_tensors = [expert.w2.weight.t() for expert in module.mlp]
                w2_weight_concat = torch.cat(w2_weight_tensors)
                export_experts.mlp.w2_weight = torch.nn.Parameter(w2_weight_concat, requires_grad=False)

                if module.mlp[0].w2.weight_quantizer is not None:
                    w2_weight_scale_tensors = [expert.w2.weight_quantizer.scale for expert in module.mlp]
                    w2_weight_scale_concat = torch.stack(w2_weight_scale_tensors)
                    export_experts.mlp.w2_weight_scale = torch.nn.Parameter(w2_weight_scale_concat, requires_grad=False)

                if module.mlp[0].w2.input_quantizer is not None:
                    w2_input_scale_tensors = [expert.w2.input_quantizer.scale for expert in module.mlp]
                    w2_input_scale_concat = torch.stack(w2_input_scale_tensors)
                    export_experts.mlp.w2_input_scale = torch.nn.Parameter(w2_input_scale_concat, requires_grad=False)

                if module.mlp[0].w2.output_quantizer is not None:
                    w2_output_scale_tensors = [expert.w2.output_quantizer.scale for expert in module.mlp]
                    w2_output_scale_concat = torch.stack(w2_output_scale_tensors)
                    export_experts.mlp.w2_output_scale = torch.nn.Parameter(w2_output_scale_concat, requires_grad=False)
                setattr_recursive(self._model, name, export_experts)
