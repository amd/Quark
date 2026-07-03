#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

from quark.torch.export.nn.modules.realquantizer import (
    RealQuantizerBase,
    SequentialRealQuantizer,
    get_real_quantizer,
)
from quark.torch.export.prequantized_config_converter import convert_prequantized_module_to_quark_config
from quark.torch.quantization.config.config import QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.inverse_quantizer import is_compressed_tensors_module, is_prequantized_linear
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.utils import create_pack_method

if TYPE_CHECKING:
    from quark.torch.export.nn.modules.qparamslinear import QParamsLinear


class QParamsLinearBuilder(ABC):
    """Abstract builder that defines the construction protocol for QParamsLinear."""

    def build(self, target: QParamsLinear) -> None:
        """Orchestrate the full construction of *target*.

        Steps executed in order:
        1. resolve_device
        2. build_bias
        3. build_quantizers (creates quantizers and weight parameter)
        4. finalize (optional post-processing, e.g. real quantization)
        """
        device = self.resolve_device()
        target.bias = self.build_bias(device)
        self.build_quantizers(target, device)
        self.finalize(target)

    @abstractmethod
    def resolve_device(self) -> torch.device:
        """Determine the device for tensor allocation."""
        ...

    @abstractmethod
    def build_bias(self, device: torch.device) -> nn.Parameter | None:
        """Construct the bias parameter (or ``None``)."""
        ...

    @abstractmethod
    def build_quantizers(self, target: QParamsLinear, device: torch.device) -> None:
        """Create all quantizers and the weight parameter on *target*."""
        ...

    # -- Optional hook --------------------------------------------------------

    def finalize(self, target: QParamsLinear) -> None:
        """Post-construction hook. Subclasses can override it for extra processing."""
        return None

    # -- Shared helpers -------------------------------------------------------

    def _build_other_quantizers_from_config(
        self,
        target: QParamsLinear,
        quantization_config: QLayerConfig,
        reorder: bool,
        device: torch.device,
        float_dtype: torch.dtype,
    ) -> None:
        """Create bias / input / output quantizers from *quantization_config*."""
        if quantization_config.bias is not None:
            self._create_and_set_quantizer(
                target,
                "bias",
                quantization_config.bias,
                reorder,
                device,
                float_dtype,
                real_quantized=True,
            )

        if quantization_config.input_tensors is not None:
            self._create_and_set_quantizer(
                target,
                "input",
                quantization_config.input_tensors,
                reorder,
                device,
                float_dtype,
                real_quantized=False,
            )

        if quantization_config.output_tensors is not None:
            self._create_and_set_quantizer(
                target,
                "output",
                quantization_config.output_tensors,
                reorder,
                device,
                float_dtype,
                real_quantized=False,
            )

    @staticmethod
    def _create_and_set_quantizer(
        target: QParamsLinear,
        quantizer_name: str,
        spec: QTensorConfig | list[QTensorConfig],
        reorder: bool,
        device: torch.device,
        float_dtype: torch.dtype,
        *,
        real_quantized: bool,
    ) -> None:
        """Validate, create, and assign a single quantizer on *target*."""
        specs_list: list[QTensorConfig] = [spec] if not isinstance(spec, list) else spec
        error_message = (
            f"Reloading a quantized model using QParamsLinear with the {quantizer_name} "
            "static quantized per channel or per group is not supported. "
            "Please open an issue."
        )
        assert all(
            tensor_spec.qscheme == QSchemeType.per_tensor or tensor_spec.is_dynamic for tensor_spec in specs_list
        ), error_message

        quantizer = get_real_quantizer(
            qspec=spec,
            quantizer=None,
            reorder=reorder,
            real_quantized=real_quantized,
            device=device,
            float_dtype=float_dtype,
        )

        if quantizer_name == "bias" and hasattr(quantizer, "transpose_scale"):
            quantizer.transpose_scale = False  # type: ignore[union-attr]

        if quantizer_name == "weight":
            target.weight_quantizer = quantizer
        elif quantizer_name == "bias":
            target.bias_quantizer = quantizer
        elif quantizer_name == "input":
            target.input_quantizer = quantizer
        elif quantizer_name == "output":
            target.output_quantizer = quantizer


class ExportBuilder(QParamsLinearBuilder):
    """Build :class:`QParamsLinear` from a :class:`QuantLinear` during safetensors export."""

    def __init__(self, source: QuantLinear, reorder: bool, custom_mode: str) -> None:
        self._source = source
        self._reorder = reorder
        self._custom_mode = custom_mode

    def resolve_device(self) -> torch.device:
        source = self._source
        if source.is_prequantized or (source.weight is not None and source.weight.device != torch.device("meta")):
            return source.weight.device
        return source._hf_hook.execution_device

    def build_bias(self, device: torch.device) -> nn.Parameter | None:
        source = self._source
        if source.bias is not None:
            if source.bias.device == torch.device("meta"):
                return nn.Parameter(source._hf_hook.weights_map["bias"].data, requires_grad=False)
            return source.bias
        return None

    def build_quantizers(self, target: QParamsLinear, device: torch.device) -> None:
        float_dtype = torch.float32
        source = self._source

        if source.weight_qspec is not None and source.weight_quantizer is not None:
            target.weight_quantizer = get_real_quantizer(
                qspec=source.weight_qspec,
                quantizer=source.weight_quantizer,
                reorder=self._reorder,
                real_quantized=True,
                device=device,
                float_dtype=float_dtype,
            )

        if source.bias_qspec is not None and source.bias_quantizer is not None:
            target.bias_quantizer = get_real_quantizer(
                qspec=source.bias_qspec,
                quantizer=source.bias_quantizer,
                reorder=self._reorder,
                real_quantized=True,
                device=device,
                float_dtype=float_dtype,
            )

        if source.input_qspec is not None and source.input_quantizer is not None:
            target.input_quantizer = get_real_quantizer(
                qspec=source.input_qspec,
                quantizer=source.input_quantizer,
                reorder=self._reorder,
                real_quantized=False,
                device=device,
                float_dtype=float_dtype,
            )

        if source.output_qspec is not None and source.output_quantizer is not None:
            target.output_quantizer = get_real_quantizer(
                qspec=source.output_qspec,
                quantizer=source.output_quantizer,
                reorder=self._reorder,
                real_quantized=False,
                device=device,
                float_dtype=float_dtype,
            )

    def finalize(self, target: QParamsLinear) -> None:
        float_weight = self._get_float_weight()
        self._real_quantize(target, float_weight)

    def _get_float_weight(self) -> torch.Tensor:
        """Get dequantized weight from QuantLinear for export."""
        source = self._source
        if source.is_prequantized:
            return source._weight_quantizer_inv.dequantize(source.weight)  # type: ignore[union-attr]
        if source.weight is not None and source.weight.device != torch.device("meta"):
            return source.weight
        return source._hf_hook.weights_map["weight"].data

    @staticmethod
    def _real_quantize(target: QParamsLinear, float_weight: torch.Tensor | None = None) -> None:
        """Quantize weight and bias on low-bit datatypes, then pack scale/zero_point."""
        # Quantize weight
        if target.weight_quantizer is not None and target.weight_quantizer.is_dynamic is False:
            if float_weight is not None:
                quantized_weight = target.weight_quantizer.to_real_quantize_params(float_weight)
                del float_weight
            else:
                old_weight = target.weight.data
                target.weight = None  # type: ignore[assignment]
                quantized_weight = target.weight_quantizer.to_real_quantize_params(old_weight)
                del old_weight
            target.weight = nn.Parameter(quantized_weight, requires_grad=False)
        elif float_weight is not None:
            target.weight = nn.Parameter(float_weight, requires_grad=False)

        # Quantize bias
        if target.bias is not None and target.bias_quantizer is not None and target.bias_quantizer.is_dynamic is False:
            old_bias = target.bias.data
            target.bias = None  # type: ignore[assignment]
            quantized_bias = target.bias_quantizer.to_real_quantize_params(old_bias)
            del old_bias
            target.bias = nn.Parameter(quantized_bias, requires_grad=False)

        # Pack scale and zero_point for all quantizers
        ExportBuilder._pack_quantizer_info(target)

    @staticmethod
    def _pack_quantizer_info(target: QParamsLinear) -> None:
        """Pack scale and zero_point for all quantizers on *target*."""
        for quantizer in (
            target.weight_quantizer,
            target.bias_quantizer,
            target.input_quantizer,
            target.output_quantizer,
        ):
            if quantizer is not None:
                quantizer.maybe_convert_and_transpose_scale()
                quantizer.pack_zero_point()


class ImportBuilder(QParamsLinearBuilder):
    """Build :class:`QParamsLinear` from ``nn.Linear`` (or pre-quantized subclass) during model import."""

    def __init__(
        self,
        source: nn.Linear,
        reorder: bool,
        custom_mode: str,
        quantization_config: QLayerConfig,
    ) -> None:
        self._source = source
        self._reorder = reorder
        self._custom_mode = custom_mode
        self._quantization_config = quantization_config

    def resolve_device(self) -> torch.device:
        return self._source.weight.device

    def build_bias(self, device: torch.device) -> nn.Parameter | None:
        if self._source.bias is None:
            return None
        return torch.nn.Parameter(
            torch.empty((self._source.out_features,), device=device, dtype=torch.float32),
            requires_grad=False,
        )

    def build_quantizers(self, target: QParamsLinear, device: torch.device) -> None:
        float_dtype = torch.float32
        source = self._source
        quantization_config = self._quantization_config

        if quantization_config.weight is not None:
            weight_configs = (
                [quantization_config.weight]
                if not isinstance(quantization_config.weight, list)
                else quantization_config.weight
            )
            assert all(weight_spec.is_dynamic is not True for weight_spec in weight_configs), (
                "Dynamic quantization is not supported for weight in `QParamsLinear`, "
                "got quantization_config.weight.is_dynamic=True."
            )
            self._build_weight_quantizer_from_config(
                target,
                source.out_features,
                source.in_features,
                quantization_config.weight,
                self._reorder,
                device,
                float_dtype,
            )
        else:
            target.weight = torch.nn.Parameter(
                torch.empty((source.out_features, source.in_features), device=device, dtype=float_dtype),
                requires_grad=False,
            )

        self._build_other_quantizers_from_config(
            target,
            quantization_config,
            self._reorder,
            device,
            float_dtype,
        )

    def _build_weight_quantizer_from_config(
        self,
        target: QParamsLinear,
        out_features: int,
        in_features: int,
        weight_spec: QTensorConfig | list[QTensorConfig],
        reorder: bool,
        device: torch.device,
        float_dtype: torch.dtype,
    ) -> None:
        """Create weight quantizer and weight placeholder from *weight_spec*."""
        weight_specs = [weight_spec] if not isinstance(weight_spec, list) else weight_spec

        weight_shapes: list[tuple[int, ...]] = []
        scale_shapes: list[tuple[int, ...]] = []
        zero_point_shapes: list[tuple[int, ...]] = []
        quantized_torch_dtypes: list[torch.dtype] = []
        last_tensor_quantizer_index = 0

        for spec_index, spec in enumerate(weight_specs):
            if not spec.is_scale_quant:
                last_tensor_quantizer_index = spec_index

            quantized_torch_dtype = spec.dtype.to_torch_packed_dtype()
            pack_method = create_pack_method(
                qscheme=spec.qscheme.value,  # type: ignore[union-attr]
                dtype=spec.dtype.value,
            )

            # For scale quant, the quantized tensor is the scale of the previous quantizer
            unpacked_shape: tuple[int, ...] = (
                (out_features, in_features) if not spec.is_scale_quant else scale_shapes[spec_index - 1]
            )
            weight_shape, scale_shape, zero_point_shape = pack_method.infer_packed_shape(
                unpacked_shape=unpacked_shape,
                quantization_spec=spec,
                legacy=False,
                custom_mode=target._custom_mode,
            )
            weight_shapes.append(weight_shape)
            scale_shapes.append(scale_shape)
            zero_point_shapes.append(zero_point_shape)
            quantized_torch_dtypes.append(quantized_torch_dtype)

        final_weight_shape = weight_shapes[last_tensor_quantizer_index]
        final_quantized_dtype = quantized_torch_dtypes[last_tensor_quantizer_index]
        target.weight = torch.nn.Parameter(
            torch.empty(final_weight_shape, device=device, dtype=final_quantized_dtype),
            requires_grad=False,
        )

        scale_shape_for_quantizer: tuple[int, ...] | list[tuple[int, ...]] = (
            scale_shapes[0] if isinstance(weight_spec, QTensorConfig) else scale_shapes
        )
        zero_point_shape_for_quantizer: tuple[int, ...] | list[tuple[int, ...]] = (
            zero_point_shapes[0] if isinstance(weight_spec, QTensorConfig) else zero_point_shapes
        )
        target.weight_quantizer = get_real_quantizer(
            qspec=weight_spec,
            quantizer=None,
            reorder=reorder,
            real_quantized=True,
            device=device,
            scale_shape=scale_shape_for_quantizer,
            zero_point_shape=zero_point_shape_for_quantizer,
            float_dtype=float_dtype,
        )


class PreserveBuilder(QParamsLinearBuilder):
    """Build :class:`QParamsLinear` from a pre-quantized module (e.g. compressed-tensors, ``FP8Linear``).

    Directly copies the quantized weight and scale tensors into QParamsLinear's
    weight_quantizer structure without dequantize/requantize, preserving bit-exact
    values for non-packed formats.
    """

    def __init__(
        self,
        source: nn.Module,
        reorder: bool,
    ) -> None:
        self._source = source
        self._reorder = reorder
        self._quantization_config = convert_prequantized_module_to_quark_config(source)
        if self._quantization_config is None:
            raise ValueError(
                f"Unsupported prequantized format: {type(source).__name__}. "
                "Cannot derive quantization config from this module."
            )
        self._quantized_weight, self._weight_scale, self._weight_zero_point = self._extract_weight_tensors(
            source, self._quantization_config, self._reorder
        )

    @staticmethod
    def _extract_weight_tensors(
        source: nn.Module,
        quantization_config: QLayerConfig,
        reorder: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Extract weight, scale, and zero_point from *source* in Quark on-disk layout.

        Two storage shapes are recognised:

        - **Non-packed** (FP8Linear, compressed-tensors FP8 / INT8 per_tensor or
          per_channel): tensors are read straight from ``source.weight`` /
          ``weight_scale`` / ``weight_zero_point`` and used as-is.
        - **Packed** (compressed-tensors W4A16 with ``weight_packed`` int32 storage):
          unpack to int8, repack into Quark's ``[in, out/8]`` int32 layout, transpose
          the per-group scale, and pack the per-group zero_point. Other packed
          combinations raise :class:`ValueError` so the handler falls back to
          dequantize.
        """
        if is_compressed_tensors_module(source) and getattr(source, "weight_packed", None) is not None:
            return PreserveBuilder._extract_packed_weight_tensors(source, quantization_config, reorder)

        weight = getattr(source, "weight", None)
        if weight is None:
            raise ValueError(
                f"Cannot extract quantized weight from {type(source).__name__}: no weight or weight_packed attribute."
            )

        weight_scale_inv = getattr(source, "weight_scale_inv", None)
        weight_scale_attribute = getattr(source, "weight_scale", None)
        weight_scale = weight_scale_inv if weight_scale_inv is not None else weight_scale_attribute
        if weight_scale is None:
            raise ValueError(
                f"Cannot extract weight scale from {type(source).__name__}. "
                "Expected weight_scale or weight_scale_inv attribute."
            )

        weight_zero_point = getattr(source, "weight_zero_point", None)
        return (
            weight.detach(),
            weight_scale.detach(),
            weight_zero_point.detach() if weight_zero_point is not None else None,
        )

    @staticmethod
    def _extract_packed_weight_tensors(
        source: nn.Module,
        quantization_config: QLayerConfig,
        reorder: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Convert compressed-tensors packed W4A16 storage to Quark's packed layout.

        Compressed-tensors stores ``weight_packed`` as int32 ``[out, in/8]`` with
        packed_dim=1 (8 little-endian int4 values per int32, +8 unsigned offset).
        Quark stores the same int values as int32 ``[in, out/8]`` with the
        ``[0,2,4,6,1,3,5,7]`` reorder applied along the packed (``out``) axis.
        Both layouts dequantize to the same matrix; only the byte arrangement
        differs.
        """
        weight_config = quantization_config.weight
        if weight_config is None:
            raise ValueError("Packed weight extraction requires a weight QTensorConfig.")
        if isinstance(weight_config, list):
            raise ValueError("Packed weight extraction does not support list-typed weight configs.")
        if not (
            weight_config.qscheme == QSchemeType.per_group
            and weight_config.dtype == Dtype.int4
            and weight_config.group_size is not None
        ):
            raise ValueError(
                f"Packed weight extraction is only implemented for int4 per_group "
                f"(W4A16); got dtype={weight_config.dtype}, qscheme={weight_config.qscheme}."
            )

        # Local import: compressed_tensors is an optional dep guarded at the
        # call sites that may reach this branch.
        try:
            from compressed_tensors.compressors.pack_quantized import unpack_from_int32
        except ImportError:
            from compressed_tensors.compressors.quantized_compressors.pack_quantized import unpack_from_int32

        weight_packed = source.weight_packed
        weight_scale = getattr(source, "weight_scale", None)
        if weight_scale is None:
            raise ValueError("Packed compressed-tensors weight is missing weight_scale.")

        weight_shape = getattr(source, "weight_shape", None)
        if weight_shape is None:
            raise ValueError("Packed compressed-tensors weight is missing weight_shape.")
        out_features, in_features = int(weight_shape[0]), int(weight_shape[1])

        # Step 1: unpack to [out, in] in [-8, 7]. unpack_from_int32 returns
        # int8, but Pack_4_bits.pack performs left-shifts up to 28 bits and
        # silently truncates int8 inputs, so cast to int32.
        unpacked_weight = unpack_from_int32(
            weight_packed.detach(), num_bits=4, shape=torch.Size([out_features, in_features]), packed_dim=1
        ).to(torch.int32)

        # Step 2: repack into Quark layout [in, out/8] using Pack_4_bits, which
        # internally transposes [out, in] -> [in, out] for per_group and packs
        # 8 int4 values per int32 along the new last axis.
        pack_method = create_pack_method(qscheme="per_group", dtype="int4")
        repacked_weight = pack_method.pack(unpacked_weight, reorder=reorder)

        # Step 3: scale [out, in/group] -> [in/group, out]
        repacked_scale = weight_scale.detach().t().contiguous()

        # Step 4: zero_point. Compressed-tensors stores it packed along dim=0
        # with shape [ceil(out/8), in/group] when present; unpack -> [out, in/group]
        # in [-8, 7], then pack via Pack_4_bits to get Quark's [in/group, out/8].
        # For symmetric W4A16 the checkpoint omits weight_zero_point entirely,
        # but realquantizer.unpack_params still calls pack_method.unpack on the
        # zero_point buffer at forward time — so we must hand it a properly
        # shaped zero tensor, otherwise the buffer defaults to 0-D and unpack
        # crashes with IndexError at pack.py.
        weight_zero_point = getattr(source, "weight_zero_point", None)
        if weight_zero_point is not None:
            zp_unpacked_shape = torch.Size([out_features, int(weight_scale.shape[-1])])
            unpacked_zp = unpack_from_int32(
                weight_zero_point.detach(), num_bits=4, shape=zp_unpacked_shape, packed_dim=0
            ).to(torch.int32)
            repacked_zp = pack_method.pack(unpacked_zp, reorder=reorder)
        else:
            zp_shape = (int(weight_scale.shape[-1]), out_features // pack_method.qparams_per_item)
            repacked_zp = torch.zeros(zp_shape, dtype=torch.int32, device=weight_packed.device)

        return repacked_weight, repacked_scale, repacked_zp

    def resolve_device(self) -> torch.device:
        return self._quantized_weight.device

    def build_bias(self, device: torch.device) -> nn.Parameter | None:
        if self._source.bias is not None:
            return nn.Parameter(self._source.bias.detach(), requires_grad=False)
        return None

    def build_quantizers(self, target: QParamsLinear, device: torch.device) -> None:
        target.weight = nn.Parameter(self._quantized_weight, requires_grad=False)

        assert self._quantization_config is not None
        weight_config = self._quantization_config.weight
        if weight_config is None:
            return

        weight_quantizer = get_real_quantizer(
            qspec=weight_config,
            quantizer=None,
            reorder=self._reorder,
            real_quantized=True,
            device=device,
            scale_shape=tuple(self._weight_scale.shape),
            zero_point_shape=(tuple(self._weight_zero_point.shape) if self._weight_zero_point is not None else None),
            float_dtype=torch.float32,
        )

        if isinstance(weight_quantizer, SequentialRealQuantizer):
            raise ValueError(
                "SequentialRealQuantizer (e.g. two-stage FP4+FP8) is not supported "
                "for prequantized layer preservation. Only single-stage quantization "
                "configs (per_tensor, per_channel, per_group, per_block) are supported."
            )

        self._copy_scale_and_zero_point(weight_quantizer, self._weight_scale, self._weight_zero_point)
        target.weight_quantizer = weight_quantizer

        # Build input/output/bias quantizers from the QLayerConfig so the preserve
        # path matches the reload path — otherwise reload's input_quantizer is
        # active while preserve's is None, causing PPL drift.
        self._build_other_quantizers_from_config(
            target,
            self._quantization_config,
            self._reorder,
            device,
            torch.float32,
        )

    @staticmethod
    def _copy_scale_and_zero_point(
        weight_quantizer: RealQuantizerBase,
        weight_scale: torch.Tensor,
        weight_zero_point: torch.Tensor | None,
    ) -> None:
        """Copy scale and zero_point tensors into the quantizer's buffers."""
        if not hasattr(weight_quantizer, "scale"):
            return

        weight_quantizer.scale.data.copy_(weight_scale)
        if weight_zero_point is not None and hasattr(weight_quantizer, "zero_point"):
            weight_quantizer.zero_point.data.copy_(weight_zero_point)

    def finalize(self, target: QParamsLinear) -> None:
        target._quant_config = self._quantization_config


def create_builder(
    source: nn.Module,
    reorder: bool,
    custom_mode: str,
    quantization_config: QLayerConfig | None = None,
) -> QParamsLinearBuilder:
    """Select and create the appropriate builder for the given *source* module.

    Args:
        source: The source module (QuantLinear for export, nn.Linear or
            pre-quantized subclass for import/preserve).
        reorder: Whether to reorder parameters (derived from pack_method).
        custom_mode: Custom export mode string (e.g. ``"awq"``).
        quantization_config: Required for import/preserve path; must be ``None`` for export.

    Returns:
        A :class:`QParamsLinearBuilder` instance ready to call :meth:`build`.

    Raises:
        ValueError: If no builder matches the given combination of source type
            and quantization_config.
    """
    if isinstance(source, QuantLinear) and quantization_config is None:
        return ExportBuilder(source, reorder, custom_mode)
    if is_prequantized_linear(source):
        # Upstream invariant: a prequantized module only survives to this dispatch
        # when the preserve decision has already been made — at export time by
        # `ModelPostProcessor` (keep_prequantized_layers=True path), at reload time
        # by the per-layer scan in `_build_quantized_model` (saved config matches
        # native format). Otherwise upstream has already replaced it with nn.Linear.
        # So no flag check here; type alone encodes the decision.
        #
        # PreserveBuilder derives its qparams from the source module itself, which
        # is correct in both cases — the source carries the authoritative quantized
        # weights/scales, and ImportBuilder would try to re-pack them and fail on
        # schemes it doesn't support (e.g. FP8 per_block).
        return PreserveBuilder(source, reorder)
    if isinstance(source, nn.Linear) and quantization_config is not None:
        return ImportBuilder(source, reorder, custom_mode, quantization_config)
    raise ValueError(
        f"No builder available for source type {type(source).__name__} "
        f"with quantization_config={'provided' if quantization_config is not None else 'None'}. "
        "Expected QuantLinear (without config) for export, "
        "pre-quantized linear (without config) for preserve, "
        "or nn.Linear (with config) for import."
    )
