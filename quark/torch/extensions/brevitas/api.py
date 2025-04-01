#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quantization API for Brevitas."""
from __future__ import annotations

import torch
import torch.utils
import torch.utils.data

import quark.torch.quantization.config.type as quark_config_type
import quark.torch.extensions.brevitas.config as brevitas_config
import quark.torch.extensions.brevitas.algos as brevitas_algos
import quark.torch.extensions.brevitas.verification as brevitas_verify

try:
    import brevitas.nn  # type: ignore[import-not-found]
    import brevitas.core.scaling  # type: ignore[import-not-found]
    import brevitas.core.zero_point  # type: ignore[import-not-found]
    import brevitas.quant  # type: ignore[import-not-found]
    import brevitas.graph.quantize  # type: ignore[import-not-found]
    import brevitas.quant.scaled_int  # type: ignore[import-not-found]
    import brevitas.export  # type: ignore[import-not-found]
    import quark.torch.extensions.brevitas.mapping as brevitas_mapping
    has_brevitas = True
except ModuleNotFoundError:
    has_brevitas = False

from typing import Any, Optional, Union, Dict, Tuple

from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

# remove this when the API stabilises
logger.warning(
    "You are using the Brevitas-Quark API which is experimental and may be subject to change. Production code should not rely on it."
)

from importlib.metadata import version
from packaging.version import Version

# update this version to whatever brevitas version we expect
expected_brevitas_version = Version("0.10.3")

# check for brevitas and its version, guide the end user if its not installed
if has_brevitas:
    brevitas_version = Version(version('brevitas'))
    if brevitas_version != expected_brevitas_version:
        logger.warning(
            f"This API expects Brevitas version {expected_brevitas_version} but {brevitas_version} is installed, there may be issues."
        )
else:
    logger.warning(f"Brevitas is not installed, you should run: pip install brevitas=={expected_brevitas_version}")

__all__ = ["ModelQuantizer", "ModelExporter"]


class ModelQuantizer:
    """
    Provides an API for quantizing deep learning models using Brevitas.

    The way this class interacts with Brevitas is based on the brevitas ptq example found here:
    https://github.com/Xilinx/brevitas/tree/master/src/brevitas_examples/imagenet_classification/ptq

    Example usage:
        weight_spec = QuantizationSpec()
        global_config = QuantizationConfig(weight=weight_spec)
        config = Config(global_quant_config=global_config)
        quantizer = ModelQuantizer(config)
        quant_model = quantizer.quantize_model(model, calib_dataloader)
    """

    @classmethod
    def _parse_bias_quant_spec(
            cls, spec: Union[brevitas_config.QuantizationSpec, None]) -> Union[brevitas.quant.scaled_int.IntBias, None]:
        if spec is None:
            return None

        return brevitas_mapping.BIAS_QUANT_MAP[spec.bit_width]

    @classmethod
    def _parse_activation_quant_spec(cls,
                                     spec: Union[brevitas_config.QuantizationSpec, None],
                                     force_symmetric: bool = False,
                                     force_per_tensor: bool = False) -> Any:
        if spec is None:
            return None

        # `high_percentile_q = 99.999` is Brevitas default.
        params: Dict[str, Any] = {'high_percentile_q': 99.999, 'bit_width': spec.bit_width, "narrow_range": False}

        if spec.quant_type is brevitas_config.QuantType.float_quant:
            params['exponent_bit_width'] = spec.exponent_bit_width
            params['mantissa_bit_width'] = spec.mantissa_bit_width

        if spec.symmetric is False:
            params['low_percentile_q'] = 100.0 - params['high_percentile_q']

        qscheme = spec.qscheme if force_per_tensor is False else quark_config_type.QSchemeType.per_tensor
        sym_type = 'sym' if spec.symmetric is True or force_symmetric is True else 'asym'

        quant = brevitas_mapping.INPUT_QUANT_MAP[spec.quant_type][spec.scale_type][spec.param_type][qscheme][sym_type]

        quant = quant.let(**params)

        return quant

    @classmethod
    def _parse_weight_quant_spec(cls, spec: Union[brevitas_config.QuantizationSpec, None]) -> Any:
        if spec is None:
            return None

        # By default, Brevitas Int8WeightPerTensorFloat (https://github.com/Xilinx/brevitas/blob/v0.10.3/src/brevitas/quant/scaled_int.py#L159) uses `narrow_range=True`, while Int8ActPerTensorFloat uses `narrow_range=False`. quark.torch does not use narrow range (as of 0.5.0).
        params = {
            'bit_width': spec.bit_width,
            'narrow_range': True,
            'scaling_impl': brevitas.core.scaling.standalone.ParameterFromStatsFromParameterScaling
        }

        if spec.exponent_bit_width is not None:
            params['exponent_bit_width'] = spec.exponent_bit_width

        if spec.mantissa_bit_width is not None:
            params['mantissa_bit_width'] = spec.mantissa_bit_width

        if spec.symmetric is False:
            params['zero_point_impl'] = brevitas.core.zero_point.ParameterFromStatsFromParameterZeroPoint

        quant = brevitas_mapping.WEIGHT_QUANT_MAP[spec.quant_type][spec.scale_type][spec.param_type][
            spec.qscheme]['sym' if spec.symmetric else 'asym']

        quant = quant.let(**params)

        return quant

    def __init__(self, config: brevitas_config.Config) -> None:
        if has_brevitas is False:
            raise RuntimeError("Brevitas needs to be installed to use this ModelQuantizer!")

        self.config = config

        # Add new quantize methods here as more backends are added
        # To keep it simple, the initial version will just have layerwise quantization
        self.quantize_fn = {
            brevitas_config.Backend.layerwise: brevitas.graph.quantize.layerwise_quantize
        }[self.config.backend]

        brevitas_verify.ConfigVerifier.verify_config(config)

    @classmethod
    def _create_layer_map(cls, config: brevitas_config.Config, device: str) -> Any:
        common_kwargs = {'device': device, 'return_quant_tensor': False, 'dtype': torch.float}

        bias_quant = cls._parse_bias_quant_spec(config.global_quant_config.bias)
        weight_quant = cls._parse_weight_quant_spec(config.global_quant_config.weight)

        # for some parts of multihead attention the activation quantization MUST be symmetric
        sym_act_quant = cls._parse_activation_quant_spec(config.global_quant_config.input_tensors, True)

        kwargs = {**common_kwargs, 'bias_quant': bias_quant, 'weight_quant': weight_quant}

        mha_kwargs = {
            **common_kwargs, 'in_proj_weight_quant': weight_quant,
            'in_proj_input_quant': None,
            'in_proj_bias_quant': bias_quant,
            'packed_in_proj': True,
            'out_proj_bias_quant': bias_quant,
            'softmax_input_quant': None,
            'out_proj_weight_quant': weight_quant,
            'out_proj_input_quant': cls._parse_activation_quant_spec(config.global_quant_config.input_tensors),
            'out_proj_bias_quant': bias_quant,
            'out_proj_output_quant': None,
            'attn_output_weights_quant': sym_act_quant,
            'q_scaled_quant': sym_act_quant,
            'k_transposed_quant': sym_act_quant,
            'v_quant': sym_act_quant
        }

        per_tensor_act_quant = cls._parse_activation_quant_spec(config.global_quant_config.input_tensors,
                                                                force_per_tensor=True)

        kwargs['input_quant'] = per_tensor_act_quant
        kwargs['output_quant'] = cls._parse_activation_quant_spec(config.global_quant_config.output_tensors)

        mha_kwargs['in_proj_input_quant'] = per_tensor_act_quant

        # todo other layer types
        layer_map = {
            torch.nn.Linear: (brevitas.nn.QuantLinear, kwargs),
            torch.nn.Conv1d: (brevitas.nn.QuantConv1d, kwargs),
            torch.nn.Conv2d: (brevitas.nn.QuantConv2d, kwargs),
            torch.nn.ConvTranspose1d: (brevitas.nn.QuantConvTranspose1d, kwargs),
            torch.nn.ConvTranspose2d: (brevitas.nn.QuantConvTranspose2d, kwargs),
            torch.nn.MultiheadAttention: (brevitas.nn.QuantMultiheadAttention, mha_kwargs)
        }

        return layer_map

    def quantize_model(
            self,
            model: torch.nn.Module,
            calib_loader: Optional[torch.utils.data.DataLoader] = None) -> torch.nn.Module:  # type: ignore[type-arg]
        """
        Quantizes the given model.

        - `model`: The model to be quantized.
        - `calib_loader`: A dataloader for calibration data, technically optional but required for most quantization processes.
        """

        # at present all brevitas backends use the layermap
        device = next(model.parameters()).device
        self.quantize_params = {'compute_layer_map': self._create_layer_map(self.config, str(device))}

        # put here any params needed by other backends when added

        # apply pre quantization optimizations
        for x in self.config.pre_quant_opt_config:
            logger.info(f"Applying {x.name}.")
            model = x.apply(model, calib_loader=calib_loader)

        logger.info("Applying Quantization.")
        quant_model = self.quantize_fn(model, **self.quantize_params)

        if calib_loader is not None:
            logger.info("Applying Calibration.")
            quant_model = brevitas_algos._calibrate(calib_loader, quant_model)

        # apply post quantization algorithms
        for x in self.config.algo_config:
            logger.info(f"Applying {x.name}.")
            quant_model = x.apply(quant_model, calib_loader=calib_loader)

        return quant_model  # type: ignore[no-any-return]


class ModelExporter:
    """
    Provides an API for exporting pytorch models quantized with Brevitas.
    This class converts the quantized model to an onnx graph, and saves it to the specified export_path.

    Example usage:
        exporter = ModelExporter("model.onnx")
        exporter.export_onnx_model(quant_model, args=torch.ones(1, 1, 784))
    """

    def __init__(self, export_path: str) -> None:
        self.export_path = export_path

    def export_onnx_model(self, model: torch.nn.Module, args: Union[torch.Tensor, Tuple[torch.Tensor]]) -> None:
        """
        Exports a model to onnx.

        - `model`: The pytorch model to export.
        - `args`: Representative tensor(s) in the same shape as the expected input(s) (can be zero, random, ones or even real data).
        """
        if has_brevitas is False:
            raise RuntimeError("Brevitas needs to be installed to use this ModelExporter!")

        brevitas.export.export_onnx_qcdq(model, args=args, export_path=self.export_path)
