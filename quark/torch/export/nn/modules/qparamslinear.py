#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any

import torch
from torch import nn
from torch.distributed._tensor import DTensor, Replicate, distribute_tensor  # type: ignore[attr-defined]
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from quark.torch.algorithm.rotation.hadamard import _get_hadamard_K
from quark.torch.algorithm.rotation.rotation_utils import HadamardTransform, OrthogonalTransform
from quark.torch.export.constants import SCALED_MM_AVAILABLE_DEV
from quark.torch.export.nn.modules.qparamslinear_builder import create_builder
from quark.torch.export.nn.modules.quark_linear_base import QuarkLinearBase
from quark.torch.export.nn.modules.realquantizer import RealQuantizerBase, SequentialRealQuantizer
from quark.torch.quantization.config.config import AlgoConfig, QLayerConfig, RotationConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.utils import e4m3fn_to_e4m3fnuz


def normalize_e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor, qinput: torch.Tensor, weight_scale: torch.Tensor, input_scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """normalize_e4m3fn_to_e4m3fnuz for amd gpu"""
    assert weight.dtype == torch.float8_e4m3fn
    assert qinput.dtype == torch.float8_e4m3fn
    ROCM_FP8_NAN_AS_INT = -128

    weight_as_int8 = weight.view(torch.int8)
    weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)

    qinput_as_int8 = qinput.view(torch.int8)
    qinput_as_int8[qinput_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
    qinput = qinput_as_int8.view(torch.float8_e4m3fnuz)

    weight_scale = weight_scale * 2.0
    if input_scale is not None:
        input_scale = input_scale * 2.0
    return weight, qinput, weight_scale, input_scale


class QparamsOperator(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.weight_quantizer: RealQuantizerBase | SequentialRealQuantizer | None = None
        self.bias_quantizer: RealQuantizerBase | SequentialRealQuantizer | None = None
        self.input_quantizer: RealQuantizerBase | SequentialRealQuantizer | None = None
        self.output_quantizer: RealQuantizerBase | SequentialRealQuantizer | None = None


class QParamsLinear(torch.nn.Linear, QparamsOperator, QuarkLinearBase):
    def __init__(
        self,
        linear: nn.Linear,
        custom_mode: str,
        pack_method: str | None = "reorder",
        quant_config: QLayerConfig | None = None,
        algo_config: list[AlgoConfig] | None = None,
    ):
        bias = True if linear.bias is not None else False
        super().__init__(linear.in_features, linear.out_features, bias)

        self._custom_mode: str = custom_mode
        self._quant_config: QLayerConfig | None = quant_config  # Store for cache quantization check

        builder = create_builder(linear, pack_method == "reorder", custom_mode, quant_config)
        builder.build(self)

        self._quant_dict = None
        self.algo_config = algo_config

    # In the original __init__ function of torch.nn.Linear,
    # the reset_parameters function is called, which takes up a lot of time.
    # This is the reason why inplace ops replacement is slow.
    # Therefore, overload this function in this class to skip the parameter
    # allocation operation, reducing the time of inplace ops replacement.
    def reset_parameters(self) -> None:
        pass

    def can_use_fp8_kernel(self) -> bool:
        """check use_fp8_kernel or not"""
        # pertensor only now, w and inp should be quantized
        if SCALED_MM_AVAILABLE_DEV is None:
            return False

        if not (self.input_quantizer and self.weight_quantizer):
            return False

        if isinstance(self.input_quantizer, SequentialRealQuantizer) or isinstance(
            self.weight_quantizer, SequentialRealQuantizer
        ):
            return False

        input_qspec = self.input_quantizer.qspec
        weight_qspec = self.weight_quantizer.qspec

        conditions = [
            input_qspec.dtype == Dtype.fp8_e4m3,
            weight_qspec.dtype == Dtype.fp8_e4m3,
            not input_qspec.is_dynamic,
            not weight_qspec.is_dynamic,
            input_qspec.qscheme == QSchemeType.per_tensor,
            weight_qspec.qscheme == QSchemeType.per_tensor,
        ]

        return all(conditions)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Dequantizes quantized weight/bias, runs a linear in high precision and apply QDQ on the (input)activation/output if required.
        """
        input_tensor = args[0]
        dtype = input_tensor.dtype

        if self.can_use_fp8_kernel():
            return self._forward_fp8(input_tensor, dtype)
        return self._forward_generic(input_tensor, dtype)

    def _forward_fp8(self, input_tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """FP8 scaled_mm forward path (dispatches Tensor vs DTensor)."""
        bias = self._prepare_fp8_bias(input_tensor, dtype)
        if isinstance(input_tensor, DTensor):
            output, output_shape = self._forward_fp8_dtensor(input_tensor, bias, dtype)
        else:
            output, output_shape = self._forward_fp8_tensor(input_tensor, bias, dtype)

        quantized_output: torch.Tensor = self._get_qoutput(output).to(dtype)  # type: ignore[arg-type]
        return quantized_output.view(*output_shape)

    def _prepare_fp8_bias(self, input_tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor | None:
        if self.bias is None:
            return None
        if dtype == torch.float32:
            raise ValueError("Bias is not supported when out_dtype is set to Float32")
        if self.bias.dtype == torch.float32:
            return self.bias.to(torch.float16)
        return self.bias.to(input_tensor.dtype)

    def _forward_fp8_tensor(
        self, input_tensor: torch.Tensor, bias: torch.Tensor | None, dtype: torch.dtype
    ) -> tuple[torch.Tensor, list[int]]:
        """FP8 forward for regular (non-DTensor) inputs."""
        assert self.input_quantizer is not None
        assert self.weight_quantizer is not None

        max_value = 448 if self.input_quantizer.qspec.dtype == Dtype.fp8_e4m3 else 57344
        input_2d = input_tensor.view(-1, input_tensor.shape[-1])
        input_2d = input_2d / self.input_quantizer.scale
        input_2d = torch.clamp(input_2d, min=-max_value, max=max_value)
        quantized_input = input_2d.to(self.input_quantizer.qspec.dtype.to_torch_packed_dtype())

        transposed_weight = self.weight.t()
        output_shape = [*input_tensor.shape[:-1], transposed_weight.shape[1]]
        input_scale = self.input_quantizer.scale
        weight_scale = self.weight_quantizer.scale

        if SCALED_MM_AVAILABLE_DEV == "hip":
            transposed_weight, quantized_input, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                weight=transposed_weight,
                qinput=quantized_input,
                weight_scale=weight_scale,
                input_scale=input_scale,
            )

        output = torch._scaled_mm(
            quantized_input,
            transposed_weight,
            out_dtype=dtype,
            scale_a=input_scale.to(torch.float32),
            scale_b=weight_scale.to(torch.float32),
            bias=bias,
        )
        # torch._scaled_mm returns tuple for torch < 2.5
        if isinstance(output, tuple) and len(output) == 2:
            output = output[0]

        return output, output_shape

    def _forward_fp8_dtensor(
        self, input_tensor: DTensor, bias: torch.Tensor | None, dtype: torch.dtype
    ) -> tuple[torch.Tensor, list[int]]:
        """FP8 forward for DTensor inputs (tensor parallel)."""
        assert self._quant_dict is not None
        if self.input_quantizer is None:
            self.input_quantizer = self._quant_dict["input_quantizer"]
        if self.weight_quantizer is None:
            self.weight_quantizer = self._quant_dict["weight_quantizer"]

        assert self.input_quantizer is not None
        assert self.weight_quantizer is not None

        input_scale = self.input_quantizer.scale
        weight_scale = self.weight_quantizer.scale

        if not isinstance(input_scale, DTensor):
            input_scale = distribute_tensor(
                input_scale.to(torch.float32), device_mesh=input_tensor.device_mesh, placements=[Replicate()]
            )
            self.input_quantizer.scale = input_scale

        if not isinstance(weight_scale, DTensor):
            weight_scale = distribute_tensor(
                weight_scale.to(torch.float32), device_mesh=input_tensor.device_mesh, placements=[Replicate()]
            )
            self.weight_quantizer.scale = weight_scale

        max_value = 448 if self.input_quantizer.qspec.dtype == Dtype.fp8_e4m3 else 57344
        input_2d = input_tensor.view(-1, input_tensor.shape[-1])
        input_2d = input_2d / input_scale
        input_2d = torch.clamp(input_2d, min=-max_value, max=max_value)
        quantized_input = input_2d.to(self.input_quantizer.qspec.dtype.to_torch_packed_dtype())

        transposed_weight = self.weight.permute(1, 0)
        output_shape = [*input_tensor.shape[:-1], transposed_weight.shape[1]]

        if SCALED_MM_AVAILABLE_DEV == "hip":
            quantized_input, input_scale = e4m3fn_to_e4m3fnuz(tensor=quantized_input, tensor_scale=input_scale)

        output = torch._scaled_mm(
            quantized_input,
            transposed_weight,
            out_dtype=dtype,
            scale_a=input_scale,
            scale_b=weight_scale,
            bias=None,
        )
        if type(output) is tuple and len(output) == 2:
            output = output[0]

        if self.bias is not None:
            output = output + bias

        return output, output_shape

    def _forward_generic(self, input_tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Generic forward path using F.linear with dequantized weight."""
        qinput = self._get_qinput(input_tensor).to(dtype)
        qweight = self._get_qweight(self.weight).to(dtype)
        qbias = self._get_qbias(self.bias)
        if qbias is not None:
            qbias = qbias.to(dtype)
        # Ensure qweight and qbias are on the same device as qinput
        if qweight.device != qinput.device:
            qweight = qweight.to(qinput.device)
        if qbias is not None and qbias.device != qinput.device:
            qbias = qbias.to(qinput.device)
        qoutput = F.linear(qinput, qweight, bias=qbias)
        return self._get_qoutput(qoutput).to(dtype)

    def _get_qweight(self, x: Parameter) -> torch.Tensor:
        weight_quantizer = self.weight_quantizer
        if self._quant_dict is not None:
            weight_quantizer = self._quant_dict["weight_quantizer"]

        if weight_quantizer is not None:
            x = weight_quantizer(x.data)
            assert isinstance(x, torch.Tensor)
            return x
        else:
            return x.data

    def _get_qbias(self, x: Parameter | None) -> torch.Tensor | None:
        bias_quantizer = self.bias_quantizer
        if self._quant_dict is not None and "bias_quantizer" in self._quant_dict:
            bias_quantizer = self._quant_dict["bias_quantizer"]

        if bias_quantizer is not None and x is not None:
            x = bias_quantizer(x.data)
            assert isinstance(x, torch.Tensor)
            return x
        else:
            return x.data if x is not None else x

    def _get_qinput(self, x: torch.Tensor) -> torch.Tensor:
        input_quantizer = self.input_quantizer
        if self._quant_dict is not None and "input_quantizer" in self._quant_dict:
            input_quantizer = self._quant_dict["input_quantizer"]

        if input_quantizer is not None:
            x = input_quantizer(x)
            assert isinstance(x, torch.Tensor)
            return x
        else:
            return x

    def _get_qoutput(self, x: torch.Tensor) -> torch.Tensor:
        output_quantizer = self.output_quantizer
        if self._quant_dict is not None and "output_quantizer" in self._quant_dict:
            output_quantizer = self._quant_dict["output_quantizer"]

        if output_quantizer is not None:
            x = output_quantizer(x)
            assert isinstance(x, torch.Tensor)
            return x
        else:
            return x


class QParamsLinearWithRotation(QParamsLinear):
    def __init__(
        self,
        linear: nn.Linear,
        custom_mode: str,
        pack_method: str | None = "reorder",
        quant_config: QLayerConfig | None = None,
        algo_config: list[AlgoConfig] | None = None,
    ):
        if algo_config is None:
            raise ValueError(
                f"The argument algo_config is required when initializing QParamsLinearWithRotation, got algo_config={algo_config}. Please open an issue."
            )

        super().__init__(
            linear=linear,
            custom_mode=custom_mode,
            pack_method=pack_method,
            quant_config=quant_config,
            algo_config=algo_config,
        )

        rotation_config = None
        for algo_conf in algo_config:
            if isinstance(algo_conf, RotationConfig):
                rotation_config = algo_conf
                break
        else:
            raise ValueError(
                f"Attempted to initialize a QParamsLinearWithRotation instance, but a RotationConfig was not found among algo_config={algo_config}. Please open an issue."
            )

        rotation_size = rotation_config.rotation_size
        trainable = rotation_config.trainable

        if rotation_size is None:
            rotation_size = linear.in_features

        if isinstance(linear, QuantLinear):
            input_rotation = linear.input_rotation
        elif isinstance(linear, nn.Linear):
            if trainable:
                rotation_dtype = torch.float64  # TODO: use lower precision.
            else:
                # In case hadamard transform is used (non-trained case), it is serialized as torch.int8 with only `-1` and `1` values.
                rotation_dtype = torch.int8
        else:
            raise ValueError(f"Unsupported linear type: {type(linear)}")

        input_rotation = torch.zeros((rotation_size, rotation_size), device=linear.weight.device, dtype=rotation_dtype)
        self.register_buffer("input_rotation", input_rotation)

        self.rotation_size = rotation_size
        self.trainable = trainable

    def post_process_after_loading(self) -> None:
        # TODO: make sure this function gets called as well in AutoModelForCausalLM.from_pretrained(quantized_model_id).

        if self.trainable:
            self.transform = OrthogonalTransform(rotation_matrix=self.input_rotation)  # type: ignore[has-type]
        else:
            if self.rotation_size == self.in_features:
                # inp = inp @ self.input_rotation

                # inp_dtype = inp.dtype
                # inp = inp.to(torch.float64) @ self.input_rotation
                # inp = inp.to(inp_dtype)

                # TODO: the two approaches above seem to be not strictly numerically equivalent compared to matmul_hadU (see `test_serialization_and_reload` in test_rotation.py), leaving matmul_hadU for now, verify end-to-end metrics for the influence of the two.
                use_matmul_hadU = True
                hadamard_K, K = _get_hadamard_K(self.rotation_size)
                hadamard_K = hadamard_K.to(self.input_rotation.device)  # type: ignore[has-type]
                self.input_rotation = None
            else:
                use_matmul_hadU = False
                K = None

                # In case hadamard transform is used (non-trained case), it is serialized as torch.int8 with only `-1` and `1` values.
                float_dtype = torch.float32
                self.input_rotation = self.input_rotation.to(float_dtype)  # type: ignore

                hadamard_K = self.input_rotation

            self.transform = HadamardTransform(
                rotation_size=self.rotation_size, use_matmul_hadU=use_matmul_hadU, hadamard_K=hadamard_K, K=K
            )

        delattr(self, "input_rotation")

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Dequantizes quantized weight/bias, runs a linear in high precision and apply QDQ on the (input)activation/output if required.
        """
        assert len(args) == 1
        inp = args[0]

        inp = self.transform(inp)

        return super().forward(inp)
