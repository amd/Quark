#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import importlib
import os
import platform
import sys
from typing import Any

import torch
import torch.nn.functional as F

from quark.common.utils.torch_utils import torch_supports_stable_abi
from quark.onnx.operators.custom_ops import get_legacy_torch_library_path, get_library_path


class FakeCustomTorchOps:
    """
    This class provides alternative operations to prevent errors if
    the custom operations library fails to import, because there is
    a probability that the library was not compiled successfully or
    compiled but did not define export functions.
    """

    @staticmethod
    def bfp(tensor: torch.Tensor, *args: Any) -> torch.Tensor:
        """This is a fake function for BFP quant-dequant operation
        :param tensor: The input tensor in torch tensor format
        :return the result tensor (it's just the input tensor for simplicity)
        """
        return tensor

    @staticmethod
    def bfp_prime(tensor: torch.Tensor, *args: Any) -> torch.Tensor:
        """This is a fake function for Microexponents quant-dequant operation
        :param tensor: The input tensor in torch tensor format
        :return the result tensor (it's just the input tensor for simplicity)
        """
        return tensor

    @staticmethod
    def mx(tensor: torch.Tensor, *args: Any) -> torch.Tensor:
        """This is a fake function for Microscaling quant-dequant operation
        :param tensor: The input tensor in torch tensor format
        :return the result tensor (it's just the input tensor for simplicity)
        """
        return tensor


def _load_legacy_ops(gpu: bool = False) -> Any:
    """Import the legacy pybind11 custom-ops module, or ``None`` if unavailable.

    The artifact differs by torch version: on torch >= 2.10 it's the slim
    test-only ``libcustom_ops_torch_legacy{_gpu}``; on older torch the
    surface ships inside ``libcustom_ops{_gpu}``.
    """
    if torch_supports_stable_abi():
        base = "custom_ops_torch_legacy"
        library_dir = os.path.split(get_legacy_torch_library_path())[0]
    else:
        base = "custom_ops"
        library_dir = os.path.split(get_library_path())[0]

    if library_dir not in sys.path:
        sys.path.append(library_dir)

    suffix = "_gpu" if gpu else ""
    if platform.system().lower() == "windows":
        module_name = f"{base}{suffix}"
    else:
        module_name = f"lib{base}{suffix}"

    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _resolve_ops() -> tuple[Any, Any]:
    """Resolve ``(cpu_ops, gpu_ops)`` strictly by torch version.

    No cross-tier fallback: if the version-appropriate backend isn't loaded,
    both entries fall back to :class:`FakeCustomTorchOps` so a stable-ABI
    failure can't silently route to the legacy pybind11 surface.
    """
    if torch_supports_stable_abi():
        try:
            _ = torch.ops.quark_custom_ops.mx
            ops = torch.ops.quark_custom_ops
            return ops, ops
        except (AttributeError, RuntimeError):
            return FakeCustomTorchOps, FakeCustomTorchOps

    cpu_ops = _load_legacy_ops(gpu=False) or FakeCustomTorchOps
    gpu_ops = _load_legacy_ops(gpu=True) or FakeCustomTorchOps
    return cpu_ops, gpu_ops


custom_torch_ops, custom_torch_ops_gpu = _resolve_ops()  # pragma: no cover


class BFPQuantDequantFunction(torch.autograd.Function):  # type: ignore
    @staticmethod
    def forward(
        ctx: torch.autograd.Function,
        tensor: torch.Tensor,
        bit_width: int,
        block_size: int,
        rounding_mode: int,
        kernel_version: int,
    ) -> Any:
        if tensor.device.type == "cuda":
            return custom_torch_ops_gpu.bfp(tensor, bit_width, block_size, rounding_mode, kernel_version)
        else:
            return custom_torch_ops.bfp(tensor, bit_width, block_size, rounding_mode, kernel_version)

    @staticmethod  # type: ignore
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, None, None, None, None]:
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class BFPPrimeQuantDequantFunction(torch.autograd.Function):  # type: ignore
    @staticmethod
    def forward(
        ctx: torch.autograd.Function,
        tensor: torch.Tensor,
        bit_width: int,
        block_size: int,
        sub_block_size: int,
        sub_block_shift_bits: int,
        rounding_mode: int,
    ) -> Any:
        if tensor.device.type == "cuda":
            return custom_torch_ops_gpu.bfp_prime(
                tensor, bit_width, block_size, sub_block_size, sub_block_shift_bits, rounding_mode
            )
        else:
            return custom_torch_ops.bfp_prime(
                tensor, bit_width, block_size, sub_block_size, sub_block_shift_bits, rounding_mode
            )

    @staticmethod  # type: ignore
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, None, None, None, None, None]:
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None


# Do not use the common function to replace the autograd function, otherwise it may cause
# "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
# because Torch cannot automatically derive the gradient of the BFP kernel
bfp_quant_dequant_func = BFPQuantDequantFunction.apply
bfp_prime_quant_dequant_func = BFPPrimeQuantDequantFunction.apply


class BFPQuantizer(torch.nn.Module):  # type: ignore
    """A quantizer has a similar behavior as custom BFP ops"""

    def __init__(self, attrs: dict[str, Any]) -> None:
        super().__init__()

        self.bfp_method = attrs["bfp_method"] if "bfp_method" in attrs else "to_bfp"
        self.axis = attrs["axis"] if "axis" in attrs else 1
        self.bit_width = attrs["bit_width"] if "bit_width" in attrs else 16
        self.block_size = attrs["block_size"] if "block_size" in attrs else 8
        self.sub_block_size = attrs["sub_block_size"] if "sub_block_size" in attrs else 2
        self.sub_block_shift_bits = attrs["sub_block_shift_bits"] if "sub_block_shift_bits" in attrs else 2
        self.rounding_mode = attrs["rounding_mode"] if "rounding_mode" in attrs else 0
        self.convert_to_bfloat_before_bfp = (
            attrs["convert_to_bfloat_before_bfp"] if "convert_to_bfloat_before_bfp" in attrs else 0
        )
        self.use_compiler_version_cpu_kernel = (
            attrs["use_compiler_version_cpu_kernel"] if "use_compiler_version_cpu_kernel" in attrs else 0
        )

        # This flag is used to be compatiable with QDQQuantizer
        # For BFP, it's always false
        self.q_folded = False

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        # This function is used to be compatiable with QDQQuantizer
        # For BFP, do not support quantizing a tensor to a bfp tensor yet
        return tensor

    def dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        # This function is used to be compatiable with QDQQuantizer
        # For BFP, do not support dequantizing a bfp tensor to a float tensor yet
        return tensor

    def quantize_dequantize(self, tensor: torch.Tensor) -> Any:
        if self.bfp_method == "to_bfp":
            return bfp_quant_dequant_func(
                tensor, self.bit_width, self.block_size, self.rounding_mode, self.use_compiler_version_cpu_kernel
            )
        else:
            return bfp_prime_quant_dequant_func(
                tensor,
                self.bit_width,
                self.block_size,
                self.sub_block_size,
                self.sub_block_shift_bits,
                self.rounding_mode,
            )

    def forward(self, tensor: torch.Tensor) -> Any:
        # Do nothing for scalar
        if tensor.numel() <= 1:
            return tensor

        # Convert the tensor to bfloat16 before converting to bfp
        if self.convert_to_bfloat_before_bfp == 1:
            bfloat16_tensor = tensor.to(dtype=torch.bfloat16)
            transformed_tensor = bfloat16_tensor.to(dtype=torch.float32)
        else:
            transformed_tensor = tensor

        # Transpose the target axis to the last dimension
        if tensor.dim() > 1 and (self.axis != tensor.dim() - 1 and self.axis != -1):
            transformed_tensor = transformed_tensor.transpose(self.axis, tensor.dim() - 1)

        # Pad the last dimension to make sure it could be divisible by integers
        origin_size = transformed_tensor.shape[-1]
        remainder = origin_size % self.block_size
        pad_size = 0 if remainder == 0 else self.block_size - remainder

        if pad_size > 0:
            transformed_tensor = F.pad(transformed_tensor, (0, pad_size), mode="constant", value=0)

        # Call quant-dequant
        out_tensor = self.quantize_dequantize(transformed_tensor)

        # Remove the padded data
        if pad_size > 0:
            out_tensor = out_tensor[..., :origin_size]

        # Transpose the axis back
        if tensor.dim() > 1 and (self.axis != tensor.dim() - 1 and self.axis != -1):
            out_tensor = out_tensor.transpose(self.axis, tensor.dim() - 1)

        return out_tensor


class MXQuantDequantFunction(torch.autograd.Function):  # type: ignore
    @staticmethod
    def forward(
        ctx: torch.autograd.Function,
        tensor: torch.Tensor,
        block_size: int,
        ebits: int,
        mbits: int,
        emax: int,
        max_norm: float,
        min_norm: float,
        rounding_mode: int,
    ) -> Any:
        if tensor.device.type == "cuda":
            return custom_torch_ops_gpu.mx(tensor, block_size, ebits, mbits, emax, max_norm, min_norm, rounding_mode)
        else:
            return custom_torch_ops.mx(tensor, block_size, ebits, mbits, emax, max_norm, min_norm, rounding_mode)

    @staticmethod  # type: ignore
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, None, None, None, None, None, None, None]:
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None, None, None


mx_quant_dequant_func = MXQuantDequantFunction.apply


class MXQuantizer(BFPQuantizer):
    """A quantizer has a similar behavior as custom MX ops"""

    def __init__(self, attrs: dict[str, Any]) -> None:
        super().__init__({})

        self.element_dtype = attrs["element_dtype"] if "element_dtype" in attrs else "int8"
        self.axis = attrs["axis"] if "axis" in attrs else 1
        self.block_size = attrs["block_size"] if "block_size" in attrs else 8
        self.rounding_mode = attrs["rounding_mode"] if "rounding_mode" in attrs else 0

        # Parameters for the MX data types
        self.ebits = 0  # bit numbers of exponent
        self.mbits = 0  # bit numbers of mantissa
        self.emax = 0  # max exponent value
        self.max_norm = 0.0  # max normal value
        self.min_norm = 0.0  # min normal value
        if self.element_dtype == "fp8_e5m2":
            self.ebits = 5
            self.mbits = 2
            self.emax = 15
            self.max_norm = 57344.0
            self.min_norm = -57344.0
        elif self.element_dtype == "fp8_e4m3":
            self.ebits = 4
            self.mbits = 3
            self.emax = 8
            self.max_norm = 448.0
            self.min_norm = -448.0
        elif self.element_dtype == "fp6_e3m2":
            self.ebits = 3
            self.mbits = 2
            self.emax = 4
            self.max_norm = 28.0
            self.min_norm = -28.0
        elif self.element_dtype == "fp6_e2m3":
            self.ebits = 2
            self.mbits = 3
            self.emax = 2
            self.max_norm = 7.5
            self.min_norm = -7.5
        elif self.element_dtype == "fp4_e2m1":
            self.ebits = 2
            self.mbits = 1
            self.emax = 2
            self.max_norm = 6.0
            self.min_norm = -6.0
        elif self.element_dtype == "int8":
            self.ebits = 0
            self.mbits = 8
            self.emax = 0  # MXINT8 has a implicit scale 2^-6
            self.max_norm = 127.0
            self.min_norm = -128.0
        else:
            raise ValueError(f"Unexpected type of elements: {self.element_dtype}")

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        # This function is used to be compatiable with QDQQuantizer
        # For BFP, do not support quantizing a tensor to a bfp tensor yet
        return tensor

    def dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        # This function is used to be compatiable with QDQQuantizer
        # For BFP, do not support dequantizing a bfp tensor to a float tensor yet
        return tensor

    def quantize_dequantize(self, tensor: torch.Tensor) -> Any:
        return mx_quant_dequant_func(
            tensor, self.block_size, self.ebits, self.mbits, self.emax, self.max_norm, self.min_norm, self.rounding_mode
        )

    def forward(self, tensor: torch.Tensor) -> Any:
        # Do nothing for scalar
        if tensor.numel() <= 1:
            return tensor

        transformed_tensor = tensor

        # Transpose the target axis to the last dimension
        if tensor.dim() > 1 and (self.axis != tensor.dim() - 1 and self.axis != -1):
            transformed_tensor = transformed_tensor.transpose(self.axis, tensor.dim() - 1)

        # Pad the last dimension to make sure it could be divisible by integers
        origin_size = transformed_tensor.shape[-1]
        remainder = origin_size % self.block_size
        pad_size = 0 if remainder == 0 else self.block_size - remainder

        if pad_size > 0:
            transformed_tensor = F.pad(transformed_tensor, (0, pad_size), mode="constant", value=0)

        # Call quant-dequant
        out_tensor = self.quantize_dequantize(transformed_tensor)

        # Remove the padded data
        if pad_size > 0:
            out_tensor = out_tensor[..., :origin_size]

        # Transpose the axis back
        if tensor.dim() > 1 and (self.axis != tensor.dim() - 1 and self.axis != -1):
            out_tensor = out_tensor.transpose(self.axis, tensor.dim() - 1)

        return out_tensor
