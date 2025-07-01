#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Optional, TypeVar, Tuple
import torch
from functools import reduce
from torch import Tensor
from quark.torch.quantization.config.config import QuantizationSpec
from quark.torch.quantization.config.type import QSchemeType
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.utils import get_dtype_params

T = TypeVar('T', bound='PackMethod')


def _pack(x: Tensor, n_bits: int) -> Tensor:
    return reduce(
        torch.bitwise_or,
        [x[..., i::(8 // n_bits)] << (8 - (i + 1) * n_bits) for i in range(8 // n_bits)],
    )


def _unpack(x: Tensor, n_bits: int) -> Tensor:
    return torch.stack(
        [(x >> (8 - (i + 1) * n_bits)) & ((1 << n_bits) - 1) for i in range(8 // n_bits)],
        dim=-1,
    ).flatten(-2)


class PackMethod:

    def __init__(self, qscheme: Optional[str], dtype: str) -> None:
        self.qscheme = qscheme
        self.dtype = dtype
        self.qparams_per_item = 1

    def pack(self, to_pack: torch.Tensor, reorder: bool) -> torch.Tensor:
        return to_pack

    def unpack(self,
               to_unpack: torch.Tensor,
               reorder: bool,
               origin_packed_axis_size: Optional[int] = None) -> torch.Tensor:
        return to_unpack

    def transpose(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def _infer_scale_zero_point_shape(self,
                                      unpacked_shape: Tuple[int, ...],
                                      quantization_spec: QuantizationSpec,
                                      legacy: bool = False,
                                      custom_mode: str = "quark") -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        shape_list = list(unpacked_shape)
        if quantization_spec.qscheme == QSchemeType.per_tensor:
            zero_point_shape: Tuple[int, ...] = ()
            # TODO: del here safely
            # For per-tensor integer models, there used to be a bug in the export in quark<1.0 where serialized scale for per-tensor quantization would be of shape `torch.Size([1])` instead of the expected `torch.Size([])`.
            # if legacy and quantization_spec.dtype in INT_QUANT_DTYPES:
            #     scale_shape: Tuple[int, ...] = (1, )
            # else:
            #     scale_shape = ()
            scale_shape: Tuple[int, ...] = ()
        elif quantization_spec.qscheme == QSchemeType.per_channel:
            axis = quantization_spec.ch_axis
            assert axis is not None, "ch_axis should be specified for per_channel quantization"
            scale_shape = (shape_list[axis], )
            if shape_list[axis] % self.qparams_per_item != 0:
                raise ValueError(
                    f"shape_list[axis]={shape_list[axis]} is not divisible by the packing size qparams_per_item={self.qparams_per_item}. Please open an issue."
                )
            zero_point_shape = (shape_list[axis] // self.qparams_per_item, )
        elif quantization_spec.qscheme == QSchemeType.per_group:
            if quantization_spec.group_size is None:
                raise ValueError(
                    "The group_size should be specified in the quantization config when using qscheme=QSchemeType.per_group. Got group_size=None."
                )
            if len(shape_list) != 2:
                raise ValueError(
                    "Per-group quantization is only supported for 2D tensors. Got a tensor with shape {shape_list}.")
            group_size = quantization_spec.group_size
            if quantization_spec.ch_axis in [1, -1]:
                if not legacy and quantization_spec.dtype in [
                        Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.uint8, Dtype.int2
                ]:
                    scale_shape = (shape_list[-1] // group_size, shape_list[0])
                elif legacy and custom_mode == "awq":
                    scale_shape = (shape_list[-1] // group_size, shape_list[0])
                else:  # pragma: no cover
                    # PR #1070 added a transpose for the scale for uint4/int4 data types, whenever using per-group quantization.
                    # Before quark==1.0, only custom AWQ models used to transpose the scale.
                    scale_shape = (shape_list[0], shape_list[-1] // group_size)
                zero_point_shape = (shape_list[-1] // group_size, shape_list[0] // self.qparams_per_item)
            else:
                raise NotImplementedError(
                    f"Packed shape inference for per group quantization with `ch_axis={quantization_spec.ch_axis}` is not implemented in Quark. Please open an issue."
                )

        return scale_shape, zero_point_shape

    def _infer_tensor_shape(self, unpacked_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return unpacked_shape

    def infer_packed_shape(self,
                           unpacked_shape: Tuple[int, ...],
                           quantization_spec: QuantizationSpec,
                           legacy: bool = False,
                           custom_mode: str = "quark") -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        packed_tensor_shape = self._infer_tensor_shape(unpacked_shape=unpacked_shape)

        scale_shape, zero_point_shape = self._infer_scale_zero_point_shape(unpacked_shape=unpacked_shape,
                                                                           quantization_spec=quantization_spec,
                                                                           legacy=legacy,
                                                                           custom_mode=custom_mode)

        return packed_tensor_shape, scale_shape, zero_point_shape


# TODO：Implement the pack func @hongwei
class Pack_2_bits(PackMethod):

    def __init__(self, qscheme: Optional[str], dtype: str) -> None:
        super().__init__(qscheme, dtype)

    def pack(self, to_pack: torch.Tensor, reorder: bool) -> torch.Tensor:
        to_pack = self.transpose(to_pack)  # per_group quantization transposes the weight.
        return to_pack

    def unpack(self,
               to_unpack: torch.Tensor,
               reorder: bool,
               origin_packed_axis_size: Optional[int] = None) -> torch.Tensor:
        to_unpack = self.transpose(to_unpack)
        return to_unpack

    def transpose(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim > 2:
            raise ValueError("Only supports tensors with dimensions not greater than 2.")
        if self.qscheme == "per_group":  # pragma: no cover
            tensor = tensor.t().contiguous()
        return tensor

    def _infer_tensor_shape(self, unpacked_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        out_features, in_features = unpacked_shape

        if self.qscheme == "per_group":
            shape = (in_features, out_features)
        else:
            shape = (out_features, in_features)

        return shape


class Pack_4_bits(PackMethod):

    def __init__(self, qscheme: Optional[str], dtype: str) -> None:
        super().__init__(qscheme, dtype)
        self.qparams_per_item = 8

    def pack(self, to_pack: torch.Tensor, reorder: bool = True) -> torch.Tensor:
        if to_pack.ndim > 2:
            raise ValueError("Pack: Only supports tensors with dimensions not greater than 2.")

        to_pack = self.transpose(to_pack)

        org_ndim = to_pack.ndim
        if org_ndim == 1:
            to_pack = to_pack.unsqueeze(0)
        if reorder:
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        else:
            order_map = [0, 1, 2, 3, 4, 5, 6, 7]
        pack_num = 8
        if to_pack.ndim == 2:
            new_channel_num = (to_pack.shape[1] + pack_num - 1) // pack_num
            packed = torch.zeros(to_pack.shape[0], new_channel_num, dtype=torch.int32, device=to_pack.device)
            for c in range(new_channel_num):
                for i in range(pack_num):
                    # Use -3 as an example, high_position is 11111111,cause bit_or generate errors, so we can't use int4 directly                    idx = c * pack_num + order_map[i]
                    idx = c * pack_num + order_map[i]
                    # skip padding data
                    if idx >= to_pack.shape[1]:
                        continue
                    packed_col = to_pack[:, idx]
                    if self.dtype == "int4":
                        packed_col = packed_col & 0x0F
                    packed[:, c] = torch.bitwise_or(packed[:, c], torch.bitwise_left_shift(packed_col, i * 4))
        elif to_pack.ndim == 0:
            packed = to_pack.to(torch.int32)
        if org_ndim == 1:
            packed = packed.squeeze(0)

        return packed

    def unpack(self,
               to_unpack: torch.Tensor,
               reorder: bool = True,
               origin_packed_axis_size: Optional[int] = None) -> torch.Tensor:
        if to_unpack.ndim > 2:
            raise ValueError("Unpack: Only supports tensors with dimensions not greater than 2.")

        shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], device=to_unpack.device)
        ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
        org_ndim = to_unpack.ndim
        if org_ndim == 1:
            to_unpack = to_unpack.unsqueeze(0)
        if to_unpack.ndim == 2:
            unpacked = (to_unpack.unsqueeze(-1) >> shifts.view(1, 1, -1)).view(to_unpack.shape[0], -1).to(torch.int8)
            if reorder:
                order_tensor = torch.arange(
                    unpacked.shape[-1],
                    dtype=torch.int32,
                    device=unpacked.device,
                )
                order_tensor = order_tensor.view(-1, 8)
                order_tensor = order_tensor[:, ORDER].view(-1)
                unpacked = unpacked[:, order_tensor]

        elif to_unpack.ndim == 0:
            unpacked = to_unpack

        unpacked &= 0b1111
        # Use -3 as an example, we have to restore 00001101 to 11111101, so we can check the fourth digit of the unzipped number,
        # and if the fourth digit == 1 it proves that the number is negative
        if self.dtype == "int4":
            mask = (unpacked & 0x08).bool()
            unpacked[mask] = unpacked[mask] | 0xF0

        if org_ndim == 1:
            unpacked = unpacked.squeeze(0)
        unpacked = self.transpose(unpacked)
        if origin_packed_axis_size is not None and origin_packed_axis_size != unpacked.shape[0]:
            if unpacked.dim() == 2:
                unpacked = unpacked[:origin_packed_axis_size, :]
            if unpacked.dim() == 1:
                unpacked = unpacked[:origin_packed_axis_size]

        return unpacked

    def transpose(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim > 2:
            raise ValueError("Only supports tensors with dimensions not greater than 2.")
        if self.qscheme == "per_group":
            tensor = tensor.t().contiguous()
        return tensor

    def _infer_tensor_shape(self, unpacked_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        shape_list = list(unpacked_shape)
        if self.qscheme == "per_group":
            # reverse the first dimennsion number to the last dimension number
            shape_list[0], shape_list[-1] = shape_list[-1], shape_list[0]
        shape_list[-1] = shape_list[-1] // self.qparams_per_item
        return tuple(shape_list)


class Pack_8_bits(PackMethod):

    def __init__(self, qscheme: Optional[str], dtype: str) -> None:
        super().__init__(qscheme, dtype)

    def pack(self, to_pack: torch.Tensor, reorder: bool) -> torch.Tensor:
        to_pack = self.transpose(to_pack)  # per_group quantization transposes the weight.

        if self.dtype == "uint8":
            return to_pack.to(torch.uint8).contiguous()
        else:
            return to_pack.to(torch.int8).contiguous()

    def unpack(self,
               to_unpack: torch.Tensor,
               reorder: bool,
               origin_packed_axis_size: Optional[int] = None) -> torch.Tensor:
        to_unpack = self.transpose(to_unpack)
        return to_unpack.to(torch.int32).contiguous()

    def transpose(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim > 2:
            raise ValueError("Only supports tensors with dimensions not greater than 2.")
        if self.qscheme == "per_group":  # pragma: no cover
            tensor = tensor.t().contiguous()
        return tensor

    def _infer_tensor_shape(self, unpacked_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        out_features, in_features = unpacked_shape

        if self.qscheme == "per_group":
            shape = (in_features, out_features)
        else:
            shape = (out_features, in_features)

        return shape


class Pack_mxfp4(PackMethod):

    def __init__(self, qscheme: Optional[str], dtype: str) -> None:
        super().__init__(qscheme, dtype)

    def pack(self, tensor: torch.Tensor, reorder: bool, axis: int = -1) -> torch.Tensor:
        input_shape = list(tensor.shape)
        input_shape[axis], input_shape[-1] = input_shape[-1], input_shape[axis]
        tensor = tensor.transpose(axis, -1)

        tensor = tensor.reshape(-1, 33)
        scale_part = torch.log2(tensor[:, :1]).to(torch.int8).view(torch.uint8)
        element_part = (tensor[:, 1:] / (2.0**(127 - 1))).view(torch.int32)
        element_part = ((element_part >> 22) & 0x07) + ((element_part >> 28) & 0x08)
        element_part = element_part.to(torch.uint8)

        element_part = element_part.reshape(-1, 16, 2)
        # using little endian to pack the element_part,
        # for example, there is a torch.tensor([0x1, 0x2, 0x3, 0x4])
        # the packed tensor result should be torch.tensor([0x21, 0x43])
        element_part = (element_part[:, :, 1] << 4) + element_part[:, :, 0]
        ret = torch.cat([scale_part, element_part], dim=-1)

        input_shape[-1] = input_shape[-1] // 33 * 17
        return ret.reshape(input_shape).transpose(axis, -1)

    def unpack(self,
               tensor: torch.Tensor,
               reorder: bool,
               origin_packed_axis_size: Optional[int] = None,
               axis: int = -1) -> torch.Tensor:
        input_shape = list(tensor.shape)
        input_shape[axis], input_shape[-1] = input_shape[-1], input_shape[axis]
        tensor = tensor.transpose(axis, -1)
        tensor = tensor.reshape(-1, 17)
        scale_part = tensor[:, :1].view(torch.int8).to(torch.float32)
        element_part = tensor[:, 1:]

        # Unpack the 4-bit values from each byte
        unpacked = torch.zeros(element_part.shape[0],
                               element_part.shape[1] * 2,
                               dtype=torch.uint8,
                               device=tensor.device)
        # using little endian to unpack the element_part,
        # for example, there is a torch.tensor([0x21, 0x43])
        # the unpacked tensor result should be torch.tensor([0x1, 0x2, 0x3, 0x4])
        unpacked[:, 1::2] = (element_part >> 4) & 0x0F  # Extract high 4 bits
        unpacked[:, ::2] = element_part & 0x0F  # Extract low 4 bits

        # Convert back to int32 and restore the original scaling
        unpacked = unpacked.to(torch.int32)
        unpacked = ((unpacked & 0x07) << 22) | ((unpacked & 0x08) << 28)
        unpacked = unpacked.view(torch.float32) * (2.0**(127 - 1))

        # Concatenate scale and element parts
        ret = torch.cat([2**scale_part, unpacked.reshape(scale_part.shape[0], -1)], dim=-1)

        # Reshape back to original shape
        input_shape[-1] = input_shape[-1] * 33 // 17
        return ret.reshape(input_shape).transpose(axis, -1)

    def transpose(self, tensor: torch.Tensor) -> torch.Tensor:  # to del
        if tensor.ndim > 2:
            raise ValueError("Only supports tensors with dimensions not greater than 2.")
        if self.qscheme == "per_group":  # pragma: no cover
            tensor = tensor.t().contiguous()
        return tensor


class Pack_mxfp6(PackMethod):

    def __init__(self, qscheme: Optional[str], dtype: str, e_bits: int, m_bits: int) -> None:
        super().__init__(qscheme, dtype)
        self.e_bits = e_bits
        self.m_bits = m_bits

    def pack(self, tensor: torch.Tensor, reorder: bool, axis: int = -1) -> torch.Tensor:
        input_shape = list(tensor.shape)
        input_shape[axis], input_shape[-1] = input_shape[-1], input_shape[axis]
        tensor = tensor.transpose(axis, -1)

        tensor = tensor.reshape(-1, 33)
        scale_part = torch.log2(tensor[:, :1]).to(torch.int8).view(torch.uint8)
        fp6_ebias = (1 << (self.e_bits - 1)) - 1
        element_part = (tensor[:, 1:] / (2.0**(127 - fp6_ebias))).view(torch.int32)
        fp6_mbias = 23 - self.m_bits
        element_part = ((element_part >> fp6_mbias) & 0x1f) + ((element_part >> 26) & 0x20)
        element_part = element_part.to(torch.uint8)

        element_part = element_part.reshape(-1, 8, 4)

        tensor_2bit = (element_part >> 4) & 0b11
        tensor_2bit = _pack(tensor_2bit, 2)

        tensor_4bit = element_part & 0b1111
        tensor_4bit = _pack(tensor_4bit, 4)
        element_part = torch.cat([tensor_2bit, tensor_4bit], dim=-1).reshape(-1, 24)
        ret = torch.cat([scale_part, element_part], dim=-1)
        input_shape[-1] = input_shape[-1] // 33 * 25
        return ret.reshape(input_shape).transpose(axis, -1)

    def unpack(
        self,
        tensor: torch.Tensor,
        reorder: bool,
        origin_packed_axis_size: Optional[int] = None,
        axis: int = -1,
    ) -> torch.Tensor:
        input_shape = list(tensor.shape)
        input_shape[axis], input_shape[-1] = input_shape[-1], input_shape[axis]
        tensor = tensor.transpose(axis, -1)
        tensor = tensor.reshape(-1, 25)
        scale_part = tensor[:, :1].view(torch.int8).to(torch.float32)
        element_part = tensor[:, 1:].reshape(-1, 8, 3)

        # Unpack 2-bit and 4-bit portions
        tensor_2bit = element_part[:, :, :1].reshape(-1, 8)
        tensor_4bit = element_part[:, :, 1:].reshape(-1, 16)

        tensor_2bit = _unpack(tensor_2bit, 2)
        tensor_4bit = _unpack(tensor_4bit, 4)

        # Combine unpacked values
        element_part = torch.zeros(tensor_2bit.shape[0], 32, dtype=torch.uint8, device=tensor.device)
        element_part = ((tensor_2bit << 4) & 0x30) | (tensor_4bit & 0x0F)

        # Convert back to float32 with proper scaling
        fp6_ebias = (1 << (self.e_bits - 1)) - 1
        fp6_mbias = 23 - self.m_bits
        element_part = element_part.to(torch.int32)
        element_part = ((element_part & 0x1f) << fp6_mbias) | ((element_part & 0x20) << 26)
        element_part = element_part.view(torch.float32) * (2.0**(127 - fp6_ebias))

        # Apply scale and reshape
        ret = torch.cat([2**scale_part, element_part.reshape(scale_part.shape[0], -1)], dim=-1)
        input_shape[-1] = input_shape[-1] * 33 // 25
        return ret.reshape(input_shape).transpose(axis, -1)


class Pack_fp4(PackMethod):

    def __init__(self, qscheme: Optional[str], dtype: str) -> None:
        super().__init__(qscheme, dtype)
        self.qparams_per_item = 2

    def pack(self, tensor: torch.Tensor, reorder: bool) -> torch.Tensor:
        input_shape = list(tensor.shape)
        tensor = (tensor / (2.0**(127 - 1))).view(torch.int32)
        tensor = ((tensor >> 22) & 0x07) + ((tensor >> 28) & 0x08)
        tensor = tensor.to(torch.uint8)

        tensor = tensor.reshape(*input_shape[:-1], input_shape[-1] // 2, 2)
        # using little endian to pack the element_part,
        # for example, there is a torch.tensor([0x1, 0x2])
        # the packed tensor result should be torch.tensor([0x21])
        tensor = (tensor[..., 1] << 4) + tensor[..., 0]
        return tensor

    def unpack(self,
               tensor: torch.Tensor,
               reorder: bool,
               origin_packed_axis_size: Optional[int] = None) -> torch.Tensor:
        # Unpack the 4-bit values from each byte
        original_shape = tensor.shape[:-1]

        tensor = tensor.reshape(-1, tensor.shape[-1])
        unpacked = torch.zeros(tensor.shape[0], tensor.shape[1] * 2, dtype=torch.uint8, device=tensor.device)
        # using little endian to unpack the element_part,
        # for example, there is a torch.tensor([0x21])
        # the unpacked tensor result should be torch.tensor([0x1, 0x2])
        unpacked[:, 1::2] = (tensor >> 4) & 0x0F  # Extract high 4 bits
        unpacked[:, ::2] = tensor & 0x0F  # Extract low 4 bits

        # Convert back to int32 and restore the original scaling
        unpacked = unpacked.to(torch.int32)
        unpacked = ((unpacked & 0x07) << 22) | ((unpacked & 0x08) << 28)
        unpacked = unpacked.view(torch.float32) * (2.0**(127 - 1))

        # Restore the original (-2, -3, ...) dimensions (if any)
        unpacked = unpacked.reshape(*original_shape, unpacked.shape[-1])

        return unpacked

    def _infer_tensor_shape(self, unpacked_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        shape_list = list(unpacked_shape)
        shape_list[-1] = shape_list[-1] // self.qparams_per_item
        return tuple(shape_list)


class Pack_fp6(PackMethod):

    def __init__(self, qscheme: Optional[str], dtype: str, e_bits: int, m_bits: int) -> None:
        super().__init__(qscheme, dtype)
        self.e_bits = e_bits
        self.m_bits = m_bits

    def pack(self, tensor: torch.Tensor, reorder: bool) -> torch.Tensor:
        input_shape = list(tensor.shape)
        fp6_ebias = (1 << (self.e_bits - 1)) - 1
        tensor = (tensor / (2.0**(127 - fp6_ebias))).view(torch.int32)
        fp6_mbias = 23 - self.m_bits
        tensor = ((tensor >> fp6_mbias) & 0x1f) + ((tensor >> 26) & 0x20)
        tensor = tensor.to(torch.uint8)

        tensor = tensor.reshape(-1, 8, 4)
        tensor_2bit = (tensor >> 4) & 0b11
        tensor_2bit = _pack(tensor_2bit, 2)

        tensor_4bit = tensor & 0b1111
        tensor_4bit = _pack(tensor_4bit, 4)
        tensor = torch.cat([tensor_2bit, tensor_4bit], dim=-1).reshape(-1, 24)
        input_shape[-1] = input_shape[-1] // 4 * 3
        return tensor.reshape(input_shape)

    def unpack(
        self,
        tensor: torch.Tensor,
        reorder: bool,
        origin_packed_axis_size: Optional[int] = None,
    ) -> torch.Tensor:
        input_shape = list(tensor.shape)
        tensor = tensor.reshape(-1, 8, 3)

        # Unpack 2-bit and 4-bit portions
        tensor_2bit = tensor[:, :, :1].reshape(-1, 8)
        tensor_4bit = tensor[:, :, 1:].reshape(-1, 16)

        tensor_2bit = _unpack(tensor_2bit, 2)
        tensor_4bit = _unpack(tensor_4bit, 4)

        # Combine unpacked values
        unpacked = torch.zeros(tensor_2bit.shape[0], 32, dtype=torch.uint8, device=tensor.device)
        unpacked = ((tensor_2bit << 4) & 0x30) | (tensor_4bit & 0x0F)

        # Convert back to float32 with proper scaling
        fp6_ebias = (1 << (self.e_bits - 1)) - 1
        fp6_mbias = 23 - self.m_bits
        unpacked = unpacked.to(torch.int32)
        unpacked = ((unpacked & 0x1f) << fp6_mbias) | ((unpacked & 0x20) << 26)
        unpacked = unpacked.view(torch.float32) * (2.0**(127 - fp6_ebias))

        # Apply scale and reshape
        input_shape[-1] = input_shape[-1] * 4 // 3
        return unpacked.reshape(input_shape)

    def _infer_tensor_shape(self, unpacked_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        shape_list = list(unpacked_shape)
        shape_list[-1] = shape_list[-1] // 4 * 3
        return tuple(shape_list)


def create_pack_method(qscheme: Optional[str], dtype: str, mx_element_dtype: Optional[str] = None) -> PackMethod:
    if dtype == "int4" or dtype == "uint4":
        return Pack_4_bits(qscheme, dtype)
    elif dtype == "int2":
        return Pack_2_bits(qscheme, dtype)
    elif dtype == "int8" or dtype == "uint8":
        return Pack_8_bits(qscheme, dtype)
    elif dtype == "mx" and mx_element_dtype == "fp4":
        return Pack_mxfp4(qscheme, dtype)
    elif dtype == "mx" and mx_element_dtype in ["fp6_e2m3", "fp6_e3m2"]:
        element_dtype = Dtype(mx_element_dtype)
        e_bits, m_bits, _ = get_dtype_params(element_dtype)
        return Pack_mxfp6(qscheme, dtype, e_bits, m_bits)
    elif dtype == "fp4":
        return Pack_fp4(qscheme, dtype)
    elif dtype in ["fp6_e2m3", "fp6_e3m2"]:
        element_dtype = Dtype(dtype)
        e_bits, m_bits, _ = get_dtype_params(element_dtype)
        return Pack_fp6(qscheme, dtype, e_bits, m_bits)
    else:
        return PackMethod(qscheme, dtype)
