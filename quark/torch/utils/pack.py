#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Optional, TypeVar, Tuple
import torch
from quark.torch.quantization.config.config import QuantizationSpec
from quark.torch.quantization.config.type import QSchemeType
from quark.torch.quantization.config.type import Dtype
import numpy as np

T = TypeVar('T', bound='PackMethod')
INT_QUANT_DTYPES = [Dtype.int2, Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.uint8]


class PackMethod:

    def __init__(self, qscheme: Optional[str], dtype: str) -> None:
        self.qscheme = qscheme
        self.dtype = dtype

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
                                      qparams_per_item: int,
                                      legacy: bool = False,
                                      custom_mode: str = "quark") -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        (out_features, in_features) = unpacked_shape

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
            if quantization_spec.ch_axis == 0:
                scale_shape = (out_features, )

                if out_features % qparams_per_item != 0:
                    raise ValueError(
                        f"out_features={out_features} is not divisible by the packing size qparams_per_item={qparams_per_item}. Please open an issue."
                    )

                zero_point_shape = (out_features // qparams_per_item, )
            else:  # pragma: no cover
                scale_shape = (in_features, )

                if in_features % qparams_per_item != 0:
                    raise ValueError(
                        f"in_features={in_features} is not divisible by the packing size qparams_per_item={qparams_per_item}. Please open an issue."
                    )

                zero_point_shape = (in_features // qparams_per_item, )
        elif quantization_spec.qscheme == QSchemeType.per_group:
            if quantization_spec.group_size is None:
                raise ValueError(
                    "The group_size should be specified in the quantization config when using qscheme=QSchemeType.per_group. Got group_size=None."
                )

            group_size = quantization_spec.group_size

            if quantization_spec.ch_axis == 1:
                if not legacy and quantization_spec.dtype in [Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.uint8]:
                    scale_shape = (in_features // group_size, out_features)
                elif legacy and custom_mode == "awq":
                    scale_shape = (in_features // group_size, out_features)
                else:  # pragma: no cover
                    # PR #1070 added a transpose for the scale for uint4/int4 data types, whenever using per-group quantization.
                    # Before quark==1.0, only custom AWQ models used to transpose the scale.
                    scale_shape = (out_features, in_features // group_size)
                zero_point_shape = (in_features // group_size, out_features // qparams_per_item)
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
                                                                           custom_mode=custom_mode,
                                                                           qparams_per_item=1)

        return packed_tensor_shape, scale_shape, zero_point_shape


class Pack_4_bits(PackMethod):

    def __init__(self, qscheme: Optional[str], dtype: str) -> None:
        super().__init__(qscheme, dtype)

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
        out_features, in_features = unpacked_shape
        qparams_per_item = 8  # int4 is packed on int32, in the out_features dimension.

        if self.qscheme == "per_group":
            shape = (in_features, out_features // qparams_per_item)
        else:
            shape = (out_features, in_features // qparams_per_item)

        return shape

    def infer_packed_shape(self,
                           unpacked_shape: Tuple[int, ...],
                           quantization_spec: QuantizationSpec,
                           legacy: bool = False,
                           custom_mode: str = "quark") -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        packed_tensor_shape = self._infer_tensor_shape(unpacked_shape=unpacked_shape)

        # uint4/int4 pack 8 values on an int32 value, hence `qparams_per_item=8`.
        scale_shape, zero_point_shape = self._infer_scale_zero_point_shape(unpacked_shape=unpacked_shape,
                                                                           quantization_spec=quantization_spec,
                                                                           legacy=legacy,
                                                                           custom_mode=custom_mode,
                                                                           qparams_per_item=8)

        return packed_tensor_shape, scale_shape, zero_point_shape


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


class Pack_mx(PackMethod):

    def __init__(self, qscheme: Optional[str], dtype: str) -> None:
        super().__init__(qscheme, dtype)

    def pack(self, tensor: torch.Tensor, reorder: bool, axis: int = -1) -> torch.Tensor:
        input_shape = list(tensor.shape)
        input_shape[axis], input_shape[-1] = input_shape[-1], input_shape[axis]
        tensor = tensor.transpose(axis, -1)

        tensor = tensor.reshape(-1, 33)
        scale_part = torch.log2(tensor[:, :1]).to(torch.uint8)
        element_part = (tensor[:, 1:] / (2.0**(127 - 1))).view(torch.uint32).numpy()
        element_part = (np.bitwise_and(element_part >> 22, 0x07) + np.bitwise_and(element_part >> 28, 0x08)).astype(
            np.uint8)

        element_part = element_part.reshape(-1, 16, 2)
        element_part = (element_part[:, :, 0] << 4) + element_part[:, :, 1]
        ret = torch.cat([scale_part, torch.from_numpy(element_part)], dim=-1)

        input_shape[-1] = input_shape[-1] // 33 * 17
        return ret.reshape(input_shape).transpose(axis, -1)

    def unpack(self,
               to_unpack: torch.Tensor,
               reorder: bool,
               origin_packed_axis_size: Optional[int] = None) -> torch.Tensor:  # TODO: finish it
        to_unpack = self.transpose(to_unpack)
        return to_unpack.to(torch.int32).contiguous()

    def transpose(self, tensor: torch.Tensor) -> torch.Tensor:  # to del
        if tensor.ndim > 2:
            raise ValueError("Only supports tensors with dimensions not greater than 2.")
        if self.qscheme == "per_group":  # pragma: no cover
            tensor = tensor.t().contiguous()
        return tensor


def create_pack_method(qscheme: Optional[str], dtype: str) -> PackMethod:
    if dtype == "int4" or dtype == "uint4":
        return Pack_4_bits(qscheme, dtype)
    elif dtype == "int8" or dtype == "uint8":
        return Pack_8_bits(qscheme, dtype)
    elif dtype == "mx":
        return Pack_mx(qscheme, dtype)
    else:
        return PackMethod(qscheme, dtype)
