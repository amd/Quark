#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING, Any, cast, Optional, List
from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from collections import Counter
from torch.ao.quantization import HistogramObserver
import quark.torch.kernel  # noqa
if TYPE_CHECKING:
    from quark.torch.quantization.config.config import QuantizationSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType, ZeroPointType, ScaleType
from quark.torch.quantization.nn.utils import check_min_max_valid
from quark.torch.quantization.utils import calculate_qmin_qmax, reshape_to_blocks, get_dtype_params, t_exponent, even_round
from quark.shares.utils.log import ScreenLogger, log_errors
from quark.shares.utils.import_utils import is_torch_greater_or_equal_2_5
from quark.torch.kernel.hw_emulation.hw_emulation_interface import fake_quantize_int  # type: ignore

logger = ScreenLogger(__name__)


class ObserverBase(ABC, nn.Module):

    def __init__(self, qspec: QuantizationSpec, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.dtype = qspec.dtype

        self.scale_torch_dtype = None
        if qspec.scale_type in [ScaleType.float32, ScaleType.float16, ScaleType.bfloat16]:
            self.scale_torch_dtype = qspec.scale_type.to_torch_dtype()

        self._num_observed_tokens: Optional[int] = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Any:
        pass

    @abstractmethod
    def _calculate_qparams(self) -> Any:
        pass

    @abstractmethod
    def reset_state(self) -> None:
        pass

    def record_observed_tokens(self, batch_tensor: torch.Tensor) -> None:
        """
        Recording the number of tokens observed during the
        calibration process.

        Note: The batch_tensor is expected to have two dimensions
            (batch_size * sequence_length, num_features). This is the
            general shape expected by the forward pass of the expert
            layers in a MOE model. If the input tensor does not have
            two dimensions, the _num_observed_tokens attribute will be set
            to None.
        """
        if not isinstance(batch_tensor, torch.Tensor):
            raise ValueError(f"Expected value to be a tensor, got {type(batch_tensor)}")

        if batch_tensor.ndim != 2:
            logger.debug("The input tensor is expected to have two dimensions "
                         "(batch_size * sequence_length, num_features). "
                         f"But the input tensor has {batch_tensor.ndim} dimensions.")
            return

        if self._num_observed_tokens is None:
            # initialize the count
            self._num_observed_tokens = 0

        # batch_tensor (batch_size * sequence_length, num_features)
        # observed_tokens (batch_size * sequence_length)
        observed_tokens, _ = batch_tensor.shape
        self._num_observed_tokens += observed_tokens


class PlaceholderObserver(ObserverBase):
    r"""
    Observer only passes its configuration to the quantized module's ``.from_float()``.

    Does not have any calculation.

    Only can be used for quantization to float16 and bfloat16 which doesn't require determining
    ranges.
    """

    def __init__(
        self,
        qspec: QuantizationSpec,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(qspec, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _calculate_qparams(self) -> None:
        pass

    def reset_state(self) -> None:
        pass

    def extra_repr(self) -> str:
        return f"dtype={self.dtype}"


class UniformScalingObserver(ObserverBase):
    """
    Observer for uniform scaling quantizer. For example 'int uniform quantizer' or 'fp8 uniform scaling'.

    """

    eps: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self,
                 qspec: QuantizationSpec,
                 device: Optional[torch.device] = None,
                 eps: float = torch.finfo(torch.float32).eps) -> None:
        super().__init__(qspec, device)

        self.qspec = qspec
        self.symmetric = qspec.symmetric
        self.scale_type = qspec.scale_type
        self.qscheme = qspec.qscheme
        self.is_dynamic = qspec.is_dynamic
        self.zero_point_type = qspec.zero_point_type
        self.scale_format = qspec.scale_format
        self.scale_calculation_mode = qspec.scale_calculation_mode

        self.register_buffer("min_val", torch.tensor(float("inf"), device=device), persistent=False)
        self.register_buffer("max_val", torch.tensor(float("-inf"), device=device), persistent=False)
        self.register_buffer("eps", torch.tensor(eps, device=device), persistent=False)

        self.quant_min, self.quant_max = calculate_qmin_qmax(qspec.dtype)

    def _calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters."""
        return self.calculate_qparams(self.min_val, self.max_val)

    def calculate_qparams(self, min_val: torch.Tensor, max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters."""
        if self.dtype in [Dtype.fp8_e4m3, Dtype.fp8_e5m2]:
            return self.calculate_fp8_quant_parameters(min_val, max_val)
        elif self.dtype == Dtype.fp4:
            return self.calculate_fp4_quant_parameters(min_val, max_val)
        else:
            return self.calculate_int_quant_params(min_val, max_val)

    def calculate_int_quant_params(self, min_val: torch.Tensor,
                                   max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # TODO setup eps device when init
        self.eps = self.eps.to(min_val.dtype).to(min_val.device)

        if not check_min_max_valid(min_val, max_val):
            return torch.tensor([1.0], device=min_val.device.type), torch.tensor([0], device=min_val.device.type)

        quant_min, quant_max = self.quant_min, self.quant_max
        assert isinstance(quant_min, int)
        assert isinstance(quant_max, int)
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)

        # TODO: This makes the assumption that signed integer dtype is used when doing symmetric quantization. There
        # is no enforcement on that in quark.
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int32, device=device)

        if self.symmetric:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
        else:
            if self.zero_point_type == ZeroPointType.float32:
                scale = (max_val - min_val) / float(quant_max - quant_min)
                scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))
                zero_point = -1 * min_val / scale
            else:
                # AWQ
                scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
                # TODO: reset eps's device
                self.eps = self.eps.to(scale.device)
                scale = torch.max(scale, self.eps)
                zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
                zero_point = torch.clamp(zero_point, quant_min, quant_max)

        return scale.to(self.scale_torch_dtype), zero_point

    def calculate_fp8_quant_parameters(self, min_val: torch.Tensor,
                                       max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int32, device=device)

        amax = torch.maximum(torch.abs(min_val_neg), torch.abs(max_val_pos))
        _, max_norm = calculate_qmin_qmax(self.dtype)
        scale = amax / max_norm
        return scale.to(self.scale_torch_dtype), zero_point

    def calculate_fp4_quant_parameters(self, min_val: torch.Tensor,
                                       max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: reduce redundancy with calculate_qparams in PerBlockMXObserver, need to re-design
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        amax = torch.maximum(torch.abs(min_val_neg), torch.abs(max_val_pos))
        _, _, emax = get_dtype_params(self.dtype)
        _, quant_max = calculate_qmin_qmax(self.dtype)

        if self.scale_format == "e8m0":
            if self.scale_calculation_mode is None or self.scale_calculation_mode == "even":
                scale = even_round(amax, self.dtype)
            elif self.scale_calculation_mode == "floor":
                scale = torch.pow(2, torch.floor(torch.log2(amax)) - emax)
                scale = scale.masked_fill(scale == 0.0, self.eps)
            elif self.scale_calculation_mode == "ceil":
                scale = torch.pow(2, torch.ceil(torch.log2(amax)) - emax)
                scale = scale.masked_fill(scale == 0.0, self.eps)
            else:
                raise ValueError(f"Unsupported scale_calculation_mode: {self.scale_calculation_mode}")
            zero_point = torch.zeros_like(scale)
        elif self.scale_format == "e4m3":
            _, quant_max = calculate_qmin_qmax(self.qspec.dtype)
            scale = torch.div(amax, quant_max)
            scale = scale.masked_fill(scale == 0.0, self.eps)

            # two-level scale
            orginal_dtype = scale.dtype
            local_amax = amax.float()
            elem_format_max = quant_max
            global_amax = amax.max().float()
            scale_format_max = 448.0  # FP8E4M3 scale
            local_unscale = local_amax / elem_format_max
            two_level_scale = scale_format_max * (elem_format_max / global_amax)
            scale = (local_unscale * two_level_scale).to(torch.float8_e4m3fn).float() / two_level_scale
            scale = scale.to(orginal_dtype)

            zero_point = torch.zeros_like(scale)
        elif self.scale_format == "float32":
            _, quant_max = calculate_qmin_qmax(self.qspec.dtype)
            scale = torch.div(amax, quant_max)
            scale = scale.masked_fill(scale == 0.0, self.eps)
            zero_point = torch.zeros_like(scale)
        else:
            raise KeyError("scale_format must be either e8m0, e4m3 or float32.")

        return scale.masked_fill(scale == 0.0, torch.finfo(torch.float32).eps).to(self.scale_torch_dtype), zero_point

    def extra_repr(self) -> str:
        return f"min_val={self.min_val}, max_val={self.max_val}"

    def reset_state(self) -> None:
        self.reset_min_max_vals()

    def reset_min_max_vals(self) -> None:
        """Resets the min/max values."""
        self.min_val = torch.tensor(float("inf"))
        self.max_val = torch.tensor(float("-inf"))

    def reset_min_max_for_dynamic(self) -> None:
        if self.is_dynamic:
            self.reset_min_max_vals()


class PerTensorMinMaxObserver(UniformScalingObserver):

    def __init__(self, qspec: QuantizationSpec, device: Optional[torch.device] = None) -> None:
        super().__init__(qspec, device)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.detach()  # avoid keeping autograd tape
        self.reset_min_max_for_dynamic()
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

        self.min_val = self.min_val.to(x_orig.dtype)
        self.max_val = self.max_val.to(x_orig.dtype)
        return x_orig


class PerTensorPowOf2MinMaxObserver(PerTensorMinMaxObserver):

    def __init__(self, qspec: QuantizationSpec, device: Optional[torch.device] = None) -> None:
        super().__init__(qspec, device)
        assert qspec.dtype in [Dtype.int8, Dtype.uint8, Dtype.int32
                               ], "Currently PerTensorPowOf2Observer only support int8, uint8 and int32 Dtype"

        self.record_scale: List[float] = []
        self.record_zp: List[float] = []

    def _scale2pos(self, scale: torch.Tensor) -> torch.Tensor:
        """
        Ref: quark/onnx/quant_utils.py scale2pos
        """
        target_device = scale.device
        scale = min(max(scale.item(), float(2**-127)), float(2**127))
        return torch.round(-torch.log2(torch.tensor(scale))).to(device=target_device)

    def _pos2scale(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Ref: quark/onnx/quant_utils.py pos2scale
        """
        return torch.pow(2.0, -pos)

    def _calculate_one_batch_quant_param(self, min_val: torch.Tensor,
                                         max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ref: quark/onnx/quant_utils.py: compute_scale_zp
        self.eps = self.eps.to(min_val.dtype).to(min_val.device)

        if not check_min_max_valid(min_val, max_val):
            return torch.tensor([1.0], device=min_val.device.type), torch.tensor([0], device=min_val.device.type)

        quant_min, quant_max = self.quant_min, self.quant_max  # e.g -128, 127
        assert isinstance(quant_min, int)
        assert isinstance(quant_max, int)
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        # quantize range
        if self.symmetric:
            abs_max = torch.max(min_val_neg.abs(), max_val_pos.abs())
            min_val_neg = -1 * abs_max
            max_val_pos = 1 * abs_max

        dr = max_val_pos - min_val_neg
        dq = quant_max - quant_min
        scale = dr / dq if dr != 0 else torch.tensor(1.0, dtype=torch.float32)
        assert scale > 0, "scale should larger than 0, please check"
        zero_point = torch.tensor(quant_min - min_val_neg / scale).to(dtype=torch.int32)

        pos = self._scale2pos(scale)
        pof2_scale = self._pos2scale(pos)
        new_rmin = torch.min((quant_min - zero_point) * pof2_scale, torch.tensor(0, dtype=torch.int32))
        new_zero_point = torch.round(quant_min - new_rmin / pof2_scale).to(dtype=torch.int32)
        # TODO refine other dtype haoliang
        if self.symmetric and self.qspec.dtype == Dtype.uint8 and new_zero_point.item() == 127:  # [0, 255]
            new_zero_point = torch.tensor(128, dtype=torch.int32)
        return pof2_scale.to(self.scale_torch_dtype), new_zero_point

    def calculate_int_quant_params(self, min_val: torch.Tensor,
                                   max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        device = min_val.device
        one_batch_scale, one_batch_zero_point = self._calculate_one_batch_quant_param(min_val, max_val)
        self.record_scale.append(one_batch_scale.item())
        self.record_zp.append(one_batch_zero_point.item())

        # select the most common scale & zp
        matrix = torch.tensor([self.record_scale, self.record_zp])
        columns = matrix.T
        keys = [tuple(col.tolist()) for col in columns]
        counter = Counter(keys)
        most_common_key, count = counter.most_common(1)[0]

        po2_scale = torch.tensor(most_common_key[0]).to(device)
        zero_point = torch.tensor(most_common_key[1]).to(device)
        return po2_scale, zero_point

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        r"""Records the running minimum and maximum of ``x``."""
        x = super().forward(x_orig)
        return x


class PerTensorPowOf2MinMSEObserver(PerTensorPowOf2MinMaxObserver):
    '''
    MINMSE method to get the the proper scale for target input tensor
    More ref: Quark/quark/onnx/quant_utils.py func: quantize_data_pof2s
    '''

    def __init__(self, qspec: QuantizationSpec, device: Optional[torch.device] = None) -> None:
        super().__init__(qspec, device)
        self.original_tensor: Optional[
            torch.Tensor] = None  # not use `register_buffer`, as this function will save dict, waist disk space
        self.pos_range = 5
        self.compute_device = 'cpu'

    def _fake_quantize_data(self, data: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        cliplow, cliphigh = self.quant_min, self.quant_max
        arr_fp32 = torch.round(data / scale) + zero_point
        clip_tensor = torch.clamp(arr_fp32, min=cliplow, max=cliphigh)
        return clip_tensor

    def _fake_dequantize_data(self, data: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        deq_tensor = (data - zero_point) * scale
        return deq_tensor

    def _calculate_one_batch_quant_param(self, min_val: torch.Tensor,
                                         max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.original_tensor is not None
        # ref: quark/onnx/quant_utils.py quantize_data_pof2s
        # Step 1, give a initial power of 2 scale
        scale, zero_point = PerTensorPowOf2MinMaxObserver._calculate_one_batch_quant_param(self, min_val, max_val)
        scale_device = scale.device
        # Step 2, find the best through the MSE TODO
        # NOTE move to cpu to compute
        scale = scale.to(device=self.compute_device)
        zero_point = zero_point.to(device=self.compute_device)  # NOTE will not change
        data = self.original_tensor
        scale_mse = scale
        diff_min = float("inf")
        for i in range(self.pos_range):
            new_scale = self._pos2scale(self._scale2pos(scale) + i - 2)
            new_quantized_data = self._fake_quantize_data(data, scale=new_scale, zero_point=zero_point)
            new_data = self._fake_dequantize_data(new_quantized_data, new_scale, zero_point=zero_point)
            diff = torch.pow(new_data - data, 2).sum().item()
            if diff < diff_min:
                diff_min = diff
                scale_mse = new_scale
        # # NOTE used for debug
        # rmin_mse = (self.quant_min - zero_point) * scale_mse # NOTE
        # rmax_mse = (self.quant_max - zero_point) * scale_mse

        # NOTE after using self.original_tensor we can free self.original_tensor
        self.original_tensor = None  # to save memory
        return scale_mse.to(scale_device), zero_point.to(scale_device)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        '''
        collect alll the original tensor, we then perform mes to calculate the best scale
        '''
        if x_orig.numel() == 0:
            return x_orig
        _ = super().forward(x_orig)  # this will init the min & max val
        x = x_orig.detach().to(self.compute_device)  # avoid keeping autograd tape
        self.original_tensor = x
        return x_orig


class PerChannelMinMaxObserver(UniformScalingObserver):

    def __init__(self,
                 qspec: QuantizationSpec,
                 device: Optional[torch.device] = None,
                 eps: float = torch.finfo(torch.float32).eps) -> None:
        super().__init__(qspec, device)

        self.qspec = qspec
        self.ch_axis = qspec.ch_axis

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        return self._forward(x_orig)

    @log_errors
    def _forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        x_orig_device = x_orig.device
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        self.reset_min_max_for_dynamic()
        min_val = self.min_val.to(x_orig_device)
        max_val = self.max_val.to(x_orig_device)
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        if self.ch_axis is not None:
            new_axis_list[self.ch_axis] = 0
        else:
            raise ValueError("ch_axis cannot be None")
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        y = y.unsqueeze(-1) if y.dim() == 1 else y  # if y is bias, usually bias has shape [n]
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch.aminmax(y, dim=1)
        else:
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val = torch.min(min_val_cur, min_val)
            max_val = torch.max(max_val_cur, max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

        input_origin_type = x_orig.dtype
        self.min_val = self.min_val.to(input_origin_type)
        self.max_val = self.max_val.to(input_origin_type)
        return x_orig

    def reset_state(self) -> None:
        self.reset_min_max_vals()

    def reset_min_max_vals(self) -> None:  # TODO
        """Resets the min/max values."""
        if not hasattr(self, "min_val") or not hasattr(self, "max_val"):
            raise AttributeError("PerChannelMinMaxObserver has no attribute 'min_val' or 'max_val'.")
        self.min_val = torch.tensor(float("inf"))
        self.max_val = torch.tensor(float("-inf"))


class PerChannelPowOf2MinMaxObserver(PerChannelMinMaxObserver):

    def __init__(self, qspec: QuantizationSpec, device: Optional[torch.device] = None) -> None:
        super().__init__(qspec, device)
        assert qspec.dtype in [Dtype.int8, Dtype.uint8, Dtype.int32
                               ], "Currently PerChannelPowOf2MinMaxObserver only support int8, uint8 and int32 Dtype"

        self.record_scale_zp: List[torch.Tensor] = []
        self.compute_device = 'cpu'

    def _scale2pos(self, scale: torch.Tensor) -> torch.Tensor:
        """
        Ref: quark/onnx/quant_utils.py scale2pos
        """
        target_device = scale.device
        scale = torch.clamp(scale, min=float(2**-127), max=float(2**127))
        return torch.round(-torch.log2(scale)).to(device=target_device)

    def _pos2scale(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Ref: quark/onnx/quant_utils.py pos2scale
        """
        return torch.pow(2.0, -pos)

    def _calculate_one_batch_quant_param(self, min_val: torch.Tensor,
                                         max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ref: quark/onnx/quant_utils.py: compute_scale_zp
        self.eps = self.eps.to(min_val.dtype).to(min_val.device)

        if not check_min_max_valid(min_val, max_val):
            return torch.tensor([1.0], device=min_val.device.type), torch.tensor([0], device=min_val.device.type)

        quant_min, quant_max = self.quant_min, self.quant_max  # e.g -128, 127
        assert isinstance(quant_min, int)
        assert isinstance(quant_max, int)
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        # quantize range
        if self.symmetric:
            abs_max = torch.max(min_val_neg.abs(), max_val_pos.abs())
            min_val_neg = -1 * abs_max
            max_val_pos = 1 * abs_max

        dr = max_val_pos - min_val_neg
        dq = quant_max - quant_min
        scale = dr / dq
        # scale = dr / dq if dr != 0 else torch.ones(min_val.size(), dtype=torch.float32)
        # assert scale > 0, "scale should larger than 0, please check"
        # NOTE asser 0
        zero_point = torch.tensor(quant_min - min_val_neg / scale).to(dtype=torch.int32)

        pos = self._scale2pos(scale)
        pof2_scale = self._pos2scale(pos)
        new_rmin = torch.min((quant_min - zero_point) * pof2_scale, torch.zeros_like(min_val, dtype=torch.int32))
        new_zero_point = torch.round(quant_min - new_rmin / pof2_scale).to(dtype=torch.int32)
        # TODO refine other dtype haoliang
        if self.symmetric and self.qspec.dtype == Dtype.uint8 and torch.all(new_zero_point == 127):  # [0, 255]
            new_zero_point = torch.full_like(zero_point, 128, dtype=torch.int32)
        return pof2_scale.to(self.scale_torch_dtype), new_zero_point

    def calculate_int_quant_params(self, min_val: torch.Tensor,
                                   max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        device = min_val.device
        one_batch_scale, one_batch_zero_point = self._calculate_one_batch_quant_param(min_val, max_val)
        cat_scale_zp = torch.cat([one_batch_scale, one_batch_zero_point])
        self.record_scale_zp.append(cat_scale_zp.to(device=self.compute_device))
        scale_size = int(cat_scale_zp.size()[0] / 2)
        # select the most common scale & zp
        tensor_tuples = [tuple(tensor.tolist()) for tensor in self.record_scale_zp]
        counter = Counter(tensor_tuples)

        most_common_tensor, freq = counter.most_common(1)[0]

        po2_scale = torch.tensor(most_common_tensor[:scale_size]).to(device)
        zero_point = torch.tensor(most_common_tensor[scale_size:], dtype=torch.int32).to(device).to()
        return po2_scale, zero_point

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        r"""Records the running minimum and maximum of ``x``."""
        x = super().forward(x_orig)
        return x


class PerChannelPowOf2MinMSEObserver(PerChannelPowOf2MinMaxObserver):
    '''
    MINMSE method to get the the proper scale for target input tensor
    More ref: Quark/quark/onnx/quant_utils.py func: quantize_data_pof2s
    '''

    def __init__(self, qspec: QuantizationSpec, device: Optional[torch.device] = None) -> None:
        super().__init__(qspec, device)
        self.original_tensor: Optional[
            torch.Tensor] = None  # not use `register_buffer`, as this function will save dict, waist disk space
        self.pos_range = 5
        self.compute_device = 'cpu'

    def _fake_quantize_data(self, data: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        cliplow, cliphigh = self.quant_min, self.quant_max
        scale_shape = [1 for _ in range(data.dim())]
        assert isinstance(self.ch_axis, int)
        scale_shape[self.ch_axis] = scale.numel()

        arr_fp32 = torch.round(data / scale.view(scale_shape)) + zero_point.view(scale_shape)
        clip_tensor = torch.clamp(arr_fp32, min=cliplow, max=cliphigh)
        return clip_tensor

    def _fake_dequantize_data(self, data: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        scale_shape = [1 for _ in range(data.dim())]
        assert isinstance(self.ch_axis, int)
        scale_shape[self.ch_axis] = scale.numel()
        deq_tensor = (data - zero_point.view(scale_shape)) * scale.view(scale_shape)
        return deq_tensor

    def _calculate_one_batch_quant_param(self, min_val: torch.Tensor,
                                         max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.original_tensor is not None
        # ref: quark/onnx/quant_utils.py quantize_data_pof2s
        # Step 1, give a initial power of 2 scale
        scale, zero_point = PerChannelPowOf2MinMaxObserver._calculate_one_batch_quant_param(self, min_val, max_val)
        scale_device = scale.device
        # Step 2, find the best through the MSE
        # NOTE move to cpu to compute
        scale = scale.to(device=self.compute_device)
        zero_point = zero_point.to(device=self.compute_device)  # NOTE will not change
        data = self.original_tensor
        scale_mse = scale
        diff_min = float("inf")
        for i in range(self.pos_range):
            new_scale = self._pos2scale(self._scale2pos(scale) + i - 2)
            new_quantized_data = self._fake_quantize_data(data, scale=new_scale, zero_point=zero_point)
            new_data = self._fake_dequantize_data(new_quantized_data, new_scale, zero_point=zero_point)
            diff = torch.pow(new_data - data, 2).sum().item()
            if diff < diff_min:
                diff_min = diff
                scale_mse = new_scale
        # # NOTE used for debug
        # rmin_mse = (self.quant_min - zero_point) * scale_mse
        # rmax_mse = (self.quant_max - zero_point) * scale_mse

        # NOTE after using self.original_tensor we can free self.original_tensor
        self.original_tensor = None  # to save memory
        return scale_mse.to(scale_device), zero_point.to(scale_device)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        '''
        collect alll the original tensor, we then perform mes to calculate the best scale
        '''
        if x_orig.numel() == 0:
            return x_orig
        _ = super().forward(x_orig)  # this will init the min & max val
        x = x_orig.detach().to(self.compute_device)  # avoid keeping autograd tape
        self.original_tensor = x
        return x_orig


class PerBlockMXObserver(ObserverBase):

    def __init__(self,
                 qspec: QuantizationSpec,
                 device: Optional[torch.device] = None,
                 eps: float = torch.finfo(torch.float32).eps) -> None:
        super().__init__(qspec=qspec, device=device)
        self.qspec = qspec
        assert self.qspec.dtype in [
            Dtype.mx, Dtype.mx6, Dtype.mx9, Dtype.fp4, Dtype.fp8_e4m3, Dtype.fp8_e5m2, Dtype.fp6_e2m3, Dtype.fp6_e3m2
        ]
        assert qspec.group_size is not None
        assert qspec.ch_axis is not None
        if self.qspec.dtype == Dtype.mx:
            assert qspec.mx_element_dtype is not None
            self.quant_bit = None
        else:
            if self.qspec.dtype == Dtype.mx6:
                self.quant_bit = 5
            elif self.qspec.dtype == Dtype.mx9:
                self.quant_bit = 8

        self.block_size = qspec.group_size
        self.axis = qspec.ch_axis
        self.mx_element_dtype = qspec.mx_element_dtype
        self.eps = eps
        self.is_dynamic = qspec.is_dynamic
        self.amax = torch.tensor(0.0, dtype=torch.float)
        self.scale_format = qspec.scale_format
        self.scale_calculation_mode = qspec.scale_calculation_mode

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        block_x = reshape_to_blocks(x_orig, self.block_size, self.axis)
        if self.qspec.dtype in [Dtype.mx6, Dtype.mx9]:
            block_x.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        amax, _ = torch.max(torch.abs(block_x), dim=-1, keepdim=True)
        amax = amax.squeeze(-1)
        if self.is_dynamic:
            self.amax = amax
        else:
            if self.amax.shape != amax.shape and self.amax.dim() > 0:
                self.amax = torch.tensor(0.0, dtype=torch.float32)
            self.amax = torch.max(self.amax, amax)

        return x_orig

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.scale_format == "e8m0":
            dtype = self.qspec.dtype if self.qspec.mx_element_dtype is None else self.qspec.mx_element_dtype
            _, _, emax = get_dtype_params(dtype)
            if self.scale_calculation_mode is None or self.scale_calculation_mode == "even":
                scale = even_round(self.amax, dtype)
            elif self.scale_calculation_mode == "floor":
                scale = torch.pow(2, torch.floor(torch.log2(self.amax)) - emax)
                scale = scale.masked_fill(scale == 0.0, self.eps)
            elif self.scale_calculation_mode == "ceil":
                scale = torch.pow(2, torch.ceil(torch.log2(self.amax)) - emax)
                scale = scale.masked_fill(scale == 0.0, self.eps)
            else:
                raise ValueError(f"Unsupported scale_calculation_mode: {self.scale_calculation_mode}")
            zero_point = torch.zeros_like(scale)
        elif self.scale_format == "e4m3":
            _, quant_max = calculate_qmin_qmax(self.qspec.dtype)
            scale = torch.div(self.amax, quant_max)
            scale = scale.masked_fill(scale == 0.0, self.eps)

            # two-level scale
            orginal_dtype = scale.dtype
            local_amax = self.amax.float()
            elem_format_max = quant_max
            global_amax = self.amax.max().float()
            scale_format_max = 448.0  # FP8E4M3 scale
            local_unscale = local_amax / elem_format_max
            two_level_scale = scale_format_max * (elem_format_max / global_amax)
            scale = (local_unscale * two_level_scale).to(torch.float8_e4m3fn).float() / two_level_scale
            scale = scale.to(orginal_dtype)

            zero_point = torch.zeros_like(scale)
        elif self.scale_format == "float32":
            _, quant_max = calculate_qmin_qmax(self.qspec.dtype)
            scale = torch.div(self.amax, quant_max)
            scale = scale.masked_fill(scale == 0.0, self.eps)
            zero_point = torch.zeros_like(scale)
        else:
            # TODO: there is no test covering this case.
            scale = t_exponent(self.amax)  # pragma: no cover
            zero_point = torch.zeros_like(scale)

        scale = scale.to(self.scale_torch_dtype)
        return scale, zero_point

    def _calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.calculate_qparams()

    def reset_state(self) -> None:
        self.reset_min_max_vals()

    def reset_min_max_vals(self) -> None:
        """Resets the amax values."""
        if not hasattr(self, "amax"):
            raise AttributeError("PerBlockMXObserver has no attribute 'amax'.")
        self.amax = torch.tensor(float(0.0))


class PerBlockMXDiffsObserver(PerBlockMXObserver):

    def __init__(self,
                 qspec: QuantizationSpec,
                 device: Optional[torch.device] = None,
                 eps: float = torch.finfo(torch.float32).eps,
                 scope: int = 1) -> None:
        super().__init__(qspec=qspec, device=device, eps=eps)
        self.quant_min, self.quant_max = calculate_qmin_qmax(self.qspec.dtype)
        self.round_method = getattr(self.qspec.round_method, "value", None)
        self.qscheme_str_name = getattr(self.qspec.qscheme, "value", None)
        self.scope = scope
        self.mx_element_dtype_value = 'None' if self.mx_element_dtype is None else self.mx_element_dtype.value
        self.mx_dtype = self.mx_element_dtype if self.mx_element_dtype is not None else self.qspec.dtype

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, emax = get_dtype_params(self.mx_dtype)
        init_scale = torch.ceil(torch.log2(self.amax)) - emax
        best_scale = init_scale.clone()

        block_x = reshape_to_blocks(self.last_x, self.block_size, self.axis)
        best_diffs = torch.full(init_scale.shape, float('inf'), device=self.last_x.device)

        for scale_offset in range(0, -self.scope - 1, -1):
            test_scales = init_scale + scale_offset
            scale_factors = torch.pow(2.0, test_scales)
            zero_points = torch.zeros_like(scale_factors)

            # Quantize entire tensor at once using per-group quantization
            quantized = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
                self.dtype.value, self.last_x, scale_factors, zero_points.to(torch.int), self.axis, self.block_size,
                self.quant_min, self.quant_max, self.round_method, self.qscheme_str_name, self.mx_element_dtype_value)
            block_quantized = reshape_to_blocks(quantized, self.block_size, self.axis)

            # Calculate MSE for all blocks at once
            diffs = torch.sum((block_x - block_quantized)**2, dim=-1)

            # Update best_scale only where new diff is better
            better_mask = diffs < best_diffs
            best_scale = torch.where(better_mask, test_scales, best_scale)
            best_diffs = torch.minimum(diffs, best_diffs)

        self.scale = torch.pow(2.0, torch.tensor(best_scale, device=self.amax.device))
        self.scale = self.scale.masked_fill(self.scale == 0.0, self.eps)
        self.zero_point = torch.zeros_like(self.scale)
        return self.scale, self.zero_point

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        # Store input for diffs calculation
        self.last_x = x_orig.detach()
        return super().forward(x_orig)


class PerBlockBFPObserver(ObserverBase):

    def __init__(self,
                 qspec: QuantizationSpec,
                 device: Optional[torch.device] = None,
                 eps: float = torch.finfo(torch.float32).eps) -> None:
        super().__init__(qspec=qspec, device=device)
        self.qspec = qspec
        self.block_size = qspec.group_size
        self.axis = qspec.ch_axis
        self.eps = eps
        self.is_dynamic = qspec.is_dynamic
        self.amax = torch.tensor(0.0, dtype=torch.float)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        block_x = reshape_to_blocks(x_orig.detach(), self.block_size, self.axis)  # type: ignore
        block_x = block_x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        amax, _ = torch.max(torch.abs(block_x), dim=-1)
        if self.is_dynamic:
            self.amax = amax
        else:
            self.amax = torch.max(self.amax, amax)
        scale, zero_point = self.calculate_qparams()
        quantized_block_x_int = fake_quantize_int(inputs=block_x,
                                                  scale=scale,
                                                  zero_point=zero_point,
                                                  axis=-1,
                                                  group_size=self.block_size,
                                                  quant_min=-129,
                                                  quant_max=128,
                                                  qscheme=QSchemeType.per_group.value) / scale.unsqueeze(-1)
        bool_mask = torch.logical_or(quantized_block_x_int >= 128, quantized_block_x_int < -128)
        scale_adjust = torch.pow(2, torch.any(bool_mask, dim=-1).to(torch.float32))
        self.amax *= scale_adjust
        return x_orig

    # TODO: there is no test covering this method.
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        scale = torch.pow(2, torch.floor(torch.log2(self.amax)) - 6)
        scale = scale.masked_fill(scale == 0.0, self.eps)
        scale = scale.to(self.scale_torch_dtype)
        zero_point = torch.zeros_like(scale).to(torch.int32)
        return scale, zero_point

    def _calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.calculate_qparams()

    def reset_state(self) -> None:
        self.reset_min_max_vals()

    def reset_min_max_vals(self) -> None:
        """Resets the min/max values."""
        self.min_val = torch.tensor(float("inf"))
        self.max_val = torch.tensor(float("-inf"))


class PerGroupMinMaxObserver(UniformScalingObserver):

    def __init__(self,
                 qspec: QuantizationSpec,
                 device: Optional[torch.device] = None,
                 eps: float = torch.finfo(torch.float32).eps) -> None:
        super().__init__(qspec, device)

        self.qspec = qspec
        self.ch_axis = cast(int, qspec.ch_axis)
        self.group_size = cast(int, qspec.group_size)
        self.group_count = 1  # Set default value

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        return self._forward(x_orig)

    def _forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        x_orig_device = x_orig.device
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        self.reset_min_max_for_dynamic()
        min_val = self.min_val.to(x_orig_device)
        max_val = self.max_val.to(x_orig_device)

        x_dim = x.size()

        # Get group_count, group_count * group_size = dim_of_axis
        x_group_channel_element_num = x_dim[self.ch_axis]
        if self.group_size == -1:
            self.group_count = 1
        elif x_group_channel_element_num % self.group_size != 0:
            raise ValueError(
                f"The number of element per dimension ch_axis={self.ch_axis} is {x_group_channel_element_num} which is not divisible by group_size={self.group_size}. The `group_size` used should be updated in the QuantizationSpec."
            )
        else:
            self.group_count = x_group_channel_element_num // self.group_size

        # Transpose the dim of ch_axis and the end dim of the tensor.
        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        if self.ch_axis is not None:
            new_axis_list[self.ch_axis] = -1
        else:
            raise ValueError("ch_axis cannot be None")
        new_axis_list[-1] = self.ch_axis
        if self.group_size is not None:
            x = x.permute(new_axis_list)
            if self.group_size > 0:
                x = x.reshape(-1, self.group_size)
        else:
            raise ValueError("group_size cannot be None")

        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch.aminmax(x, dim=1)
        else:
            min_val_cur, max_val_cur = torch.aminmax(x, dim=1)
            min_val = torch.min(min_val_cur, min_val)
            max_val = torch.max(max_val_cur, max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

        input_origin_type = x_orig.dtype
        self.min_val = self.min_val.to(input_origin_type)
        self.max_val = self.max_val.to(input_origin_type)
        return x_orig

    def calculate_qparams(self, min_val: torch.Tensor, max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters."""
        if self.dtype in [Dtype.fp8_e4m3, Dtype.fp8_e5m2]:
            _scale, _zero_point = super().calculate_fp8_quant_parameters(min_val, max_val)
        else:
            _scale, _zero_point = super().calculate_int_quant_params(min_val, max_val)
        _scale = _scale.reshape(-1, self.group_count)
        _zero_point = _zero_point.reshape(-1, self.group_count)
        return _scale, _zero_point


class PerTensorHistogramObserver(UniformScalingObserver):

    calib_bin_edges: torch.Tensor
    calib_hist: torch.Tensor

    def __init__(self, qspec: QuantizationSpec, device: Optional[torch.device] = None) -> None:
        super().__init__(qspec)

        self.register_buffer("calib_bin_edges", torch.tensor([], device=device), persistent=False)
        self.register_buffer("calib_hist", torch.tensor([], device=device), persistent=False)

        # TODO: make the value can be set
        self._skip_zeros = False
        self._num_bins = 2048

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        """
        Records the running histogram of ``x_orig``.

        Raises:
        - ValueError: If the `self.symmetric` argument is False.

        """
        self.device = x_orig.device
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.float()
        self.reset_min_max_for_dynamic()

        with torch.no_grad():
            if self._skip_zeros:
                x = x[torch.where(x != 0)]

            assert isinstance(x, torch.Tensor)
            if self.symmetric is not None and self.symmetric is False:
                x_max = x.max().item()
                x_min = x.min().item()
            else:
                if torch.min(x) < 0.0:
                    x = x.abs()
                x_max = x.max().item()
                x_min = 0.0

            if self.calib_bin_edges.nelement() == 0 and self.calib_hist.nelement() == 0:
                self.calib_hist = torch.histc(x, bins=self._num_bins, min=x_min, max=x_max)
                self.calib_bin_edges = torch.linspace(x_min, x_max, self._num_bins + 1)
            else:
                if x_min < self.calib_bin_edges[0]:
                    width = (self.calib_bin_edges[1] - self.calib_bin_edges[0]).item()
                    self._num_bins += int(((self.calib_bin_edges[0] - x_min) / width).ceil().item())
                    self.calib_bin_edges = torch.arange(x_min - width,
                                                        self.calib_bin_edges[-1].item(),
                                                        width,
                                                        device=x.device)
                if x_max > self.calib_bin_edges[-1]:
                    width = (self.calib_bin_edges[1] - self.calib_bin_edges[0]).item()
                    self._num_bins += int(((x_max - self.calib_bin_edges[-1]) / width).ceil().item())
                    self.calib_bin_edges = torch.arange(self.calib_bin_edges[0].item(),
                                                        x_max + width,
                                                        width,
                                                        device=x.device)
                assert x_max <= self.calib_bin_edges[-1]
                assert x_min >= self.calib_bin_edges[0]

                hist = torch.histc(x,
                                   bins=self._num_bins,
                                   min=self.calib_bin_edges[0].item(),
                                   max=self.calib_bin_edges[-1].item())
                hist[:self.calib_hist.numel()] += self.calib_hist
                self.calib_hist = hist

            assert isinstance(self.calib_hist, torch.Tensor)
            assert isinstance(self.calib_bin_edges, torch.Tensor)
            self.calib_hist = self.calib_hist.to(self.device)
            self.calib_bin_edges = self.calib_bin_edges.to(self.device)

        return x_orig


class PerTensorHistogramObserverPro(UniformScalingObserver):
    '''
    A wrap of pytorch version observer: HistogramObserver
    '''

    def __init__(self,
                 qspec: QuantizationSpec,
                 device: Optional[torch.device] = None,
                 bins: int = 256,
                 reduce_range: bool = False,
                 upsample_rate: int = 384) -> None:
        super().__init__(qspec, device)
        self.qscheme = qspec.qscheme
        self.symmetric = qspec.symmetric

        if self.qscheme != QSchemeType.per_tensor:
            raise ValueError("PerTensorHistogramObserverPro only supports per_tensor")
        torch_qscheme = torch.per_tensor_symmetric if self.symmetric else torch.per_tensor_affine

        dtype = qspec.dtype
        if dtype not in [Dtype.uint4, Dtype.int8, Dtype.uint8]:
            raise ValueError("PerTensorHistogramObserverPro only supports 4bit and 8bit")
        else:
            quant_min, quant_max = calculate_qmin_qmax(dtype)
            if dtype == Dtype.uint4:
                torch_dtype = torch.quint4x2
            elif dtype == Dtype.int8:
                torch_dtype = torch.qint8
            elif dtype == Dtype.uint8:
                torch_dtype = torch.quint8

        # The argument `upsample_rate` was removed from `HistogramObserver` in torch 2.5.
        if is_torch_greater_or_equal_2_5():
            kwargs = {}
        else:  # pragma: no cover
            kwargs = {"upsample_rate": upsample_rate}

        self.histogram = HistogramObserver(qscheme=torch_qscheme,
                                           dtype=torch_dtype,
                                           bins=bins,
                                           quant_min=quant_min,
                                           quant_max=quant_max,
                                           reduce_range=reduce_range,
                                           **kwargs)

    def forward(self, x_orig: torch.Tensor) -> Any:
        return self.histogram(x_orig)

    def _calculate_qparams(self) -> Any:
        new_min, new_max = self.histogram._non_linear_param_search()
        return self.histogram._calculate_qparams(new_min, new_max)


class PerTensorPercentileObserver(PerTensorHistogramObserver):

    def __init__(self, qspec: QuantizationSpec, device: Optional[torch.device] = None) -> None:
        super().__init__(qspec, device)

        # TODO: make the value can be set
        self._skip_zeros = True
        self._num_bins = 4200
        self.percentile = 99.99999999

    def _calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters."""
        self.min_val, self.max_val = self._calculate_min_and_max_using_percentile()
        return self.calculate_qparams(self.min_val, self.max_val)

    def _calculate_min_and_max_using_percentile(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(self.calib_hist, torch.Tensor)
        assert isinstance(self.calib_bin_edges, torch.Tensor)
        return self.get_min_max_by_percentile(self.calib_hist, self.calib_bin_edges, self.percentile)

    def get_min_max_by_percentile(self, histogram: torch.Tensor, bin_edges: torch.Tensor,
                                  percentile: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the minimum and maximum values of a histogram at a specified percentile.

        Parameters:
        - histogram (torch.Tensor): A tensor representing the histogram of the data. Each element
        in the histogram represents the frequency of data in the corresponding bin.
        - bin_edges (torch.Tensor): A tensor containing the edge values that correspond to the
        bins represented in the histogram. There should be one more element in `bin_edges` than
        in `histogram`.
        - percentile (int): The percentile at which to determine the minimum and maximum values.
        The value should be an integer between 0 and 100.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors. The first tensor
        is the value at the specified percentile, and the second tensor is the value at the
        complementary percentile (i.e., 100-percentile).

        Raises:
        - ValueError: If the `percentile` argument is not within the range 0 to 100.
        """
        if percentile < 0 or percentile > 100:
            raise ValueError("Percentile value must be between 0 and 100.")

        # Return None if no data is available
        if bin_edges is None and histogram is None:
            return None

        # Calculate cumulative distribution function
        hist_total = histogram.sum()
        cumulative_dist = torch.cumsum(histogram / hist_total, dim=0)

        if self.symmetric is not None and self.symmetric is False:
            target_pct_one_side = (100.0 - percentile) / 200.0

            upper_idx = (cumulative_dist >= target_pct_one_side).nonzero().min().item()
            assert isinstance(upper_idx, int), "Index must be an integer"
            max_value = bin_edges[upper_idx]

            lower_idx = (cumulative_dist <= (1 - target_pct_one_side)).nonzero().min().item()
            assert isinstance(lower_idx, int), "Index must be an integer"
            min_value = bin_edges[lower_idx]

        else:
            target_pct = percentile / 100.0
            cumulative_dist_max = cumulative_dist[-1].item()
            assert isinstance(cumulative_dist_max, float)
            target_pct = min(target_pct, cumulative_dist_max)

            upper_idx = (cumulative_dist >= target_pct).nonzero().min().item()
            assert isinstance(upper_idx, int), "Index must be an integer"
            max_value = bin_edges[upper_idx]

            min_value = torch.tensor(0, device='cpu')

        max_value = max_value.to(self.device)
        min_value = min_value.to(self.device)
        return min_value, max_value


class PerTensorMSEObserver(PerTensorHistogramObserver):

    def __init__(self, qspec: QuantizationSpec, device: Optional[torch.device] = None) -> None:
        super().__init__(qspec, device)

    def _calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.min_val, self.max_val = self._calculate_min_and_max_using_mse()
        return self.calculate_qparams(self.min_val, self.max_val)

    def _calculate_min_and_max_using_mse(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(self.calib_hist, torch.Tensor)
        assert isinstance(self.calib_bin_edges, torch.Tensor)
        return self.get_min_max_by_mse(self.calib_hist, self.calib_bin_edges)

    def get_min_max_by_mse(self,
                           calib_hist: torch.Tensor,
                           calib_bin_edges: torch.Tensor,
                           stride: int = 1,
                           start_bin: int = 2045) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns amax that minimizes MSE of the collected histogram."""
        # If calibrator hasn't collected any data, return none
        if calib_bin_edges is None and calib_hist is None:
            return None

        counts = calib_hist
        edges = calib_bin_edges

        counts = counts.to(self.device)
        edges = edges.to(self.device)

        centers = (edges[1:] + edges[:-1]) / 2

        mses = []
        arguments = []

        min_value = torch.tensor(0, device='cpu')
        min_value = min_value.to(self.device)

        for i in range(start_bin, len(centers), stride):
            amax = centers[i]
            if self.dtype in [Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.uint8]:
                quant_centers = self.int_fake_tensor_quant(centers, min_value, amax)
            elif self.dtype in [Dtype.fp8_e4m3, Dtype.fp8_e5m2]:
                quant_centers = self.scaling_fp8(centers, amax)
            else:
                raise ValueError(f"Invalid dtype {self.dtype}. dtype must be a positive integer, fp8_e4m3 or fp8_e5m2."
                                 )  # pragma: no cover

            mse = ((quant_centers - centers)**2 * counts).mean()

            mses.append(mse.cpu())
            arguments.append(i)

        argmin = torch.argmin(torch.stack(mses))
        calib_amax = centers[arguments[argmin]]

        calib_amax = calib_amax.to(self.device)

        return min_value, calib_amax

    def int_fake_tensor_quant(self, X: torch.Tensor, min_value: torch.Tensor, max_value: torch.Tensor) -> torch.Tensor:
        scale, zero_point = self.calculate_int_quant_params(min_value, max_value)
        assert isinstance(self.quant_min, int)
        assert isinstance(self.quant_max, int)
        X = torch.fake_quantize_per_tensor_affine(X, scale.to(torch.float), zero_point.to(torch.int), self.quant_min,
                                                  self.quant_max)
        return X

    @log_errors
    def scaling_fp8(self, X: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
        min_norm, max_norm = calculate_qmin_qmax(self.dtype)
        X_orig_dtype = X.dtype
        scale = amax / max_norm
        X = X / scale
        X = torch.clamp(X, min=min_norm, max=max_norm)
        if self.dtype == Dtype.fp8_e4m3:
            fp8_dtype = torch.float8_e4m3fn
        elif self.dtype == Dtype.fp8_e5m2:
            fp8_dtype = torch.float8_e5m2
        else:
            raise ValueError(
                f"Invalid dtype {self.dtype}. Supported fp8 formats: fp8_e4m3, fp8_e5m2.")  # pragma: no cover
        X = X.to(fp8_dtype).to(X_orig_dtype) * scale
        return X


OBSERVER_CLASSES = {
    PlaceholderObserver, UniformScalingObserver, PerTensorMinMaxObserver, PerChannelMinMaxObserver, PerBlockMXObserver,
    PerBlockBFPObserver, PerGroupMinMaxObserver, PerTensorHistogramObserver, PerTensorPercentileObserver,
    PerTensorMSEObserver, PerTensorHistogramObserverPro, PerBlockMXDiffsObserver
}

OBSERVER_MAP = {observer_cls.__name__: observer_cls for observer_cls in OBSERVER_CLASSES}
