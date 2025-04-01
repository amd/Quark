#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any, Optional, List, Dict, Tuple, Union
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.version
from quark.torch.quantization.config.config import QuantizationConfig
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.utils.pack import create_pack_method
from quark.torch.quantization.config.type import QSchemeType
from quark.torch.export.constants import AWQ_LOAD_MAP, LOAD_MAP, SAVE_MAP, AWQ_SAVE_MAP
from quark.torch.export.nn.modules.realquantizer import RealQuantizerBase
from collections import OrderedDict

SCALED_MM_AVAILABLE_DEV: Optional[str] = None


def normalize_e4m3fn_to_e4m3fnuz(weight: torch.Tensor,
                                 qinput: torch.Tensor,
                                 weight_scale: torch.Tensor,
                                 input_scale: Optional[torch.Tensor] = None
                                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    '''normalize_e4m3fn_to_e4m3fnuz for amd gpu'''
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
        self.weight_quantizer: Optional[RealQuantizerBase] = None
        self.bias_quantizer: Optional[RealQuantizerBase] = None
        self.input_quantizer: Optional[RealQuantizerBase] = None
        self.output_quantizer: Optional[RealQuantizerBase] = None


class QParamsLinear(QparamsOperator):

    def __init__(self,
                 linear: nn.Linear,
                 custom_mode: str,
                 pack_method: Optional[str] = "reorder",
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()

        reorder = True if pack_method == "reorder" else False
        self._custom_mode: str = custom_mode
        self._init_qparamlinear(linear, reorder, quant_config)

    def _init_qparamlinear(self,
                           linear: nn.Linear,
                           reorder: bool,
                           quant_config: Optional[QuantizationConfig] = None) -> None:
        if isinstance(linear, QuantLinear) and quant_config is None:
            self.weight: torch.nn.Parameter = torch.nn.Parameter(linear.weight)
            self.bias: Optional[torch.nn.Parameter] = linear.bias if linear.bias is not None else None

            device = linear.weight.device

            # We always pass `torch.float32` as `float_dtype` for now, meaning that
            # initialized scales will always be in float32, even if serialized in float16.
            # Although suboptimal, in case float32 scales are used and the pre-loaded model is in float16,
            # we can not rely on e.g. `linear.weight.dtype`, as later on `load_state_dict` would load fp32 scales
            # into fp16 instanciated parameters.
            float_dtype = torch.float32

            if linear.weight_qspec is not None and linear.weight_quantizer is not None:
                self.weight_quantizer = RealQuantizerBase.from_fake_quantizer(qspec=linear.weight_qspec,
                                                                              quantizer=linear.weight_quantizer,
                                                                              reorder=reorder,
                                                                              real_quantized=True,
                                                                              device=device,
                                                                              float_dtype=float_dtype)

            if linear.bias_qspec is not None and linear.bias_quantizer is not None:
                self.bias_quantizer = RealQuantizerBase.from_fake_quantizer(qspec=linear.bias_qspec,
                                                                            quantizer=linear.bias_quantizer,
                                                                            reorder=reorder,
                                                                            real_quantized=True,
                                                                            device=device,
                                                                            float_dtype=float_dtype)

            if linear.input_qspec is not None and linear.input_quantizer is not None:
                self.input_quantizer = RealQuantizerBase.from_fake_quantizer(qspec=linear.input_qspec,
                                                                             quantizer=linear.input_quantizer,
                                                                             reorder=reorder,
                                                                             real_quantized=False,
                                                                             device=device,
                                                                             float_dtype=float_dtype)

            if linear.output_qspec is not None and linear.output_quantizer is not None:
                self.output_quantizer = RealQuantizerBase.from_fake_quantizer(qspec=linear.output_qspec,
                                                                              quantizer=linear.output_quantizer,
                                                                              reorder=reorder,
                                                                              real_quantized=False,
                                                                              device=device,
                                                                              float_dtype=float_dtype)
            self._real_quantize()
        elif isinstance(linear, nn.Linear) and quant_config is not None:
            device = linear.weight.device
            float_dtype = torch.float32
            in_features = linear.in_features
            out_features = linear.out_features

            if linear.bias is not None:
                self.bias = torch.nn.Parameter(torch.empty((out_features, ), device=device, dtype=float_dtype),
                                               requires_grad=False)
            else:
                self.bias = None

            # create weight, scale, zeropoint and initialize weight quantized parameters with correct shape, dtype.
            # TODO: weight uses infer_packed_shape func , what about sacle zeropoint
            if quant_config.weight is not None:
                quant_torch_dtype = quant_config.weight.dtype.to_torch_packed_dtype()
                pack_method = create_pack_method(
                    qscheme=quant_config.weight.qscheme.value,  # type: ignore[union-attr]
                    dtype=quant_config.weight.dtype.value)

                # Retrieve the quantized weight shape. For example, int4/uint4 does packing on torch.int32.
                weight_shape, scale_shape, zero_point_shape = pack_method.infer_packed_shape(
                    unpacked_shape=(out_features, in_features),
                    quantization_spec=quant_config.weight,
                    legacy=False,
                    custom_mode=self._custom_mode)

                self.weight = torch.nn.Parameter(torch.empty(weight_shape, device=device, dtype=quant_torch_dtype),
                                                 requires_grad=False)
                self.weight_quantizer = RealQuantizerBase.from_fake_quantizer(qspec=quant_config.weight,
                                                                              quantizer=None,
                                                                              reorder=reorder,
                                                                              real_quantized=True,
                                                                              device=device,
                                                                              scale_shape=scale_shape,
                                                                              zero_point_shape=zero_point_shape,
                                                                              float_dtype=float_dtype)
            else:
                self.weight = torch.nn.Parameter(torch.empty((out_features, in_features),
                                                             device=device,
                                                             dtype=float_dtype),
                                                 requires_grad=False)

            # Initialize bias quantized parameters with correct shape, dtype.
            if quant_config.bias is not None:
                self.bias_quantizer = RealQuantizerBase.from_fake_quantizer(qspec=quant_config.bias,
                                                                            quantizer=None,
                                                                            reorder=reorder,
                                                                            real_quantized=True,
                                                                            device=device,
                                                                            float_dtype=float_dtype)
                if quant_config.bias.qscheme != QSchemeType.per_tensor:
                    raise NotImplementedError(
                        "Reloading a quantized model using QParamsLinear with the bias quantized per channel or per group is not supported."
                    )
                if hasattr(self.bias_quantizer, "transpose_scale"):
                    # bias need not transpose_scale
                    self.bias_quantizer.transpose_scale = False  # type: ignore

            # Initialize input quantized parameters with correct shape, dtype.
            if quant_config.input_tensors is not None:
                self.input_quantizer = RealQuantizerBase.from_fake_quantizer(qspec=quant_config.input_tensors,
                                                                             quantizer=None,
                                                                             reorder=reorder,
                                                                             real_quantized=False,
                                                                             device=device,
                                                                             float_dtype=float_dtype)
                if quant_config.input_tensors.is_dynamic:
                    raise NotImplementedError(
                        "QParamsLinear does not support is_dynamic=True for now. Please open an issue.")
                if quant_config.input_tensors.qscheme != QSchemeType.per_tensor:
                    raise NotImplementedError(
                        "Reloading a quantized model using QParamsLinear with the input quantized per channel or per group is not supported. Please open an issue."
                    )

            # Initialize output quantized parameters with correct shape, dtype.
            if quant_config.output_tensors is not None:
                self.output_quantizer = RealQuantizerBase.from_fake_quantizer(qspec=quant_config.output_tensors,
                                                                              quantizer=None,
                                                                              reorder=reorder,
                                                                              real_quantized=False,
                                                                              device=device,
                                                                              float_dtype=float_dtype)
                if quant_config.output_tensors.is_dynamic:
                    raise NotImplementedError(
                        "QParamsLinear does not support is_dynamic=True for now. Please open an issue.")
                if quant_config.output_tensors.qscheme != QSchemeType.per_tensor:
                    raise NotImplementedError(
                        "Reloading a quantized model using QParamsLinear with the output quantized per channel or per group is not supported."
                    )
        else:
            raise ValueError(f"Unsupported module type: {type(linear)}")

    @classmethod
    def from_module(
        cls,
        linear: nn.Linear,
        custom_mode: str,
        pack_method: Optional[str] = "reorder",
        quant_config: Optional[QuantizationConfig] = None,
    ) -> "QParamsLinear":
        '''
        Build a QParamsLinear from a QuantLinear or nn.Linear.
        Initialize the shape and data type of weight and bias in importing.
        Initialize weight and bias in exporting.
        '''
        qparamslinear = cls(linear=linear, custom_mode=custom_mode, pack_method=pack_method, quant_config=quant_config)
        return qparamslinear

    def can_use_fp8_kernel(self) -> bool:
        '''check use_fp8_kernel or not'''
        # pertensor only now, w and inp should be quantized
        if SCALED_MM_AVAILABLE_DEV is None:
            return False

        if not (self.input_quantizer and self.weight_quantizer):
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
        '''
        Dequantizes quantized weight/bias, runs a linear in high precision and apply QDQ on the (input)activation/output if required.
        '''
        dtype = args[0].dtype
        use_fp8_kernel = self.can_use_fp8_kernel()
        if use_fp8_kernel:
            assert self.input_quantizer is not None
            assert self.weight_quantizer is not None
            input = args[0]
            if self.bias is not None:
                if dtype == torch.float32:
                    raise ValueError("Bias is not supported when out_dtype is set to Float32")
                if self.bias.dtype == torch.float32:
                    # Bias must be either Half or BFloat16.
                    bias = self.bias.to(torch.float16)
                else:
                    bias = self.bias.to(input.dtype)
            else:
                bias = None

            max_value = 448 if self.input_quantizer.qspec.dtype == Dtype.fp8_e4m3 else 57344
            input_2d = input.view(-1, input.shape[-1])
            input_2d = input_2d / self.input_quantizer.scale
            input_2d = torch.clamp(input_2d, min=-max_value, max=max_value)
            qinput = input_2d.to(self.input_quantizer.qspec.dtype.to_torch_packed_dtype())

            weight = self.weight.t()
            output_shape = [*input.shape[:-1], weight.shape[1]]
            input_scale = self.input_quantizer.scale
            weight_scale = self.weight_quantizer.scale
            if SCALED_MM_AVAILABLE_DEV == "hip":
                weight, qinput, weight_scale, input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                                weight=weight,
                                qinput=qinput,
                                weight_scale=weight_scale,
                                input_scale=input_scale)

            output: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            # Both scale_a and scale_b must be float (fp32) tensors.
            output = torch._scaled_mm(qinput,
                                      weight,
                                      out_dtype=dtype,
                                      scale_a=input_scale.to(torch.float32),
                                      scale_b=weight_scale.to(torch.float32),
                                      bias=bias)
            # returns tuple for torch < 2.5 and a single value in torch >= 2.5
            if type(output) is tuple and len(output) == 2:
                output = output[0]
            quant_output: torch.Tensor = self._get_qoutput(output).to(dtype)  # type: ignore
            quant_output = quant_output.view(*output_shape)
        else:
            qinput = self._get_qinput(args[0]).to(dtype)
            qweight = self._get_qweight(self.weight).to(dtype)
            qbias = self._get_qbias(self.bias)
            if qbias is not None:
                qbias = qbias.to(dtype)
            qoutput = F.linear(qinput, qweight, bias=qbias)
            quant_output = self._get_qoutput(qoutput).to(dtype)

        return quant_output

    def _get_qweight(self, x: Parameter) -> torch.Tensor:
        if self.weight_quantizer is not None:
            x = self.weight_quantizer(x.data)
            assert isinstance(x, torch.Tensor)
            return x
        else:
            return x.data

    def _get_qbias(self, x: Optional[Parameter]) -> Optional[torch.Tensor]:
        if self.bias_quantizer is not None and x is not None:
            x = self.bias_quantizer(x.data)
            assert isinstance(x, torch.Tensor)
            return x
        else:
            return x.data if x is not None else x

    def _get_qinput(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_quantizer is not None:
            x = self.input_quantizer(x)
            assert isinstance(x, torch.Tensor)
            return x
        else:
            return x

    def _get_qoutput(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_quantizer is not None:
            x = self.output_quantizer(x)
            assert isinstance(x, torch.Tensor)
            return x
        else:
            return x

    def _real_quantize(self) -> None:
        '''
        Calls `_to_real_quantize_params` to do weight and bias real quantization on low-bit datatypes, and calls `pack_qinfo` to do scale and zero_point packing.
        '''
        self._to_real_quantize_params()
        self.pack_qinfo()

    def _to_real_quantize_params(self) -> None:
        '''
        Calls `to_real_quantize_params` of real_quantizer to do weight and bias real quantization on low-bit datatypes
        '''
        if self.weight_quantizer is not None and self.weight_quantizer.qspec.is_dynamic is False:
            w_res = self.weight_quantizer.to_real_quantize_params(self.weight)
            self.weight = nn.Parameter(w_res, requires_grad=False)

        # Replaces the high-precision fake quantized bias (QDQ) by a low-precision bias.
        if self.bias is not None and self.bias_quantizer is not None and self.bias_quantizer.qspec.is_dynamic is False:
            b_res = self.bias_quantizer.to_real_quantize_params(self.bias)
            self.bias = nn.Parameter(b_res, requires_grad=False)

    def pack_qinfo(self) -> None:
        '''
        Calls `RealQuantizer.pack_zero_point`` and `RealQuantizer.maybe_transpose_scale` to do scale, zero_point packing if required.
        '''
        if self.weight_quantizer is not None:
            self.weight_quantizer.pack_zero_point()
            self.weight_quantizer.maybe_transpose_scale()
        if self.bias_quantizer is not None:
            self.bias_quantizer.pack_zero_point()
            self.bias_quantizer.maybe_transpose_scale()
        if self.input_quantizer is not None:
            self.input_quantizer.pack_zero_point()
            self.input_quantizer.maybe_transpose_scale()
        if self.output_quantizer is not None:
            self.output_quantizer.pack_zero_point()
            self.output_quantizer.maybe_transpose_scale()

    def state_dict(self, *args: Any, destination: Any = None, prefix: str = "", keep_vars: bool = False) -> Any:
        # Save scale, zeropoint of realquantizer directly at the qparamlinear level.
        # Since the recursive call of `state_dict`, Overloading `_save_to_state_dict` can not prevent real_quantizer from calling its `_save_to_state_dict`.
        if self._custom_mode == "awq":
            name_map = AWQ_SAVE_MAP
        elif self._custom_mode == "fp8" or self._custom_mode == "quark":
            name_map = SAVE_MAP
        else:
            raise ValueError(f"Not supported custom_mode{self._custom_mode}")

        if destination is None:  # pragma: no cover
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        # seperate scale from tensor when mx
        is_mx_export = self.weight_quantizer is not None and self.weight_quantizer.qspec.dtype.value == "mx"
        if self.weight_quantizer is not None and is_mx_export:
            scale_weight_shape = list(self.weight.shape)
            scale_weight = self.weight.reshape(-1, 17)
            scale = scale_weight[:, :1].reshape(scale_weight_shape[0], -1).contiguous()
            weight = scale_weight[:, 1:].reshape(scale_weight_shape[0], -1).contiguous()

            destination[prefix + name_map["weight"]] = weight
            destination[prefix + name_map["weight_scale"]] = scale
            # return destination

        if self.weight is not None and not is_mx_export:
            destination[prefix + name_map["weight"]] = self.weight if keep_vars else self.weight.detach()
        if self.bias is not None:
            destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()

        # scales and zero_points
        if self.weight_quantizer is not None and self.weight_quantizer.qspec.is_dynamic is False and not is_mx_export:
            if hasattr(self.weight_quantizer, "scale") and self.weight_quantizer.scale is not None:
                destination[prefix + name_map[
                    "weight_scale"]] = self.weight_quantizer.scale if keep_vars else self.weight_quantizer.scale.detach(
                    )
            if hasattr(self.weight_quantizer, "zero_point") and self.weight_quantizer.zero_point is not None:
                destination[prefix + name_map[
                    "weight_zero_point"]] = self.weight_quantizer.zero_point if keep_vars else self.weight_quantizer.zero_point.detach(
                    )

        # TODO: Does bias get special treatment in awq cases?
        if self.bias_quantizer is not None and self.bias_quantizer.qspec.is_dynamic is False:
            if self.bias_quantizer.scale is not None:
                destination[
                    prefix +
                    "bias_scale"] = self.bias_quantizer.scale if keep_vars else self.bias_quantizer.scale.detach()
            if self.bias_quantizer.zero_point is not None:
                destination[
                    prefix +
                    "bias_zero_point"] = self.bias_quantizer.zero_point if keep_vars else self.bias_quantizer.zero_point.detach(
                    )
        if self.input_quantizer is not None and self.input_quantizer.qspec.is_dynamic is False:
            if self.input_quantizer.scale is not None:
                destination[
                    prefix +
                    "input_scale"] = self.input_quantizer.scale if keep_vars else self.input_quantizer.scale.detach()
            if self.input_quantizer.zero_point is not None:
                destination[
                    prefix +
                    "input_zero_point"] = self.input_quantizer.zero_point if keep_vars else self.input_quantizer.zero_point.detach(
                    )
        if self.output_quantizer is not None and self.output_quantizer.qspec.is_dynamic is False:
            if self.output_quantizer.scale is not None:
                destination[
                    prefix +
                    "output_scale"] = self.output_quantizer.scale if keep_vars else self.output_quantizer.scale.detach(
                    )
            if self.output_quantizer.zero_point is not None:
                destination[
                    prefix +
                    "output_zero_point"] = self.output_quantizer.zero_point if keep_vars else self.output_quantizer.zero_point.detach(
                    )
        if hasattr(self, "weight_scale1") and self.weight_scale1 is not None:
            destination[prefix + "weight_scale1"] = self.weight_scale1 if keep_vars else self.weight_scale1.detach()
        return destination

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        if self._custom_mode == "awq":
            rename_map = AWQ_LOAD_MAP
        else:
            rename_map = LOAD_MAP
        keys = list(state_dict.keys())

        for name in keys:
            full_name = name
            # is None represent common custom_mode
            if rename_map is not None:
                to_remap = name[len(prefix):]
                suffix = rename_map[to_remap]
                full_name = prefix + suffix
            param = state_dict[name]
            # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
            if ("scale" in name or "zero_point" in name) and param.ndim == 0 and param.ndim == 1:
                param = param[0]
            if full_name != name:
                state_dict[full_name] = param
                del state_dict[name]

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)  # type: ignore
