#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any, Optional, List, Dict, Tuple, Union, cast
import torch
import re
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.version
from quark.torch.quantization.config.config import QuantizationConfig, QuantizationSpec
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.utils.pack import create_pack_method
from quark.torch.quantization.config.type import QSchemeType
from quark.torch.export.constants import AWQ_LOAD_MAP, AWQ_SAVE_MAP
from quark.torch.export.nn.modules.realquantizer import RealQuantizerBase, SequentialRealQuantizer, get_real_quantizer
from torch.distributed._tensor import distribute_tensor, DTensor, Replicate
from quark.torch.utils.device import e4m3fn_to_e4m3fnuz

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
        self.weight_quantizer: Union[RealQuantizerBase, SequentialRealQuantizer, None] = None
        self.bias_quantizer: Union[RealQuantizerBase, SequentialRealQuantizer, None] = None
        self.input_quantizer: Union[RealQuantizerBase, SequentialRealQuantizer, None] = None
        self.output_quantizer: Union[RealQuantizerBase, SequentialRealQuantizer, None] = None


class QParamsLinear(torch.nn.Linear, QparamsOperator):

    def __init__(self,
                 linear: nn.Linear,
                 custom_mode: str,
                 pack_method: Optional[str] = "reorder",
                 quant_config: Optional[QuantizationConfig] = None):
        bias = True if linear.bias is not None else False
        super(QParamsLinear, self).__init__(linear.in_features, linear.out_features, bias)

        reorder = True if pack_method == "reorder" else False
        self._custom_mode: str = custom_mode
        self._init_qparamlinear(linear, reorder, quant_config)
        self._quant_dict = None

    # In the original __init__ function of torch.nn.Linear,
    # the reset_parameters function is called, which takes up a lot of time.
    # This is the reason why inplace ops replacement is slow.
    # Therefore, overload this function in this class to skip the parameter
    # allocation operation, reducing the time of inplace ops replacement.
    def reset_parameters(self) -> None:
        pass

    def _init_qparamlinear(self,
                           linear: nn.Linear,
                           reorder: bool,
                           quant_config: Optional[QuantizationConfig] = None) -> None:
        """Initialize QParamsLinear from either a QuantLinear or nn.Linear module.

        Args:
            linear: Input linear module (QuantLinear or nn.Linear)
            reorder: Whether to reorder parameters
            quant_config: Optional quantization configuration
        """
        if isinstance(linear, QuantLinear) and quant_config is None:
            self._init_from_quantlinear(linear, reorder)
        elif isinstance(linear, nn.Linear) and quant_config is not None:
            self._init_from_linear(linear, reorder, quant_config)
        else:
            raise ValueError(f"Unsupported module type: {type(linear)}")

    def _init_from_quantlinear(self, linear: QuantLinear, reorder: bool) -> None:
        if linear.weight.device != torch.device("meta"):
            self.weight: torch.nn.Parameter = torch.nn.Parameter(linear.weight)  # Keep it in the CPU.
            self.bias = linear.bias if linear.bias is not None else None
            device = linear.weight.device
        else:  # we can copy it directly, don't care device because just export.
            self.weight = torch.nn.Parameter(linear._hf_hook.weights_map["weight"].data)
            self.bias = torch.nn.Parameter(
                linear._hf_hook.weights_map["bias"].data) if linear.bias is not None else None
            device = linear._hf_hook.execution_device

        float_dtype = torch.float32

        # Initialize quantizers if they exist
        quantizer_configs = [("weight", linear.weight_qspec, linear.weight_quantizer, True),
                             ("bias", linear.bias_qspec, linear.bias_quantizer, True),
                             ("input", linear.input_qspec, linear.input_quantizer, False),
                             ("output", linear.output_qspec, linear.output_quantizer, False)]

        for name, qspec, quantizer, real_quantized in quantizer_configs:
            if qspec is not None and quantizer is not None:
                setattr(
                    self, f"{name}_quantizer",
                    get_real_quantizer(qspec=qspec,
                                       quantizer=quantizer,
                                       reorder=reorder,
                                       real_quantized=real_quantized,
                                       device=device,
                                       float_dtype=float_dtype))
        self._real_quantize()

    def _init_from_linear(self, linear: nn.Linear, reorder: bool, quant_config: QuantizationConfig) -> None:
        device = linear.weight.device
        float_dtype = torch.float32
        in_features = linear.in_features
        out_features = linear.out_features

        # Initialize bias
        if linear.bias is not None:
            self.bias = torch.nn.Parameter(torch.empty((out_features, ), device=device, dtype=float_dtype),
                                           requires_grad=False)
        else:
            self.bias = None

        # Initialize weight and weight quantizer
        if quant_config.weight is not None:
            weight_configs = [quant_config.weight] if not isinstance(quant_config.weight, list) else quant_config.weight
            assert all(weight_spec.is_dynamic is not True for weight_spec in weight_configs), \
                "Dynamic quantization is not supported for weight in `QParamsLinear`, " \
                "got quant_config.weight.is_dynamic=True."
            self._init_weight_quantizer(linear, quant_config.weight, reorder, device, float_dtype)
        else:
            self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=float_dtype),
                                             requires_grad=False)

        # Initialize other quantizers
        self._init_other_quantizers(quant_config, reorder, device, float_dtype)

    def _init_weight_quantizer(self, linear: nn.Linear, weight_spec: Union[QuantizationSpec, List[QuantizationSpec]],
                               reorder: bool, device: torch.device, float_dtype: torch.dtype) -> None:
        weight_specs = [weight_spec] if not isinstance(weight_spec, list) else weight_spec

        weight_shapes: List[Tuple[int, ...]] = []
        scale_shapes: List[Tuple[int, ...]] = []
        zero_point_shapes: List[Tuple[int, ...]] = []
        quant_torch_dtypes: List[torch.dtype] = []
        last_tensor_quantizer_index = 0
        for i, spec in enumerate(weight_specs):
            # record the index of the last tensor quantizer
            if not spec.is_scale_quant:
                last_tensor_quantizer_index = i
            quant_torch_dtype = spec.dtype.to_torch_packed_dtype()
            pack_method = create_pack_method(
                qscheme=spec.qscheme.value,  # type: ignore[union-attr]
                dtype=spec.dtype.value)

            # for scale quant, the quantized tensor is the scale tensor of the previous quantizer
            # so we need to get the scale shape of the previous quantizer
            unpacked_shape = (linear.out_features,
                              linear.in_features) if not spec.is_scale_quant else scale_shapes[i - 1]
            weight_shape, scale_shape, zero_point_shape = pack_method.infer_packed_shape(unpacked_shape=unpacked_shape,
                                                                                         quantization_spec=spec,
                                                                                         legacy=False,
                                                                                         custom_mode=self._custom_mode)
            weight_shapes.append(weight_shape)
            scale_shapes.append(scale_shape)
            zero_point_shapes.append(zero_point_shape)
            quant_torch_dtypes.append(quant_torch_dtype)
        # the quantized weight shape is determined by the last tensor quantizer
        weight_shape = weight_shapes[last_tensor_quantizer_index]
        quant_torch_dtype = quant_torch_dtypes[last_tensor_quantizer_index]
        self.weight = torch.nn.Parameter(torch.empty(weight_shape, device=device, dtype=quant_torch_dtype),
                                         requires_grad=False)

        s_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]] = scale_shapes[0] if isinstance(
            weight_spec, QuantizationSpec) else scale_shapes
        zp_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]] = zero_point_shapes[0] if isinstance(
            weight_spec, QuantizationSpec) else zero_point_shapes
        self.weight_quantizer: Union[RealQuantizerBase,
                                     SequentialRealQuantizer] = get_real_quantizer(qspec=weight_spec,
                                                                                   quantizer=None,
                                                                                   reorder=reorder,
                                                                                   real_quantized=True,
                                                                                   device=device,
                                                                                   scale_shape=s_shape,
                                                                                   zero_point_shape=zp_shape,
                                                                                   float_dtype=float_dtype)

    def _init_other_quantizers(self, quant_config: QuantizationConfig, reorder: bool, device: torch.device,
                               float_dtype: torch.dtype) -> None:
        # Define quantizer configurations
        quantizer_specs = {
            'bias': {
                'spec': quant_config.bias,
                'real_quantized': True
            },
            'input': {
                'spec': quant_config.input_tensors,
                'real_quantized': False
            },
            'output': {
                'spec': quant_config.output_tensors,
                'real_quantized': False
            }
        }

        for name, config in quantizer_specs.items():
            spec = config['spec']
            spec = cast(Optional[Union[QuantizationSpec, List[QuantizationSpec]]], spec)
            if spec is not None:
                # Validate quantization scheme
                error_msg = (f"Reloading a quantized model using QParamsLinear with the {name} "
                             "static quantized per channel or per group is not supported. "
                             "Please open an issue.")

                specs: List[QuantizationSpec] = [spec] if not isinstance(spec, list) else spec
                assert all(spec.qscheme == QSchemeType.per_tensor or spec.is_dynamic for spec in specs), error_msg

                # Create quantizer
                quantizer = get_real_quantizer(qspec=spec,
                                               quantizer=None,
                                               reorder=reorder,
                                               real_quantized=bool(config["real_quantized"]),
                                               device=device,
                                               float_dtype=float_dtype)

                # Handle transpose_scale for bias quantizer
                if name == "bias" and hasattr(quantizer, "transpose_scale"):
                    quantizer.transpose_scale = False  # type: ignore

                # Set the quantizer
                setattr(self, f"{name}_quantizer", quantizer)

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

        if isinstance(self.input_quantizer, SequentialRealQuantizer) or isinstance(self.weight_quantizer,
                                                                                   SequentialRealQuantizer):
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
        output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        use_fp8_kernel = self.can_use_fp8_kernel()
        if use_fp8_kernel:
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

            if not isinstance(input, DTensor):
                assert self.input_quantizer is not None
                assert self.weight_quantizer is not None

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
            else:
                assert self._quant_dict is not None
                if self.input_quantizer is None:
                    self.input_quantizer = self._quant_dict['input_quantizer']
                if self.weight_quantizer is None:
                    self.weight_quantizer = self._quant_dict['weight_quantizer']

                assert self.input_quantizer is not None
                assert self.weight_quantizer is not None

                input_scale = self.input_quantizer.scale
                weight_scale = self.weight_quantizer.scale

                # Distribute the tensor to create a DTensor
                if not isinstance(input_scale, DTensor):
                    input_scale = distribute_tensor(input_scale.to(torch.float32),
                                                    device_mesh=input.device_mesh,
                                                    placements=[Replicate()])
                    self.input_quantizer.scale = input_scale

                if not isinstance(weight_scale, DTensor):
                    weight_scale = distribute_tensor(weight_scale.to(torch.float32),
                                                     device_mesh=input.device_mesh,
                                                     placements=[Replicate()])
                    self.weight_quantizer.scale = weight_scale

                max_value = 448 if self.input_quantizer.qspec.dtype == Dtype.fp8_e4m3 else 57344
                input_2d = input.view(-1, input.shape[-1])
                input_2d = input_2d / input_scale
                input_2d = torch.clamp(input_2d, min=-max_value, max=max_value)
                qinput = input_2d.to(self.input_quantizer.qspec.dtype.to_torch_packed_dtype())

                weight = self.weight
                weight = weight.permute(1, 0)

                output_shape = [*input.shape[:-1], weight.shape[1]]
                if SCALED_MM_AVAILABLE_DEV == "hip":
                    qinput, input_scale = e4m3fn_to_e4m3fnuz(tensor=qinput, tensor_scale=input_scale)

                output = torch._scaled_mm(qinput,
                                          weight,
                                          out_dtype=dtype,
                                          scale_a=input_scale,
                                          scale_b=weight_scale,
                                          bias=None)
                if type(output) is tuple and len(output) == 2:
                    output = output[0]

                if self.bias is not None:
                    output = output + bias

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
        weight_quantizer = self.weight_quantizer
        if self._quant_dict is not None:
            weight_quantizer = self._quant_dict['weight_quantizer']

        if weight_quantizer is not None:
            x = weight_quantizer(x.data)
            assert isinstance(x, torch.Tensor)
            return x
        else:
            return x.data

    def _get_qbias(self, x: Optional[Parameter]) -> Optional[torch.Tensor]:
        bias_quantizer = self.bias_quantizer
        if self._quant_dict is not None and 'bias_quantizer' in self._quant_dict:
            bias_quantizer = self._quant_dict['bias_quantizer']

        if bias_quantizer is not None and x is not None:
            x = bias_quantizer(x.data)
            assert isinstance(x, torch.Tensor)
            return x
        else:
            return x.data if x is not None else x

    def _get_qinput(self, x: torch.Tensor) -> torch.Tensor:
        input_quantizer = self.input_quantizer
        if self._quant_dict is not None and 'input_quantizer' in self._quant_dict:
            input_quantizer = self._quant_dict['input_quantizer']

        if input_quantizer is not None:
            x = input_quantizer(x)
            assert isinstance(x, torch.Tensor)
            return x
        else:
            return x

    def _get_qoutput(self, x: torch.Tensor) -> torch.Tensor:
        output_quantizer = self.output_quantizer
        if self._quant_dict is not None and 'output_quantizer' in self._quant_dict:
            output_quantizer = self._quant_dict['output_quantizer']

        if output_quantizer is not None:
            x = output_quantizer(x)
            assert isinstance(x, torch.Tensor)
            return x
        else:
            return x

    def _real_quantize(self) -> None:
        '''
        Calls `_to_real_quantize_params` to do weight and bias real quantization on low-bit datatypes, and calls `pack_qinfo` to do scale and zero_point packing.
        '''
        # the order of maybe_convert_and_transpose_scale and pack_zero_point could not be changed
        self._to_real_quantize_params()
        self.pack_qinfo()

    def _to_real_quantize_params(self) -> None:
        '''
        Calls `to_real_quantize_params` of real_quantizer to do weight and bias real quantization on low-bit datatypes
        '''
        if self.weight_quantizer is not None and self.weight_quantizer.is_dynamic is False:
            w_res = self.weight_quantizer.to_real_quantize_params(self.weight)
            self.weight = nn.Parameter(w_res, requires_grad=False)

        # Replaces the high-precision fake quantized bias (QDQ) by a low-precision bias.
        if self.bias is not None and self.bias_quantizer is not None and self.bias_quantizer.is_dynamic is False:
            b_res = self.bias_quantizer.to_real_quantize_params(self.bias)
            self.bias = nn.Parameter(b_res, requires_grad=False)

    def pack_qinfo(self) -> None:
        '''
        Calls `RealQuantizer.pack_zero_point`` and `RealQuantizer.maybe_convert_and_transpose_scale` to do scale, zero_point packing if required.
        '''
        quantizers_names = ["weight_quantizer", "bias_quantizer", "input_quantizer", "output_quantizer"]
        for name in quantizers_names:
            quantizer = getattr(self, f"{name}", None)
            if quantizer is not None:
                # the order of maybe_convert_and_transpose_scale and pack_zero_point could not be changed
                quantizer.maybe_convert_and_transpose_scale()
                quantizer.pack_zero_point()

    def state_dict(self, *args: Any, destination: Any = None, prefix: str = "", keep_vars: bool = False) -> Any:
        # Save scale, zeropoint of realquantizer directly at the qparamlinear level.
        # Since the recursive call of `state_dict`, Overloading `_save_to_state_dict` can not prevent real_quantizer from calling its `_save_to_state_dict`.
        destination = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        params_names = [
            "weight_quantizer.*scale", "bias_quantizer.*scale", "input_quantizer.*scale", "output_quantizer.*scale"
        ]
        for param_name in params_names:
            # find all keys that both contains prefix string and param_name, param_name is a regex
            keys = [key for key in destination.keys() if re.match(prefix + param_name, key)]
            if len(keys) == 0:
                continue
            index_keys = [key.split(".")[-2] for key in keys]
            if len(keys) == 1 and not index_keys[0].isdigit():
                tensor_name = index_keys[0].split("_")[-2]
                destination[prefix + tensor_name + "_" + "scale"] = destination[keys[0]]
                # replace the last "scale" in keys[0] with "zero_point"
                zero_point_key = keys[0].rsplit(".", 1)[0] + ".zero_point"
                if zero_point_key in destination:
                    destination[prefix + tensor_name + "_" + "zero_point"] = destination[zero_point_key]
                    del destination[zero_point_key]
                del destination[keys[0]]
            elif all(index_key.isdigit() for index_key in index_keys):
                # sort keys by index_keys from small to large
                keys = [x for _, x in sorted(zip(index_keys, keys), key=lambda pair: pair[0])]
                tensor_name = keys[0].split(".")[-3].split("_")[-2]
                for i, key in enumerate(keys):
                    if i == 0:
                        suffix = ""
                    else:
                        suffix = "_" + str(i + 1)
                    destination[prefix + tensor_name + "_" + "scale" + suffix] = destination[key]
                    # replace the last "scale" in key with "zero_point"
                    zero_point_key = key.rsplit(".", 1)[0] + ".zero_point"
                    if zero_point_key in destination:
                        destination[prefix + tensor_name + "_" + "zero_point" + suffix] = destination[zero_point_key]
                        del destination[zero_point_key]
                    del destination[key]

        if self._custom_mode == "awq":
            for quark_name, awq_name in AWQ_SAVE_MAP.items():
                for key in list(destination.keys()):
                    if (prefix + quark_name) == key:
                        destination[prefix + awq_name] = destination[key]
                        del destination[key]

        is_mx_export = self.weight_quantizer is not None and \
            not isinstance(self.weight_quantizer, SequentialRealQuantizer) and \
            self.weight_quantizer.qspec.dtype.value == "mx"
        if is_mx_export:
            assert self.weight_quantizer.qspec.mx_element_dtype is not None, \
                "mx_element_dtype should not be None"
            mx_element_dtype = self.weight_quantizer.qspec.mx_element_dtype.value
            reshape_shape = 17 if mx_element_dtype == "fp4" else 25
            scale_weight_shape = list(self.weight.shape)
            scale_weight = self.weight.reshape(-1, reshape_shape)
            scale = scale_weight[:, :1].reshape(scale_weight_shape[0], -1).contiguous()
            weight = scale_weight[:, 1:].reshape(scale_weight_shape[0], -1).contiguous()
            destination[prefix + "weight"] = weight
            destination[prefix + "weight_scale"] = scale.view(torch.uint8)

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
        scale_quantizer_map = {
            "weight_scale*": "weight_quantizer",
            "bias_scale*": "bias_quantizer",
            "input_scale*": "input_quantizer",
            "output_scale*": "output_quantizer"
        }
        for scale_key, quantizer_name in scale_quantizer_map.items():
            keys = [key for key in state_dict.keys() if re.match(prefix + scale_key, key)]
            if len(keys) == 0:
                continue
            # Sort: non-numbered keys first, then numbered keys by numerical order
            # for example, if keys is ["weight_scale_1", "weight_scale_2", "weight_scale"],
            # the sorted keys should be ["weight_scale", "weight_scale_1", "weight_scale_2"]
            sorted_keys = sorted(keys, key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0)
            quantizer = getattr(self, quantizer_name, None)
            if quantizer is not None:
                if isinstance(quantizer, RealQuantizerBase):
                    real_key = prefix + quantizer_name + ".scale"
                    state_dict[real_key] = state_dict[sorted_keys[0]]
                    del state_dict[sorted_keys[0]]
                    zero_point_key = prefix + sorted_keys[0].split(".")[-1].replace("scale", "zero_point")
                    if zero_point_key in state_dict and getattr(quantizer, "zero_point", None) is not None:
                        real_zero_point_key = prefix + quantizer_name + ".zero_point"
                        state_dict[real_zero_point_key] = state_dict[zero_point_key]
                        del state_dict[zero_point_key]
                elif isinstance(quantizer, SequentialRealQuantizer):
                    key_index = 0
                    for i, module in enumerate(quantizer):
                        real_key = prefix + quantizer_name + "." + str(i) + ".scale"
                        if getattr(module, "scale", None) is not None and module.has_static_scale():
                            state_dict[real_key] = state_dict[sorted_keys[key_index]]
                            del state_dict[sorted_keys[key_index]]
                            zero_point_key = prefix + sorted_keys[key_index].split(".")[-1].replace(
                                "scale", "zero_point")
                            if zero_point_key in state_dict and getattr(module, "zero_point", None) is not None:
                                real_zero_point_key = prefix + quantizer_name + "." + str(i) + ".zero_point"
                                state_dict[real_zero_point_key] = state_dict[zero_point_key]
                                del state_dict[zero_point_key]
                            key_index += 1

        if self._custom_mode == "awq":
            for quark_name, awq_name in AWQ_LOAD_MAP.items():
                keys = [key for key in state_dict.keys() if (prefix + quark_name) == key]
                for key in keys:
                    state_dict[prefix + awq_name] = state_dict[key]
                    del state_dict[key]

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)  # type: ignore
