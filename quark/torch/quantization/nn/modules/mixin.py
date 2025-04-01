#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
from typing import Optional, Any, Dict
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase
from quark.torch.quantization.config.config import QuantizationConfig, QuantizationSpec


class QuantMixin(torch.nn.Module):

    def init_quantizer(self, quant_config: QuantizationConfig, device: torch.device, **kwargs: Any) -> None:

        self._input_qspec = quant_config.input_tensors
        self._output_qspec = quant_config.output_tensors
        self._weight_qspec = quant_config.weight
        self._bias_qspec = quant_config.bias
        self._input_quantizer = FakeQuantizeBase.get_fake_quantize(self._input_qspec, device, **
                                                                   kwargs) if self._input_qspec is not None else None
        self._output_quantizer = FakeQuantizeBase.get_fake_quantize(self._output_qspec, device, **
                                                                    kwargs) if self._output_qspec is not None else None
        self._weight_quantizer = FakeQuantizeBase.get_fake_quantize(self._weight_qspec, device, **
                                                                    kwargs) if self._weight_qspec is not None else None
        self._bias_quantizer = FakeQuantizeBase.get_fake_quantize(self._bias_qspec, device, **
                                                                  kwargs) if self._bias_qspec is not None else None

    @property
    def input_quantizer(self) -> Optional[FakeQuantizeBase]:
        return self._input_quantizer

    @property
    def weight_quantizer(self) -> Optional[FakeQuantizeBase]:
        return self._weight_quantizer

    @property
    def output_quantizer(self) -> Optional[FakeQuantizeBase]:
        return self._output_quantizer

    @property
    def bias_quantizer(self) -> Optional[FakeQuantizeBase]:
        return self._bias_quantizer

    @property
    def input_qspec(self) -> Optional[QuantizationSpec]:
        return self._input_qspec

    @property
    def output_qspec(self) -> Optional[QuantizationSpec]:
        return self._output_qspec

    @property
    def weight_qspec(self) -> Optional[QuantizationSpec]:
        return self._weight_qspec

    @property
    def bias_qspec(self) -> Optional[QuantizationSpec]:
        return self._bias_qspec

    def get_quant_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._input_quantizer is not None:
            x = self._input_quantizer(x)
            assert isinstance(x, torch.Tensor)
            return x
        else:
            return x

    def get_quant_output(self, x: torch.Tensor) -> torch.Tensor:
        if self._output_quantizer is not None:
            x = self._output_quantizer(x)
            assert isinstance(x, torch.Tensor)
            return x
        else:
            return x

    def get_quant_weight(self, x: torch.Tensor) -> torch.Tensor:
        if self._weight_quantizer is not None:
            x = self._weight_quantizer(x)
            assert isinstance(x, torch.Tensor)
            return x
        else:
            return x

    def get_quant_bias(self, x: Any) -> Any:
        if self._bias_quantizer and x is not None:
            x = self._bias_quantizer(x)
            assert isinstance(x, torch.Tensor)
            return x
        else:
            return x

    def load_quant_params(self, params_dict: Dict[str, torch.Tensor]) -> None:
        device = next(self.parameters()).device
        if hasattr(self, "_input_quantizer") and self._input_quantizer is not None:
            if params_dict.get("input_scale", None) is not None:
                self._input_quantizer.scale.data = params_dict["input_scale"].to(device)
            if params_dict.get("input_zero_point", None) is not None:
                self._input_quantizer.zero_point.data = params_dict["input_zero_point"].to(device)

        if hasattr(self, "_output_quantizer") and self._output_quantizer is not None:
            if params_dict.get("output_scale", None) is not None:
                self._output_quantizer.scale.data = params_dict["output_scale"].to(device)
            if params_dict.get("output_zero_point", None) is not None:
                self._output_quantizer.zero_point.data = params_dict["output_zero_point"].to(device)

        if hasattr(self, "_weight_quantizer") and self._weight_quantizer is not None:
            if params_dict.get("weight_scale", None) is not None:
                self._weight_quantizer.scale.data = params_dict["weight_scale"].to(device)
            if params_dict.get("weight_zero_point", None) is not None:
                self._weight_quantizer.zero_point.data = params_dict["weight_zero_point"].to(device)

        if hasattr(self, "_bias_quantizer") and self._bias_quantizer is not None:
            if params_dict.get("bias_scale", None) is not None:
                self._bias_quantizer.scale.data = params_dict["bias_scale"].to(device)
            if params_dict.get("bias_zero_point", None) is not None:
                self._bias_quantizer.zero_point.data = params_dict["bias_zero_point"].to(device)
