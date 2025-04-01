#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Optional, Dict, Any, Union, Tuple
import gc
import torch
import torch.nn as nn
from quark.torch.quantization.config.type import Dtype
from quark.shares.utils.log import ScreenLogger, log_errors

logger = ScreenLogger(__name__)


def clear_memory(weight: Optional[torch.Tensor] = None) -> None:
    if weight is not None:
        del weight
    gc.collect()
    torch.cuda.empty_cache()


def validate_qmin_qmax(quant_min: int, quant_max: int) -> None:
    assert (quant_min < quant_max), "qmin must be less than qmax."


def calculate_qmin_qmax(dtype: Dtype) -> tuple[Union[int, float], Union[int, float]]:

    # Fallback onto default 8-bit qmin and qmax calculation if dynamic range is not used.
    if dtype == Dtype.int8:
        return -128, 127
    elif dtype == Dtype.uint8:
        return 0, 255
    elif dtype == Dtype.int4:
        return -8, 7
    elif dtype == Dtype.uint4:
        return 0, 15
    elif dtype == Dtype.fp8_e4m3:
        return -448, 448
    elif dtype == Dtype.fp8_e5m2:
        return -57344, 57344
    elif dtype == Dtype.bfloat16:
        return torch.finfo(torch.bfloat16).min, torch.finfo(torch.bfloat16).max
    elif dtype == Dtype.float16:
        return torch.finfo(torch.float16).min, torch.finfo(torch.float16).max
    elif dtype == Dtype.fp6_e3m2:
        return -28.0, 28.0
    elif dtype == Dtype.fp6_e2m3:
        return -7.5, 7.5
    elif dtype == Dtype.fp4:
        return -6.0, 6.0
    else:
        raise ValueError("The qmin and qmax of {dtype} are not defined")


def get_num_bits(dtype: Dtype) -> Optional[Union[int, Tuple[int, int]]]:
    if dtype in [Dtype.int4, Dtype.uint4]:
        return 4
    elif dtype in [Dtype.int8, Dtype.uint8]:
        return 8
    elif dtype == Dtype.fp8_e4m3:
        return (4, 3)
    else:
        return None


def set_op_by_name(layer: Union[nn.Module, nn.ModuleList], name: str, new_module: nn.Module) -> None:
    """
    Replaces a submodule in a given neural network layer with a new module(e.g. quantized module). The submodule to be
    replaced is identified by the 'name' parameter, which specifies the name of the submodule
    using dot notation. If the name includes dots, it navigates through nested submodules
    to find the specific layer to replace. Otherwise, it directly replaces the submodule in the
    provided layer.

    Parameters:
    - layer: The top-level module containing the submodule.
    - name: name of the submodule, split by dots.
    - new_module: The new module to replace the existing one, for example the quantized module.
    """
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit() and isinstance(mod_, nn.ModuleList):
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def get_op_by_name(layer: Union[nn.Module, nn.ModuleList], name: str) -> Union[nn.Module, nn.ModuleList]:
    levels = name.split('.')
    mod_ = layer
    for l_idx in range(len(levels)):
        if levels[l_idx].isdigit() and isinstance(mod_, nn.ModuleList):
            mod_ = mod_[int(levels[l_idx])]
        else:
            mod_ = getattr(mod_, levels[l_idx])
    return mod_


def deep_compare(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> bool:
    if type(dict1) != type(dict2):
        return False
    if isinstance(dict1, dict):
        if dict1.keys() != dict2.keys():
            return False
        return all(deep_compare(dict1[k], dict2[k]) for k in dict1)
    elif isinstance(dict1, list):
        return set(dict1) == set(dict2)
    else:
        return dict1 == dict2


_FORMAT_CACHE: Dict[Dtype, tuple[int, int, int]] = {}


def get_dtype_params(dtype: Union[str, Dtype]) -> tuple[int, int, int]:
    if isinstance(dtype, str):
        dtype = Dtype.from_str(dtype)

    if dtype in _FORMAT_CACHE:
        return _FORMAT_CACHE[dtype]

    if dtype == Dtype.int8:
        ebits, mbits = 0, 8
        emax = 0
    elif dtype == Dtype.int4:
        ebits, mbits = 0, 4
        emax = 0
    elif dtype == Dtype.int2:
        ebits, mbits = 0, 2
        emax = 0
    elif dtype == Dtype.fp8_e5m2:
        ebits, mbits = 5, 2
        emax = 2**(ebits - 1) - 1
    elif dtype == Dtype.fp8_e4m3:
        ebits, mbits = 4, 3
        emax = 2**(ebits - 1)
    elif dtype == Dtype.fp6_e3m2:
        ebits, mbits = 3, 2
        emax = 2**(ebits - 1)
    elif dtype == Dtype.fp6_e2m3:
        ebits, mbits = 2, 3
        emax = 2**(ebits - 1)
    elif dtype == Dtype.fp4:
        ebits, mbits = 2, 1
        emax = 2**(ebits - 1)
    elif dtype == Dtype.float16:
        ebits, mbits = 5, 10
        emax = 2**(ebits - 1) - 1
    elif dtype == Dtype.bfloat16:
        ebits, mbits = 8, 7
        emax = 2**(ebits - 1) - 1
    else:
        raise Exception("Unknown element format %s" % dtype)

    _FORMAT_CACHE[dtype] = (ebits, mbits, emax)

    return ebits, mbits, emax


def pad_to_blocks(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    num_elem_to_be_padded = block_size - x.size(-1) % block_size
    if num_elem_to_be_padded == block_size:
        return x, 0
    return torch.nn.functional.pad(x, (0, num_elem_to_be_padded)), num_elem_to_be_padded


def reshape_to_blocks(x: torch.Tensor, block_size: int, axis: int) -> torch.Tensor:
    if axis > x.dim() - 1:
        raise IndexError('Axis is larger than number of tensor dimensions')

    x = x.transpose(axis, -1)
    x = x.reshape(-1, x.size(-1))
    x, _ = pad_to_blocks(x, block_size)
    return x.reshape(x.size(0), x.size(1) // block_size, block_size)


@log_errors
def exponent_frexp_no_exception(t: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        t_dtype = t.dtype

        if t_dtype == torch.float32:
            int_tensor = t.view(torch.int32)
            t_exp = ((int_tensor >> 23) & 0xFF) - 127
        elif t_dtype == torch.bfloat16:
            int_tensor = t.view(torch.int16)
            t_exp = ((int_tensor >> 7) & 0xFF) - 127
        elif t_dtype == torch.float16:
            # zero has a different exponent here comparing to the original version
            # exponent bias is now defined as -15
            int_tensor = t.view(torch.int16)
            t_exp = ((int_tensor >> 10) & 0x1F) - 15
        else:
            raise ValueError(f"Unsupported data type: {t_dtype}")  # pragma: no cover

        return t_exp


def t_exponent(t: torch.Tensor) -> torch.Tensor:
    """Get element exponents

    Args:
        t (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Exponents for each elements. NaN and Inf are treated as zeros.

    """
    with torch.no_grad():
        t = torch.nan_to_num(t, nan=0, posinf=0, neginf=0)
        t_exp = exponent_frexp_no_exception(t)

        return t_exp
