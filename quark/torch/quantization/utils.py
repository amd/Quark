#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import gc
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quark.common.data_type import BaseFP8_E5M3
from quark.common.utils.import_utils import is_transformers_available
from quark.common.utils.log import ScreenLogger, log_errors
from quark.torch.quantization.config.type import Dtype
from quark.torch.utils.accelerate_helper import clone_align_devices_hook
from quark.torch.utils.torch_utils import setattr_recursive

if is_transformers_available():
    from transformers.feature_extraction_utils import BatchFeature

logger = ScreenLogger(__name__)

MOE_EXPERT_INPUT_QUANTIZER_PATTERN = re.compile(
    r"^(.*\.mlp\.experts\.)\d+\.(gate_proj|up_proj|down_proj|w1|w3|w2|v1|linear|linear_v|linear_1)\._?input_quantizer(?:\.(\d+))?$"
)


def is_aiter_available() -> bool:
    """Backward-compatible proxy for Aiter availability checks."""
    from quark.torch.kernel.aiter import is_aiter_available as _is_aiter_available

    return _is_aiter_available()


def _transfer_hf_hook(old_module: nn.Module, new_module: nn.Module) -> None:
    """Move accelerate's ``_hf_hook`` from a replaced module to its replacement.

    When a model is dispatched with ``device_map`` (e.g. ``"balanced"`` for a
    large diffusion transformer), accelerate attaches an ``AlignDevicesHook``
    that moves a module's inputs to its execution device and its outputs back.
    Swapping the module via ``setattr`` would drop that hook and break
    cross-device dispatch, so we clone it onto the replacement. Best-effort:
    failures (or no hook / no accelerate) leave the replacement hook-free.
    """
    if not hasattr(old_module, "_hf_hook"):
        return
    try:
        from accelerate.hooks import add_hook_to_module

        add_hook_to_module(new_module, clone_align_devices_hook(old_module._hf_hook))
    except Exception as e:  # pragma: no cover - depends on accelerate / device_map
        logger.warning(f"Could not transfer accelerate _hf_hook to native layer: {e}")


@dataclass(frozen=True)
class RuntimeOptions:
    """Runtime toggles for native inference conversion."""

    # Select native linear implementation at conversion time.
    # Supported values: "auto", "fp8_per_tensor", "mxfp4".
    native_linear_mode: str = "auto"
    # Enable preshuffle kernels for the FP8 per-tensor native linear.
    use_preshuffle: bool = False
    # For SVDQuant ErrorCorrectedModule layers: run the low-rank correction on
    # a second CUDA stream so it overlaps the residual MXFP4 GEMM.
    svdquant_overlap_streams: bool = False


def clear_memory(weight: torch.Tensor | None = None) -> None:
    if weight is not None:
        del weight
    gc.collect()
    torch.cuda.empty_cache()


def validate_qmin_qmax(quant_min: int, quant_max: int) -> None:
    assert quant_min < quant_max, "qmin must be less than qmax."


def calculate_qmin_qmax(dtype: Dtype) -> tuple[int | float, int | float]:
    # Fallback onto default 8-bit qmin and qmax calculation if dynamic range is not used.
    if dtype == Dtype.int8:
        return -128, 127
    elif dtype == Dtype.uint8:
        return 0, 255
    elif dtype == Dtype.int16:
        return -(2**15), 2**15 - 1
    elif dtype == Dtype.int32:
        return -(2**31), 2**31 - 1
    elif dtype == Dtype.int4:
        return -8, 7
    elif dtype == Dtype.uint4:
        return 0, 15
    elif dtype == Dtype.int3:
        return -4, 3
    elif dtype == Dtype.int2:
        return -2, 1
    elif dtype == Dtype.fp8_e4m3:
        return -448, 448
    elif dtype == Dtype.fp8_e5m2:
        return -57344, 57344
    elif dtype == Dtype.fp8_e5m3:
        return BaseFP8_E5M3.min_value, BaseFP8_E5M3.max_value
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
        raise ValueError(f"The qmin and qmax of {dtype} are not defined")


def get_num_bits(dtype: Dtype) -> int | tuple[int, int] | None:
    if dtype in [Dtype.int4, Dtype.uint4]:
        return 4
    elif dtype in [Dtype.int8, Dtype.uint8]:
        return 8
    elif dtype in [Dtype.int16, Dtype.uint16]:
        return 16
    elif dtype in [Dtype.int32]:
        return 32
    elif dtype == Dtype.fp8_e4m3:
        return (4, 3)
    else:
        return None


def deep_compare(dict1: dict[str, Any], dict2: dict[str, Any]) -> bool:
    if type(dict1) is not type(dict2):
        return False
    if isinstance(dict1, dict):
        if dict1.keys() != dict2.keys():
            return False
        return all(deep_compare(dict1[k], dict2[k]) for k in dict1)
    elif isinstance(dict1, list):
        return set(dict1) == set(dict2)
    else:
        return dict1 == dict2


_FORMAT_CACHE: dict[Dtype, tuple[int, int, int]] = {}


def get_dtype_params(dtype: str | Dtype) -> tuple[int, int, int]:
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
    elif dtype == Dtype.int3:
        ebits, mbits = 0, 3
        emax = 0
    elif dtype == Dtype.int2:
        ebits, mbits = 0, 2
        emax = 0
    elif dtype == Dtype.fp8_e5m2:
        ebits, mbits = 5, 2
        emax = 2 ** (ebits - 1) - 1
    elif dtype == Dtype.fp8_e4m3:
        ebits, mbits = 4, 3
        emax = 2 ** (ebits - 1)
    elif dtype == Dtype.fp6_e3m2:
        ebits, mbits = 3, 2
        emax = 2 ** (ebits - 1)
    elif dtype == Dtype.fp6_e2m3:
        ebits, mbits = 2, 3
        emax = 2 ** (ebits - 1)
    elif dtype == Dtype.fp4:
        ebits, mbits = 2, 1
        emax = 2 ** (ebits - 1)
    elif dtype == Dtype.float16:
        ebits, mbits = 5, 10
        emax = 2 ** (ebits - 1) - 1
    elif dtype == Dtype.bfloat16:
        ebits, mbits = 8, 7
        emax = 2 ** (ebits - 1) - 1
    else:
        raise ValueError(f"Unknown element format {dtype}")

    _FORMAT_CACHE[dtype] = (ebits, mbits, emax)

    return ebits, mbits, emax


def pad_to_blocks(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    num_elem_to_be_padded = block_size - x.size(-1) % block_size
    if num_elem_to_be_padded == block_size:
        return x, 0
    return torch.nn.functional.pad(x, (0, num_elem_to_be_padded)), num_elem_to_be_padded


def reshape_to_blocks(x: torch.Tensor, block_size: int, axis: int) -> torch.Tensor:
    if axis > x.dim() - 1:
        raise IndexError("Axis is larger than number of tensor dimensions")

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


def even_round(max_abs: torch.Tensor, dtype: Dtype | str) -> torch.Tensor:
    f32_min_normal = 2 ** (-127 + 1)
    zero_fill_value = torch.tensor(f32_min_normal, dtype=max_abs.dtype).to(torch.float32).item()
    if max_abs.dtype == torch.float32:
        max_abs_float32 = max_abs.clone()
    else:
        max_abs_float32 = max_abs.to(torch.float32)

    nan_mask = torch.isnan(max_abs_float32)
    zero_mask = max_abs_float32 == 0
    max_abs_as_int32 = max_abs_float32.view(torch.int32)
    _ebits, mbits, emax = get_dtype_params(dtype)

    # Rounding strategy between [2**n, 2**(n+1)]:
    # x in [2**n, 2**n *(1 + 0.5 + 0.25)[ => round to 2**n
    # x in [2**n * 1.75, 2**(n + 1)] => round to 2**(n+1)
    #
    # `val_to_add` overflows on the exponent bits in case we round up.
    val_to_add = 1 << (23 - mbits - 1)

    # Mask for the 9 leftmost bits (1 sign, 8 exponent) of the float32 representation.
    fp32_sign_exponent_mask = ((1 << (8 + 1)) - 1) << 23

    # Use in-place integer operations to avoid creating large temporary tensors.
    max_abs_as_int32.add_(val_to_add)
    max_abs_as_int32.bitwise_and_(fp32_sign_exponent_mask)
    max_abs_float32 = max_abs_as_int32.view(torch.float32)
    max_abs_float32.masked_fill_(zero_mask, zero_fill_value)
    max_abs_float32.masked_fill_(nan_mask, float("nan"))
    max_abs_float32.log2_()
    max_abs_float32.floor_()
    max_abs_float32.sub_(emax)
    max_abs_float32.clamp_(min=-127, max=127)
    max_abs_float32.exp2_()

    # See section 6.3 of https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    # for the computation below.
    return max_abs_float32


def count_calibration_tokens(
    dataloader: DataLoader[torch.Tensor]
    | DataLoader[list[dict[str, torch.Tensor]]]
    | DataLoader[dict[str, torch.Tensor]]
    | DataLoader[list["BatchFeature"]],
) -> int:
    total_tokens = 0
    for data in dataloader:
        if isinstance(data, dict) or (is_transformers_available() and isinstance(data, BatchFeature)):
            if "input_ids" in data:
                if isinstance(data["input_ids"], torch.Tensor):
                    total_tokens += data["input_ids"].numel()
                else:
                    logger.warning(
                        "Counting calibration tokens, "
                        f"unsupported calibration data type {type(data['input_ids'])}, returning 0."
                    )
                    return 0
        elif isinstance(data, torch.Tensor):
            total_tokens += data.numel()
        else:
            logger.warning(f"Counting calibration tokens, unsupported calibration data type {type(data)}, returning 0.")
            return 0

    return total_tokens


def enable_native_inference(
    model: nn.Module,
    *,
    runtime_options: RuntimeOptions | None = None,
) -> int:
    """
    Enable native inference mode for quantized linear layers in the model.

    Replaces eligible ``QuantLinear`` / ``QParamsLinear`` modules with
    Aiter-backed native inference layers that use optimized AMD GEMM
    kernels for native inference. Export compatibility is preserved through
    shared state_dict serialization helpers.

    This can be called directly on a quantized model:
    ``QuantLinear -> NativeInferenceLinear`` conversion is handled internally.

    When preshuffle is enabled (via ``runtime_options.use_preshuffle``), the
    weight is shuffled in-place (~1× weight memory); ``state_dict()``
    temporarily un-shuffles for export.

    To revert back to the export-format module (``QParamsLinear`` with the
    ``scaled_mm`` / dequant fallback forward path), call
    :py:func:`disable_native_inference` directly — there is no boolean toggle.

    :param torch.nn.Module model: The quantized model containing
        ``QuantLinear`` / ``QParamsLinear`` layers.
    :param RuntimeOptions runtime_options: Runtime conversion options,
        including native linear selection and preshuffle behavior.
    :return: Number of layers converted.
    :rtype: int
    :raises ImportError: If Aiter is not installed.
    """
    from quark.torch.algorithm.svdquant.svdquant import ErrorCorrectedModule
    from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
    from quark.torch.kernel.aiter import is_aiter_available
    from quark.torch.quantization.nn.modules import (
        QuantLinear,
        aiter_fp4_inference_linear,  # noqa: F401
    )
    from quark.torch.quantization.nn.modules.aiter_fp8_inference_linear import (
        aiter_native_linear_from_module,
    )
    from quark.torch.quantization.nn.modules.aiter_svdquant_inference_linear import (
        svdquant_native_linear_from_error_corrected_module,
    )
    from quark.torch.quantization.nn.modules.native_inference_linear_common import (
        NativeInferenceLinear,
        NativeInferenceMode,
    )

    if not is_aiter_available():
        raise ImportError(
            "Native inference requires AMD Aiter. Please install Aiter from https://github.com/ROCm/aiter"
        )

    options = runtime_options or RuntimeOptions()
    mode_map = {
        "auto": None,
        "fp8_per_tensor": NativeInferenceMode.FP8_PER_TENSOR,
        "mxfp4": NativeInferenceMode.MXFP4,
    }
    if options.native_linear_mode not in mode_map:
        raise ValueError(
            f"Unsupported RuntimeOptions.native_linear_mode='{options.native_linear_mode}'. "
            f"Supported values: {list(mode_map.keys())}."
        )
    forced_mode = mode_map[options.native_linear_mode]

    # SVDQuant emits ErrorCorrectedModule wrappers (residual QParamsLinear +
    # low-rank correction). Convert the whole wrapper to a single fused native
    # layer rather than only its inner residual. ECMs are MXFP4, so they are
    # converted only when the mode allows MXFP4 (auto / mxfp4). Inner submodules
    # of every ECM are excluded from the plain QParamsLinear pass below so the
    # residual is not also converted standalone (whether or not the ECM itself
    # is converted here).
    ecm_names = [name for name, module in model.named_modules() if isinstance(module, ErrorCorrectedModule)]
    ecm_child_prefixes = tuple(name + "." for name in ecm_names)
    convert_ecm = forced_mode in (None, NativeInferenceMode.MXFP4)

    candidates: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, NativeInferenceLinear):
            continue
        if name.startswith(ecm_child_prefixes):
            continue
        is_convertible_source = isinstance(module, QParamsLinear | QuantLinear)
        if is_convertible_source and getattr(module, "weight_quantizer", None) is not None:
            candidates.append(name)

    converted = 0

    if convert_ecm:
        for name in ecm_names:
            ecm = model.get_submodule(name)
            try:
                native_layer = svdquant_native_linear_from_error_corrected_module(
                    ecm,
                    overlap_streams=options.svdquant_overlap_streams,
                    use_preshuffle=options.use_preshuffle,
                )
            except (ValueError, ImportError) as e:
                logger.warning(
                    f"Could not enable SVDQuant native inference for '{name}': {e}. "
                    f"Layer will use the eager ErrorCorrectedModule forward path."
                )
                continue
            _transfer_hf_hook(ecm, native_layer)
            setattr_recursive(model, name, native_layer)
            del ecm
            converted += 1

    for name in candidates:
        module = model.get_submodule(name)
        try:
            native_layer = aiter_native_linear_from_module(
                module,
                use_preshuffle=options.use_preshuffle,
                forced_mode=forced_mode,
            )
        except (ValueError, ImportError) as e:
            logger.warning(
                f"Could not enable native inference for '{name}': {e}. Layer will use fallback forward path."
            )
            continue
        _transfer_hf_hook(module, native_layer)
        setattr_recursive(model, name, native_layer)
        del module
        converted += 1

    logger.info(f"Native inference enabled for {converted} layers.")
    return converted


def disable_native_inference(model: nn.Module) -> int:
    """
    Convert native inference layers back to base ``QParamsLinear``.

    The base ``QParamsLinear.forward()`` uses ``scaled_mm`` (FP8 per-tensor)
    or ``dequant + F.linear`` as a fallback, so no Aiter dependency is needed
    after disabling.

    :param torch.nn.Module model: The model containing native inference layers.
    :return: Number of layers converted back to base ``QParamsLinear``.
    :rtype: int
    """
    from quark.torch.quantization.nn.modules.aiter_svdquant_inference_linear import (
        AiterSVDQuantMXFP4NativeInferenceLinear,
    )
    from quark.torch.quantization.nn.modules.native_inference_linear_common import (
        NativeInferenceLinear,
    )

    # SVDQuant composites must be reverted to an ErrorCorrectedModule (not a
    # single QParamsLinear), and their inner residual native linear must not be
    # reverted independently.
    svdquant_names = [
        name for name, module in model.named_modules() if isinstance(module, AiterSVDQuantMXFP4NativeInferenceLinear)
    ]
    svdquant_child_prefixes = tuple(name + "." for name in svdquant_names)

    replacements: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, AiterSVDQuantMXFP4NativeInferenceLinear):
            replacements.append((name, module.to_error_corrected_module()))
        elif isinstance(module, NativeInferenceLinear):
            if name.startswith(svdquant_child_prefixes):
                continue
            replacements.append((name, module.to_qparams_linear()))

    for name, restored in replacements:
        old_module = model.get_submodule(name)
        _transfer_hf_hook(old_module, restored)
        setattr_recursive(model, name, restored)

    logger.info(f"Native inference disabled for {len(replacements)} layers.")
    return len(replacements)


def _compute_observer_amax_from_quantizer(module: torch.nn.Module) -> float | None:
    """Read amax from a quantizer observer's recorded min/max range."""
    observer = getattr(module, "observer", None)
    if observer is None:
        return None

    observer_min = getattr(observer, "min_val", None)
    observer_max = getattr(observer, "max_val", None)
    if observer_min is None or observer_max is None:
        return None
    if observer_min.numel() != 1 or observer_max.numel() != 1:
        return None
    if observer_min.item() == float("inf") or observer_max.item() == float("-inf"):
        return None

    absolute_minimum = observer_min.to(torch.float32).abs()
    absolute_maximum = observer_max.to(torch.float32).abs()
    return torch.max(absolute_minimum, absolute_maximum).item()


def _is_asymmetric_quantizer(module: torch.nn.Module) -> bool:
    """Return whether a quantizer explicitly uses asymmetric qparams."""
    quantizer_symmetric = getattr(module, "symmetric", None)
    if quantizer_symmetric is not None:
        return quantizer_symmetric is False

    observer = getattr(module, "observer", None)
    if observer is None:
        return False

    return getattr(observer, "symmetric", None) is False


def sync_moe_expert_input_quantizer_qparams(model: torch.nn.Module) -> int:
    """Synchronize MoE expert input quantizer qparams after calibration.

    Experts from the same MoE layer and projection name share a single
    post-calibration amax. For ``SequentialQuantize`` input quantizers, each
    substage is synchronized independently. The quantizer scale/zero_point are
    then recomputed from the synced observer ranges so eager-mode inference,
    freeze, and export all see the same qparams. Asymmetric quantizers are
    skipped because this synchronization assumes symmetric min/max ranges.

    :param torch.nn.Module model: The quantized eager-mode model.
    :return: Number of synchronized quantizers.
    """
    shared_amax_by_projection: dict[tuple[str, str], float] = {}
    skipped_asymmetric_module_names: list[str] = []

    for module_name, module in model.named_modules():
        if not hasattr(module, "observer") or not hasattr(module, "update_buffer") or not hasattr(module, "scale"):
            continue
        if _is_asymmetric_quantizer(module):
            skipped_asymmetric_module_names.append(module_name)
            continue

        pattern_match = MOE_EXPERT_INPUT_QUANTIZER_PATTERN.match(module_name)
        if pattern_match is None:
            continue

        current_amax = _compute_observer_amax_from_quantizer(module)
        if current_amax is None:
            continue

        expert_prefix, projection_name, substage_index = pattern_match.groups()
        normalized_substage_index = "" if substage_index is None else substage_index
        projection_key = (expert_prefix, f"{projection_name}:{normalized_substage_index}")
        existing_amax = shared_amax_by_projection.get(projection_key)
        if existing_amax is None:
            shared_amax_by_projection[projection_key] = current_amax
        else:
            shared_amax_by_projection[projection_key] = max(existing_amax, current_amax)

    if skipped_asymmetric_module_names:
        logger.warning(
            "Skipped MoE expert input amax synchronization for %d asymmetric quantizer(s): %s.",
            len(skipped_asymmetric_module_names),
            ", ".join(skipped_asymmetric_module_names),
        )

    synchronized_module_count = 0
    for module_name, module in model.named_modules():
        if not hasattr(module, "observer") or not hasattr(module, "update_buffer") or not hasattr(module, "scale"):
            continue
        if _is_asymmetric_quantizer(module):
            continue

        pattern_match = MOE_EXPERT_INPUT_QUANTIZER_PATTERN.match(module_name)
        if pattern_match is None:
            continue

        expert_prefix, projection_name, substage_index = pattern_match.groups()
        normalized_substage_index = "" if substage_index is None else substage_index
        projection_key = (expert_prefix, f"{projection_name}:{normalized_substage_index}")
        shared_amax = shared_amax_by_projection.get(projection_key)
        if shared_amax is None:
            continue

        observer = getattr(module, "observer", None)
        if observer is None:
            continue

        observer_min = getattr(observer, "min_val", None)
        observer_max = getattr(observer, "max_val", None)
        if observer_min is None or observer_max is None:
            continue

        synced_amax_tensor = torch.tensor(
            shared_amax,
            dtype=observer_max.dtype,
            device=observer_max.device,
        )
        observer_max.copy_(synced_amax_tensor)
        observer_min.copy_(-synced_amax_tensor)

        qparams = observer._calculate_qparams()
        if qparams is None:
            continue

        scale_tensor, zero_point_tensor = qparams
        module.update_buffer("scale", scale_tensor, module.scale.device)
        module.update_buffer("zero_point", zero_point_tensor, module.scale.device)
        synchronized_module_count += 1

    return synchronized_module_count
