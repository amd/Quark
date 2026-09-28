#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""AutoRound's own weight-only/weight+activation quantize-model helper.

Forked from ``examples/torch/language_modeling/llm_qat/efficientqat/quantization_schemes.py``:
this driver follows ``quark/torch/quantization/config/template.py``'s scheme-naming convention,
while EfficientQAT keeps its own pre-existing names -- do not re-merge without re-checking both
callers' scheme-name expectations.
"""

import torch.nn as nn

from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import (
    Int2PerGroupSpec,
    OCP_MXFP4Spec,
    QConfig,
    QLayerConfig,
    QTensorConfig,
)
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver

# Naming follows template.py's `_wo`/`_wa` (weight-only/weight+activation) convention for the INT
# schemes, e.g. `int4_wo_64` -- group_size is a separate, runtime `--group_size` CLI flag here
# rather than baked into the scheme name. INT2 has no template.py equivalent and is named in the
# same style.
INT4_QUANT_SCHEMES = [
    "uint4_wo_asym",
    "int4_wo_sym",
    "int4_wo_asym",
]
INT2_QUANT_SCHEMES = [
    "int2_wo_asym",
]
# MXFP4 weight-only. Uses Quark's default "even" scale_calculation_mode (matches Quark's HIP
# production kernel).
MX_QUANT_SCHEMES = [
    "mxfp4_weight_only",
]
# Weight+activation MXFP4 (matches template.py's bare "mxfp4" name): weight is
# static/calibrated/tunable like MX_QUANT_SCHEMES; activation is dynamic (recomputed every
# forward, no calibration or learnable params -- AutoRound never tunes activations).
MX_WA_QUANT_SCHEMES = [
    "mxfp4",
]
# OCP MX group size is fixed by the format spec, not user-selectable via --group_size.
MX_GROUP_SIZE = 32

SUPPORTED_QUANT_SCHEMES = [
    *INT4_QUANT_SCHEMES,
    *INT2_QUANT_SCHEMES,
    *MX_QUANT_SCHEMES,
    *MX_WA_QUANT_SCHEMES,
]


def _build_int2_weight_spec(group_size: int) -> QTensorConfig:
    if group_size <= 0:
        raise ValueError(f"group_size must be a positive integer for INT2 quantization, got {group_size}")
    return Int2PerGroupSpec(
        ch_axis=-1,
        group_size=group_size,
        symmetric=False,
        scale_type="float",
        round_method="half_even",
        is_dynamic=False,
    ).to_quantization_spec()


def _build_int4_or_uint4_weight_spec(quant_scheme: str, group_size: int) -> QTensorConfig:
    if group_size <= 0:
        raise ValueError(f"group_size must be a positive integer for PTQ, got {group_size}")
    if quant_scheme == "uint4_wo_asym":
        dtype = Dtype.uint4
        symmetric = False
    elif quant_scheme == "int4_wo_sym":
        dtype = Dtype.int4
        symmetric = True
    elif quant_scheme == "int4_wo_asym":
        dtype = Dtype.int4
        symmetric = False
    else:
        raise ValueError(f"Unsupported quant_scheme for PTQ: {quant_scheme}")
    return QTensorConfig(
        dtype=dtype,
        observer_cls=PerGroupMinMaxObserver,
        symmetric=symmetric,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        qscheme=QSchemeType.per_group,
        ch_axis=1,
        is_dynamic=False,
        group_size=group_size,
    )


def _build_mxfp4_weight_spec() -> QTensorConfig:
    """OCP MXFP4: group size is fixed at 32 by the format spec (not user-selectable) --
    see ``main.py``'s ``--group_size`` guard for the caller-facing warning when a non-32 value
    was requested. Uses Quark's default ``"even"`` scale_calculation_mode, which matches Quark's
    HIP production kernel.
    """
    return OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False, scale_calculation_mode="even").to_quantization_spec()


def _build_mxfp4_activation_spec() -> QTensorConfig:
    """OCP MXFP4 activation spec: ``is_dynamic=True`` -- no calibration, scale recomputed fresh
    from each forward's actual activation values (no learnable parameters; AutoRound never tunes
    activations). Uses the same ``"even"`` scale_calculation_mode as the weight spec."""
    return OCP_MXFP4Spec(ch_axis=-1, is_dynamic=True, scale_calculation_mode="even").to_quantization_spec()


def quantize_model(model: nn.Module, quant_scheme: str, group_size: int) -> nn.Module:
    """Apply weight-only (or, for ``MX_WA_QUANT_SCHEMES``, weight+activation) quantization to a
    PyTorch model. Named generically (not ``weight_only_quantize``) because it also handles the
    weight+activation ``MX_WA_QUANT_SCHEMES`` schemes.

    Args:
        model: Model to quantize.
        quant_scheme: Quantization scheme name — see ``SUPPORTED_QUANT_SCHEMES`` (INT2/INT4/UINT4
            weight-only, or MXFP4 weight-only/weight+activation — the MX schemes use Quark's
            "even" scale_calculation_mode).
        group_size: Per-group quantization size. Must be a positive integer. Ignored for MX
            schemes (fixed at ``MX_GROUP_SIZE``) — see ``main.py``'s ``--group_size`` guard for
            the caller-facing warning when a non-32 value was requested.

    Returns:
        Quantized model. ``lm_head`` is excluded to reduce eval/PPL drift.
    """
    input_spec = None
    if quant_scheme in INT2_QUANT_SCHEMES:
        weight_spec = _build_int2_weight_spec(group_size)
    elif quant_scheme in MX_WA_QUANT_SCHEMES:
        weight_spec = _build_mxfp4_weight_spec()
        input_spec = _build_mxfp4_activation_spec()
    elif quant_scheme in MX_QUANT_SCHEMES:
        weight_spec = _build_mxfp4_weight_spec()
    else:
        weight_spec = _build_int4_or_uint4_weight_spec(quant_scheme, group_size)

    quant_config = QConfig(
        global_quant_config=QLayerConfig(weight=weight_spec, input_tensors=input_spec),
        # Keep lm_head in full precision to avoid eval/PPL drift.
        exclude=["lm_head", "*lm_head"],
    )
    quantizer = ModelQuantizer(quant_config)
    model = quantizer.quantize_model(model, None)
    return model
