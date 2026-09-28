#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from quark.torch.quantization.config.config import QTensorConfig
from quark.torch.utils.pack import create_pack_method

from ._serialization import StrictSchema
from .errors import SchemaValidationError
from .hardware_capability import get_scheme_config

_SOURCE_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass(frozen=True, slots=True)
class WeightStorageCost(StrictSchema):
    weight_bits: int
    metadata_bits: int
    total_bits: int
    num_params: int
    effective_bits: float

    def __post_init__(self) -> None:
        if self.weight_bits < 0 or self.metadata_bits < 0 or self.num_params <= 0:
            raise SchemaValidationError("Weight storage values must be non-negative with positive num_params.")
        if self.total_bits != self.weight_bits + self.metadata_bits:
            raise SchemaValidationError("total_bits must equal weight_bits plus metadata_bits.")
        if not math.isclose(self.effective_bits, self.total_bits / self.num_params, rel_tol=0.0, abs_tol=1e-12):
            raise SchemaValidationError("effective_bits does not match total_bits / num_params.")


def _numel(shape: tuple[int, ...]) -> int:
    if not shape or any(dimension <= 0 for dimension in shape):
        raise SchemaValidationError(f"Invalid tensor shape: {shape}.")
    return math.prod(shape)


def calculate_weight_storage(
    weight_shape: tuple[int, ...],
    source_dtype: str,
    scheme: str,
    *,
    model_type: str,
) -> WeightStorageCost:
    """Calculate serialized weight and scale storage using Quark's pack-shape rules."""
    num_params = _numel(weight_shape)
    if source_dtype not in _SOURCE_DTYPES:
        raise SchemaValidationError(f"Unsupported source dtype: {source_dtype!r}.")
    source_torch_dtype = _SOURCE_DTYPES[source_dtype]

    if scheme == "native":
        weight_bits = num_params * torch.empty((), dtype=source_torch_dtype).element_size() * 8
        return WeightStorageCost(
            weight_bits=weight_bits,
            metadata_bits=0,
            total_bits=weight_bits,
            num_params=num_params,
            effective_bits=weight_bits / num_params,
        )

    layer_config = get_scheme_config(model_type, scheme)
    weight_spec = layer_config.weight
    if not isinstance(weight_spec, QTensorConfig):
        raise SchemaValidationError(f"MVP scheme {scheme!r} must have one weight QTensorConfig.")
    if weight_spec.qscheme is None:
        raise SchemaValidationError(f"MVP scheme {scheme!r} has no weight qscheme.")

    pack_method = create_pack_method(
        qscheme=weight_spec.qscheme.value,
        dtype=weight_spec.dtype.value,
        mx_element_dtype=weight_spec.mx_element_dtype.value if weight_spec.mx_element_dtype is not None else None,
    )
    packed_shape, scale_shape, _ = pack_method.infer_packed_shape(
        unpacked_shape=weight_shape,
        quantization_spec=weight_spec,
        legacy=False,
        custom_mode="quark",
    )
    packed_dtype = weight_spec.dtype.to_torch_packed_dtype()
    weight_bits = math.prod(packed_shape) * torch.empty((), dtype=packed_dtype).element_size() * 8
    scale_bits = math.prod(scale_shape) * torch.empty((), dtype=source_torch_dtype).element_size() * 8
    total_bits = weight_bits + scale_bits
    return WeightStorageCost(
        weight_bits=weight_bits,
        metadata_bits=scale_bits,
        total_bits=total_bits,
        num_params=num_params,
        effective_bits=total_bits / num_params,
    )


__all__ = ["WeightStorageCost", "calculate_weight_storage"]
