#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import fnmatch
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import nn

from ._serialization import StrictSchema, sha256_json
from .errors import SchemaValidationError

_LAYER_INDEX = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)")
_SUPPORTED_DTYPES = {torch.float16: "float16", torch.bfloat16: "bfloat16"}


class ModuleStatus(StrEnum):
    SEARCHABLE = "searchable"
    FORCED_NATIVE_USER = "forced_native_user"
    FORCED_NATIVE_ALIAS = "forced_native_alias"
    FORCED_NATIVE_DEPLOYMENT = "forced_native_deployment"
    TEMPLATE_EXCLUDED = "template_excluded"


@dataclass(frozen=True, slots=True)
class QuantizableModule(StrictSchema):
    name: str
    weight_shape: tuple[int, ...]
    source_dtype: str
    num_params: int
    layer_index: int | None
    storage_id: str
    storage_offset: int
    weight_stride: tuple[int, ...]
    storage_aliases: tuple[str, ...]
    status: ModuleStatus

    def __post_init__(self) -> None:
        if not self.name or len(self.weight_shape) != 2 or any(dimension <= 0 for dimension in self.weight_shape):
            raise SchemaValidationError("Quantizable modules require a named rank-2 weight with positive dimensions.")
        if self.source_dtype not in {"float16", "bfloat16"}:
            raise SchemaValidationError("Quantizable module source dtype must be float16 or bfloat16.")
        if self.num_params != math.prod(self.weight_shape):
            raise SchemaValidationError("Quantizable module num_params does not match its weight shape.")
        if not self.storage_id.startswith("sha256:") or self.storage_offset < 0:
            raise SchemaValidationError("Quantizable module storage metadata is invalid.")
        if len(self.weight_stride) != len(self.weight_shape) or any(stride <= 0 for stride in self.weight_stride):
            raise SchemaValidationError("Quantizable module weight stride is invalid.")
        if self.storage_aliases != tuple(sorted(set(self.storage_aliases))):
            raise SchemaValidationError("Quantizable module storage aliases must be sorted and unique.")
        parameter_name = f"{self.name}.weight"
        if parameter_name not in self.storage_aliases:
            raise SchemaValidationError("Quantizable module storage aliases must include its weight parameter.")


@dataclass(frozen=True, slots=True)
class Inventory:
    modules: tuple[QuantizableModule, ...]
    matched_user_patterns: dict[str, tuple[str, ...]]


def _matches(name: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(name, pattern) or name == pattern or name.endswith(f".{pattern}")


def _storage_key(tensor: torch.Tensor) -> tuple[str, int, int]:
    if tensor.device.type == "meta":
        raise SchemaValidationError("MVP inventory requires materialized weights, not meta tensors.")
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


def _parameter_storage_views(
    model: nn.Module,
) -> dict[tuple[str, int, int], dict[str, tuple[int, tuple[int, ...], tuple[int, ...]]]]:
    views: dict[tuple[str, int, int], dict[str, tuple[int, tuple[int, ...], tuple[int, ...]]]] = defaultdict(dict)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        views[_storage_key(parameter)][name] = (
            parameter.storage_offset(),
            tuple(parameter.shape),
            tuple(parameter.stride()),
        )
    return dict(views)


def _layer_index(name: str) -> int | None:
    match = _LAYER_INDEX.search(name)
    return int(match.group(1)) if match else None


def enumerate_linear_modules(
    model: nn.Module,
    *,
    template_excludes: tuple[str, ...],
    user_excludes: tuple[str, ...],
) -> Inventory:
    """Enumerate post-preprocess Linear modules and assign an explicit status."""
    storage_views = _parameter_storage_views(model)
    linear_items = sorted(
        (
            (name, module)
            for name, module in model.named_modules(remove_duplicate=False)
            if isinstance(module, nn.Linear)
        ),
        key=lambda item: item[0],
    )
    if not linear_items:
        raise SchemaValidationError("The model contains no nn.Linear modules.")

    matched_user: dict[str, list[str]] = {pattern: [] for pattern in user_excludes}
    modules: list[QuantizableModule] = []
    for name, module in linear_items:
        weight = module.weight
        if weight.ndim != 2:
            raise SchemaValidationError(f"Linear weight {name!r} must be rank 2, got shape {tuple(weight.shape)}.")
        if weight.dtype not in _SUPPORTED_DTYPES:
            raise SchemaValidationError(
                f"Linear weight {name!r} must use BF16 or FP16, got {str(weight.dtype).removeprefix('torch.')}."
            )

        key = _storage_key(weight)
        views = storage_views[key]
        aliases = tuple(sorted(views))
        current_view = (weight.storage_offset(), tuple(weight.shape), tuple(weight.stride()))
        expected_nbytes = weight.numel() * weight.element_size()
        if weight.untyped_storage().nbytes() != expected_nbytes or any(view != current_view for view in views.values()):
            raise SchemaValidationError(
                f"Linear weight {name!r} is a partial or non-identical storage view; MVP only supports full aliases."
            )
        template_excluded = any(_matches(name, pattern) for pattern in template_excludes)
        matched_patterns = [pattern for pattern in user_excludes if _matches(name, pattern)]
        for pattern in matched_patterns:
            matched_user[pattern].append(name)

        shares_storage = len(aliases) > 1
        if template_excluded:
            status = ModuleStatus.TEMPLATE_EXCLUDED
        elif matched_patterns:
            status = ModuleStatus.FORCED_NATIVE_USER
        elif shares_storage:
            status = ModuleStatus.FORCED_NATIVE_ALIAS
        else:
            status = ModuleStatus.SEARCHABLE

        modules.append(
            QuantizableModule(
                name=name,
                weight_shape=tuple(weight.shape),
                source_dtype=_SUPPORTED_DTYPES[weight.dtype],
                num_params=weight.numel(),
                layer_index=_layer_index(name),
                storage_id=sha256_json({"parameter_names": aliases}),
                storage_offset=weight.storage_offset(),
                weight_stride=tuple(weight.stride()),
                storage_aliases=aliases,
                status=status,
            )
        )

    unmatched = sorted(pattern for pattern, names in matched_user.items() if not names)
    if unmatched:
        raise SchemaValidationError(f"exclude_patterns did not match any Linear module: {unmatched}.")

    return Inventory(
        modules=tuple(modules),
        matched_user_patterns={pattern: tuple(names) for pattern, names in matched_user.items()},
    )


def model_structure_fingerprint(model_type: str, modules: tuple[QuantizableModule, ...]) -> str:
    return sha256_json(
        {
            "model_type": model_type,
            "modules": [
                {
                    "name": module.name,
                    "weight_shape": module.weight_shape,
                    "source_dtype": module.source_dtype,
                    "storage_id": module.storage_id,
                    "storage_offset": module.storage_offset,
                    "weight_stride": module.weight_stride,
                    "storage_aliases": module.storage_aliases,
                }
                for module in modules
            ],
        }
    )


__all__ = [
    "Inventory",
    "ModuleStatus",
    "QuantizableModule",
    "enumerate_linear_modules",
    "model_structure_fingerprint",
]
