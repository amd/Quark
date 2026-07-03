#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Portions of this file are derived from HuggingFace transformers
# (https://github.com/huggingface/transformers), licensed under the Apache License 2.0.
# Original source: transformers.core_model_loading
#

"""
Standalone weight conversion utilities extracted from transformers.core_model_loading.

This module is self-contained and depends only on torch and the Python standard library.
It provides WeightConverter / WeightRenaming along with all ConversionOps needed to
perform checkpoint weight transformations without importing transformers.
"""

from __future__ import annotations

import re
import traceback
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Generator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

# ---------------------------------------------------------------------------
# LoadStateDictInfo  (standalone replica)
# ---------------------------------------------------------------------------


@dataclass
class LoadStateDictInfo:
    """Mutable container for state-dict loading results and diagnostics."""

    missing_keys: set[str]
    unexpected_keys: set[str]
    mismatched_keys: set[tuple[str, tuple[int], tuple[int]]]
    error_msgs: list[str]
    conversion_errors: dict[str, str]

    def missing_and_mismatched(self) -> set[str]:
        mismatched_names: set[str] = set()
        for mismatched_entry in self.mismatched_keys:
            mismatched_names.add(mismatched_entry[0])
        return self.missing_keys | mismatched_names


# ---------------------------------------------------------------------------
# ConversionOps  (base + all built-in subclasses)
# ---------------------------------------------------------------------------


class ConversionOps:
    """Base class for weight conversion operations."""

    def __repr__(self) -> str:
        if hasattr(self, "dim"):
            return f"{self.__class__.__name__}(dim={self.dim})"
        return f"{self.__class__.__name__}"

    @abstractmethod
    def convert(
        self, input_dict: dict[str, Any], source_patterns: list[str], target_patterns: list[str], **kwargs: Any
    ) -> dict[str, list[torch.Tensor]]:
        raise NotImplementedError

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError


class Chunk(ConversionOps):
    """Split a tensor along ``dim`` into equally sized chunks."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        targets = self.get_target_patterns(input_dict, target_patterns)
        sizes = len(targets)
        chunks = torch.chunk(tensor, sizes, dim=self.dim)
        return dict(zip(targets, chunks, strict=False))

    def get_target_patterns(self, input_dict: dict[str, Any], target_patterns: list[str]) -> list[str]:
        if len(input_dict) > 1 or len(target_patterns) == 1:
            raise ValueError("Undefined Operation encountered!")
        return target_patterns

    @property
    def reverse_op(self) -> ConversionOps:
        return Concatenate(self.dim)


class SplitFusedExperts(ConversionOps):
    """Unfuse a stacked MoE expert tensor of shape ``[num_experts, ...]`` into per-expert tensors.

    For two target patterns (e.g. gate/up) the per-expert tensor is additionally split along
    ``split_axis`` into equal halves. For one target pattern (e.g. down) only the per-expert
    unfusion is applied. Output keys are prefixed with ``{expert_index}.`` so downstream
    expansion produces ``...experts.{i}.gate_proj.weight`` etc.

    :param int split_axis: Axis on the per-expert (post-unfuse) tensor along which to split
        gate/up halves. Defaults to ``0``, matching the qwen3_5_moe layout where each
        per-expert ``gate_up`` tensor has shape ``(2*intermediate, hidden)``.
    """

    def __init__(self, split_axis: int = 0):
        self.split_axis = split_axis

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        tensor_values = next(iter(input_dict.values()))
        fused_expert_tensor = tensor_values[0] if isinstance(tensor_values, list) else tensor_values

        if fused_expert_tensor.ndim != 3:
            raise ValueError(f"Expected a 3D fused expert tensor, but got shape {tuple(fused_expert_tensor.shape)}.")

        converted_tensors: dict[str, torch.Tensor] = {}

        if len(target_patterns) == 1:
            target_pattern = target_patterns[0]
            for expert_index in range(fused_expert_tensor.shape[0]):
                converted_tensors[f"{expert_index}.{target_pattern}"] = fused_expert_tensor[expert_index]
            return converted_tensors

        if len(target_patterns) != 2:
            raise ValueError(f"Expected 1 or 2 target patterns, but got {len(target_patterns)}.")

        # Per-expert view is one dim smaller than the fused tensor. Map split_axis onto it
        # and validate the split dim is even.
        per_expert_split_dim = self.split_axis
        per_expert_ndim = fused_expert_tensor.ndim - 1
        if per_expert_split_dim < 0:
            per_expert_split_dim += per_expert_ndim
        if not 0 <= per_expert_split_dim < per_expert_ndim:
            raise ValueError(
                f"split_axis={self.split_axis} is out of range for per-expert tensor of ndim {per_expert_ndim}."
            )
        fused_split_dim = per_expert_split_dim + 1  # account for the leading num_experts dim
        if fused_expert_tensor.shape[fused_split_dim] % 2 != 0:
            raise ValueError(
                f"Expected the fused gate_up tensor to have an even size along axis {fused_split_dim}, "
                f"but got shape {tuple(fused_expert_tensor.shape)}."
            )

        gate_target_pattern, up_target_pattern = target_patterns
        for expert_index in range(fused_expert_tensor.shape[0]):
            expert_gate_up_tensor = fused_expert_tensor[expert_index]
            gate_half, up_half = torch.chunk(expert_gate_up_tensor, 2, dim=per_expert_split_dim)
            converted_tensors[f"{expert_index}.{gate_target_pattern}"] = gate_half
            converted_tensors[f"{expert_index}.{up_target_pattern}"] = up_half
        return converted_tensors


class Concatenate(ConversionOps):
    """Concatenate tensors along ``dim``."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        target_pattern = self.get_target_pattern(target_patterns)
        all_tensors = []
        for source_pattern in source_patterns:
            tensors = input_dict[source_pattern]
            if isinstance(tensors, list):
                all_tensors.extend(tensors)
            else:
                all_tensors.append(tensors)
        return {target_pattern: torch.cat(all_tensors, dim=self.dim)}

    def get_target_pattern(self, target_patterns: list[str]) -> str:
        if len(target_patterns) > 1:
            raise ValueError("Undefined Operation encountered!")
        return target_patterns[0]

    @property
    def reverse_op(self) -> ConversionOps:
        return Chunk(self.dim)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INTERNAL_MANY_TO_MANY_CONVERSIONS: tuple[type[ConversionOps], ...] = ()


class SkipParameters(Exception):
    """Control-flow sentinel: abort processing of the current parameters only."""

    pass


@contextmanager
def log_conversion_errors(
    first_target_key: str,
    loading_info: LoadStateDictInfo | None,
    extras: Any = None,
    operation: list[ConversionOps] | ConversionOps | None = None,
) -> Generator[None, None, None]:
    """Catch exceptions during ``convert`` calls, log them, and raise :class:`SkipParameters`."""
    try:
        yield
    except Exception as e:
        if loading_info is None:
            raise

        def _format_operation_name(current_operation: list[ConversionOps] | ConversionOps | None) -> str | None:
            if current_operation is None:
                return None
            if isinstance(current_operation, list | tuple | set):
                names = []
                for single_operation in current_operation:
                    if single_operation is not None:
                        names.append(single_operation.__class__.__name__)
                return ", ".join(names) if names else None
            return current_operation.__class__.__name__

        operation_name = _format_operation_name(operation)

        traceback_string = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        if isinstance(extras, tuple) and len(extras) == 2:
            length, target_keys = extras
            descriptor = f"{operation_name} " if operation_name else ""
            loading_info.conversion_errors[first_target_key] = (
                f"{traceback_string}{e}\nError: {descriptor}on tensors destined for {target_keys}. Ckpt contains: {length}"
            )
        elif isinstance(extras, str):
            suffix = f" via {operation_name}" if operation_name else ""
            loading_info.conversion_errors[first_target_key] = (
                f"{traceback_string}{e}\nError{suffix} when processing parameter {extras}"
            )
        elif extras is None and operation_name:
            loading_info.conversion_errors[first_target_key] = f"{operation_name}: {e}"
        else:
            loading_info.conversion_errors[first_target_key] = f"{extras} |Error: {e}"

        raise SkipParameters() from e


def process_target_pattern(pattern: str) -> tuple[str, str | None]:
    """
    Process a target pattern for reverse mapping (when targets become sources).

    Handles ``^`` / ``$`` anchors, negative lookahead/lookbehind, and capturing groups.
    """
    pattern = pattern.removeprefix("^")
    pattern = pattern.removesuffix("$")
    pattern = re.sub(r"\(\?.+\)", "", pattern)
    capturing_group_match = re.search(r"\(.+?\)", pattern)
    captured_group = None
    if capturing_group_match:
        captured_group = capturing_group_match.group(0)
        pattern = pattern.replace(captured_group, r"\1", 1)
    return pattern, captured_group


# ---------------------------------------------------------------------------
# WeightTransform  (base dataclass)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WeightTransform:
    source_patterns: str | list[str] = field(init=True)
    target_patterns: str | list[str] = field(init=True)
    compiled_sources: re.Pattern[str] = field(init=False)

    distributed_operation: Any | None = None
    quantization_operation: ConversionOps | None = None

    collected_tensors: dict[str, list[Future[torch.Tensor | None]]] = field(
        default_factory=lambda: defaultdict(list), init=False
    )
    layer_targets: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set), init=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("source_patterns", "target_patterns"):
            if hasattr(self, name):
                raise ValueError(f"Cannot assign to field {name}, you should create a new instance")
            elif isinstance(value, str):
                value = [value]
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        target_capturing_groups: list[str] = []
        for pattern_index, pattern in enumerate(self.target_patterns):
            self.target_patterns[pattern_index], captured_group = process_target_pattern(pattern)
            if captured_group is not None:
                target_capturing_groups.append(captured_group)

        unique_capturing_groups = set(target_capturing_groups)
        if len(unique_capturing_groups) > 1:
            raise ValueError(
                f"Multiple different capturing groups found in target_patterns: {unique_capturing_groups}. "
                f"All target patterns must use the same capturing group pattern."
            )
        unique_capturing_group = unique_capturing_groups.pop() if unique_capturing_groups else None

        for pattern_index, pattern in enumerate(self.source_patterns):
            if r"\1" in pattern:
                if unique_capturing_group is None:
                    raise ValueError(
                        f"Source pattern '{pattern}' contains \\1 backreference, but no capturing groups "
                        f"found in target_patterns."
                    )
                pattern = pattern.replace(r"\1", unique_capturing_group, 1)
            self.source_patterns[pattern_index] = pattern

        branches = []
        for pattern_index, source_pattern in enumerate(self.source_patterns):
            group_name = f"g{pattern_index}"
            pattern_body = source_pattern.replace(".*.", r"\..*\.")
            branches.append(f"(?P<{group_name}>{pattern_body})")
        self.compiled_sources = re.compile("|".join(branches))

    def add_tensor(
        self, target_key: str, source_key: str, source_pattern: str, future: Future[torch.Tensor | None]
    ) -> None:
        self.collected_tensors[source_pattern].append(future)
        self.layer_targets[target_key].add(source_key)

    def rename_source_key(self, source_key: str) -> tuple[str, str | None]:
        """Return ``(renamed_key, source_pattern_producing_the_match)``."""
        match_object = self.compiled_sources.search(source_key)
        if match_object is None:
            return source_key, None

        matching_group_name = next(name for name, val in match_object.groupdict().items() if val is not None)
        source_pattern_that_matched = self.source_patterns[int(matching_group_name[1:])]
        replacement = self.target_patterns[0]
        if r"\1" in replacement:
            replaced_group_idx = self.compiled_sources.groupindex[matching_group_name] + 1
            replacement = replacement.replace(r"\1", match_object.group(replaced_group_idx))
        renamed_key = source_key.replace(match_object.group(0), replacement)
        return renamed_key, source_pattern_that_matched

    def reverse_transform(self) -> WeightTransform:
        """Reverse the current transform for saving with opposite weight transformations."""
        if self.quantization_operation is not None:
            raise ValueError("Cannot reverse the transform with TP or quantization")

        kwargs = {}
        if hasattr(self, "operations"):
            kwargs["operations"] = [operation.reverse_op for operation in self.operations[::-1]]

        return self.__class__(source_patterns=self.target_patterns, target_patterns=self.source_patterns, **kwargs)

    def materialize_tensors(self) -> dict[str, list[torch.Tensor]]:
        """
        Materialize all collected tensors.

        Handles three cases:
        - async loading: tensors are ``Future`` instances
        - sync loading: tensors are ``Callable``
        - saving: tensors are already ``torch.Tensor``
        """
        collected_tensors = {}
        for tensor_key in set(self.collected_tensors.keys()):
            pending_tensors = self.collected_tensors.pop(tensor_key)
            if isinstance(pending_tensors[0], Future):
                materialized = []
                for future in pending_tensors:
                    resolved = future.result()
                    if resolved is not None:
                        materialized.append(resolved)
                pending_tensors = materialized
            elif callable(pending_tensors[0]):
                materialized = []
                for loader_function in pending_tensors:
                    materialized.append(loader_function())
                pending_tensors = materialized
            collected_tensors[tensor_key] = pending_tensors
        return collected_tensors


# ---------------------------------------------------------------------------
# WeightConverter
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WeightConverter(WeightTransform):
    operations: list[ConversionOps] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        WeightTransform.__post_init__(self)
        if bool(len(self.source_patterns) - 1) + bool(len(self.target_patterns) - 1) >= 2:
            if not any(isinstance(op, _INTERNAL_MANY_TO_MANY_CONVERSIONS) for op in self.operations):
                raise ValueError(
                    f"source keys={self.source_patterns}, target_patterns={self.target_patterns} "
                    f"but you can only have one to many, one to one or many to one."
                )
        if not self.operations:
            raise ValueError("WeightConverter requires at least one operation.")

    def convert(
        self,
        layer_name: str,
        model: Any = None,
        config: Any = None,
        hf_quantizer: Any = None,
        loading_info: LoadStateDictInfo | None = None,
    ) -> dict[str, Any]:
        collected_tensors = self.materialize_tensors()

        for operation in self.operations:
            with log_conversion_errors(layer_name, loading_info, (len(collected_tensors), layer_name), operation):
                collected_tensors = operation.convert(
                    collected_tensors,
                    source_patterns=self.source_patterns,
                    target_patterns=self.target_patterns,
                    full_layer_name=layer_name,
                    model=model,
                    config=config,
                    missing_keys=loading_info.missing_keys if loading_info else None,
                )

        full_name = layer_name
        if ".*." in layer_name:
            full_name = layer_name.replace(".*.", ".0.")

        try:
            matching_key = next(tensor_key for tensor_key in collected_tensors if tensor_key in full_name)
            prefix, _, suffix = full_name.partition(matching_key)
            expanded_tensors = {}
            for tensor_key, tensor_value in collected_tensors.items():
                expanded_tensors[prefix + tensor_key + suffix] = tensor_value
            collected_tensors = expanded_tensors
        except StopIteration:
            pass

        if hf_quantizer is not None and self.quantization_operation is not None:
            with log_conversion_errors(
                layer_name, loading_info, (len(collected_tensors), layer_name), self.quantization_operation
            ):
                collected_tensors = self.quantization_operation.convert(
                    collected_tensors,
                    source_patterns=self.source_patterns,
                    target_patterns=self.target_patterns,
                    full_layer_name=layer_name,
                    config=config,
                    model=model,
                    missing_keys=loading_info.missing_keys if loading_info else None,
                )
        return collected_tensors
