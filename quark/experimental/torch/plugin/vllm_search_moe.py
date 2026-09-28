#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Search-only negotiation between Quark QDQ adapters and vLLM MoE selectors.

The registry describes concrete execution paths, not model names or vLLM's
backend families. Add an adapter only alongside an a2 hook and execution tests.
vLLM remains responsible for GPU, activation, shape and parallelism support.
"""

from __future__ import annotations

import copy
import functools
import importlib
import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from quark.experimental.torch.plugin.vllm_inverse_quantizer import weight_quantization_formats_match

_EXPERTS = "vllm.model_executor.layers.fused_moe.experts."
_ORACLE = "vllm.model_executor.layers.fused_moe.oracle."


@dataclass(frozen=True)
class SearchMoeAdapter:
    """A concrete QDQ implementation and the vLLM family used to request it."""

    backend: str
    expert_class: str
    source_kinds: tuple[str, ...]
    backend_names: tuple[str, ...]
    a2_hook: str
    vllm_backend: str
    weight_conversion: bool = False
    automatic: bool = True


SEARCH_MOE_ADAPTERS = (
    SearchMoeAdapter(
        "triton",
        _EXPERTS + "triton_moe.TritonExperts",
        ("unquantized", "fp8"),
        ("TRITON",),
        "moe_kernel_quantize_input / invoke_fused_moe_triton_kernel",
        vllm_backend="triton",
        weight_conversion=True,
    ),
    SearchMoeAdapter(
        "triton_unfused",
        _EXPERTS + "gpt_oss_triton_kernels_moe.UnfusedOAITritonExperts",
        ("mxfp4",),
        ("TRITON_UNFUSED",),
        "UnfusedOAITritonExperts.activation",
        vllm_backend="triton_unfused",
    ),
    SearchMoeAdapter(
        "aiter_mxfp4_bf16",
        _EXPERTS + "rocm_aiter_moe.AiterExperts",
        ("mxfp4",),
        ("AITER", "AITER_MXFP4_BF16"),
        "get_2stage_cfgs.stage2",
        vllm_backend="aiter",
    ),
    # Explicit compatibility option only: source activation semantics can differ
    # from W4A16, so auto must not silently fall back to emulation.
    SearchMoeAdapter(
        "emulation",
        _EXPERTS + "ocp_mx_emulation_moe.OCP_MXQuantizationEmulationTritonExperts",
        ("mxfp4",),
        ("EMULATION",),
        "Triton activation helpers",
        vllm_backend="emulation",
        automatic=False,
    ),
)

_SELECTORS = (
    ("unquantized", "select_unquantized_moe_backend", "unquantized"),
    ("fp8", "select_fp8_moe_backend", "fp8"),
    ("mxfp4", "select_mxfp4_moe_backend", "mxfp4"),
    ("mxfp4", "select_deepseek_v4_mxfp4_moe_backend", "mxfp4"),
)


def to_vllm_moe_backend(backend: str) -> str:
    """Translate a Quark adapter ID, preserving vLLM family names and auto."""
    for adapter in SEARCH_MOE_ADAPTERS:
        if adapter.backend == backend:
            return adapter.vllm_backend
    return backend


def _class_name(cls: Any) -> str:
    return f"{cls.__module__}.{cls.__qualname__}" if cls is not None else "None"


def _backend_name(backend: Any) -> str:
    return str(getattr(backend, "name", backend)).upper()


def find_search_moe_adapter(backend: Any, experts_cls: Any, source_kind: str) -> SearchMoeAdapter:
    """Do not infer support for subclasses: they can override the hooked path."""
    for adapter in SEARCH_MOE_ADAPTERS:
        if (
            _class_name(experts_cls) == adapter.expert_class
            and _backend_name(backend) in adapter.backend_names
            and source_kind in adapter.source_kinds
        ):
            return adapter
    raise ValueError(
        f"Quark plugin has no search a2-QDQ adapter for backend={_backend_name(backend)}, "
        f"experts={_class_name(experts_cls)}, source={source_kind}"
    )


class SearchMoeSelector:
    """Installed in a search worker before model construction and weight loading."""

    def __init__(self, policy: dict[str, Any]) -> None:
        from quark.experimental.torch.mix_precision.config import get_layer_config, is_native_mode

        self.requested = policy["requested"]
        self.targets = [
            config
            for mode in policy["target_modes"]
            if not is_native_mode(mode) and (config := get_layer_config(mode)) is not None
        ]
        self.records: list[dict[str, Any]] = []
        self.inventory: list[dict[str, Any]] = []
        self.probes: list[dict[str, Any]] = []
        self._bindings: list[tuple[Any, str, Any, Any]] = []
        self._selected: dict[int, tuple[str, str, str]] = {}

    def _targets_for_source(self, source_kind: str) -> list[Any]:
        # Match create_qconfig_from_quant_config's per-layer precision floor.
        # In a heterogeneous checkpoint a W4 layer may see an FP8 candidate,
        # which keeps its W4 weights and changes only activation precision.
        source_bits = {"mxfp4": 4, "fp8": 8}.get(source_kind)
        targets = []
        for target in self.targets:
            weight = target.weight
            if isinstance(weight, list):
                weight = weight[0] if weight else None
            bitwidth = getattr(getattr(weight, "dtype", None), "to_bitwidth", None)
            if source_bits is not None and callable(bitwidth) and bitwidth() > source_bits:
                target = replace(target, weight=None)
            targets.append(target)
        return targets

    def _requires_weight_conversion(self, source_kind: str) -> bool:
        if source_kind == "unquantized":
            return False
        if source_kind == "mxfp4":
            from quark.torch.quantization.config.template import MXFP4Scheme

            source_weight = MXFP4Scheme().config.weight
            return any(
                target.weight is not None and not weight_quantization_formats_match(source_weight, target.weight)
                for target in self._targets_for_source(source_kind)
            )
        # FP8 scale granularity is confirmed against loaded tensors below.
        return True

    def wrap(self, original: Callable[..., Any], source_kind: str) -> Callable[..., Any]:
        signature = inspect.signature(original)
        config_arg = "moe_config" if "moe_config" in signature.parameters else "config"

        @functools.wraps(original)
        def select(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            config = bound.arguments[config_arg]
            if not self.targets:
                return original(*args, **kwargs)

            adapters = [a for a in SEARCH_MOE_ADAPTERS if source_kind in a.source_kinds and a.automatic]
            candidates = (
                list(dict.fromkeys(a.vllm_backend for a in adapters))
                if self.requested == "auto"
                else [to_vllm_moe_backend(self.requested)]
            )
            record: dict[str, Any] = {
                "selector": original.__name__,
                "source": source_kind,
                "activation": str(getattr(config, "activation", None)),
                "hidden_dim": getattr(config, "hidden_dim", None),
                "intermediate_size": getattr(config, "intermediate_size_per_partition", None),
                "rejected": [],
            }
            for candidate in candidates:
                candidate_config = copy.copy(config)
                candidate_config.moe_backend = candidate
                bound.arguments[config_arg] = candidate_config
                try:
                    backend, experts_cls = original(*bound.args, **bound.kwargs)
                    adapter = find_search_moe_adapter(backend, experts_cls, source_kind)
                    if self._requires_weight_conversion(source_kind) and not adapter.weight_conversion:
                        raise ValueError(
                            "Quark adapter preserves packed weights; source-to-target weight conversion is unsupported"
                        )
                except (ValueError, NotImplementedError, ImportError) as exc:
                    record["rejected"].append({"candidate": candidate, "reason": str(exc)})
                    continue
                record.update(
                    selected=adapter.backend,
                    backend=_backend_name(backend),
                    experts=_class_name(experts_cls),
                    a2_hook=adapter.a2_hook,
                )
                self.records.append(record)
                self._selected[id(config)] = (_backend_name(backend), _class_name(experts_cls), source_kind)
                # Auto enables AITER discovery globally, but a Triton layer
                # must still use Triton's expert-map representation under EP.
                if hasattr(config, "rocm_aiter_fmoe_enabled"):
                    config.rocm_aiter_fmoe_enabled = adapter.backend.startswith("aiter")
                return backend, experts_cls
            self.records.append(record)
            reasons = "; ".join(f"{r['candidate']}: {r['reason']}" for r in record["rejected"])
            raise RuntimeError(
                f"No Quark-compatible search MoE backend (requested={self.requested}, source={source_kind}, "
                f"activation={record['activation']}). {reasons}. "
                "Search requires a plugin adapter for both a1 and a2 QDQ and the source weight layout. "
                "Use --search-moe-backend=auto to try registered adapters; an unsupported implementation needs a new adapter."
            )

        return select

    def install(self) -> None:
        """Patch source functions and already imported vLLM by-name bindings."""
        if self._bindings or not self.targets:
            return
        for module_name, function_name, source_kind in _SELECTORS:
            try:
                module = importlib.import_module(_ORACLE + module_name)
            except ModuleNotFoundError as exc:
                if exc.name and (_ORACLE + module_name).startswith(exc.name):
                    continue
                raise
            original = getattr(module, function_name, None)
            if original is None:
                continue
            replacement = self.wrap(original, source_kind)
            for name, imported in list(sys.modules.items()):
                if imported is None or not name.startswith("vllm."):
                    continue
                for attr, value in list(vars(imported).items()):
                    if value is original:
                        self._bindings.append((imported, attr, original, replacement))
                        setattr(imported, attr, replacement)
        # vLLM 0.25 copies this flag to RoutedExperts before selecting its
        # quant_method. Reconcile that copy before create_weights/kernel setup.
        try:
            routed_module = importlib.import_module("vllm.model_executor.layers.fused_moe.routed_experts")
        except ModuleNotFoundError as exc:
            if exc.name and "vllm.model_executor.layers.fused_moe.routed_experts".startswith(exc.name):
                return
            raise
        routed_cls = getattr(routed_module, "RoutedExperts", None)
        original_get_method = getattr(routed_cls, "_get_quant_method", None)
        if routed_cls is not None and original_get_method is not None:

            @functools.wraps(original_get_method)
            def get_method(layer: Any, *args: Any, **kwargs: Any) -> Any:
                method = original_get_method(layer, *args, **kwargs)
                config = getattr(method, "moe", None)
                if config is not None and id(config) in self._selected:
                    layer.rocm_aiter_fmoe_enabled = config.rocm_aiter_fmoe_enabled
                return method

            self._bindings.append((routed_cls, "_get_quant_method", original_get_method, get_method))
            routed_cls._get_quant_method = get_method

    def restore(self) -> None:
        # Include bindings imported after install, without retaining whole models.
        replacements = {id(replacement): (replacement, original) for _, _, original, replacement in self._bindings}
        for owner, attr, original, replacement in reversed(self._bindings):
            if getattr(owner, attr, None) is replacement:
                setattr(owner, attr, original)
        for name, module in list(sys.modules.items()):
            if module is None or not name.startswith("vllm."):
                continue
            for attr, value in list(vars(module).items()):
                pair = replacements.get(id(value))
                if pair is not None and value is pair[0]:
                    setattr(module, attr, pair[1])
        self._bindings.clear()

    def validate_loaded_model(self, model: Any) -> None:
        """Catch unadapted selectors and verify codecs against the actual weights."""
        from quark.experimental.torch.plugin.vllm_inverse_quantizer import (
            create_vllm_moe_inverse_quantizers,
            vllm_source_weight_matches_target,
        )

        self.inventory = []
        if not self.targets:
            return
        for name, layer in model.named_modules():
            if not hasattr(layer, "w13_weight") or not hasattr(layer, "w2_weight"):
                continue
            method = getattr(layer, "quant_method", None)
            source_kind, backend = next(
                (
                    (kind, getattr(method, attr))
                    for kind, attr in (
                        ("mxfp4", "mxfp4_backend"),
                        ("fp8", "fp8_backend"),
                        ("unquantized", "unquantized_backend"),
                    )
                    if getattr(method, attr, None) is not None
                ),
                (None, None),
            )
            kernel = getattr(method, "moe_kernel", None)
            experts = getattr(kernel, "fused_experts", None)
            experts_cls = type(experts) if experts is not None else getattr(method, "experts_cls", None)
            config = getattr(method, "moe", None)
            selection = self._selected.get(id(config))
            if selection is None:
                # Some model methods construct a supported expert directly,
                # without invoking an oracle selector. Validate that concrete
                # implementation against the same finite registry; the worker
                # still requires execution probes before evaluating candidates.
                if config is None or source_kind is None:
                    raise RuntimeError(
                        f"MoE layer {name} bypassed Quark search backend negotiation without source metadata. "
                        "Register a selector and plugin QDQ adapter for this implementation."
                    )
                adapter = find_search_moe_adapter(backend, experts_cls, source_kind)
                if (self.requested == "auto" and not adapter.automatic) or (
                    self.requested != "auto" and self.requested not in (adapter.backend, adapter.vllm_backend)
                ):
                    raise RuntimeError(
                        f"MoE layer {name} selected native backend={adapter.backend}, "
                        f"which does not satisfy requested search backend={self.requested}."
                    )
                self.records.append(
                    {
                        "selector": "model_native",
                        "source": source_kind,
                        "activation": str(getattr(config, "activation", None)),
                        "hidden_dim": getattr(config, "hidden_dim", None),
                        "intermediate_size": getattr(config, "intermediate_size_per_partition", None),
                        "rejected": [],
                        "selected": adapter.backend,
                        "backend": _backend_name(backend),
                        "experts": _class_name(experts_cls),
                        "a2_hook": adapter.a2_hook,
                    }
                )
            elif selection[:2] != (_backend_name(backend), _class_name(experts_cls)):
                raise RuntimeError(
                    f"MoE layer {name} changed implementation after Quark search backend negotiation: "
                    f"method={_class_name(type(method))}, experts={_class_name(experts_cls)}. "
                    "Register a selector and plugin QDQ adapter for this implementation."
                )
            else:
                source_kind = selection[2]
            adapter = find_search_moe_adapter(backend, experts_cls, source_kind)
            needs_conversion = source_kind != "unquantized" and any(
                not vllm_source_weight_matches_target(layer, target) for target in self._targets_for_source(source_kind)
            )
            if needs_conversion:
                if not adapter.weight_conversion:
                    raise RuntimeError(f"MoE layer {name}: {adapter.backend} cannot convert the source weight layout.")
                # Construction validates metadata/scales without dequantizing or
                # copying the model. Conversion still uses the existing codec.
                create_vllm_moe_inverse_quantizers(layer)
            weight_handling = "quantize"
            if source_kind != "unquantized":
                weight_handling = "convert" if needs_conversion else "preserve"
            self.inventory.append(
                {
                    "layer": name,
                    "selected": adapter.backend,
                    "experts": adapter.expert_class,
                    "weight_handling": weight_handling,
                }
            )

    def report(self) -> dict[str, Any]:
        selected = sorted({record["selected"] for record in self.records if "selected" in record})
        return {
            "selected": selected[0] if len(selected) == 1 else "mixed" if selected else "not_required",
            "records": list(self.records),
            "layers": list(self.inventory),
            "probes": list(self.probes),
        }
