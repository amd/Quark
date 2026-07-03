#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Per-directory conftest: install vLLM stubs when vLLM is not available.

The Quark online-quant plugin imports a lot of symbols from vLLM. In CI
environments without vLLM installed, we substitute minimal stub modules
into ``sys.modules`` before any test imports the plugin, so:

* the plugin's ``from vllm... import ...`` statements succeed,
* the plugin's class definitions (which inherit from vLLM classes) work,
* structural tests (MRO, configs, registry, public surface) still pass.

Tests that need *real* vLLM behavior (kernel calls, ``QuarkConfig``
parsing, layerwise pipeline, etc.) check the ``vllm_is_mocked`` fixture
and skip themselves. They can be enumerated with ``pytest -k mocked``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Detection — are we using a real vLLM or did this conftest stub it?
# ---------------------------------------------------------------------------

_VLLM_STUB_SENTINEL = "_quark_test_vllm_stub"
_REQUIRED_VLLM_MODULES = (
    "vllm",
    "vllm.distributed.communication_op",
)


def _module_spec_exists(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _vllm_real_available() -> bool:
    # If we already installed our stub, ``find_spec`` would still see a
    # module — distinguish via the sentinel.
    mod = sys.modules.get("vllm")
    if mod is not None and getattr(mod, _VLLM_STUB_SENTINEL, False):
        return False
    return all(_module_spec_exists(module_name) for module_name in _REQUIRED_VLLM_MODULES)


def _vllm_is_mocked() -> bool:
    mod = sys.modules.get("vllm")
    return mod is not None and getattr(mod, _VLLM_STUB_SENTINEL, False)


@pytest.fixture
def vllm_is_mocked() -> bool:
    """True if the test session is running against the stub vLLM."""
    return _vllm_is_mocked()


# ---------------------------------------------------------------------------
# Stub installation (executed at conftest import time, before tests run)
# ---------------------------------------------------------------------------


def _install_module(name: str, attrs: dict[str, Any]) -> types.ModuleType:
    """Create a stub module with a proper ``__spec__`` so ``pytest.importorskip``
    (which validates ``module.__spec__ is not None``) accepts it. Bare
    ``types.ModuleType(name)`` leaves ``__spec__ = None`` and trips the
    "module.__spec__ is None" guard during test collection.
    """
    import importlib.machinery

    mod = types.ModuleType(name)
    spec = importlib.machinery.ModuleSpec(name, loader=None)
    # Mark packages (any name with a child registered in sys.modules) by
    # giving them ``__path__`` — required so ``import vllm.x.y`` works.
    spec.submodule_search_locations = []
    mod.__spec__ = spec
    mod.__path__ = []  # treat every stub as a package for simplicity
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _install_vllm_stubs() -> None:
    """Create a minimal vLLM in ``sys.modules`` that satisfies the imports the
    Quark online-quant plugin makes. Behavior is intentionally minimal —
    tests that depend on real behavior should skip via ``vllm_is_mocked``.
    """
    import torch
    import torch.nn as nn

    # ------------------------------------------------------------------
    # vllm + vllm.logger + vllm.platforms
    # ------------------------------------------------------------------
    vllm_root = _install_module("vllm", {_VLLM_STUB_SENTINEL: True})

    def _init_logger(name: str):
        import logging

        return logging.getLogger(name)

    _install_module("vllm.logger", {"init_logger": _init_logger})

    class _CurrentPlatform:
        @staticmethod
        def is_fp8_fnuz() -> bool:
            return False

        @staticmethod
        def is_rocm() -> bool:
            return False

        @staticmethod
        def is_cuda() -> bool:
            return torch.cuda.is_available()

        @staticmethod
        def supports_mx() -> bool:
            return False

        @staticmethod
        def fp8_dtype():
            return torch.float8_e4m3fn

    _install_module("vllm.platforms", {"current_platform": _CurrentPlatform()})

    # ------------------------------------------------------------------
    # vllm._custom_ops
    # ------------------------------------------------------------------
    def _scaled_fp8_quant(
        input,
        scale=None,
        use_per_token_if_dynamic: bool = False,
        **_kwargs,
    ):
        # Trivial per-channel/per-tensor "quant" — just cast to fp8 and
        # compute amax/448. Sufficient for unit-test param-shape checks.
        if scale is None:
            if use_per_token_if_dynamic:
                amax = input.abs().amax(dim=1, keepdim=True).to(torch.float32)
            else:
                amax = input.abs().amax().to(torch.float32).view(1)
            scale = (amax / 448.0).clamp_min(1e-12)
        else:
            scale = scale.to(torch.float32)
        out = ((input.to(torch.float32) / scale).clamp(-448.0, 448.0)).to(torch.float8_e4m3fn)
        if not use_per_token_if_dynamic and scale.numel() == 1:
            return out, scale.view(1)
        return out, scale

    _install_module("vllm._custom_ops", {"scaled_fp8_quant": _scaled_fp8_quant})

    # ------------------------------------------------------------------
    # vllm.distributed.communication_op
    # ------------------------------------------------------------------
    _install_module("vllm.distributed", {})
    _install_module(
        "vllm.distributed.communication_op",
        {"tensor_model_parallel_all_gather": lambda x, *args, **kwargs: x},
    )

    # ------------------------------------------------------------------
    # vllm.config (set_current_vllm_config + VllmConfig stubs)
    # ------------------------------------------------------------------
    from contextlib import contextmanager

    class _VllmConfig:
        def __init__(self):
            self.model_config = types.SimpleNamespace(dtype=torch.bfloat16, hf_config=types.SimpleNamespace())

    @contextmanager
    def _set_current_vllm_config(cfg):
        yield cfg

    def _get_current_vllm_config():
        return _VllmConfig()

    _install_module(
        "vllm.config",
        {
            "VllmConfig": _VllmConfig,
            "set_current_vllm_config": _set_current_vllm_config,
            "get_current_vllm_config": _get_current_vllm_config,
        },
    )

    # ------------------------------------------------------------------
    # vllm.model_executor.{layers,kernels,model_loader,parameter,utils}
    # ------------------------------------------------------------------

    # Parent packages — empty namespaces.
    for name in [
        "vllm.model_executor",
        "vllm.model_executor.kernels",
        "vllm.model_executor.kernels.linear",
        "vllm.model_executor.kernels.linear.scaled_mm",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.layers.quantization.quark",
        "vllm.model_executor.layers.quantization.quark.schemes",
        "vllm.model_executor.layers.quantization.utils",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.reload",
    ]:
        if name not in sys.modules:
            _install_module(name, {})

    # ---- parameter ----
    class ModelWeightParameter(nn.Parameter):
        def __new__(
            cls,
            data,
            input_dim=None,
            output_dim=None,
            weight_loader=None,
        ):
            inst = super().__new__(cls, data, requires_grad=False)
            inst.input_dim = input_dim
            inst.output_dim = output_dim
            inst.weight_loader = weight_loader
            return inst

    class _ScaleParameter(nn.Parameter):
        def __new__(cls, data, weight_loader=None, **_kwargs):
            inst = super().__new__(cls, data, requires_grad=False)
            inst.weight_loader = weight_loader
            return inst

    _install_module(
        "vllm.model_executor.parameter",
        {
            "ModelWeightParameter": ModelWeightParameter,
            "ChannelQuantScaleParameter": _ScaleParameter,
            "PerTensorScaleParameter": _ScaleParameter,
        },
    )

    # ---- utils.replace_parameter ----
    def _replace_parameter(layer, name, value):
        # nn.Module.register_parameter is the canonical API; wrap if needed.
        if not isinstance(value, nn.Parameter):
            value = nn.Parameter(value, requires_grad=False)
        layer.register_parameter(name, value)

    _install_module(
        "vllm.model_executor.utils",
        {"replace_parameter": _replace_parameter},
    )

    # ---- quantization.base_config ----
    import abc

    class QuantizationConfig:
        def __init__(self):
            self.packed_modules_mapping: dict = {}

        def apply_vllm_mapper(self, hf_to_vllm_mapper):
            pass

        def get_cache_scale(self, name):
            return None

        @classmethod
        def get_supported_act_dtypes(cls):
            return [torch.float16, torch.bfloat16]

        @classmethod
        def get_min_capability(cls):
            return 70

        @classmethod
        def get_config_filenames(cls):
            return []

        @classmethod
        def get_name(cls):
            return "stub"

        @classmethod
        def from_config(cls, config):
            return cls()

        def get_quant_method(self, layer, prefix):
            return None

    # ``QuantizeMethodBase`` is intentionally an empty ABC — vLLM uses
    # it as a virtual-subclass anchor (``QuantizeMethodBase.register(...)``)
    # rather than declaring abstract methods. Suppress B024.
    class QuantizeMethodBase(metaclass=abc.ABCMeta):  # noqa: B024
        pass

    _install_module(
        "vllm.model_executor.layers.quantization.base_config",
        {
            "QuantizationConfig": QuantizationConfig,
            "QuantizeMethodBase": QuantizeMethodBase,
        },
    )

    # ---- quantization (registry + types) ----
    _REGISTRY: dict[str, type] = {}

    def register_quantization_config(name: str):
        def deco(cls):
            _REGISTRY[name] = cls
            return cls

        return deco

    def get_quantization_config(name: str):
        if name not in _REGISTRY:
            raise KeyError(f"no quant config registered for {name!r}")
        return _REGISTRY[name]

    # Pre-register "fp8" so scenario-B tests can resolve the offline class.
    class _StubFp8Config(QuantizationConfig):
        def __init__(self, quant_config=None, **kwargs):
            super().__init__()
            # Store the original config dict under ``quant_config`` (the
            # convention real Quark uses) so ``_offline_dict`` can find it.
            self.quant_config = dict(quant_config or {})
            self.weight_block_size = self.quant_config.get("weight_block_size")

        @classmethod
        def get_name(cls):
            return "fp8"

        @classmethod
        def from_config(cls, config):
            return cls(quant_config=dict(config))

        def get_quant_method(self, layer, prefix):
            return None

    _REGISTRY["fp8"] = _StubFp8Config

    _install_module(
        "vllm.model_executor.layers.quantization",
        {
            "QuantizationMethods": str,  # type alias
            "register_quantization_config": register_quantization_config,
            "get_quantization_config": get_quantization_config,
        },
    )

    # ---- quantization.quark.utils ----
    import fnmatch as _fnmatch

    def should_ignore_layer(prefix, ignore=None, fused_mapping=None):
        if not ignore:
            return False
        for pat in ignore:
            if "*" in pat and _fnmatch.fnmatch(prefix, pat):
                return True
            if pat == prefix or prefix.endswith("." + pat) or pat in prefix:
                return True
        return False

    _install_module(
        "vllm.model_executor.layers.quantization.quark.utils",
        {"should_ignore_layer": should_ignore_layer},
    )

    # ---- quantization.quark.quark.QuarkConfig ----
    class QuarkConfig(QuantizationConfig):
        def __init__(self, quant_config=None, **_kwargs):
            super().__init__()
            self.quant_config = quant_config or {}

        @classmethod
        def from_config(cls, config):
            inst = cls(quant_config=config)
            return inst

        def _find_matched_config(self, layer_name, module):
            # Walk layer_quant_config patterns, fnmatch on prefix, fall back
            # to global_quant_config. Matches real QuarkConfig semantics
            # closely enough for our dispatch tests.
            lqc = self.quant_config.get("layer_quant_config") or {}
            for pat, cfg in lqc.items():
                if "*" in pat:
                    if _fnmatch.fnmatch(layer_name, pat):
                        return cfg
                elif layer_name == pat:
                    return cfg
            return self.quant_config.get("global_quant_config") or {}

        @classmethod
        def get_name(cls):
            return "quark"

    _install_module(
        "vllm.model_executor.layers.quantization.quark.quark",
        {"QuarkConfig": QuarkConfig},
    )

    # ---- quantization.utils.quant_utils ----
    class _GroupShape:
        PER_TOKEN = "per_token"
        PER_TENSOR = "per_tensor"

    class _QuantKey:
        def __init__(self, name, group_shape):
            self.name = name
            self.scale = types.SimpleNamespace(group_shape=group_shape)

    _kFp8DynamicTokenSym = _QuantKey("dyn_tok", _GroupShape.PER_TOKEN)
    _kFp8StaticTokenSym = _QuantKey("static_tok", _GroupShape.PER_TOKEN)
    _kFp8StaticTensorSym = _QuantKey("static_tensor", _GroupShape.PER_TENSOR)

    _install_module(
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        {
            "GroupShape": _GroupShape,
            "kFp8DynamicTokenSym": _kFp8DynamicTokenSym,
            "kFp8StaticTokenSym": _kFp8StaticTokenSym,
            "kFp8StaticTensorSym": _kFp8StaticTensorSym,
        },
    )

    # ---- quantization.utils.w8a8_utils ----
    def _normalize_e4m3fn_to_e4m3fnuz(weight, weight_scale, input_scale=None):
        # In the stub world we pretend no normalization is needed.
        return weight, weight_scale, input_scale

    _install_module(
        "vllm.model_executor.layers.quantization.utils.w8a8_utils",
        {"normalize_e4m3fn_to_e4m3fnuz": _normalize_e4m3fn_to_e4m3fnuz},
    )

    # ---- quantization.utils.ocp_mx_utils ----
    _install_module(
        "vllm.model_executor.layers.quantization.utils.ocp_mx_utils",
        {"OCP_MX_BLOCK_SIZE": 32},
    )

    # ---- quantization.utils.mxfp4_utils ----
    def _dequant_mxfp4(weight, scale, out_dtype):
        return weight.to(out_dtype)

    def _quant_dequant_mxfp4(x):
        return x

    _install_module(
        "vllm.model_executor.layers.quantization.utils.mxfp4_utils",
        {
            "dequant_mxfp4": _dequant_mxfp4,
            "quant_dequant_mxfp4": _quant_dequant_mxfp4,
        },
    )

    # ---- kernels.linear.* (init_fp8_linear_kernel + Marlin kernel marker) ----
    class _FakeFP8Kernel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def apply_weights(self, layer, x, bias=None):
            return x

        def process_weights_after_loading(self, layer):
            pass

    def _init_fp8_linear_kernel(**kwargs):
        return _FakeFP8Kernel(**kwargs)

    _install_module(
        "vllm.model_executor.kernels.linear",
        {"init_fp8_linear_kernel": _init_fp8_linear_kernel},
    )

    class _MarlinFP8ScaledMMLinearKernel(_FakeFP8Kernel):
        pass

    _install_module(
        "vllm.model_executor.kernels.linear.scaled_mm",
        {},
    )
    _install_module(
        "vllm.model_executor.kernels.linear.scaled_mm.marlin",
        {"MarlinFP8ScaledMMLinearKernel": _MarlinFP8ScaledMMLinearKernel},
    )

    # ---- layers.linear ----
    # LinearMethodBase uses ABCMeta so the plugin's
    # ``LinearMethodBase.register(MyClass)`` calls work.
    class LinearMethodBase(QuantizeMethodBase, metaclass=abc.ABCMeta):
        def create_weights(self, *args, **kwargs):
            raise NotImplementedError

        def apply(self, layer, x, bias=None):
            raise NotImplementedError

    class LinearBase(nn.Module):
        def __init__(self):
            super().__init__()

    class UnquantizedLinearMethod(LinearMethodBase):
        def create_weights(self, layer, **kwargs):
            return None

        def apply(self, layer, x, bias=None):
            return x

    _install_module(
        "vllm.model_executor.layers.linear",
        {
            "LinearBase": LinearBase,
            "LinearMethodBase": LinearMethodBase,
            "UnquantizedLinearMethod": UnquantizedLinearMethod,
        },
    )

    # ---- layers.fused_moe ----
    class FusedMoEMethodBase(QuantizeMethodBase):
        def __init__(self, moe=None):
            self.moe = moe

    class FusedMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe_config = types.SimpleNamespace(has_bias=False)

    # Real vLLM has ``RoutedExperts: TypeAlias = FusedMoE``.
    RoutedExperts = FusedMoE

    _install_module(
        "vllm.model_executor.layers.fused_moe",
        {
            "FusedMoEMethodBase": FusedMoEMethodBase,
            "FusedMoE": FusedMoE,
            "RoutedExperts": RoutedExperts,
        },
    )
    _install_module(
        "vllm.model_executor.layers.fused_moe.layer",
        {"FusedMoE": FusedMoE},
    )

    # ---- quantization.quark.schemes.* ----
    class _QuarkScheme:
        def create_weights(self, layer, *args, **kwargs):
            return None

        def apply_weights(self, layer, x, bias=None):
            return x

        def process_weights_after_loading(self, layer):
            pass

    class QuarkW8A8Fp8(_QuarkScheme):
        def __init__(self, weight_config, input_config):
            self.weight_config = weight_config
            self.input_config = input_config
            self.weight_qscheme = weight_config.get("qscheme") if weight_config else None
            self.is_static_input_scheme = bool(input_config) and not input_config.get("is_dynamic")
            per_token = self.weight_qscheme == "per_channel"
            self.weight_quant_key = _kFp8StaticTokenSym if per_token else _kFp8StaticTensorSym
            self.activation_quant_key = _kFp8DynamicTokenSym if per_token else _kFp8StaticTensorSym
            self.input_dtype = torch.bfloat16
            self.out_dtype = torch.bfloat16

    _install_module(
        "vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_fp8",
        {"QuarkW8A8Fp8": QuarkW8A8Fp8},
    )

    class QuarkOCP_MX(_QuarkScheme):
        def __init__(self, weight_config=None, input_config=None):
            self.weight_config = weight_config
            self.input_config = input_config

    _install_module(
        "vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx",
        {
            "QuarkOCP_MX": QuarkOCP_MX,
            "is_rocm_aiter_fp4_asm_gemm_enabled": lambda: False,
        },
    )

    # ---- quantization.quark.quark_moe ----
    class QuarkW8A8Fp8MoEMethod(FusedMoEMethodBase):
        def __init__(self, weight_config, input_config, moe_config):
            super().__init__(moe=moe_config)
            self.weight_config = weight_config
            self.input_config = input_config
            self.rocm_aiter_moe_enabled = False
            self.use_marlin = False

    class QuarkOCP_MX_MoEMethod(FusedMoEMethodBase):
        def __init__(self, weight_config, input_config, moe_config):
            super().__init__(moe=moe_config)
            self.weight_config = weight_config
            self.input_config = input_config

    _install_module(
        "vllm.model_executor.layers.quantization.quark.quark_moe",
        {
            "QuarkW8A8Fp8MoEMethod": QuarkW8A8Fp8MoEMethod,
            "QuarkOCP_MX_MoEMethod": QuarkOCP_MX_MoEMethod,
        },
    )

    # ---- model_loader.reload.layerwise ----
    def _initialize_online_processing(layer):
        # Real version wires the layerwise loader; stub does nothing.
        pass

    _install_module(
        "vllm.model_executor.model_loader.reload.layerwise",
        {"initialize_online_processing": _initialize_online_processing},
    )

    # ---- model_loader.weight_utils ----
    def _initialize_single_dummy_weight(param):
        with torch.no_grad():
            param.data = torch.empty(param.shape, dtype=param.dtype).uniform_(-1e-3, 1e-3)

    _install_module(
        "vllm.model_executor.model_loader.weight_utils",
        {"initialize_single_dummy_weight": _initialize_single_dummy_weight},
    )

    # Re-expose subpackages on parent modules so attribute traversal works
    # (e.g. ``vllm.model_executor.layers.linear`` accessed from the top).
    for full in list(sys.modules.keys()):
        if not full.startswith("vllm."):
            continue
        parent, _, leaf = full.rpartition(".")
        if parent in sys.modules:
            setattr(sys.modules[parent], leaf, sys.modules[full])

    # Mark our root with the sentinel for detection.
    setattr(vllm_root, _VLLM_STUB_SENTINEL, True)


# Run on conftest load so the stub is in place before any test imports.
if not _vllm_real_available():
    _install_vllm_stubs()


# ---------------------------------------------------------------------------
# Marker: needs_real_vllm — auto-skip when stubs are in use.
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    if not _vllm_is_mocked():
        return
    skip = pytest.mark.skip(reason="needs real vLLM behavior (stubbed in CI)")
    for item in items:
        if "needs_real_vllm" in item.keywords:
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_real_vllm: test requires real vLLM (not stubbed)")
