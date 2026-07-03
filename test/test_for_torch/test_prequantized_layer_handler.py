#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Tests for :mod:`quark.torch.export.prequantized_layer_handler`.

Focused unit tests for the routing/preserve/dequant entry points and the
private helpers (collapse, lost-mxfp4 detection, on-disk config fallback,
preserved-config injection).
"""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from quark.torch.export import prequantized_layer_handler as handler
from quark.torch.export.prequantized_layer_handler import (
    _collapse_names_to_patterns,
    _collect_lost_mxfp4_layers,
    _inject_preserved_configs,
    _on_disk_quantization_config,
    _resolve_model_default_dtype,
    apply_prequantized_routing,
    dequantize_prequantized_linears,
    preserve_prequantized_layers,
)
from quark.torch.quantization.config.config import QConfig, QLayerConfig
from quark.torch.quantization.config.template import LLMTemplate, MXFP4WeightOnlyScheme

_GLOBAL_CFG = LLMTemplate._SCHEME_COLLECTION.get_scheme("fp8").config


# ============================================================================
#  _collapse_names_to_patterns
# ============================================================================


def test_collapse_empty_names_returns_empty():
    assert _collapse_names_to_patterns([], namespace=set()) == []


def test_collapse_single_name_unchanged():
    """A single name in a group cannot collapse to a wildcard."""
    out = _collapse_names_to_patterns(["layers.0.q_proj"], namespace={"layers.0.q_proj"})
    assert out == ["layers.0.q_proj"]


def test_collapse_collapses_shared_signature():
    """Two names with the same digit-shape collapse to a single wildcard pattern."""
    names = ["layers.0.q_proj", "layers.1.q_proj"]
    namespace = set(names)
    out = _collapse_names_to_patterns(names, namespace)
    assert out == ["layers.*.q_proj"]


def test_collapse_keeps_names_when_pattern_would_match_forbidden():
    """Collapse falls back to literal names when wildcard would catch a non-target."""
    names = ["layers.0.q_proj", "layers.1.q_proj"]
    # Forbidden name shares the shape signature, so the wildcard would also match it.
    namespace = set(names) | {"layers.2.q_proj"}
    out = _collapse_names_to_patterns(names, namespace)
    assert sorted(out) == sorted(names)


# ============================================================================
#  _collect_lost_mxfp4_layers
# ============================================================================


def _model_with_qconfig(qconfig, name_or_path: str | None = None):
    model = nn.Module()
    model.config = type("C", (), {"quantization_config": qconfig, "_name_or_path": name_or_path})()
    return model


def test_collect_lost_layers_returns_empty_when_no_quant_config():
    model = _model_with_qconfig(None)
    lost, scheme = _collect_lost_mxfp4_layers(model, linear_namespace={"a.b"})
    assert lost == []
    assert scheme is None


def _typed_qconfig(quant_method, modules_to_not_convert=None):
    """Stand-in for HF's typed quantization config object (e.g. Mxfp4Config)."""
    return type(
        "TypedQConfig",
        (),
        {"quant_method": quant_method, "modules_to_not_convert": modules_to_not_convert},
    )()


def test_collect_lost_layers_returns_empty_for_unknown_quant_method():
    model = _model_with_qconfig(_typed_qconfig("unknown", []))
    lost, scheme = _collect_lost_mxfp4_layers(model, linear_namespace={"a.b"})
    assert lost == []
    assert scheme is None


def test_collect_lost_layers_filters_by_modules_to_not_convert():
    """Names matching skip_patterns must NOT be returned."""
    model = _model_with_qconfig(_typed_qconfig("mxfp4", ["router"]))
    namespace = {"layers.0.mlp.gate", "layers.0.mlp.router", "router.dense"}
    lost, scheme = _collect_lost_mxfp4_layers(model, namespace)
    assert scheme == "mxfp4_weight_only"
    assert "layers.0.mlp.gate" in lost
    assert "router.dense" not in lost  # excluded
    assert "layers.0.mlp.router" in lost  # only top-level "router" matches


def test_collect_lost_layers_filters_by_wildcard_modules_to_not_convert():
    """Wildcard skip_patterns (e.g. ``*.mlp.router``) must match via fnmatch; the old
    ``startswith`` matcher would leak ``layers.0.mlp.router`` into ``lost``."""
    model = _model_with_qconfig(_typed_qconfig("mxfp4", ["*.mlp.router"]))
    namespace = {"layers.0.mlp.gate", "layers.0.mlp.router", "layers.1.mlp.router"}
    lost, scheme = _collect_lost_mxfp4_layers(model, namespace)
    assert scheme == "mxfp4_weight_only"
    assert "layers.0.mlp.gate" in lost
    assert "layers.0.mlp.router" not in lost
    assert "layers.1.mlp.router" not in lost


class _EnumMethod:
    value = "mxfp4"


@pytest.mark.parametrize(
    "qconfig",
    [
        _typed_qconfig("mxfp4", []),  # string quant_method
        _typed_qconfig(_EnumMethod(), []),  # enum quant_method with .value
    ],
    ids=["string_quant_method", "enum_quant_method"],
)
def test_collect_lost_layers_extracts_quant_method_from_variants(qconfig):
    """quant_method may be a plain string or an enum with .value — both yield mxfp4."""
    model = _model_with_qconfig(qconfig)
    _, scheme = _collect_lost_mxfp4_layers(model, linear_namespace={"layers.0.q_proj"})
    assert scheme == "mxfp4_weight_only"


def test_collect_lost_layers_uses_on_disk_fallback(tmp_path):
    """When in-memory config is wiped, _on_disk_quantization_config is consulted."""
    cfg_dir = tmp_path
    (cfg_dir / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "mxfp4", "modules_to_not_convert": ["router"]}})
    )
    # In-memory config is wiped (None); both required fields come from disk.
    model = _model_with_qconfig(None, name_or_path=str(cfg_dir))
    namespace = {"layers.0.mlp.gate"}
    lost, scheme = _collect_lost_mxfp4_layers(model, namespace)
    assert scheme == "mxfp4_weight_only"
    assert "layers.0.mlp.gate" in lost


# ============================================================================
#  _on_disk_quantization_config
# ============================================================================


def test_on_disk_config_returns_empty_without_name_or_path():
    model = _model_with_qconfig(None, name_or_path=None)
    assert _on_disk_quantization_config(model) == {}


def test_on_disk_config_returns_empty_when_file_missing(tmp_path):
    """Local dir exists but no config.json → returns {} (no crash)."""
    model = _model_with_qconfig(None, name_or_path=str(tmp_path))
    assert _on_disk_quantization_config(model) == {}


def test_on_disk_config_returns_empty_on_json_error(tmp_path):
    (tmp_path / "config.json").write_text("not json at all {")
    model = _model_with_qconfig(None, name_or_path=str(tmp_path))
    assert _on_disk_quantization_config(model) == {}


def test_on_disk_config_returns_parsed_block(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "mxfp4"}}))
    model = _model_with_qconfig(None, name_or_path=str(tmp_path))
    assert _on_disk_quantization_config(model) == {"quant_method": "mxfp4"}


def test_on_disk_config_consults_hf_cache_for_repo_id():
    """Non-dir name_or_path is treated as an HF repo id."""
    model = _model_with_qconfig(None, name_or_path="org/repo-name")
    # try_to_load_from_cache returns None for missing → returns {}
    with (
        patch.object(handler, "_on_disk_quantization_config", wraps=_on_disk_quantization_config),
        patch.object(handler, "try_to_load_from_cache", return_value=None),
    ):
        assert _on_disk_quantization_config(model) == {}


def test_on_disk_config_returns_empty_when_hub_unavailable():
    """Repo-id path but huggingface_hub not installed → returns {} (no crash)."""
    model = _model_with_qconfig(None, name_or_path="org/repo-name")
    with patch.object(handler, "is_huggingface_hub_available", return_value=False):
        assert _on_disk_quantization_config(model) == {}


# ============================================================================
#  _resolve_model_default_dtype
# ============================================================================


def test_resolve_dtype_from_torch_dtype_attr():
    """model.dtype as a supported torch.dtype is returned directly."""
    model = nn.Module()
    model.dtype = torch.bfloat16
    assert _resolve_model_default_dtype(model) == torch.bfloat16


def test_resolve_dtype_skips_unsupported_and_reads_string_config():
    """Unsupported torch.dtype is skipped; falls through to a string config dtype."""
    model = nn.Module()
    model.dtype = torch.int8  # unsupported → continue
    model.config = type("C", (), {"torch_dtype": "float16"})()
    assert _resolve_model_default_dtype(model) == torch.float16


# ============================================================================
#  apply_prequantized_routing
# ============================================================================


def _make_qconfig(exclude=None, keep=True):
    return QConfig(global_quant_config=_GLOBAL_CFG, exclude=exclude or [], keep_prequantized_layers=keep)


def test_apply_routing_noop_when_keep_disabled():
    """keep_prequantized_layers=False → routing returns immediately."""
    qc = _make_qconfig(keep=False)
    model = _model_with_qconfig(None)
    # Should not raise even though the model has no real structure.
    apply_prequantized_routing(qc, model)
    assert qc.layer_quant_config == {}


def test_apply_routing_noop_when_no_excluded_lost_layers():
    """No lost-mxfp4 names match exclude → no mutation."""
    qc = _make_qconfig(exclude=["fnord"])
    model = _model_with_qconfig(_typed_qconfig("mxfp4", []))
    # No nn.Linear modules on this skeleton model → no lost names.
    apply_prequantized_routing(qc, model)
    assert qc.layer_quant_config == {}


def test_apply_routing_injects_scheme_and_drops_exclude():
    """Excluded lost-mxfp4 names get a layer_quant_config entry; matching exclude is dropped."""
    parent = nn.Module()
    parent.layers_0_q_proj = nn.Linear(8, 8)
    parent.layers_1_q_proj = nn.Linear(8, 8)
    # Inject names with dots that match collapse semantics.
    real_model = nn.Module()
    real_model.add_module("layers", nn.Module())
    real_model.layers.add_module("0", nn.Linear(8, 8))
    real_model.layers.add_module("1", nn.Linear(8, 8))
    real_model.config = type(
        "C",
        (),
        {
            "quantization_config": _typed_qconfig("mxfp4", []),
            "_name_or_path": None,
        },
    )()

    qc = _make_qconfig(exclude=["layers.*"])
    apply_prequantized_routing(qc, real_model)

    # Both layer.0 and layer.1 are excluded → collapsed pattern injected.
    assert any("layers.*" in k for k in qc.layer_quant_config)
    # The matching exclude pattern is dropped (otherwise exclude would override).
    assert "layers.*" not in qc.exclude


def test_apply_routing_keeps_wildcard_protection_for_non_lost_siblings():
    """Wildcard that straddles lost-mxfp4 and non-lost layers (e.g. ``*.gate``
    covering both ``mlp.gate`` and ``router.gate``) should expand into the
    non-lost names rather than being dropped wholesale."""
    real_model = nn.Module()
    real_model.add_module("mlp", nn.Module())
    real_model.mlp.add_module("gate", nn.Linear(8, 8))  # lost-mxfp4
    real_model.add_module("router", nn.Module())
    real_model.router.add_module("gate", nn.Linear(8, 8))  # plain bf16
    real_model.config = type(
        "C",
        (),
        {"quantization_config": _typed_qconfig("mxfp4", ["router"]), "_name_or_path": None},
    )()

    qc = _make_qconfig(exclude=["*.gate"])
    apply_prequantized_routing(qc, real_model)

    # router.gate is still protected; mlp.gate moved into layer_quant_config.
    assert "router.gate" in qc.exclude
    assert "mlp.gate" not in qc.exclude


# ============================================================================
#  preserve_prequantized_layers
# ============================================================================


def test_preserve_prequantized_layers_noop_when_nothing_to_preserve():
    """No pre-quantized modules → no mutation, no exception."""
    model = nn.Module()
    model.lin = nn.Linear(8, 8)
    qc = _make_qconfig(exclude=[])
    preserve_prequantized_layers(model, custom_mode="fp8", pack_method=None, quantization_config=qc)
    assert qc.layer_quant_config == {}


def test_preserve_prequantized_layers_injects_configs_and_drops_exclude():
    """Pre-quantized modules are rewrapped; their configs reach layer_quant_config."""
    model = nn.Module()
    fake_module = nn.Linear(8, 8)
    model.add_module("layers", nn.Module())
    model.layers.add_module("0", fake_module)

    qc = _make_qconfig(exclude=["layers.0"])

    fake_layer_cfg = Mock(spec=QLayerConfig)
    fake_layer_cfg.to_dict.return_value = {"k": "v"}

    fake_qparams_linear = nn.Linear(8, 8)
    fake_qparams_linear._quant_config = fake_layer_cfg

    with (
        patch.object(handler, "find_prequantized_linears", return_value=[("layers.0", fake_module)]),
        patch(
            "quark.torch.export.nn.modules.qparamslinear.QParamsLinear.from_module", return_value=fake_qparams_linear
        ),
    ):
        preserve_prequantized_layers(model, custom_mode="fp8", pack_method=None, quantization_config=qc)

    # The exclude entry was dropped.
    assert "layers.0" not in qc.exclude
    # And the layer config was injected.
    assert any(cfg is fake_layer_cfg for cfg in qc.layer_quant_config.values())


# ============================================================================
#  dequantize_prequantized_linears
# ============================================================================


def test_dequantize_prequantized_linears_noop_when_no_prequant_modules():
    """When no pre-quantized linears are found, nothing happens."""
    model = nn.Module()
    model.lin = nn.Linear(8, 8)
    with patch.object(handler, "find_prequantized_linears", return_value=[]):
        dequantize_prequantized_linears(model)  # should not raise


def test_dequantize_prequantized_linears_replaces_modules():
    """Each pre-quantized module is replaced via dequantize_prequantized_to_linear."""
    model = nn.Module()
    original = nn.Linear(8, 8)
    model.add_module("inner", nn.Module())
    model.inner.add_module("lin", original)

    replacement = nn.Linear(8, 8, bias=False)
    with (
        patch.object(handler, "find_prequantized_linears", return_value=[("inner.lin", original)]),
        patch.object(handler, "dequantize_prequantized_to_linear", return_value=replacement),
    ):
        dequantize_prequantized_linears(model)

    assert model.inner.lin is replacement


# ============================================================================
#  _inject_preserved_configs
# ============================================================================


def test_inject_preserved_configs_collapses_shared_configs():
    """Names that share an identical config are collapsed into one wildcard."""
    cfg = Mock(spec=QLayerConfig)
    cfg.to_dict.return_value = {"k": "v"}

    preserved = {"layers.0.q_proj": cfg, "layers.1.q_proj": cfg}
    namespace = set(preserved)
    qc = _make_qconfig(exclude=["layers.*.q_proj"])

    _inject_preserved_configs(preserved, qc, namespace)

    assert "layers.*.q_proj" in qc.layer_quant_config
    # The matching exclude pattern is dropped (it matches the preserved names).
    assert "layers.*.q_proj" not in qc.exclude


def test_mxfp4_weight_only_scheme_registered_and_weight_only():
    """The mxfp4_weight_only scheme is registered and produces a weight-only QLayerConfig."""

    scheme = LLMTemplate._SCHEME_COLLECTION.get_scheme("mxfp4_weight_only")
    assert isinstance(scheme, MXFP4WeightOnlyScheme)
    cfg = scheme.config
    assert cfg.weight is not None
    # Weight-only: no input/output specs.
    assert cfg.input_tensors is None
    assert cfg.output_tensors is None


def test_inject_preserved_configs_keeps_unrelated_exclude():
    """Exclude patterns that DON'T match any preserved name stay put."""
    cfg = Mock(spec=QLayerConfig)
    cfg.to_dict.return_value = {"k": "v"}
    preserved = {"layers.0.q_proj": cfg}
    qc = _make_qconfig(exclude=["other.*"])
    _inject_preserved_configs(preserved, qc, namespace=set(preserved))
    assert "other.*" in qc.exclude
