#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Search backend negotiation requires both vLLM support and a plugin QDQ adapter."""

import json
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from quark.experimental.torch.mix_precision.moe_backend import (
    resolve_search_moe_backend,
    search_moe_backend_environment,
    split_moe_backend_args,
)
from quark.experimental.torch.plugin.vllm_search_moe import (
    SEARCH_MOE_ADAPTERS,
    SearchMoeSelector,
    find_search_moe_adapter,
)


def expert_class(adapter):
    module, name = adapter.expert_class.rsplit(".", 1)
    return type(name, (), {"__module__": module})


def selector(requested="auto", modes=("mxfp4",)):
    return SearchMoeSelector({"requested": requested, "target_modes": modes})


@pytest.mark.parametrize("model_type", ["gpt_oss", "kimi_k3", "unseen_model"])
def test_driver_defers_to_worker_without_model_name_rules(model_type):
    args, decision = resolve_search_moe_backend(
        [],
        [{"routed_moe_mode": "mxfp4"}],
        {"routed_moe": 4},
        model_config={"model_type": model_type},
        source_weight_mode="mxfp4",
    )
    assert args == ["--moe-backend=auto"]
    assert decision.selected == "pending"
    assert decision.target_modes == ("mxfp4",)
    assert not decision.requires_weight_requantization


@pytest.mark.parametrize("source_kind,index", [("unquantized", 0), ("fp8", 0), ("mxfp4", 1)])
def test_auto_selects_registered_implementation_for_new_model(source_kind, index):
    adapter = SEARCH_MOE_ADAPTERS[index]
    config = SimpleNamespace(moe_backend="auto", activation="silu", rocm_aiter_fmoe_enabled=True)
    calls = []

    def original(config, activation_key=None):
        calls.append(config.moe_backend)
        assert activation_key == "source-activation"
        return SimpleNamespace(name=adapter.backend_names[0]), expert_class(adapter)

    selected = selector()
    selected.wrap(original, source_kind)(config, activation_key="source-activation")
    assert calls == [adapter.backend]
    assert selected.report()["selected"] == adapter.backend
    assert config.moe_backend == "auto"
    assert not config.rocm_aiter_fmoe_enabled


def test_vllm_activation_rejection_tries_next_plugin_adapter():
    aiter = SEARCH_MOE_ADAPTERS[2]
    calls = []

    def original(config):
        calls.append(config.moe_backend)
        if config.moe_backend == "triton_unfused":
            raise ValueError("activation not supported")
        if config.moe_backend != "aiter":
            raise ValueError(f"Unsupported vLLM backend: {config.moe_backend}")
        return SimpleNamespace(name="AITER_MXFP4_BF16"), expert_class(aiter)

    selected = selector()
    selected.wrap(original, "mxfp4")(SimpleNamespace(moe_backend="auto", activation="new_activation"))
    assert calls == ["triton_unfused", "aiter"]
    assert selected.report()["selected"] == "aiter_mxfp4_bf16"
    assert selected.records[0]["rejected"] == [{"candidate": "triton_unfused", "reason": "activation not supported"}]


@pytest.mark.parametrize("requested", ["aiter", "aiter_mxfp4_bf16"])
def test_explicit_aiter_alias_uses_vllm_family(requested):
    calls = []

    def original(config):
        calls.append(config.moe_backend)
        if config.moe_backend != "aiter":
            raise ValueError(f"Unsupported vLLM backend: {config.moe_backend}")
        return SimpleNamespace(name="AITER_MXFP4_BF16"), expert_class(SEARCH_MOE_ADAPTERS[2])

    selected = selector(requested)
    config = SimpleNamespace(moe_backend="aiter")
    selected.wrap(original, "mxfp4")(config)
    assert calls == ["aiter"]
    assert selected.requested == requested
    assert selected.report()["selected"] == "aiter_mxfp4_bf16"
    assert config.moe_backend == "aiter"


@pytest.mark.parametrize("requested", ["aiter", "aiter_mxfp4_bf16"])
@pytest.mark.parametrize("backend_name", ["AITER_MXFP4_FP8", "AITER_MXFP4_MXFP4"])
def test_aiter_family_does_not_accept_unregistered_variants(requested, backend_name):
    def original(config):
        return SimpleNamespace(name=backend_name), expert_class(SEARCH_MOE_ADAPTERS[2])

    with pytest.raises(RuntimeError, match="no search a2-QDQ adapter"):
        selector(requested).wrap(original, "mxfp4")(SimpleNamespace(moe_backend="aiter"))


@pytest.mark.parametrize("requested", ["auto", "aiter", "aiter_mxfp4_bf16"])
def test_aiter_candidates_pass_real_vllm_backend_parsing(requested):
    oracle = pytest.importorskip("vllm.model_executor.layers.fused_moe.oracle.mxfp4")
    from vllm.config.kernel import KernelConfig

    args, decision = resolve_search_moe_backend(
        [f"--moe-backend={requested}"],
        [{"routed_moe_mode": "mxfp4"}],
        {"routed_moe": 4},
        model_config=None,
        source_weight_mode="mxfp4",
    )
    runtime_backend = args[-1].split("=", 1)[1]
    KernelConfig(moe_backend=runtime_backend)
    calls = []

    def original(config):
        calls.append(config.moe_backend)
        backends = oracle.map_mxfp4_backend(config.moe_backend)
        if oracle.Mxfp4MoeBackend.TRITON_UNFUSED in backends:
            raise ValueError("Controlled Triton rejection to exercise AITER fallback")
        assert oracle.Mxfp4MoeBackend.AITER_MXFP4_BF16 in backends
        # Only the capability result is supplied here; vLLM validates the names.
        return oracle.Mxfp4MoeBackend.AITER_MXFP4_BF16, expert_class(SEARCH_MOE_ADAPTERS[2])

    selected = selector(decision.requested)
    selected.wrap(original, "mxfp4")(SimpleNamespace(moe_backend=runtime_backend))
    assert calls == (["triton_unfused", "aiter"] if requested == "auto" else ["aiter"])
    assert selected.report()["selected"] == "aiter_mxfp4_bf16"


@pytest.mark.parametrize("expert_name", ["OAITritonExperts", "OAITritonMxfp4ExpertsMonolithic", "UnknownExperts"])
def test_backend_family_name_does_not_prove_a2_support(expert_name):
    cls = type(
        expert_name, (), {"__module__": "vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe"}
    )

    def original(config):
        return SimpleNamespace(name="TRITON"), cls

    with pytest.raises(RuntimeError, match="no search a2-QDQ adapter"):
        selector("triton").wrap(original, "mxfp4")(SimpleNamespace(moe_backend="triton"))


def test_subclass_and_aiter_w4a4_are_not_implicitly_supported():
    adapter = SEARCH_MOE_ADAPTERS[2]
    cls = expert_class(adapter)
    with pytest.raises(ValueError, match="no search a2-QDQ adapter"):
        find_search_moe_adapter("AITER_MXFP4_MXFP4", cls, "mxfp4")
    with pytest.raises(ValueError, match="no search a2-QDQ adapter"):
        find_search_moe_adapter("AITER_MXFP4_BF16", type("NewExperts", (cls,), {}), "mxfp4")


def test_no_candidate_is_silently_accepted_and_errors_keep_both_reasons():
    def original(config):
        raise ValueError(config.moe_backend + " lacks TP/EP support")

    selected = selector()
    with pytest.raises(RuntimeError, match="triton_unfused.*aiter"):
        selected.wrap(original, "mxfp4")(SimpleNamespace(moe_backend="auto"))
    assert [rejected["candidate"] for rejected in selected.records[0]["rejected"]] == ["triton_unfused", "aiter"]


def test_explicit_backend_is_not_overridden():
    calls = []

    def original(config):
        calls.append(config.moe_backend)
        raise ValueError("unsupported")

    with pytest.raises(RuntimeError, match="requested=triton"):
        selector("triton").wrap(original, "mxfp4")(SimpleNamespace(moe_backend="triton"))
    assert calls == ["triton"]


def test_packed_weight_conversion_is_rejected():
    adapter = SEARCH_MOE_ADAPTERS[1]

    def original(config):
        return SimpleNamespace(name="TRITON_UNFUSED"), expert_class(adapter)

    selected = selector("triton_unfused")
    selected.targets = [SimpleNamespace(weight={"dtype": "fp4", "group_size": 64})]
    with pytest.raises(RuntimeError, match="weight conversion is unsupported"):
        selected.wrap(original, "mxfp4")(SimpleNamespace(moe_backend="auto"))


def test_heterogeneous_source_floor_keeps_w4_for_higher_precision_activation():
    selected = selector(modes=("fp8", "mxfp4"))
    fp8_target, mxfp4_target = selected._targets_for_source("mxfp4")
    assert fp8_target.weight is None
    assert fp8_target.input_tensors is not None
    assert mxfp4_target.weight is not None
    assert not selected._requires_weight_conversion("mxfp4")


def test_native_moe_requires_no_patch_and_keeps_original_selector():
    def original(config):
        return "opaque-native-runtime", None

    selected = selector(modes=("native",))
    assert selected.wrap(original, "mxfp4")(SimpleNamespace(moe_backend="auto")) == ("opaque-native-runtime", None)
    assert selected.report()["selected"] == "not_required"


def test_unexpected_vllm_bug_is_not_treated_as_incompatibility():
    def original(config):
        raise TypeError("changed selector ABI")

    with pytest.raises(TypeError, match="ABI"):
        selector().wrap(original, "mxfp4")(SimpleNamespace(moe_backend="auto"))


def test_install_patches_existing_and_future_import_bindings(monkeypatch):
    import quark.experimental.torch.plugin.vllm_search_moe as module

    adapter = SEARCH_MOE_ADAPTERS[0]

    def original(moe_config):
        return SimpleNamespace(name="TRITON"), expert_class(adapter)

    source = ModuleType("vllm.oracle_for_test")
    source.select_unquantized_moe_backend = original

    class RoutedExperts:
        def _get_quant_method(self, config):
            source.select_unquantized_moe_backend(config)
            return SimpleNamespace(moe=config)

    original_get_method = RoutedExperts._get_quant_method
    source.RoutedExperts = RoutedExperts
    consumer = ModuleType("vllm.consumer_for_test")
    consumer.aliased_selector = original
    monkeypatch.setitem(sys.modules, source.__name__, source)
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    monkeypatch.setattr(module, "_SELECTORS", (("unquantized", "select_unquantized_moe_backend", "unquantized"),))
    monkeypatch.setattr(module.importlib, "import_module", lambda name: source)
    selected = selector()
    selected.install()
    replacement = source.select_unquantized_moe_backend
    assert replacement is not original and consumer.aliased_selector is replacement
    selected.install()
    assert source.select_unquantized_moe_backend is replacement
    layer = RoutedExperts()
    layer.rocm_aiter_fmoe_enabled = True
    layer._get_quant_method(SimpleNamespace(moe_backend="auto", rocm_aiter_fmoe_enabled=True))
    assert not layer.rocm_aiter_fmoe_enabled
    future = ModuleType("vllm.future_for_test")
    future.selector = replacement
    monkeypatch.setitem(sys.modules, future.__name__, future)
    selected.restore()
    assert consumer.aliased_selector is original
    assert future.selector is original
    assert RoutedExperts._get_quant_method is original_get_method


def test_loaded_model_cannot_bypass_negotiation():
    layer = SimpleNamespace(w13_weight=object(), w2_weight=object(), quant_method=object())
    with pytest.raises(RuntimeError, match="bypassed Quark"):
        selector().validate_loaded_model(SimpleNamespace(named_modules=lambda: [("layer.experts", layer)]))


def native_mxfp4_layer(adapter_index=2):
    adapter = SEARCH_MOE_ADAPTERS[adapter_index]
    cls = expert_class(adapter)
    method = SimpleNamespace(
        moe=SimpleNamespace(activation="new_activation", hidden_dim=7168, intermediate_size_per_partition=384),
        weight_dtype="mxfp4",
        mxfp4_backend=SimpleNamespace(name=adapter.backend_names[0]),
        experts_cls=cls,
        moe_kernel=SimpleNamespace(fused_experts=cls()),
    )
    return SimpleNamespace(w13_weight=object(), w2_weight=object(), quant_method=method)


@pytest.mark.parametrize("requested", ["auto", "aiter", "aiter_mxfp4_bf16"])
def test_native_selection_requires_registered_adapter_and_matching_weights(requested):
    layer = native_mxfp4_layer()
    selected = selector(requested, modes=("mxfp4", "fp8"))
    selected.validate_loaded_model(SimpleNamespace(named_modules=lambda: [("moe", layer)]))
    assert selected.report()["selected"] == "aiter_mxfp4_bf16"
    assert selected.records[0]["selector"] == "model_native"
    assert selected.inventory[0]["weight_handling"] == "preserve"
    # Selection alone is not execution evidence.
    assert selected.probes == []


@pytest.mark.parametrize("requested,index", [("triton", 2), ("triton_unfused", 2), ("auto", 3)])
def test_native_selection_cannot_override_explicit_choice_or_enable_emulation(requested, index):
    layer = native_mxfp4_layer(index)
    with pytest.raises(RuntimeError, match="does not satisfy requested search backend"):
        selector(requested).validate_loaded_model(SimpleNamespace(named_modules=lambda: [("moe", layer)]))


def test_native_selection_validates_actual_loaded_experts():
    layer = native_mxfp4_layer()
    # The declared class is supported, but the instantiated implementation differs.
    layer.quant_method.moe_kernel.fused_experts = object()
    with pytest.raises(ValueError, match="no search a2-QDQ adapter"):
        selector().validate_loaded_model(SimpleNamespace(named_modules=lambda: [("moe", layer)]))


def test_native_selection_rejects_unknown_source_weight_format():
    layer = native_mxfp4_layer()
    layer.quant_method.weight_dtype = "unknown"
    with pytest.raises(RuntimeError, match="cannot convert the source weight layout"):
        selector().validate_loaded_model(SimpleNamespace(named_modules=lambda: [("moe", layer)]))


def test_negotiated_backend_cannot_change_to_another_supported_adapter():
    layer = native_mxfp4_layer()
    selected = selector()
    adapter = SEARCH_MOE_ADAPTERS[1]
    selected._selected[id(layer.quant_method.moe)] = ("TRITON_UNFUSED", adapter.expert_class, "mxfp4")
    with pytest.raises(RuntimeError, match="changed implementation after Quark"):
        selected.validate_loaded_model(SimpleNamespace(named_modules=lambda: [("moe", layer)]))


def test_loaded_float_and_fp8_layers_keep_distinct_source_requirements(monkeypatch):
    from quark.experimental.torch.plugin import vllm_inverse_quantizer as inverse

    selected = selector()
    cls = expert_class(SEARCH_MOE_ADAPTERS[0])
    layers = []
    converted = []
    for source in ("fp8", "unquantized"):
        config = SimpleNamespace(moe_backend="auto")

        def original(config):
            return SimpleNamespace(name="TRITON"), cls

        backend, _ = selected.wrap(original, source)(config)
        method = SimpleNamespace(moe=config, experts_cls=cls, **{source + "_backend": backend})
        layers.append((source, SimpleNamespace(w13_weight=object(), w2_weight=object(), quant_method=method)))
    monkeypatch.setattr(inverse, "vllm_source_weight_matches_target", lambda layer, target: False)
    monkeypatch.setattr(inverse, "create_vllm_moe_inverse_quantizers", lambda layer: converted.append(layer))
    selected.validate_loaded_model(SimpleNamespace(named_modules=lambda: layers))
    assert converted == [layers[0][1]]
    assert [layer["weight_handling"] for layer in selected.inventory] == ["convert", "quantize"]


def test_loaded_codec_failure_is_not_deferred_to_accuracy(monkeypatch):
    from quark.experimental.torch.plugin import vllm_inverse_quantizer as inverse

    selected = selector()
    config = SimpleNamespace(moe_backend="auto")
    cls = expert_class(SEARCH_MOE_ADAPTERS[0])

    def original(config):
        return SimpleNamespace(name="TRITON"), cls

    backend, _ = selected.wrap(original, "fp8")(config)
    method = SimpleNamespace(moe=config, experts_cls=cls, fp8_backend=backend)
    layer = SimpleNamespace(w13_weight=object(), w2_weight=object(), quant_method=method)
    monkeypatch.setattr(inverse, "vllm_source_weight_matches_target", lambda layer, target: False)

    def missing_scales(layer):
        raise ValueError("source scale metadata is missing")

    monkeypatch.setattr(inverse, "create_vllm_moe_inverse_quantizers", missing_scales)
    with pytest.raises(ValueError, match="source scale metadata"):
        selected.validate_loaded_model(SimpleNamespace(named_modules=lambda: [("moe", layer)]))


def test_explicit_choice_keeps_other_arguments():
    args, decision = resolve_search_moe_backend(
        ["--tensor-parallel-size", "4", "--moe_backend", "triton-unfused"],
        [{"routed_moe_mode": "mxfp4"}],
        {"routed_moe": 4},
        model_config=None,
        source_weight_mode="mxfp4",
    )
    assert args == ["--tensor-parallel-size", "4", "--moe-backend=triton_unfused"]
    assert decision.requested == "triton_unfused"


@pytest.mark.parametrize("backend", ["aiter", "aiter_mxfp4_bf16", "aiter-mxfp4-bf16"])
def test_driver_normalizes_aiter_alias_and_preserves_request(backend):
    args, decision = resolve_search_moe_backend(
        ["--tensor-parallel-size", "4", "--moe_backend", backend],
        [{"routed_moe_mode": "mxfp4"}],
        {"routed_moe": 4},
        model_config=None,
        source_weight_mode="mxfp4",
    )
    assert args == ["--tensor-parallel-size", "4", "--moe-backend=aiter"]
    assert decision.requested == backend.replace("-", "_")


def test_conflicting_overrides_are_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        split_moe_backend_args(["--moe-backend=triton", "--moe_backend", "aiter"])


@pytest.mark.parametrize("backend", ["auto", "aiter", "triton", "triton_unfused"])
def test_environment_and_worker_policy_are_restored_after_failure(monkeypatch, backend):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "0")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FLYDSL_MOE", "1")
    monkeypatch.setenv("QUARK_SEARCH_MOE_POLICY", "previous")
    previous = dict(os.environ)
    policy = {"requested": backend, "target_modes": ["mxfp4"]}
    with pytest.raises(RuntimeError, match="startup"), search_moe_backend_environment(backend, policy):
        assert os.environ["VLLM_ROCM_USE_AITER_MOE"] == ("1" if backend in ("auto", "aiter") else "0")
        assert json.loads(os.environ["QUARK_SEARCH_MOE_POLICY"]) == policy
        assert "VLLM_ROCM_USE_AITER_FLYDSL_MOE" not in os.environ
        raise RuntimeError("startup")
    assert dict(os.environ) == previous


@pytest.mark.parametrize("bad_result", ["shape", "dtype", "nan"])
def test_qdq_probe_rejects_invalid_activation(bad_result):
    from quark.experimental.torch.plugin.vllm_plugin import _audit_search_moe_quantizer

    tensor = torch.ones(2, 4)
    bad = {"shape": tensor[:1], "dtype": tensor.to(torch.float16), "nan": tensor * float("nan")}[bad_result]
    with pytest.raises(RuntimeError, match="QDQ"):
        _audit_search_moe_quantizer(lambda x: bad, {"a2_calls": 0}, "a2")(tensor)


@pytest.mark.parametrize("skip_second_a2", [False, True])
def test_worker_probe_checks_both_lengths_and_cleans_up(monkeypatch, skip_second_a2):
    from quark.experimental.torch.plugin import vllm_plugin as plugin
    from quark.experimental.torch.plugin.fakequant_worker import QuarkFakeQuantWorker

    class Wrapper(torch.nn.Module):
        _source_matches_target = False
        _runtime_handles_moe_activation_quantization = False

        def _get_moe_quantizers(self):
            return torch.clone, torch.clone, None, None

    layer = Wrapper()
    model = torch.nn.Sequential(layer)
    monkeypatch.setattr(plugin, "QuantVLLMFusedMoE", Wrapper)
    selected = selector()
    lengths = []

    def execute(token_ids):
        lengths.append(len(token_ids))
        audit = layer.__dict__["_search_qdq_audit"]
        audit["a1_calls"] += 1
        if not (skip_second_a2 and len(token_ids) == 8):
            # Several hits in the first probe cannot hide a missing second hit.
            audit["a2_calls"] += 3

    worker = SimpleNamespace(_search_moe_selector=selected, _execute_calibration_step=execute)
    if skip_second_a2:
        with pytest.raises(RuntimeError, match="a2 QDQ.*length=8"):
            QuarkFakeQuantWorker._validate_search_moe_qdq(worker, model)
        assert selected.probes[0]["status"] == "failed"
    else:
        QuarkFakeQuantWorker._validate_search_moe_qdq(worker, model)
        assert selected.probes[0]["status"] == "passed"
    assert lengths == [1, 8]
    assert "_search_qdq_audit" not in layer.__dict__
