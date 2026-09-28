#
# Copyright (C) 2025 - 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Processors must not do raw CPU roundtrips when accelerate manages device placement.

Moving a layer with ``.to("cpu")`` desyncs accelerate's hook-tracked device state from the
module's real device, which crashes with a cuda/cpu mismatch under ``--multi_gpu auto``.
Every processor that offloads decoder blocks has to gate that on ``using_accelerate``.
"""

import ast
import inspect
from unittest.mock import MagicMock, call

import pytest
import torch
import torch.nn as nn

from quark.torch.algorithm.awq import auto_smooth as auto_smooth_module
from quark.torch.algorithm.awq.auto_smooth import AutoSmoothQuantProcessor


@pytest.mark.parametrize("has_hook", [True, False])
def test_using_accelerate_tracks_hf_hook(monkeypatch: pytest.MonkeyPatch, has_hook: bool) -> None:
    """``using_accelerate`` mirrors accelerate's ``_hf_hook``, which gates the offloads below."""
    monkeypatch.setattr(AutoSmoothQuantProcessor, "init_quant", lambda self: ([], {}, []))
    monkeypatch.setattr(auto_smooth_module, "init_device_map", lambda model: {})
    monkeypatch.setattr(auto_smooth_module, "get_num_attn_heads_from_model", lambda model: (8, 8))

    model = nn.Module()
    model.device = torch.device("cpu")
    if has_hook:
        model._hf_hook = MagicMock()

    processor = AutoSmoothQuantProcessor(model, MagicMock(), data_loader=MagicMock())

    assert processor.using_accelerate is has_hook


def _fake_layer() -> MagicMock:
    """A stand-in decoder layer that records ``.to(...)`` calls and reports itself on CPU."""
    layer = MagicMock()
    layer.to.return_value = layer
    layer.parameters.side_effect = lambda: iter([nn.Parameter(torch.zeros(1))])
    return layer


@pytest.mark.parametrize("using_accelerate", [True, False])
def test_init_quant_cpu_offload_respects_accelerate(monkeypatch: pytest.MonkeyPatch, using_accelerate: bool) -> None:
    """``init_quant`` offloads all but the first layer to CPU only when accelerate is absent."""
    layers = [_fake_layer(), _fake_layer()]
    monkeypatch.setattr(auto_smooth_module, "get_model_layers", lambda model, path: layers)
    monkeypatch.setattr(auto_smooth_module, "reset_model_kv_cache", lambda model, use_cache: True)
    monkeypatch.setattr(auto_smooth_module, "cache_model_inps", lambda model, mods, loader: (mods, {}, []))
    monkeypatch.setattr(auto_smooth_module, "clear_memory", lambda: None)

    processor = object.__new__(AutoSmoothQuantProcessor)
    processor.model = MagicMock()
    processor.data_loader = MagicMock()
    processor.model_decoder_layers = "layers"
    processor.using_accelerate = using_accelerate

    processor.init_quant()

    assert layers[0].to.call_args_list == []
    assert layers[1].to.call_args_list == ([] if using_accelerate else [call("cpu")])


@pytest.mark.parametrize("using_accelerate", [True, False])
def test_apply_cpu_offload_respects_accelerate(monkeypatch: pytest.MonkeyPatch, using_accelerate: bool) -> None:
    """``apply`` relocates layers only when accelerate is absent.

    Covers all three moves: the CPU offload before and after each layer, and the device_map
    fallback that pulls a CPU-resident layer back onto its compute device.
    """
    layer = _fake_layer()
    monkeypatch.setattr(auto_smooth_module, "clear_memory", lambda: None)
    monkeypatch.setattr(auto_smooth_module, "get_named_quant_linears", lambda module: {})
    monkeypatch.setattr(auto_smooth_module, "get_moe_layers", lambda module: {})
    monkeypatch.setattr(auto_smooth_module, "get_layers_for_scaling", lambda *args, **kwargs: [])
    monkeypatch.setattr(auto_smooth_module, "apply_scale", lambda *args, **kwargs: None)
    monkeypatch.setattr(auto_smooth_module, "append_str_prefix", lambda scales, prefix: scales)
    monkeypatch.setattr(auto_smooth_module, "get_op_name", lambda model, module: "layer")
    monkeypatch.setattr(AutoSmoothQuantProcessor, "_get_input_feat", lambda self, layer, named: {})

    processor = object.__new__(AutoSmoothQuantProcessor)
    processor.model = MagicMock()
    processor.modules = [layer]
    # Non-CPU so the layer's own device placement stays distinguishable from an offload.
    processor.device_map = {"layers.0": "cuda:0"}
    processor.model_decoder_layers = "layers"
    processor.module_kwargs = {}
    processor.scaling_layers = []
    processor.num_attention_heads = 8
    processor.num_key_value_heads = 8
    processor.using_accelerate = using_accelerate

    processor.apply()

    moves = layer.to.call_args_list
    if using_accelerate:
        assert moves == [], f"accelerate manages placement; apply() must not relocate: {moves}"
    else:
        # offload to cpu, pull back onto the device_map device, offload again
        assert moves == [call("cpu"), call("cuda:0"), call("cpu")]


# Modules that offload decoder blocks to CPU to save memory. Each must gate that on
# `using_accelerate`, or `--multi_gpu auto` desyncs accelerate's hook-tracked placement.
_OFFLOADING_MODULES = [
    "quark.torch.algorithm.awq.awq",
    "quark.torch.algorithm.awq.auto_smooth",
    "quark.torch.algorithm.osscar.osscar",
    "quark.torch.algorithm.blockwise_tuning.blockwise_tuning",
    "quark.torch.algorithm.blockwise_joint_tuning.processor",
    "quark.torch.algorithm.depth_pruning.layer_importance",
]


def _names_cpu(node: ast.AST) -> bool:
    """True if this argument expression can evaluate to CPU.

    Includes the conditional form ``CPU if force_layer_back_to_cpu else cur_layer_device``, which
    an equality-only check misses -- and which is how blockwise_tuning and osscar return a layer
    to CPU at the end of each iteration.
    """
    if isinstance(node, ast.Constant) and node.value == "cpu":
        return True
    if isinstance(node, ast.Name) and node.id == "CPU":
        return True
    if isinstance(node, ast.IfExp):
        return _names_cpu(node.body) or _names_cpu(node.orelse)
    return False


def _is_cpu_offload(node: ast.AST) -> bool:
    """True for ``<expr>.to(<cpu>)`` and ``move_to_device(<expr>, <cpu>)``.

    Deliberately not ``.cpu()``: that idiom is used here for *tensors* -- cached activations,
    cloned weights, returned scales -- and moving a tensor to host memory is not the bug. The bug
    is relocating a module whose placement accelerate is tracking.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "to":
        return any(_names_cpu(a) for a in node.args) or any(_names_cpu(k.value) for k in node.keywords)
    if isinstance(func, ast.Name) and func.id == "move_to_device":
        return len(node.args) > 1 and _names_cpu(node.args[1])
    return False


def _guards_on_accelerate(test: ast.AST) -> bool:
    return any(isinstance(n, ast.Attribute) and n.attr == "using_accelerate" for n in ast.walk(test))


@pytest.mark.parametrize("module_name", _OFFLOADING_MODULES)
def test_cpu_offloads_are_guarded_on_using_accelerate(module_name: str) -> None:
    """No processor may unconditionally push decoder blocks to CPU."""
    module = __import__(module_name, fromlist=["_"])
    tree = ast.parse(inspect.getsource(module))

    # Span of the guarded body only -- an `else:` branch is deliberately excluded.
    guarded_spans: list[tuple[int, int]] = [
        (node.body[0].lineno, node.body[-1].end_lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and _guards_on_accelerate(node.test)
    ]

    # Two kinds of annotated site are skipped, both stating their reason at the source line, and
    # keyed on the comment rather than a line-number allowlist that rots on the first edit above it:
    #   accelerate-exempt: the guard does not apply (the layer is not this model's, or is about to
    #                      be deleted, so there is no dispatch state to desync).
    #   accelerate-todo:   a real unguarded offload, deliberately left for a follow-up because
    #                      covering it needs the full per-layer body.
    def _annotated(line: str) -> bool:
        return "accelerate-exempt:" in line or "accelerate-todo:" in line

    source_lines = inspect.getsource(module).splitlines()
    exempt = {i + offset for i, line in enumerate(source_lines) if _annotated(line) for offset in range(2, 7)}

    unguarded = [
        node.lineno
        for node in ast.walk(tree)
        if _is_cpu_offload(node)
        and node.lineno not in exempt
        and not any(lo <= node.lineno <= hi for lo, hi in guarded_spans)
    ]
    assert not unguarded, (
        f"{module_name} offloads to CPU without a using_accelerate guard at line(s) {unguarded}; "
        "this desyncs accelerate's device_map dispatch under --multi_gpu auto"
    )


# ---------------------------------------------------------------------------------------------
# The remaining offloading processors.
#
# The AST test above proves each guard is *present*; these prove it *behaves*, by executing the
# offload branch with and without accelerate and asserting which relocations were requested.
# Nothing is really relocated -- the layers are mocks that record .to(...) -- so these do not
# exercise an accelerate hook, only the decision to ask for a move.
#
# The heavy per-layer body is skipped by patching the module-level tqdm to yield nothing, which
# leaves the bulk offload above it running against a non-empty module list.
# ---------------------------------------------------------------------------------------------


class _StopAfterOffload(Exception):
    """Raised from the first call made after the offload under test, to skip the heavy body.

    A distinct type, not a bare Exception: pytest.raises must not be satisfied by an unrelated
    production error that happened to fire before the offload ran.
    """


def _stop(*args: object, **kwargs: object) -> None:
    raise _StopAfterOffload


@pytest.mark.parametrize("using_accelerate", [True, False])
def test_blockwise_tuning_apply_cpu_offload_respects_accelerate(
    monkeypatch: pytest.MonkeyPatch, using_accelerate: bool
) -> None:
    from quark.torch.algorithm.blockwise_tuning import blockwise_tuning as blockwise_module
    from quark.torch.algorithm.blockwise_tuning.blockwise_tuning import BlockwiseTuningProcessor

    layer = _fake_layer()
    monkeypatch.setattr(blockwise_module, "clear_memory", lambda: None)
    monkeypatch.setattr(blockwise_module, "get_device", _stop)

    processor = object.__new__(BlockwiseTuningProcessor)
    processor.model = MagicMock()
    processor.fp_model = MagicMock()
    processor.modules = [layer]
    processor.modules_fp = [layer]
    processor.inps = []
    processor.using_accelerate = using_accelerate

    with pytest.raises(_StopAfterOffload):
        processor.apply()

    assert layer.to.call_args_list == ([] if using_accelerate else [call("cpu")])


@pytest.mark.parametrize("using_accelerate", [True, False])
def test_osscar_apply_cpu_offload_respects_accelerate(monkeypatch: pytest.MonkeyPatch, using_accelerate: bool) -> None:
    from quark.torch.algorithm.osscar import osscar as osscar_module
    from quark.torch.algorithm.osscar.osscar import OsscarProcessor

    layer = _fake_layer()
    monkeypatch.setattr(osscar_module, "clear_memory", lambda: None)
    monkeypatch.setattr(osscar_module, "get_device", _stop)

    processor = object.__new__(OsscarProcessor)
    processor.model = MagicMock()
    processor.modules = [layer]
    processor.inps = []
    processor.using_accelerate = using_accelerate

    with pytest.raises(_StopAfterOffload):
        processor.apply()

    assert layer.to.call_args_list == ([] if using_accelerate else [call("cpu")])


@pytest.mark.parametrize("has_hook", [True, False])
def test_blockwise_tuning_init_tracks_hf_hook(monkeypatch: pytest.MonkeyPatch, has_hook: bool) -> None:
    from quark.torch.algorithm.blockwise_tuning import blockwise_tuning as blockwise_module
    from quark.torch.algorithm.blockwise_tuning.blockwise_tuning import BlockwiseTuningProcessor

    monkeypatch.setattr(blockwise_module, "init_device_map", lambda model: {})
    monkeypatch.setattr(blockwise_module, "init_blockwise_algo", lambda model, path, loader: ([], {}, []))
    monkeypatch.setattr(blockwise_module, "get_model_layers", lambda model, path: [])

    model = nn.Module()
    if has_hook:
        model._hf_hook = MagicMock()

    processor = BlockwiseTuningProcessor(nn.Module(), model, MagicMock(), data_loader=MagicMock())

    assert processor.using_accelerate is has_hook


@pytest.mark.parametrize("has_hook", [True, False])
def test_osscar_init_tracks_hf_hook(monkeypatch: pytest.MonkeyPatch, has_hook: bool) -> None:
    from quark.torch.algorithm.osscar import osscar as osscar_module
    from quark.torch.algorithm.osscar.osscar import OsscarProcessor

    monkeypatch.setattr(osscar_module, "init_device_map", lambda model: {})
    monkeypatch.setattr(osscar_module, "init_blockwise_algo", lambda model, path, loader: ([], {}, []))

    model = nn.Module()
    if has_hook:
        model._hf_hook = MagicMock()

    processor = OsscarProcessor(model, MagicMock(), data_loader=MagicMock())

    assert processor.using_accelerate is has_hook


def _layer_importance_config() -> MagicMock:
    config = MagicMock()
    config.model_decoder_layers = "layers"
    config.layer_norm_field = "norm"
    config.layer_num_field = "num_hidden_layers"
    config.delete_layers_index = []
    config.delete_layer_num = 0
    config.save_gpu_memory = True
    return config


@pytest.mark.parametrize("has_hook", [True, False])
def test_layer_importance_init_cpu_offload_respects_accelerate(monkeypatch: pytest.MonkeyPatch, has_hook: bool) -> None:
    """With save_gpu_memory the pruner parks every decoder layer on CPU -- unless accelerate placed them."""
    from quark.torch.algorithm.depth_pruning import layer_importance as pruning_module
    from quark.torch.algorithm.depth_pruning.layer_importance import LayerImportancePrunerProcessor

    layer = _fake_layer()
    monkeypatch.setattr(pruning_module, "init_device_map", lambda model: {})
    monkeypatch.setattr(pruning_module, "get_model_layers", lambda model, path: [layer])
    monkeypatch.setattr(pruning_module, "init_blockwise_algo", lambda model, path, loader: ([], {}, []))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    model = nn.Module()
    model.config = MagicMock()
    model.config.num_hidden_layers = 1
    if has_hook:
        model._hf_hook = MagicMock()

    processor = LayerImportancePrunerProcessor(model, _layer_importance_config(), data_loader=[torch.zeros(4)])

    assert processor.using_accelerate is has_hook
    assert layer.to.call_args_list == ([] if has_hook else [call("cpu")])


@pytest.mark.parametrize("using_accelerate", [True, False])
def test_layer_importance_slow_eval_cpu_offload_respects_accelerate(
    monkeypatch: pytest.MonkeyPatch, using_accelerate: bool
) -> None:
    """The layer-by-layer PPL pass sends each layer back to CPU once it is done with it."""
    from quark.torch.algorithm.depth_pruning import layer_importance as pruning_module
    from quark.torch.algorithm.depth_pruning.layer_importance import LayerImportancePrunerProcessor

    layer = _fake_layer()

    def fake_get_model_layers(model: object, path: str) -> object:
        # Path-aware rather than call-ordered: the decoder lookup yields our layer, and the
        # post-loop norm lookup is where we stop. An order-based fake would break the moment
        # anything else looked a layer up, even with the offload still correct.
        if path == "layers":
            return [layer]
        raise _StopAfterOffload

    monkeypatch.setattr(pruning_module, "get_model_layers", fake_get_model_layers)
    moved: list[object] = []
    monkeypatch.setattr(pruning_module, "move_to_device", lambda obj, device: moved.append((obj, device)) or obj)
    monkeypatch.setattr(pruning_module, "resolve_per_layer_kwargs", lambda layer, kwargs: {})
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    processor = object.__new__(LayerImportancePrunerProcessor)
    processor.model = MagicMock()
    processor.model_decoder_layers = "layers"
    processor.layer_norm_field = "norm"
    processor.device_map = {}
    processor.module_kwargs = {}
    processor.layer_inputs = []
    processor.using_accelerate = using_accelerate

    with pytest.raises(_StopAfterOffload):
        processor._slow_eval_model(MagicMock(), remain_layer_idx=[0])

    offloaded = [obj for obj, device in moved if device is pruning_module.CPU]
    assert offloaded == ([] if using_accelerate else [layer])


@pytest.mark.parametrize("has_hook", [True, False])
def test_blockwise_joint_tuning_init_cpu_offload_respects_accelerate(
    monkeypatch: pytest.MonkeyPatch, has_hook: bool
) -> None:
    """This processor does its bulk offload in __init__, and parks three lists, not one.

    Covered here rather than left to the GPU-gated integration test, so the guard stays verified
    on a CPU-only CI runner.
    """
    from quark.torch.algorithm.blockwise_joint_tuning import processor as joint_module
    from quark.torch.algorithm.blockwise_joint_tuning.processor import BlockwiseJointTuningProcessor

    quant_layer, fp_layer, fp_val_layer = _fake_layer(), _fake_layer(), _fake_layer()
    monkeypatch.setattr(joint_module, "init_device_map", lambda model: {})
    monkeypatch.setattr(joint_module, "get_model_layers", lambda model, path: [fp_layer])
    monkeypatch.setattr(joint_module, "clear_memory", lambda: None)
    # The quantized model's blocks come from the first call, the fp val blocks from the second.
    blockwise_results = iter([([quant_layer], {}, []), ([fp_val_layer], {}, [])])
    monkeypatch.setattr(joint_module, "init_blockwise_algo", lambda *args: next(blockwise_results))

    model = nn.Module()
    if has_hook:
        model._hf_hook = MagicMock()

    processor = BlockwiseJointTuningProcessor(nn.Module(), model, MagicMock(), data_loader=MagicMock())

    assert processor.using_accelerate is has_hook
    expected = [] if has_hook else [call("cpu")]
    assert quant_layer.to.call_args_list == expected
    assert fp_layer.to.call_args_list == expected
    assert fp_val_layer.to.call_args_list == expected


@pytest.mark.parametrize("using_accelerate", [True, False])
def test_awq_apply_cpu_offload_respects_accelerate(monkeypatch: pytest.MonkeyPatch, using_accelerate: bool) -> None:
    """AWQ's three relocations, the same shape as the AutoSmoothQuant case above.

    Covered on CPU rather than left to the GPU-gated AWQ tests, so the guard stays verified on a
    runner without an accelerator. Note the device_map lookup at the top of the loop must happen
    either way -- `common_device` is used further down -- so only the *move* is gated.
    """
    from quark.torch.algorithm.awq import awq as awq_module
    from quark.torch.algorithm.awq.awq import AwqProcessor

    layer = _fake_layer()
    monkeypatch.setattr(awq_module, "clear_memory", lambda: None)
    monkeypatch.setattr(awq_module, "get_named_quant_linears", lambda module: {})
    monkeypatch.setattr(awq_module, "get_moe_layers", lambda module: {})
    monkeypatch.setattr(awq_module, "get_layers_for_scaling", lambda *args, **kwargs: [])
    monkeypatch.setattr(awq_module, "apply_scale", lambda *args, **kwargs: None)
    monkeypatch.setattr(awq_module, "apply_clip", lambda *args, **kwargs: None)
    monkeypatch.setattr(awq_module, "append_str_prefix", lambda scales, prefix: scales)
    monkeypatch.setattr(awq_module, "get_op_name", lambda model, module: "layer")
    monkeypatch.setattr(AwqProcessor, "_get_input_feat", lambda self, layer, named: {})
    monkeypatch.setattr(AwqProcessor, "_search_best_clip", lambda self, named, feat: [])
    monkeypatch.setattr(AwqProcessor, "_apply_quant", lambda self, named: None)

    processor = object.__new__(AwqProcessor)
    processor.model = MagicMock()
    processor.modules = [layer]
    # Non-CPU so the layer's own placement stays distinguishable from an offload.
    processor.device_map = {"layers.0": "cuda:0"}
    processor.model_decoder_layers = "layers"
    processor.module_kwargs = {}
    processor.scaling_layers = []
    processor.num_attention_heads = 8
    processor.num_key_value_heads = 8
    processor.global_scales_list = []
    processor.recover_attn_implementation = "eager"
    processor.using_accelerate = using_accelerate

    processor.apply()

    moves = layer.to.call_args_list
    if using_accelerate:
        assert moves == [], f"accelerate manages placement; apply() must not relocate: {moves}"
    else:
        assert moves == [call("cpu"), call("cuda:0"), call("cpu")]
