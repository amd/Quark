#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Oracle test for blockwise calibration capture/replay fidelity.
Blockwise algorithms (GPTQ/AWQ/blockwise tuning/...) don't run the real model during quantization;
they capture decoder-layer inputs once (``cache_model_inps``) and *replay* each layer in isolation
(``resolve_per_layer_kwargs`` selecting that layer's forward kwargs). For replay to be correct, the
output of every replayed layer must equal what that layer produced in a real forward pass.
That real forward is the ground truth -- it is correct by construction, whatever masks/rotary/kwargs
the model decided each layer needs. So instead of asserting anything about *which* kwargs a layer
should receive (attention masks, RoPE tables, ...), we record each layer's output during a genuine
``model(input)`` and assert the replay reproduces it bit-for-bit. Any per-layer kwarg the capture
mishandles -- today's known ones or any added by transformers later -- corrupts the replayed output
and fails the test automatically.
The bug this guards against: ``cache_model_inps`` used to record only layer 0's kwargs and replay them
through every layer. For models that interleave ``sliding_attention``/``full_attention`` (Gemma2 differ
by mask, Gemma3 also by local/global RoPE) that hands later layers the wrong mask/rotary.
"""

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    Gemma2Config,
    Gemma2ForCausalLM,
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
)

from quark.common.utils.testing_utils import require_torch_multi_gpu, skip_if_no_gpu, slow_test, torch_device
from quark.torch import ModelQuantizer
from quark.torch.algorithm.depth_pruning.layer_importance import LayerImportancePrunerProcessor
from quark.torch.algorithm.utils.module import get_layer_idx, resolve_per_layer_kwargs
from quark.torch.algorithm.utils.prepare import cache_model_inps
from quark.torch.pruning.config import LayerImportancePruneConfig
from quark.torch.quantization.config.config import (
    AutoSmoothQuantConfig,
    AWQConfig,
    GPTAQConfig,
    GPTQConfig,
    QConfig,
    QLayerConfig,
    QronosConfig,
    QTensorConfig,
    SmoothQuantConfig,
)
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.observer.observer import PlaceholderObserver

SEQLEN = 16
SLIDING_WINDOW = 4  # < SEQLEN, so sliding and full causal masks genuinely diverge

# Small configs shared by the mixed-attention families. Alternating layer_types so both attention
# variants appear; sliding_window < SEQLEN so the masks (and, for Gemma3, the rotary tables) differ.
_LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
_COMMON = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=64,
)


def _build_gemma2() -> Gemma2ForCausalLM:
    """Build a Gemma2 model with mixed attention layers for testing.
    Returns:
        Gemma2ForCausalLM: A Gemma2 model configured with alternating sliding and full attention layers.
    """
    cfg = Gemma2Config(sliding_window=SLIDING_WINDOW, layer_types=_LAYER_TYPES, query_pre_attn_scalar=8, **_COMMON)
    return Gemma2ForCausalLM(cfg)


def _build_gemma3() -> Gemma3ForCausalLM:
    """Build a small Gemma3ForCausalLM model for testing.
    Creates a Gemma3 model with mixed attention types (sliding and full) for testing
    per-layer kwargs handling in blockwise calibration.
    Returns:
        Gemma3ForCausalLM: A configured Gemma3 causal language model.
    """
    cfg = Gemma3TextConfig(sliding_window=SLIDING_WINDOW, layer_types=_LAYER_TYPES, query_pre_attn_scalar=8, **_COMMON)
    return Gemma3ForCausalLM(cfg)


def _build_llama() -> LlamaForCausalLM:
    """Build a LlamaForCausalLM model with common test configuration.
    Returns:
        LlamaForCausalLM: A Llama model configured with shared test parameters.
    """
    return LlamaForCausalLM(LlamaConfig(**_COMMON))


# (builder, mixed) -- mixed models need per-layer kwargs; uniform models share one dict.
MODELS: list[tuple[str, Callable[[], torch.nn.Module], bool]] = [
    ("gemma3_mixed", _build_gemma3, True),  # rotary (local vs global RoPE) diverges per layer type
    ("gemma2_mixed", _build_gemma2, True),  # attention_mask (sliding vs full) diverges per layer type
    ("llama_uniform", _build_llama, False),  # uniform attention -- one shared kwargs dict
]


def _decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Extract the decoder layers from a transformer model.
    Args:
        model: A transformer model containing decoder layers.
    Returns:
        ModuleList containing the model's decoder layers.
    """
    return model.model.layers


def _hook_layer_outputs(model: torch.nn.Module, store: dict[int, torch.Tensor]) -> list[Any]:
    """Register forward hooks recording each decoder layer's output into ``store`` (last write wins).
    Returns the handles for the caller to remove. Outputs are cast to float so bf16 and fp32 models
    compare in the same dtype."""

    def make_hook(idx: int) -> Callable:
        def hook(_module, _inp, out):
            store[idx] = (out[0] if isinstance(out, tuple) else out).detach().float().clone()

        return hook

    return [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(_decoder_layers(model))]


def _real_forward_outputs(model: torch.nn.Module, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """Ground truth: each decoder layer's output during a genuine full forward."""
    outputs: dict[int, torch.Tensor] = {}
    handles = _hook_layer_outputs(model, outputs)
    with torch.no_grad():
        model(input_ids, use_cache=False)
    for h in handles:
        h.remove()
    return [outputs[i] for i in range(len(outputs))]


def _replay_outputs(
    model: torch.nn.Module,
    layer_kwargs: dict,
    inps: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Replay each decoder layer on the captured inputs, chaining output -> next input."""
    hidden = inps[0]
    results: list[torch.Tensor] = []
    with torch.no_grad():
        for layer in _decoder_layers(model):
            out = layer(hidden, **resolve_per_layer_kwargs(layer, layer_kwargs))
            hidden = out[0] if isinstance(out, tuple) else out
            results.append(hidden)
    return results


@pytest.mark.parametrize("name,builder,mixed", MODELS, ids=[m[0] for m in MODELS])
def test_replay_matches_real_forward(name: str, builder: Callable[[], torch.nn.Module], mixed: bool) -> None:
    """Test that blockwise calibration capture and replay produces identical outputs to a real forward pass.
    This test verifies the correctness of the capture/replay mechanism used by blockwise quantization
    algorithms. It captures decoder layer inputs once during a real forward pass and replays each layer
    in isolation, then compares the replayed outputs against the ground truth outputs from the real forward.
    Args:
        name: Name identifier for the model being tested.
        builder: Callable that constructs and returns the model instance.
        mixed: Boolean indicating whether the model uses mixed attention types requiring per-layer kwargs.
    Returns:
        None. Raises AssertionError if replay outputs diverge from real forward outputs.
    """
    torch.manual_seed(0)
    model = builder().eval().to(torch_device)
    input_ids = torch.randint(0, _COMMON["vocab_size"], (1, SEQLEN)).to(torch_device)

    oracle = _real_forward_outputs(model, input_ids)

    modules, layer_kwargs, inps = cache_model_inps(model, _decoder_layers(model), DataLoader(input_ids))
    assert len(inps) == 1, "expected one captured calibration input"

    assert ("_per_layer_kwargs" in layer_kwargs) == mixed

    replayed = _replay_outputs(model, layer_kwargs, inps)
    for i, (got, want) in enumerate(zip(replayed, oracle, strict=True)):
        torch.testing.assert_close(got, want, msg=f"{name} layer {i}: replay diverged from real forward")


# ---------------------------------------------------------------------------------------------------
# Consumer (end-to-end) test
#
# The oracle test above proves the capture/replay *mechanism* is correct in isolation. This second test
# proves the quantization *algorithms* actually route through it -- a call site that forgot to call
# resolve_per_layer_kwargs (or passed layer 0's kwargs) would pass the oracle test but be caught here.
#
# Like the oracle, this is output-based: it records each decoder layer's output during a real forward
# (ground truth) and again during the algorithm's replay, and asserts they match -- naming no kwarg. The
# obstacle is that real quantization perturbs weights as it walks the stack, so a replayed layer's input
# carries quantization noise from earlier layers, swamping any kwargs error. We dodge that with a
# *lossless* config: ``Dtype.bfloat16`` is a cast-only "quant" (NonScaledFakeQuantize + PlaceholderObserver),
# so on a bf16 model the round-trip is the identity -- zero quant error, zero contamination -- and any
# per-layer-output divergence is purely a wrong-kwargs bug.
# ---------------------------------------------------------------------------------------------------

_INSIDE_LAYER_MODULES = [
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.q_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]
_SCALING_LAYERS = [
    {
        "prev_op": "input_layernorm",
        "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "inp": "self_attn.q_proj",
        "module2inspect": "self_attn",
    },
]
_DECODER_LAYERS_PATH = "model.layers"

_W_BF16_LOSSLESS = QTensorConfig(dtype=Dtype.bfloat16, observer_cls=PlaceholderObserver, ch_axis=0, group_size=0)


def _make_algo_config(algo: str) -> Any:
    """Build one algorithm's config. GPTQ-family takes inside_layer_modules (block_forward replay);
    AWQ auto-derives scale pairs (empty scaling_layers); SmoothQuant takes explicit pairs."""
    if algo == "gptq":
        return GPTQConfig(inside_layer_modules=_INSIDE_LAYER_MODULES, model_decoder_layers=_DECODER_LAYERS_PATH)
    if algo == "gptaq":
        return GPTAQConfig(inside_layer_modules=_INSIDE_LAYER_MODULES, model_decoder_layers=_DECODER_LAYERS_PATH)
    if algo == "qronos":
        return QronosConfig(inside_layer_modules=_INSIDE_LAYER_MODULES, model_decoder_layers=_DECODER_LAYERS_PATH)
    if algo == "awq":
        return AWQConfig(scaling_layers=[], model_decoder_layers=_DECODER_LAYERS_PATH)
    if algo == "smoothquant":
        return SmoothQuantConfig(scaling_layers=_SCALING_LAYERS, model_decoder_layers=_DECODER_LAYERS_PATH)
    if algo == "autosmoothquant":
        return AutoSmoothQuantConfig(scaling_layers=_SCALING_LAYERS, model_decoder_layers=_DECODER_LAYERS_PATH)
    raise ValueError(algo)


ALGORITHMS = ["gptq", "gptaq", "qronos", "awq", "smoothquant", "autosmoothquant"]


@slow_test
@pytest.mark.parametrize("algo", ALGORITHMS)
@pytest.mark.parametrize("name,builder,mixed", MODELS, ids=[m[0] for m in MODELS])
def test_algorithm_replay_matches_real_forward(
    name: str, builder: Callable[[], torch.nn.Module], mixed: bool, algo: str
) -> None:
    torch.manual_seed(0)
    model = builder().eval().to(torch_device).to(torch.bfloat16)
    input_ids = torch.randint(0, _COMMON["vocab_size"], (1, SEQLEN)).to(torch_device)

    oracle = _real_forward_outputs(model, input_ids)

    seen: dict[int, torch.Tensor] = {}
    handles = _hook_layer_outputs(model, seen)
    quant_config = QConfig(
        global_quant_config=QLayerConfig(weight=_W_BF16_LOSSLESS), algo_config=[_make_algo_config(algo)]
    )
    ModelQuantizer(quant_config).quantize_model(model, DataLoader(input_ids))
    for h in handles:
        h.remove()

    for i, want in enumerate(oracle):
        assert i in seen, f"{name}/{algo}: layer {i} never ran during quantization"
        torch.testing.assert_close(
            seen[i], want.float(), msg=f"{name}/{algo} layer {i}: replay diverged from real forward"
        )


def test_get_layer_idx_raises_without_layer_idx() -> None:
    """A module (and its submodules) with no ``layer_idx`` must raise, not silently mis-resolve kwargs."""

    class NoIdx(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    with pytest.raises(AttributeError, match="Could not find layer_idx"):
        get_layer_idx(NoIdx())


@skip_if_no_gpu
def test_cache_model_inps_pulls_offloaded_layers() -> None:
    """Offloaded layers must be pulled on-device just before they run and sent back afterwards.
    Mixed models run a full forward to record every layer's kwargs; when later layers live on CPU
    (device_map offload) the pre-hook moves each onto the hidden-states device and the post-hook restores
    it. Placing layers 1+ on CPU while the model runs on GPU exercises both branches.
    """
    torch.manual_seed(0)
    model = _build_gemma3().eval().to(torch_device)
    layers = _decoder_layers(model)
    for i in range(1, len(layers)):
        layers[i].to("cpu")
    before = [next(layer.parameters()).device for layer in layers]

    input_ids = torch.randint(0, _COMMON["vocab_size"], (1, SEQLEN)).to(torch_device)
    _, layer_kwargs, _ = cache_model_inps(model, layers, DataLoader(input_ids))

    assert sorted(layer_kwargs["_per_layer_kwargs"].keys()) == list(range(len(layers)))
    after = [next(layer.parameters()).device for layer in layers]
    assert after == before, "offloaded layers were not restored to their original device"


@skip_if_no_gpu
def test_autosmoothquant_keeps_accelerate_managed_layers_resident() -> None:
    """ASQ must not park accelerate-managed layers on CPU.

    ``init_quant`` parks layers 1+ expecting the pre-hook above to pull them back, but it skips
    accelerate-managed modules, stranding them. Only mixed models notice, since uniform ones
    early-exit at the never-parked layer 0. The hook reproduces multi-GPU on one device.
    """
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    torch.manual_seed(0)
    model = _build_gemma3().eval().to(torch_device).to(torch.bfloat16)
    for layer in _decoder_layers(model):
        add_hook_to_module(layer, AlignDevicesHook(execution_device=torch.device(torch_device)))
    # `dispatch_model` also hooks the root (its `io_same_device` hook), and that is what
    # `using_accelerate` keys on -- without it this fixture is not a dispatched model.
    add_hook_to_module(model, AlignDevicesHook(io_same_device=True))

    input_ids = torch.randint(0, _COMMON["vocab_size"], (1, SEQLEN)).to(torch_device)
    quant_config = QConfig(
        global_quant_config=QLayerConfig(weight=_W_BF16_LOSSLESS),
        algo_config=[_make_algo_config("autosmoothquant")],
    )
    ModelQuantizer(quant_config).quantize_model(model, DataLoader(input_ids))

    for i, layer in enumerate(_decoder_layers(model)):
        assert next(layer.parameters()).device.type == torch.device(torch_device).type, (
            f"layer {i} was left off-device; accelerate-managed layers must not be parked"
        )


@slow_test
@require_torch_multi_gpu
def test_autosmoothquant_survives_real_accelerate_dispatch(tmp_path) -> None:
    """Ground truth for the test above, using a genuine ``device_map="auto"`` dispatch.

    The hook-injection version runs anywhere but only approximates accelerate. This one is the
    real thing -- and it needs two visible GPUs, because accelerate attaches no hook when it can
    fit the model on one device, which is precisely why single-GPU CI never caught the bug.
    """
    torch.manual_seed(0)
    _build_gemma3().save_pretrained(tmp_path)
    # A tiny budget on GPU 0 forces a real split rather than a single-device placement.
    model = AutoModelForCausalLM.from_pretrained(
        tmp_path, device_map="auto", max_memory={0: "1MiB", 1: "600MiB"}, dtype=torch.bfloat16
    ).eval()

    layers = _decoder_layers(model)
    assert any(hasattr(layer, "_hf_hook") for layer in layers), "accelerate did not dispatch; test is vacuous"
    assert len({str(next(layer.parameters()).device) for layer in layers}) > 1, "model was not split across devices"

    input_ids = torch.randint(0, _COMMON["vocab_size"], (1, SEQLEN))
    quant_config = QConfig(
        global_quant_config=QLayerConfig(weight=_W_BF16_LOSSLESS),
        algo_config=[_make_algo_config("autosmoothquant")],
    )
    # Pre-fix this raises "Expected all tensors to be on the same device, cuda:0 and cpu".
    ModelQuantizer(quant_config).quantize_model(model, DataLoader(input_ids))


# ---------------------------------------------------------------------------------------------------
# Depth-pruning consumer test
#
# LayerImportancePrunerProcessor has two PPL evaluators. `_fast_eval_model` runs a genuine
# `model(batch)` forward (ground truth). `_slow_eval_model` (save_gpu_memory=True) instead replays the
# decoder stack layer-by-layer using the `module_kwargs` captured by `init_blockwise_algo`. On a
# uniform model the two agree (the smoke test cross-checks this on OPT); on a mixed-attention model the
# slow path must select each layer's own kwargs or it feeds `full_attention` layers layer 0's sliding
# mask / local RoPE. This test drives both evaluators on the mixed Gemma families (with uniform Llama as
# a control) and asserts they produce the same perplexity -- naming no kwarg, exactly like the oracle.
# ---------------------------------------------------------------------------------------------------

_LAYER_NORM_FIELD = "model.norm"


def _make_pruner(model: torch.nn.Module, save_gpu_memory: bool, test_dataset: list[torch.Tensor]):
    cfg = LayerImportancePruneConfig(
        delete_layer_num=1,
        model_decoder_layers=_DECODER_LAYERS_PATH,
        layer_norm_field=_LAYER_NORM_FIELD,
        layer_num_field="num_hidden_layers",
        save_gpu_memory=save_gpu_memory,
    )
    return LayerImportancePrunerProcessor(model, cfg, test_dataset)


@slow_test
@skip_if_no_gpu
@pytest.mark.parametrize("name,builder,mixed", MODELS, ids=[m[0] for m in MODELS])
def test_depth_prune_slow_eval_matches_fast_eval(
    name: str, builder: Callable[[], torch.nn.Module], mixed: bool
) -> None:
    """Slow (layer-replay) PPL must equal fast (real-forward) PPL for the depth-pruning evaluator.
    For mixed-attention models this fails unless the slow path resolves each layer's own kwargs.
    """
    torch.manual_seed(0)
    model = builder().eval().to(torch_device)
    num_layers = _COMMON["num_hidden_layers"]
    remain = list(range(num_layers))
    input_ids = torch.randint(0, _COMMON["vocab_size"], (1, SEQLEN)).to(torch_device)
    test_dataset = [input_ids]

    fast_ppl = _make_pruner(model, save_gpu_memory=False, test_dataset=test_dataset).eval_func(
        model, remain_layer_idx=remain
    )
    slow_ppl = _make_pruner(model, save_gpu_memory=True, test_dataset=test_dataset).eval_func(
        model, remain_layer_idx=remain
    )

    torch.testing.assert_close(
        slow_ppl,
        fast_ppl,
        rtol=1e-3,
        atol=1e-3,
        msg=f"{name}: slow-eval PPL {slow_ppl.item()} != fast-eval PPL {fast_ppl.item()}",
    )


if __name__ == "__main__":
    for _name, _builder, _mixed in MODELS:
        test_replay_matches_real_forward(_name, _builder, _mixed)
        print(f"PASS oracle: {_name}")
