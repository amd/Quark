#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import contextlib
import pathlib
import sys

import pytest
import torch
from accelerate.hooks import AlignDevicesHook
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, Llama4Config, Llama4ForConditionalGeneration
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from quark.common.utils.log import ScreenLogger
from quark.common.utils.testing_utils import (
    FROM_PRETRAINED_KWARGS,
    set_environment_variables,
    skip_if_no_gpu,
    slow_test,
    torch_device,
)
from quark.torch import ModelQuantizer
from quark.torch.quantization import (
    FP8E4M3PerTensorSpec,
    OCP_MXFP4Spec,
    Uint4PerChannelSpec,
    load_pre_optimization_config_from_file,
    load_quant_algo_config_from_file,
)
from quark.torch.quantization.config.config import GPTAQConfig, QConfig, QLayerConfig, QronosConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.observer.observer import PlaceholderObserver
from quark.torch.utils.llm import preprocess_for_quantization

logger = ScreenLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def _setup_env():
    """Set environment variables for all MoE algorithm tests."""
    with set_environment_variables(QUARK_ALGO_DEBUG="1"):
        yield


FLOAT16_SPEC = QTensorConfig(dtype=Dtype.float16, observer_cls=PlaceholderObserver)
FLOAT16_CONFIG = QLayerConfig(input_tensors=FLOAT16_SPEC, weight=FLOAT16_SPEC)
FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(is_dynamic=False).to_quantization_spec()
W_FP8_A_FP8_PER_TENSOR_CONFIG = QLayerConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC)
W_MXFP4_A_DYN_MXFP4_CONFIG = QLayerConfig(
    input_tensors=OCP_MXFP4Spec(ch_axis=-1).to_quantization_spec(),
    weight=OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec(),
)
UINT4_PER_CHANNEL_ASYM_SPEC = Uint4PerChannelSpec(ch_axis=0, is_dynamic=False).to_quantization_spec()
W_UINT4_PER_CHANNEL_ASYM_CONFIG = QLayerConfig(weight=UINT4_PER_CHANNEL_ASYM_SPEC)
sys.path.append("..")


def get_dataloader(model_name="facebook/opt-125m", device=torch_device):
    seq_length = 4
    tokenized_outputs = {}
    torch.manual_seed(42)
    tokenized_outputs["input_ids"] = torch.randint(10, (1, seq_length), device=device)
    tokenized_outputs["attention_mask"] = torch.ones((1, seq_length), dtype=torch.int64, device=device)
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"].to(device))
    return calib_dataloader


def test_moe_gptq():
    # dataset
    config_path = "./test/test_for_torch/configs/moe_model/mixtral"
    dataloader = get_dataloader(config_path, torch_device)

    # original results
    config = AutoConfig.from_pretrained(config_path + "/config.json", trust_remote_code=True, **FROM_PRETRAINED_KWARGS)
    model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16).to(torch_device).eval()
    logits_original = model(dataloader.dataset).logits

    # algorithm config
    algo_config = load_quant_algo_config_from_file(config_path + "/gptq_config.json")
    quant_config = QConfig(
        global_quant_config=W_UINT4_PER_CHANNEL_ASYM_CONFIG, algo_config=[algo_config], exclude=["lm_head", "*gate"]
    )

    # apply algorithm
    quantizer = ModelQuantizer(quant_config)
    model = quantizer._prepare_model(model)
    model = quantizer._apply_advanced_quant_algo(model, dataloader)

    # check results
    logits_smooth = model(dataloader.dataset).logits
    assert (logits_original - logits_smooth).abs().max().item() < 5e-3
    logger.info("MoE GPTQ is checked valid!")


def test_moe_qronos():
    # dataset
    config_path = "./test/test_for_torch/configs/moe_model/mixtral"
    dataloader = get_dataloader(config_path, torch_device)

    # original results
    config = AutoConfig.from_pretrained(config_path + "/config.json", trust_remote_code=True, **FROM_PRETRAINED_KWARGS)
    model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16).to(torch_device).eval()
    logits_original = model(dataloader.dataset).logits

    # algorithm config
    algo_config = load_quant_algo_config_from_file(config_path + "/qronos_config.json")
    quant_config = QConfig(
        global_quant_config=W_UINT4_PER_CHANNEL_ASYM_CONFIG, algo_config=[algo_config], exclude=["lm_head", "*gate"]
    )

    # apply algorithm
    quantizer = ModelQuantizer(quant_config)
    model = quantizer._prepare_model(model)
    model = quantizer._apply_advanced_quant_algo(model, dataloader)

    # check results
    logits_smooth = model(dataloader.dataset).logits
    assert (logits_original - logits_smooth).abs().max().item() < 5e-3


def test_moe_gptaq():
    # dataset
    config_path = "./test/test_for_torch/configs/moe_model/mixtral"
    dataloader = get_dataloader(config_path, torch_device)

    # original results
    config = AutoConfig.from_pretrained(config_path + "/config.json", trust_remote_code=True, **FROM_PRETRAINED_KWARGS)
    model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16).to(torch_device).eval()
    logits_original = model(dataloader.dataset).logits

    # algorithm config
    algo_config = load_quant_algo_config_from_file(config_path + "/gptaq_config.json")
    quant_config = QConfig(
        global_quant_config=W_UINT4_PER_CHANNEL_ASYM_CONFIG, algo_config=[algo_config], exclude=["lm_head", "*gate"]
    )

    # apply algorithm
    quantizer = ModelQuantizer(quant_config)
    model = quantizer._prepare_model(model)
    model = quantizer._apply_advanced_quant_algo(model, dataloader)

    # check results
    logits_smooth = model(dataloader.dataset).logits
    assert (logits_original - logits_smooth).abs().max().item() < 5e-3


def test_moe_awq():
    # dataset
    config_path = "./test/test_for_torch/configs/moe_model/mixtral"
    dataloader = get_dataloader(config_path, torch_device)

    # original results
    config = AutoConfig.from_pretrained(config_path + "/config.json", trust_remote_code=True, **FROM_PRETRAINED_KWARGS)
    model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16).to(torch_device).eval()
    logits_original = model(dataloader.dataset).logits

    # algorithm config
    algo_config = load_quant_algo_config_from_file(config_path + "/awq_config.json")
    quant_config = QConfig(
        global_quant_config=W_UINT4_PER_CHANNEL_ASYM_CONFIG, algo_config=[algo_config], exclude=["lm_head", "*gate"]
    )

    # apply algorithm
    quantizer = ModelQuantizer(quant_config)
    model = quantizer._prepare_model(model)
    model = quantizer._apply_advanced_quant_algo(model, dataloader)

    # check results
    logits_smooth = model(dataloader.dataset).logits
    assert (logits_original - logits_smooth).abs().max().item() < 6.5e-3
    logger.info("MoE AWQ is checked valid!")


@pytest.mark.parametrize(
    "global_quant_config,model_config_name",
    [
        (W_FP8_A_FP8_PER_TENSOR_CONFIG, "config.json"),
        pytest.param(
            W_MXFP4_A_DYN_MXFP4_CONFIG,
            "config_128.json",
            marks=pytest.mark.skipif(
                not (isinstance(torch_device, torch.device) and torch_device.type == "cuda"),
                reason="MXFP4 (qdq_mxfp4) is only implemented for CUDA devices",
            ),
        ),
    ],
)
def test_moe_autosmoothquant(global_quant_config, model_config_name):
    # dataset
    config_path = "./test/test_for_torch/configs/moe_model/mixtral"
    dataloader = get_dataloader(config_path, torch_device)

    # original results
    config = AutoConfig.from_pretrained(
        config_path + "/" + model_config_name, trust_remote_code=True, **FROM_PRETRAINED_KWARGS
    )
    model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16).to(torch_device).eval()
    logits_original = model(dataloader.dataset).logits

    # algorithm config
    algo_config = load_quant_algo_config_from_file(config_path + "/autosmoothquant_config.json")
    quant_config = QConfig(
        global_quant_config=global_quant_config, algo_config=[algo_config], exclude=["lm_head", "*gate"]
    )

    # apply algorithm
    quantizer = ModelQuantizer(quant_config)
    model = quantizer._prepare_model(model)
    model = quantizer._apply_advanced_quant_algo(model, dataloader)

    # check results
    logits_smooth = model(dataloader.dataset).logits
    assert (logits_original - logits_smooth).abs().max().item() < 5e-3
    logger.info("MoE AutoSmoothQuant is checked valid!")


def test_moe_autosmoothquant_only_hooks_consumed_inputs():
    """AutoSmoothQuant must not capture activations it never reads.

    Hooking every linear copies each input to host once per calibration sample; on a MoE the
    expert linears dominate runtime while contributing nothing.
    """
    from quark.torch.algorithm.awq.auto_smooth import AutoSmoothQuantProcessor

    processor = object.__new__(AutoSmoothQuantProcessor)
    processor.scaling_layers = [
        {"prev_op": "input_layernorm", "layers": ["self_attn.q_proj"], "inp": "self_attn.q_proj"},
    ]
    named_input_layers = {
        "self_attn.q_proj": torch.nn.Linear(4, 4),
        "self_attn.k_proj": torch.nn.Linear(4, 4),
        **{f"block_sparse_moe.experts.{i}.w1": torch.nn.Linear(4, 4) for i in range(32)},
    }

    selected = processor._select_input_hook_targets(named_input_layers)

    assert set(selected) == {"self_attn.q_proj"}, "expert linears must not be hooked"

    # A config naming no ``inp`` must not silently disable capture altogether.
    processor.scaling_layers = [{"prev_op": "input_layernorm", "layers": ["self_attn.q_proj"]}]
    assert processor._select_input_hook_targets(named_input_layers) == named_input_layers

    # MoE configs name ``inp`` by suffix; ``get_layers_for_scaling`` resolves it with
    # ``fnmatch("*" + inp)``, so selection must keep those experts or smoothing silently stops.
    processor.scaling_layers = [
        {"prev_op": "input_layernorm", "layers": ["self_attn.q_proj"], "inp": "self_attn.q_proj"},
        {"prev_op": "w3", "layers": ["w2"], "inp": "w2"},
    ]
    moe_input_layers = {
        **named_input_layers,
        **{f"block_sparse_moe.experts.{i}.w2": torch.nn.Linear(4, 4) for i in range(32)},
        **{f"block_sparse_moe.experts.{i}.w3": torch.nn.Linear(4, 4) for i in range(32)},
    }

    selected = processor._select_input_hook_targets(moe_input_layers)

    assert set(selected) == {"self_attn.q_proj"} | {f"block_sparse_moe.experts.{i}.w2" for i in range(32)}, (
        "suffix-named MoE inp must be hooked, and only that expert linear"
    )


def _bare_autosmoothquant_processor(modules, device_map, using_accelerate=False):
    """An `AutoSmoothQuantProcessor` with only what device resolution reads; `__init__` calibrates."""
    from quark.torch.algorithm.awq.auto_smooth import AutoSmoothQuantProcessor

    processor = object.__new__(AutoSmoothQuantProcessor)
    processor.modules = modules
    processor.device_map = device_map
    processor.model_decoder_layers = "model.layers"
    processor.using_accelerate = using_accelerate
    return processor


def test_autosmoothquant_does_not_move_accelerate_managed_layers():
    """Under accelerate the device_map device is still reported, but the layer is not relocated:
    moving a module accelerate tracks desyncs its dispatch state."""
    layer = torch.nn.Linear(4, 4)
    processor = _bare_autosmoothquant_processor(
        [layer], {"model.layers.0": torch.device("cuda:0")}, using_accelerate=True
    )

    device, pull_on_device = processor._layer_device(0)

    assert device == torch.device("cuda:0"), "the device_map device must be used for the step"
    assert pull_on_device is False, "a managed layer must not be moved, nor parked back afterwards"
    assert next(layer.parameters()).device.type == "cpu", "accelerate's placement must be untouched"


def test_autosmoothquant_rejects_offloaded_layers():
    """Offloaded layers are `meta` between forwards, so smoothing them would be a silent no-op."""
    layer = torch.nn.Linear(4, 4, device="meta")
    layer._hf_hook = AlignDevicesHook(execution_device=torch.device("cpu"))
    processor = _bare_autosmoothquant_processor([layer], {"model.layers.0": torch.device("cpu")})

    with pytest.raises(NotImplementedError, match="offloaded"):
        processor._layer_device(0)


def test_autosmoothquant_pulls_unmanaged_layers_onto_their_mapped_device():
    """Without accelerate, `apply` parks layers on CPU and each step pulls its own layer back."""
    layer = torch.nn.Linear(4, 4)
    mapped_device = torch.device(torch_device)
    processor = _bare_autosmoothquant_processor([layer], {"model.layers.0": mapped_device})

    device, pull_on_device = processor._layer_device(0)

    assert device == mapped_device
    assert pull_on_device is True, "a layer the caller moves must be parked back on CPU after the step"
    # The query itself is pure: `_layer_on_device` owns the move.
    assert next(processor.modules[0].parameters()).device.type == "cpu"


@pytest.mark.parametrize("failing_step", [False, True], ids=["step_succeeds", "step_raises"])
def test_autosmoothquant_parks_pulled_layers_back_on_cpu(failing_step: bool):
    """`_layer_on_device` pulls an unmanaged layer for the step and always parks it back."""
    layer = torch.nn.Linear(4, 4)
    mapped_device = torch.device(torch_device)
    processor = _bare_autosmoothquant_processor([layer], {"model.layers.0": mapped_device})

    expect_failure = pytest.raises(ZeroDivisionError) if failing_step else contextlib.nullcontext()
    with expect_failure, processor._layer_on_device(0) as device:
        assert device == mapped_device
        assert next(processor.modules[0].parameters()).device.type == mapped_device.type, (
            "the layer must be on its mapped device for the duration of the step"
        )
        if failing_step:
            raise ZeroDivisionError

    assert next(processor.modules[0].parameters()).device.type == "cpu", "a pulled layer must be parked back"


def test_autosmoothquant_leaves_accelerate_managed_layers_where_they_are():
    """`_layer_on_device` must not move a managed layer, on entry or on exit."""
    layer = torch.nn.Linear(4, 4)
    processor = _bare_autosmoothquant_processor(
        [layer], {"model.layers.0": torch.device("cuda:0")}, using_accelerate=True
    )

    with processor._layer_on_device(0) as device:
        assert device == torch.device("cuda:0"), "the device_map device must be used for the step"
        assert next(processor.modules[0].parameters()).device.type == "cpu", "accelerate's placement stands"

    assert next(processor.modules[0].parameters()).device.type == "cpu"


def test_moe_smoothquant():
    # dataset
    config_path = "./test/test_for_torch/configs/moe_model/llama4"
    dataloader = get_dataloader(config_path, torch_device)

    # model
    config = Llama4Config.from_pretrained(config_path, **FROM_PRETRAINED_KWARGS)

    model = Llama4ForConditionalGeneration(config).to(torch_device).to(torch.bfloat16).eval()
    # init parameters defined using torch.empty
    gate_up_proj_init = model.language_model.model.layers[0].feed_forward.experts.gate_up_proj
    down_proj_init = model.language_model.model.layers[0].feed_forward.experts.down_proj
    gate_up_proj_init.data = torch.rand(
        gate_up_proj_init.shape, device=gate_up_proj_init.device, dtype=gate_up_proj_init.dtype
    )
    down_proj_init.data = torch.rand(down_proj_init.shape, device=down_proj_init.device, dtype=down_proj_init.dtype)
    logits_original = model(dataloader.dataset).logits

    # algorithm config
    algo_config = load_pre_optimization_config_from_file(config_path + "/smoothquant_config.json")
    quant_config = QConfig(global_quant_config=FLOAT16_CONFIG, algo_config=[algo_config], exclude=["lm_head", "*gate"])

    # algorithm
    quantizer = ModelQuantizer(quant_config)
    model = quantizer._apply_advanced_quant_algo(model, dataloader)

    # check results
    logits_smooth = model(dataloader.dataset).logits
    assert (logits_original - logits_smooth).abs().max().item() < 5e-3
    logger.info("MoE SmoothQuant is checked valid!")


OLMOE_MODEL_ID = "allenai/OLMoE-1B-7B-0924"
OLMOE_INSIDE_LAYER_MODULES = [
    "mlp.experts.*.gate_proj",
    "mlp.experts.*.up_proj",
    "mlp.experts.*.down_proj",
]


def _build_olmoe():
    torch.manual_seed(0)
    input_ids = torch.randint(1000, (1, 128), device=torch_device)
    dataloader = DataLoader(input_ids)

    model = AutoModelForCausalLM.from_pretrained(OLMOE_MODEL_ID, dtype=torch.float16, **FROM_PRETRAINED_KWARGS)
    model = model.to(torch_device).eval()
    # OLMoE ships fused experts (OlmoeExperts); split into per-expert linears so the algorithm can quantize them.
    preprocess_for_quantization(model)
    return model, dataloader


def _max_logit_err(model, dataloader, quant_config):
    with torch.no_grad():
        logits_original = model(dataloader.dataset).logits.float()
    quantizer = ModelQuantizer(quant_config)
    model = quantizer.quantize_model(model, dataloader)
    with torch.no_grad():
        logits_quant = model(dataloader.dataset).logits.float()
    return (logits_original - logits_quant).abs().max().item()


def _assert_algo_beats_rtn(algo_config):
    model, dataloader = _build_olmoe()
    rtn_config = QConfig(global_quant_config=W_UINT4_PER_CHANNEL_ASYM_CONFIG, exclude=["lm_head", "*gate"])
    err_rtn = _max_logit_err(model, dataloader, rtn_config)

    model, dataloader = _build_olmoe()
    algo_quant_config = QConfig(
        global_quant_config=W_UINT4_PER_CHANNEL_ASYM_CONFIG, algo_config=[algo_config], exclude=["lm_head", "*gate"]
    )
    err_algo = _max_logit_err(model, dataloader, algo_quant_config)

    assert err_algo < err_rtn, f"{algo_config.name} did not beat RTN: algo={err_algo:.6f} rtn={err_rtn:.6f}"
    logger.info(f"MoE {algo_config.name} beats RTN on OLMoE: algo={err_algo:.6f} < rtn={err_rtn:.6f}")


@slow_test
@skip_if_no_gpu
def test_moe_gptaq_beats_rtn():
    """GPTAQ on a real MoE checkpoint must beat round-to-nearest.

    GPTAQ runs two forward passes per block (quantized weights, then original weights) and the
    MoE router recomputes independently on each. The router-alignment fix replays the
    quantized-pass routing on the original pass so both passes agree on token->expert routing;
    without it the passes diverge on a real router. This asserts the algorithm produces lower
    quantization error than plain RTN on OLMoE.
    """
    _assert_algo_beats_rtn(
        GPTAQConfig(
            model_decoder_layers="model.layers",
            inside_layer_modules=OLMOE_INSIDE_LAYER_MODULES,
            block_size=128,
        )
    )


@slow_test
@skip_if_no_gpu
def test_moe_qronos_beats_rtn():
    """Qronos on a real MoE checkpoint must beat round-to-nearest.

    Like GPTAQ, Qronos runs a quantized pass and an original-weights pass per block and relies on
    the router-alignment fix to keep token->expert routing identical across the two. This asserts
    the algorithm produces lower quantization error than plain RTN on OLMoE.
    """
    _assert_algo_beats_rtn(
        QronosConfig(
            model_decoder_layers="model.layers",
            inside_layer_modules=OLMOE_INSIDE_LAYER_MODULES,
            block_size=128,
        )
    )


# ------------------------------------------------------------------------------------------------
# qwen4_exp (Qwen3.8-Flash-Next)
#
# Hybrid decoder: linear-attention and full-attention layers, MoE on every one, with the routed
# experts stored fused until preprocess_for_quantization splits them into per-expert linears. Only
# those routed experts are quantized, so these tests pin that the shipped configs actually resolve
# against the preprocessed tree -- the failure they guard against is patterns that look reasonable
# and match nothing, which quantizes the model but leaves the algorithm a no-op.
# ------------------------------------------------------------------------------------------------

QWEN4_EXP_CONFIG_PATH = "./test/test_for_torch/configs/moe_model/qwen4_exp"

# The architecture ships with transformers, not with Quark, so older transformers releases cannot
# build the fixture at all. Gate on the architecture being registered rather than on a version
# number: that is what these tests actually need, and it stays correct if the release that added
# it changes.
_QWEN4_EXP_IN_TRANSFORMERS = "qwen4_exp_text" in CONFIG_MAPPING_NAMES
requires_qwen4_exp = pytest.mark.skipif(
    not _QWEN4_EXP_IN_TRANSFORMERS,
    reason="the installed transformers does not provide the qwen4_exp architecture",
)
QWEN4_EXP_NUM_LAYERS = 2
QWEN4_EXP_NUM_EXPERTS = 4


def _qwen4_exp_quant_config():
    """The shipped template config, so the tests exercise what users actually get.

    Deliberately NOT used as the oracle for what ends up quantized: asserting the template's own
    exclusions against a scope derived from those same exclusions proves nothing. The expected
    scope is written out independently in
    :func:`test_qwen4_exp_template_quantizes_experts_only`.
    """
    from quark.torch.quantization.config.template import LLMTemplate

    return LLMTemplate.get("qwen4_exp_text").get_config("uint4_wo_128")


def _build_qwen4_exp():
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(QWEN4_EXP_CONFIG_PATH + "/config.json", **FROM_PRETRAINED_KWARGS)
    model = AutoModelForCausalLM.from_config(config=config, dtype=torch.float32).to(torch_device).eval()
    # Routed experts ship fused (Qwen4ExpTextExperts); split them into per-expert linears, which is
    # what every qwen4_exp algorithm config addresses.
    preprocess_for_quantization(model)
    input_ids = torch.randint(10, (1, 4), device=torch_device)
    return model, DataLoader(input_ids)


@requires_qwen4_exp
def test_qwen4_exp_preprocessing_exposes_every_routed_expert():
    model, _ = _build_qwen4_exp()
    names = [name for name, _ in model.named_modules()]
    per_expert = [n for n in names if ".mlp.experts." in n and n.endswith(("gate_proj", "up_proj", "down_proj"))]
    assert len(per_expert) == QWEN4_EXP_NUM_LAYERS * QWEN4_EXP_NUM_EXPERTS * 3, sorted(per_expert)[:8]
    # The shared expert has identically-named leaves; the routed-expert patterns must not claim it.
    assert any("shared_expert.down_proj" in n for n in names)


@pytest.mark.parametrize("algo", ["gptq", "gptaq", "qronos"])
@requires_qwen4_exp
def test_qwen4_exp_hessian_configs_select_every_expert_projection(algo):
    """inside_layer_modules must resolve against the indexed per-expert names.

    Unindexed spellings such as mlp.experts.down_proj never match mlp.experts.0.down_proj under
    fnmatch, so the processors would select nothing and quantization would silently fall back to
    plain round-to-nearest.
    """
    import fnmatch

    from quark.torch.algorithm.utils.module import get_named_quant_linears

    model, _ = _build_qwen4_exp()
    quantizer = ModelQuantizer(_qwen4_exp_quant_config())
    prepared = quantizer._prepare_model(model)
    algo_config = load_quant_algo_config_from_file(f"{QWEN4_EXP_CONFIG_PATH}/{algo}_config.json")

    for layer in prepared.model.layers:
        candidates = list(get_named_quant_linears(layer))
        selected = set()
        for pattern in algo_config.inside_layer_modules:
            selected |= set(fnmatch.filter(candidates, "*" + pattern))
        assert len(selected) == QWEN4_EXP_NUM_EXPERTS * 3 + 3, (
            f"{algo} selected {len(selected)}: {sorted(selected)[:6]}"
        )
        # The shared expert is quantized too, so it must be selected -- exactly once.
        assert len([n for n in selected if "shared_expert" in n]) == 3


@pytest.mark.parametrize("algo", ["awq", "autosmoothquant"])
@requires_qwen4_exp
def test_qwen4_exp_scaling_configs_resolve_to_every_expert(algo):
    """scaling_layers must expand to one up_proj -> down_proj pair per routed expert."""
    from quark.torch.algorithm.utils.module import get_named_quant_linears
    from quark.torch.algorithm.utils.prepare import get_layers_for_scaling

    model, _ = _build_qwen4_exp()
    quantizer = ModelQuantizer(_qwen4_exp_quant_config())
    prepared = quantizer._prepare_model(model)
    algo_config = load_quant_algo_config_from_file(f"{QWEN4_EXP_CONFIG_PATH}/{algo}_config.json")

    for layer in prepared.model.layers:
        # Every candidate hooked, so the only thing that can drop a group is the pattern itself.
        input_feat = {name: torch.zeros(1, 2) for name in get_named_quant_linears(layer)}
        resolved = get_layers_for_scaling(layer, input_feat, {}, algo_config.scaling_layers)
        targets = sum(len(entry["layers"]) for entry in resolved)
        assert targets == QWEN4_EXP_NUM_EXPERTS + 1, f"{algo} resolved {targets} target(s)"


@pytest.mark.parametrize(
    "algo,map_name,field",
    [
        ("gptq", "GPTQ_MAP", "inside_layer_modules"),
        ("gptaq", "GPTAQ_MAP", "inside_layer_modules"),
        ("qronos", "QRONOS_MAP", "inside_layer_modules"),
        ("awq", "AWQ_MAP", "scaling_layers"),
        ("autosmoothquant", "AUTOSMOOTHQUANT_MAP", "scaling_layers"),
    ],
)
def test_qwen4_exp_json_configs_match_the_shipped_maps(algo, map_name, field):
    """These JSONs exist to exercise the shipped configs, so they must not drift from them."""
    from quark.torch.quantization.config import algo_configs

    shipped = getattr(getattr(algo_configs, map_name)["qwen4_exp_text"], field)
    from_file = getattr(load_quant_algo_config_from_file(f"{QWEN4_EXP_CONFIG_PATH}/{algo}_config.json"), field)
    assert from_file == shipped, f"{algo}_config.json has drifted from {map_name}['qwen4_exp_text']"


@requires_qwen4_exp
def test_qwen4_exp_template_quantizes_experts_only():
    """The shipped template must quantize the routed and shared experts and nothing else.

    The expected set is written out here rather than derived from the template's own
    exclude_layers_name: using those exclusions as the oracle for the scope they produce is
    circular and would pass no matter which of them were deleted. Dropping "*.self_attn.*",
    "*.linear_attn.*", "*mlp.gate" or "*shared_expert_gate*" from the template makes this fail.
    """
    from quark.torch.algorithm.utils.module import get_named_quant_linears

    model, _ = _build_qwen4_exp()
    prepared = ModelQuantizer(_qwen4_exp_quant_config())._prepare_model(model)

    quantized = set(get_named_quant_linears(prepared))
    projections = ("gate_proj", "up_proj", "down_proj")
    expected = {
        f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
        for layer in range(QWEN4_EXP_NUM_LAYERS)
        for expert in range(QWEN4_EXP_NUM_EXPERTS)
        for proj in projections
    } | {
        # The shared expert is quantized as well; only its gate stays excluded.
        f"model.layers.{layer}.mlp.shared_expert.{proj}"
        for layer in range(QWEN4_EXP_NUM_LAYERS)
        for proj in projections
    }
    assert quantized == expected, (
        f"unexpected: {sorted(quantized - expected)[:8]}; missing: {sorted(expected - quantized)[:8]}"
    )

    # The excluded components exist in this fixture, so their absence above is a real result and
    # not an artefact of a model that never had them.
    present = {name for name, _ in prepared.named_modules()}
    for component in ("self_attn", "linear_attn", "mlp.gate", "shared_expert_gate"):
        assert any(component in name for name in present), f"fixture has no {component} to exclude"


def _qwen4_exp_text_config_dict() -> dict:
    import json

    cfg = json.loads((pathlib.Path(QWEN4_EXP_CONFIG_PATH) / "config.json").read_text())
    cfg.pop("architectures", None)
    return cfg


@requires_qwen4_exp
def test_qwen4_exp_config_exposes_both_model_type_spellings():
    """One checkpoint reports two model_types, which is why both map entries exist.

    file2file reads the raw top-level config ("qwen4_exp"); live loading resolves the nested
    text config ("qwen4_exp_text"). Registering only one silently leaves the other entry point
    without a config.
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    config = CONFIG_MAPPING["qwen4_exp"](text_config=_qwen4_exp_text_config_dict())
    assert config.model_type == "qwen4_exp"
    assert config.text_config.model_type == "qwen4_exp_text"


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text"])
@requires_qwen4_exp
def test_qwen4_exp_model_decoder_layers_resolve_on_the_real_class(model_type):
    """model_decoder_layers must resolve on the class each model_type actually loads as.

    Asserted against a built model rather than as a string: the two spellings differ only in this
    path, and a wrong one is only discovered when a run fails to find the decoder stack.
    """
    import torch.nn as nn
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from quark.torch.algorithm.utils.module import get_nested_attr_from_module
    from quark.torch.quantization.config.algo_configs import AWQ_MAP, GPTQ_MAP

    text = _qwen4_exp_text_config_dict()
    if model_type == "qwen4_exp":
        from transformers.models.qwen4_exp import Qwen4ExpForConditionalGeneration

        vision = {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_heads": 2,
            "intermediate_size": 16,
            "patch_size": 14,
            "in_channels": 3,
            "out_hidden_size": 16,
        }
        model = Qwen4ExpForConditionalGeneration(CONFIG_MAPPING["qwen4_exp"](text_config=text, vision_config=vision))
    else:
        model = AutoModelForCausalLM.from_config(config=CONFIG_MAPPING["qwen4_exp_text"](**text))

    for algo_map, name in ((AWQ_MAP, "AWQ_MAP"), (GPTQ_MAP, "GPTQ_MAP")):
        path = algo_map[model_type].model_decoder_layers
        layers = get_nested_attr_from_module(model, path)
        assert isinstance(layers, nn.ModuleList), f"{name}[{model_type}].model_decoder_layers={path!r}"
        assert len(layers) == QWEN4_EXP_NUM_LAYERS


def test_qwen4_exp_composite_model_type_loads_the_vision_wrapper(monkeypatch):
    """model_type "qwen4_exp" must load the composite class, not the text-only one.

    AutoModelForCausalLM resolves the nested text config and builds the text model, which has no
    vision tower -- so the exported checkpoint would silently lose it. Only the class dispatch is
    checked, with the config and from_pretrained both stubbed, so this needs neither a checkpoint
    nor a transformers release that ships the architecture.
    """
    from quark.torch.utils.llm import model_preparation

    class _Config:
        """Just enough config for the dispatch, which branches on model_type alone."""

        model_type = "qwen4_exp"

        def to_dict(self):
            return {"model_type": self.model_type}

    class _Dispatched(Exception):
        """Raised by the stub so the test stops at the class choice, before any model work."""

    class _Stub:
        @staticmethod
        def from_pretrained(path, **kwargs):
            raise _Dispatched

    monkeypatch.setattr(model_preparation, "Qwen4ExpForConditionalGeneration", _Stub, raising=False)
    monkeypatch.setattr(model_preparation.AutoConfig, "from_pretrained", staticmethod(lambda *a, **k: _Config()))
    with pytest.raises(_Dispatched):
        model_preparation.get_model("unused-path", data_type="bfloat16", device="cpu")
