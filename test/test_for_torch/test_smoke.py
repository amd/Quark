#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
import tempfile
from dataclasses import replace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.shares.utils.testing_utils import (
    require_torch_higher_or_equal,
    require_torch_lower_or_equal,
    torch_device,
    use_temporary_directory,
)
from quark.testing import slow_test
from quark.torch import ModelQuantizer, export_onnx, export_safetensors, load_params, save_params
from quark.torch.quantization import (
    AutoSmoothQuantConfig,
    AWQConfig,
    Bfloat16Spec,
    Float16Spec,
    FP8E4M3PerTensorSpec,
    FP8E5M2PerTensorSpec,
    Int2PerGroupSpec,
    Int4PerChannelSpec,
    Int4PerGroupSpec,
    Int4PerTensorSpec,
    Int8PerTensorSpec,
    QConfig,
    QLayerConfig,
    RotationConfig,
    SmoothQuantConfig,
    Uint4PerGroupSpec,
    Uint8PerGroupSpec,
)
from quark.torch.quantization.observer.observer import PerTensorMSEObserver, PerTensorPercentileObserver

# Quant_Spec
FLOAT16_SPEC = Float16Spec().to_quantization_spec()

BFLOAT16_SPEC = Bfloat16Spec().to_quantization_spec()

FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(is_dynamic=False).to_quantization_spec()

FP8_E5M2_PER_TENSOR_SPEC = FP8E5M2PerTensorSpec(is_dynamic=False).to_quantization_spec()


INT2_PER_GROUP_ASYM_SPEC = Int2PerGroupSpec(
    symmetric=False, ch_axis=1, is_dynamic=False, group_size=128
).to_quantization_spec()

INT4_PER_TENSOR_SPEC = Int4PerTensorSpec(is_dynamic=False).to_quantization_spec()

INT4_PER_CHANNEL_SPEC = Int4PerChannelSpec(ch_axis=0, is_dynamic=False).to_quantization_spec()

INT4_PER_GROUP_SYM_SPEC = Int4PerGroupSpec(ch_axis=1, is_dynamic=False, group_size=128).to_quantization_spec()

DEFAULT_UINT4_PER_GROUP_ASYM_SPEC = Uint4PerGroupSpec(
    ch_axis=1, is_dynamic=False, group_size=128
).to_quantization_spec()

DEFAULT_UINT8_PER_GROUP_ASYM_SPEC = Uint8PerGroupSpec(
    symmetric=False, scale_type="float", round_method="half_even", ch_axis=1, is_dynamic=False, group_size=128
).to_quantization_spec()

INT8_PER_TENSOR_SPEC = Int8PerTensorSpec(is_dynamic=False).to_quantization_spec()

INT8_PER_TENSOR_DYNAMIC_SPEC = Int8PerTensorSpec(is_dynamic=True).to_quantization_spec()

# Float16 config
DEFAULT_FLOAT16_CONFIG = QLayerConfig(input_tensors=FLOAT16_SPEC, weight=FLOAT16_SPEC)

# Fp8(e4m3) config
DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG = QLayerConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC)

DEFAULT_W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG = QLayerConfig(
    input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
)

# Fp8(e5m2) config
DEFAULT_W_FP8E5M2_A_FP8E5M2_PER_TENSOR_CONFIG = QLayerConfig(
    input_tensors=FP8_E5M2_PER_TENSOR_SPEC, weight=FP8_E5M2_PER_TENSOR_SPEC
)

DEFAULT_W_FP8E5M2_A_FP8E5M2_OFP8E5M2_PER_TENSOR_CONFIG = QLayerConfig(
    input_tensors=FP8_E5M2_PER_TENSOR_SPEC, weight=FP8_E5M2_PER_TENSOR_SPEC, output_tensors=FP8_E5M2_PER_TENSOR_SPEC
)

# Per tensor config
DEFAULT_W_INT2_PER_GROUP_CONFIG = QLayerConfig(weight=INT2_PER_GROUP_ASYM_SPEC)

DEFAULT_W_INT4_PER_TENSOR_CONFIG = QLayerConfig(weight=INT4_PER_TENSOR_SPEC)

DEFAULT_W_INT4_BIAS_INT4_PER_TENSOR_CONFIG = QLayerConfig(weight=INT4_PER_TENSOR_SPEC, bias=INT4_PER_TENSOR_SPEC)

DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG = QLayerConfig(input_tensors=INT8_PER_TENSOR_SPEC, weight=INT8_PER_TENSOR_SPEC)

DEFAULT_W_INT8_A_INT8_O_INT8_PER_TENSOR_CONFIG = QLayerConfig(
    input_tensors=INT8_PER_TENSOR_SPEC, weight=INT8_PER_TENSOR_SPEC, output_tensors=INT8_PER_TENSOR_SPEC
)

DEFAULT_W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG = QLayerConfig(
    input_tensors=INT8_PER_TENSOR_DYNAMIC_SPEC, weight=INT8_PER_TENSOR_DYNAMIC_SPEC
)

# Per Channel QConfig
DEFAULT_W_INT4_PER_CHANNEL_CONFIG = QLayerConfig(weight=INT4_PER_CHANNEL_SPEC)

# Per Group QConfig
DEFAULT_W_INT4_PER_GROUP_SYM_CONFIG = QLayerConfig(weight=INT4_PER_GROUP_SYM_SPEC)

DEFAULT_W_UINT4_PER_GROUP_CONFIG = QLayerConfig(weight=DEFAULT_UINT4_PER_GROUP_ASYM_SPEC)

DEFAULT_W_UINT8_PER_GROUP_CONFIG = QLayerConfig(weight=DEFAULT_UINT8_PER_GROUP_ASYM_SPEC)

DEFAULT_W_UINT4_A_BFLOAT16_PER_GROUP_CONFIG = QLayerConfig(
    input_tensors=BFLOAT16_SPEC, weight=DEFAULT_UINT4_PER_GROUP_ASYM_SPEC
)

DEFAULT_W_UINT8_A_BFLOAT16_PER_GROUP_CONFIG = QLayerConfig(
    input_tensors=BFLOAT16_SPEC, weight=DEFAULT_UINT8_PER_GROUP_ASYM_SPEC
)

# Default AWQ QConfig
DEFAULT_AWQ_CONFIG = QConfig(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG, algo_config=[AWQConfig()])

# Default SmoothQuant QConfig
DEFAULT_SMOOTH_QUANT_CONFIG = QConfig(
    global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG, algo_config=[SmoothQuantConfig()]
)

# Default AutoSmoothQuant QConfig
DEFAULT_AutoSmoothQuant_CONFIG = QConfig(
    global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG, algo_config=[AutoSmoothQuantConfig()]
)

# mutil_gpu should disable lm_head replacement in opt, qwen, llama, and it is needed in algos.
EXCLUDE_LAYERS = ["lm_head"]

sys.path.append("..")


def get_dataloader(model_name="facebook/opt-125m", device=torch_device):
    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"].to(device))
    return calib_dataloader


def quantize_model(
    quant_config, input_args=None, model_name="facebook/opt-125m", multi_gpu=False, freeze: bool = False
):
    # Get quantizer
    quantizer = ModelQuantizer(quant_config)

    if multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True
        )
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        model.eval()
        model = model.to(torch_device)
    # Get dataloader, if multi_gpu, give the first layer's device
    calib_dataloader = get_dataloader(model_name, model.device)

    quant_model = quantizer.quantize_model(model, calib_dataloader)

    if input_args is not None:
        with torch.no_grad():
            res_bef = quant_model(input_args).logits

    if freeze:
        quant_model = quantizer.freeze(quant_model)

    if input_args is not None:
        with torch.no_grad():
            res_aft = quant_model(input_args).logits

        assert torch.allclose(res_bef, res_aft)

    # Inference with quantized model
    for i in calib_dataloader:
        quant_model(i)

    return quant_model


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize(
    "quant_config",
    [
        (DEFAULT_W_INT2_PER_GROUP_CONFIG),
        (DEFAULT_FLOAT16_CONFIG),
        (DEFAULT_W_INT4_PER_GROUP_SYM_CONFIG),
        (DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG),
        (DEFAULT_W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG),
        (DEFAULT_W_FP8E5M2_A_FP8E5M2_PER_TENSOR_CONFIG),
        (DEFAULT_W_FP8E5M2_A_FP8E5M2_OFP8E5M2_PER_TENSOR_CONFIG),
        (DEFAULT_W_INT4_PER_TENSOR_CONFIG),
        (DEFAULT_W_INT4_BIAS_INT4_PER_TENSOR_CONFIG),
        (DEFAULT_W_INT4_PER_CHANNEL_CONFIG),
        (DEFAULT_W_UINT4_PER_GROUP_CONFIG),
        (DEFAULT_W_UINT4_A_BFLOAT16_PER_GROUP_CONFIG),
        (DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG),
        (DEFAULT_W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG),
        (DEFAULT_W_UINT8_PER_GROUP_CONFIG),
        (DEFAULT_W_UINT8_A_BFLOAT16_PER_GROUP_CONFIG),
    ],
)
def test_smoke_basic_quantization(quant_config):
    """
    Test Features:
        Data Type:                Float16 / Bfloat16 / Int4 / Uint4 / Int8 / FP8(e4m3fn) / FP8(e5m2)
        Quantization Strategies:  Post Training Weight-Only Quantization / Post Training Dynamic Quantization / Post Training Static Quantization
        Quantization Scheme:      Per tensor / Per channel / Per group
        Symmetric:                Symmetric / Asymmetric
        In-Place Replace OP:      nn.Linear
    """
    quant_config = QConfig(global_quant_config=quant_config, exclude=EXCLUDE_LAYERS)
    quantize_model(quant_config)
    # quantize_model(quant_config, multi_gpu=True)# TODO: uncomment after ROCM support multi-GPU


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize(
    "global_config,kv_config",
    [
        (DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG, FP8_PER_TENSOR_SPEC),
        (DEFAULT_W_FP8E5M2_A_FP8E5M2_PER_TENSOR_CONFIG, FP8_E5M2_PER_TENSOR_SPEC),
    ],
)
def test_smoke_kv_cache_quant(global_config, kv_config):
    """
    Test Features:
        KV-Cache Quant:           FP8 KV-Cache Quant
    """
    quant_config = QConfig(global_quant_config=global_config)
    KV_CACHE_CFG = {
        "*v_proj": QLayerConfig(input_tensors=kv_config, output_tensors=kv_config, weight=kv_config),
        "*k_proj": QLayerConfig(input_tensors=kv_config, output_tensors=kv_config, weight=kv_config),
    }
    quant_config = replace(quant_config, layer_quant_config=KV_CACHE_CFG, exclude=EXCLUDE_LAYERS)

    quantize_model(quant_config)
    # quantize_model(quant_config, multi_gpu=True)# TODO: uncomment after ROCM support multi-GPU


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize(
    "global_config,softmax_quant_spec",
    [
        (DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG, FP8_PER_TENSOR_SPEC),
    ],
)
def test_smoke_fp8_attn_quant(global_config, softmax_quant_spec):
    """
    Test Features:
        FP8 Attn Quant:        FP8 Attention Quant
    """
    quant_config = QConfig(
        global_quant_config=global_config, softmax_quant_spec=softmax_quant_spec, exclude=EXCLUDE_LAYERS
    )

    with torch.inference_mode(), tempfile.TemporaryDirectory() as tmpdir:
        #        for multi_gpu in [False, True]:# TODO: uncomment after ROCM support multi-GPU
        for multi_gpu in [False]:
            model = quantize_model(quant_config, multi_gpu=multi_gpu)
            export_safetensors(model=model, output_dir=tmpdir, weight_format="real_quantized", pack_method="reorder")


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize(
    "global_config,observer_cls",
    [
        (DEFAULT_W_INT4_PER_TENSOR_CONFIG, PerTensorPercentileObserver),
        (DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG, PerTensorPercentileObserver),
        (DEFAULT_W_FP8E5M2_A_FP8E5M2_PER_TENSOR_CONFIG, PerTensorPercentileObserver),
        (DEFAULT_W_INT4_PER_TENSOR_CONFIG, PerTensorMSEObserver),
        (DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG, PerTensorMSEObserver),
        (DEFAULT_W_FP8E5M2_A_FP8E5M2_PER_TENSOR_CONFIG, PerTensorMSEObserver),
    ],
)
def test_smoke_calibration_method(global_config, observer_cls):
    """
    Test Features:
        Calibration method:       MinMax / Percentile / MSE
    """
    quant_config = QConfig(global_quant_config=global_config, exclude=EXCLUDE_LAYERS)
    quant_config.global_quant_config.weight = replace(
        quant_config.global_quant_config.weight, observer_cls=observer_cls
    )
    quantize_model(quant_config)
    # quantize_model(quant_config, multi_gpu=True)# TODO: uncomment after ROCM support multi-GPU


@pytest.mark.parametrize(
    "global_config",
    [
        (DEFAULT_W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG),
        (DEFAULT_W_FP8E5M2_A_FP8E5M2_OFP8E5M2_PER_TENSOR_CONFIG),
    ],
)
def test_smoke_export_hf_format_fp8(global_config):
    """
    Test Features:
        Export Format: HF / safetensors
    """
    quant_config = QConfig(global_quant_config=global_config, exclude=EXCLUDE_LAYERS)

    with torch.inference_mode(), tempfile.TemporaryDirectory() as tmpdir:
        #        for multi_gpu in [False, True]:# TODO: uncomment after ROCM support multi-GPU
        for multi_gpu in [False]:
            model = quantize_model(quant_config, model_name="Qwen/Qwen1.5-0.5B", multi_gpu=multi_gpu)
            export_safetensors(model=model, output_dir=tmpdir, weight_format="fake_quantized", pack_method="reorder")


@use_temporary_directory
def test_smoke_export_hf_format_pergroup(tmpdir: str):
    """
    Test Features:
        Export Format: HF / safetensors
    """
    quant_config = QConfig(global_quant_config=DEFAULT_W_INT4_PER_GROUP_SYM_CONFIG, exclude=EXCLUDE_LAYERS)

    with torch.inference_mode():
        #        for multi_gpu in [False, True]:# TODO: uncomment after ROCM support multi-GPU
        for multi_gpu in [False]:
            model = quantize_model(quant_config, model_name="Qwen/Qwen1.5-0.5B", multi_gpu=multi_gpu)
            export_safetensors(model=model, output_dir=tmpdir, weight_format="fake_quantized", pack_method="reorder")


@use_temporary_directory
def test_smoke_export_hf_format_pertensor(tmpdir: str):
    """
    Test Features:
        Export Format: HF / safetensors
    """
    quant_config = QConfig(global_quant_config=DEFAULT_W_INT8_A_INT8_O_INT8_PER_TENSOR_CONFIG, exclude=EXCLUDE_LAYERS)

    with torch.inference_mode():
        #        for multi_gpu in [False, True]:# TODO: uncomment after ROCM support multi-GPU
        for multi_gpu in [False]:
            model = quantize_model(quant_config, model_name="Qwen/Qwen1.5-0.5B", multi_gpu=multi_gpu)
            export_safetensors(model=model, output_dir=tmpdir, weight_format="fake_quantized", pack_method="reorder")


@use_temporary_directory
def test_smoke_export_quark_perchannel_safetensors(tmpdir: str):
    """
    Test Features:
        Export Format: HF / safetensors
    """
    quant_config = QConfig(global_quant_config=DEFAULT_W_INT4_PER_CHANNEL_CONFIG, exclude=EXCLUDE_LAYERS)

    with torch.inference_mode():
        #        for multi_gpu in [False, True]:# TODO: uncomment after ROCM support multi-GPU
        for multi_gpu in [False]:
            model = quantize_model(quant_config, model_name="Qwen/Qwen1.5-0.5B", multi_gpu=multi_gpu)
            export_safetensors(model=model, output_dir=tmpdir, weight_format="real_quantized", pack_method="reorder")


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 20)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.relu(self.lin(x))


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize(
    "global_config",
    [
        (DEFAULT_W_INT4_PER_TENSOR_CONFIG),
        (DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG),
        (DEFAULT_W_FP8E5M2_A_FP8E5M2_PER_TENSOR_CONFIG),
    ],
)
def test_smoke_export_onnx(global_config):
    """
    Test Features:
        Export Format:            ONNX
    """
    quant_config = QConfig(global_quant_config=global_config, exclude=EXCLUDE_LAYERS)
    with torch.inference_mode():
        with torch_device:
            model = MyModel()
            model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            quantizer = ModelQuantizer(quant_config)

            calib_dataloader = DataLoader(torch.rand(3, 10, device=torch_device))

            quant_model = quantizer.quantize_model(model, calib_dataloader)

            # Inference with quantized model
            for inp in calib_dataloader:
                quant_model(inp)

            export_onnx(model=model, output_dir=tmpdir, input_args=(torch.rand(4, 10, device=torch_device),))


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
def test_smoke_eager_save_load():
    """
    Test Features:
        Eager mode save and load functions
    """
    quant_config = QConfig(global_quant_config=DEFAULT_W_INT4_PER_GROUP_SYM_CONFIG, exclude=EXCLUDE_LAYERS)

    with torch.inference_mode():
        calib_dataloader = get_dataloader()
        batch_iter = iter(calib_dataloader)
        input_args = next(batch_iter)

        # TODO: add multi_gpu=True case after ROCM support multi-GPU
        multi_gpu = False

        with tempfile.TemporaryDirectory() as tmpdir:
            quant_model = quantize_model(
                quant_config, model_name="facebook/opt-125m", multi_gpu=multi_gpu, freeze=True, input_args=input_args
            )
            saved_out = quant_model(input_args)
            save_params(quant_model, model_type="opt", export_dir=tmpdir)

            model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype="auto")
            model.eval()
            model = model.to(torch_device)
            json_path = tmpdir + "/opt.json"
            safetensors_path = tmpdir + "/opt.safetensors"
            loaded_model = load_params(model, json_path=json_path, safetensors_path=safetensors_path)

            print("loaded_model here", loaded_model)

            loaded_out = loaded_model(input_args)

            assert torch.allclose(saved_out["logits"], loaded_out["logits"], atol=1e-2)


def set_config_for_awq_or_smooth(algo_config):
    algo_config.scaling_layers = [
        {
            "prev_op": "self_attn_layer_norm",
            "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            "inp": "self_attn.q_proj",
            "module2inspect": "self_attn",
        },
        {
            "prev_op": "self_attn.v_proj",
            "layers": ["self_attn.out_proj"],
            "inp": "self_attn.out_proj",
        },
        {
            "prev_op": "final_layer_norm",
            "layers": ["fc1"],
            "inp": "fc1",
        },
        {
            "prev_op": "fc1",
            "layers": ["fc2"],
            "inp": "fc2",
        },
    ]
    algo_config.model_decoder_layers = "model.decoder.layers"
    return algo_config


def set_config_for_autosmooth_mae(algo_config):
    algo_config.scaling_layers = [
        {
            "prev_op": "self_attn_layer_norm",
            "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            "inp": "self_attn.q_proj",
            "module2inspect": "self_attn",
        },
        {
            "prev_op": "self_attn.v_proj",
            "layers": ["self_attn.out_proj"],
            "inp": "self_attn.out_proj",
        },
        {
            "prev_op": "final_layer_norm",
            "layers": ["fc1"],
            "inp": "fc1",
        },
        {
            "prev_op": "fc1",
            "layers": ["fc2"],
            "inp": "fc2",
        },
    ]
    algo_config.model_decoder_layers = "model.decoder.layers"
    algo_config.compute_scale_loss = "MAE"
    return algo_config


def set_config_for_autosmooth_mse(algo_config):
    algo_config.scaling_layers = [
        {
            "prev_op": "self_attn_layer_norm",
            "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            "inp": "self_attn.q_proj",
            "module2inspect": "self_attn",
        },
        {
            "prev_op": "self_attn.v_proj",
            "layers": ["self_attn.out_proj"],
            "inp": "self_attn.out_proj",
        },
        {
            "prev_op": "final_layer_norm",
            "layers": ["fc1"],
            "inp": "fc1",
        },
        {
            "prev_op": "fc1",
            "layers": ["fc2"],
            "inp": "fc2",
        },
    ]
    algo_config.model_decoder_layers = "model.decoder.layers"
    algo_config.compute_scale_loss = "MSE"
    return algo_config


def set_config_for_autosmooth_rmse(algo_config):
    algo_config.scaling_layers = [
        {
            "prev_op": "self_attn_layer_norm",
            "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            "inp": "self_attn.q_proj",
            "module2inspect": "self_attn",
        },
        {
            "prev_op": "self_attn.v_proj",
            "layers": ["self_attn.out_proj"],
            "inp": "self_attn.out_proj",
        },
        {
            "prev_op": "final_layer_norm",
            "layers": ["fc1"],
            "inp": "fc1",
        },
        {
            "prev_op": "fc1",
            "layers": ["fc2"],
            "inp": "fc2",
        },
    ]
    algo_config.model_decoder_layers = "model.decoder.layers"
    algo_config.compute_scale_loss = "RMSE"
    return algo_config


def set_config_for_autosmooth_rmae(algo_config):
    algo_config.scaling_layers = [
        {
            "prev_op": "self_attn_layer_norm",
            "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            "inp": "self_attn.q_proj",
            "module2inspect": "self_attn",
        },
        {
            "prev_op": "self_attn.v_proj",
            "layers": ["self_attn.out_proj"],
            "inp": "self_attn.out_proj",
        },
        {
            "prev_op": "final_layer_norm",
            "layers": ["fc1"],
            "inp": "fc1",
        },
        {
            "prev_op": "fc1",
            "layers": ["fc2"],
            "inp": "fc2",
        },
    ]
    algo_config.model_decoder_layers = "model.decoder.layers"
    algo_config.compute_scale_loss = "RMAE"
    return algo_config


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
def test_smoke_smooth_quant_quantization():
    """
    Test Features:
        Pre-Quant Optimization:   SmoothQuant
    """
    quant_config = DEFAULT_SMOOTH_QUANT_CONFIG
    quant_config.algo_config = [
        set_config_for_awq_or_smooth(quant_config.algo_config[0]),
    ]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)
    quantize_model(quant_config)
    # quantize_model(quant_config, multi_gpu=True)# TODO: uncomment after ROCM support multi-GPU


@slow_test
def test_smoke_awq_quantization():
    """
    Test Features:
        Quant Algorithm:          AWQ
    """
    quant_config = DEFAULT_AWQ_CONFIG
    quant_config.algo_config = [set_config_for_awq_or_smooth(quant_config.algo_config[0])]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)
    quantize_model(quant_config)
    # quantize_model(quant_config, multi_gpu=True)# TODO: uncomment after ROCM support multi-GPU


@slow_test
def test_smoke_autosmoothquant_quantization():
    """
    Test Features:
        Quant Algorithm:          AWQ
    """
    quant_config = DEFAULT_AutoSmoothQuant_CONFIG
    quant_config.algo_config = [set_config_for_autosmooth_mae(quant_config.algo_config[0])]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)
    quantize_model(quant_config)

    quant_config = DEFAULT_AutoSmoothQuant_CONFIG
    quant_config.algo_config = [set_config_for_autosmooth_mse(quant_config.algo_config[0])]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)
    quantize_model(quant_config)

    quant_config = DEFAULT_AutoSmoothQuant_CONFIG
    quant_config.algo_config = [set_config_for_autosmooth_rmse(quant_config.algo_config[0])]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)
    quantize_model(quant_config)

    quant_config = DEFAULT_AutoSmoothQuant_CONFIG
    quant_config.algo_config = [set_config_for_autosmooth_rmae(quant_config.algo_config[0])]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)
    try:
        quantize_model(quant_config)
    except Exception:
        print("Expected 'MAE', 'MSE' or 'RMSE'.")
    # quantize_model(quant_config, multi_gpu=True)# TODO: uncomment after ROCM support multi-GPU


# TODO: This case may cause a core dump under specific versions (2.8.0, 2.9.0)
@require_torch_lower_or_equal("2.7.0")
@slow_test
def test_smoke_smooth_quant_and_awq_quantization():
    """
    Test Features:
        Pre-Quant Optimization:   SmoothQuant
        Quant Algorithm:          AWQ
    """
    quant_config = QConfig(
        global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG,
        algo_config=[SmoothQuantConfig(), AWQConfig()],
        exclude=EXCLUDE_LAYERS,
    )
    quant_config.algo_config = [
        set_config_for_awq_or_smooth(quant_config.algo_config[0]),
        set_config_for_awq_or_smooth(quant_config.algo_config[1]),
    ]
    quantize_model(quant_config)


def set_config_for_rotation_quarot():
    scaling_layers = {
        "first_layer": [
            {
                "prev_modules": ["model.embed_tokens"],
                "norm_module": "model.layers.layer_id.input_layernorm",
                "next_modules": [
                    "model.layers.layer_id.self_attn.q_proj",
                    "model.layers.layer_id.self_attn.k_proj",
                    "model.layers.layer_id.self_attn.v_proj",
                ],
            },
            {
                "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
                "norm_module": "model.layers.layer_id.post_attention_layernorm",
                "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"],
            },
        ],
        "middle_layers": [
            {
                "prev_modules": ["model.layers.pre_layer_id.mlp.down_proj"],
                "norm_module": "model.layers.layer_id.input_layernorm",
                "next_modules": [
                    "model.layers.layer_id.self_attn.q_proj",
                    "model.layers.layer_id.self_attn.k_proj",
                    "model.layers.layer_id.self_attn.v_proj",
                ],
            },
            {
                "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
                "norm_module": "model.layers.layer_id.post_attention_layernorm",
                "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"],
            },
        ],
        "last_layer": [
            {
                "prev_modules": ["model.layers.layer_id.mlp.down_proj"],
                "norm_module": "model.norm",
                "next_modules": ["lm_head"],
            }
        ],
    }
    return scaling_layers


def test_smoke_rotation():
    """
    Test Features:
        Pre-Quant Optimization:   Rotation
    """
    rotation_config = RotationConfig(
        scaling_layers=set_config_for_rotation_quarot(), model_decoder_layers="model.layers"
    )

    quant_config = QConfig(
        global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG,
        algo_config=[
            rotation_config,
        ],
        exclude=EXCLUDE_LAYERS,
    )
    quantize_model(quant_config, model_name="Qwen/Qwen1.5-0.5B")
    # quantize_model(quant_config, multi_gpu=True)# TODO: uncomment after ROCM support multi-GPU


def test_smoke_quarot():
    """
    Test Features:
        Pre-Quant Optimization:   QuaRot
    """
    quarot_config = RotationConfig(
        scaling_layers=set_config_for_rotation_quarot(),
        backbone="model",
        model_decoder_layers="model.layers",
        v_proj="self_attn.v_proj",
        o_proj="self_attn.o_proj",
        self_attn="self_attn",
        mlp="mlp",
        r2=True,
        r3=False,
        r4=False,
    )

    quant_config = QConfig(
        global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG,
        algo_config=[
            quarot_config,
        ],
        exclude=EXCLUDE_LAYERS,
    )
    quantize_model(quant_config, model_name="Qwen/Qwen1.5-0.5B")
    # quantize_model(quant_config, multi_gpu=True)# TODO: uncomment after ROCM support multi-GPU
