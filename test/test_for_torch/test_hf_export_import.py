
#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from torch.utils.data import DataLoader
from dataclasses import replace
from quark.torch import ModelQuantizer
from quark.torch.export.main_export.quant_config_parser import QuantConfigParser
from quark.torch.quantization.config.config import Config, QuantizationSpec, QuantizationConfig, AWQConfig, GPTQConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType
from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver, PerTensorMinMaxObserver, PerChannelMinMaxObserver
from quark.torch import ModelExporter, ModelImporter
from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
from quark.torch.export.safetensors import _load_weights_from_safetensors
from quark.torch.quantization import OCP_MXFP8E4M3Spec, OCP_MXFP4Spec, FP8E4M3PerTensorSpec, Int4PerChannelSpec, \
                                     FP4PerGroupSpec, ScaleQuantSpec, FP6E2M3PerGroupSpec, FP6E3M2PerGroupSpec
from quark.shares.utils.testing_utils import require_accelerate, require_torch_multi_gpu
from quark.testing import slow_test

import quark

import tempfile
from typing import Optional, List

from quark.torch.quantization.utils import set_op_by_name
from quark.shares.utils.testing_utils import torch_device, use_temporary_directory, retry_flaky_test, delete_directory_content
import os
import gc
from transformers.models.dbrx.modeling_dbrx import DbrxForCausalLM, DbrxExperts
import huggingface_hub

from dbrx_expert import DbrxExperts_

MODEL_DIR = "facebook/opt-125m"
torch.manual_seed(42)
INPUT_IDS = torch.randint(0, 1024, (1, 10)).to(torch_device)

GPTQ_CONFIG = GPTQConfig(model_decoder_layers="model.decoder.layers", inside_layer_modules=["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.out_proj", "fc1", "fc2"])

UINT4_PER_GROUP_ASYM_SPEC = QuantizationSpec(dtype=Dtype.uint4, observer_cls=PerGroupMinMaxObserver, symmetric=False, scale_type=ScaleType.float, round_method=RoundType.half_even, qscheme=QSchemeType.per_group, ch_axis=1, is_dynamic=False, group_size=32)

def get_dataloader(model_name="facebook/opt-125m", device=torch_device):
    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs['input_ids'].to(device))
    return calib_dataloader

def quantize_model(quant_config, model_name="facebook/opt-125m", multi_gpu=False, device_map: Optional[str] = "auto"):
    # Get quantizer
    quantizer = ModelQuantizer(quant_config)

    if multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     device_map=device_map,
                                                     torch_dtype="auto",
                                                     trust_remote_code=True)
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        model = model.to(torch_device)
    # Get dataloader, if multi_gpu, give the first layer's device
    calib_dataloader = get_dataloader(model_name, model.device)

    quant_model = quantizer.quantize_model(model, calib_dataloader)
    # Inference with quantized model
    for i in calib_dataloader:
        quant_model(i)
    quant_model = quantizer.freeze(quant_model)

    return quant_model

@require_accelerate
@require_torch_multi_gpu
@pytest.mark.parametrize("weight_format", ["real_quantized", "fake_quantized"])
def test_load_multi_device(weight_format: str):
    INT8_PER_TENSER_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                            qscheme=QSchemeType.per_tensor,
                                            observer_cls=PerTensorMinMaxObserver,
                                            symmetric=True,
                                            scale_type=ScaleType.float,
                                            round_method=RoundType.half_even,
                                            is_dynamic=False)

    INT8_PER_TENSOR_CONFIG = QuantizationConfig(weight=INT8_PER_TENSER_SPEC, input_tensors=INT8_PER_TENSER_SPEC, output_tensors=INT8_PER_TENSER_SPEC, bias=INT8_PER_TENSER_SPEC)
    quant_config = Config(global_quant_config=INT8_PER_TENSOR_CONFIG)

    EXCLUDE_LAYERS = ["lm_head", "*.gate", "*.shared_expert_gate"]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)

    with tempfile.TemporaryDirectory() as tmpdir:
        with torch.inference_mode():
            quant_model = quantize_model(quant_config, model_name=MODEL_DIR, multi_gpu=False)
            NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format=weight_format, pack_method="reorder")
            export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)

            model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
            model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config)

            quant_model(INPUT_IDS).to_tuple()

            with torch.no_grad():
                ref_outputs = quant_model(INPUT_IDS).to_tuple()

            device_map = {
                "model.decoder.embed_tokens": "cuda:0",
                "model.decoder.embed_positions": "cuda:0",
                "model.decoder.final_layer_norm": "cuda:0",
                "model.decoder.layers": "cuda:1",
                "lm_head": "cuda:0"
            }

            original_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map=device_map)

            importer = ModelImporter(tmpdir, saved_format="safetensors")
            q_model = importer.import_model_info(original_model)
            q_model = q_model.eval()

            with torch.no_grad():
                outputs = q_model(INPUT_IDS).to_tuple()

            for ref_output, output in zip(ref_outputs[0], outputs[0]):
                assert torch.allclose(output, ref_output, atol=1e-4)

            # Make sure the quantized parameters are on the specified device
            # in the original device_map.
            for param_name, param in q_model.named_parameters():
                for device_map_key, device_map_device in device_map.items():
                    if device_map_key in param_name:
                        assert param.device == torch.device(device_map_device)
                        break
                else:
                    raise RuntimeError("should not go here")

@pytest.mark.parametrize("qscheme", [
    pytest.param(qscheme, id=str(qscheme)) for qscheme in [QSchemeType.per_tensor, QSchemeType.per_channel, QSchemeType.per_group]
])
@pytest.mark.parametrize("weight_format", ["real_quantized", "fake_quantized"])
def test_int2_import_export(qscheme: QSchemeType, weight_format: str):
    '''
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      INT4
    '''
    model_id = "facebook/opt-125m"
    qscheme_to_observer = {
        QSchemeType.per_tensor: PerTensorMinMaxObserver,
        QSchemeType.per_channel: PerChannelMinMaxObserver,
        QSchemeType.per_group: PerGroupMinMaxObserver,
    }
    qscheme_to_ch_axis = {
        QSchemeType.per_tensor: None,
        QSchemeType.per_channel: 0,
        QSchemeType.per_group: 1,
    }
    quant_spec = QuantizationSpec(dtype=Dtype.int2,
                                  qscheme=qscheme,
                                  observer_cls=qscheme_to_observer[qscheme],
                                  symmetric=True,
                                  scale_type=ScaleType.float,
                                  round_method=RoundType.half_even,
                                  is_dynamic=False,
                                  ch_axis=qscheme_to_ch_axis[qscheme],
                                  group_size=8 if qscheme == QSchemeType.per_group else None)

    quant_config = Config(global_quant_config=QuantizationConfig(weight=quant_spec))

    quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False, device_map=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_config = ExporterConfig(json_export_config=JsonExporterConfig(weight_format=weight_format))

        model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
        model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config)

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        for device in ["meta", torch_device]:
            config = AutoConfig.from_pretrained(model_id)
            with torch.device(device):
                original_model = AutoModelForCausalLM.from_config(config)

            # Used for later comparison.
            model_state_dict = _load_weights_from_safetensors(tmpdir)

            importer = ModelImporter(tmpdir, saved_format="safetensors")
            q_model = importer.import_model_info(original_model)
            q_model = q_model.eval()

            q_model_state_dict = q_model.state_dict()

            if weight_format == "real_quantized":
                for key in model_state_dict.keys():
                    assert model_state_dict[key].dtype == q_model_state_dict[key].dtype
                    assert model_state_dict[key].shape == q_model_state_dict[key].shape

            for _, param in q_model.named_parameters():
                assert param.device != "meta"
            for _, param in q_model.named_buffers():
                assert param.device != "meta"

            if device == "meta":
                q_model = q_model.to(torch_device)

            with torch.no_grad():
                outputs = q_model(INPUT_IDS).to_tuple()

            for ref_output, output in zip(ref_outputs[0], outputs[0]):
                if torch_device.type == "cpu":
                    assert torch.equal(ref_output, output)
                else:
                    assert torch.allclose(output, ref_output, atol=1e-4)


@slow_test
@pytest.mark.parametrize("qscheme", [
    pytest.param(qscheme, id=str(qscheme)) for qscheme in [QSchemeType.per_tensor, QSchemeType.per_channel, QSchemeType.per_group]
])
@pytest.mark.parametrize("weight_format", ["real_quantized", "fake_quantized"])
def test_int4_import_export(qscheme: QSchemeType, weight_format: str):
    '''
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      INT4
    '''
    model_id = "facebook/opt-125m"
    qscheme_to_observer = {
        QSchemeType.per_tensor: PerTensorMinMaxObserver,
        QSchemeType.per_channel: PerChannelMinMaxObserver,
        QSchemeType.per_group: PerGroupMinMaxObserver,
    }
    qscheme_to_ch_axis = {
        QSchemeType.per_tensor: None,
        QSchemeType.per_channel: 0,
        QSchemeType.per_group: 1,
    }
    quant_spec = QuantizationSpec(dtype=Dtype.int4,
                                  qscheme=qscheme,
                                  observer_cls=qscheme_to_observer[qscheme],
                                  symmetric=True,
                                  scale_type=ScaleType.float,
                                  round_method=RoundType.half_even,
                                  is_dynamic=False,
                                  ch_axis=qscheme_to_ch_axis[qscheme],
                                  group_size=8 if qscheme == QSchemeType.per_group else None)

    quant_config = Config(global_quant_config=QuantizationConfig(weight=quant_spec))

    quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False, device_map=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_config = ExporterConfig(json_export_config=JsonExporterConfig(weight_format=weight_format))

        model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
        model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config)

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        for device in ["meta", torch_device]:
            config = AutoConfig.from_pretrained(model_id)
            with torch.device(device):
                original_model = AutoModelForCausalLM.from_config(config)

            # Used for later comparison.
            model_state_dict = _load_weights_from_safetensors(tmpdir)

            importer = ModelImporter(tmpdir, saved_format="safetensors")
            q_model = importer.import_model_info(original_model)
            q_model = q_model.eval()

            q_model_state_dict = q_model.state_dict()

            if weight_format == "real_quantized":
                for key in model_state_dict.keys():
                    assert model_state_dict[key].dtype == q_model_state_dict[key].dtype
                    assert model_state_dict[key].shape == q_model_state_dict[key].shape

            for _, param in q_model.named_parameters():
                assert param.device != "meta"
            for _, param in q_model.named_buffers():
                assert param.device != "meta"

            if device == "meta":
                q_model = q_model.to(torch_device)

            with torch.no_grad():
                outputs = q_model(INPUT_IDS).to_tuple()

            for ref_output, output in zip(ref_outputs[0], outputs[0]):
                if torch_device.type == "cpu":
                    assert torch.equal(ref_output, output)
                else:
                    assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.

@pytest.mark.parametrize("qscheme", [
    pytest.param(qscheme, id=str(qscheme)) for qscheme in [QSchemeType.per_tensor, QSchemeType.per_channel, QSchemeType.per_group]
])
@pytest.mark.parametrize("weight_format", ["real_quantized", "fake_quantized"])
def test_int8_import_export(qscheme: QSchemeType, weight_format: str):
    '''
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      INT8
    '''
    qscheme_to_observer = {
        QSchemeType.per_tensor: PerTensorMinMaxObserver,
        QSchemeType.per_channel: PerChannelMinMaxObserver,
        QSchemeType.per_group: PerGroupMinMaxObserver,
    }
    qscheme_to_ch_axis = {
        QSchemeType.per_tensor: None,
        QSchemeType.per_channel: 0,
        QSchemeType.per_group: 1,
    }
    quant_spec = QuantizationSpec(dtype=Dtype.int8,
                                  qscheme=qscheme,
                                  observer_cls=qscheme_to_observer[qscheme],
                                  symmetric=True,
                                  scale_type=ScaleType.float,
                                  round_method=RoundType.half_even,
                                  is_dynamic=False,
                                  ch_axis=qscheme_to_ch_axis[qscheme],
                                  group_size=8 if qscheme == QSchemeType.per_group else None)

    quant_config = Config(global_quant_config=QuantizationConfig(weight=quant_spec))

    EXCLUDE_LAYERS = ["lm_head", "*.gate", "*.shared_expert_gate"]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)

    with tempfile.TemporaryDirectory() as tmpdir:
        with torch.inference_mode():
            quant_model = quantize_model(quant_config, model_name=MODEL_DIR, multi_gpu=False)
            NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format=weight_format, pack_method="reorder")
            export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)

            model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
            model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config)

            quant_model(INPUT_IDS).to_tuple()

            with torch.no_grad():
                ref_outputs = quant_model(INPUT_IDS).to_tuple()

            for device in ["meta", torch_device]:
                config = AutoConfig.from_pretrained(MODEL_DIR)
                with torch.device(device):
                    original_model = AutoModelForCausalLM.from_config(config)

                importer = ModelImporter(tmpdir, saved_format="safetensors")
                q_model = importer.import_model_info(original_model)

                q_model = q_model.eval()

                for _, param in q_model.named_parameters():
                    assert param.device != "meta"
                for _, param in q_model.named_buffers():
                    assert param.device != "meta"

                if device == "meta":
                    q_model = q_model.to(torch_device)

                with torch.no_grad():
                    outputs = q_model(INPUT_IDS).to_tuple()

                for ref_output, output in zip(ref_outputs[0], outputs[0]):
                    if torch_device.type == "cpu":
                        assert torch.equal(ref_output, output)
                    else:
                        assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


@slow_test
@pytest.mark.parametrize("weight_format", [
    pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]
])
def test_awq_import(weight_format: str):
    '''
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      AWQ
    '''
    AWQ_CONFIG = AWQConfig(
        scaling_layers=[
            {
                "prev_op": "self_attn_layer_norm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn"
            },
            {
                "prev_op": "self_attn.v_proj",
                "layers": ["self_attn.out_proj"],
                "inp": "self_attn.out_proj"
            },
            {
                "prev_op": "final_layer_norm",
                "layers": ["fc1"],
                "inp": "fc1"
            },
            {
                "prev_op": "fc1",
                "layers": ["fc2"],
                "inp": "fc2"
            }
        ],
        model_decoder_layers="model.decoder.layers"
    )
    EXCLUDE_LAYERS = ["lm_head"]

    W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)
    quant_config = Config(global_quant_config=W_UINT4_PER_GROUP_CONFIG, algo_config=AWQ_CONFIG, exclude=EXCLUDE_LAYERS)
    with tempfile.TemporaryDirectory() as tmpdir:
        with torch.inference_mode():
            for custom_mode in ["awq", "quark"]:
                if custom_mode == "awq" and weight_format == "fake_quantized":
                    continue
                quant_model = quantize_model(quant_config, model_name=MODEL_DIR, multi_gpu=False)
                NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format=weight_format, pack_method="reorder")
                export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)

                model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
                model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config)

                quant_model(INPUT_IDS).to_tuple()

                with torch.no_grad():
                    ref_outputs = quant_model(INPUT_IDS).to_tuple()

                for device in ["meta", torch_device]:
                    config = AutoConfig.from_pretrained(MODEL_DIR)
                    with torch.device(device):
                        original_model = AutoModelForCausalLM.from_config(config)

                    importer = ModelImporter(tmpdir, saved_format="safetensors")
                    q_model = importer.import_model_info(original_model)

                    for _, param in q_model.named_parameters():
                        assert param.device != "meta"
                    for _, param in q_model.named_buffers():
                        assert param.device != "meta"

                    q_model = q_model.eval()

                    if device == "meta":
                        q_model = q_model.to(torch_device)

                    with torch.no_grad():
                        outputs = q_model(INPUT_IDS).to_tuple()
                    for ref_output, output in zip(ref_outputs[0], outputs[0]):
                        if torch_device.type == "cpu":
                            assert torch.equal(ref_output, output)
                        else:
                            assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


@pytest.mark.parametrize("weight_format", [
    pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]
])
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU with max abs diff ~0.1.
def test_fp8_inp_weight_out_import(weight_format: str):
    '''
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      FP8
    '''
    FP8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.fp8_e4m3,
                                           qscheme=QSchemeType.per_tensor,
                                           observer_cls=PerTensorMinMaxObserver,
                                           is_dynamic=False)

    EXCLUDE_LAYERS = ["lm_head"]

    W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                                            weight=FP8_PER_TENSOR_SPEC,
                                                            output_tensors=FP8_PER_TENSOR_SPEC)
    quant_config = Config(global_quant_config=W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG, exclude=EXCLUDE_LAYERS)
    with tempfile.TemporaryDirectory() as tmpdir:
        with torch.inference_mode():
            quant_model = quantize_model(quant_config, model_name=MODEL_DIR, multi_gpu=False)
            MERGE_REALQ_CONFIG = JsonExporterConfig(weight_merge_groups=[["*up_proj", "*gate_proj"], ["*q_proj", "*k_proj", "*v_proj"]], weight_format=weight_format, pack_method="reorder")
            export_config = ExporterConfig(json_export_config=MERGE_REALQ_CONFIG)

            model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
            model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config)

            quant_model(INPUT_IDS).to_tuple()

            with torch.no_grad():
                ref_outputs = quant_model(INPUT_IDS).to_tuple()

            for device in ["meta", torch_device]:
                config = AutoConfig.from_pretrained(MODEL_DIR)
                with torch.device(device):
                    original_model = AutoModelForCausalLM.from_config(config)

                importer = ModelImporter(tmpdir, saved_format="safetensors")
                q_model = importer.import_model_info(original_model)
                for _, param in q_model.named_parameters():
                    assert param.device != "meta"
                for _, param in q_model.named_buffers():
                    assert param.device != "meta"

                q_model = q_model.eval()
                # scaled_mm has exclusive tests, only naive mode is tested here.
                quark.torch.export.nn.modules.qparamslinear.SCALED_MM_AVAILABLE_DEV = None

                if device == "meta":
                    q_model = q_model.to(torch_device)

                with torch.no_grad():
                    outputs = q_model(INPUT_IDS).to_tuple()

                for ref_output, output in zip(ref_outputs[0], outputs[0]):
                    if torch_device.type == "cpu":
                        assert torch.equal(ref_output, output)
                    else:
                        assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.

@pytest.mark.parametrize("kv_cache_group", [
    pytest.param(kv_cache_group, id=str(kv_cache_group)) for kv_cache_group in [[], ["*k_proj", "*v_proj"]]
])
@pytest.mark.parametrize("weight_format", ["real_quantized", "fake_quantized"])
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU.
def test_fp8_kv_cache_import(kv_cache_group: List[str], weight_format: str):
    '''
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      FP8 KV_Cache_FP8
    '''

    FP8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.fp8_e4m3,
                                           qscheme=QSchemeType.per_tensor,
                                           observer_cls=PerTensorMinMaxObserver,
                                           is_dynamic=False)

    EXCLUDE_LAYERS = ["lm_head"]

    W_FP8_A_FP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                                       weight=FP8_PER_TENSOR_SPEC)
    kv_cache_quant_config = {}
    if len(kv_cache_group) > 0:
        layer_quant_config = {
                "*v_proj":
                QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                   weight=FP8_PER_TENSOR_SPEC,
                                   output_tensors=FP8_PER_TENSOR_SPEC),
                "*k_proj":
                QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                   weight=FP8_PER_TENSOR_SPEC,
                                   output_tensors=FP8_PER_TENSOR_SPEC),
        }
        kv_cache_quant_config = layer_quant_config.copy()
    else:
        layer_quant_config = {}

    quant_config = Config(global_quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG,
                          layer_quant_config=layer_quant_config,
                          kv_cache_quant_config=kv_cache_quant_config,
                          exclude=EXCLUDE_LAYERS)
    with torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=MODEL_DIR, multi_gpu=False)

        NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format=weight_format, kv_cache_group=kv_cache_group, pack_method="reorder")
        export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)

        with tempfile.TemporaryDirectory() as tmpdir:
            for custom_mode in ["quark", "fp8"]:
                if custom_mode == "fp8" and weight_format == "fake_quantized":
                    continue

                model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
                model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config, custom_mode=custom_mode)

                ref_outputs = quant_model(INPUT_IDS).to_tuple()

                for device in ["meta", torch_device]:
                    config = AutoConfig.from_pretrained(MODEL_DIR)
                    with torch.device(device):
                        original_model = AutoModelForCausalLM.from_config(config)

                    original_model.eval()

                    importer = ModelImporter(tmpdir, saved_format="safetensors")
                    q_model = importer.import_model_info(original_model)
                    # scaled_mm has exclusive tests, only naive mode is tested here.
                    quark.torch.export.nn.modules.qparamslinear.SCALED_MM_AVAILABLE_DEV = None

                    for _, param in q_model.named_parameters():
                        assert param.device != "meta"
                    for _, param in q_model.named_buffers():
                        assert param.device != "meta"

                    if device == "meta":
                        q_model = q_model.to(torch_device)

                    outputs = q_model(INPUT_IDS).to_tuple()

                    for ref_output, output in zip(ref_outputs[0], outputs[0]):
                        # When `kv_cache_group` is specified, a single scale is used for key/value linear after export,
                        # which does not match the behavior prior to export.
                        if len(kv_cache_group) == 0:
                            if torch_device.type == "cpu":
                                assert torch.equal(ref_output, output)
                            else:
                                assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.

@slow_test
@pytest.mark.parametrize("weight_format", [
    pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]
])
def test_gptq_import(weight_format: str):
    '''
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      GPTQ
    '''
    W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)
    quant_config = Config(global_quant_config=W_UINT4_PER_GROUP_CONFIG, algo_config=GPTQ_CONFIG)

    EXCLUDE_LAYERS = ["lm_head", "*.gate", "*.shared_expert_gate"]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)

    with tempfile.TemporaryDirectory() as tmpdir:
        with torch.inference_mode():
            quant_model = quantize_model(quant_config, model_name=MODEL_DIR, multi_gpu=False)
            NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format=weight_format, pack_method="reorder")
            export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)

            model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
            model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config)

            quant_model(INPUT_IDS).to_tuple()

            with torch.no_grad():
                ref_outputs = quant_model(INPUT_IDS).to_tuple()

            for device in ["meta", torch_device]:
                config = AutoConfig.from_pretrained(MODEL_DIR)
                with torch.device(device):
                    original_model = AutoModelForCausalLM.from_config(config)

                importer = ModelImporter(tmpdir, saved_format="safetensors")
                q_model = importer.import_model_info(original_model)

                for _, param in q_model.named_parameters():
                    assert param.device != "meta"
                for _, param in q_model.named_buffers():
                    assert param.device != "meta"

                q_model = q_model.eval()

                if device == "meta":
                    q_model = q_model.to(torch_device)

                with torch.no_grad():
                    outputs = q_model(INPUT_IDS).to_tuple()

                for ref_output, output in zip(ref_outputs[0], outputs[0]):
                    if torch_device.type == "cpu":
                        assert torch.equal(ref_output, output)
                    else:
                        assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.

@use_temporary_directory
def test_non_quantized_import(tmpdir: str):
    with torch.inference_mode():
        non_quantized_model = AutoModelForCausalLM.from_pretrained("haoyang-amd/non_quantized_model")
        non_quantized_model.save_pretrained(tmpdir)
        config = AutoConfig.from_pretrained(MODEL_DIR)
        with torch.device("meta"):
            original_model = AutoModelForCausalLM.from_config(config)

        importer = ModelImporter(model_info_dir=tmpdir)
        model_config = importer.get_model_config()
        model_state_dict = _load_weights_from_safetensors(tmpdir)
        model_config.config_dict["quantization_config"] = None
        _ = importer._build_model(original_model, model_config, model_state_dict)

@slow_test
@use_temporary_directory
def test_dbrx_import(tmpdir: str):
    '''
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      FP8
    '''
    FP8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.fp8_e4m3,
                                           qscheme=QSchemeType.per_tensor,
                                           observer_cls=PerTensorMinMaxObserver,
                                           is_dynamic=False)

    EXCLUDE_LAYERS = ["lm_head"]

    W_FP8_A_FP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                                       weight=FP8_PER_TENSOR_SPEC,
                                                       output_tensors=FP8_PER_TENSOR_SPEC)
    layer_quant_config = {
            "*Wqkv":
            QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                               weight=FP8_PER_TENSOR_SPEC,
                               output_tensors=FP8_PER_TENSOR_SPEC),
    }

    quant_config = Config(global_quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG,
                          layer_quant_config=layer_quant_config,
                          exclude=EXCLUDE_LAYERS)


    with torch.inference_mode():
        quantizer = ModelQuantizer(quant_config)
        dbrx_id = "haoyang-amd/dbrx_layer1"

        config = AutoConfig.from_pretrained(dbrx_id, trust_remote_code=True)
        original_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        original_model.eval()
        original_model = original_model.to(torch_device)
        # dbrx replace
        if isinstance(original_model, DbrxForCausalLM):
            for name, module in original_model.named_modules(remove_duplicate=False):
                if isinstance(module, DbrxExperts):
                    new_experts = DbrxExperts_.from_float(module)
                    set_op_by_name(original_model, name, new_experts)
                    print(f"module {name} has been replaced")
        # Get dataloader, if multi_gpu, give the first layer's device
        calib_dataloader = get_dataloader()

        quant_model = quantizer.quantize_model(original_model, calib_dataloader)
        # Inference with quantized model
        for i in calib_dataloader:
            quant_model(i)
        quant_model = quantizer.freeze(quant_model)

        NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized", kv_cache_group=["*Wqkv"], pack_method="reorder")

        export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)

        model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
        model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config)
        gc.collect()
        torch.cuda.empty_cache()

        # TODO: trust_remote_code=True is dangerous, to be removed.
        original_model_config = AutoConfig.from_pretrained(dbrx_id, trust_remote_code=True)
        original_model2 = AutoModelForCausalLM.from_config(original_model_config, trust_remote_code=True)
        original_model2.eval()
        original_model2 = original_model2.to(torch_device)

        # dbrx replace
        if isinstance(original_model2, DbrxForCausalLM):
            for name, module in original_model2.named_modules(remove_duplicate=False):
                if isinstance(module, DbrxExperts):
                    new_experts = DbrxExperts_.from_float(module)
                    set_op_by_name(original_model2, name, new_experts)

        importer = ModelImporter(tmpdir, saved_format="safetensors")
        q_model = importer.import_model_info(original_model2)
        q_model = q_model.to(torch_device)


@slow_test
@use_temporary_directory
def test_custom_mode_export(tmpdir: str):
    # AWQ model.
    W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)
    AWQ_CONFIG = AWQConfig(
        scaling_layers=[
            {
                "prev_op": "self_attn_layer_norm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn"
            },
            {
                "prev_op": "self_attn.v_proj",
                "layers": ["self_attn.out_proj"],
                "inp": "self_attn.out_proj"
            },
            {
                "prev_op": "final_layer_norm",
                "layers": ["fc1"],
                "inp": "fc1"
            },
            {
                "prev_op": "fc1",
                "layers": ["fc2"],
                "inp": "fc2"
            }
        ],
        model_decoder_layers="model.decoder.layers"
    )

    EXCLUDE_LAYERS = ["lm_head"]
    quant_config = Config(global_quant_config=W_UINT4_PER_GROUP_CONFIG, algo_config=AWQ_CONFIG, exclude=EXCLUDE_LAYERS)
    quant_model = quantize_model(quant_config, model_name=MODEL_DIR, multi_gpu=False)
    export_config = ExporterConfig(json_export_config=JsonExporterConfig(weight_format="real_quantized"))

    model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
    model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config, custom_mode="awq")

    config = AutoConfig.from_pretrained(tmpdir)
    assert config.quantization_config["quant_method"] == "awq"

    delete_directory_content(tmpdir)

    # FP8 model.
    FP8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.fp8_e4m3,
                                           qscheme=QSchemeType.per_tensor,
                                           observer_cls=PerTensorMinMaxObserver,
                                           is_dynamic=False)


    W_FP8_A_FP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                                       weight=FP8_PER_TENSOR_SPEC)
    quant_config = Config(global_quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG, exclude=EXCLUDE_LAYERS)
    quant_model = quantize_model(quant_config, model_name=MODEL_DIR, multi_gpu=False)
    export_config = ExporterConfig(json_export_config=JsonExporterConfig(weight_format="real_quantized"))

    model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
    model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config, custom_mode="fp8")

    config = AutoConfig.from_pretrained(tmpdir)
    assert config.quantization_config["quant_method"] == "fp8"
    assert "activation_scheme" in config.quantization_config
    assert "kv_cache_scheme" in config.quantization_config
    assert "export" in config.quantization_config

def test_multi_safetensors_load():
    script_dir = os.path.dirname(__file__)

    safetensors_dir = os.path.join(script_dir, "simple_model")
    model_state_dict = _load_weights_from_safetensors(safetensors_dir)
    assert(len(model_state_dict) == 10)

@pytest.mark.parametrize("model_id", [
    pytest.param(model_id, id=model_id) for model_id in ["amd/Meta-Llama-3.1-8B-Instruct-FP8-KV", "amd-quark/dummy-config-awq"]
])
def test_custom_config_remap(model_id: str):
    hf_config = AutoConfig.from_pretrained(model_id)

    _ = QuantConfigParser.from_custom_config(hf_config.quantization_config, is_bias_quantized=False, is_kv_cache=False, kv_layers_name=None)

@pytest.mark.parametrize("model_id", [
    pytest.param(model_id, id=model_id) for model_id in ["amd-quark/quark-legacy-awq"]
])
def test_custom_import(model_id: str):
    # TODO: enhance this test over all quark previous versions.

    if "awq" not in model_id:
        original_model_id = "fxmarty/tiny-llama-fast-tokenizer"
    else:
        original_model_id = "fxmarty/small-llama-testing"

    model = AutoModelForCausalLM.from_pretrained(original_model_id, torch_dtype="auto", attn_implementation="eager")

    model.eval()
    model = model.to(torch_device)

    custom_model_path = huggingface_hub.snapshot_download(repo_id=model_id, repo_type="model")

    importer = ModelImporter(custom_model_path, saved_format="safetensors")
    _ = importer.import_model_info(model)


# TODO: When the import function of mx is complete, this function should be upgraded to "import"
@pytest.mark.parametrize("weight_format", [
    pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]
])
@retry_flaky_test()
def test_wmxfp4_afp8_export(weight_format: str):
    '''
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      wmxfp4_afp8
    '''
    FP8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.fp8_e4m3,
                                           qscheme=QSchemeType.per_tensor,
                                           observer_cls=PerTensorMinMaxObserver,
                                           is_dynamic=False)
    MXFP4_PER_GROUP_SYM_SPEC = OCP_MXFP4Spec(is_dynamic=False).to_quantization_spec()
    EXCLUDE_LAYERS = ["lm_head"]

    W_MXFP4_A_FP8_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=MXFP4_PER_GROUP_SYM_SPEC, input_tensors=FP8_PER_TENSOR_SPEC)
    quant_config = Config(global_quant_config=W_MXFP4_A_FP8_PER_GROUP_SYM_CONFIG, exclude=EXCLUDE_LAYERS)
    with tempfile.TemporaryDirectory() as tmpdir:
        with torch.inference_mode():
            quant_model = quantize_model(quant_config, model_name=MODEL_DIR, multi_gpu=False)
            MERGE_REALQ_CONFIG = JsonExporterConfig(weight_merge_groups=[["*up_proj", "*gate_proj"], ["*q_proj", "*k_proj", "*v_proj"]], weight_format=weight_format, pack_method="reorder")
            export_config = ExporterConfig(json_export_config=MERGE_REALQ_CONFIG)

            model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
            model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config)

            quant_model(INPUT_IDS).to_tuple()

            with torch.no_grad():
                ref_outputs = quant_model(INPUT_IDS).to_tuple()


@pytest.mark.parametrize("weight_format", [
    pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]
])
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU with max abs diff ~0.1.
def test_wfp4_afp8_import(weight_format: str):
    '''
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      wfp4_afp8
    '''
    MX_SEPERATED_FP8_E4M3_PER_GROUP_SYM_SPEC = OCP_MXFP8E4M3Spec(ch_axis=-1,
                                                                 is_dynamic=True).to_quantization_spec()
    MX_SEPERATED_FP4_PER_GROUP_SYM_SPEC = OCP_MXFP4Spec(ch_axis=-1,
                                                        is_dynamic=False).to_quantization_spec()

    EXCLUDE_LAYERS = ["lm_head"]
    W_MXFP4_A_MXFP8 = QuantizationConfig(input_tensors=MX_SEPERATED_FP8_E4M3_PER_GROUP_SYM_SPEC,
                                         weight=MX_SEPERATED_FP4_PER_GROUP_SYM_SPEC)

    quant_config = Config(global_quant_config=W_MXFP4_A_MXFP8, exclude=EXCLUDE_LAYERS)
    with tempfile.TemporaryDirectory() as tmpdir:
        with torch.inference_mode():
            quant_model = quantize_model(quant_config, model_name=MODEL_DIR, multi_gpu=False)
            NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format=weight_format, pack_method="reorder")
            export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)

            model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
            model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config)

            quant_model(INPUT_IDS).to_tuple()

            with torch.no_grad():
                ref_outputs = quant_model(INPUT_IDS).to_tuple()

            for device in ["meta", torch_device]:
                config = AutoConfig.from_pretrained(MODEL_DIR)
                with torch.device(device):
                    original_model = AutoModelForCausalLM.from_config(config)

                importer = ModelImporter(tmpdir, saved_format="safetensors")
                q_model = importer.import_model_info(original_model)

                q_model = q_model.eval()

                if device == "meta":
                    q_model = q_model.to(torch_device)

                with torch.no_grad():
                    outputs = q_model(INPUT_IDS).to_tuple()

                for ref_output, output in zip(ref_outputs[0], outputs[0]):
                    if torch_device.type == "cpu":
                        assert torch.equal(ref_output, output)
                    else:
                        assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


@pytest.mark.parametrize("weight_format", [
    pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]
])
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU with max abs diff ~0.1.
def test_wfp6_e2m3_afp6_e2m3_import(weight_format: str):
    '''
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      wfp6_e2m3_afp6_e2m3
    '''

    def FP6_E2M3_PER_GROUP_SYM_SPEC(group_size, scale_format="e8m0", scale_calculation_mode="even", is_dynamic=True):
        return FP6E2M3PerGroupSpec(ch_axis=-1,
                                   group_size=group_size,
                                   scale_format=scale_format,
                                   scale_calculation_mode=scale_calculation_mode,
                                   is_dynamic=is_dynamic).to_quantization_spec()
    EXCLUDE_LAYERS = ["lm_head"]
    global_quant_config = QuantizationConfig(input_tensors=FP6_E2M3_PER_GROUP_SYM_SPEC(32, "e8m0", "even", True),
                                             weight=FP6_E2M3_PER_GROUP_SYM_SPEC(32, "e8m0", "even", False))
    quant_config = Config(global_quant_config=global_quant_config, exclude=EXCLUDE_LAYERS)

    with tempfile.TemporaryDirectory() as tmpdir:
        with torch.inference_mode():
            quant_model = quantize_model(quant_config, model_name=MODEL_DIR, multi_gpu=False)
            NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format=weight_format, pack_method="reorder")
            export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)

            model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
            model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config)

            quant_model(INPUT_IDS).to_tuple()

            with torch.no_grad():
                ref_outputs = quant_model(INPUT_IDS).to_tuple()

            for device in ["meta", torch_device]:
                config = AutoConfig.from_pretrained(MODEL_DIR)
                with torch.device(device):
                    original_model = AutoModelForCausalLM.from_config(config)

                importer = ModelImporter(tmpdir, saved_format="safetensors")
                q_model = importer.import_model_info(original_model)

                q_model = q_model.eval()

                if device == "meta":
                    q_model = q_model.to(torch_device)

                with torch.no_grad():
                    outputs = q_model(INPUT_IDS).to_tuple()

                for ref_output, output in zip(ref_outputs[0], outputs[0]):
                    if torch_device.type == "cpu":
                        assert torch.equal(ref_output, output)
                    else:
                        assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


@pytest.mark.parametrize("weight_format", [
    pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]
])
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU with max abs diff ~0.1.
def test_wfp6_e3m2_afp6_e3m2_import(weight_format: str):
    '''
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      wfp6_e3m2_afp6_e3m2
    '''

    def FP6_E3M2_PER_GROUP_SYM_SPEC(group_size, scale_format="e8m0", scale_calculation_mode="even", is_dynamic=True):
        return FP6E3M2PerGroupSpec(ch_axis=-1,
                                   group_size=group_size,
                                   scale_format=scale_format,
                                   scale_calculation_mode=scale_calculation_mode,
                                   is_dynamic=is_dynamic).to_quantization_spec()
    EXCLUDE_LAYERS = ["lm_head"]
    global_quant_config = QuantizationConfig(input_tensors=FP6_E3M2_PER_GROUP_SYM_SPEC(32, "e8m0", "even", True),
                                             weight=FP6_E3M2_PER_GROUP_SYM_SPEC(32, "e8m0", "even", False))
    quant_config = Config(global_quant_config=global_quant_config, exclude=EXCLUDE_LAYERS)

    with tempfile.TemporaryDirectory() as tmpdir:
        with torch.inference_mode():
            quant_model = quantize_model(quant_config, model_name=MODEL_DIR, multi_gpu=False)
            NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format=weight_format, pack_method="reorder")
            export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)

            model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
            model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config)

            quant_model(INPUT_IDS).to_tuple()

            with torch.no_grad():
                ref_outputs = quant_model(INPUT_IDS).to_tuple()

            for device in ["meta", torch_device]:
                config = AutoConfig.from_pretrained(MODEL_DIR)
                with torch.device(device):
                    original_model = AutoModelForCausalLM.from_config(config)

                importer = ModelImporter(tmpdir, saved_format="safetensors")
                q_model = importer.import_model_info(original_model)

                q_model = q_model.eval()

                if device == "meta":
                    q_model = q_model.to(torch_device)

                with torch.no_grad():
                    outputs = q_model(INPUT_IDS).to_tuple()

                for ref_output, output in zip(ref_outputs[0], outputs[0]):
                    if torch_device.type == "cpu":
                        assert torch.equal(ref_output, output)
                    else:
                        assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


@pytest.mark.parametrize("weight_format", [
    pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]
])
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU with max abs diff ~0.1.
def test_fp4_per_group_fp8_per_tensor_scale_export_import(weight_format: str):
    '''
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      w_fp4_per_group_fp8_per_tensor_scale_a_fp4_per_group_fp8_per_tensor_scale
    '''
    FP4_PER_GROUP_FP8_PER_TENSOR_SCALE_SPEC = ScaleQuantSpec(
        first_stage=FP4PerGroupSpec(group_size=16, is_dynamic=False),
        second_stage=FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False)
    ).to_quantization_spec()

    FP4_PER_GROUP_FP8_PER_TENSOR_SCALE_SPEC_DYNAMIC = ScaleQuantSpec(
        first_stage=FP4PerGroupSpec(group_size=16, is_dynamic=True),
        second_stage=FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=True)
    ).to_quantization_spec()

    EXCLUDE_LAYERS = ["lm_head"]
    W_FP4_A_FP4_SCALE_FP8_CONFIG = QuantizationConfig(input_tensors=FP4_PER_GROUP_FP8_PER_TENSOR_SCALE_SPEC_DYNAMIC,
                                                      weight=FP4_PER_GROUP_FP8_PER_TENSOR_SCALE_SPEC)

    quant_config = Config(global_quant_config=W_FP4_A_FP4_SCALE_FP8_CONFIG, exclude=EXCLUDE_LAYERS)
    with tempfile.TemporaryDirectory() as tmpdir:
        with torch.inference_mode():
            quant_model = quantize_model(quant_config, model_name=MODEL_DIR, multi_gpu=False)
            NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format=weight_format, pack_method="reorder")
            export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)

            model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
            model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config)

            quant_model(INPUT_IDS).to_tuple()

            with torch.no_grad():
                ref_outputs = quant_model(INPUT_IDS).to_tuple()

            for device in ["meta", torch_device]:
                config = AutoConfig.from_pretrained(MODEL_DIR)
                with torch.device(device):
                    original_model = AutoModelForCausalLM.from_config(config)

                importer = ModelImporter(tmpdir, saved_format="safetensors")
                q_model = importer.import_model_info(original_model)

                q_model = q_model.eval()

                if device == "meta":
                    q_model = q_model.to(torch_device)

                with torch.no_grad():
                    outputs = q_model(INPUT_IDS).to_tuple()

                for ref_output, output in zip(ref_outputs[0], outputs[0]):
                    if torch_device.type == "cpu":
                        assert torch.equal(ref_output, output)
                    else:
                        assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


@pytest.mark.parametrize("weight_format", [
    pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]
])
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU with max abs diff ~0.1.
def test_wfp8_int4perchannel_afp8_import(weight_format: str):
    '''
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      wfp8_int4perchannel_afp8
    '''
    FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(observer_method="min_max",
                                               is_dynamic=False).to_quantization_spec()
    INT4_PER_CHANNEL_SPEC = Int4PerChannelSpec(symmetric=True,
                                               scale_type="float",
                                               round_method="half_even",
                                               ch_axis=0,
                                               is_dynamic=False).to_quantization_spec()
    FP8_INT4_PER_CHANNEL_SPEC = [FP8_PER_TENSOR_SPEC, INT4_PER_CHANNEL_SPEC]

    EXCLUDE_LAYERS = ["lm_head"]
    W_FP8_A_INT4_PER_CHANNEL = QuantizationConfig(weight=FP8_INT4_PER_CHANNEL_SPEC, input_tensors=FP8_PER_TENSOR_SPEC)

    quant_config = Config(global_quant_config=W_FP8_A_INT4_PER_CHANNEL, exclude=EXCLUDE_LAYERS)
    with tempfile.TemporaryDirectory() as tmpdir:
        with torch.inference_mode():
            quant_model = quantize_model(quant_config, model_name=MODEL_DIR, multi_gpu=False)
            NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format=weight_format, pack_method="reorder")
            export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)

            model_exporter = ModelExporter(config=export_config, export_dir=tmpdir)
            model_exporter.export_safetensors_model(model=quant_model, quant_config=quant_config)

            quant_model(INPUT_IDS).to_tuple()

            with torch.no_grad():
                ref_outputs = quant_model(INPUT_IDS).to_tuple()

            for device in ["meta", torch_device]:
                config = AutoConfig.from_pretrained(MODEL_DIR)
                with torch.device(device):
                    original_model = AutoModelForCausalLM.from_config(config)

                importer = ModelImporter(tmpdir, saved_format="safetensors")
                q_model = importer.import_model_info(original_model)

                q_model = q_model.eval()

                if device == "meta":
                    q_model = q_model.to(torch_device)

                with torch.no_grad():
                    outputs = q_model(INPUT_IDS).to_tuple()

                for ref_output, output in zip(ref_outputs[0], outputs[0]):
                    if torch_device.type == "cpu":
                        assert torch.equal(ref_output, output)
                    else:
                        assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.
