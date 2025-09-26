#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import tempfile
from dataclasses import replace
from typing import List, Optional

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.shares.utils.testing_utils import (
    require_torch_higher_or_equal,
    retry_flaky_test,
    torch_device,
    use_temporary_directory,
)
from quark.testing import slow_test
from quark.torch import ModelExporter, ModelImporter, ModelQuantizer
from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
from quark.torch.export.utils import _build_quantized_model
from quark.torch.quantization.config.config import AWQConfig, Config, GPTQConfig, QuantizationConfig, QuantizationSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import (
    PerChannelMinMaxObserver,
    PerGroupMinMaxObserver,
    PerTensorMinMaxObserver,
)
from quark.torch.utils import getattr_recursive, setattr_recursive

model_dir = "facebook/opt-125m"
torch.manual_seed(42)
input_ids = torch.randint(0, 1024, (1, 10)).to(torch_device)

GPTQ_CONFIG = GPTQConfig(
    model_decoder_layers="model.decoder.layers",
    inside_layer_modules=[
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_proj",
        "self_attn.out_proj",
        "fc1",
        "fc2",
    ],
)

UINT4_PER_GROUP_ASYM_SPEC = QuantizationSpec(
    dtype=Dtype.uint4,
    observer_cls=PerGroupMinMaxObserver,
    symmetric=False,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    qscheme=QSchemeType.per_group,
    ch_axis=1,
    is_dynamic=False,
    group_size=32,
)


def get_dataloader(model_name="facebook/opt-125m", device=torch_device):
    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"].to(device))
    return calib_dataloader


def quantize_model(quant_config, model_name="facebook/opt-125m", multi_gpu=False, device_map: str | None = "auto"):
    # Get quantizer
    quantizer = ModelQuantizer(quant_config)

    if multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, torch_dtype="auto", trust_remote_code=True
        )
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

    model = quantizer.freeze(model)

    return quant_model


@slow_test
@pytest.mark.parametrize(
    "weight_format",
    [pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]],
)
@pytest.mark.parametrize(
    "qscheme",
    [
        pytest.param(qscheme, id=str(qscheme))
        for qscheme in [QSchemeType.per_tensor, QSchemeType.per_channel, QSchemeType.per_group]
    ],
)
def test_int4_import(qscheme: QSchemeType, weight_format: str):
    """
    Test Features:
        Import Format:            Json-pth
        Quantization Method:      int4_per_tensor_per_channel_per_group
    """
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
    quant_spec = QuantizationSpec(
        dtype=Dtype.int4,
        qscheme=qscheme,
        observer_cls=qscheme_to_observer[qscheme],
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
        ch_axis=qscheme_to_ch_axis[qscheme],
        group_size=8 if qscheme == QSchemeType.per_group else None,
    )

    quant_config = Config(global_quant_config=QuantizationConfig(weight=quant_spec))

    quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False, device_map=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_config = ExporterConfig(json_export_config=JsonExporterConfig(weight_format=weight_format))
        exporter = ModelExporter(config=export_config, export_dir=tmpdir)
        exporter.export_quark_model(quant_model, quant_config=quant_config)
        quant_model(input_ids).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(input_ids).to_tuple()

        original_model = AutoModelForCausalLM.from_pretrained(model_id)
        q_model = AutoModelForCausalLM.from_pretrained(model_id)
        # Transformers is bit bugged with `with torch.device("meta"):` so not using it here.
        if weight_format == "real_quantized":
            q_model = q_model.to("meta")
        q_model = q_model.eval()

        importer = ModelImporter(tmpdir)
        model_config = importer.get_model_config()
        model_state_dict = importer.get_model_state_dict()
        q_model = _build_quantized_model(q_model, model_config, model_state_dict)
        q_model_state_dict = q_model.state_dict()

        if weight_format == "real_quantized":
            for key in model_state_dict.keys():
                assert q_model_state_dict[key].device.type == "meta"
                assert model_state_dict[key].dtype == q_model_state_dict[key].dtype
                # fake_quantized model loads without reasoning about shape for a while
                assert model_state_dict[key].shape == q_model_state_dict[key].shape

        q_model.load_state_dict(model_state_dict, assign=True)
        # inv_freq is a non persistent buffer, hence not overriden with the `load_state_dict` above.
        for name, param in q_model.named_parameters():
            if param.device.type == "meta":
                assert "inv_freq" in name
                setattr_recursive(q_model, name, getattr_recursive(original_model, name))

        for name, param in q_model.named_buffers():
            if param.device.type == "meta":
                assert "inv_freq" in name
                setattr_recursive(q_model, name, getattr_recursive(original_model, name))
        q_model = q_model.to(torch_device)

        with torch.no_grad():
            outputs = q_model(input_ids).to_tuple()

        for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
            if torch_device.type == "cpu":
                assert torch.equal(ref_output, output)
            else:
                assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


@slow_test
@use_temporary_directory
def test_awq_import(tmpdir: str):
    """
    Test Features:
        Import Format:            Json-pth
        Quantization Method:      AWQ
    """
    AWQ_CONFIG = AWQConfig(
        scaling_layers=[
            {
                "prev_op": "self_attn_layer_norm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.out_proj"], "inp": "self_attn.out_proj"},
            {"prev_op": "final_layer_norm", "layers": ["fc1"], "inp": "fc1"},
            {"prev_op": "fc1", "layers": ["fc2"], "inp": "fc2"},
        ],
        model_decoder_layers="model.decoder.layers",
    )
    EXCLUDE_LAYERS = ["lm_head"]

    W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)
    quant_config = Config(
        global_quant_config=W_UINT4_PER_GROUP_CONFIG, algo_config=[AWQ_CONFIG], exclude=EXCLUDE_LAYERS
    )
    with torch.inference_mode():
        q_model = quantize_model(quant_config, model_name=model_dir, multi_gpu=False)
        ref_outputs = q_model(input_ids).to_tuple()
        NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized", pack_method="reorder")
        config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)
        exporter = ModelExporter(config=config, export_dir=tmpdir)
        exporter.export_quark_model(q_model, quant_config=quant_config)
        q_model(input_ids).to_tuple()

        model = AutoModelForCausalLM.from_pretrained(model_dir)
        model.eval()
        model = model.to(torch_device)

        importer = ModelImporter(model_info_dir=tmpdir)
        reload_model = importer.import_model_info(model)

        outputs = reload_model(input_ids).to_tuple()

        for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
            assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@use_temporary_directory
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU with max abs diff ~0.1.
def test_fp8_import(tmpdir: str):
    """
    Test Features:
        Import Format:            Json-pth
        Quantization Method:      FP8
    """
    FP8_PER_TENSOR_SPEC = QuantizationSpec(
        dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False
    )

    EXCLUDE_LAYERS = ["lm_head"]

    W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG = QuantizationConfig(
        input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
    )
    quant_config = Config(global_quant_config=W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG, exclude=EXCLUDE_LAYERS)
    with torch.inference_mode():
        q_model = quantize_model(quant_config, model_name=model_dir, multi_gpu=False)
        ref_outputs = q_model(input_ids).to_tuple()
        NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized", pack_method="reorder")
        config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)
        exporter = ModelExporter(config=config, export_dir=tmpdir)
        exporter.export_quark_model(q_model, quant_config=quant_config)
        q_model(input_ids).to_tuple()

        model = AutoModelForCausalLM.from_pretrained(model_dir)
        model.eval()
        model = model.to(torch_device)

        importer = ModelImporter(model_info_dir=tmpdir)
        reload_model = importer.import_model_info(model)

        outputs = reload_model(input_ids).to_tuple()

        for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
            if torch_device.type == "cpu":
                assert torch.equal(ref_output, output)
            else:
                assert torch.allclose(output, ref_output, atol=1e-4)


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize(
    "kv_cache_group",
    [pytest.param(kv_cache_group, id=str(kv_cache_group)) for kv_cache_group in [[], ["*k_proj", "*v_proj"]]],
)
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU.
def test_fp8_kv_cache_import(kv_cache_group: list[str]):
    """
    Test Features:
        Import Format:            Json-pth
        Quantization Method:      FP8 KV_Cache_FP8
    """
    FP8_PER_TENSOR_SPEC = QuantizationSpec(
        dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False
    )

    EXCLUDE_LAYERS = ["lm_head"]

    W_FP8_A_FP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC)
    layer_quant_config = {
        "*v_proj": QuantizationConfig(
            input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
        ),
        "*k_proj": QuantizationConfig(
            input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
        ),
    }

    quant_config = Config(
        global_quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG, layer_quant_config=layer_quant_config, exclude=EXCLUDE_LAYERS
    )

    with torch.inference_mode():
        q_model = quantize_model(quant_config, model_name=model_dir, multi_gpu=False)

        NO_MERGE_REALQ_CONFIG = JsonExporterConfig(
            weight_format="real_quantized", kv_cache_group=kv_cache_group, pack_method="reorder"
        )
        config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(config=config, export_dir=tmpdir)
            exporter.export_quark_model(q_model, quant_config=quant_config)
            ref_outputs = q_model(input_ids).to_tuple()

            model = AutoModelForCausalLM.from_pretrained(model_dir)
            model.eval()
            model = model.to(torch_device)
            importer = ModelImporter(model_info_dir=tmpdir)
            reload_model = importer.import_model_info(model)

            outputs = reload_model(input_ids).to_tuple()

            for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                # When `kv_cache_group` is specified, a single scale is used for key/value linear after export,
                # which does not match the behavior prior to export.
                if len(kv_cache_group) == 0:
                    if torch_device.type == "cpu":
                        assert torch.equal(ref_output, output)
                    else:
                        assert torch.allclose(output, ref_output, atol=1e-4)


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize(
    "weight_format",
    [pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]],
)
@pytest.mark.parametrize(
    "kv_cache_group",
    [pytest.param(kv_cache_group, id=str(kv_cache_group)) for kv_cache_group in [[], ["*k_proj", "*v_proj"]]],
)
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU.
def test_fp8_kv_cache_attn_import(kv_cache_group: list[str], weight_format: str):
    """
    Test Features:
        Import Format:            Json-pth
        Quantization Method:      FP8 KV_Cache_FP8 Attn_FP8
    """
    FP8_PER_TENSOR_SPEC = QuantizationSpec(
        dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False
    )

    EXCLUDE_LAYERS = ["lm_head"]

    W_FP8_A_FP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC)
    layer_quant_config = {
        "*v_proj": QuantizationConfig(
            input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
        ),
        "*k_proj": QuantizationConfig(
            input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
        ),
        "*q_proj": QuantizationConfig(
            input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
        ),
    }

    quant_config = Config(
        global_quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG,
        layer_quant_config=layer_quant_config,
        softmax_quant_spec=FP8_PER_TENSOR_SPEC,
        exclude=EXCLUDE_LAYERS,
    )

    with torch.inference_mode():
        q_model = quantize_model(quant_config, model_name=model_dir, multi_gpu=False)

        NO_MERGE_REALQ_CONFIG = JsonExporterConfig(
            weight_format=weight_format, kv_cache_group=kv_cache_group, pack_method="reorder"
        )
        config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(config=config, export_dir=tmpdir)
            exporter.export_quark_model(q_model, quant_config=quant_config)
            ref_outputs = q_model(input_ids).to_tuple()

            model = AutoModelForCausalLM.from_pretrained(model_dir)
            model.eval()
            model = model.to(torch_device)
            importer = ModelImporter(model_info_dir=tmpdir)
            reload_model = importer.import_model_info(model)

            outputs = reload_model(input_ids).to_tuple()

            for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                # When `kv_cache_group` is specified, a single scale is used for key/value linear after export,
                # which does not match the behavior prior to export.
                if len(kv_cache_group) == 0:
                    if torch_device.type == "cpu":
                        assert torch.equal(ref_output, output)
                    else:
                        assert torch.allclose(output, ref_output, atol=1e-4)


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize(
    "weight_format",
    [pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]],
)
def test_int8_import(weight_format: str):
    """
    Test Features:
        Import Format:            Json-pth
        Quantization Method:      INT8
    """
    INT8_PER_TENSOR_SPEC = QuantizationSpec(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )

    INT8_PER_TENSOR_CONFIG = QuantizationConfig(weight=INT8_PER_TENSOR_SPEC, bias=INT8_PER_TENSOR_SPEC)
    quant_config = Config(global_quant_config=INT8_PER_TENSOR_CONFIG)

    EXCLUDE_LAYERS = ["lm_head", "*.gate", "*.shared_expert_gate"]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)
    with tempfile.TemporaryDirectory() as tmpdir:
        with torch.inference_mode():
            q_model = quantize_model(quant_config, model_name=model_dir, multi_gpu=False)
            NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format=weight_format, pack_method="reorder")
            config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)
            exporter = ModelExporter(config=config, export_dir=tmpdir)
            exporter.export_quark_model(q_model, quant_config=quant_config)
            ref_outputs = q_model(input_ids).to_tuple()

            model = AutoModelForCausalLM.from_pretrained(model_dir)
            model.eval()
            model = model.to(torch_device)

            importer = ModelImporter(model_info_dir=tmpdir)
            reload_model = importer.import_model_info(model)

            outputs = reload_model(input_ids).to_tuple()

            for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                if torch_device.type == "cpu":
                    assert torch.equal(ref_output, output)
                else:
                    assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


@slow_test
@use_temporary_directory
def test_gptq_import(tmpdir: str):
    """
    Test Features:
        Import Format:            Json-pth
        Quantization Method:      GPTQ
    """
    W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)
    quant_config = Config(global_quant_config=W_UINT4_PER_GROUP_CONFIG, algo_config=[GPTQ_CONFIG])

    EXCLUDE_LAYERS = ["lm_head", "*.gate", "*.shared_expert_gate"]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)

    with torch.inference_mode():
        q_model = quantize_model(quant_config, model_name=model_dir, multi_gpu=False)
        NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized", pack_method="reorder")
        config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG)
        exporter = ModelExporter(config=config, export_dir=tmpdir)
        exporter.export_quark_model(q_model, quant_config=quant_config)

        ref_outputs = q_model(input_ids).to_tuple()

        model = AutoModelForCausalLM.from_pretrained(model_dir)
        model.eval()
        model = model.to(torch_device)

        importer = ModelImporter(model_info_dir=tmpdir)
        reload_model = importer.import_model_info(model)
        outputs = reload_model(input_ids).to_tuple()

        for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
            assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.
