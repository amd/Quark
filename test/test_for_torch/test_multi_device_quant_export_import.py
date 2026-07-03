#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import tempfile

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.common.utils.testing_utils import (
    PatchEverywhere,
    require_torch_higher_or_equal,
    retry_flaky_test,
    skip_if_no_gpu,
    torch_device,
)
from quark.torch import ModelQuantizer, export_safetensors, import_model_from_safetensors
from quark.torch.export.safetensors import _load_weights_from_safetensors
from quark.torch.export.utils import _fix_loaded_weights_key_mismatch
from quark.torch.quantization.config.config import GPTQConfig, QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import (
    PerChannelMinMaxObserver,
    PerGroupMinMaxObserver,
    PerTensorMinMaxObserver,
)
from quark.torch.utils import QPARAMSLINEAR_OVERRIDES_STATE_DICT

MODEL_DIR = "facebook/opt-125m"
torch.manual_seed(42)
INPUT_IDS = torch.randint(0, 1024, (1, 10)).to(torch_device)

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

UINT4_PER_GROUP_ASYM_SPEC = QTensorConfig(
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


def get_accelerate_cpu_model(model_name="facebook/opt-125m"):
    max_memory = {"cpu": "100GB"}
    # cpu and one gpu; This one is the best for opt
    if torch.cuda.is_available():
        first_gpu = 0
        max_memory[first_gpu] = "0GB"
        if torch.cuda.device_count() > 1:
            for i in range(1, torch.cuda.device_count()):
                max_memory[i] = "0GB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto", max_memory=max_memory, trust_remote_code=True
    )
    # transformers 5.x (huggingface/transformers#43019) omits `hf_device_map`
    # when `device_map="auto"` lands on a single device (e.g. CPU-only here);
    # read it defensively so the helper still works in that case.
    print(getattr(model, "hf_device_map", None))
    return model


def get_multi_device_model(model_name="facebook/opt-125m"):
    max_memory = {"cpu": "100GB"}
    # cpu and one gpu; This one is the best for opt
    if torch.cuda.is_available():
        first_gpu = 0
        max_memory[first_gpu] = "0.15GB"
        if torch.cuda.device_count() > 1:
            for i in range(1, torch.cuda.device_count()):
                max_memory[i] = "0GB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto", max_memory=max_memory, trust_remote_code=True
    )
    print(getattr(model, "hf_device_map", None))
    return model


def quantize_model(
    quant_config, model_name="facebook/opt-125m", multi_gpu=False, multi_device=True, device_map: str | None = "auto"
):
    # Get quantizer
    if not multi_device:
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
    else:
        max_memory = {"cpu": "100GB"}
        # cpu and one gpu; This one is the best for opt
        if torch.cuda.is_available():
            first_gpu = 0
            max_memory[first_gpu] = "0.2GB"
            if torch.cuda.device_count() > 1:
                for i in range(1, torch.cuda.device_count()):
                    max_memory[i] = "0GB"

        quantizer = ModelQuantizer(quant_config, multi_device=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype="auto", max_memory=max_memory, trust_remote_code=True
        )
        print(getattr(model, "hf_device_map", None))

    # Get dataloader, if multi_gpu, give the first layer's device
    calib_dataloader = get_dataloader(model_name, model.device)

    quant_model = quantizer.quantize_model(model, calib_dataloader)
    # Inference with quantized model
    for i in calib_dataloader:
        quant_model(i)

    return quant_model


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize(
    "qscheme",
    [
        pytest.param(qscheme, id=str(qscheme))
        for qscheme in [QSchemeType.per_tensor, QSchemeType.per_channel, QSchemeType.per_group]
    ],
)
@pytest.mark.parametrize("weight_format", ["real_quantized", "fake_quantized"])
def test_int4_import_export(qscheme: QSchemeType, weight_format: str):
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
    quant_spec = QTensorConfig(
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

    quant_config = QConfig(global_quant_config=QLayerConfig(weight=quant_spec))

    quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False, device_map=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        original_model = get_multi_device_model(model_id)

        # Used for later comparison.
        weight_dict = _load_weights_from_safetensors(tmpdir)
        if not QPARAMSLINEAR_OVERRIDES_STATE_DICT:
            weight_dict = _fix_loaded_weights_key_mismatch(
                weight_dict, weight_format=weight_format, custom_mode="quark"
            )

        q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=True)
        q_model = q_model.eval()

        q_model_state_dict = q_model.state_dict()

        if weight_format == "real_quantized":
            for key in weight_dict:
                assert weight_dict[key].dtype == q_model_state_dict[key].dtype
                assert weight_dict[key].shape == q_model_state_dict[key].shape

        with torch.no_grad():
            outputs = q_model(INPUT_IDS).to_tuple()

        for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
            if torch_device.type == "cpu":
                assert torch.equal(ref_output, output)
            else:
                assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize(
    "qscheme",
    [
        pytest.param(qscheme, id=str(qscheme))
        for qscheme in [QSchemeType.per_tensor, QSchemeType.per_channel, QSchemeType.per_group]
    ],
)
@pytest.mark.parametrize("weight_format", ["real_quantized", "fake_quantized"])
def test_int8_import_export(qscheme: QSchemeType, weight_format: str):
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
    quant_spec = QTensorConfig(
        dtype=Dtype.int8,
        qscheme=qscheme,
        observer_cls=qscheme_to_observer[qscheme],
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
        ch_axis=qscheme_to_ch_axis[qscheme],
        group_size=8 if qscheme == QSchemeType.per_group else None,
    )

    quant_config = QConfig(global_quant_config=QLayerConfig(weight=quant_spec))

    quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False, device_map=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        original_model = get_multi_device_model(model_id)

        # Used for later comparison.
        weight_dict = _load_weights_from_safetensors(tmpdir)

        if not QPARAMSLINEAR_OVERRIDES_STATE_DICT:
            weight_dict = _fix_loaded_weights_key_mismatch(
                weight_dict, weight_format=weight_format, custom_mode="quark"
            )

        q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=True)
        q_model = q_model.eval()

        q_model_state_dict = q_model.state_dict()

        if weight_format == "real_quantized":
            for key in weight_dict:
                assert weight_dict[key].dtype == q_model_state_dict[key].dtype
                assert weight_dict[key].shape == q_model_state_dict[key].shape

        with torch.no_grad():
            outputs = q_model(INPUT_IDS).to_tuple()

        for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
            if torch_device.type == "cpu":
                assert torch.equal(ref_output, output)
            else:
                assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize(
    "kv_cache_group,kv_cache_post_rope",
    [
        pytest.param([], False, id="no-kv"),
        pytest.param(["*k_proj", "*v_proj"], False, id="kv-pre-rope"),
        pytest.param(["*k_proj", "*v_proj"], True, id="kv-post-rope"),
    ],
)
@pytest.mark.parametrize("weight_format", ["real_quantized", "fake_quantized"])
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU.
def test_fp8_kv_cache_import(kv_cache_group: list[str], kv_cache_post_rope: bool, weight_format: str):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      FP8 KV_Cache_FP8
    """

    FP8_PER_TENSOR_SPEC = QTensorConfig(
        dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False
    )

    EXCLUDE_LAYERS = ["lm_head"]

    W_FP8_A_FP8_PER_TENSOR_CONFIG = QLayerConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC)
    kv_cache_quant_config = {}
    if len(kv_cache_group) > 0:
        layer_quant_config = {
            "*v_proj": QLayerConfig(
                input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
            ),
            "*k_proj": QLayerConfig(
                input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
            ),
        }
        kv_cache_quant_config = layer_quant_config.copy()
    else:
        layer_quant_config = {}

    quant_config = QConfig(
        global_quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG,
        layer_quant_config=layer_quant_config,
        kv_cache_quant_config=kv_cache_quant_config,
        kv_cache_group=kv_cache_group,
        exclude=EXCLUDE_LAYERS,
    )
    if len(kv_cache_group) > 0:
        quant_config.kv_cache_post_rope = kv_cache_post_rope  # type: ignore[attr-defined]
    with torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=MODEL_DIR, multi_gpu=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            for custom_mode in ["quark", "fp8"]:
                if custom_mode == "fp8" and weight_format == "fake_quantized":
                    continue

                export_safetensors(
                    model=quant_model,
                    output_dir=tmpdir,
                    custom_mode=custom_mode,
                    weight_format=weight_format,
                    pack_method="reorder",
                )

                ref_outputs = quant_model(INPUT_IDS).to_tuple()

                original_model = get_multi_device_model()

                original_model.eval()

                q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=True)

                # scaled_mm has exclusive tests, only naive mode is tested here.
                with PatchEverywhere("SCALED_MM_AVAILABLE_DEV", None, module_name_prefix="quark"):
                    outputs = q_model(INPUT_IDS).to_tuple()

                    for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                        # When `kv_cache_group` is specified, a single scale is used for key/value linear after export,
                        # which does not match the behavior prior to export.
                        if len(kv_cache_group) == 0:
                            if torch_device.type == "cpu":
                                assert torch.equal(ref_output, output)
                            else:
                                assert torch.allclose(
                                    output, ref_output, atol=1e-4
                                )  # This one appears not to be flaky.


# For torch requirement, refer to /pull/2529#issuecomment-235620
# Requires a GPU: the OOM hint is only raised when accelerate offloads to CPU
# because GPU memory is insufficient. On CPU-only the model is single-device,
# so transformers 5.x sets no `hf_device_map`, no offload occurs, and the
# product correctly does not raise (there is no OOM to flag).
@skip_if_no_gpu
@require_torch_higher_or_equal("2.6")
def test_OOM():
    model_id = "facebook/opt-125m"
    accelerate_cpu_model = get_multi_device_model(model_id)
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
    qscheme = QSchemeType.per_tensor
    quant_spec = QTensorConfig(
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

    quant_config = QConfig(global_quant_config=QLayerConfig(weight=quant_spec))
    quantizer = ModelQuantizer(quant_config, multi_device=False)
    # Get dataloader, if multi_gpu, give the first layer's device
    calib_dataloader = get_dataloader(model_id, accelerate_cpu_model.device)
    try:
        _ = quantizer.quantize_model(accelerate_cpu_model, calib_dataloader)
    except MemoryError as e:
        assert (
            "Out of memory. The available GPU memory is insufficient to load the entire model. You can try adding '--multi_device' "
            in str(e)
        )
    else:
        raise ValueError("Expected ValueError was not raised.")


if __name__ == "__main__":
    test_int4_import_export(qscheme=QSchemeType.per_tensor, weight_format="real_quantized")
