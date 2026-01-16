#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from unittest.mock import patch

import pytest
import torch
from gguf import GGMLQuantizationType

from quark.shares.utils.testing_utils import require_torch_higher_or_equal, torch_device, use_temporary_directory
from quark.torch import export_gguf
from quark.torch.export.gguf_export.tensor_convert import convert_from_gguf, convert_to_gguf
from quark.torch.quantization import QConfig, QLayerConfig, Uint4PerGroupSpec


# flake8: noqa: C901
def generate_test_data():
    test_data = {
        "inpt": torch.tensor(
            [
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    160,
                    55,
                    64,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    224,
                    13,
                    61,
                    0,
                    32,
                    61,
                    61,
                    0,
                    32,
                    61,
                    60,
                    0,
                    192,
                    212,
                    189,
                    0,
                    0,
                    0,
                    0,
                    0,
                    32,
                    61,
                    60,
                    0,
                    0,
                    0,
                    0,
                    0,
                    224,
                    13,
                    189,
                    0,
                    32,
                    189,
                    188,
                    0,
                    224,
                    13,
                    61,
                    0,
                    32,
                    61,
                    61,
                    0,
                    224,
                    141,
                    61,
                    0,
                    32,
                    61,
                    188,
                    0,
                    32,
                    61,
                    61,
                    0,
                    0,
                    0,
                    0,
                    0,
                    32,
                    61,
                    188,
                    0,
                    32,
                    61,
                    60,
                    0,
                    224,
                    13,
                    189,
                    0,
                    224,
                    141,
                    189,
                    0,
                    0,
                    0,
                    0,
                    0,
                    224,
                    141,
                    61,
                    0,
                    32,
                    61,
                    60,
                    0,
                    32,
                    189,
                    188,
                    0,
                    32,
                    61,
                    60,
                    0,
                    32,
                    61,
                    60,
                    0,
                    32,
                    189,
                    188,
                    0,
                    224,
                    141,
                    61,
                    0,
                    32,
                    189,
                    60,
                    0,
                    32,
                    61,
                    60,
                    0,
                    224,
                    13,
                    189,
                    0,
                    32,
                    61,
                    188,
                    0,
                    32,
                    61,
                    60,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    32,
                    159,
                    192,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    96,
                    229,
                    188,
                    0,
                    96,
                    143,
                    189,
                    0,
                    96,
                    101,
                    188,
                    0,
                    0,
                    172,
                    61,
                    0,
                    96,
                    101,
                    60,
                    0,
                    96,
                    101,
                    60,
                    0,
                    96,
                    229,
                    188,
                    0,
                    96,
                    229,
                    60,
                    0,
                    96,
                    229,
                    60,
                    0,
                    96,
                    101,
                    189,
                    0,
                    96,
                    101,
                    189,
                    0,
                    96,
                    101,
                    189,
                    0,
                    96,
                    101,
                    188,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    44,
                    189,
                    0,
                    96,
                    229,
                    188,
                    0,
                    96,
                    143,
                    61,
                    0,
                    96,
                    229,
                    61,
                    0,
                    96,
                    101,
                    188,
                    0,
                    0,
                    44,
                    189,
                    0,
                    96,
                    143,
                    189,
                    0,
                    96,
                    229,
                    60,
                    0,
                    96,
                    229,
                    188,
                    0,
                    96,
                    229,
                    188,
                    0,
                    0,
                    44,
                    189,
                    0,
                    192,
                    200,
                    189,
                    0,
                    96,
                    101,
                    189,
                    0,
                    96,
                    101,
                    188,
                    0,
                    96,
                    229,
                    60,
                    0,
                    96,
                    101,
                    60,
                    0,
                    96,
                    229,
                    188,
                ],
            ],
            dtype=torch.uint8,
        ),
        "scale": torch.tensor([[0, 224, 67, 62, 0, 32, 61, 60], [0, 192, 169, 62, 0, 96, 101, 60]], dtype=torch.uint8),
        "zero_point": torch.tensor([[0, 0, 0, 0, 0, 0, 16, 65], [0, 0, 112, 65, 0, 0, 224, 64]], dtype=torch.uint8),
        "result": torch.tensor(
            [
                [
                    31,
                    50,
                    0,
                    128,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    240,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    233,
                    33,
                    166,
                    174,
                    172,
                    109,
                    58,
                    144,
                    249,
                    170,
                    121,
                    166,
                    167,
                    124,
                    253,
                    191,
                    168,
                    109,
                    137,
                    168,
                ],
                [
                    78,
                    53,
                    249,
                    196,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255,
                    15,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255,
                    43,
                    35,
                    70,
                    174,
                    85,
                    194,
                    246,
                    109,
                    72,
                    40,
                    149,
                    89,
                    89,
                    67,
                    3,
                    51,
                    102,
                    151,
                    135,
                    84,
                ],
            ],
            dtype=torch.uint8,
        ),
    }
    test_data["inpt"] = test_data["inpt"].view(torch.float32)
    test_data["scale"] = test_data["scale"].view(torch.float32)
    test_data["zero_point"] = test_data["zero_point"].view(torch.float32)
    return test_data


def test_convert_to_gguf():
    test_data = generate_test_data()
    inpt = test_data["inpt"]
    scale = test_data["scale"]
    zero_point = test_data["zero_point"]
    result = test_data["result"]

    # test whether convertion works
    converted_tensor = convert_to_gguf(
        inpt=inpt, scale=scale, zero_point=zero_point, gguf_type=GGMLQuantizationType.Q4_1
    )

    assert result.shape == converted_tensor.shape
    assert (result - converted_tensor).abs().max() == 0

    # test whether gguf type check works
    with pytest.raises(TypeError):
        converted_tensor = convert_to_gguf(
            inpt=inpt, scale=scale, zero_point=zero_point, gguf_type=GGMLQuantizationType.Q8_K
        )

    # test whether tensor shape check works
    with pytest.raises(AssertionError):
        converted_tensor = convert_to_gguf(
            inpt=inpt[:, :-1], scale=scale, zero_point=zero_point, gguf_type=GGMLQuantizationType.Q4_1
        )

        converted_tensor = convert_to_gguf(
            inpt=inpt, scale=scale[:, :-1], zero_point=zero_point, gguf_type=GGMLQuantizationType.Q4_1
        )

        converted_tensor = convert_to_gguf(
            inpt=inpt, scale=scale, zero_point=zero_point[:, :-1], gguf_type=GGMLQuantizationType.Q4_1
        )


def test_convert_from_gguf():
    test_data = generate_test_data()
    inpt = test_data["inpt"]
    scale = test_data["scale"]
    zero_point = test_data["zero_point"]
    result = test_data["result"]

    # test whether convertion works
    converted_inpt, converted_scale, converted_zp = convert_from_gguf(inpt=result, gguf_type=GGMLQuantizationType.Q4_1)

    assert scale.shape == converted_scale.shape
    assert (scale - converted_scale).abs().max() == 0

    assert zero_point.shape == converted_zp.shape
    assert (zero_point - converted_zp.round()).abs().max() == 0

    block_size = 32
    origin_shape = inpt.shape
    inpt = inpt.reshape(-1, block_size)
    scale = scale.reshape(-1, 1)
    zero_point = zero_point.reshape(-1, 1)

    min_val = -scale * zero_point
    scale_inverse = (1 / scale).masked_fill(scale == 0.0, 0)
    quant_inpt = torch.round((inpt - min_val) * scale_inverse).to(torch.uint8).clamp(0, 15)
    dequant_inpt = quant_inpt.to(torch.float32) * scale.to(torch.float32) + min_val.to(torch.float32)
    dequant_inpt = dequant_inpt.reshape(origin_shape)

    assert dequant_inpt.shape == converted_inpt.shape
    assert torch.allclose(dequant_inpt, converted_inpt, atol=1e-3, rtol=1e-3)

    # assert inpt.shape == converted_inpt.shape
    # assert (inpt - converted_inpt).abs().max() == 0

    # test whether gguf type check works
    with pytest.raises(TypeError):
        converted_inpt, converted_scale, converted_zp = convert_from_gguf(
            inpt=result, gguf_type=GGMLQuantizationType.Q8_K
        )

    # test whether tensor shape check works
    with pytest.raises(AssertionError):
        converted_inpt, converted_scale, converted_zp = convert_from_gguf(
            inpt=result[:, :-1], gguf_type=GGMLQuantizationType.Q4_1
        )


def get_dataloader(model_name="facebook/opt-125m", device=torch_device):
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"].to(device))
    return calib_dataloader


def quantize_model(quant_config, model_name="facebook/opt-125m", multi_gpu=False):
    from transformers import AutoModelForCausalLM

    from quark.torch import ModelQuantizer

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
    # Inference with quantized model
    for i in calib_dataloader:
        quant_model(i)

    return quant_model


@require_torch_higher_or_equal("2.6")
@use_temporary_directory
def test_gguf_export(tmpdir: str):
    """
    Test Features:
        Export Format:            GGUF
    """
    UINT4_PER_GROUP_ASYM_SPEC = Uint4PerGroupSpec(ch_axis=1, is_dynamic=False, group_size=32).to_quantization_spec()
    W_UINT4_PER_GROUP_CONFIG = QLayerConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)
    quant_config = QConfig(global_quant_config=W_UINT4_PER_GROUP_CONFIG)
    with patch("quark.torch.export.api.convert_exported_model_to_gguf"), torch.inference_mode():
        for multi_gpu in [False]:
            model = quantize_model(quant_config, model_name="facebook/opt-125m", multi_gpu=multi_gpu)
            export_gguf(model, output_dir=tmpdir, model_type="llama", tokenizer_path="facebook/opt-125m")
