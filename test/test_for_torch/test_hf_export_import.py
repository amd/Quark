#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import copy
import gc
import json
import os
import re
import tempfile
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import huggingface_hub
import pytest
import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

from quark.common.utils.import_utils import (
    is_transformers_available,
    is_transformers_version_higher_or_equal,
    is_triton_available,
)
from quark.common.utils.testing_utils import (
    PatchEverywhere,
    local_test_only,
    require_accelerate,
    require_torch_higher_or_equal,
    require_torch_multi_gpu,
    retry_flaky_test,
    skip_torch_version,
    slow_test,
    torch_device,
    use_temporary_directory,
)
from quark.torch import LLMTemplate, ModelQuantizer, export_safetensors, import_model_from_safetensors
from quark.torch.export.main_export.quant_config_parser import QuantConfigParser
from quark.torch.export.main_import.pretrained_config import PretrainedConfig
from quark.torch.export.safetensors import _load_weights_from_safetensors, export_hf_model
from quark.torch.export.utils import _build_quantized_model, _fix_loaded_weights_key_mismatch, preprocess_import_info
from quark.torch.quantization import (
    FP4PerGroupSpec,
    FP6E2M3PerGroupSpec,
    FP6E3M2PerGroupSpec,
    FP8E4M3PerTensorSpec,
    Int4PerChannelSpec,
    OCP_MXFP4Spec,
    OCP_MXFP8E4M3Spec,
    ScaleQuantSpec,
)
from quark.torch.quantization.cache_integration import QuarkQuantizedCache
from quark.torch.quantization.config.config import AWQConfig, GPTQConfig, QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.inverse_quantizer import is_prequantized_linear
from quark.torch.quantization.nn.modules.mixin import QuantMixin
from quark.torch.quantization.observer.observer import (
    PerChannelMinMaxObserver,
    PerGroupMinMaxObserver,
    PerTensorMinMaxObserver,
)
from quark.torch.utils import QPARAMSLINEAR_OVERRIDES_STATE_DICT
from quark.torch.utils.llm import preprocess_for_quantization
from quark.torch.utils.llm.model_preparation import get_model
from quark.torch.utils.llm.preprocessing import maybe_load_preprocessors, maybe_save_preprocessors

if is_transformers_version_higher_or_equal("5.0"):
    from transformers.initialization import no_init_weights
else:
    from transformers.modeling_utils import no_init_weights

TEST_MODELS = {"opt-125m:": "facebook/opt-125m", "qwen3-tiny": "amd-quark/tiny-random-qwen3_moe"}

MODEL_PREFIX = {"facebook/opt-125m": "model.decoder.layers", "amd-quark/tiny-random-qwen3_moe": "model.layers"}


torch.manual_seed(42)
INPUT_IDS = torch.randint(0, 1024, (1, 10)).to(torch_device)

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


def init_model(config, device, torch_dtype=None):
    if device == "meta":
        context = ExitStack()
        context.enter_context(no_init_weights())
        context.enter_context(init_empty_weights())
    else:
        context = torch.device(device)

    with context:
        original_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)

    return original_model


def get_dataloader(model_name="facebook/opt-125m", device=torch_device):
    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"].to(device))
    return calib_dataloader


def quantize_model(
    quant_config,
    model_name="facebook/opt-125m",
    multi_gpu=False,
    device_map: str | None = "auto",
    torch_dtype: str | None = "auto",
):
    # Get quantizer
    quantizer = ModelQuantizer(quant_config)

    if multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, torch_dtype="auto", trust_remote_code=True
        )
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        model.eval()
        model = model.to(torch_device)
    # Get dataloader, if multi_gpu, give the first layer's device
    calib_dataloader = get_dataloader(model_name, model.device)

    preprocess_for_quantization(model)

    quant_model = quantizer.quantize_model(model, calib_dataloader)
    # Inference with quantized model
    for i in calib_dataloader:
        quant_model(i)
    quant_model = quantizer.freeze(quant_model)

    return quant_model


def assert_direct_gates_are_excluded(quant_model: torch.nn.Module) -> None:
    """Verify direct MoE gates remain float linears with upstream state-dict names."""
    gates = [(name, module) for name, module in quant_model.named_modules() if name.endswith(".gate")]
    if not gates:
        assert getattr(quant_model.config, "model_type", None) != "qwen3_moe"
        return

    assert all(not isinstance(module, QuantMixin) for _, module in gates)
    state_dict_keys = quant_model.state_dict().keys()
    assert not any("gate.linear" in key for key in state_dict_keys)


@require_accelerate
@require_torch_multi_gpu
@pytest.mark.parametrize("weight_format", ["real_quantized", "fake_quantized"])
@pytest.mark.parametrize(
    "model_id", [pytest.param(model_name, id=f"model_id:{key}") for key, model_name in TEST_MODELS.items()]
)
def test_load_multi_device(weight_format: str, model_id: str):
    INT8_PER_TENSER_SPEC = QTensorConfig(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )

    INT8_PER_TENSOR_CONFIG = QLayerConfig(
        weight=INT8_PER_TENSER_SPEC,
        input_tensors=INT8_PER_TENSER_SPEC,
        output_tensors=INT8_PER_TENSER_SPEC,
        bias=INT8_PER_TENSER_SPEC,
    )
    quant_config = QConfig(global_quant_config=INT8_PER_TENSOR_CONFIG)

    EXCLUDE_LAYERS = ["lm_head", "*.gate", "*.shared_expert_gate"]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)

    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False)
        assert_direct_gates_are_excluded(quant_model)
        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        device_map = {
            "model.decoder.embed_tokens": "cuda:0",
            "model.decoder.embed_positions": "cuda:0",
            "model.decoder.final_layer_norm": "cuda:0",
            MODEL_PREFIX[model_id]: "cuda:1",
            "lm_head": "cuda:0",
            "model.norm": "cuda:1",
            "model.embed_tokens": "cuda:0",
        }

        original_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, torch_dtype="auto")

        q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)
        q_model = q_model.eval()

        with torch.no_grad():
            outputs = q_model(INPUT_IDS).to_tuple()

        for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
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


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize(
    "qscheme",
    [
        pytest.param(qscheme, id=str(qscheme))
        for qscheme in [QSchemeType.per_tensor, QSchemeType.per_channel, QSchemeType.per_group]
    ],
)
@pytest.mark.parametrize("exclude", [True, False])
@pytest.mark.parametrize("weight_format", ["real_quantized", "fake_quantized"])
def test_int2_import_export(qscheme: QSchemeType, weight_format: str, exclude: bool):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      INT4
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
    quant_spec = QTensorConfig(
        dtype=Dtype.int2,
        qscheme=qscheme,
        observer_cls=qscheme_to_observer[qscheme],
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
        ch_axis=qscheme_to_ch_axis[qscheme],
        group_size=8 if qscheme == QSchemeType.per_group else None,
    )

    if exclude:
        exclude = ["*lm_head*", "*embed_tokens*"]
    else:
        exclude = []

    quant_config = QConfig(global_quant_config=QLayerConfig(weight=quant_spec), exclude=exclude)

    quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False, device_map=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        for device in ["meta", torch_device]:
            config = AutoConfig.from_pretrained(model_id)

            original_model = init_model(config, device)

            # Used for later comparison.
            weight_dict = _load_weights_from_safetensors(tmpdir)

            if not QPARAMSLINEAR_OVERRIDES_STATE_DICT:
                weight_dict = _fix_loaded_weights_key_mismatch(
                    weight_dict, weight_format=weight_format, custom_mode="quark"
                )

            q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)
            q_model = q_model.eval()

            q_model_state_dict = q_model.state_dict()

            if weight_format == "real_quantized":
                for key in weight_dict:
                    assert weight_dict[key].dtype == q_model_state_dict[key].dtype
                    assert weight_dict[key].shape == q_model_state_dict[key].shape

            for _, param in q_model.named_parameters():
                assert param.device != "meta"
            for _, param in q_model.named_buffers():
                assert param.device != "meta"

            if device == "meta":
                q_model = q_model.to(torch_device)

            with torch.no_grad():
                outputs = q_model(INPUT_IDS).to_tuple()

            for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                if torch_device.type == "cpu":
                    assert torch.equal(ref_output, output)
                else:
                    assert torch.allclose(output, ref_output, atol=1e-4)


@slow_test
@pytest.mark.parametrize(
    "qscheme",
    [
        pytest.param(qscheme, id=str(qscheme))
        for qscheme in [QSchemeType.per_tensor, QSchemeType.per_channel, QSchemeType.per_group]
    ],
)
@pytest.mark.parametrize("weight_format", ["real_quantized", "fake_quantized"])
def test_int4_import_export(qscheme: QSchemeType, weight_format: str):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      INT4
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

        for device in ["meta", torch_device]:
            config = AutoConfig.from_pretrained(model_id)

            original_model = init_model(config, device)

            # Used for later comparison.
            weight_dict = _load_weights_from_safetensors(tmpdir)

            if not QPARAMSLINEAR_OVERRIDES_STATE_DICT:
                weight_dict = _fix_loaded_weights_key_mismatch(
                    weight_dict, weight_format=weight_format, custom_mode="quark"
                )

            q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)
            q_model = q_model.eval()

            q_model_state_dict = q_model.state_dict()

            if weight_format == "real_quantized":
                for key in weight_dict:
                    assert weight_dict[key].dtype == q_model_state_dict[key].dtype
                    assert weight_dict[key].shape == q_model_state_dict[key].shape

            for _, param in q_model.named_parameters():
                assert param.device != "meta"
            for _, param in q_model.named_buffers():
                assert param.device != "meta"

            if device == "meta":
                q_model = q_model.to(torch_device)

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
@pytest.mark.parametrize(
    "model_id", [pytest.param(model_name, id=f"model_id:{key}") for key, model_name in TEST_MODELS.items()]
)
def test_int8_import_export(qscheme: QSchemeType, weight_format: str, model_id: str):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      INT8
    """
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

    EXCLUDE_LAYERS = ["lm_head", "*.gate", "*.shared_expert_gate"]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)

    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False)
        assert_direct_gates_are_excluded(quant_model)

        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

        # `quant_flow`/`gpu_resident_blocks` are a runtime execution strategy read off
        # `QConfig` in-memory; they must not leak into the exported `config.json`.
        exported_config = AutoConfig.from_pretrained(tmpdir)
        assert "quant_flow" not in exported_config.quantization_config
        assert "gpu_resident_blocks" not in exported_config.quantization_config

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        with safe_open(Path(tmpdir, "model.safetensors"), framework="pt") as f:
            checkpoint_keys = f.keys()

            assert f"{MODEL_PREFIX[model_id]}.1.self_attn.k_proj.weight_scale" in checkpoint_keys

        for device in ["meta", torch_device]:
            config = AutoConfig.from_pretrained(model_id)

            original_model = init_model(config, device)

            q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

            q_model = q_model.eval()

            for _, param in q_model.named_parameters():
                assert param.device != "meta"
            for _, param in q_model.named_buffers():
                assert param.device != "meta"

            if device == "meta":
                q_model = q_model.to(torch_device)

            with torch.no_grad():
                outputs = q_model(INPUT_IDS).to_tuple()

            for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                if torch_device.type == "cpu":
                    assert torch.equal(ref_output, output)
                else:
                    assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


@pytest.mark.parametrize("affected_version", [True, False], ids=["transformers>=5.6", "transformers<5.6"])
def test_awq_export_warns_about_corrupted_keys_only_on_affected_transformers(monkeypatch, affected_version: bool):
    """The AWQ export path must announce the known key-name corruption on transformers>=5.6 only.

    ``test_awq_import`` below is the only test that reaches this branch through a real export, but
    it is slow-gated and xfailed on the affected versions, so drive the helper directly.
    """
    from quark.torch.export import api as export_api

    emitted: list[str] = []
    monkeypatch.setattr(export_api.logger, "warning", lambda message, *args, **kwargs: emitted.append(message))
    monkeypatch.setattr(export_api, "is_transformers_version_higher_or_equal", lambda version: affected_version)

    export_api._warn_if_awq_export_keys_are_corrupted()

    if affected_version:
        assert len(emitted) == 1
        assert "qscales" in emitted[0] and "qqzeros" in emitted[0]
    else:
        assert emitted == []


@slow_test
@pytest.mark.parametrize(
    "weight_format",
    [pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]],
)
@pytest.mark.parametrize("custom_mode", [pytest.param(key, id=f"custom_mode:{key}") for key in ["awq", "quark"]])
def test_awq_import(weight_format: str, custom_mode: str):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      AWQ
    """
    # TODO: test MOE here.
    model_id = "facebook/opt-125m"

    # KNOWN REGRESSION (tracked), NOT a compatibility no-op: from transformers 5.6.0 the AWQ
    # real_quantized export produces a corrupted checkpoint. quark's anchored AWQ_SAVE_MAP save path
    # (apply_export_state_dict_mappings) is gated by QPARAMSLINEAR_OVERRIDES_STATE_DICT, which is False
    # for transformers>=4.57, so AWQ falls back to `_weight_conversions = QUARK_AWQ_WEIGHT_CONVERSIONS`.
    # transformers reverse-applies those unanchored renamings cumulatively, and 5.6.0 added
    # `weight_conversions[::-1]` in core_model_loading.revert_weight_conversion, which flips their order
    # so `weight`->`qweight` runs first; `qweight_quantizer.scale` then still matches the later
    # `weight_quantizer.scale` entry, yielding `qscales` (and `qqzeros`) instead of `scales`/`qzeros`.
    # Bisected against the released wheels: 5.5.0 is clean, 5.6.0 onwards is not.
    # The export path emits a runtime warning (see _warn_if_awq_export_keys_are_corrupted). This xfail
    # exists ONLY to keep the regression visible/tracked while the real fix (moving the AWQ save off the
    # reverse rename) is developed and validated on a GPU env; it must be removed once that fix lands.
    # Tracking: huggingface/transformers#46650 + quark AWQ-save follow-up.
    if custom_mode == "awq" and is_transformers_version_higher_or_equal("5.6.0"):
        pytest.xfail(
            "KNOWN REGRESSION (tracked): AWQ real_quantized export is corrupted on transformers>=5.6 "
            "(weight_quantizer scales/zero_points serialize as qscales/qqzeros). Remove this xfail once "
            "the AWQ save path is moved off the transformers reverse WeightRenaming. "
            "See huggingface/transformers#46650."
        )

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
        model_decoder_layers=MODEL_PREFIX[model_id],
    )
    EXCLUDE_LAYERS = ["lm_head"]

    W_UINT4_PER_GROUP_CONFIG = QLayerConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)
    quant_config = QConfig(
        global_quant_config=W_UINT4_PER_GROUP_CONFIG, algo_config=[AWQ_CONFIG], exclude=EXCLUDE_LAYERS
    )
    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        if custom_mode == "awq" and weight_format == "fake_quantized":
            pytest.skip("skip custom_mode=awq, weight_format=fake_quantized")
        quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False)

        export_safetensors(
            model=quant_model,
            output_dir=tmpdir,
            weight_format=weight_format,
            pack_method="reorder",
            custom_mode=custom_mode,
        )

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        for device in ["meta", torch_device]:
            config = AutoConfig.from_pretrained(model_id)

            original_model = init_model(config, device)

            with safe_open(Path(tmpdir, "model.safetensors"), framework="pt") as f:
                checkpoint_keys = list(f.keys())

                if custom_mode == "quark":
                    assert f"{MODEL_PREFIX[model_id]}.0.self_attn.k_proj.weight" in checkpoint_keys
                    assert f"{MODEL_PREFIX[model_id]}.0.self_attn.k_proj.weight_scale" in checkpoint_keys
                else:
                    assert f"{MODEL_PREFIX[model_id]}.0.self_attn.k_proj.qweight" in checkpoint_keys
                    assert f"{MODEL_PREFIX[model_id]}.0.self_attn.k_proj.scales" in checkpoint_keys
                    assert f"{MODEL_PREFIX[model_id]}.0.self_attn.k_proj.qzeros" in checkpoint_keys

            q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

            for _, param in q_model.named_parameters():
                assert param.device != "meta"
            for _, param in q_model.named_buffers():
                assert param.device != "meta"

            q_model = q_model.eval()

            if device == "meta":
                q_model = q_model.to(torch_device)

            with torch.no_grad():
                outputs = q_model(INPUT_IDS).to_tuple()
            for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                if torch_device.type == "cpu":
                    assert torch.equal(ref_output, output)
                else:
                    assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@skip_torch_version("2.8")
@pytest.mark.parametrize(
    "weight_format",
    [pytest.param(val, id=f"weight_format:{val}") for val in ["real_quantized", "fake_quantized"]],
)
@pytest.mark.parametrize(
    "torch_dtype",
    [pytest.param(val, id=f"torch_dtype:{val}") for val in [torch.float16, torch.bfloat16, torch.float32]],
)
@pytest.mark.parametrize(
    "model_id", [pytest.param(model_name, id=f"model_id:{key}") for key, model_name in TEST_MODELS.items()]
)
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU with max abs diff ~0.1.
def test_fp8_inp_weight_out_import(weight_format: str, torch_dtype: torch.dtype, model_id: str):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      FP8
    """

    FP8_PER_TENSOR_SPEC = QTensorConfig(
        dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False
    )

    EXCLUDE_LAYERS = ["lm_head"]

    W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG = QLayerConfig(
        input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
    )

    quant_config = QConfig(global_quant_config=W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG, exclude=EXCLUDE_LAYERS)
    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False, torch_dtype=torch_dtype)

        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        for device in ["meta", torch_device]:
            config = AutoConfig.from_pretrained(model_id)

            original_model = init_model(config, device, torch_dtype=torch_dtype)

            q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)
            for _, param in q_model.named_parameters():
                assert param.device != "meta"
            for _, param in q_model.named_buffers():
                assert param.device != "meta"

            q_model = q_model.eval()

            # scaled_mm has exclusive tests, only naive mode is tested here.
            with PatchEverywhere("SCALED_MM_AVAILABLE_DEV", None, module_name_prefix="quark"):
                if device == "meta":
                    q_model = q_model.to(torch_device)

                with torch.no_grad():
                    outputs = q_model(INPUT_IDS).to_tuple()

                for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                    if torch_device.type == "cpu":
                        assert torch.equal(ref_output, output)
                    else:
                        assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@skip_torch_version("2.8")
@pytest.mark.parametrize(
    "torch_dtype",
    [pytest.param(val, id=f"torch_dtype:{val}") for val in [torch.float16, torch.bfloat16, torch.float32]],
)
@pytest.mark.parametrize(
    "kv_cache_group,kv_cache_post_rope",
    [
        pytest.param([], False, id="no-kv"),
        pytest.param(["*k_proj", "*v_proj"], False, id="kv-pre-rope"),
        pytest.param(["*k_proj", "*v_proj"], True, id="kv-post-rope"),
    ],
)
@pytest.mark.parametrize("weight_format", ["real_quantized", "fake_quantized"])
@pytest.mark.parametrize(
    "model_id", [pytest.param(model_name, id=f"model_id:{key}") for key, model_name in TEST_MODELS.items()]
)
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU.
def test_fp8_kv_cache_import(
    kv_cache_group: list[str], kv_cache_post_rope: bool, weight_format: str, torch_dtype: torch.dtype, model_id: str
):
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
    # Toggle post-RoPE path only when kv cache is enabled
    if len(kv_cache_group) > 0:
        quant_config.kv_cache_post_rope = kv_cache_post_rope  # type: ignore[attr-defined]
    with torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False, torch_dtype=torch_dtype)

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

                for device in ["meta", torch_device]:
                    config = AutoConfig.from_pretrained(model_id)

                    original_model = init_model(config, device, torch_dtype=torch_dtype)

                    original_model.eval()

                    q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

                    # scaled_mm has exclusive tests, only naive mode is tested here.
                    with PatchEverywhere("SCALED_MM_AVAILABLE_DEV", None, module_name_prefix="quark"):
                        for _, param in q_model.named_parameters():
                            assert param.device != "meta"
                        for _, param in q_model.named_buffers():
                            assert param.device != "meta"

                        if device == "meta":
                            q_model = q_model.to(torch_device)

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


@pytest.mark.parametrize(
    "kv_layers_name",
    [
        pytest.param(["*k_proj", "*v_proj"], id="single-segment"),
        pytest.param(["*self_attn.k_proj", "*self_attn.v_proj"], id="multi-segment-qwen3_5"),
        pytest.param(["*language_model.*.k_proj", "*language_model.*.v_proj"], id="multi-segment-llama4"),
    ],
)
def test_preprocess_import_info_restores_kv_scale_for_multi_segment_kv_layers_name(kv_layers_name: list[str]):
    """
    Verify that ``preprocess_import_info`` reconstructs ``output_quantizer.scale`` keys correctly
    regardless of how many path segments the model's ``kv_layers_name`` pattern contains.

    Regression test for a bug where multi-segment patterns (e.g. ``*self_attn.k_proj`` used by
    qwen3_5, or ``*language_model.*.k_proj`` used by llama4) caused the reconstructed key to
    duplicate part of the checkpoint prefix (e.g. ``...self_attn.self_attn.k_proj.output_scale``
    instead of ``...self_attn.k_proj.output_scale``), so the real ``output_quantizer.scale``
    tensors were never populated and reload failed with missing keys.

    :param list[str] kv_layers_name: The k_proj/v_proj layer name patterns to test, taken from
        real model templates registered in ``quark.torch.quantization.config.template``.
    """
    prefix = "model.language_model.layers.3.self_attn."
    # k_scale and v_scale are exported as a single shared kv_scale (vLLM/HF FP8 convention), so
    # `preprocess_import_info` only reads the k_scale entry and copies it onto both k_proj and
    # v_proj output_scale keys, then drops the now-redundant v_scale entry.
    kv_scale = torch.tensor([1.0])
    model_state_dict = {
        prefix + "k_scale": kv_scale,
        prefix + "v_scale": torch.tensor([2.0]),
    }

    updated_state_dict, is_kv_cache, returned_kv_layers_name = preprocess_import_info(
        model_state_dict, is_kv_cache=False, kv_layers_name=kv_layers_name, custom_mode="fp8"
    )

    assert is_kv_cache
    assert returned_kv_layers_name == kv_layers_name
    assert prefix + "k_scale" not in updated_state_dict
    assert prefix + "v_scale" not in updated_state_dict
    assert torch.equal(updated_state_dict[prefix + "k_proj.output_scale"], kv_scale)
    assert torch.equal(updated_state_dict[prefix + "v_proj.output_scale"], kv_scale)


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize("weight_format", ["real_quantized", "fake_quantized"])
@pytest.mark.parametrize(
    "model_id", [pytest.param(model_name, id=f"model_id:{key}") for key, model_name in TEST_MODELS.items()]
)
@retry_flaky_test()
def test_kv_cache_post_rope_integration(weight_format: str, model_id: str):
    """
    Test Features:
        Specifically verify post-RoPE KV cache quantization integration.
        This test validates that:
        1. QuarkQuantizedCache is properly attached when kv_cache_post_rope=True
        2. Output quantizers are moved from k_proj/v_proj to cache level
        3. Cache quantization state is properly exported/imported
        4. Cache works correctly during inference with use_cache=True
    """
    FP8_PER_TENSOR_SPEC = QTensorConfig(
        dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False
    )

    EXCLUDE_LAYERS = ["lm_head"]

    W_FP8_A_FP8_PER_TENSOR_CONFIG = QLayerConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC)
    layer_quant_config = {
        "*v_proj": QLayerConfig(
            input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
        ),
        "*k_proj": QLayerConfig(
            input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
        ),
    }
    kv_cache_quant_config = layer_quant_config.copy()
    kv_cache_group = ["*k_proj", "*v_proj"]

    quant_config = QConfig(
        global_quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG,
        layer_quant_config=layer_quant_config,
        kv_cache_quant_config=kv_cache_quant_config,
        kv_cache_group=kv_cache_group,
        exclude=EXCLUDE_LAYERS,
    )
    # Enable post-RoPE quantization
    quant_config.kv_cache_post_rope = True  # type: ignore[attr-defined]

    with torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False)

        # ✅ VERIFY 1: QuarkQuantizedCache is attached
        assert hasattr(quant_model, "_quark_cache"), (
            "Model should have _quark_cache attribute when kv_cache_post_rope=True"
        )
        assert isinstance(quant_model._quark_cache, QuarkQuantizedCache), "Cache should be QuarkQuantizedCache instance"
        assert len(quant_model._quark_cache.quantized_layers) > 0, "Cache should have quantized layers configured"

        # ✅ VERIFY 2: k_proj/v_proj output quantizers are disabled (moved to cache)
        disabled_count = 0
        preserved_count = 0
        for name, module in quant_model.named_modules():
            if ("k_proj" in name or "v_proj" in name) and hasattr(module, "_output_quantizer"):
                # Output quantizer should be disabled
                assert module._output_quantizer is None, (
                    f"{name} output_quantizer should be None (disabled) with post-RoPE, "
                    "as quantization now happens in cache"
                )
                disabled_count += 1

                # But quantizer should be preserved for cache use
                if hasattr(module, "_quark_cache_output_quantizer"):
                    preserved_count += 1

        assert disabled_count > 0, "Should have found and disabled k_proj/v_proj output quantizers"
        assert preserved_count > 0, "Should have preserved quantizers for cache use"

        with tempfile.TemporaryDirectory() as tmpdir:
            export_safetensors(
                model=quant_model,
                output_dir=tmpdir,
                custom_mode="quark",
                weight_format=weight_format,
                pack_method="reorder",
            )

            # ✅ VERIFY 3: Cache quantization scales are exported to safetensors
            with safe_open(Path(tmpdir, "model.safetensors"), framework="pt") as f:
                checkpoint_keys = list(f.keys())

                # Look for output_scale keys (these are the cache quantization scales)
                output_scale_keys = [
                    k for k in checkpoint_keys if ".output_scale" in k and ("k_proj" in k or "v_proj" in k)
                ]

                assert len(output_scale_keys) > 0, (
                    f"Cache quantization scales should be exported with post-RoPE. "
                    f"Found keys: {[k for k in checkpoint_keys if 'output_scale' in k]}"
                )

            # Get reference outputs before import
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

            # ✅ VERIFY 4: Test with use_cache=True
            with torch.no_grad():
                cached_output = quant_model(INPUT_IDS, use_cache=True)
                assert cached_output.past_key_values is not None, "Should return past_key_values when use_cache=True"
                # Verify it's our QuarkQuantizedCache
                assert isinstance(cached_output.past_key_values, QuarkQuantizedCache), (
                    "past_key_values should be QuarkQuantizedCache instance"
                )
                # Verify cache has layers populated after inference
                assert len(cached_output.past_key_values.layers) > 0, (
                    "Cache should have layers populated after inference"
                )

            # ✅ VERIFY 5: Import and verify cache integration is restored
            for device in ["meta", torch_device]:
                config = AutoConfig.from_pretrained(model_id)

                original_model = init_model(config, device)

                original_model.eval()

                q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

                # Verify imported model has cache attached
                assert hasattr(q_model, "_quark_cache"), "Imported model should have _quark_cache attribute restored"
                assert isinstance(q_model._quark_cache, QuarkQuantizedCache), (
                    "Imported cache should be QuarkQuantizedCache instance"
                )

            with PatchEverywhere("SCALED_MM_AVAILABLE_DEV", None, module_name_prefix="quark"):
                for _, param in q_model.named_parameters():
                    assert param.device != "meta"
                for _, param in q_model.named_buffers():
                    assert param.device != "meta"

                if device == "meta":
                    q_model = q_model.to(torch_device)

                # Test basic inference
                outputs = q_model(INPUT_IDS).to_tuple()

                # Verify outputs are valid (may differ slightly from ref due to export/import)
                for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                    assert output.shape == ref_output.shape, "Output shapes should match"
                    assert not torch.isnan(output).any(), "Outputs should not contain NaN"
                    assert not torch.isinf(output).any(), "Outputs should not contain Inf"

                # ✅ VERIFY 6: Test use_cache=True on imported model
                with torch.no_grad():
                    cached_output_imported = q_model(INPUT_IDS, use_cache=True)
                    assert cached_output_imported.past_key_values is not None, (
                        "Imported model should support use_cache=True"
                    )
                    assert isinstance(cached_output_imported.past_key_values, QuarkQuantizedCache), (
                        "Imported model should use QuarkQuantizedCache"
                    )


@slow_test
@pytest.mark.parametrize(
    "weight_format",
    [pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]],
)
@pytest.mark.parametrize(
    "model_id", [pytest.param(model_name, id=f"model_id:{key}") for key, model_name in TEST_MODELS.items()]
)
def test_gptq_import(weight_format: str, model_id: str):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      GPTQ
    """
    gptq_config = GPTQConfig(
        model_decoder_layers=MODEL_PREFIX[model_id],
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.out_proj",
            "fc1",
            "fc2",
        ],
    )

    W_UINT4_PER_GROUP_CONFIG = QLayerConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)
    quant_config = QConfig(global_quant_config=W_UINT4_PER_GROUP_CONFIG, algo_config=[gptq_config])

    EXCLUDE_LAYERS = ["lm_head", "*.gate", "*.shared_expert_gate"]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)

    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False)
        assert_direct_gates_are_excluded(quant_model)

        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        for device in ["meta", torch_device]:
            config = AutoConfig.from_pretrained(model_id)

            original_model = init_model(config, device)

            q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

            for _, param in q_model.named_parameters():
                assert param.device != "meta"
            for _, param in q_model.named_buffers():
                assert param.device != "meta"

            q_model = q_model.eval()

            if device == "meta":
                q_model = q_model.to(torch_device)

            with torch.no_grad():
                outputs = q_model(INPUT_IDS).to_tuple()

            for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                if torch_device.type == "cpu":
                    assert torch.equal(ref_output, output)
                else:
                    assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


@pytest.mark.parametrize(
    "model_id", [pytest.param(model_name, id=f"model_id:{key}") for key, model_name in TEST_MODELS.items()]
)
def test_non_quantized_import(model_id: str):
    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        non_quantized_model = AutoModelForCausalLM.from_pretrained(
            "haoyang-amd/non_quantized_model", torch_dtype="auto"
        )
        non_quantized_model.save_pretrained(tmpdir)
        config = AutoConfig.from_pretrained(model_id)
        with torch.device("meta"):
            original_model = AutoModelForCausalLM.from_config(config)

        model_config = PretrainedConfig(pretrained_dir=tmpdir)
        model_state_dict = _load_weights_from_safetensors(tmpdir)
        model_config.config_dict["quantization_config"] = None
        _ = _build_quantized_model(original_model, model_config, model_state_dict)


@slow_test
@use_temporary_directory
@pytest.mark.skipif(
    not is_transformers_available() or is_transformers_version_higher_or_equal("5.0.0"),
    reason="requires a version of Transfomers lower than 5.0.0",
)
def test_dbrx_import(tmpdir: str):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      FP8
    """
    FP8_PER_TENSOR_SPEC = QTensorConfig(
        dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False
    )

    EXCLUDE_LAYERS = ["lm_head"]

    W_FP8_A_FP8_PER_TENSOR_CONFIG = QLayerConfig(
        input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
    )
    layer_quant_config = {
        "*Wqkv": QLayerConfig(
            input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
        ),
    }

    quant_config = QConfig(
        global_quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG, layer_quant_config=layer_quant_config, exclude=EXCLUDE_LAYERS
    )

    with torch.inference_mode():
        quantizer = ModelQuantizer(quant_config)
        dbrx_id = "haoyang-amd/dbrx_layer1"

        config = AutoConfig.from_pretrained(dbrx_id, trust_remote_code=True)
        original_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        original_model.eval()
        original_model = original_model.to(torch_device)

        preprocess_for_quantization(original_model)

        # Get dataloader, if multi_gpu, give the first layer's device
        calib_dataloader = get_dataloader()

        quant_model = quantizer.quantize_model(original_model, calib_dataloader)
        # Inference with quantized model
        for i in calib_dataloader:
            quant_model(i)
        quant_model = quantizer.freeze(quant_model)

        export_safetensors(model=quant_model, output_dir=tmpdir, pack_method="reorder")
        gc.collect()
        torch.cuda.empty_cache()

        # TODO: trust_remote_code=True is dangerous, to be removed.
        original_model_config = AutoConfig.from_pretrained(dbrx_id, trust_remote_code=True)
        original_model2 = AutoModelForCausalLM.from_config(original_model_config, trust_remote_code=True)
        original_model2.eval()
        original_model2 = original_model2.to(torch_device)

        preprocess_for_quantization(original_model2)

        q_model = import_model_from_safetensors(original_model2, model_dir=tmpdir, multi_device=False)
        q_model = q_model.to(torch_device)


@slow_test
@pytest.mark.parametrize(
    "model_id", [pytest.param(model_name, id=f"model_id:{key}") for key, model_name in TEST_MODELS.items()]
)
def test_custom_mode_export(model_id: str):
    # AWQ model.
    W_UINT4_PER_GROUP_CONFIG = QLayerConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)
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
        model_decoder_layers=MODEL_PREFIX[model_id],
    )

    EXCLUDE_LAYERS = ["lm_head"]
    quant_config = QConfig(
        global_quant_config=W_UINT4_PER_GROUP_CONFIG, algo_config=[AWQ_CONFIG], exclude=EXCLUDE_LAYERS
    )
    quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_safetensors(
            model=quant_model,
            output_dir=tmpdir,
            custom_mode="awq",
            weight_format="real_quantized",
            pack_method="reorder",
        )

        config = AutoConfig.from_pretrained(tmpdir)
        assert config.quantization_config["quant_method"] == "awq"

        # FP8 model.
        FP8_PER_TENSOR_SPEC = QTensorConfig(
            dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False
        )

        W_FP8_A_FP8_PER_TENSOR_CONFIG = QLayerConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC)
        quant_config = QConfig(global_quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG, exclude=EXCLUDE_LAYERS)
        quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False)

        export_safetensors(
            model=quant_model,
            output_dir=tmpdir,
            custom_mode="fp8",
            weight_format="real_quantized",
            pack_method="reorder",
        )

        config = AutoConfig.from_pretrained(tmpdir)
        assert config.quantization_config["quant_method"] == "fp8"
        assert "activation_scheme" in config.quantization_config
        assert "kv_cache_scheme" in config.quantization_config
        assert "export" in config.quantization_config


@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize(
    "model_id", [pytest.param(model_name, id=f"model_id:{key}") for key, model_name in TEST_MODELS.items()]
)
def test_export_safetensors_invalid_parameters(model_id: str):
    """Test that export_safetensors raises ValueError for invalid custom_mode, weight_format, pack_method, and quant_config=None."""
    quant_spec = QTensorConfig(
        dtype=Dtype.int4,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    quant_config = QConfig(global_quant_config=QLayerConfig(weight=quant_spec))
    quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Invalid custom_mode
        with pytest.raises(ValueError, match=r"Custom_mode must be one of `quark`, `fp8`, `awq`.*invalid_mode"):
            export_safetensors(
                model=quant_model,
                output_dir=tmpdir,
                custom_mode="invalid_mode",
                weight_format="real_quantized",
                pack_method="reorder",
            )
        # Invalid weight_format
        with pytest.raises(
            ValueError, match=r"Weight_format must be one of `real_quantized`, `fake_quantized`.*invalid_weight"
        ):
            export_safetensors(
                model=quant_model,
                output_dir=tmpdir,
                custom_mode="quark",
                weight_format="invalid_weight",
                pack_method="reorder",
            )
        # Invalid pack_method
        with pytest.raises(ValueError, match=r"Pack_method must be one of `reorder`, `order`.*invalid_pack"):
            export_safetensors(
                model=quant_model,
                output_dir=tmpdir,
                custom_mode="quark",
                weight_format="real_quantized",
                pack_method="invalid_pack",
            )

        # Model without quant_config attribute
        if getattr(quant_model, "quark_quantized", False) and hasattr(quant_model, "quant_config"):
            delattr(quant_model, "quant_config")
        with pytest.raises(
            ValueError, match=r"Model must have a 'quant_config' attribute if it is quantized with quark."
        ):
            export_safetensors(
                model=quant_model,
                output_dir=tmpdir,
                custom_mode="quark",
                weight_format="real_quantized",
                pack_method="reorder",
            )


def test_multi_safetensors_load():
    script_dir = os.path.dirname(__file__)

    safetensors_dir = os.path.join(script_dir, "simple_model")
    model_state_dict = _load_weights_from_safetensors(safetensors_dir)
    assert len(model_state_dict) == 10


@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param(model_id, id=model_id)
        for model_id in ["amd/Meta-Llama-3.1-8B-Instruct-FP8-KV", "amd-quark/dummy-config-awq"]
    ],
)
def test_custom_config_remap(model_id: str):
    hf_config = AutoConfig.from_pretrained(model_id)

    _ = QuantConfigParser.from_custom_config(
        hf_config.quantization_config, is_bias_quantized=False, is_kv_cache=False, kv_layers_name=None
    )


@pytest.mark.parametrize(
    "model_id", [pytest.param(model_id, id=model_id) for model_id in ["amd-quark/quark-legacy-awq"]]
)
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

    _ = import_model_from_safetensors(model, model_dir=custom_model_path, multi_device=False)


# TODO: When the import function of mx is complete, this function should be upgraded to "import"
# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize(
    "weight_format",
    [pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]],
)
@pytest.mark.parametrize(
    "model_id", [pytest.param(model_name, id=f"model_id:{key}") for key, model_name in TEST_MODELS.items()]
)
@retry_flaky_test()
def test_wmxfp4_afp8_export(weight_format: str, model_id: str):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      wmxfp4_afp8
    """
    FP8_PER_TENSOR_SPEC = QTensorConfig(
        dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False
    )
    MXFP4_PER_GROUP_SYM_SPEC = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
    EXCLUDE_LAYERS = ["lm_head"]

    W_MXFP4_A_FP8_PER_GROUP_SYM_CONFIG = QLayerConfig(
        weight=MXFP4_PER_GROUP_SYM_SPEC, input_tensors=FP8_PER_TENSOR_SPEC
    )
    quant_config = QConfig(global_quant_config=W_MXFP4_A_FP8_PER_GROUP_SYM_CONFIG, exclude=EXCLUDE_LAYERS)
    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False)

        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            _ = quant_model(INPUT_IDS).to_tuple()


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize(
    "weight_format",
    [pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]],
)
@pytest.mark.parametrize(
    "model_id", [pytest.param(model_name, id=f"model_id:{key}") for key, model_name in TEST_MODELS.items()]
)
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU with max abs diff ~0.1.
def test_wfp4_afp8_import(weight_format: str, model_id: str):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      wfp4_afp8
    """
    MX_SEPARATED_FP8_E4M3_PER_GROUP_SYM_SPEC = OCP_MXFP8E4M3Spec(ch_axis=-1, is_dynamic=True).to_quantization_spec()
    MX_SEPARATED_FP4_PER_GROUP_SYM_SPEC = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()

    EXCLUDE_LAYERS = ["lm_head"]
    W_MXFP4_A_MXFP8 = QLayerConfig(
        input_tensors=MX_SEPARATED_FP8_E4M3_PER_GROUP_SYM_SPEC, weight=MX_SEPARATED_FP4_PER_GROUP_SYM_SPEC
    )

    quant_config = QConfig(global_quant_config=W_MXFP4_A_MXFP8, exclude=EXCLUDE_LAYERS)
    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False)

        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        for device in ["meta", torch_device]:
            config = AutoConfig.from_pretrained(model_id)

            original_model = init_model(config, device)

            q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

            q_model = q_model.eval()

            if device == "meta":
                q_model = q_model.to(torch_device)

            with torch.no_grad():
                outputs = q_model(INPUT_IDS).to_tuple()

            for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                if torch_device.type == "cpu":
                    assert torch.equal(ref_output, output)
                else:
                    assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


@pytest.mark.parametrize("weight_format", ["real_quantized", "fake_quantized"])
@pytest.mark.parametrize(
    "model_id", [pytest.param(model_name, id=f"model_id:{key}") for key, model_name in TEST_MODELS.items()]
)
def test_amdfp4_export_import(weight_format: str, model_id: str):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      amdfp4
    """
    # Load model to get config
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Get template and config for amdfp4 scheme
    template = LLMTemplate(model_type=model.config.model_type)
    quant_config = template.get_config("amdfp4", exclude_layers=["lm_head"])

    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False)

        quant_model(INPUT_IDS).to_tuple()

        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

        # Verify dtype of saved tensors for real_quantized
        if weight_format == "real_quantized":
            safetensors_path = Path(tmpdir) / "model.safetensors"
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                # Check q_proj weight (fp4 packed) is uint8
                q_proj_weight = f.get_tensor(f"{MODEL_PREFIX[model_id]}.0.self_attn.q_proj.weight")
                assert q_proj_weight.dtype == torch.uint8

                # Check q_proj weight_scale (E5M3) is uint8
                q_proj_scale = f.get_tensor(f"{MODEL_PREFIX[model_id]}.0.self_attn.q_proj.weight_scale")
                assert q_proj_scale.dtype == torch.uint8

                # Check shape relationship: weight.shape[-1] * 2 == weight_scale.shape[-1] * 16
                # This verifies FP4 packing (2 values per byte) and group_size=16 (1 scale per 16 elements)
                assert q_proj_weight.shape[-1] * 2 == q_proj_scale.shape[-1] * 16, (
                    f"Shape mismatch: weight.shape[-1] * 2 = {q_proj_weight.shape[-1] * 2}, "
                    f"weight_scale.shape[-1] * 16 = {q_proj_scale.shape[-1] * 16}"
                )

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        for device in ["meta", torch_device]:
            config = AutoConfig.from_pretrained(model_id)

            original_model = init_model(config, device)

            q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

            q_model = q_model.eval()

            if device == "meta":
                q_model = q_model.to(torch_device)

            with torch.no_grad():
                outputs = q_model(INPUT_IDS).to_tuple()

            for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                if torch_device.type == "cpu":
                    assert torch.equal(ref_output, output)
                else:
                    assert torch.allclose(output, ref_output, atol=1e-4)


# TODO: run test_kv_layers_exclude_import with MOE models.
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize("kv_cache_post_rope", [False, True])
def test_kv_layers_exclude_import(kv_cache_post_rope: bool):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      wfp8_afp8, kv_layers excluded in quantization config, but kv cache still need to be quantized
    """
    FP8_PER_TENSOR_SPEC = QTensorConfig(
        dtype=Dtype.fp8_e4m3, qscheme=QSchemeType.per_tensor, observer_cls=PerTensorMinMaxObserver, is_dynamic=False
    )

    EXCLUDE_LAYERS = ["lm_head", "*.k_proj", "*.v_proj"]

    model_id = "facebook/opt-125m"

    W_FP8_A_FP8_PER_TENSOR_CONFIG = QLayerConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC)
    layer_quant_config = {
        "*v_proj": QLayerConfig(
            input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
        ),
        "*k_proj": QLayerConfig(
            input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
        ),
    }
    kv_cache_quant_config = layer_quant_config.copy()

    quant_config = QConfig(
        global_quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG,
        layer_quant_config=layer_quant_config,
        kv_cache_quant_config=kv_cache_quant_config,
        exclude=EXCLUDE_LAYERS,
    )
    quant_config.kv_cache_post_rope = kv_cache_post_rope  # type: ignore[attr-defined]
    with torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_safetensors(model=quant_model, output_dir=tmpdir, pack_method="reorder")

            ref_outputs = quant_model(INPUT_IDS).to_tuple()

            for device in ["meta", torch_device]:
                config = AutoConfig.from_pretrained(model_id)

                original_model = init_model(config, device)

                original_model.eval()

                q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

                # scaled_mm has exclusive tests, only naive mode is tested here.
                with PatchEverywhere("SCALED_MM_AVAILABLE_DEV", None, module_name_prefix="quark"):
                    for _, param in q_model.named_parameters():
                        assert param.device != "meta"
                    for _, param in q_model.named_buffers():
                        assert param.device != "meta"

                    if device == "meta":
                        q_model = q_model.to(torch_device)

                    outputs = q_model(INPUT_IDS).to_tuple()

                    for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                        if torch_device.type == "cpu":
                            assert torch.equal(ref_output, output)
                        else:
                            assert torch.allclose(output, ref_output, atol=1e-2)  # This one appears not to be flaky.


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@pytest.mark.parametrize(
    "weight_format",
    [pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]],
)
@pytest.mark.parametrize(
    "model_id", [pytest.param(model_name, id=f"model_id:{key}") for key, model_name in TEST_MODELS.items()]
)
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU with max abs diff ~0.1.
def test_wfp6_e2m3_afp6_e2m3_import(weight_format: str, model_id: str):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      wfp6_e2m3_afp6_e2m3
    """

    def FP6_E2M3_PER_GROUP_SYM_SPEC(group_size, scale_format="e8m0", scale_calculation_mode="even", is_dynamic=True):
        return FP6E2M3PerGroupSpec(
            ch_axis=-1,
            group_size=group_size,
            scale_format=scale_format,
            scale_calculation_mode=scale_calculation_mode,
            is_dynamic=is_dynamic,
        ).to_quantization_spec()

    EXCLUDE_LAYERS = ["lm_head"]
    global_quant_config = QLayerConfig(
        input_tensors=FP6_E2M3_PER_GROUP_SYM_SPEC(32, "e8m0", "even", True),
        weight=FP6_E2M3_PER_GROUP_SYM_SPEC(32, "e8m0", "even", False),
    )
    quant_config = QConfig(global_quant_config=global_quant_config, exclude=EXCLUDE_LAYERS)

    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False)

        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        for device in ["meta", torch_device]:
            config = AutoConfig.from_pretrained(model_id)

            original_model = init_model(config, device)

            q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

            q_model = q_model.eval()

            if device == "meta":
                q_model = q_model.to(torch_device)

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
    "weight_format",
    [pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]],
)
@pytest.mark.parametrize(
    "model_id", [pytest.param(model_name, id=f"model_id:{key}") for key, model_name in TEST_MODELS.items()]
)
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU with max abs diff ~0.1.
def test_wfp6_e3m2_afp6_e3m2_import(weight_format: str, model_id: str):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      wfp6_e3m2_afp6_e3m2
    """

    def FP6_E3M2_PER_GROUP_SYM_SPEC(group_size, scale_format="e8m0", scale_calculation_mode="even", is_dynamic=True):
        return FP6E3M2PerGroupSpec(
            ch_axis=-1,
            group_size=group_size,
            scale_format=scale_format,
            scale_calculation_mode=scale_calculation_mode,
            is_dynamic=is_dynamic,
        ).to_quantization_spec()

    EXCLUDE_LAYERS = ["lm_head"]
    global_quant_config = QLayerConfig(
        input_tensors=FP6_E3M2_PER_GROUP_SYM_SPEC(32, "e8m0", "even", True),
        weight=FP6_E3M2_PER_GROUP_SYM_SPEC(32, "e8m0", "even", False),
    )
    quant_config = QConfig(global_quant_config=global_quant_config, exclude=EXCLUDE_LAYERS)

    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False)

        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

        with safe_open(Path(tmpdir, "model.safetensors"), framework="pt") as f:
            checkpoint_keys = f.keys()

            assert f"{MODEL_PREFIX[model_id]}.1.self_attn.k_proj.weight_scale" in checkpoint_keys

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        for device in ["meta", torch_device]:
            config = AutoConfig.from_pretrained(model_id)

            original_model = init_model(config, device)

            q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

            q_model = q_model.eval()

            if device == "meta":
                q_model = q_model.to(torch_device)

            with torch.no_grad():
                outputs = q_model(INPUT_IDS).to_tuple()

            for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                if torch_device.type == "cpu":
                    assert torch.equal(ref_output, output)
                else:
                    assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


# For torch requirement, refer to /pull/2529#issuecomment-235620
# Before PyTorch 2.9.0, FP8 operations on AMD GPUs exhibited numerical instability,
# causing slightly different outputs for the same inputs. PyTorch 2.9.0 has fixed
# these issues, and FP8 computation is now deterministic.
# Using `2.8.99` so that this test runs as well on e.g. 2.9.0a0+git1c57644.
@require_torch_higher_or_equal("2.8.99")
@pytest.mark.parametrize(
    "weight_format",
    [pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]],
)
@pytest.mark.parametrize(
    "model_id", [pytest.param(model_name, id=f"model_id:{key}") for key, model_name in TEST_MODELS.items()]
)
def test_fp4_per_group_fp8_per_tensor_scale_export_import(weight_format: str, model_id: str):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      w_fp4_per_group_fp8_per_tensor_scale_a_fp4_per_group_fp8_per_tensor_scale
    """
    FP4_PER_GROUP_FP8_PER_TENSOR_SCALE_SPEC = ScaleQuantSpec(
        first_stage=FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=False),
        second_stage=FP8E4M3PerTensorSpec(is_dynamic=False),
    ).to_quantization_spec()

    FP4_PER_GROUP_FP8_PER_TENSOR_SCALE_SPEC_MIXED_DYNAMIC = ScaleQuantSpec(
        first_stage=FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=True),
        second_stage=FP8E4M3PerTensorSpec(is_dynamic=False),
    ).to_quantization_spec()

    EXCLUDE_LAYERS = ["lm_head"]
    W_FP4_A_FP4_SCALE_FP8_CONFIG = QLayerConfig(
        input_tensors=FP4_PER_GROUP_FP8_PER_TENSOR_SCALE_SPEC_MIXED_DYNAMIC,
        weight=FP4_PER_GROUP_FP8_PER_TENSOR_SCALE_SPEC,
    )

    quant_config = QConfig(global_quant_config=W_FP4_A_FP4_SCALE_FP8_CONFIG, exclude=EXCLUDE_LAYERS)
    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False, torch_dtype=None)

        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

        with safe_open(Path(tmpdir, "model.safetensors"), framework="pt") as f:
            checkpoint_keys = f.keys()

            assert f"{MODEL_PREFIX[model_id]}.1.self_attn.k_proj.weight_scale" in checkpoint_keys
            assert f"{MODEL_PREFIX[model_id]}.1.self_attn.k_proj.weight_scale_2" in checkpoint_keys

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        for device in ["meta", torch_device]:
            config = AutoConfig.from_pretrained(model_id)

            original_model = init_model(config, device)

            q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

            q_model = q_model.eval()

            if device == "meta":
                q_model = q_model.to(torch_device)

            with torch.no_grad():
                outputs = q_model(INPUT_IDS).to_tuple()

            for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                if torch_device.type == "cpu":
                    assert torch.equal(ref_output, output)
                else:
                    assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@skip_torch_version("2.8")
@pytest.mark.parametrize(
    "torch_dtype",
    [pytest.param(val, id=f"torch_dtype:{val}") for val in [torch.float16, torch.bfloat16, torch.float32]],
)
@pytest.mark.parametrize(
    "weight_format",
    [pytest.param(weight_format, id=str(weight_format)) for weight_format in ["real_quantized", "fake_quantized"]],
)
@pytest.mark.parametrize(
    "model_id", [pytest.param(model_name, id=f"model_id:{key}") for key, model_name in TEST_MODELS.items()]
)
@retry_flaky_test()  # Test is flaky (~1/50 fail on MI250) on GPU with max abs diff ~0.1.
def test_wfp8_int4perchannel_afp8_import(weight_format: str, torch_dtype: torch.dtype, model_id: str):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      wfp8_int4perchannel_afp8
    """
    FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(is_dynamic=False).to_quantization_spec()
    INT4_PER_CHANNEL_SPEC = Int4PerChannelSpec(ch_axis=0, is_dynamic=False).to_quantization_spec()
    FP8_INT4_PER_CHANNEL_SPEC = [FP8_PER_TENSOR_SPEC, INT4_PER_CHANNEL_SPEC]

    # NOTE: qwen3_moe tiny's gate is too small to be quantized here.
    EXCLUDE_LAYERS = ["lm_head", "*mlp.gate"]
    W_FP8_A_INT4_PER_CHANNEL = QLayerConfig(weight=FP8_INT4_PER_CHANNEL_SPEC, input_tensors=FP8_PER_TENSOR_SPEC)

    quant_config = QConfig(global_quant_config=W_FP8_A_INT4_PER_CHANNEL, exclude=EXCLUDE_LAYERS)
    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False, torch_dtype=torch_dtype)
        assert_direct_gates_are_excluded(quant_model)

        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

        quant_model(INPUT_IDS).to_tuple()

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        for device in ["meta", torch_device]:
            config = AutoConfig.from_pretrained(model_id)

            original_model = init_model(config, device, torch_dtype=torch_dtype)

            q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

            q_model = q_model.eval()

            if device == "meta":
                q_model = q_model.to(torch_device)

            with torch.no_grad():
                outputs = q_model(INPUT_IDS).to_tuple()

            for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                if torch_device.type == "cpu":
                    assert torch.equal(ref_output, output)
                else:
                    assert torch.allclose(output, ref_output, atol=1e-4)  # This one appears not to be flaky.


@use_temporary_directory
def test_import_raise_error_non_persistent_buffer(tmpdir: str):
    model_dir = "amd-quark/tiny-llama-fast-tokenizer"

    quant_spec = QTensorConfig(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )

    quant_config = QConfig(global_quant_config=QLayerConfig(weight=quant_spec))

    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=model_dir, multi_gpu=False)

        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format="real_quantized", pack_method="reorder")

        config = AutoConfig.from_pretrained(model_dir)
        with torch.device("meta"):
            original_model = AutoModelForCausalLM.from_config(config)

        with pytest.raises(Exception) as exc_info:
            _ = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

        assert "Importing a model containing non-persistent buffers on meta device" in str(exc_info.value)


@use_temporary_directory
def test_checkpoint_conversion_mapping(tmpdir: str):
    model_dir = "amd-quark/tiny-llama-fast-tokenizer"

    quant_spec = QTensorConfig(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )

    quant_config = QConfig(global_quant_config=QLayerConfig(weight=quant_spec))
    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        quant_model = quantize_model(quant_config, model_name=model_dir, multi_gpu=False)

        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format="real_quantized", pack_method="reorder")

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        original_weights = _load_weights_from_safetensors(tmpdir)
        renamed_weights = {}
        key_mapping_applied = False

        for key, value in original_weights.items():
            # rename keys: model.layers.X -> model.blocks.X
            if "model.layers" in key:
                new_key = key.replace("model.layers", "model.blocks")
                renamed_weights[new_key] = value
                key_mapping_applied = True
            else:
                renamed_weights[key] = value

        # make sure we actually renamed some keys
        assert key_mapping_applied, "Test setup failed: no keys were renamed"

        # save the renamed weights back to safetensors
        safetensors_path = Path(tmpdir) / "model.safetensors"
        save_file(renamed_weights, str(safetensors_path))

        config = AutoConfig.from_pretrained(model_dir)
        original_model = AutoModelForCausalLM.from_config(config)
        original_model = original_model.to(torch_device)

        # set the checkpoint conversion mapping to reverse the renaming
        # pattern: "model.blocks" -> "model.layers"
        original_model._checkpoint_conversion_mapping = {
            r"^model.blocks": "model.layers",
        }

        q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)
        q_model = q_model.eval()

        with torch.no_grad():
            outputs = q_model(INPUT_IDS).to_tuple()

        for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
            if torch_device.type == "cpu":
                assert torch.equal(ref_output, output)
            else:
                assert torch.allclose(output, ref_output, atol=1e-4)


# ---------------------------------------------------------------------------
# Pre-quantized model tests
# ---------------------------------------------------------------------------


def _run_prequantized_export_import_test(
    model_id: str, expected_layer_type: str, weight_format: str, trust_remote_code: bool = True
) -> None:
    """Helper to test quantizing, exporting, and importing a pre-quantized model.

    Loads a pre-quantized HuggingFace model (with FP8Linear or compressed-tensors layers),
    re-quantizes it with Quark using FP8 per-tensor, exports to safetensors,
    and verifies the import produces matching outputs.
    """
    if expected_layer_type == "compressed-tensors":
        pytest.importorskip("compressed_tensors", reason="compressed_tensors required for compressed-tensors tests")

    FP8_PER_TENSOR_SPEC = QTensorConfig(
        dtype=Dtype.fp8_e4m3,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
    )
    EXCLUDE_LAYERS = ["lm_head"]
    quant_config = QConfig(
        global_quant_config=QLayerConfig(
            weight=FP8_PER_TENSOR_SPEC,
            input_tensors=FP8_PER_TENSOR_SPEC,
        ),
        exclude=EXCLUDE_LAYERS,
    )

    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        # Load pre-quantized model
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", trust_remote_code=trust_remote_code)
        model.eval()
        model = model.to(torch_device)

        # Verify pre-quantized layers are present
        prequant_count = sum(1 for m in model.modules() if is_prequantized_linear(m))
        assert prequant_count > 0, f"Expected pre-quantized layers in {model_id}, found {prequant_count}"

        # Create calibration dataloader using model's tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        text = "Hello, how are you?"
        tokenized_outputs = tokenizer(text, return_tensors="pt")
        calib_dataloader = DataLoader(tokenized_outputs["input_ids"].to(model.device))

        # Quantize with Quark
        quantizer = ModelQuantizer(quant_config)
        quant_model = quantizer.quantize_model(model, calib_dataloader)
        for i in calib_dataloader:
            quant_model(i)
        quant_model = quantizer.freeze(quant_model)

        # Export
        export_safetensors(
            model=quant_model,
            output_dir=tmpdir,
            weight_format=weight_format,
            pack_method="reorder",
        )

        # Reference outputs
        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        # Import and verify
        for device in ["meta", torch_device]:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            # Remove quantization_config to create a standard (non-quantized) model
            if hasattr(config, "quantization_config"):
                del config.quantization_config
            if device == "meta":
                context = ExitStack()
                context.enter_context(no_init_weights())
                context.enter_context(init_empty_weights())
            else:
                context = torch.device(device)
            with context:
                original_model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)

            q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)
            q_model = q_model.eval()

            for _, param in q_model.named_parameters():
                assert param.device != "meta"
            for _, param in q_model.named_buffers():
                assert param.device != "meta"

            # Use PatchEverywhere to disable scaled_mm for consistent testing
            with PatchEverywhere("SCALED_MM_AVAILABLE_DEV", None, module_name_prefix="quark"):
                if device == "meta":
                    q_model = q_model.to(torch_device)

                with torch.no_grad():
                    outputs = q_model(INPUT_IDS).to_tuple()

                for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
                    if torch_device.type == "cpu":
                        assert torch.equal(ref_output, output)
                    else:
                        assert torch.allclose(output, ref_output, atol=1e-4)


@require_torch_higher_or_equal("2.6")
@skip_torch_version("2.8")
@pytest.mark.parametrize(
    "model_id, expected_layer_type",
    [
        pytest.param(
            "Qwen/Qwen3-4B-FP8",
            "FP8Linear",
            id="fp8-native",
            # transformers FP8Linear dequant path pulls in a Triton kernel; absent on CPU-only builds.
            marks=pytest.mark.skipif(not is_triton_available(), reason="Triton is not installed."),
        ),
        pytest.param("RedHatAI/Qwen2.5-0.5B-quantized.w4a16", "compressed-tensors", id="compressed-w4a16"),
        pytest.param("RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic", "compressed-tensors", id="compressed-fp8-dynamic"),
    ],
)
@pytest.mark.parametrize("weight_format", ["real_quantized"])
@retry_flaky_test()
def test_prequantized_export_import(model_id: str, expected_layer_type: str, weight_format: str):
    """
    Test Features:
        Pre-quantized Model:      FP8Linear (Qwen3-4B-FP8) / compressed-tensors (w4a16, FP8-dynamic)
        Re-quantization Method:   FP8 per-tensor (weight + activation)
        Import Format:            Json-safetensors
    """
    _run_prequantized_export_import_test(model_id, expected_layer_type, weight_format)


MOE_TEST_MODELS = [
    pytest.param(
        "optimum-intel-internal-testing/tiny-random-llama4",
        "AutoModelForImageTextToText",
        id="llama4",
    ),
    pytest.param(
        "amd-quark/tiny-random-qwen3_moe",
        "AutoModelForCausalLM",
        id="qwen3_moe",
    ),
    pytest.param(
        "optimum-intel-internal-testing/tiny-random-granitemoehybrid",
        "AutoModelForCausalLM",
        id="granitemoehybrid",
    ),
    pytest.param(
        "optimum-intel-internal-testing/tiny-random-gpt-oss-mxfp4",
        "AutoModelForCausalLM",
        id="gpt_oss",
    ),
]


@pytest.mark.parametrize("model_id, auto_cls_name", MOE_TEST_MODELS)
@pytest.mark.parametrize("weight_format", ["real_quantized"])
def test_moe_reload_with_prepare_for_moe_quant(model_id: str, auto_cls_name: str, weight_format: str):
    """
    Test that prepare_for_moe_quant preserves model outputs and that MoE models
    can be correctly reloaded using prepare_for_moe_quant(reload=True).

    The test verifies:
    1. prepare_for_moe_quant does not change model outputs (module replacement is numerically equivalent)
    2. Full reload flow: quantize -> export -> skeleton from_config -> import -> verify outputs

    Test Features:
        Models:                   llama4, qwen3_moe, granitemoehybrid, gpt_oss
        MoE Preparation:          prepare_for_moe_quant / prepare_for_moe_quant(reload=True)
        Import Format:            Json-safetensors
        Quantization Method:      INT8 per-tensor
    """
    import transformers

    # The shared `optimum-intel-internal-testing/tiny-random-llama4` checkpoint predates transformers
    # switching `attn_temperature_tuning` from int to bool: its config.json stores `4`, which fails
    # huggingface_hub `@strict` bool validation at load time, before any quark code runs. Configs became
    # `@strict` dataclasses in 5.4 (5.3 has none), so gate on that rather than on 5.0, which would skip
    # needlessly on 5.2/5.3. Skip until the shared checkpoint is refreshed.
    if "llama4" in model_id.lower() and is_transformers_version_higher_or_equal("5.4.0"):
        pytest.skip(
            "tiny-random-llama4 config stores legacy int `attn_temperature_tuning=4`, which fails "
            "transformers>=5.4 @strict bool validation; needs an updated checkpoint."
        )

    auto_cls = getattr(transformers, auto_cls_name)

    # `kernels` in the test env disables transformers' mxfp4 auto-dequantize, and the native triton
    # path's save reshape hardcodes GPT-OSS-20B's hidden_size. Load dequantized, like `get_model` does.
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "attn_implementation": "eager"}
    if "gpt-oss" in model_id:
        load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)

    # 1. Verify prepare_for_moe_quant does not change model outputs
    original_model = auto_cls.from_pretrained(model_id, **load_kwargs)
    original_model.eval()
    original_model = original_model.to(torch_device)

    with torch.no_grad():
        pre_patch_outputs = original_model(INPUT_IDS).to_tuple()

    preprocess_for_quantization(original_model)

    with torch.no_grad():
        post_patch_outputs = original_model(INPUT_IDS).to_tuple()

    for pre, post in zip(pre_patch_outputs[0], post_patch_outputs[0], strict=False):
        assert torch.allclose(pre, post, atol=1e-3), "prepare_for_moe_quant should not change model outputs"

    del original_model
    gc.collect()

    # 2. Full reload flow: quantize -> export -> reload with prepare_for_moe_quant(reload=True)
    INT8_SPEC = QTensorConfig(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )

    EXCLUDE_LAYERS = ["lm_head", "*.gate", "*.router"]
    quant_config = QConfig(global_quant_config=QLayerConfig(weight=INT8_SPEC), exclude=EXCLUDE_LAYERS)

    with tempfile.TemporaryDirectory() as tmpdir, torch.inference_mode():
        model = auto_cls.from_pretrained(model_id, **load_kwargs)
        model.eval()
        model = model.to(torch_device)

        # NOTE: This model has a wrong modules_to_not_convert not in line with
        # openai/gpt-oss-20b. TODO: use an other model and avoid this patching.
        if model_id == "optimum-intel-internal-testing/tiny-random-gpt-oss-mxfp4":
            model.config.quantization_config = {
                "quant_method": "mxfp4",
                "modules_to_not_convert": [
                    "model.layers.*.self_attn",
                    "model.layers.*.mlp.router",
                    "model.embed_tokens",
                    "lm_head",
                ],
            }

        original_router_dtypes = {
            name: module.weight.dtype
            for name, module in model.named_modules()
            if name.endswith((".gate", ".router")) and hasattr(module, "weight")
        }
        preprocess_for_quantization(model)

        try:
            calib_dataloader = get_dataloader(model_id, model.device)
        except Exception:
            calib_dataloader = DataLoader(INPUT_IDS.to(model.device))

        quantizer = ModelQuantizer(quant_config)
        quant_model = quantizer.quantize_model(model, calib_dataloader)
        for i in calib_dataloader:
            quant_model(i)
        quant_model = quantizer.freeze(quant_model)

        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

        if model_id == "amd-quark/tiny-random-qwen3_moe":
            excluded_names = ("model.layers.0.mlp.gate", "model.layers.1.mlp.gate")
        elif model_id == "optimum-intel-internal-testing/tiny-random-gpt-oss-mxfp4":
            excluded_names = tuple(name for name in original_router_dtypes if name.endswith(".router"))
            assert excluded_names
        else:
            excluded_names = ()

        if excluded_names:
            assert set(excluded_names).issubset(original_router_dtypes)
            with open(Path(tmpdir, "config.json")) as config_file:
                exported_exclude = json.load(config_file)["quantization_config"]["exclude"]

            assert set(exported_exclude) == {*excluded_names, "lm_head"}
            if model_id == "amd-quark/tiny-random-qwen3_moe":
                assert exported_exclude == [*excluded_names, "lm_head"]

            with safe_open(Path(tmpdir, "model.safetensors"), framework="pt") as safetensors_file:
                checkpoint_keys = set(safetensors_file.keys())
                for excluded_name in excluded_names:
                    assert f"{excluded_name}.weight" in checkpoint_keys
                    assert f"{excluded_name}.weight_scale" not in checkpoint_keys
                    assert f"{excluded_name}.weight_zero_point" not in checkpoint_keys
                    exported_weight = safetensors_file.get_tensor(f"{excluded_name}.weight")
                    assert exported_weight.dtype is original_router_dtypes[excluded_name]

        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        del quant_model
        gc.collect()

        q_model = import_model_from_safetensors(
            None,
            model_dir=tmpdir,
            device=torch_device.type,
            attn_implementation="eager",
        )

        for _, param in q_model.named_parameters():
            assert param.device != torch.device("meta"), "All parameters should be materialized"
        for _, buf in q_model.named_buffers():
            assert buf.device != torch.device("meta"), "All buffers should be materialized"

        with torch.no_grad():
            outputs = q_model(INPUT_IDS).to_tuple()

        for ref_output, output in zip(ref_outputs[0], outputs[0], strict=False):
            if torch_device.type == "cpu":
                assert torch.equal(ref_output, output)
            else:
                assert torch.allclose(output, ref_output, atol=1e-4)


# TODO: verify whether Kimi-K2.5 / Kimi-K2.6 custom modeling code is compatible with Transformers v5.
@local_test_only
@pytest.mark.skipif(
    is_transformers_version_higher_or_equal("5.0"),
    reason="requires transformers < 5.0",
)
def test_kimi_loading():
    # NOTE: This test passes only for transformers==4.57 as of April 2026.

    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise RuntimeError("test_kimi_loading requires at least 4 CUDA devices")

    model_id = "moonshotai/Kimi-K2.6"

    model_dir = huggingface_hub.snapshot_download(model_id, revision="refs/pr/33")
    model, _ = get_model(model_dir, multi_gpu=True)

    model.eval()

    assert not any(p.device.type == "meta" for _, p in model.named_parameters())

    expected_dtypes = {
        "language_model.model.layers.0.self_attn.q_a_proj.weight": torch.bfloat16,
        "language_model.model.layers.0.mlp.down_proj.weight": torch.bfloat16,
        "language_model.model.layers.1.mlp.experts.99.down_proj.weight_packed": torch.int32,
        "language_model.model.layers.1.mlp.experts.99.gate_proj.weight_scale": torch.bfloat16,
        "language_model.model.layers.1.mlp.gate.e_score_correction_bias": torch.float32,
    }
    params = dict(model.named_parameters())
    for name, expected_dtype in expected_dtypes.items():
        assert name in params
        assert params[name].dtype == expected_dtype


def test_preprocessors_load_save():
    _ = maybe_load_preprocessors("facebook/opt-125m")

    with tempfile.TemporaryDirectory() as tmpdir:
        maybe_save_preprocessors("facebook/opt-125m", tmpdir)


@slow_test
@require_torch_higher_or_equal("2.6")
def test_patch_missing_weights_restores_mtp(tmp_path):
    """Weights that transformers drops via _keys_to_ignore_on_load_unexpected (e.g. mtp.*)
    must be restored into the hf_format export from the source checkpoint.

    Regression test for the bug where Qwen3.5 mtp.* weights were silently absent from the
    exported safetensors because Qwen3_5ForConditionalGeneration does not instantiate them
    in __init__ and lists `^mtp.*` in _keys_to_ignore_on_load_unexpected. This test fails on
    `main` (mtp weights missing from export) and passes once the missing weights are merged
    into the state_dict passed to `model.save_pretrained`.
    """
    model_id = "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", attn_implementation="eager")
    model.eval()

    # transformers drops mtp.* during from_pretrained, so it never reaches state_dict.
    assert not any("mtp" in k for k in model.state_dict()), "expected transformers to drop mtp.* keys"
    assert any(re.search(p, "mtp.fc.weight") for p in model._keys_to_ignore_on_load_unexpected)

    # Build a source checkpoint that DOES contain an mtp weight (mirrors a real Qwen3.5 checkpoint).
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    cache_path = huggingface_hub.snapshot_download(repo_id=model_id, repo_type="model")
    source_sd = _load_weights_from_safetensors(cache_path)
    mtp_weight = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    source_sd["mtp.fc.weight"] = mtp_weight
    save_file(source_sd, str(source_dir / "model.safetensors"), metadata={"format": "pt"})
    model.config._name_or_path = str(source_dir)

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    export_hf_model(model, export_dir)

    exported_sd = _load_weights_from_safetensors(str(export_dir))
    assert "mtp.fc.weight" in exported_sd, "mtp.fc.weight should be restored into the export"
    assert torch.equal(exported_sd["mtp.fc.weight"], mtp_weight)


@slow_test
@require_torch_higher_or_equal("2.6")
def test_patch_missing_weights_does_not_leak_compressed_tensors_artifacts(tmp_path):
    """End-to-end guard with a real compressed-tensors checkpoint: the missing-weights merge must
    only restore weights matching the model's _keys_to_ignore_on_load_unexpected patterns,
    and must never transfer compressed-tensors artifacts (weight_scale_inv, etc.) into the export.

    Uses Qwen/Qwen3-0.6B-FP8 truncated to 2 layers. The full source checkpoint on disk holds all
    28 layers (with ~196 weight_scale_inv tensors); the export only contains the 2 truncated layers.
    The source therefore has many keys absent from the export, but none of them match the model's
    ignore patterns, so none must be restored.
    """
    model_id = "Qwen/Qwen3-0.6B-FP8"
    config = AutoConfig.from_pretrained(model_id)
    num_layers = 2
    config.num_hidden_layers = num_layers
    if getattr(config, "layer_types", None) is not None:
        config.layer_types = config.layer_types[:num_layers]

    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, torch_dtype="auto", device_map="cpu", attn_implementation="eager"
    )
    model.eval()

    # Qwen3ForCausalLM declares no ignore list. Older releases leave the instance attribute at None; by
    # 5.15 PreTrainedModel.__init__ normalizes it to an (empty) set() -- modeling_utils.py:1396, absent
    # in 5.2 --
    # `self._keys_to_ignore_on_load_unexpected = set(self._keys_to_ignore_on_load_unexpected or [])`. Either
    # way it is falsy, so the missing-weights merge restores nothing here -- never scale_inv artifacts.
    assert not model._keys_to_ignore_on_load_unexpected

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    export_hf_model(model, export_dir)

    exported_sd = _load_weights_from_safetensors(str(export_dir))

    # No key belonging only to the truncated-away layers (2..27) may leak into the export, in
    # particular none of the compressed-tensors weight_scale_inv artifacts.
    leaked = [k for k in exported_sd if re.search(r"\.layers\.(?:[2-9]|1[0-9]|2[0-7])\.", k)]
    assert leaked == [], f"source-only keys must not be copied into the export, leaked: {leaked[:10]}"
    assert not any("mtp" in k for k in exported_sd), "no mtp weights exist in this model, none should appear"


def test_patch_missing_weights_respects_explicit_empty_ignore_list(tmp_path):
    """An explicitly empty _keys_to_ignore_on_load_unexpected ([]) means "nothing is
    ignored" and must be a no-op, while an absent attribute (None) falls back to the
    default mtp.* pattern. Regression test for the falsy-`[]` bug where an explicit
    empty list wrongly triggered the default pattern.
    """
    import types

    # Source checkpoint contains an mtp weight; the model's own state_dict does not.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    save_file({"mtp.fc.weight": torch.ones(2, 2)}, str(source_dir / "model.safetensors"), metadata={"format": "pt"})

    def _make_model(ignore_attr):
        model = types.SimpleNamespace()
        model.config = types.SimpleNamespace(_name_or_path=str(source_dir))
        model._keys_to_ignore_on_load_unexpected = ignore_attr
        model.state_dict = lambda: {"model.embed.weight": torch.zeros(2, 2)}
        model.save_pretrained = lambda export_dir, state_dict: save_file(
            state_dict, str(Path(export_dir) / "model.safetensors"), metadata={"format": "pt"}
        )
        model.generation_config = None
        return model

    # Case 1: explicit empty list -> no-op, mtp must NOT be restored.
    export_empty = tmp_path / "export_empty"
    export_empty.mkdir()
    export_hf_model(_make_model([]), export_empty)
    assert "mtp.fc.weight" not in _load_weights_from_safetensors(str(export_empty))

    # Case 2: absent / None -> default pattern applies, mtp IS restored.
    export_none = tmp_path / "export_none"
    export_none.mkdir()
    export_hf_model(_make_model(None), export_none)
    assert "mtp.fc.weight" in _load_weights_from_safetensors(str(export_none))


def test_restore_fp32_params_from_source(tmp_path):
    """
    _restore_fp32_params_from_source must upgrade parameters that are float32 in the
    source safetensors checkpoint but were downcast (e.g. to bfloat16) during loading.

    Regression test for issue #5141: Kimi-K2.5 e_score_correction_bias is float32 in the
    source checkpoint but gets cast to bfloat16 when loaded with torch_dtype="auto" because
    the model class is missing _keep_in_fp32_modules = ["MoEGate"].
    """
    from quark.torch.utils.llm.model_preparation import _restore_fp32_params_from_source

    fp32_val = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    bf16_val = torch.tensor([0.5, 1.5], dtype=torch.bfloat16)

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    save_file(
        {"e_score_correction_bias": fp32_val, "weight": bf16_val},
        str(source_dir / "model.safetensors"),
        metadata={"format": "pt"},
    )

    # Simulate the downcast: model has e_score_correction_bias as bfloat16
    # (as would happen when loading Kimi-K2.5 with torch_dtype="auto").
    model = torch.nn.Module()
    model.e_score_correction_bias = torch.nn.Parameter(fp32_val.to(torch.bfloat16))
    model.weight = torch.nn.Parameter(bf16_val.clone())

    assert model.e_score_correction_bias.dtype == torch.bfloat16
    assert model.weight.dtype == torch.bfloat16

    _restore_fp32_params_from_source(model, str(source_dir))

    assert model.e_score_correction_bias.dtype == torch.float32, "e_score_correction_bias must be restored to float32"
    assert torch.equal(model.e_score_correction_bias.data, fp32_val), "restored values must match source"
    assert model.weight.dtype == torch.bfloat16, "bfloat16 params must remain bfloat16"


def test_restore_fp32_params_from_source_sharded(tmp_path):
    """
    _restore_fp32_params_from_source must work for sharded safetensors checkpoints
    (model.safetensors.index.json + multiple shard files), as used by large models like Kimi-K2.5.
    """
    import json

    from quark.torch.utils.llm.model_preparation import _restore_fp32_params_from_source

    fp32_val = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    bf16_val = torch.tensor([0.5, 1.5], dtype=torch.bfloat16)

    source_dir = tmp_path / "source"
    source_dir.mkdir()

    # Shard 1: contains the float32 param (e.g. gate bias in shard N of a large model).
    save_file(
        {"e_score_correction_bias": fp32_val},
        str(source_dir / "model-00001-of-00002.safetensors"),
        metadata={"format": "pt"},
    )
    # Shard 2: contains the bfloat16 weight.
    save_file(
        {"weight": bf16_val},
        str(source_dir / "model-00002-of-00002.safetensors"),
        metadata={"format": "pt"},
    )
    index = {
        "metadata": {"total_size": 20},
        "weight_map": {
            "e_score_correction_bias": "model-00001-of-00002.safetensors",
            "weight": "model-00002-of-00002.safetensors",
        },
    }
    with open(source_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    model = torch.nn.Module()
    model.e_score_correction_bias = torch.nn.Parameter(fp32_val.to(torch.bfloat16))
    model.weight = torch.nn.Parameter(bf16_val.clone())

    _restore_fp32_params_from_source(model, str(source_dir))

    assert model.e_score_correction_bias.dtype == torch.float32
    assert torch.equal(model.e_score_correction_bias.data, fp32_val)
    assert model.weight.dtype == torch.bfloat16


def test_restore_fp32_params_from_source_buffer(tmp_path):
    """
    _restore_fp32_params_from_source must also restore downcast float32 buffers, not just
    parameters, since some model classes register correction biases as buffers.
    """
    from quark.torch.utils.llm.model_preparation import _restore_fp32_params_from_source

    fp32_val = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    save_file(
        {"e_score_correction_bias": fp32_val},
        str(source_dir / "model.safetensors"),
        metadata={"format": "pt"},
    )

    model = torch.nn.Module()
    model.register_buffer("e_score_correction_bias", fp32_val.to(torch.bfloat16))

    assert model.e_score_correction_bias.dtype == torch.bfloat16

    _restore_fp32_params_from_source(model, str(source_dir))

    assert model.e_score_correction_bias.dtype == torch.float32
    assert torch.equal(model.e_score_correction_bias.data, fp32_val)


def test_restore_fp32_params_from_source_no_safetensors_available(tmp_path):
    """_restore_fp32_params_from_source must no-op when safetensors is not installed."""
    from quark.torch.utils.llm import model_preparation

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.bfloat16))

    with patch.object(model_preparation, "is_safetensors_available", return_value=False):
        model_preparation._restore_fp32_params_from_source(model, str(tmp_path))

    assert model.weight.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "no_transformers",
    [
        pytest.param(True, id="transformers_unavailable"),
        pytest.param(False, id="cached_file_lookup_raises"),
    ],
)
def test_restore_fp32_params_from_source_unresolvable_ckpt_path_is_noop(tmp_path, no_transformers):
    """
    When ckpt_path doesn't exist locally and can't be resolved to a source checkpoint dir
    (either because transformers is unavailable, or because the HF cache lookup raises for
    a ckpt_path that isn't a cached HF repo id), it must be treated as best-effort and no-op.
    """
    from quark.torch.utils.llm import model_preparation

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.bfloat16))

    with ExitStack() as stack:
        if no_transformers:
            stack.enter_context(patch.object(model_preparation, "is_transformers_available", return_value=False))
        model_preparation._restore_fp32_params_from_source(model, "not-a-local-dir/and-not-a-hf-repo")

    assert model.weight.dtype == torch.bfloat16


def test_restore_fp32_params_from_source_no_safetensors_files(tmp_path):
    """When the checkpoint dir has no safetensors files (e.g. pytorch_model.bin only), it must no-op."""
    from quark.torch.utils.llm.model_preparation import _restore_fp32_params_from_source

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "pytorch_model.bin").write_bytes(b"not-a-real-checkpoint")

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.bfloat16))

    _restore_fp32_params_from_source(model, str(source_dir))

    assert model.weight.dtype == torch.bfloat16


def test_restore_fp32_params_from_source_unreadable_shard_is_skipped(tmp_path):
    """A shard that fails to open (corrupt/unreadable) must be skipped rather than raising."""
    import json

    from quark.torch.utils.llm.model_preparation import _restore_fp32_params_from_source

    fp32_val = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    save_file(
        {"e_score_correction_bias": fp32_val},
        str(source_dir / "model-00001-of-00002.safetensors"),
        metadata={"format": "pt"},
    )
    # Corrupt shard: not a valid safetensors file, so safe_open raises SafetensorError.
    (source_dir / "model-00002-of-00002.safetensors").write_bytes(b"not-a-real-safetensors-file")
    index = {
        "metadata": {"total_size": 20},
        "weight_map": {
            "e_score_correction_bias": "model-00001-of-00002.safetensors",
            "weight": "model-00002-of-00002.safetensors",
        },
    }
    with open(source_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    model = torch.nn.Module()
    model.e_score_correction_bias = torch.nn.Parameter(fp32_val.to(torch.bfloat16))

    _restore_fp32_params_from_source(model, str(source_dir))

    # The good shard is still processed despite the corrupt one being skipped.
    assert model.e_score_correction_bias.dtype == torch.float32
    assert torch.equal(model.e_score_correction_bias.data, fp32_val)


# TODO: Extend this test with all supported architectures.
@pytest.mark.parametrize(
    "model_spec",
    [
        ("qwen3_5", "optimum-intel-internal-testing/tiny-random-qwen3.5"),
        ("gemma4", "amd/tiny-random-gemma4-moe"),
        ("gemma4_unified", "optimum-intel-internal-testing/tiny-random-gemma4-unified"),
    ],
)
def test_quantization_and_export_validity(model_spec: tuple[str, str]):
    """
    Test Features:
        Import Format:            Json-safetensors
        Quantization Method:      mxfp4, via `LLMTemplate`

    Verifies that quantizing and exporting a model with an `LLMTemplate`-provided
    configuration preserves the original `config.json` `model_type`, and that layers
    matched by the template's `exclude_layers_name` are not quantized.
    """
    model_id = model_spec[1]

    original_checkpoint_dir = huggingface_hub.snapshot_download(repo_id=model_id, repo_type="model")
    original_checkpoint_weights = _load_weights_from_safetensors(original_checkpoint_dir)

    model, _ = get_model(model_id, device=torch_device)
    original_model_type = model.config.model_type

    assert original_model_type == model_spec[0]

    # Source keys that are not materialized as parameters/buffers in the loaded `nn.Module` (e.g.
    # weights dropped through `_keys_to_ignore_on_load_unexpected`) must still be carried over into
    # the exported checkpoint untouched. They are matched by value rather than by name: transformers
    # renames some submodules on load (gemma4_unified loads the vision block `vision_embedder` as
    # `embed_vision.multimodal_embedder`), and export emits the loaded name, so the source key string
    # is not expected to survive verbatim -- only the tensor value must.
    loaded_state_dict_keys = set(model.state_dict().keys())
    keys_missing_from_model = set(original_checkpoint_weights.keys()) - loaded_state_dict_keys

    template = LLMTemplate.get(model.config.model_type)
    quant_config = template.get_config("mxfp4")

    model.eval()

    preprocess_for_quantization(model)

    quantizer = ModelQuantizer(copy.deepcopy(quant_config))
    quant_model = quantizer.quantize_model(model, dataloader=None)
    quant_model = quantizer.freeze(quant_model)

    with tempfile.TemporaryDirectory() as export_dir:
        with torch.no_grad():
            export_safetensors(
                model=quant_model,
                output_dir=export_dir,
                weight_format="real_quantized",
                pack_method="reorder",
            )

        exported_config = AutoConfig.from_pretrained(export_dir, trust_remote_code=True)
        assert exported_config.model_type == original_model_type

        # Match carried-over weights by value, consuming each exported tensor at most once so that N
        # identical (e.g. all-zero bias) source weights require N surviving copies in the export.
        exported_values = list(_load_weights_from_safetensors(export_dir).values())
        for key in keys_missing_from_model:
            source_value = original_checkpoint_weights[key]
            match_index = next(
                (
                    index
                    for index, exported_value in enumerate(exported_values)
                    if exported_value.shape == source_value.shape and torch.equal(exported_value, source_value)
                ),
                None,
            )
            assert match_index is not None, f"carried-over weight for {key} missing from exported checkpoint"
            exported_values.pop(match_index)

        with safe_open(Path(export_dir, "model.safetensors"), framework="pt") as f:
            checkpoint_keys = list(f.keys())

        for excluded_layer_name in template.exclude_layers_name:
            excluded_pattern = excluded_layer_name.strip("*")
            assert not any(excluded_pattern in key and key.endswith("weight_scale") for key in checkpoint_keys)

        assert any(key.endswith("weight_scale") for key in checkpoint_keys)
