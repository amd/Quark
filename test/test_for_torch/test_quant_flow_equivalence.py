#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Cross-flow equivalence for `QConfig.quant_flow`.

`quant_flow` picks *where* weights live while quantizing, never *how* they are quantized,
so two flows given the same config and the same calibration data must produce the same
quantizer state and the same outputs.

Loaded on the same device, `standard` and `per_block` agree bit for bit: streaming blocks
through CPU RAM and back is a pure copy, so it perturbs nothing.

The one caveat is the *load* device, not the flow. Weight calibration runs before the block
loader is installed, so a CPU-loaded model calibrates weights on CPU. `quantize_quark.py`
forces a CPU load for `per_block`, and CPU and GPU weight scales differ by fp32 epsilon
(~6e-11 on this fixture, activation scales unaffected). That is a device effect the standard
flow shows too, so the tests below pin the flows against each other on one device.

`file2file` is a separate entry point (`direct_quantize_checkpoint`) that never builds the
model, so it cannot be driven through `quantize_model` alongside the other two.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quark.common.utils.testing_utils import skip_if_no_gpu
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, QuantFlow, RoundType, ScaleType
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver

pytest.importorskip("transformers")

VOCAB_SIZE = 64
SEQ_LEN = 8


def _tiny_causal_lm_checkpoint(tmp_path) -> str:
    """Save a small randomly initialised Llama checkpoint and return its directory.

    `per_block` reads the checkpoint directory off `model.config._name_or_path`, so the
    model has to come from `from_pretrained`.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=SEQ_LEN,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    save_dir = str(tmp_path / "tiny-llama")
    model.save_pretrained(save_dir)
    return save_dir


def _load(model_dir: str, device: torch.device) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    return model.to(device).eval()


def _calib_dataloader(device: torch.device) -> DataLoader:
    torch.manual_seed(1234)
    samples = [torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=device) for _ in range(4)]
    return DataLoader(samples, batch_size=None)


def _static_act_config() -> QConfig:
    spec = QTensorConfig(
        dtype=Dtype.int8,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
        qscheme=QSchemeType.per_tensor,
        symmetric=True,
        round_method=RoundType.half_even,
        scale_type=ScaleType.float,
    )
    return QConfig(global_quant_config=QLayerConfig(weight=spec, input_tensors=spec), exclude=["lm_head"])


def _quantizer_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Every quantizer scale / zero_point in the model, keyed by module path."""
    state = {}
    for name, module in model.named_modules():
        for attr in ("scale", "zero_point"):
            value = getattr(module, attr, None)
            if isinstance(value, torch.Tensor) and not value.is_meta:
                state[f"{name}.{attr}"] = value.detach().float().cpu()
    return state


def _assert_same_quantizer_state(reference: dict[str, torch.Tensor], other: dict[str, torch.Tensor]) -> None:
    assert reference, "no quantizer state was collected; the comparison would be vacuous"
    assert sorted(reference) == sorted(other)
    mismatched = [key for key in reference if not torch.equal(reference[key], other[key])]
    assert not mismatched, f"quantizer state differs between flows for {mismatched[:8]}"


def _quantize(model_dir: str, device: torch.device, flow: QuantFlow, gpu_resident_blocks: int = 0) -> nn.Module:
    config = _static_act_config()
    config.quant_flow = flow
    config.gpu_resident_blocks = gpu_resident_blocks
    model = ModelQuantizer(config).quantize_model(_load(model_dir, device), _calib_dataloader(device))
    return model.to("cpu")


@skip_if_no_gpu
def test_per_block_matches_standard_flow(tmp_path):
    """Same device, same config: `per_block` must be indistinguishable from `standard`."""
    model_dir = _tiny_causal_lm_checkpoint(tmp_path)
    device = torch.device("cuda")
    input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

    standard_model = _quantize(model_dir, device, QuantFlow.standard)
    per_block_model = _quantize(model_dir, device, QuantFlow.per_block)

    _assert_same_quantizer_state(_quantizer_state(standard_model), _quantizer_state(per_block_model))

    with torch.no_grad():
        assert torch.equal(standard_model(input_ids).logits, per_block_model(input_ids).logits)


@skip_if_no_gpu
def test_per_block_gpu_resident_blocks_does_not_change_result(tmp_path):
    """`gpu_resident_blocks` is a memory knob; it must not move a single qparam."""
    model_dir = _tiny_causal_lm_checkpoint(tmp_path)
    device = torch.device("cuda")

    states = [
        _quantizer_state(_quantize(model_dir, device, QuantFlow.per_block, gpu_resident_blocks=n)) for n in (0, 2)
    ]
    _assert_same_quantizer_state(states[0], states[1])


@pytest.mark.parametrize(("scheme", "per_block_supported"), [("nvfp4", True), ("mxfp4", False)])
def test_per_block_supports_dynamic_activations_with_a_per_tensor_scale(scheme, per_block_supported):
    """Both schemes quantize activations dynamically; only the one with a per-tensor scale
    to calibrate runs a forward pass, and only that one per_block can drive. Pins the rule
    against being re-tightened to "static activations only", which would lock out nvfp4."""
    from quark.torch import LLMTemplate

    verifier = ModelQuantizer(LLMTemplate.get("llama").get_config(scheme)).config_verifier
    assert verifier.is_act_dynamic

    # Mirrors the guard in `ModelQuantizer.quantize_model`.
    rejected = (
        verifier.is_all_dynamic
        or verifier.is_weight_only
        or (verifier.is_act_dynamic and not verifier.is_act_contain_scale_per_tensor)
    )
    assert rejected is not per_block_supported


@skip_if_no_gpu
def test_per_block_streams_blocks_onto_the_calibration_device(tmp_path):
    """Blocks must land on the device the calibration data is on, not a hardcoded `cuda`."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs a second GPU to tell a device choice apart from the default")

    from quark.torch.quantization.api import _infer_calibration_device

    device = torch.device("cuda:1")
    assert _infer_calibration_device(_calib_dataloader(device)) == device

    # And end to end: a block streamed onto the wrong device would fail the forward.
    config = _static_act_config()
    config.quant_flow = QuantFlow.per_block
    model = _load(_tiny_causal_lm_checkpoint(tmp_path), torch.device("cpu"))
    ModelQuantizer(config).quantize_model(model, _calib_dataloader(device))


def test_per_block_rejects_cpu_calibration_data(tmp_path):
    """CPU is the offload target, so streaming blocks onto it is meaningless; say so."""
    model_dir = _tiny_causal_lm_checkpoint(tmp_path)
    cpu = torch.device("cpu")

    config = _static_act_config()
    config.quant_flow = QuantFlow.per_block
    with pytest.raises(ValueError, match="calibration data is on CPU"):
        ModelQuantizer(config).quantize_model(_load(model_dir, cpu), _calib_dataloader(cpu))


@skip_if_no_gpu
def test_per_block_rejects_multi_device_placement(tmp_path):
    """An accelerate-sharded model would leave activations and weights on different devices."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs a second GPU to build a multi-device placement")

    from transformers import AutoModelForCausalLM

    model_dir = _tiny_causal_lm_checkpoint(tmp_path)
    device_map = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.2": 1,
        "model.layers.3": 1,
        "model.norm": 1,
        "lm_head": 1,
    }
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, device_map=device_map).eval()

    config = _static_act_config()
    config.quant_flow = QuantFlow.per_block
    with pytest.raises(ValueError, match="sharded over"):
        ModelQuantizer(config).quantize_model(model, _calib_dataloader(torch.device("cuda:0")))


@skip_if_no_gpu
def test_calibration_device_is_found_in_a_batchfeature():
    """`BatchFeature` is a `UserDict`, not a `dict`; a dict-only check misses every VLM batch."""
    from transformers.feature_extraction_utils import BatchFeature

    from quark.torch.quantization.api import _infer_calibration_device

    device = torch.device("cuda:0")
    batch = BatchFeature({"input_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device)})
    for dataloader in (
        DataLoader([batch], batch_size=None, collate_fn=lambda x: x),
        DataLoader([[batch]], batch_size=None, collate_fn=lambda x: x),
    ):
        assert _infer_calibration_device(dataloader) == device
