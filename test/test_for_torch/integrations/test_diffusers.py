#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from quark.common.utils.import_utils import is_diffusers_available
from quark.common.utils.testing_utils import use_temporary_directory
from quark.torch import ModelQuantizer, export_safetensors
from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
from quark.torch.quantization.config.config import Int8PerTensorSpec, QConfig, QLayerConfig
from quark.torch.quantization.nn.modules.mixin import QuantMixin

if is_diffusers_available():
    from diffusers import UNet2DModel

    import quark.integrations.diffusers  # noqa: F401


requires_diffusers = pytest.mark.skipif(not is_diffusers_available(), reason="diffusers not installed")


def _make_tiny_unet() -> "UNet2DModel":
    """Create a minimal UNet2DModel for testing (< 1M params)."""
    return UNet2DModel(
        sample_size=32,
        in_channels=1,
        out_channels=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
    )


def _make_mock_quant_config(**overrides: object) -> SimpleNamespace:
    """Create a minimal object that quacks like QConfig for testing."""
    defaults = {"quant_method": "quark", "quant_scheme": "w_int8", "group_size": 128}
    defaults.update(overrides)
    return SimpleNamespace(to_dict=lambda: dict(defaults))


@requires_diffusers
@use_temporary_directory
def test_diffusers_export_roundtrip(tmpdir: str):
    """
    Test Features:
        Export Format:  diffusers / safetensors
        Verify exported model can be reloaded via from_pretrained.
    """
    model = _make_tiny_unet()
    original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    export_safetensors(model, tmpdir)

    reloaded = UNet2DModel.from_pretrained(tmpdir)

    for key in original_state_dict:
        assert torch.equal(reloaded.state_dict()[key], original_state_dict[key]), f"Mismatch after reload for {key}"


@requires_diffusers
@use_temporary_directory
def test_diffusers_export_ignores_transformers_kwargs(tmpdir: str):
    """
    Test Features:
        Export Format:  diffusers / safetensors
        Verify transformers-specific kwargs do not affect diffusers export path.
    """
    model = _make_tiny_unet()
    export_safetensors(model, tmpdir, custom_mode="quark", weight_format="real_quantized", pack_method="reorder")

    filenames = {f.name for f in Path(tmpdir).iterdir()}
    assert "diffusion_pytorch_model.safetensors" in filenames


@requires_diffusers
@use_temporary_directory
def test_diffusers_export_embeds_quantization_config(tmpdir: str):
    """
    Test Features:
        Export Format:  diffusers / safetensors
        Verify quantization_config is written into config.json when model has quant_config.
    """
    model = _make_tiny_unet()
    model.quant_config = _make_mock_quant_config()  # type: ignore[attr-defined]

    export_safetensors(model, tmpdir)

    with open(Path(tmpdir) / "config.json") as f:
        config = json.load(f)

    assert "quantization_config" in config
    qc = config["quantization_config"]
    assert qc["quant_method"] == "quark"
    assert qc["quant_scheme"] == "w_int8"
    assert qc["group_size"] == 128


@requires_diffusers
@use_temporary_directory
def test_diffusers_export_no_quantization_config_when_unquantized(tmpdir: str):
    """
    Test Features:
        Export Format:  diffusers / safetensors
        Verify config.json does not contain quantization_config for un-quantized models.
    """
    model = _make_tiny_unet()
    export_safetensors(model, tmpdir)

    with open(Path(tmpdir) / "config.json") as f:
        config = json.load(f)

    assert "quantization_config" not in config


@requires_diffusers
@use_temporary_directory
def test_diffusers_export_quantized_model(tmpdir: str):
    """
    Test Features:
        Export Format:  diffusers / safetensors
        Quantize a tiny UNet with ModelQuantizer, export, and verify
        that the real QConfig serializes correctly into config.json.
    """
    model = _make_tiny_unet()

    weight_spec = Int8PerTensorSpec(
        observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
    ).to_quantization_spec()
    quant_config = QConfig(global_quant_config=QLayerConfig(weight=weight_spec))

    quantizer = ModelQuantizer(quant_config)
    model = quantizer.quantize_model(model, dataloader=None)

    export_safetensors(model, tmpdir)

    filenames = {f.name for f in Path(tmpdir).iterdir()}
    assert "diffusion_pytorch_model.safetensors" in filenames
    assert "config.json" in filenames

    with open(Path(tmpdir) / "config.json") as f:
        config = json.load(f)

    assert "quantization_config" in config
    qc = config["quantization_config"]
    assert qc["quant_method"] == "quark"
    assert "global_quant_config" in qc


@requires_diffusers
def test_on_the_fly_quantizer_rejects_activation_config():
    """The on-the-fly load path rejects activation-quantized configs early.

    ``_process_model_before_weight_loading`` with ``pre_quantized=False``
    must raise ``NotImplementedError`` for a config that declares
    activation quantizers, directing the user to the offline workflow.
    """
    from quark.integrations.diffusers import QuarkDiffusersQuantizer, QuarkQuantizationConfig

    weight_spec = Int8PerTensorSpec(
        observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
    ).to_quantization_spec()
    w8a8 = QConfig(global_quant_config=QLayerConfig(weight=weight_spec, input_tensors=weight_spec))

    quantizer = QuarkDiffusersQuantizer(QuarkQuantizationConfig(w8a8.to_dict()), pre_quantized=False)
    model = _make_tiny_unet()

    with pytest.raises(NotImplementedError, match="weight-only"):
        quantizer._process_model_before_weight_loading(model)


@requires_diffusers
def test_on_the_fly_quantizer_weight_only_quantizes_in_place():
    """The on-the-fly load path quantizes a vanilla UNet with a weight-only config."""
    from quark.integrations.diffusers import QuarkDiffusersQuantizer, QuarkQuantizationConfig

    weight_spec = Int8PerTensorSpec(
        observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
    ).to_quantization_spec()
    weight_only = QConfig(global_quant_config=QLayerConfig(weight=weight_spec))

    quantizer = QuarkDiffusersQuantizer(QuarkQuantizationConfig(weight_only.to_dict()), pre_quantized=False)
    model = _make_tiny_unet()

    # Simulate the from_pretrained sequence: before-load (no transform for
    # the on-the-fly path) then after-load (quantize in place).
    quantizer._process_model_before_weight_loading(model)
    quantizer._process_model_after_weight_loading(model)

    assert any(isinstance(m, QuantMixin) for m in model.modules()), (
        "on-the-fly weight-only quantization should produce QuantMixin modules"
    )


def _quantize_tiny_unet(model: "UNet2DModel") -> tuple["UNet2DModel", QConfig]:
    """Quantize a tiny UNet with weight-only INT8 and return (model, config)."""
    weight_spec = Int8PerTensorSpec(
        observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
    ).to_quantization_spec()
    qconfig = QConfig(global_quant_config=QLayerConfig(weight=weight_spec))
    quantizer = ModelQuantizer(qconfig)
    model = quantizer.quantize_model(model, dataloader=None)
    return model, qconfig


@requires_diffusers
@pytest.mark.parametrize("weight_format", ["fake_quantized", "real_quantized"])
def test_quantized_export_reload_roundtrip(weight_format: str):
    """
    Test Features:
        Export Format:  diffusers / safetensors
        Quantize, export, reload via from_pretrained, and verify that
        the reloaded model contains QuantLinear/QuantConv2d layers with
        matching weights and produces the same forward-pass output.

    The two formats reload into different module types, which is the point of
    parametrizing:

    * ``fake_quantized`` -- QuantLinear/QuantConv2d with FakeQuantize submodules,
      scales stored as ``<layer>._weight_quantizer.scale``.
    * ``real_quantized`` -- packed ``QParamsLinear`` (not a ``QuantMixin``) with scales at
      ``<layer>.weight_quantizer.scale``, plus fake-quantized QuantConv2d for the module
      types that have no packed form.

    Regression guard: running ModelPostProcessor unconditionally renamed the Linear
    scales to the packed convention even for ``fake_quantized``, so the rebuilt
    ``_weight_quantizer.scale`` buffers had nothing to load from and stayed on the
    meta device. The key *count* was unchanged (only Linear scales moved, Conv were
    untouched), so only a name-aware check catches it.
    """
    model, _ = _quantize_tiny_unet(_make_tiny_unet())
    assert any(isinstance(m, QuantMixin) for m in model.modules()), (
        "Quantization should produce at least one QuantMixin module"
    )
    model.eval()

    sample = torch.randn(1, 1, 32, 32)
    timestep = torch.tensor([1.0])
    with torch.no_grad():
        original_output = model(sample, timestep).sample

    with tempfile.TemporaryDirectory() as tmpdir:
        export_safetensors(model, tmpdir, weight_format=weight_format)

        reloaded = UNet2DModel.from_pretrained(tmpdir)
        reloaded.eval()

        # real_quantized reloads as packed QParamsLinear, which is deliberately NOT a
        # QuantMixin (MRO: QParamsLinear -> Linear -> QparamsOperator -> QuarkLinearBase).
        quantized_types: tuple[type, ...] = (QuantMixin, QParamsLinear)
        has_quant_module = any(isinstance(m, quantized_types) for m in reloaded.modules())
        assert has_quant_module, (
            f"Reloaded model should contain at least one quantized module "
            f"(QuantMixin or QParamsLinear) for weight_format={weight_format}"
        )
        if weight_format == "real_quantized":
            assert any(isinstance(m, QParamsLinear) for m in reloaded.modules()), (
                "real_quantized reload should produce packed QParamsLinear modules"
            )
            # Only nn.Linear has a packed form. The other quantized module types must
            # still come back carrying their quantizers -- a plain Conv2d here means the
            # layer reloaded on its full-precision master weight and runs unquantized.
            assert not any(type(m) is torch.nn.Conv2d for m in reloaded.modules()), (
                "quantized Conv must not reload as a plain Conv2d holding the master weight"
            )

        with torch.no_grad():
            reloaded_output = reloaded(sample, timestep).sample

        assert original_output.shape == reloaded_output.shape
        assert torch.allclose(original_output, reloaded_output, atol=1e-5), (
            f"Max diff: {(original_output - reloaded_output).abs().max().item()}"
        )


@requires_diffusers
def test_diffusers_export_defaults_to_fake_quantized():
    """Packed output is opt-in for diffusers.

    ``export_safetensors`` defaults to ``real_quantized`` for LLMs, but a packed
    diffusers checkpoint is only readable by a Quark that has the packed reload path,
    so the diffusers exporter keeps the fake-quantized layout unless asked otherwise.
    """
    model, _ = _quantize_tiny_unet(_make_tiny_unet())
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        export_safetensors(model, tmpdir)

        with open(Path(tmpdir) / "config.json") as f:
            config = json.load(f)
        assert config["quantization_config"]["export"]["weight_format"] == "fake_quantized"

        reloaded = UNet2DModel.from_pretrained(tmpdir)
        assert not any(isinstance(m, QParamsLinear) for m in reloaded.modules()), (
            "the default diffusers export must not write a packed checkpoint"
        )
