#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import importlib
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from quark.common.utils.testing_utils import torch_device
from quark.torch.algorithm.awq.awq import AwqProcessor
from quark.torch.algorithm.awq.scale import apply_scale, scale_ln_fcs
from quark.torch.algorithm.utils.utils import get_model_type_norm_constant, get_num_attn_heads_from_model
from quark.torch.utils.exceptions import LossError


class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(2 * output_size, 2 * output_size)

    def forward(self, x):
        x = self.gelu(x)
        x = self.fc1(x)
        x = torch.cat((x, x), dim=1)
        x = self.fc2(x)
        return x


def test_apply_scale_for_gelu_fc():
    model = SimpleNN(input_size=6, output_size=6)
    scale = torch.Tensor([0.8687, 1.0146, 0.8218, 0.8765, 0.8521, 0.9272])
    scales_list = [("gelu", ("fc1",), scale)]
    weight_old = model.fc1.weight.data
    weight_golden = weight_old * scale
    apply_scale(model, scales_list)
    assert torch.equal(model.fc1.weight.data, weight_golden)


def test_apply_scale_for_fc_fc():
    model = SimpleNN(input_size=3, output_size=3)
    scale = torch.Tensor([0.8687, 1.0146, 0.8218, 0.8765, 0.8521, 0.9272])
    scales_list = [("fc1", ("fc2",), scale)]
    weight_old = model.fc2.weight.data
    weight_old.mul_(scale.to(model.fc2.weight.device).view(1, -1))
    apply_scale(model, scales_list, num_attention_heads=2, num_key_value_heads=1)
    assert torch.equal(model.fc2.weight.data, weight_old)


def test_awq_save_activation_scales():
    """Test that AwqProcessor.apply() saves activation scales when QUARK_SAVE_ACTIVATION_SCALES is enabled."""
    # Bypass __init__ and set only the attributes needed by apply()
    processor = object.__new__(AwqProcessor)
    processor.using_accelerate = False
    processor.modules = []  # empty so the AWQ loop doesn't execute
    processor.model = MagicMock()
    processor.model.config._attn_implementation = "eager"
    processor.recover_attn_implementation = "eager"
    processor.global_scales_list = [("layer.0", ("fc1",), torch.ones(10))]

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        scales_file = f.name

    try:
        with (
            patch("quark.torch.algorithm.awq.awq.QUARK_SAVE_ACTIVATION_SCALES", True),
            patch("quark.torch.algorithm.awq.awq.QUARK_ACTIVATION_SCALES_FILENAME", scales_file),
        ):
            processor.apply()

        # Verify the scales were saved correctly
        loaded = torch.load(scales_file, weights_only=False)
        assert len(loaded) == 1
        assert loaded[0][0] == "layer.0"
        assert loaded[0][1] == ("fc1",)
        assert torch.equal(loaded[0][2], torch.ones(10))
    finally:
        os.unlink(scales_file)


class _FakeWeightQuantizer:
    """Minimal stand-in for a weight quantizer: returns the weight unchanged (identity quant).

    Deliberately NOT an nn.Module, so ``linear_layer.modules()`` in ``pseudo_quantize_tensor``
    does not recurse into it and the enable/disable-observer calls are skipped. This keeps the
    quantization path functional (returns a real tensor) without pulling in the full observer
    machinery, so the grid search runs end to end on unpatched code.
    """

    group_size = -1

    def __init__(self):
        self.observer = MagicMock()

    def __call__(self, w):
        return w


class _QuantizedLinear(nn.Linear):
    """Linear with the ``_weight_quantizer`` attribute that ``_search_best_scale`` reads."""

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self._weight_quantizer = _FakeWeightQuantizer()


class _OverflowModule(nn.Module):
    """A module2inspect whose forward output overflows to NaN/Inf in a chosen dtype."""

    def __init__(self, fc, fill, out_dtype):
        super().__init__()
        self.fc = fc
        self._fill = fill
        self._out_dtype = out_dtype

    def forward(self, x, **kwargs):
        out = self.fc(x)
        return torch.full_like(out, self._fill, dtype=self._out_dtype)


def _make_search_processor():
    processor = object.__new__(AwqProcessor)
    processor.using_accelerate = False
    processor.module_kwargs = {}
    processor.inps = []
    processor.device = torch.device(torch_device)
    return processor


@pytest.mark.parametrize("fill", [float("nan"), float("inf")])
def test_search_best_scale_fails_fast_on_fp16_reference_recommends_bfloat16(fill):
    """
    Verify that ``_search_best_scale`` fails fast with a ``LossError`` recommending ``bfloat16``
    when a ``float16`` reference output overflows to NaN/Inf.

    Regression test for https://github.com/amd/Quark/issues/5: on Qwen2.5-7B/1.5B the unquantized
    forward overflows in float16, so the reference output fed to the grid search is all NaN. Every
    per-step loss is then non-finite and ``best_ratio`` stays at its init value. Before the fix the
    grid search ran to completion and raised the opaque "best_ratio was not updated" error (a bare
    ``raise Exception`` in even older versions) with no hint that the root cause is fp16 overflow.
    On unpatched code this test still fails, but with the generic error and no ``bfloat16`` text,
    so the ``match=`` assertion catches the regression.

    :param float fill: The non-finite value (``nan`` or ``inf``) the mock module forward emits to
        simulate an overflowing float16 reference output.
    """
    processor = _make_search_processor()

    fc = _QuantizedLinear(4, 4)
    module2inspect = _OverflowModule(fc, fill, out_dtype=torch.float16)
    inp = torch.randn(2, 4)

    with pytest.raises(LossError, match="bfloat16"):
        processor._search_best_scale(
            module=module2inspect,
            prev_op=nn.Identity(),
            layers=[fc],
            inp=inp,
            module2inspect=module2inspect,
        )


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_search_best_scale_non_fp16_overflow_does_not_recommend_switching_precision(out_dtype):
    """
    Verify that the fail-fast diagnosis is derived from the actual reference dtype: when a
    ``bfloat16`` or ``float32`` reference output overflows, the error must report that dtype and
    must NOT recommend switching precision (switching would not help, and recommending bfloat16
    for an already-bfloat16 run would be misleading).

    :param torch.dtype out_dtype: The non-float16 dtype of the overflowing reference output.
    """
    processor = _make_search_processor()

    fc = _QuantizedLinear(4, 4)
    module2inspect = _OverflowModule(fc, float("nan"), out_dtype=out_dtype)
    inp = torch.randn(2, 4)

    with pytest.raises(LossError) as exc_info:
        processor._search_best_scale(
            module=module2inspect,
            prev_op=nn.Identity(),
            layers=[fc],
            inp=inp,
            module2inspect=module2inspect,
        )

    message = str(exc_info.value)
    assert str(out_dtype) in message
    assert "retry with --data_type bfloat16" not in message
    assert "switching precision is unlikely to help" in message


def test_compute_loss_preserves_per_step_tolerance_on_nan():
    """
    Verify that ``_compute_loss`` does NOT raise when a single grid step overflows: it returns
    ``nan`` so that ``nan < best_error`` is False and that step is skipped, letting other steps win.

    This guards the tolerance behaviour that lets larger models (14B/32B in issue #5) succeed even
    when individual scale candidates overflow. A previous attempt to raise inside ``_compute_loss``
    on any NaN broke this and would have regressed those working cases.
    """
    processor = _make_search_processor()
    device = torch.device(torch_device)

    loss = processor._compute_loss(torch.zeros(4, 4), torch.full((4, 4), float("nan")), device)
    assert loss != loss  # nan, so `loss < best_error` is False and the step is skipped

    best_error = float("inf")
    assert not (loss < best_error)


def test_compute_loss_valid_returns_scalar():
    """Verify that ``_compute_loss`` returns a finite scalar MSE when both tensors are finite."""
    processor = _make_search_processor()
    device = torch.device(torch_device)

    loss = processor._compute_loss(torch.ones(4, 4), torch.zeros(4, 4), device)
    assert isinstance(loss, float)
    assert loss == pytest.approx(1.0)


def test_awq_memory_optimization_constant():
    """Test that QUARK_AWQ_MEMORY_OPTIMIZATION constant is correctly read from the environment."""
    import quark.torch.utils.constants as constants_module

    # Verify default (env var not set) is False
    assert constants_module.QUARK_AWQ_MEMORY_OPTIMIZATION is False

    # Reload with env var set to verify the constant and its log message are exercised
    with patch.dict(os.environ, {"QUARK_AWQ_MEMORY_OPTIMIZATION": "1"}):
        importlib.reload(constants_module)
        assert constants_module.QUARK_AWQ_MEMORY_OPTIMIZATION is True

    # Restore original state
    importlib.reload(constants_module)
    assert constants_module.QUARK_AWQ_MEMORY_OPTIMIZATION is False


# ---------------------------------------------------------------------------------------------------
# gemma4 AWQ scaling support
#
# Gemma4RMSNorm/Gemma4UnifiedRMSNorm apply their weight directly (no (1+w) offset), unlike pre-gemma4
# Gemma and qwen3_5 which use the centered ``(1 + w)`` convention. ``scale_ln_fcs`` branches on
# ``get_model_type_norm_constant``, so the tests below cover both the offset and plain div_ paths.
#
# The detection matches on the exact class name, so the toy norms below are named after the real
# transformers classes -- the names are what is under test.
# ---------------------------------------------------------------------------------------------------


class _Gemma4UnifiedRMSNorm(nn.Module):
    """Applies weight directly (no ``1 + w`` offset) -- must use the plain div_ scaling path."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))


class Gemma3RMSNorm(nn.Module):
    """Pre-gemma4 Gemma norm using the ``1 + w`` offset convention."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))


def test_gemma4_rmsnorm_uses_plain_div_path():
    """gemma4 RMSNorm must scale as ``w / s`` (no offset)."""
    hidden = 4
    ln = _Gemma4UnifiedRMSNorm(hidden)
    ln.weight.data = torch.tensor([2.0, 4.0, 6.0, 8.0])
    fc = nn.Linear(hidden, hidden, bias=False)
    scales = torch.tensor([2.0, 2.0, 2.0, 2.0])

    scale_ln_fcs(ln, [fc], scales)

    torch.testing.assert_close(ln.weight.data, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_pre_gemma4_rmsnorm_uses_offset_path():
    """Pre-gemma4 Gemma RMSNorm must keep the ``(w + 1) / s - 1`` offset."""
    hidden = 4
    ln = Gemma3RMSNorm(hidden)
    ln.weight.data = torch.tensor([1.0, 3.0, 5.0, 7.0])
    fc = nn.Linear(hidden, hidden, bias=False)
    scales = torch.tensor([2.0, 2.0, 2.0, 2.0])

    scale_ln_fcs(ln, [fc], scales)

    expected = (torch.tensor([1.0, 3.0, 5.0, 7.0]) + 1.0) / 2.0 - 1.0
    torch.testing.assert_close(ln.weight.data, expected)


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))


class _Qwen35RMSNormGated(nn.Module):
    """qwen3_5's gated norm initializes the weight at ones and applies it directly (no offset)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))


class _LlamaRMSNorm(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))


class MuseGlimmerTextCenteredRMSNorm(nn.Module):
    """Muse-Glimmer's centered norm; uses the same ``1 + w`` convention as pre-gemma4 Gemma/qwen3_5."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))


def test_get_model_type_norm_constant():
    """Centered RMSNorm (pre-gemma4 Gemma, qwen3_5, Muse-Glimmer) -> 1.0; gemma4/gated/standard -> 0.0."""
    hidden = 4
    assert get_model_type_norm_constant(Gemma3RMSNorm(hidden)) == 1.0
    assert get_model_type_norm_constant(Qwen3_5RMSNorm(hidden)) == 1.0
    assert get_model_type_norm_constant(MuseGlimmerTextCenteredRMSNorm(hidden)) == 1.0
    assert get_model_type_norm_constant(_Gemma4UnifiedRMSNorm(hidden)) == 0.0
    assert get_model_type_norm_constant(_Qwen35RMSNormGated(hidden)) == 0.0
    assert get_model_type_norm_constant(_LlamaRMSNorm(hidden)) == 0.0


class _RMSNormWithForward(nn.Module):
    """Toy RMSNorm applying ``weight + c``, where ``c`` follows from the class name."""

    def __init__(self, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(hidden) + 0.5)
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = get_model_type_norm_constant(self)
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight + c) * x


# A different unit-offset entry than the `Gemma3RMSNorm` stub above, to avoid a name clash.
class GemmaRMSNorm(_RMSNormWithForward):
    pass


class _LlamaRMSNormWithForward(_RMSNormWithForward):
    pass


@pytest.mark.parametrize("norm_class", [_LlamaRMSNormWithForward, GemmaRMSNorm])
def test_scale_ln_fcs_is_non_destructive(norm_class):
    """Scaling the norm down and the following linear up must leave the composition unchanged."""
    torch.manual_seed(0)
    hidden = 64
    norm = norm_class(hidden)
    linear = nn.Linear(hidden, hidden, bias=False)
    inp = torch.randn(4, hidden)

    with torch.no_grad():
        output_original = linear(norm(inp))

    scale_ln_fcs(ln=norm, fcs=[linear], scales=torch.rand(hidden) + 0.5)

    with torch.no_grad():
        output_scaled = linear(norm(inp))

    assert float(torch.norm(output_original - output_scaled)) < 1e-4


# ---------------------------------------------------------------------------------------------------
# get_num_attn_heads_from_model on heterogeneous per-layer configs (QUARK-1102)
#
# transformers >= 5.15 gives models like gemma-4-12B-it a heterogeneous config: the head counts are
# defined per layer (full vs sliding attention), so reading a single global value raises
# ``AmbiguousGlobalPerLayerAttributeError`` (a RuntimeError, which ``hasattr`` does NOT swallow). The
# helper must degrade to the ``-1`` sentinel instead of propagating the crash; the sentinel disables
# the group-query-attention scale path, which is correct when no global head layout exists.
# ---------------------------------------------------------------------------------------------------


class _AmbiguousGlobalPerLayerAttributeError(RuntimeError):
    """Stand-in for the transformers >= 5.15 exception (also a RuntimeError, not AttributeError)."""


class _HeterogeneousConfig:
    """Config whose global head counts are ambiguous, like a per-layer gemma4_unified text config."""

    def __getattr__(self, name: str) -> object:
        if name in ("num_attention_heads", "num_key_value_heads", "head_dim"):
            raise _AmbiguousGlobalPerLayerAttributeError(name)
        raise AttributeError(name)


class _HomogeneousConfig:
    def __init__(self, num_attention_heads: int, num_key_value_heads: int) -> None:
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads


class _FakeModel:
    def __init__(self, config: object) -> None:
        self.config = config


def test_get_num_attn_heads_homogeneous_config():
    model = _FakeModel(_HomogeneousConfig(32, 8))
    assert get_num_attn_heads_from_model(model) == (32, 8)


def test_get_num_attn_heads_heterogeneous_config(monkeypatch):
    """A per-layer config must yield the ``(-1, -1)`` sentinel instead of raising."""
    import quark.torch.algorithm.utils.utils as utils_module

    monkeypatch.setattr(utils_module, "_HETEROGENEOUS_CONFIG_ERRORS", (_AmbiguousGlobalPerLayerAttributeError,))
    model = _FakeModel(_HeterogeneousConfig())
    assert get_num_attn_heads_from_model(model) == (-1, -1)


def test_get_num_attn_heads_vlm_heterogeneous_text_config(monkeypatch):
    """VLM top-level config delegating to a heterogeneous text_config also degrades to the sentinel."""
    import quark.torch.algorithm.utils.utils as utils_module

    monkeypatch.setattr(utils_module, "_HETEROGENEOUS_CONFIG_ERRORS", (_AmbiguousGlobalPerLayerAttributeError,))

    class _TopConfig:
        text_config = _HeterogeneousConfig()

    assert get_num_attn_heads_from_model(_FakeModel(_TopConfig())) == (-1, -1)


def test_get_num_attn_heads_no_config():
    assert get_num_attn_heads_from_model(object()) == (-1, -1)
