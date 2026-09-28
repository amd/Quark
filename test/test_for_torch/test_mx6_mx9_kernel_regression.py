#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Regression tests pinning the three MX6/MX9 fake-quantize kernel defects.

Each test fails on the pre-fix kernel and passes on the fixed one.

The cross-backend oracle in ``test_microexponent_cross_backend.py`` covers the same
ground by differential comparison against ONNX, but it is gated on
``importorskip("onnxruntime")`` and skips wholesale in a torch-only environment. These
tests deliberately depend on torch alone, so the kernel stays pinned in every job.
"""

import numpy as np
import pytest
import torch

from quark.torch.kernel.hw_emulation.hw_emulation_interface import fake_quantize_mx6_mx9
from quark.torch.quantization.config.type import MX6, MX9

# Taken from the canonical definitions rather than restated. ``test_microexponent_cross_backend``
# deliberately keeps its own copy so that a typo in these values cannot agree with itself; that
# reasoning does not carry here, because what this file pins is behaviour that holds whatever the
# field width is -- the defect made the code width scale with the data instead of with the field.
# The values themselves are pinned by ``test_llm_template.test_microexponent_format_parameters``.
FORMATS = {"mx6": MX6, "mx9": MX9}

# All MicroeXponent formats share one two-level block geometry, so one of them speaks for all.
BLOCK_SIZE = MX6.k1
SUB_BLOCK_SIZE = MX6.k2
SUB_BLOCK_SHIFT_BITS = MX6.d2


def torch_mx_qdq(x: torch.Tensor, quant_bit: int) -> np.ndarray:
    out: np.ndarray = fake_quantize_mx6_mx9(
        x, axis=-1, block_size=BLOCK_SIZE, quant_bit=quant_bit, sub_block_size=SUB_BLOCK_SIZE
    ).numpy()
    return out


def _max_exponent(values: np.ndarray) -> np.ndarray:
    """Floor-log2 of the largest magnitude along the last axis. All-zero groups give 0."""
    amax = np.abs(values).max(axis=-1)
    exponent: np.ndarray = np.floor(np.log2(np.where(amax == 0, 1.0, amax)))
    return exponent


def sub_block_shift(original: np.ndarray) -> np.ndarray:
    """Per-element second-level shift: how far each sub-block sits below its block."""
    blocks = original.reshape(-1, BLOCK_SIZE)
    sub = blocks.reshape(-1, BLOCK_SIZE // SUB_BLOCK_SIZE, SUB_BLOCK_SIZE)
    shift = np.minimum(_max_exponent(blocks)[:, None] - _max_exponent(sub), (1 << SUB_BLOCK_SHIFT_BITS) - 1)
    per_element: np.ndarray = np.repeat(shift, SUB_BLOCK_SIZE, axis=1)
    return per_element


def element_codes(qdq: np.ndarray, original: np.ndarray, mantissa_bits: int) -> np.ndarray:
    """Recover per-element integer codes, in units of the sub-block step each is encoded against.

    Normalizing by the block step instead would report a shifted sub-block's codes at half
    magnitude, hiding an over-wide code behind an arithmetic artefact.
    """
    e_max = _max_exponent(original.reshape(-1, BLOCK_SIZE))
    step = 2.0 ** (e_max[:, None] - sub_block_shift(original) + 1 - mantissa_bits)
    codes: np.ndarray = np.abs(qdq.reshape(-1, BLOCK_SIZE)) / step
    return codes


def shifted_sample(seed: int = 0, blocks: int = 512) -> torch.Tensor:
    """Blocks whose sub-blocks straddle several binades, so the second-level shift fires."""
    rng = np.random.default_rng(seed)
    exponents = rng.integers(-4, 4, size=(blocks, BLOCK_SIZE))
    mantissas = rng.uniform(1.0, 2.0, size=(blocks, BLOCK_SIZE))
    signs = rng.choice([-1.0, 1.0], size=(blocks, BLOCK_SIZE))
    return torch.tensor(signs * mantissas * 2.0**exponents, dtype=torch.float32)


@pytest.mark.parametrize("fmt", list(FORMATS))
def test_element_codes_fit_element_field(fmt: str) -> None:
    """a shifted sub-block must not be handed a code wider than the element field."""
    spec = FORMATS[fmt]
    x = shifted_sample()
    original = x.numpy()

    # Guard against a vacuous pass: the input must actually exercise the shift path.
    assert sub_block_shift(original).max() > 0, "sample does not trigger any second-level shift"

    codes = element_codes(torch_mx_qdq(x, spec.element_bits), original, spec.mantissa_bits)
    widest = float(np.max(codes))
    # The element field carries a sign bit, so the widest magnitude is the mantissa alone.
    # This is the ONNX kernel's ``(1 << m_bfp) - 1`` bound.
    limit = 2**spec.mantissa_bits - 1
    assert widest <= limit + 1e-6, (
        f"{fmt}: element code {widest:g} exceeds the {spec.element_bits}-bit signed element field "
        f"(max {limit}); quant_max is derived from the block exponent instead of the sub-block one"
    )


@pytest.mark.parametrize("fmt", list(FORMATS))
def test_subnormals_flush_to_zero(fmt: str) -> None:
    """subnormals report IEEE exponent -127, so they must be flushed, not stepped."""
    x = torch.full((1, BLOCK_SIZE), 1e-40, dtype=torch.float32)
    out = torch_mx_qdq(x, FORMATS[fmt].element_bits)
    assert np.all(out == 0.0), f"{fmt}: subnormal block quantized to {out.flatten()[0]:g} instead of 0"


@pytest.mark.parametrize("fmt", list(FORMATS))
def test_non_finite_inputs_propagate(fmt: str) -> None:
    """clamping ±Inf to quant_max turns an overflow into a plausible finite number."""
    x = torch.ones((1, BLOCK_SIZE), dtype=torch.float32)
    x[0, 0] = float("inf")
    x[0, 1] = float("-inf")
    x[0, 2] = float("nan")
    out = torch_mx_qdq(x, FORMATS[fmt].element_bits).flatten()

    assert np.isposinf(out[0]), f"{fmt}: +Inf became {out[0]:g}"
    assert np.isneginf(out[1]), f"{fmt}: -Inf became {out[1]:g}"
    assert np.isnan(out[2]), f"{fmt}: NaN became {out[2]:g}"
