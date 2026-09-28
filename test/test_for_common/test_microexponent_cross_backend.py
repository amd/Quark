#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Cross-backend consistency oracle for the MX6 / MX9 quantization formats.

Quark implements these formats twice and independently:

* PyTorch -- :func:`quark.torch.kernel.hw_emulation.hw_emulation_interface.fake_quantize_mx6_mx9`,
  a pure-torch tensor formulation;
* ONNX -- the ``BFPQuantizeDequantize`` custom op with ``bfp_method="to_bfp_prime"``,
  a C++ custom-op kernel.

Nothing previously compared the two. These tests do. They assert bit-exact agreement -- the
divergences this oracle originally pinned turned out to share one root cause on each side, and
both fixes are one-line constant changes -- so a future change to either side is caught rather
than silently absorbed.

The ONNX fix (#6177) has landed; the torch one (#6173) has not. Until it does, the assertions
that depend on it are xfailed via ``pending_torch_kernel_fix``, which is driven by a probe of
the installed kernel rather than hard-coded -- see :func:`_torch_kernel_has_mx_fixes`. The
ONNX-side assertions are deliberately left ungated so this file still catches a regression
there in the meantime.

Requires ``onnxruntime`` plus the Quark ONNX custom-op library; skipped otherwise.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

onnxruntime = pytest.importorskip("onnxruntime", reason="cross-backend oracle needs onnxruntime")
onnx = pytest.importorskip("onnx", reason="cross-backend oracle needs onnx")

from onnx import helper  # noqa: E402
from onnx.onnx_ml_pb2 import TensorProto  # noqa: E402

from quark.onnx.operators.custom_ops import _COP_BFP_OP_NAME, _COP_DOMAIN, get_library_path  # noqa: E402
from quark.torch.kernel.hw_emulation.hw_emulation_interface import fake_quantize_mx6_mx9  # noqa: E402

# Per-format parameters, written out here rather than imported from
# ``quark.common.data_type``. This file is an oracle, so it carries its own copy of the
# values under test -- importing the constants that the implementations are built from
# would let a typo in them agree with itself.
#
#          (torch quant_bit, onnx bit_width, element mantissa bits)
FORMATS = {
    "mx6": (5, 13, 4),
    "mx9": (8, 16, 7),
}
BLOCK_SIZE = 16
SUB_BLOCK_SIZE = 2
SUB_BLOCK_SHIFT_BITS = 1  # the second-level shift field is one bit wide
PY3_ROUND = 2  # round-half-to-even, matching torch.round


def _custom_ops_available() -> bool:
    try:
        return os.path.exists(get_library_path())
    except Exception:  # pragma: no cover - environment probe only
        return False


requires_custom_ops = pytest.mark.skipif(
    not _custom_ops_available(), reason="Quark ONNX custom-op library is not built"
)


def onnx_mx_qdq(x: np.ndarray, bit_width: int, rounding_mode: int = PY3_ROUND) -> np.ndarray:
    """Quantize-dequantize ``x`` (2-D, blocks along axis 1) via BFPQuantizeDequantize."""
    node = helper.make_node(
        _COP_BFP_OP_NAME,
        ["input"],
        ["out"],
        domain=_COP_DOMAIN,
        bfp_method="to_bfp_prime",
        axis=1,
        bit_width=bit_width,
        block_size=BLOCK_SIZE,
        sub_block_size=SUB_BLOCK_SIZE,
        sub_block_shift_bits=SUB_BLOCK_SHIFT_BITS,
        rounding_mode=rounding_mode,
    )
    graph = helper.make_graph(
        [node],
        "microexponent-oracle",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, None)],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, ir_version=9, opset_imports=[helper.make_operatorsetid("", 19)])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.onnx")
        onnx.save(model, path)
        so = onnxruntime.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        session = onnxruntime.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        out: np.ndarray = session.run(None, {"input": x})[0]
        return out


def torch_mx_qdq(x: torch.Tensor, quant_bit: int) -> np.ndarray:
    out: np.ndarray = fake_quantize_mx6_mx9(
        x, axis=-1, block_size=BLOCK_SIZE, quant_bit=quant_bit, sub_block_size=SUB_BLOCK_SIZE
    ).numpy()
    return out


def _torch_kernel_has_mx_fixes() -> bool:
    """Probe whether the installed torch kernel carries the MX6/MX9 fixes from #6173.

    #6173 changes three things at once -- ``quant_max`` becomes a constant, subnormals flush to
    zero, and non-finites pass through instead of saturating -- so a single probe gates all
    three. Subnormal flush is the detector because it is an exact binary with no tolerance
    question: the pre-#6173 kernel derives a step from IEEE exponent -127 and returns
    ``9.18e-41`` for mx9, the fixed kernel returns ``0``.

    A capability probe, not an assertion. It lets the torch-dependent tests below be xfailed
    while #6173 is unmerged without hard-coding a state someone has to remember to undo: the
    marker it feeds is conditional, so it disappears on its own once the fixed kernel is
    installed.
    """
    probe = torch.full((1, BLOCK_SIZE), 1e-40, dtype=torch.float32)
    return bool(np.all(torch_mx_qdq(probe, quant_bit=8) == 0.0))


TORCH_KERNEL_HAS_MX_FIXES = _torch_kernel_has_mx_fixes()

# Applied to every assertion that depends on the torch kernel being fixed. Conditional, so it
# is inert once #6173 lands; ``strict=True``, so a suppressed test that starts passing fails
# loudly rather than lingering as a lie about what is covered. Nothing here needs deleting
# afterwards -- but the ONNX-side tests are deliberately left ungated, so this file still fails
# if the *other* backend regresses in the meantime.
pending_torch_kernel_fix = pytest.mark.xfail(
    not TORCH_KERNEL_HAS_MX_FIXES,
    reason=(
        "torch fake_quantize_mx6_mx9 predates #6173: quant_max is derived from the block "
        "exponent so shifted sub-blocks overflow the element field, subnormals are not "
        "flushed, and non-finite inputs saturate to quant_max"
    ),
    strict=True,
)


def _max_exponent(values: np.ndarray) -> np.ndarray:
    """Floor-log2 of the largest magnitude along the last axis. All-zero groups give 0."""
    amax = np.abs(values).max(axis=-1)
    exponent: np.ndarray = np.floor(np.log2(np.where(amax == 0, 1.0, amax)))
    return exponent


def sub_block_shift(original: np.ndarray) -> np.ndarray:
    """Per-element second-level shift: how far each sub-block sits below its block.

    Saturates at the width of the shift field, matching both kernels.
    """
    blocks = original.reshape(-1, BLOCK_SIZE)
    sub = blocks.reshape(-1, BLOCK_SIZE // SUB_BLOCK_SIZE, SUB_BLOCK_SIZE)
    shift = np.minimum(_max_exponent(blocks)[:, None] - _max_exponent(sub), (1 << SUB_BLOCK_SHIFT_BITS) - 1)
    per_element: np.ndarray = np.repeat(shift, SUB_BLOCK_SIZE, axis=1)
    return per_element


def element_codes(qdq: np.ndarray, original: np.ndarray, mantissa_bits: int) -> np.ndarray:
    """Recover per-element integer codes from a dequantized tensor.

    Codes are expressed in units of the step the element is actually encoded against --
    the *sub-block* step, which halves for each second-level shift. Normalizing by the
    block step instead would report a shifted sub-block's codes at half magnitude, hiding
    codes too wide for the element field behind an arithmetic artefact.
    """
    e_max = _max_exponent(original.reshape(-1, BLOCK_SIZE))
    step = 2.0 ** (e_max[:, None] - sub_block_shift(original) + 1 - mantissa_bits)
    codes: np.ndarray = np.abs(qdq.reshape(-1, BLOCK_SIZE)) / step
    return codes


def sample_tensor(seed: int = 0, rows: int = 64, cols: int = 256) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(rows, cols).float()


@pending_torch_kernel_fix
@requires_custom_ops
@pytest.mark.parametrize("fmt", list(FORMATS))
def test_torch_codes_fit_element_width(fmt: str) -> None:
    """Torch never exceeds the element field, at either second-level shift value.

    ``quant_max`` is the constant ``2^(quant_bit - 1) - 1``: the element field is
    ``quant_bit`` bits wide *including the sign*, and an element is encoded against its own
    sub-block's step, so the bound does not depend on the shift. It was previously derived
    from the *block* exponent while the step came from the sub-block, which worked out to
    ``2^(m + shift) - 1`` and let shifted sub-blocks emit an unrepresentable code -- so this
    assertion used to hold only where the shift was zero.

    :func:`test_onnx_codes_fit_element_width` asserts the same of the ONNX kernel.
    """
    quant_bit, _, mantissa_bits = FORMATS[fmt]
    x = sample_tensor()
    xn = x.numpy()
    codes = element_codes(torch_mx_qdq(x, quant_bit), xn, mantissa_bits)
    limit = 2**mantissa_bits - 1
    assert codes.max() <= limit + 1e-6, f"torch {fmt} emitted |code|={codes.max()} > {limit}"


@pending_torch_kernel_fix
@requires_custom_ops
@pytest.mark.parametrize("fmt", list(FORMATS))
@pytest.mark.parametrize("rounding_mode", [0, 1, 2])
def test_torch_matches_onnx(fmt: str, rounding_mode: int) -> None:
    """Torch and ONNX agree bit-exactly, at every rounding mode, in both formats.

    This assertion used to carry a whitelist for one known divergence: on a round-up that
    carried the code one step past the widest representable value, the ONNX kernel emitted it
    rather than clamping. Both kernels now clamp, so the whitelist is gone and this is a plain
    bit-exactness check -- which is what makes it useful as a regression guard, since either
    kernel drifting in any direction now fails it.

    Note that the two kernels are only guaranteed to agree because ``rounding_mode`` selects the
    same behaviour on both. That holds for the CPU path used here; the CUDA/HIP path implements a
    different contract for the same attribute values, which is tracked separately.
    """
    quant_bit, bit_width, _ = FORMATS[fmt]
    x = sample_tensor()
    xn = x.numpy()
    np.testing.assert_array_equal(
        torch_mx_qdq(x, quant_bit),
        onnx_mx_qdq(xn, bit_width, rounding_mode),
        err_msg=f"{fmt} at rounding_mode={rounding_mode}",
    )


@requires_custom_ops
@pytest.mark.parametrize("fmt", list(FORMATS))
def test_onnx_codes_fit_element_width(fmt: str) -> None:
    """ONNX never exceeds the element field either, at both second-level shift values.

    Mirrors :func:`test_torch_codes_fit_element_width`. ``round_bits`` in the CPU kernel used
    to bound the rounded mantissa at ``2^(m + 1) - 1``, one bit wider than the element field
    holds, so a value just under the block maximum rounded up into a code the format cannot
    represent. One constant produced both symptom classes PR 3 reported -- an unshifted
    overflow and a shifted one -- which is why a single fix closed both and both pinning tests
    were deleted in favour of this one.

    Asserted at every shift value rather than only where the shift is zero, because that is the
    guarantee a packed ``m + 1``-bit element layout actually needs: no code, anywhere, wider
    than the field it will be written into.
    """
    _, bit_width, mantissa_bits = FORMATS[fmt]
    x = sample_tensor()
    xn = x.numpy()
    codes = element_codes(onnx_mx_qdq(xn, bit_width), xn, mantissa_bits)
    limit = 2**mantissa_bits - 1
    shift = sub_block_shift(xn)
    for shift_value in (0, 1):
        selected = codes[shift == shift_value]
        if selected.size == 0:
            continue
        assert selected.max() <= limit + 1e-6, (
            f"ONNX {fmt} emitted |code|={selected.max()} > {limit} at shift={shift_value}"
        )


@requires_custom_ops
@pytest.mark.parametrize("fmt", list(FORMATS))
def test_zero_and_uniform_blocks_agree(fmt: str) -> None:
    """Degenerate blocks that the packing code must also survive."""
    quant_bit, bit_width, _ = FORMATS[fmt]
    cases = {
        "all zero": torch.zeros(1, BLOCK_SIZE),
        "single outlier": torch.cat([torch.tensor([[1e6]]), torch.full((1, BLOCK_SIZE - 1), 1e-6)], dim=1),
        "uniform": torch.full((1, BLOCK_SIZE), 3.0e38),
        "power of two": torch.full((1, BLOCK_SIZE), 0.25),
    }
    for name, x in cases.items():
        x = x.float()
        np.testing.assert_array_equal(
            torch_mx_qdq(x, quant_bit), onnx_mx_qdq(x.numpy(), bit_width), err_msg=f"{fmt} / {name}"
        )


@requires_custom_ops
@pytest.mark.parametrize("fmt", list(FORMATS))
def test_onnx_special_value_semantics(fmt: str) -> None:
    """Pin the ONNX kernel's chosen semantics for subnormal and non-finite inputs.

    Policy, not arithmetic -- the formats say nothing about either -- so this pins the kernel
    directly rather than comparing it to torch. Split out from the torch half so that it stays
    live while #6173 is unmerged: the two backends' choices are independent, and a regression
    in this one should not be hidden by a pending fix to the other.
    """
    _, bit_width, _ = FORMATS[fmt]

    subnormal = torch.full((1, BLOCK_SIZE), 1e-40, dtype=torch.float32)
    assert np.all(onnx_mx_qdq(subnormal.numpy(), bit_width) == 0.0), "ONNX should flush subnormals"

    non_finite = torch.tensor([[float("inf"), float("nan"), -float("inf")] + [1.0] * (BLOCK_SIZE - 3)])
    o = onnx_mx_qdq(non_finite.numpy(), bit_width)
    assert np.all(np.isnan(o)), "ONNX should NaN the whole block when the shared exponent is 0xff"


@pending_torch_kernel_fix
@requires_custom_ops
@pytest.mark.parametrize("fmt", list(FORMATS))
def test_torch_special_value_semantics(fmt: str) -> None:
    """Pin the torch kernel's chosen semantics, and its one deliberate divergence from ONNX.

    * **Subnormals: both backends flush to zero.** They carry an all-zero IEEE exponent field,
      so no block scale can represent them and the exponent one would derive is meaningless.
      Torch used to emit whatever fell out of a step built from exponent -127, differing per
      format (``1e-40`` gave ``0`` for mx6 and ``9.18e-41`` for mx9). #6173 makes it flush,
      matching ONNX, the CUDA/HIP kernel, and the hardware -- including the sign of the zero.

    * **Non-finites: the backends deliberately differ.** Torch propagates them per element,
      leaving the block's finite tail intact; ONNX NaNs the *entire* block once the shared
      exponent saturates to ``0xff``. Torch used to saturate ``+Inf`` to ``quant_max``, turning
      an upstream overflow into a plausible finite number -- the one option that is silently
      wrong. Callers must not depend on either kernel's choice.
    """
    quant_bit, bit_width, _ = FORMATS[fmt]

    subnormal = torch.full((1, BLOCK_SIZE), 1e-40, dtype=torch.float32)
    assert np.all(torch_mx_qdq(subnormal, quant_bit) == 0.0), "torch should flush subnormals"

    # A subnormal sharing a block with normal values: the block step dwarfs it, so it lands
    # on zero either way -- but the sign of that zero has to survive to stay bit-exact.
    mixed = torch.tensor([[1.0, -1e-40] + [1.0] * (BLOCK_SIZE - 2)], dtype=torch.float32)
    np.testing.assert_array_equal(
        torch_mx_qdq(mixed, quant_bit), onnx_mx_qdq(mixed.numpy(), bit_width), err_msg=f"{fmt} / mixed subnormal"
    )

    non_finite = torch.tensor([[float("inf"), float("nan"), -float("inf")] + [1.0] * (BLOCK_SIZE - 3)])
    t = torch_mx_qdq(non_finite, quant_bit)
    assert t[0, 0] == float("inf"), "torch should propagate +Inf, not saturate it to quant_max"
    assert np.isnan(t[0, 1]), "torch should propagate NaN"
    assert t[0, 2] == -float("inf"), "torch should propagate -Inf, not saturate it to -quant_max"
    assert np.isfinite(t[0, 3:]).all(), "torch should leave the finite tail of the block finite"
