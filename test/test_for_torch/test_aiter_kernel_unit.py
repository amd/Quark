#
# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Unit tests for aiter kernel wrappers (gemm.py and quant.py).

These tests mock the aiter backend to verify wrapper logic
(scale reshaping, input validation, fallback paths) without
requiring AMD GPU hardware or the aiter package.

Run:
    pytest test/test_for_torch/test_aiter_kernel_unit.py -v
"""

import sys
from unittest import mock

import pytest
import torch

_has_fp8 = hasattr(torch, "float8_e4m3fn")
requires_fp8 = pytest.mark.skipif(not _has_fp8, reason="torch.float8_e4m3fn not available")


# ============================================================
# Helper: run module source in a fresh namespace for import
# path coverage without triggering cascading reloads.
# ============================================================


def _exec_module_source(module, extra_sys_modules=None, patch_is_aiter=None):
    """Execute a module's source in a fresh namespace.

    Returns the resulting namespace dict.  Coverage sees the executed
    lines because the code object carries the original filename.
    """
    source_path = module.__file__
    with open(source_path, encoding="utf-8") as fh:
        source = fh.read()

    ns: dict = {"__name__": module.__name__, "__file__": source_path}

    patches = []
    if extra_sys_modules:
        patches.append(mock.patch.dict(sys.modules, extra_sys_modules))
    if patch_is_aiter is not None:
        patches.append(
            mock.patch(
                "quark.common.utils.import_utils.is_aiter_available",
                return_value=patch_is_aiter,
            )
        )

    for p in patches:
        p.start()
    try:
        exec(compile(source, source_path, "exec"), ns)  # noqa: S102
    finally:
        for p in reversed(patches):
            p.stop()

    return ns


# ============================================================
# gemm.py — module-level import paths
# ============================================================


class TestGemmModuleImport:
    """Cover the module-level try/except/else import block in gemm.py."""

    def test_import_success_with_mocked_aiter(self):
        """Aiter available and imports succeed -> kernels marked available."""
        import quark.torch.kernel.aiter.gemm as gemm_mod

        ns = _exec_module_source(
            gemm_mod,
            extra_sys_modules={
                "aiter": mock.MagicMock(),
                "aiter.ops": mock.MagicMock(),
                "aiter.ops.gemm_op_a8w8": mock.MagicMock(),
                "aiter.utility": mock.MagicMock(),
            },
            patch_is_aiter=True,
        )
        assert ns["_aiter_kernels_available"] is True
        assert ns["_aiter_import_error"] is None

    def test_import_failure_covers_except_path(self):
        """Package check passes but import raises ImportError."""
        import quark.torch.kernel.aiter.gemm as gemm_mod

        ns = _exec_module_source(gemm_mod, patch_is_aiter=True)
        assert ns["_aiter_kernels_available"] is False
        assert ns["_aiter_import_error"] is not None

    def test_else_path_when_package_unavailable(self):
        """Cover the else branch when _check_aiter_package returns False."""
        import quark.torch.kernel.aiter.gemm as gemm_mod

        ns = _exec_module_source(gemm_mod, patch_is_aiter=False)
        assert ns["_aiter_kernels_available"] is False
        assert ns["_aiter_import_error"] == "aiter package not installed"


# ============================================================
# gemm.py — _check_aiter_available
# ============================================================


class TestCheckAiterAvailable:
    def test_raises_when_unavailable(self):
        import quark.torch.kernel.aiter.gemm as gemm_mod

        with pytest.raises(ImportError, match="Native inference requires AMD Aiter"):
            gemm_mod._check_aiter_available()


# ============================================================
# gemm.py — gemm_fp8 wrapper (scale reshaping)
# ============================================================


@requires_fp8
class TestGemmFp8Wrapper:
    """Test scale reshaping logic in gemm_fp8 with mocked kernel."""

    def _call(self, gemm_mod, mk, x_scale, w_scale, m=4, k=16, n=8):
        with mock.patch.object(gemm_mod, "_aiter_gemm_a8w8", mk, create=True):
            XQ = torch.zeros(m, k, dtype=torch.float8_e4m3fn)
            WQ = torch.zeros(n, k, dtype=torch.float8_e4m3fn)
            return gemm_mod.gemm_fp8(XQ, WQ, x_scale, w_scale)

    def _mk(self, m=4, n=8):
        return mock.MagicMock(return_value=torch.zeros(m, n, dtype=torch.bfloat16))

    def test_scalar_x_scale_reshaped_to_2d(self):
        import quark.torch.kernel.aiter.gemm as gemm_mod

        mk = self._mk()
        self._call(gemm_mod, mk, x_scale=torch.tensor(1.0), w_scale=torch.ones(8))
        assert mk.call_args[0][2].shape == (1, 1)

    def test_1d_x_scale_reshaped_to_column(self):
        import quark.torch.kernel.aiter.gemm as gemm_mod

        mk = self._mk()
        self._call(gemm_mod, mk, x_scale=torch.ones(4), w_scale=torch.ones(8))
        assert mk.call_args[0][2].shape == (4, 1)

    def test_scalar_w_scale_reshaped_to_1d(self):
        import quark.torch.kernel.aiter.gemm as gemm_mod

        mk = self._mk()
        self._call(gemm_mod, mk, x_scale=torch.ones(4, 1), w_scale=torch.tensor(1.0))
        assert mk.call_args[0][3].shape == (1,)

    def test_2d_x_scale_and_1d_w_scale_unchanged(self):
        import quark.torch.kernel.aiter.gemm as gemm_mod

        mk = self._mk()
        self._call(gemm_mod, mk, x_scale=torch.ones(4, 1), w_scale=torch.ones(8))
        assert mk.call_args[0][2].shape == (4, 1)
        assert mk.call_args[0][3].shape == (8,)

    def test_scales_cast_to_bfloat16(self):
        import quark.torch.kernel.aiter.gemm as gemm_mod

        mk = self._mk()
        self._call(
            gemm_mod,
            mk,
            x_scale=torch.ones(4, 1, dtype=torch.float32),
            w_scale=torch.ones(8, dtype=torch.float32),
        )
        assert mk.call_args[0][2].dtype == torch.bfloat16
        assert mk.call_args[0][3].dtype == torch.bfloat16

    def test_bias_and_dtype_forwarded(self):
        import quark.torch.kernel.aiter.gemm as gemm_mod

        mk = mock.MagicMock(return_value=torch.zeros(4, 8, dtype=torch.float16))
        bias = torch.zeros(8)
        with mock.patch.object(gemm_mod, "_aiter_gemm_a8w8", mk, create=True):
            XQ = torch.zeros(4, 16, dtype=torch.float8_e4m3fn)
            WQ = torch.zeros(8, 16, dtype=torch.float8_e4m3fn)
            gemm_mod.gemm_fp8(
                XQ,
                WQ,
                torch.ones(4, 1),
                torch.ones(8),
                bias=bias,
                output_dtype=torch.float16,
            )
        assert mk.call_args.kwargs["bias"] is bias
        assert mk.call_args.kwargs["dtype"] == torch.float16


# ============================================================
# gemm.py — gemm_fp8_blockscale wrapper
# ============================================================


@requires_fp8
class TestGemmFp8BlockscaleWrapper:
    def test_args_forwarded_correctly(self):
        import quark.torch.kernel.aiter.gemm as gemm_mod

        expected = torch.zeros(4, 8, dtype=torch.bfloat16)
        mk = mock.MagicMock(return_value=expected)

        with mock.patch.object(gemm_mod, "_aiter_gemm_a8w8_blockscale", mk, create=True):
            XQ = torch.zeros(4, 128, dtype=torch.float8_e4m3fn)
            WQ = torch.zeros(8, 128, dtype=torch.float8_e4m3fn)
            result = gemm_mod.gemm_fp8_blockscale(XQ, WQ, torch.ones(4, 1), torch.ones(1, 1))

        mk.assert_called_once()
        assert mk.call_args.kwargs["dtype"] == torch.bfloat16
        assert result.shape == (4, 8)


# ============================================================
# gemm.py — gemm_fp8_bpreshuffle wrapper (scale reshaping)
# ============================================================


@requires_fp8
class TestGemmFp8BpreshuffleWrapper:
    """Test scale broadcast/reshape logic in gemm_fp8_bpreshuffle."""

    def _call(self, gemm_mod, mk, x_scale, w_scale, m=4, k=16, n=8):
        with mock.patch.object(gemm_mod, "_aiter_gemm_a8w8_bpreshuffle", mk, create=True):
            XQ = torch.zeros(m, k, dtype=torch.float8_e4m3fn)
            WQ = torch.zeros(n, k, dtype=torch.float8_e4m3fn)
            return gemm_mod.gemm_fp8_bpreshuffle(XQ, WQ, x_scale, w_scale)

    def _mk(self, m=4, n=8):
        return mock.MagicMock(return_value=torch.zeros(m, n, dtype=torch.bfloat16))

    def test_scalar_x_scale_broadcast_to_m1(self):
        import quark.torch.kernel.aiter.gemm as gemm_mod

        mk = self._mk()
        self._call(gemm_mod, mk, x_scale=torch.tensor(1.0), w_scale=torch.ones(1, 8))
        assert mk.call_args[0][2].shape == (4, 1)

    def test_1d_x_scale_reshaped_to_column(self):
        import quark.torch.kernel.aiter.gemm as gemm_mod

        mk = self._mk()
        self._call(gemm_mod, mk, x_scale=torch.ones(4), w_scale=torch.ones(1, 8))
        assert mk.call_args[0][2].shape == (4, 1)

    def test_2d_x_scale_unchanged(self):
        import quark.torch.kernel.aiter.gemm as gemm_mod

        mk = self._mk()
        self._call(gemm_mod, mk, x_scale=torch.ones(4, 1), w_scale=torch.ones(1, 8))
        assert mk.call_args[0][2].shape == (4, 1)

    def test_scalar_w_scale_broadcast_to_1n(self):
        import quark.torch.kernel.aiter.gemm as gemm_mod

        mk = self._mk()
        self._call(gemm_mod, mk, x_scale=torch.ones(4, 1), w_scale=torch.tensor(1.0))
        assert mk.call_args[0][3].shape == (1, 8)

    def test_1d_w_scale_reshaped_to_row(self):
        import quark.torch.kernel.aiter.gemm as gemm_mod

        mk = self._mk()
        self._call(gemm_mod, mk, x_scale=torch.ones(4, 1), w_scale=torch.ones(8))
        assert mk.call_args[0][3].shape == (1, 8)

    def test_2d_w_scale_unchanged(self):
        import quark.torch.kernel.aiter.gemm as gemm_mod

        mk = self._mk()
        self._call(gemm_mod, mk, x_scale=torch.ones(4, 1), w_scale=torch.ones(1, 8))
        assert mk.call_args[0][3].shape == (1, 8)

    def test_scales_cast_to_float32(self):
        import quark.torch.kernel.aiter.gemm as gemm_mod

        mk = self._mk()
        self._call(
            gemm_mod,
            mk,
            x_scale=torch.ones(4, 1, dtype=torch.bfloat16),
            w_scale=torch.ones(1, 8, dtype=torch.bfloat16),
        )
        assert mk.call_args[0][2].dtype == torch.float32
        assert mk.call_args[0][3].dtype == torch.float32

    def test_bias_and_dtype_forwarded(self):
        import quark.torch.kernel.aiter.gemm as gemm_mod

        mk = mock.MagicMock(return_value=torch.zeros(4, 8, dtype=torch.float16))
        bias = torch.zeros(8)
        with mock.patch.object(gemm_mod, "_aiter_gemm_a8w8_bpreshuffle", mk, create=True):
            XQ = torch.zeros(4, 16, dtype=torch.float8_e4m3fn)
            WQ = torch.zeros(8, 16, dtype=torch.float8_e4m3fn)
            gemm_mod.gemm_fp8_bpreshuffle(
                XQ,
                WQ,
                torch.ones(4, 1),
                torch.ones(1, 8),
                bias=bias,
                output_dtype=torch.float16,
            )
        assert mk.call_args.kwargs["bias"] is bias
        assert mk.call_args.kwargs["dtype"] == torch.float16


# ============================================================
# quant.py — module-level import paths
# ============================================================


class TestQuantModuleImport:
    """Cover the module-level try/except/else import block in quant.py."""

    def test_import_success_with_mocked_aiter(self):
        import quark.torch.kernel.aiter.quant as quant_mod

        ns = _exec_module_source(
            quant_mod,
            extra_sys_modules={
                "aiter": mock.MagicMock(),
                "aiter.ops": mock.MagicMock(),
                "aiter.ops.quant": mock.MagicMock(),
                "aiter.utility": mock.MagicMock(),
            },
            patch_is_aiter=True,
        )
        assert ns["_aiter_quant_available"] is True

    def test_import_failure_covers_except_path(self):
        import quark.torch.kernel.aiter.quant as quant_mod

        ns = _exec_module_source(quant_mod, patch_is_aiter=True)
        assert ns["_aiter_quant_available"] is False
        assert ns["_aiter_quant_import_error"] is not None

    def test_else_path_when_package_unavailable(self):
        import quark.torch.kernel.aiter.quant as quant_mod

        ns = _exec_module_source(quant_mod, patch_is_aiter=False)
        assert ns["_aiter_quant_available"] is False
        assert ns["_aiter_quant_import_error"] == "aiter package not installed"


# ============================================================
# quant.py — _check_aiter_quant_available
# ============================================================


class TestCheckAiterQuantAvailable:
    def test_raises_when_unavailable(self):
        import quark.torch.kernel.aiter.quant as quant_mod

        with pytest.raises(ImportError, match="Native inference quantization requires AMD Aiter"):
            quant_mod._check_aiter_quant_available()


# ============================================================
# quant.py — aiter-backed dynamic quant wrappers (mocked)
# ============================================================


@requires_fp8
class TestDynamicQuantWrappers:
    """Test aiter-backed quant wrappers with mocked kernels."""

    def test_per_token_quant_fp8(self):
        import quark.torch.kernel.aiter.quant as quant_mod

        expected = (
            torch.zeros(4, 16, dtype=torch.float8_e4m3fn),
            torch.ones(4, 1),
        )
        mk = mock.MagicMock(return_value=expected)
        with (
            mock.patch.object(quant_mod, "_aiter_quant_available", True),
            mock.patch.object(quant_mod, "_aiter_per_token_quant", mk, create=True),
            mock.patch.object(quant_mod, "_aiter_dtypes", mock.MagicMock(fp8=torch.float8_e4m3fn)),
        ):
            result = quant_mod.dynamic_per_token_quant_fp8(torch.randn(4, 16))

        mk.assert_called_once()
        assert mk.call_args.kwargs["quant_dtype"] == torch.float8_e4m3fn
        assert result == expected

    def test_per_group_quant_fp8(self):
        import quark.torch.kernel.aiter.quant as quant_mod

        expected = (
            torch.zeros(4, 128, dtype=torch.float8_e4m3fn),
            torch.ones(4, 1),
        )
        mk = mock.MagicMock(return_value=expected)
        with (
            mock.patch.object(quant_mod, "_aiter_quant_available", True),
            mock.patch.object(quant_mod, "_aiter_per_group_quant", mk, create=True),
            mock.patch.object(quant_mod, "_aiter_dtypes", mock.MagicMock(fp8=torch.float8_e4m3fn)),
        ):
            quant_mod.dynamic_per_group_quant_fp8(torch.randn(4, 128), group_size=64)

        mk.assert_called_once()
        assert mk.call_args.kwargs["group_size"] == 64
        assert mk.call_args.kwargs["quant_dtype"] == torch.float8_e4m3fn

    def test_per_tensor_quant_fp8(self):
        import quark.torch.kernel.aiter.quant as quant_mod

        expected = (
            torch.zeros(4, 16, dtype=torch.float8_e4m3fn),
            torch.ones(1),
        )
        mk = mock.MagicMock(return_value=expected)
        with (
            mock.patch.object(quant_mod, "_aiter_quant_available", True),
            mock.patch.object(quant_mod, "_aiter_per_tensor_quant", mk, create=True),
            mock.patch.object(quant_mod, "_aiter_dtypes", mock.MagicMock(fp8=torch.float8_e4m3fn)),
        ):
            quant_mod.dynamic_per_tensor_quant_fp8(torch.randn(4, 16))

        mk.assert_called_once()
        assert mk.call_args.kwargs["quant_dtype"] == torch.float8_e4m3fn

    def test_per_group_quant_fp4_2d_input(self):
        import quark.torch.kernel.aiter.quant as quant_mod

        expected = (torch.zeros(4, 16), torch.ones(4, 1))
        mk = mock.MagicMock(return_value=expected)
        with (
            mock.patch.object(quant_mod, "_aiter_quant_available", True),
            mock.patch.object(quant_mod, "_aiter_per_1x32_f4_quant", mk, create=True),
        ):
            y, scale = quant_mod.dynamic_per_group_quant_fp4(torch.randn(4, 32))

        mk.assert_called_once()
        assert mk.call_args.kwargs["shuffle"] is True
        assert y.shape == (4, 16)

    def test_per_group_quant_fp4_3d_reshape(self):
        import quark.torch.kernel.aiter.quant as quant_mod

        expected = (torch.zeros(8, 16), torch.ones(8, 1))
        mk = mock.MagicMock(return_value=expected)
        with (
            mock.patch.object(quant_mod, "_aiter_quant_available", True),
            mock.patch.object(quant_mod, "_aiter_per_1x32_f4_quant", mk, create=True),
        ):
            y, scale = quant_mod.dynamic_per_group_quant_fp4(torch.randn(2, 4, 32))

        passed_x = mk.call_args[0][0]
        assert passed_x.dim() == 2
        assert passed_x.shape == (8, 32)

    def test_per_group_quant_fp4_shuffle_false(self):
        import quark.torch.kernel.aiter.quant as quant_mod

        expected = (torch.zeros(4, 16), torch.ones(4, 1))
        mk = mock.MagicMock(return_value=expected)
        with (
            mock.patch.object(quant_mod, "_aiter_quant_available", True),
            mock.patch.object(quant_mod, "_aiter_per_1x32_f4_quant", mk, create=True),
        ):
            quant_mod.dynamic_per_group_quant_fp4(
                torch.randn(4, 32),
                shuffle_scale=False,
            )

        assert mk.call_args.kwargs["shuffle"] is False
