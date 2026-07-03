#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Coverage for build_custom_ops JIT helpers and base_fn_quantizers._resolve_ops."""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quark.common.torch_cpp_build_specs import ORT_WINDOWS_DEFINE, TORCH_TARGET_VERSION_DEFINE
from quark.common.torch_cpp_build_specs.onnx_ops import (
    onnx_ops_sources,
    onnx_ort_cpu_sources,
    ort_include_paths,
    ort_lib_jit_sources,
    torch_legacy_jit_sources,
)

_B = "quark.onnx.operators.custom_ops.build_custom_ops"
_F = "quark.onnx.algorithm.finetuning.create_torch.base_fn_quantizers"
_V = "quark.common.utils.torch_utils"


def _get(name: str, mod: str = _B):
    return getattr(importlib.import_module(mod), name)


class TestTorchVersionGate:
    @pytest.mark.parametrize(
        "version,expected",
        [
            ("2.10.0", True),
            ("2.10.0+cu124", True),
            ("2.11.1", True),
            ("2.9.1", False),
            ("1.13.0", False),
            ("not-a-version", False),
        ],
    )
    def test_version_gate(self, version, expected):
        with patch(f"{_V}.torch.__version__", version):
            assert _get("torch_supports_stable_abi", _V)() is expected

    def test_shared_helper_is_reused(self):
        from quark.common.utils import torch_utils
        from quark.onnx.algorithm.finetuning.create_torch import base_fn_quantizers
        from quark.onnx.operators.custom_ops import build_custom_ops

        assert build_custom_ops.torch_supports_stable_abi is torch_utils.torch_supports_stable_abi
        assert base_fn_quantizers.torch_supports_stable_abi is torch_utils.torch_supports_stable_abi


def _invoke_load_or_compile(device: str) -> None:
    """Drive ``_load_or_compile`` with the legacy ORT-lib wiring used in production."""
    _get("_load_or_compile")(
        device,
        _get("ORT_LIBRARY_NAME"),
        _get("compile_custom_op_cpu_legacy"),
        _get("compile_custom_op_gpu_legacy"),
        [],
        [],
    )


class TestLoadOrCompile:
    @patch(f"{_B}.torch")
    @patch(f"{_B}._legacy_lib_path", return_value="/fake/lib.so")
    @patch(f"{_B}.os.path.exists", return_value=True)
    def test_existing_lib_success(self, _exists, _glp, mock_torch):
        mock_torch.ops.load_library = MagicMock()
        _invoke_load_or_compile("CPU")

    @patch(f"{_B}.torch")
    @patch(f"{_B}._legacy_lib_path", return_value="/fake/lib.so")
    @patch(f"{_B}.os.path.exists", return_value=True)
    def test_existing_lib_failure(self, _exists, _glp, mock_torch):
        mock_torch.ops.load_library.side_effect = OSError("bad")
        _invoke_load_or_compile("CPU")

    @patch(f"{_B}.handle_generated_files")
    @patch(f"{_B}.compile_custom_op_cpu_legacy")
    @patch(f"{_B}._legacy_lib_path", return_value="/fake/build/lib.so")
    @patch(f"{_B}.os.path.exists", return_value=False)
    @patch(f"{_B}.os.makedirs")
    def test_cpu_jit(self, _mkdirs, _exists, _glp, mock_compile, *_):
        _invoke_load_or_compile("CPU")
        mock_compile.assert_called_once()

    @patch(f"{_B}.handle_generated_files")
    @patch(f"{_B}.compile_custom_op_gpu_legacy")
    @patch(f"{_B}._legacy_lib_path", return_value="/fake/build/lib_gpu.so")
    @patch(f"{_B}.os.path.exists", return_value=False)
    @patch(f"{_B}.os.makedirs")
    def test_gpu_jit(self, _mkdirs, _exists, _glp, mock_compile, *_):
        _invoke_load_or_compile("GPU")
        mock_compile.assert_called_once()


class TestInitializeKernels:
    @patch(f"{_B}._compile_torch_legacy_library")
    @patch(f"{_B}._compile_legacy_library")
    @patch(f"{_B}.load_or_jit_stable_abi")
    @patch(f"{_B}.torch")
    @patch(f"{_B}.torch_supports_stable_abi", return_value=True)
    def test_modern_torch_skips_legacy_ort_build(self, _gate, mock_torch, mock_stable, mock_ort, mock_legacy):
        """torch >= 2.10: the stable-ABI ``_C`` artifact serves both torch and ORT,
        so the legacy ORT lib must NOT be built separately. The JIT callable is the
        onnx-side thunk (carries the onnx vs torch source/define divergence)."""
        # MagicMock raises on dunder lookups, so set ``__version__`` explicitly.
        mock_torch.__version__ = "2.10.0"
        _get("_initialize_kernels")()
        mock_stable.assert_called_once()
        mock_ort.assert_not_called()
        mock_legacy.assert_not_called()
        assert mock_stable.call_args.kwargs["jit_compile_fn"] is _get("_jit_compile_stable_abi")

    @patch(f"{_B}._compile_torch_legacy_library")
    @patch(f"{_B}._compile_legacy_library")
    @patch(f"{_B}.load_or_jit_stable_abi")
    @patch(f"{_B}.torch")
    @patch(f"{_B}.torch_supports_stable_abi", return_value=False)
    def test_old_torch_builds_ort_only(self, _gate, mock_torch, mock_stable, mock_ort, mock_legacy):
        mock_torch.__version__ = "2.9.1"
        _get("_initialize_kernels")()
        mock_ort.assert_called_once()
        # Pre-2.10 torch ships pybind11 inside the ORT lib; no separate build.
        mock_legacy.assert_not_called()
        mock_stable.assert_not_called()

    @patch(f"{_B}._compile_legacy_library")
    @patch(f"{_B}.load_or_jit_stable_abi", side_effect=RuntimeError("no headers"))
    @patch(f"{_B}.torch")
    @patch(f"{_B}.torch_supports_stable_abi", return_value=True)
    def test_stable_abi_failure_propagates(self, _gate, mock_torch, _stable, _ort):
        with pytest.raises(RuntimeError):
            _get("_initialize_kernels")()

    @patch(f"{_B}._compile_legacy_library", side_effect=RuntimeError("no compiler"))
    @patch(f"{_B}.load_or_jit_stable_abi")
    @patch(f"{_B}.torch")
    @patch(f"{_B}.torch_supports_stable_abi", return_value=False)
    def test_ort_failure_propagates(self, _gate, mock_torch, _stable, _ort):
        mock_torch.__version__ = "2.9.1"
        with pytest.raises(RuntimeError):
            _get("_initialize_kernels")()


class TestCompileLegacyLibrary:
    # ORT lib carries the pybind11 surface (``-DTORCH_OP``) only on torch < 2.10.
    @pytest.mark.parametrize("supports_stable_abi,torch_op_present", [(True, False), (False, True)])
    @patch(f"{_B}._load_or_compile")
    @patch(f"{_B}.torch")
    @patch(f"{_B}.platform")
    def test_torch_op_flag_dispatches_on_torch_version(
        self, mock_plat, mock_torch, mock_lc, supports_stable_abi, torch_op_present
    ):
        mock_plat.system.return_value = "Linux"
        mock_torch.cuda.is_available.return_value = False
        with patch(f"{_B}.torch_supports_stable_abi", return_value=supports_stable_abi):
            _get("_compile_legacy_library")()
        cflags = mock_lc.call_args_list[0][0][5]
        assert ("-DTORCH_OP" in cflags) is torch_op_present

    @patch(f"{_B}._load_or_compile")
    @patch(f"{_B}.torch")
    @patch(f"{_B}.platform")
    @patch(f"{_B}.torch_supports_stable_abi", return_value=True)
    def test_cpu_only_on_no_cuda(self, _gate, mock_plat, mock_torch, mock_lc):
        mock_plat.system.return_value = "Linux"
        mock_torch.cuda.is_available.return_value = False
        _get("_compile_legacy_library")()
        assert mock_lc.call_count == 1
        assert mock_lc.call_args_list[0][0][0] == "CPU"

    @patch(f"{_B}._load_or_compile")
    @patch(f"{_B}.torch")
    @patch(f"{_B}.platform")
    @patch(f"{_B}.torch_supports_stable_abi", return_value=True)
    def test_cpu_and_gpu_on_cuda(self, _gate, mock_plat, mock_torch, mock_lc):
        mock_plat.system.return_value = "Linux"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (8, 0)
        _get("_compile_legacy_library")()
        assert mock_lc.call_count == 2
        assert [c[0][0] for c in mock_lc.call_args_list] == ["CPU", "GPU"]

    @patch(f"{_B}._load_or_compile")
    @patch(f"{_B}.torch")
    @patch(f"{_B}.platform")
    @patch(f"{_B}.torch_supports_stable_abi", return_value=True)
    def test_windows_adds_ort_dll_import(self, _gate, mock_plat, mock_torch, mock_lc):
        mock_plat.system.return_value = "Windows"
        mock_torch.cuda.is_available.return_value = False
        _get("_compile_legacy_library")()
        cflags = mock_lc.call_args[0][5]
        assert "-DORT_DLL_IMPORT" in cflags

    @patch(f"{_B}._load_or_compile")
    @patch(f"{_B}.torch")
    @patch(f"{_B}.platform")
    @patch(f"{_B}.torch_supports_stable_abi", return_value=True)
    def test_targets_ort_library_name(self, _gate, mock_plat, mock_torch, mock_lc):
        mock_plat.system.return_value = "Linux"
        mock_torch.cuda.is_available.return_value = False
        _get("_compile_legacy_library")()
        # _load_or_compile(device, lib, cpu_fn, gpu_fn, cuda_cflags, cflags)
        assert mock_lc.call_args[0][1] == _get("ORT_LIBRARY_NAME")


class TestCompileTorchLegacyLibrary:
    @patch(f"{_B}._load_or_compile")
    @patch(f"{_B}.torch")
    def test_targets_torch_legacy_library_name(self, mock_torch, mock_lc):
        mock_torch.cuda.is_available.return_value = False
        _get("_compile_torch_legacy_library")()
        assert mock_lc.call_args[0][1] == _get("TORCH_LEGACY_LIBRARY_NAME")

    @patch(f"{_B}._set_torch_cuda_arch_list")
    @patch(f"{_B}._load_or_compile")
    @patch(f"{_B}.torch")
    def test_compiles_gpu_when_cuda_available(self, mock_torch, mock_lc, mock_arch):
        mock_torch.cuda.is_available.return_value = True
        _get("_compile_torch_legacy_library")()
        assert mock_lc.call_count == 2
        assert [c[0][0] for c in mock_lc.call_args_list] == ["CPU", "GPU"]
        mock_arch.assert_called_once()

    @patch(f"{_B}._load_or_compile")
    @patch(f"{_B}.torch")
    def test_carries_torch_op_and_legacy_lib_flags(self, mock_torch, mock_lc):
        mock_torch.cuda.is_available.return_value = False
        _get("_compile_torch_legacy_library")()
        cflags = mock_lc.call_args[0][5]
        assert "-DTORCH_OP" in cflags
        # Selects the matching ``LIBRARY_FILE_NAME`` arm in legacy/torch_ops.cc.
        assert "-DTORCH_LEGACY_LIB" in cflags

    @patch(f"{_B}._load_or_compile")
    @patch(f"{_B}.torch")
    def test_uses_torch_legacy_compile_fn_pair(self, mock_torch, mock_lc):
        mock_torch.cuda.is_available.return_value = False
        _get("_compile_torch_legacy_library")()
        # _load_or_compile(device, lib, cpu_fn, gpu_fn, cuda_cflags, cflags)
        cpu_fn = mock_lc.call_args[0][2]
        gpu_fn = mock_lc.call_args[0][3]
        assert cpu_fn is _get("compile_custom_op_cpu_torch_legacy")
        assert gpu_fn is _get("compile_custom_op_gpu_torch_legacy")


class TestCompileStableAbi:
    # The actual compile pipeline lives in
    # ``quark.common.torch_cpp_ext.jit_compile_stable_abi_library``; these
    # tests pin the onnx-specific wiring (wheel-matching source list, ORT
    # include routing, namespace name, ORT Windows define, ROCm arch opt-out).
    @pytest.mark.parametrize("is_cuda", [False, True], ids=["cpu", "cuda"])
    @patch(f"{_B}.jit_compile_stable_abi_library", return_value=True)
    @patch(f"{_B}.torch")
    def test_passes_onnx_specific_wiring_to_shared_helper(self, mock_torch, mock_helper, is_cuda):
        mock_torch.cuda.is_available.return_value = is_cuda
        _get("_jit_compile_stable_abi")()
        kwargs = mock_helper.call_args.kwargs
        # JIT source list must match the wheel ``_C`` so the artifact is symbol-equivalent.
        assert set(kwargs["sources"]) == set(onnx_ops_sources(use_cuda=is_cuda))
        # ORT headers must reach the compiler via ``-I`` (extra_isolated_includes),
        # NOT via extra_include_paths — torch.cpp_extension's HIP path hipifies the
        # latter and generates ``_hip.h`` siblings that break ORT builds.
        assert set(kwargs["extra_isolated_includes"]) == set(ort_include_paths())
        assert set(ort_include_paths()).isdisjoint(kwargs["include_paths"])
        assert TORCH_TARGET_VERSION_DEFINE in kwargs["extra_defines"]
        assert ORT_WINDOWS_DEFINE in kwargs["extra_windows_defines"]
        # onnx stays opt-out of ROCm multi-arch narrowing — flipping this is a
        # behavior change, not a refactor.
        assert kwargs["pin_rocm_arch"] is False
        assert kwargs["label"] == "custom ops"


class TestCompileLegacy:
    # JIT pipeline behaviour lives in test_torch_cpp_ext_helpers.py; tests
    # here only pin onnx-specific wiring (sources spec, label, policy flags).
    @pytest.mark.parametrize("torch_op", [True, False], ids=["with_torch_op", "without_torch_op"])
    @pytest.mark.parametrize(
        "fn_name,use_cuda",
        [
            pytest.param("compile_custom_op_cpu_legacy", False, id="cpu"),
            pytest.param("compile_custom_op_gpu_legacy", True, id="gpu"),
        ],
    )
    def test_legacy_sources_track_torch_op_flag(self, fn_name, use_cuda, torch_op):
        # ``-DTORCH_OP`` gates the pybind11 surface into the ORT lib on torch < 2.10.
        extra_cflags = ["-DTORCH_OP"] if torch_op else []
        with patch(f"{_B}.jit_compile_nonabi_library") as mock_helper:
            _get(fn_name)("libcustom_ops", "/fake/build", [], extra_cflags)
        kwargs = mock_helper.call_args.kwargs
        assert set(kwargs["sources"]) == set(ort_lib_jit_sources(use_cuda=use_cuda, include_legacy_torch_ops=torch_op))
        assert kwargs["use_cuda"] is use_cuda
        assert kwargs["label"] == "custom ops"
        # CPU wrappers opt into ImportError-as-success because ``-DNO_GPU``
        # pybind11 builds raise it when the ``.so`` has no python init.
        assert kwargs.get("import_error_is_success", False) is (not use_cuda)

    @pytest.mark.parametrize("fn_name", ["compile_custom_op_cpu_legacy", "compile_custom_op_gpu_legacy"])
    def test_legacy_compile_keeps_ort_headers_out_of_extra_include_paths(self, fn_name):
        # Regression for the ROCm "redefinition of struct Ort::Exception" build
        # failure: torch.cpp_extension's HIP path hipifies every header found
        # in ``extra_include_paths``, so ORT headers must reach the compiler
        # via ``-I`` cflags only (see ``_ort_lib_cflags``).
        with patch(f"{_B}.jit_compile_nonabi_library") as mock_helper:
            _get(fn_name)("libcustom_ops", "/fake/build", [], [])
        assert set(ort_include_paths()).isdisjoint(mock_helper.call_args.kwargs["include_paths"])

    def test_ort_lib_cflags_routes_ort_headers_through_dash_I(self):
        # Pair of the regression test above: every ORT root must reach the
        # compiler via ``-I`` cflags (mirrored on the nvcc/hipcc command line)
        # so hipify never sees the ORT include dirs.
        ort_cuda_cflags, ort_cflags = _get("_ort_lib_cflags")()
        expected = {f"-I{p}" for p in ort_include_paths()}
        assert expected <= set(ort_cflags)
        assert expected <= set(ort_cuda_cflags)


class TestCompileTorchLegacy:
    @pytest.mark.parametrize(
        "fn_name,use_cuda",
        [
            pytest.param("compile_custom_op_cpu_torch_legacy", False, id="cpu"),
            pytest.param("compile_custom_op_gpu_torch_legacy", True, id="gpu"),
        ],
    )
    def test_torch_legacy_sources_match_jit_spec(self, fn_name, use_cuda):
        # pybind11 + shared kernels only; no ORT glue in the test-only artifact.
        with patch(f"{_B}.jit_compile_nonabi_library") as mock_helper:
            _get(fn_name)("libcustom_ops_torch_legacy", "/fake/build", [], [])
        kwargs = mock_helper.call_args.kwargs
        assert set(kwargs["sources"]) == set(torch_legacy_jit_sources(use_cuda=use_cuda))
        assert kwargs["use_cuda"] is use_cuda
        assert kwargs["label"] == "torch-legacy custom ops"
        # Same CPU-only opt-in as the ORT legacy wrappers above.
        assert kwargs.get("import_error_is_success", False) is (not use_cuda)


class TestGetLibraryPath:
    # On torch < 2.10, get_library_path dispatches on ``device`` and ``lib``
    # to pick among the CPU/GPU and ORT/torch-legacy JIT artifacts.
    @pytest.mark.parametrize(
        "fn_name,device,expected_suffix",
        [
            ("get_library_path", "CPU", "libcustom_ops.so"),
            ("get_library_path", "GPU", "libcustom_ops_gpu.so"),
            ("get_legacy_torch_library_path", "CPU", "libcustom_ops_torch_legacy.so"),
            ("get_legacy_torch_library_path", "GPU", "libcustom_ops_torch_legacy_gpu.so"),
        ],
    )
    @patch(f"{_B}.torch_supports_stable_abi", return_value=False)
    @patch(f"{_B}._get_build_directory", return_value="/fake/cache")
    @patch(f"{_B}.platform")
    @patch(f"{_B}.os.path.exists", return_value=True)
    def test_legacy_artifact_name_dispatch(self, _exists, mock_plat, _gbd, _gate, fn_name, device, expected_suffix):
        mock_plat.system.return_value = "Linux"
        assert _get(fn_name)(device).endswith(expected_suffix)

    @patch(f"{_B}.find_setuptools_extension", return_value=Path("/fake/site/_C.cpython.so"))
    @patch(f"{_B}.torch_supports_stable_abi", return_value=True)
    def test_modern_torch_prefers_wheel_setuptools_ext(self, _gate, mock_find):
        assert _get("get_library_path")("CPU") == "/fake/site/_C.cpython.so"
        assert _get("get_library_path")("GPU") == "/fake/site/_C.cpython.so"
        mock_find.assert_called()

    @patch(f"{_B}.glob.iglob", return_value=iter(["/fake/cache/quark_custom_ops_stable_abi.so"]))
    @patch(f"{_B}._get_build_directory", return_value="/fake/cache")
    @patch(f"{_B}.find_setuptools_extension", return_value=None)
    @patch(f"{_B}.torch_supports_stable_abi", return_value=True)
    def test_modern_torch_falls_back_to_jit_artifact(self, _gate, _find, _gbd, _ig):
        assert _get("get_library_path")("GPU") == "/fake/cache/quark_custom_ops_stable_abi.so"

    # Regression: when packaging breaks (no precompiled ``_C`` AND no JIT-cache
    # artifact), ``_get_stable_abi_custom_ops_path`` must raise so ORT
    # registration fails loudly at the source instead of getting a ``""``
    # path and decoding ORT's downstream error. The other two branches are
    # naturally exercised by build runs; this no-artifact branch only fires
    # when packaging breaks (or init was bypassed). Mirrors torch-side
    # ``load_or_jit_stable_abi`` loud-failure contract.
    @patch(f"{_B}.glob.iglob", return_value=iter([]))
    @patch(f"{_B}._get_build_directory", return_value="/fake/cache")
    @patch(f"{_B}.find_setuptools_extension", return_value=None)
    @patch(f"{_B}.torch_supports_stable_abi", return_value=True)
    def test_modern_torch_raises_when_no_artifact(self, _gate, _find, _gbd, _ig):
        with pytest.raises(RuntimeError, match="not found"):
            _get("_get_stable_abi_custom_ops_path")()

    # Falls back to the legacy lookup when an explicit ``lib`` (e.g. the
    # test-only torch-legacy artifact) is requested even on torch >= 2.10.
    @patch(f"{_B}.torch_supports_stable_abi", return_value=True)
    @patch(f"{_B}._get_build_directory", return_value="/fake/cache")
    @patch(f"{_B}.platform")
    @patch(f"{_B}.os.path.exists", return_value=True)
    def test_modern_torch_legacy_lib_arg_uses_per_device_path(self, _exists, mock_plat, _gbd, _gate):
        mock_plat.system.return_value = "Linux"
        TORCH_LEGACY = _get("TORCH_LEGACY_LIBRARY_NAME")
        assert _get("get_library_path")("CPU", lib=TORCH_LEGACY).endswith("libcustom_ops_torch_legacy.so")


class TestCpuOrtCompanion:
    @patch(f"{_B}.find_setuptools_extension", return_value=Path("/fake/site/_C_cpu.cpython.so"))
    @patch(f"{_B}.torch")
    @patch(f"{_B}.torch_supports_stable_abi", return_value=True)
    def test_gpu_build_cpu_device_prefers_precompiled_companion(self, _gate, mock_torch, mock_find):
        mock_torch.cuda.is_available.return_value = True
        assert _get("get_library_path")("CPU") == "/fake/site/_C_cpu.cpython.so"
        assert mock_find.call_args.kwargs["stem"] == "_C_cpu"

    @patch(f"{_B}.handle_generated_files")
    @patch(f"{_B}.jit_compile_nonabi_library")
    @patch(f"{_B}.os.makedirs")
    @patch(f"{_B}.os.path.exists", return_value=False)
    @patch(f"{_B}._legacy_lib_path", return_value="/fake/cache/libcustom_ops.so")
    @patch(f"{_B}.find_setuptools_extension", return_value=None)
    @patch(f"{_B}.raise_if_jit_fallback_disabled")
    @patch(f"{_B}.torch")
    @patch(f"{_B}.torch_supports_stable_abi", return_value=True)
    def test_jit_fallback_does_not_recurse_and_matches_aot_sources(
        self, _gate, mock_torch, _raise, _find, _llp, _exists, _mkdirs, mock_jit, *_
    ):
        mock_torch.cuda.is_available.return_value = True
        assert _get("get_library_path")("CPU") == "/fake/cache/libcustom_ops.so"
        assert set(mock_jit.call_args.kwargs["sources"]) == set(onnx_ort_cpu_sources())
        assert mock_jit.call_args.kwargs["use_cuda"] is False

    @patch(f"{_B}.find_setuptools_extension", return_value=None)
    @patch(f"{_B}.raise_if_jit_fallback_disabled", side_effect=RuntimeError("jit disabled"))
    @patch(f"{_B}.torch")
    @patch(f"{_B}.torch_supports_stable_abi", return_value=True)
    def test_jit_fallback_disabled_raises(self, _gate, mock_torch, _raise, _find):
        mock_torch.cuda.is_available.return_value = True
        with pytest.raises(RuntimeError, match="jit disabled"):
            _get("get_library_path")("CPU")


class TestResolveOps:
    # Modern torch: pybind11 in the slim torch-legacy artifact. Old torch: in the ORT lib.
    @pytest.mark.parametrize(
        "supports_stable_abi,plat,gpu,expected_name",
        [
            (True, "Linux", False, "libcustom_ops_torch_legacy"),
            (True, "Linux", True, "libcustom_ops_torch_legacy_gpu"),
            (True, "Windows", False, "custom_ops_torch_legacy"),
            (True, "Windows", True, "custom_ops_torch_legacy_gpu"),
            (False, "Linux", False, "libcustom_ops"),
            (False, "Linux", True, "libcustom_ops_gpu"),
            (False, "Windows", False, "custom_ops"),
            (False, "Windows", True, "custom_ops_gpu"),
        ],
    )
    @patch(f"{_F}.platform")
    @patch(f"{_F}.get_legacy_torch_library_path", return_value="/fake/lib_torch_legacy.so")
    @patch(f"{_F}.get_library_path", return_value="/fake/lib.so")
    @patch(f"{_F}.importlib")
    def test_load_legacy_ops_module_name(
        self, mock_imp, _glp, _glltp, mock_plat, supports_stable_abi, plat, gpu, expected_name
    ):
        mock_imp.import_module.return_value = MagicMock()
        mock_plat.system.return_value = plat
        with patch(f"{_F}.torch_supports_stable_abi", return_value=supports_stable_abi):
            _get("_load_legacy_ops", _F)(gpu=gpu)
        mock_imp.import_module.assert_called_with(expected_name)

    @patch(f"{_F}.platform")
    @patch(f"{_F}.get_legacy_torch_library_path", return_value="/fake/lib_torch_legacy.so")
    @patch(f"{_F}.importlib")
    @patch(f"{_F}.torch_supports_stable_abi", return_value=True)
    def test_load_legacy_ops_returns_none_on_import_error(self, _gate, mock_imp, _glltp, mock_plat):
        mock_plat.system.return_value = "Linux"
        mock_imp.import_module.side_effect = ImportError("nope")
        assert _get("_load_legacy_ops", _F)(gpu=False) is None

    @patch(f"{_F}.platform")
    @patch(f"{_F}.get_legacy_torch_library_path", return_value="/fake/dir/lib.so")
    @patch(f"{_F}.importlib")
    @patch(f"{_F}.torch_supports_stable_abi", return_value=True)
    def test_load_legacy_ops_appends_library_dir_to_sys_path(self, _gate, mock_imp, _glltp, mock_plat):
        # ``sys.path`` must contain the library dir exactly once across calls.
        fn = _get("_load_legacy_ops", _F)
        mock_plat.system.return_value = "Windows"
        mock_imp.import_module.return_value = MagicMock()
        original = list(sys.path)
        try:
            fn(gpu=False)
            fn(gpu=True)
            assert sys.path.count("/fake/dir") == 1
        finally:
            sys.path[:] = original

    @patch(f"{_F}.torch_supports_stable_abi", return_value=True)
    @patch(f"{_F}.torch")
    def test_resolve_stable_abi(self, mock_torch, _gate):
        mock_ops = MagicMock()
        mock_torch.ops.quark_custom_ops = mock_ops
        cpu, gpu = _get("_resolve_ops", _F)()
        assert cpu is mock_ops
        assert gpu is mock_ops

    @patch(f"{_F}.torch_supports_stable_abi", return_value=True)
    @patch(f"{_F}._load_legacy_ops")
    @patch(f"{_F}.torch")
    def test_resolve_modern_torch_falls_back_to_fake_not_legacy(self, mock_torch, mock_legacy, _gate):
        # Stable-ABI fallback must be Fake — never silently the legacy surface.
        mock_torch.ops.quark_custom_ops = MagicMock(spec=[])
        FakeOps = _get("FakeCustomTorchOps", _F)
        cpu, gpu = _get("_resolve_ops", _F)()
        assert cpu is FakeOps
        assert gpu is FakeOps
        mock_legacy.assert_not_called()

    @patch(f"{_F}.torch_supports_stable_abi", return_value=False)
    @patch(f"{_F}._load_legacy_ops")
    @patch(f"{_F}.torch")
    def test_resolve_old_torch_prefers_legacy(self, mock_torch, mock_legacy, _gate):
        cpu_mod, gpu_mod = MagicMock(), MagicMock()
        mock_legacy.side_effect = [cpu_mod, gpu_mod]
        cpu, gpu = _get("_resolve_ops", _F)()
        assert cpu is cpu_mod
        assert gpu is gpu_mod
        assert not mock_torch.ops.quark_custom_ops.mx.called

    @patch(f"{_F}.torch_supports_stable_abi", return_value=False)
    @patch(f"{_F}._load_legacy_ops", return_value=None)
    @patch(f"{_F}.torch")
    def test_resolve_old_torch_falls_back_to_fake(self, _torch, _legacy, _gate):
        FakeOps = _get("FakeCustomTorchOps", _F)
        cpu, gpu = _get("_resolve_ops", _F)()
        assert cpu is FakeOps
        assert gpu is FakeOps
