#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for the shared torch C++ extension helpers in ``quark.common.torch_cpp_ext``."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from quark.common import torch_cpp_ext
from quark.common.torch_cpp_build_specs import ORT_WINDOWS_DEFINE, TORCH_TARGET_VERSION_DEFINE

C = "quark.common.torch_cpp_ext"
FAKE_BASE = Path("/fake/module")
FAKE_PKG_SUBPATH = Path("quark/fake/pkg")


@pytest.mark.parametrize(
    "system,suffix,prefix",
    [("Linux", ".so", "lib"), ("Darwin", ".dylib", "lib"), ("Windows", ".dll", "")],
)
def test_get_platform_lib_suffix_and_prefix(system, suffix, prefix):
    # ``platform.system`` is the only non-Linux branch reachable on CI, so
    # both helpers are tested together to keep all OS arms in one place.
    with patch(f"{C}.platform.system", return_value=system):
        assert torch_cpp_ext.get_platform_lib_suffix() == suffix
        assert torch_cpp_ext.get_platform_lib_prefix() == prefix


# exists_seq drives Path.exists() (next-to-module, then site-packages).
# expected: None | ("base",) | ("sp", "/fake/sp").
@pytest.mark.parametrize(
    "ext_suffix,exists_seq,sitepackages,expected",
    [
        pytest.param(None, [], None, None, id="no_ext_suffix"),
        pytest.param(".so", [True], None, ("base",), id="next_to_module"),
        pytest.param(".so", [False, True], ["/fake/sp"], ("sp", "/fake/sp"), id="site_packages_fallback"),
        pytest.param(".so", [False, False], ["/fake/sp"], None, id="none_found"),
        pytest.param(".so", [False], RuntimeError("no site"), None, id="site_packages_raises"),
    ],
)
def test_find_setuptools_extension(ext_suffix, exists_seq, sitepackages, expected):
    sp_patch = (
        patch(f"{C}.site.getsitepackages", side_effect=sitepackages)
        if isinstance(sitepackages, Exception)
        else patch(f"{C}.site.getsitepackages", return_value=sitepackages or [])
    )
    exists_kwargs = {"side_effect": exists_seq} if exists_seq else {"return_value": False}
    with (
        patch(f"{C}.sysconfig.get_config_var", return_value=ext_suffix),
        patch.object(Path, "exists", **exists_kwargs),
        sp_patch,
    ):
        result = torch_cpp_ext.find_setuptools_extension(FAKE_BASE, FAKE_PKG_SUBPATH)

    if expected is None:
        assert result is None
    elif expected[0] == "base":
        assert result is not None
        assert str(result).startswith(str(FAKE_BASE))
    else:
        assert result is not None
        assert expected[1] in str(result)


@pytest.mark.parametrize(
    "exists,kwargs,expected_name",
    [
        pytest.param(True, {}, "libfoo.so", id="cpu_default"),
        pytest.param(True, {"use_gpu": True}, "libfoo_gpu.so", id="gpu_default_suffix"),
        pytest.param(True, {"use_gpu": True, "gpu_suffix": "_cuda"}, "libfoo_cuda.so", id="gpu_custom_suffix"),
        pytest.param(False, {}, None, id="missing_returns_none"),
    ],
)
def test_get_precompiled_lib_path(exists, kwargs, expected_name):
    with (
        patch(f"{C}.platform.system", return_value="Linux"),
        patch.object(Path, "exists", return_value=exists),
    ):
        result = torch_cpp_ext.get_precompiled_lib_path(FAKE_BASE, "foo", **kwargs)

    if expected_name is None:
        assert result is None
    else:
        assert result is not None
        assert result.name == expected_name


@pytest.mark.parametrize(
    "lib_path,load_side_effect,expected,expect_load_call",
    [
        pytest.param(None, None, False, False, id="none_path_short_circuits"),
        pytest.param(Path("/fake/lib.so"), None, True, True, id="success"),
        pytest.param(Path("/fake/lib.so"), OSError("bad"), False, True, id="load_raises"),
    ],
)
def test_load_precompiled_library(lib_path, load_side_effect, expected, expect_load_call):
    with patch(f"{C}.torch.ops.load_library", side_effect=load_side_effect) as mock_load:
        assert torch_cpp_ext.load_precompiled_library(lib_path) is expected
    if expect_load_call:
        mock_load.assert_called_once_with(str(lib_path))
    else:
        mock_load.assert_not_called()


@pytest.fixture
def stable_abi_patches():
    """Patch the four collaborators of ``load_stable_abi_library`` in one place.

    Each test customizes the four return values / side effects; the fixture
    only deduplicates the patch scaffold.
    """
    with (
        patch(f"{C}.find_setuptools_extension") as setup,
        patch(f"{C}.get_precompiled_lib_path") as precompiled,
        patch(f"{C}.torch.ops.load_library") as load,
        patch(f"{C}.torch.cuda.is_available") as cuda,
    ):
        yield SimpleNamespace(setup=setup, precompiled=precompiled, load=load, cuda=cuda)


def test_load_stable_abi_library_uses_setuptools_first(stable_abi_patches):
    fake_c_lib = FAKE_BASE / "_C.so"
    stable_abi_patches.setup.return_value = fake_c_lib
    assert torch_cpp_ext.load_stable_abi_library(FAKE_BASE, FAKE_PKG_SUBPATH, "foo") is True
    stable_abi_patches.load.assert_called_once_with(str(fake_c_lib))
    stable_abi_patches.precompiled.assert_not_called()


def test_load_stable_abi_library_falls_back_to_precompiled_on_setuptools_failure(stable_abi_patches):
    fake_c_lib = FAKE_BASE / "_C.so"
    fake_gpu_lib = FAKE_BASE / "lib" / "libfoo_gpu.so"
    stable_abi_patches.setup.return_value = fake_c_lib
    stable_abi_patches.load.side_effect = [OSError("boom"), None]
    stable_abi_patches.precompiled.side_effect = lambda _base, _name, *, use_gpu=False, **_kw: (
        fake_gpu_lib if use_gpu else None
    )
    stable_abi_patches.cuda.return_value = True
    assert torch_cpp_ext.load_stable_abi_library(FAKE_BASE, FAKE_PKG_SUBPATH, "foo") is True
    assert stable_abi_patches.load.call_count == 2


def test_load_stable_abi_library_prefers_gpu_flavour(stable_abi_patches):
    fake_gpu_lib = FAKE_BASE / "lib" / "libfoo_gpu.so"
    fake_cpu_lib = FAKE_BASE / "lib" / "libfoo.so"
    seen: dict[bool, int] = {}

    def fake_precompiled(_base, _name, *, use_gpu=False, **_kw):
        seen[use_gpu] = seen.get(use_gpu, 0) + 1
        return fake_gpu_lib if use_gpu else fake_cpu_lib

    stable_abi_patches.setup.return_value = None
    stable_abi_patches.precompiled.side_effect = fake_precompiled
    stable_abi_patches.cuda.return_value = True

    assert torch_cpp_ext.load_stable_abi_library(FAKE_BASE, FAKE_PKG_SUBPATH, "foo") is True
    stable_abi_patches.load.assert_called_once_with(str(fake_gpu_lib))
    # CPU flavour must not be consulted after GPU success.
    assert seen == {True: 1}


# All three "no library found" branches collapse to the same observable
# behavior: setup miss → precompiled miss → ``False``. The cuda-available
# flag and the ``use_gpu=False`` override only narrow which flavours
# ``get_precompiled_lib_path`` is queried for, so we assert that gating
# in the same test instead of one-test-per-branch.
@pytest.mark.parametrize(
    "cuda_available,use_gpu_override,expected_gpu_flavours_queried",
    [
        pytest.param(False, None, {False}, id="no_cuda_cpu_only"),
        pytest.param(True, False, {False}, id="cuda_but_override_disables_gpu"),
        pytest.param(True, None, {True, False}, id="cuda_tries_gpu_then_cpu"),
    ],
)
def test_load_stable_abi_library_returns_false_when_all_sources_missing(
    stable_abi_patches, cuda_available, use_gpu_override, expected_gpu_flavours_queried
):
    seen: set[bool] = set()

    def fake_precompiled(_base, _name, *, use_gpu=False, **_kw):
        seen.add(use_gpu)
        return None

    stable_abi_patches.setup.return_value = None
    stable_abi_patches.precompiled.side_effect = fake_precompiled
    stable_abi_patches.cuda.return_value = cuda_available

    kwargs = {} if use_gpu_override is None else {"use_gpu": use_gpu_override}
    assert torch_cpp_ext.load_stable_abi_library(FAKE_BASE, FAKE_PKG_SUBPATH, "foo", **kwargs) is False
    assert seen == expected_gpu_flavours_queried


_FAKE_LEGACY_KWARGS = dict(
    name="quark_test_legacy",
    build_directory="/fake/build",
    sources=["/fake/a.cc", "/fake/a.cu"],
    include_paths=["/fake/include"],
    label="test",
)


# --- jit_fallback_disabled / raise_if_jit_fallback_disabled / load_or_jit_stable_abi ---


@pytest.mark.parametrize(
    "env_value,expected",
    [
        pytest.param(None, False, id="unset"),
        pytest.param("0", False, id="zero_is_off"),
        pytest.param("", False, id="empty_is_off"),
        pytest.param("1", True, id="one_is_on"),
        pytest.param("true", True, id="non_zero_string_is_on"),
    ],
)
def test_jit_fallback_disabled(env_value, expected):
    env = {} if env_value is None else {"QUARK_BUILD_DISABLE_JIT_FALLBACK": env_value}
    with patch.dict(os.environ, env, clear=False):
        if env_value is None:
            os.environ.pop("QUARK_BUILD_DISABLE_JIT_FALLBACK", None)
        assert torch_cpp_ext.jit_fallback_disabled() is expected


def test_raise_if_jit_fallback_disabled_raises_when_set():
    with (
        patch.dict(os.environ, {"QUARK_BUILD_DISABLE_JIT_FALLBACK": "1"}, clear=False),
        pytest.raises(RuntimeError, match="QUARK_BUILD_DISABLE_JIT_FALLBACK"),
    ):
        torch_cpp_ext.raise_if_jit_fallback_disabled("my_label")


def test_raise_if_jit_fallback_disabled_noop_when_unset():
    with patch.dict(os.environ, {"QUARK_BUILD_DISABLE_JIT_FALLBACK": "0"}, clear=False):
        torch_cpp_ext.raise_if_jit_fallback_disabled("my_label")


@pytest.fixture
def load_or_jit_patches():
    """Mock the filesystem loader and the JIT callable so each test can control
    return values / call counts independently of the real compile pipeline."""
    from unittest.mock import MagicMock

    with patch(f"{C}.load_stable_abi_library") as loader:
        yield SimpleNamespace(loader=loader, jit_fn=MagicMock(name="jit_compile_fn"))


_LOAD_OR_JIT_KWARGS = dict(
    base_dir=FAKE_BASE,
    package_subpath=FAKE_PKG_SUBPATH,
    library_name="foo",
    display_name="Foo",
)


def test_load_or_jit_stable_abi_load_hit_short_circuits(load_or_jit_patches):
    """Precompiled load hit returns ``None`` and never invokes the JIT thunk."""
    load_or_jit_patches.loader.return_value = True
    assert (
        torch_cpp_ext.load_or_jit_stable_abi(
            jit_compile_fn=load_or_jit_patches.jit_fn,
            **_LOAD_OR_JIT_KWARGS,
        )
        is None
    )
    load_or_jit_patches.jit_fn.assert_not_called()


def test_load_or_jit_stable_abi_gate_promotes_miss_to_error(load_or_jit_patches):
    load_or_jit_patches.loader.return_value = False
    with (
        patch.dict(os.environ, {"QUARK_BUILD_DISABLE_JIT_FALLBACK": "1"}, clear=False),
        pytest.raises(RuntimeError, match="QUARK_BUILD_DISABLE_JIT_FALLBACK"),
    ):
        torch_cpp_ext.load_or_jit_stable_abi(
            jit_compile_fn=load_or_jit_patches.jit_fn,
            **_LOAD_OR_JIT_KWARGS,
        )
    load_or_jit_patches.jit_fn.assert_not_called()


def test_load_or_jit_stable_abi_jit_success_returns_none(load_or_jit_patches):
    """Precompiled miss + JIT compile success: no return value, no raise."""
    load_or_jit_patches.loader.return_value = False
    load_or_jit_patches.jit_fn.return_value = True
    with patch.dict(os.environ, {"QUARK_BUILD_DISABLE_JIT_FALLBACK": "0"}, clear=False):
        assert (
            torch_cpp_ext.load_or_jit_stable_abi(
                jit_compile_fn=load_or_jit_patches.jit_fn,
                **_LOAD_OR_JIT_KWARGS,
            )
            is None
        )
    load_or_jit_patches.jit_fn.assert_called_once()


def test_load_or_jit_stable_abi_both_fail_raises(load_or_jit_patches):
    """Precompiled miss + JIT compile failure must raise rather than returning a sentinel —
    silent continuation on stable-ABI-capable PyTorch would mask the regression behind
    whatever fallback code path the caller chose next."""
    load_or_jit_patches.loader.return_value = False
    load_or_jit_patches.jit_fn.return_value = False
    with (
        patch.dict(os.environ, {"QUARK_BUILD_DISABLE_JIT_FALLBACK": "0"}, clear=False),
        pytest.raises(RuntimeError, match=r"Stable-ABI Foo extension could not be loaded"),
    ):
        torch_cpp_ext.load_or_jit_stable_abi(
            jit_compile_fn=load_or_jit_patches.jit_fn,
            **_LOAD_OR_JIT_KWARGS,
        )
    load_or_jit_patches.jit_fn.assert_called_once()


# --- jit_compile_stable_abi_library ---


_FAKE_JIT_KWARGS = dict(
    name="quark_test_stable_abi",
    sources=["/fake/a.cc", "/fake/b.cu"],
    include_paths=["/fake/include"],
    label="test",
    extra_defines=[TORCH_TARGET_VERSION_DEFINE],
)


@pytest.fixture
def jit_legacy_patches():
    """Patch lazy-imported ``cpp_extension.load`` + ``torch.version.hip`` so the HIP branch is reachable on CPU CI."""
    with (
        patch("torch.utils.cpp_extension.load") as load,
        patch(f"{C}.torch") as mock_torch,
    ):
        mock_torch.version.hip = None
        yield SimpleNamespace(load=load, torch=mock_torch)


def _snapshot_load_kwargs(mock_load) -> dict[str, list[str]]:
    """Capture ``extra_*`` lists at the moment ``load`` is called; the
    helper's ``finally`` block restores them, so post-call assertions on
    the caller's list would always see them empty."""
    snapshot: dict[str, list[str]] = {}
    mock_load.side_effect = lambda **kwargs: snapshot.update(
        {k: list(v) for k, v in kwargs.items() if k.startswith("extra_")}
    )
    return snapshot


@pytest.mark.parametrize(
    "use_cuda,hip,want_cflag_defines,want_cuda_cflag_defines",
    [
        pytest.param(False, None, {"-DNO_GPU"}, set(), id="cpu_appends_no_gpu"),
        pytest.param(True, None, {"-DUSE_CUDA"}, {"-DUSE_CUDA"}, id="gpu_non_hip_appends_use_cuda_to_both"),
        # HIP: cpp_extension auto-defines ``USE_ROCM`` and legacy kernels gate
        # ``USE_CUDA`` separately, so appending it would mis-route the build.
        pytest.param(True, "5.7", set(), set(), id="gpu_hip_appends_nothing"),
    ],
)
def test_jit_legacy_flag_append_and_restore(
    jit_legacy_patches, use_cuda, hip, want_cflag_defines, want_cuda_cflag_defines
):
    jit_legacy_patches.torch.version.hip = hip
    extra_cflags: list[str] = []
    extra_cuda_cflags: list[str] = []
    snapshot = _snapshot_load_kwargs(jit_legacy_patches.load)
    torch_cpp_ext.jit_compile_nonabi_library(
        **_FAKE_LEGACY_KWARGS,
        use_cuda=use_cuda,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
    )
    all_defines = {"-DUSE_CUDA", "-DNO_GPU"}
    assert set(snapshot["extra_cflags"]) & all_defines == want_cflag_defines
    assert set(snapshot["extra_cuda_cflags"]) & all_defines == want_cuda_cflag_defines
    # Caller-owned lists must be untouched on return so subsequent compile
    # passes can't inherit stale defines.
    assert extra_cflags == []
    assert extra_cuda_cflags == []


@pytest.mark.parametrize(
    "import_error_is_success,exception",
    [
        # Two distinct except-branches: ImportError + opt-in routes to the
        # info-log success path; anything else routes to the warning path.
        pytest.param(True, ImportError("no init"), id="import_error_treated_as_success"),
        pytest.param(False, OSError("no compiler"), id="non_import_error_swallowed_as_warning"),
    ],
)
def test_jit_legacy_failures_never_propagate(jit_legacy_patches, import_error_is_success, exception):
    # Propagating would brick ``import quark.onnx`` on a host that can't build.
    jit_legacy_patches.load.side_effect = exception
    torch_cpp_ext.jit_compile_nonabi_library(
        **_FAKE_LEGACY_KWARGS,
        use_cuda=False,
        extra_cflags=[],
        extra_cuda_cflags=[],
        import_error_is_success=import_error_is_success,
    )


@pytest.mark.parametrize(
    "use_cuda,hip,import_error_is_success,exception,expect_hint_substrs",
    [
        # GPU + CUDA host: name CUDA-side env vars only.
        pytest.param(
            True, None, False, OSError("no compiler"), ("CUDA_HOME", "TORCH_CUDA_ARCH_LIST", "nvcc"), id="gpu_cuda"
        ),
        # GPU + HIP host: name ROCm-side env vars only — pointing HIP users
        # at CUDA_HOME would be misleading.
        pytest.param(True, "5.7", False, OSError("no compiler"), ("ROCM_PATH", "PYTORCH_ROCM_ARCH"), id="gpu_hip"),
        # CPU failure: no GPU hint — the underlying exception is already
        # actionable and there's no env-var checklist for the host toolchain.
        pytest.param(False, None, False, OSError("no compiler"), (), id="cpu_no_hint"),
        # ImportError-as-success on CPU: routed to the info-log branch, so the
        # warning + hint must not fire at all.
        pytest.param(False, None, True, ImportError("no init"), (), id="cpu_import_error_success_no_hint"),
    ],
)
def test_jit_legacy_gpu_env_hint(
    jit_legacy_patches, use_cuda, hip, import_error_is_success, exception, expect_hint_substrs
):
    jit_legacy_patches.torch.version.hip = hip
    jit_legacy_patches.load.side_effect = exception
    with patch(f"{C}.logger.warning") as mock_warn:
        torch_cpp_ext.jit_compile_nonabi_library(
            **_FAKE_LEGACY_KWARGS,
            use_cuda=use_cuda,
            extra_cflags=[],
            extra_cuda_cflags=[],
            import_error_is_success=import_error_is_success,
        )
    hint_calls = [c for c in mock_warn.call_args_list if "environment-related" in str(c)]
    if not expect_hint_substrs:
        assert hint_calls == []
        return
    assert len(hint_calls) == 1, f"expected exactly one env-hint warning, got: {mock_warn.call_args_list}"
    hint_text = str(hint_calls[0])
    for substr in expect_hint_substrs:
        assert substr in hint_text, f"missing {substr!r} in hint: {hint_text}"
    # Cross-backend leakage check: HIP hosts must not be told to check
    # CUDA_HOME and CUDA hosts must not be told to check ROCM_PATH.
    forbidden = "ROCM_PATH" if hip is None else "CUDA_HOME"
    assert forbidden not in hint_text


def test_jit_legacy_forwards_load_kwargs(jit_legacy_patches):
    # Pins the contract the onnx wrappers depend on: sources / include_paths /
    # build_directory / name reach ``cpp_extension.load`` unchanged. Keeps the
    # "ORT headers stay out of extra_include_paths" regression transitive.
    torch_cpp_ext.jit_compile_nonabi_library(
        **_FAKE_LEGACY_KWARGS,
        use_cuda=False,
        extra_cflags=[],
        extra_cuda_cflags=[],
    )
    kwargs = jit_legacy_patches.load.call_args.kwargs
    assert kwargs["name"] == _FAKE_LEGACY_KWARGS["name"]
    assert kwargs["sources"] == _FAKE_LEGACY_KWARGS["sources"]
    assert kwargs["build_directory"] == _FAKE_LEGACY_KWARGS["build_directory"]
    assert kwargs["extra_include_paths"] == _FAKE_LEGACY_KWARGS["include_paths"]
    assert kwargs["verbose"] is False


@pytest.fixture
def jit_stable_abi_patches():
    """Patch the three collaborators of ``jit_compile_stable_abi_library``.

    Only the inner ``torch.utils.cpp_extension.load`` and the ROCm-arch
    context manager need to be controlled; ``compose_compile_flags`` is
    exercised on real inputs so the flag wiring stays end-to-end tested.
    """
    with (
        patch("torch.utils.cpp_extension.load") as load,
        patch(f"{C}.set_rocm_user_architecture") as rocm_ctx,
        patch(f"{C}.torch.cuda.is_available", return_value=False),
        patch(f"{C}.platform.system", return_value="Linux"),
    ):
        rocm_ctx.return_value.__enter__ = lambda self: None
        rocm_ctx.return_value.__exit__ = lambda self, *a: None
        yield SimpleNamespace(load=load, rocm_ctx=rocm_ctx)


def test_jit_compile_stable_abi_success_forwards_load_kwargs(jit_stable_abi_patches):
    """Success path must forward sources/include_paths verbatim and pin
    ``is_python_module=False`` (the stable-ABI surface registers ops via
    ``TORCH_LIBRARY`` instead of pybind11, so a Python module is meaningless
    and would shadow the .so on the namespace)."""
    assert torch_cpp_ext.jit_compile_stable_abi_library(**_FAKE_JIT_KWARGS) is True
    kwargs = jit_stable_abi_patches.load.call_args.kwargs
    assert kwargs["name"] == _FAKE_JIT_KWARGS["name"]
    assert kwargs["sources"] == _FAKE_JIT_KWARGS["sources"]
    assert kwargs["extra_include_paths"] == _FAKE_JIT_KWARGS["include_paths"]
    assert kwargs["is_python_module"] is False
    # TORCH_TARGET_VERSION must reach the cxx side so a non-stable header trips the tripwire.
    assert TORCH_TARGET_VERSION_DEFINE in kwargs["extra_cflags"]


def test_jit_compile_stable_abi_load_failure_returns_false(jit_stable_abi_patches):
    """Compile failures must not propagate — callers rely on the bool to
    fall through to legacy/pre-compiled paths instead of crashing at import."""
    jit_stable_abi_patches.load.side_effect = RuntimeError("no compiler")
    assert torch_cpp_ext.jit_compile_stable_abi_library(**_FAKE_JIT_KWARGS) is False


@pytest.mark.parametrize(
    "pin_rocm_arch,expect_ctx_entered",
    [pytest.param(True, True, id="default_narrows"), pytest.param(False, False, id="opt_out_skips")],
)
def test_jit_compile_stable_abi_rocm_arch_opt_out(jit_stable_abi_patches, pin_rocm_arch, expect_ctx_entered):
    """``pin_rocm_arch=False`` is the onnx escape hatch; verify it skips
    the context manager entirely instead of entering a no-op one (the no-op
    case would still log on non-HIP hosts via the helper's __exit__)."""
    torch_cpp_ext.jit_compile_stable_abi_library(pin_rocm_arch=pin_rocm_arch, **_FAKE_JIT_KWARGS)
    assert jit_stable_abi_patches.rocm_ctx.called is expect_ctx_entered


def test_jit_compile_stable_abi_windows_defines_only_apply_on_windows(jit_stable_abi_patches):
    """ORT_WINDOWS_DEFINE must reach cxx flags only when the host is Windows;
    otherwise the helper would surface a Windows-only ``-D`` on Linux CI."""
    sentinel = "-DSHOULD_BE_DROPPED_ON_LINUX"
    torch_cpp_ext.jit_compile_stable_abi_library(extra_windows_defines=[sentinel], **_FAKE_JIT_KWARGS)
    assert sentinel not in jit_stable_abi_patches.load.call_args.kwargs["extra_cflags"]


# --- set_rocm_user_architecture ---
#
# Each test patches ``torch_cpp_ext.torch.version`` so the HIP-gated branch
# can be exercised on CPU CI; ``patch.dict(os.environ, ..., clear=False)``
# restores the env at exit so test ordering can't leak ``PYTORCH_ROCM_ARCH``.


def test_set_rocm_user_architecture_noop_on_non_hip():
    with (
        patch(f"{C}.torch.version") as mock_ver,
        patch.dict(os.environ, {}, clear=False),
    ):
        mock_ver.hip = None
        os.environ.pop("PYTORCH_ROCM_ARCH", None)
        with torch_cpp_ext.set_rocm_user_architecture():
            assert "PYTORCH_ROCM_ARCH" not in os.environ
        assert "PYTORCH_ROCM_ARCH" not in os.environ


def test_set_rocm_user_architecture_pins_detected_archs():
    # gcnArchName carries trailing ``:sramecc+:xnack-`` feature flags upstream;
    # the helper must strip them so PYTORCH_ROCM_ARCH stays a bare arch list.
    fake_props = SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-")
    with (
        patch(f"{C}.torch.version") as mock_ver,
        patch(f"{C}.torch.cuda.device_count", return_value=1),
        patch(f"{C}.torch.cuda.get_device_properties", return_value=fake_props),
        patch.dict(os.environ, {}, clear=False),
    ):
        mock_ver.hip = "5.7.0"
        os.environ.pop("PYTORCH_ROCM_ARCH", None)
        with torch_cpp_ext.set_rocm_user_architecture():
            assert os.environ.get("PYTORCH_ROCM_ARCH") == "gfx942"
        # __exit__ removes the value it set, restoring the unset state.
        assert "PYTORCH_ROCM_ARCH" not in os.environ


def test_set_rocm_user_architecture_skips_when_no_gcn_arch():
    with (
        patch(f"{C}.torch.version") as mock_ver,
        patch(f"{C}.torch.cuda.device_count", return_value=1),
        patch(f"{C}.torch.cuda.get_device_properties", return_value=SimpleNamespace()),
        patch.dict(os.environ, {}, clear=False),
    ):
        mock_ver.hip = "5.7.0"
        os.environ.pop("PYTORCH_ROCM_ARCH", None)
        with torch_cpp_ext.set_rocm_user_architecture():
            assert "PYTORCH_ROCM_ARCH" not in os.environ


def test_set_rocm_user_architecture_enter_respects_user_set_env():
    # ``__enter__`` must not run detection / overwrite when the user has
    # already pinned the env var (the intended escape hatch). ``__exit__``
    # is intentionally not asserted here; see helper for current semantics.
    with (
        patch(f"{C}.torch.version") as mock_ver,
        patch(f"{C}.torch.cuda.device_count") as device_count,
        patch.dict(os.environ, {"PYTORCH_ROCM_ARCH": "gfx900"}, clear=False),
    ):
        mock_ver.hip = "5.7.0"
        ctx = torch_cpp_ext.set_rocm_user_architecture()
        ctx.__enter__()
        try:
            assert os.environ.get("PYTORCH_ROCM_ARCH") == "gfx900"
            device_count.assert_not_called()
        finally:
            ctx.__exit__(None, None, None)


def test_set_rocm_user_architecture_exit_logs_on_exception():
    # Exception path must not mutate ROCm arch env; it only surfaces diagnostics.
    with patch(f"{C}.logger.error") as log_err:
        ctx = torch_cpp_ext.set_rocm_user_architecture()
        ctx.__enter__()
        exc = ValueError("boom")
        ctx.__exit__(ValueError, exc, None)
        log_err.assert_called_once()


# --- flag composition + pre-compilation factory ---


_FAKE_DEFINES = (TORCH_TARGET_VERSION_DEFINE,)
_FAKE_WINDOWS_DEFINES = (ORT_WINDOWS_DEFINE,)


@pytest.mark.parametrize(
    "is_windows,debug,want_in_cxx,want_not_in_cxx",
    [
        pytest.param(False, False, ["-O3", "-std=c++17"], ["-g", "/Zi"], id="linux_release"),
        pytest.param(False, True, ["-O0", "-g"], ["/Zi"], id="linux_debug_appends_g"),
        pytest.param(
            True, False, ["/O2", "/std:c++17", ORT_WINDOWS_DEFINE], ["/Zi"], id="windows_release_keeps_win_defines"
        ),
        pytest.param(
            True, True, ["/Od", "/std:c++17", "/Zi", ORT_WINDOWS_DEFINE], ["-g"], id="windows_debug_appends_Zi"
        ),
    ],
)
def test_compose_compile_flags(is_windows, debug, want_in_cxx, want_not_in_cxx):
    cxx, nvcc = torch_cpp_ext.compose_compile_flags(
        use_cuda=True,
        is_windows=is_windows,
        debug=debug,
        extra_defines=_FAKE_DEFINES,
        extra_windows_defines=_FAKE_WINDOWS_DEFINES,
    )
    for f in want_in_cxx:
        assert f in cxx, f"expected {f!r} in cxx flags, got {cxx}"
    for f in want_not_in_cxx:
        assert f not in cxx, f"did NOT expect {f!r} in cxx flags, got {cxx}"
    # nvcc always carries -DUSE_CUDA and the TORCH_TARGET_VERSION tripwire.
    assert "-DUSE_CUDA" in nvcc
    assert TORCH_TARGET_VERSION_DEFINE in nvcc


def test_compose_compile_flags_linux_cpu_invariants():
    # cxx gets -DNO_GPU (not -DUSE_CUDA), nvcc always carries -DUSE_CUDA
    # regardless (.cu/.hip TUs are GPU-side), and Windows-only defines are
    # silently dropped off-Windows so callers don't have to branch.
    sentinel = "-DSHOULD_BE_DROPPED"
    cxx, nvcc = torch_cpp_ext.compose_compile_flags(
        use_cuda=False,
        is_windows=False,
        debug=False,
        extra_windows_defines=(sentinel,),
    )
    assert "-DNO_GPU" in cxx
    assert "-DUSE_CUDA" not in cxx
    assert sentinel not in cxx
    assert "-DUSE_CUDA" in nvcc


# Patch torch.utils.cpp_extension via sys.modules so we can assert the kwargs
# setuptools would receive without invoking the real BuildExtension.
@pytest.fixture
def setuptools_extension_patches():
    from unittest.mock import MagicMock

    cpu_ext = MagicMock(name="CppExtension")
    cuda_ext = MagicMock(name="CUDAExtension")
    with patch.dict(
        "sys.modules",
        {"torch.utils.cpp_extension": MagicMock(CppExtension=cpu_ext, CUDAExtension=cuda_ext)},
    ):
        yield SimpleNamespace(cpu_ext=cpu_ext, cuda_ext=cuda_ext)


_BASE_PRECOMPILE_KWARGS = dict(
    name="quark_test_ext",
    sources=["/fake/a.cc", "/fake/a.cu"],
    include_paths=["/fake/include"],
    extra_defines=list(_FAKE_DEFINES),
    extra_windows_defines=list(_FAKE_WINDOWS_DEFINES),
)


@pytest.mark.parametrize(
    "use_cuda,debug,is_windows,expected_cls,expected_link_args,define_macros,kwargs_overrides",
    [
        pytest.param(False, False, False, "cpu_ext", [], [], {}, id="cpu_linux_release"),
        pytest.param(True, False, False, "cuda_ext", [], [], {}, id="cuda_linux_release_picks_CUDAExtension"),
        pytest.param(False, True, False, "cpu_ext", ["-O0", "-g"], [], {}, id="cpu_linux_debug_adds_link_args"),
        pytest.param(False, True, True, "cpu_ext", ["/DEBUG"], [], {}, id="cpu_windows_debug_adds_DEBUG_link_arg"),
        pytest.param(False, False, False, "cpu_ext", [], [("MY_DEFINE", "1")], {}, id="forwards_define_macros"),
        pytest.param(
            False,
            False,
            False,
            "cpu_ext",
            [],
            [],
            {"sources": ["relative.cc", "relative.cu"]},
            id="relative_sources_forwarded_unchanged",
        ),
    ],
)
def test_make_setuptools_extension(
    setuptools_extension_patches,
    use_cuda,
    debug,
    is_windows,
    expected_cls,
    expected_link_args,
    define_macros,
    kwargs_overrides,
):
    """Pin the full pre-compilation factory contract: class dispatch, source path rewriting,
    Linux/Windows debug link-arg branches, and structured-macro forwarding."""
    call_kwargs_in = {**_BASE_PRECOMPILE_KWARGS, **kwargs_overrides}
    with patch(f"{C}.platform.system", return_value="Windows" if is_windows else "Linux"):
        torch_cpp_ext.make_setuptools_extension(
            use_cuda=use_cuda, debug=debug, define_macros=define_macros or None, **call_kwargs_in
        )

    chosen = getattr(setuptools_extension_patches, expected_cls)
    other = (
        setuptools_extension_patches.cpu_ext if expected_cls == "cuda_ext" else setuptools_extension_patches.cuda_ext
    )
    chosen.assert_called_once()
    other.assert_not_called()

    call_kwargs = chosen.call_args.kwargs
    # Absolute paths are rewritten because setuptools' editable builder
    # rejects them in ``Extension(sources=...)``.
    assert chosen.call_args.args[0] == call_kwargs_in["name"]
    forwarded_sources = chosen.call_args.args[1]
    assert all(not os.path.isabs(s) for s in forwarded_sources)
    expected_sources = [
        os.path.relpath(s).replace(os.sep, "/") if os.path.isabs(s) else s for s in call_kwargs_in["sources"]
    ]
    assert forwarded_sources == expected_sources
    assert call_kwargs["include_dirs"] == list(call_kwargs_in["include_paths"])
    assert call_kwargs["extra_link_args"] == expected_link_args
    assert call_kwargs["define_macros"] == define_macros
    # TORCH_TARGET_VERSION must reach the cxx side so a non-stable header trips the tripwire.
    assert "cxx" in call_kwargs["extra_compile_args"]
    assert "nvcc" in call_kwargs["extra_compile_args"]
    assert TORCH_TARGET_VERSION_DEFINE in call_kwargs["extra_compile_args"]["cxx"]


def test_make_setuptools_extension_isolated_includes_bypass_include_dirs(setuptools_extension_patches):
    """``extra_isolated_includes`` must reach cxx + nvcc as ``-I`` and stay out
    of ``include_dirs`` — same hipify-skip contract as the JIT helper."""
    isolated = ["/fake/ort/include", "/fake/ort/cuda"]
    with patch(f"{C}.platform.system", return_value="Linux"):
        torch_cpp_ext.make_setuptools_extension(
            name="quark_test_ext",
            sources=["a.cc"],
            include_paths=["/fake/include"],
            extra_isolated_includes=isolated,
            use_cuda=True,
        )
    call_kwargs = setuptools_extension_patches.cuda_ext.call_args.kwargs
    cxx_flags = call_kwargs["extra_compile_args"]["cxx"]
    nvcc_flags = call_kwargs["extra_compile_args"]["nvcc"]
    for path in isolated:
        assert f"-I{path}" in cxx_flags
        assert f"-I{path}" in nvcc_flags
        assert path not in call_kwargs["include_dirs"]
