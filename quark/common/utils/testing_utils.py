#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import functools
import importlib.metadata
import logging
import os
import platform
import random
import shutil
import sys
import tempfile
import unittest
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from functools import wraps
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from packaging import version

from quark.common.utils.import_utils import is_transformers_version_higher_or_equal  # noqa: E402
from quark.common.utils.log import CustomFormatter, ScreenLogger  # noqa: E402
from quark.common.utils.torch_utils import is_torch_higher_or_equal, torch_supports_stable_abi  # noqa: E402

from .import_utils import is_accelerate_available, is_torch_available, is_vllm_available  # noqa: E402

logger = ScreenLogger(__name__)

if is_torch_available():  # pragma: no cover
    # Set env var CUDA_VISIBLE_DEVICES="" to force cpu-mode
    import torch

    torch_device: str | torch.device | None = None
    if "QUARK_TEST_DEVICE" in os.environ:
        torch_device = os.environ["QUARK_TEST_DEVICE"]

        if torch_device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                f"QUARK_TEST_DEVICE={torch_device}, but CUDA is unavailable. Please double-check your testing environment."
            )

        try:
            # try creating device to see if provided device is valid
            torch_device = torch.device(torch_device)
        except RuntimeError as e:
            raise RuntimeError(
                f"Unknown testing device specified by environment variable `TRANSFORMERS_TEST_DEVICE`: {torch_device}"
            ) from e
    elif torch.cuda.is_available():
        torch_device = torch.device("cuda")
    else:
        torch_device = torch.device("cpu")
else:  # pragma: no cover
    torch_device = None

# Enables tests that are slow to run (disabled by default)
# Used with QUARK_TEST_SKIP_FAST to run either slow or fast tests **only**.
TEST_WITH_SLOW = os.getenv("QUARK_TEST_WITH_SLOW", "0") == "1"

# Disables non-slow tests (enabled by default)
# Used with TEST_WITH_SLOW to run either slow or fast tests **only**.
TEST_SKIP_FAST = os.getenv("QUARK_TEST_SKIP_FAST", "0") == "1"

# Enables extensive/local-only tests (disabled by default)
TEST_WITH_EXTENSIVE = os.getenv("QUARK_EXTENSIVE_TEST", "0") == "1"


def find_repo_root(start_path: Path | None = None) -> Path:
    """Find the repository root directory by looking for marker files.

    Resolution when ``start_path`` is not provided, tried in order:

    1. Walk up from this module's ``__file__`` (works for editable installs).
    2. Walk up from the current working directory. Wheel installs import
       ``quark`` from ``site-packages`` -- no marker above it -- but the test
       suite runs from the repo checkout, so the CWD still resolves the root.

    An explicit ``start_path`` skips the fallbacks and walks up from it alone.

    Args:
        start_path: Starting path for search. If None, the fallbacks above
            are tried in order.

    Returns:
        Path to repository root.

    Raises:
        RuntimeError: If repository root cannot be found.

    Example:
        >>> from quark.common.utils.testing_utils import find_repo_root
        >>> repo_root = find_repo_root()
        >>> assert (repo_root / "pyproject.toml").exists()
    """
    if start_path is not None:
        starts = [start_path if isinstance(start_path, Path) else Path(start_path)]
    else:
        starts = [Path(__file__), Path.cwd()]

    for start in starts:
        start = start.resolve()
        # Include ``start`` itself: the CWD fallback is already a directory
        # that may be the root, whereas ``parents`` alone would skip it.
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").exists() or (candidate / "setup.py").exists():
                return candidate

    raise RuntimeError("Could not find repository root (no pyproject.toml or setup.py found)")


def add_repo_root_to_sys_path() -> None:
    """Add the repository root directory to sys.path if not already present.

    This is useful for tests that need to import modules from the tools directory
    or other non-package directories in the repository.

    This function can be imported without torch being installed.

    Example:
        >>> from quark.common.utils.testing_utils import add_repo_root_to_sys_path
        >>> add_repo_root_to_sys_path()
        >>> from tools.ci.some_script import some_function
    """
    repo_root = find_repo_root()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


@contextmanager
def set_environment_variables(**env_vars: str | None) -> Iterator[None]:
    """
    Temporarily set environment variables within a context.

    Environment variables are automatically restored to their original values
    when the context exits, even if an exception occurs.

    Args:
        **env_vars: Environment variables to set as keyword arguments.
                   Set to None to unset a variable.

    Usage:
        with set_environment_variables(API_URL="http://test.url", ANOTHER_VAR="value"):
            assert os.getenv("API_URL") == "http://test.url"
            # ... your code under test ...

        # Variables restored after exiting the 'with' block
        assert os.getenv("API_URL") is None

        # To unset a variable:
        with set_environment_variables(MY_VAR=None):
            assert os.getenv("MY_VAR") is None
    """
    original_env = {k: os.getenv(k) for k in env_vars}
    try:
        for k, val in env_vars.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
        yield
    finally:
        for k, val in original_env.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val


def with_env(**env_vars: str | None) -> Callable[[Any], Any]:
    """
    Decorator for setting environment variables for a test function.

    Args:
        **env_vars: Environment variables to set as keyword arguments.

    Usage:
        @with_env(QUARK_ALGO_DEBUG="1", CUDA_VISIBLE_DEVICES="0")
        def test_algorithm():
            assert os.getenv("QUARK_ALGO_DEBUG") == "1"
    """

    def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with set_environment_variables(**env_vars):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def require_torch_cuda(test_case: Any) -> Any:  # pragma: no cover
    """Decorator marking a test that requires CUDA and PyTorch."""
    return unittest.skipUnless(
        isinstance(torch_device, torch.device) and torch_device.type == "cuda", "test requires CUDA"
    )(test_case)


def require_torch_multi_gpu(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator marking a test that requires CUDA with at least two GPUs and PyTorch."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            if not (torch.cuda.is_available() and torch.cuda.device_count() >= 2):
                pytest.skip("Test requires CUDA with at least two GPUs; skipping.")
            return fn(*args, **kwargs)
        except ImportError:
            pytest.skip("PyTorch not available; skipping test.")

    return wrapper


def require_torch_hip(test_case: Any) -> Any:  # pragma: no cover
    """Decorator marking a test that requires HIP."""
    return unittest.skipUnless(torch.version.hip is not None, "test requires HIP")(test_case)


def require_accelerate(test_case: Any) -> Any:  # pragma: no cover
    """Decorator marking a test that requires Accelerate library."""
    return unittest.skipUnless(is_accelerate_available(), "test requires accelerate")(test_case)


def require_linux(test_case: Any) -> Any:  # pragma: no cover
    """Decorator marking a test that requires Linux."""
    return unittest.skipUnless(platform.system() == "Linux", "test requires Linux")(test_case)


def require_torch_higher_or_equal(min_version: str) -> Callable[[Any], Any]:
    """Decorator marking a test that requires torch >= min_version."""

    def decorator(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_torch_higher_or_equal(min_version):
                pytest.skip(f"Test requires torch >= {min_version}, current is {torch.__version__}")
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def require_torch_lower_or_equal(max_version: str) -> Callable[[Any], Any]:
    """Decorator marking a test that requires torch <= max_version."""

    def decorator(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            torch_version = version.parse(torch.__version__.split("+")[0])
            if torch_version > version.parse(max_version):
                pytest.skip(f"Test requires torch <= {max_version}, current is {torch.__version__}")
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def require_vllm(test_case: Any) -> Any:  # pragma: no cover
    """Decorator marking a test that requires vllm."""
    return unittest.skipUnless(is_vllm_available(), "test requires vllm")(test_case)


def skip_torch_version(skip_version: str) -> Callable[[Any], Any]:
    """Decorator marking a test that skips if the torch version is equal to the given version"""

    def decorator(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            torch_version = version.parse(torch.__version__.split("+")[0])
            if torch_version == version.parse(skip_version):
                pytest.skip(
                    f"Test skips if the torch version is equal to {skip_version}, current is {torch.__version__}"
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def use_temporary_directory(func):  # type: ignore
    # Preserve pytest marks (skipif/parametrize/...) on the wrapped test.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore
        # `wraps` exposes the inner signature to pytest, which will also try to inject
        # the built-in `tmpdir` fixture when the test declares `tmpdir=...`. This
        # decorator always supplies its own temp directory; drop any injected value
        # to avoid "multiple values for keyword argument 'tmpdir'".
        kwargs.pop("tmpdir", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            return func(*args, **kwargs, tmpdir=tmpdir)

    return wrapper


def delete_directory_content(directory: str) -> None:  # pragma: no cover
    """Deletes all content within a directory

    Args:
        directory (str): The path to the directory whose content should be deleted.
    """
    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"{directory} is not a valid directory.")


def retry_flaky_test(max_attempts: int = 5):  # type: ignore
    """
    Allows to retry flaky tests multiple times.
    """

    def decorator(test_func):  # type: ignore
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):  # type: ignore
            retry_count = 1

            while retry_count < max_attempts:
                try:
                    return test_func(*args, **kwargs)
                except Exception as exception:  # pragma: no cover
                    print(f"Test failed with exception {exception} at try {retry_count}/{max_attempts}.")
                    retry_count += 1

            return test_func(*args, **kwargs)  # pragma: no cover

        return wrapper

    return decorator


def skip_if_amd_quark_nightly_wheel_is_installed(test_case: Any) -> Any:  # pragma: no cover
    """Decorator marking a test that require non-nightly amd-quark packages."""

    is_not_nightly_package = True
    try:
        assert importlib.metadata.metadata("amd-quark") is not None
    except importlib.metadata.PackageNotFoundError:
        is_not_nightly_package = False

    return unittest.skipUnless(is_not_nightly_package, "test requires official `amd-quark` package")(test_case)


class PatchEverywhere:
    """
    Finds all occurences of ``attribute_name`` in the loaded modules and patches them with ``patch``, which can be a function, a variable, a class, etc.

    :param str attribute_name: The name of attribute to patch.
    :param Any patch: The patch for the attribute.
    :param Optional[str] module_name_prefix: If set, only module names starting with this prefix will be considered for patching. Defaults to ``None``.
    """

    def __init__(
        self,
        attribute_name: str,
        patch: Any,
        module_name_prefix: str | None = None,
    ):
        self.attribute_name = attribute_name
        self.patch = patch
        self.module_name_prefix = module_name_prefix

        self.originals = {}
        for name in list(sys.modules):
            module = sys.modules[name]
            if module_name_prefix is not None and not name.startswith(module_name_prefix):
                continue
            if hasattr(module, attribute_name):
                self.originals[module.__name__ + attribute_name] = getattr(module, attribute_name)

    def __enter__(self) -> None:
        for name in list(sys.modules):
            module = sys.modules[name]
            if self.module_name_prefix is not None and not name.startswith(self.module_name_prefix):
                continue
            if hasattr(module, self.attribute_name):
                setattr(module, self.attribute_name, self.patch)

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        for name in list(sys.modules):
            module = sys.modules[name]
            if self.module_name_prefix is not None and not name.startswith(self.module_name_prefix):
                continue
            if hasattr(module, self.attribute_name):
                key = module.__name__ + self.attribute_name
                if key not in self.originals:
                    raise ValueError(f"{key} not found in {self.originals.keys()}")

                setattr(module, self.attribute_name, self.originals[key])


def slow_test(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Marks the test as slow and skip it if QUARK_TEST_WITH_SLOW env var is not set.

    Note: When the test has multiple decorators, `slow_test` must be the first decorator (at the top)
    """

    @functools.wraps(fn)
    def wrapper(*args: tuple[Any] | None, **kwargs: dict[Any, Any] | None) -> Any:
        if not TEST_WITH_SLOW:
            pytest.skip("Skipping slow test; set QUARK_TEST_WITH_SLOW=1 to enable.")
        return fn(*args, **kwargs)

    wrapper.__dict__["slow_test"] = True  # Use by class TestCase(unittest.TestCase).setUp
    return wrapper


def slow_test_if(condition: bool) -> Callable[[Any], Any]:
    """Decorator to mark test as slow if `condition` is `True`."""
    return slow_test if condition else lambda fn: fn


def local_test_only(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:  # pragma: no cover
    """Marks the test as local-only (extensive) and skips it unless QUARK_EXTENSIVE_TEST=1 is set.
    These tests are too heavy or environment-specific for CI and are useful only for local debugging.
    """

    @wraps(fn)
    def wrapper(*args: tuple[Any] | None, **kwargs: dict[Any, Any] | None) -> None:
        if not TEST_WITH_EXTENSIVE:
            pytest.skip("Skipping extensive test; set QUARK_EXTENSIVE_TEST=1 to enable.")
        return fn(*args, **kwargs)

    return wrapper


def skip_if_no_gpu(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Decorator to skip the test if no GPU is available."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            if not torch.cuda.is_available():
                pytest.skip("Test requires GPU; skipping.")
            return fn(*args, **kwargs)
        except ImportError:
            logger.warning("PyTorch not detected. skip_if_no_gpu will be a no-op.")
            return fn(*args, **kwargs)

    return wrapper


class TestCase(unittest.TestCase):
    """TestCase subclass that handles slow test skipping.

    When QUARK_TEST_SKIP_FAST=1 is set, only tests marked with @slow_test will run.
    """

    def setUp(self) -> None:
        """Check if this test should be skipped based on QUARK_TEST_SKIP_FAST."""
        if TEST_SKIP_FAST:
            if not getattr(self, self._testMethodName).__dict__.get("slow_test", False):
                raise unittest.SkipTest("test is fast; we disabled it with QUARK_TEST_SKIP_FAST")


@contextmanager
def capture_quark_logs() -> Iterator[None]:
    """Context manager to capture all quark logger output to a StringIO buffer."""
    old_stderr = sys.stderr
    mystderr = StringIO()
    sys.stderr = mystderr

    # Remove existing handlers and add new ones with the captured stream
    saved_handlers = {}

    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if logger_name.startswith("quark"):
            logger_obj = logging.getLogger(logger_name)
            saved_handlers[logger_name] = logger_obj.handlers[:]
            logger_obj.handlers.clear()

            # Add new handler with captured stream
            new_handler = logging.StreamHandler(mystderr)
            new_handler.setFormatter(CustomFormatter())
            logger_obj.addHandler(new_handler)

    try:
        yield mystderr
    finally:
        # Restore original handlers
        for logger_name, handlers in saved_handlers.items():
            logger_obj = logging.getLogger(logger_name)
            logger_obj.handlers.clear()
            logger_obj.handlers.extend(handlers)

        # Restore original stderr
        sys.stderr = old_stderr


# TODO: Remove `experts_implementation="eager"` once we drop torch 2.9 support (grouped gemm not available on rocm on torch==2.9).
if is_transformers_version_higher_or_equal("5.0"):
    FROM_PRETRAINED_KWARGS = {"experts_implementation": "eager"}
else:
    FROM_PRETRAINED_KWARGS = {}


# ---------------------------------------------------------------------------
# Stable-ABI vs legacy pybind11 ops equivalence helpers for pipeline tests.
#
# An ops module (e.g. ``quark.torch.kernel.hw_emulation`` or
# ``quark.onnx.operators.custom_ops``) may expose two Python-facing entry
# points into the same set of C++ kernels: a stable-ABI ``torch.ops.<ns>``
# surface (PyTorch >= 2.10) and a legacy pybind11 ``.so``. Both forward to
# the **same** underlying kernels, so end-to-end pipeline output is required
# to match bit-for-bit.
#
# ``run_op_variants`` runs the pipeline through each provided variant,
# reseeding the RNG between runs. When both variants are provided it asserts
# ``stable == legacy`` bit-for-bit; when only the legacy variant is provided
# (typical on PyTorch < 2.10, where the stable-ABI surface isn't registered)
# it runs only that one and returns its output. CI lanes that should fail
# loudly when the stable surface didn't load are expected to gate on that
# in the caller (e.g. ``assert has_stable_X()`` before calling) rather than
# inside this helper. The output of one variant is returned (the
# stable one when both ran, since they're bit-equivalent) so callers that
# also want a looser golden check (e.g.
# ``assert_outputs_equivalent(out, golden, atol=...)``) can do so on the
# result; whenever both variants ran, ``legacy ~= golden`` follows
# transitively from the bit-exact equivalence assertion.
#
# Concrete variant-selecting context managers live next to each ops module's
# dispatch state (see e.g. ``quark.onnx.algorithm.finetuning.create_torch
# .testing_backends``); this module is intentionally agnostic about them.
# ---------------------------------------------------------------------------


def _reseed_rngs(seed: int) -> None:
    """Reset torch / numpy / stdlib RNGs so two pipeline runs start from
    identical random state.
    """
    if is_torch_available():  # pragma: no cover
        import torch as _torch

        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _to_ndarray(t: "torch.Tensor") -> np.ndarray:
    """Materialize ``t`` as a NumPy array, promoting through float32 for
    any dtype without a native NumPy mapping.

    NumPy has no native dtype for ``bfloat16`` or the ``float8_*``
    variants (and, presumably, any future low-precision dtype torch
    adds), so calling ``.numpy()`` on those raises ``TypeError``. Caught
    and retried via fp32 promotion - which is bit-exact for all such
    dtypes today: bf16/fp16/fp8 are strict bit-subsets of fp32 (literally
    fp32 with low-mantissa bits truncated), so equality on the fp32 side
    ⇔ equality on the original side.
    """
    import torch

    t = t.detach().cpu()
    try:
        return t.numpy()
    except TypeError:
        return t.to(torch.float32).numpy()


def assert_outputs_equivalent(
    a: Any,
    b: Any,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    ctx: str = "",
) -> None:
    """Recursively compare pipeline outputs (ndarrays, tensors, lists,
    tuples, scalars).

    By default, leaves ``atol`` / ``rtol`` unset so they fall through to
    ``np.testing.assert_allclose``'s own defaults (``rtol=1e-7``, ``atol=0``)
    - i.e. "approximately equal", suitable for golden checks where the user
    didn't explicitly opt into bit-exact. Callers that need strict bit-exact
    equivalence (e.g. the internal stable-vs-legacy comparison in
    :func:`run_op_variants`, where both variants reach the same C++
    kernels) should pass ``atol=0, rtol=0`` explicitly. Callers with
    legitimate sources of numeric drift can relax via larger ``atol`` /
    ``rtol`` values.

    Tensors are materialized via :func:`_to_ndarray` (with fp32 promotion
    for bf16/fp8 dtypes that lack a native NumPy mapping). Anything that
    survives the list/tuple recursion is then funnelled through
    ``np.testing.assert_allclose`` — which broadcasts internally (handling
    the ``[ndarray]`` ORT-``sess.run`` output vs bare ``ndarray`` golden
    shape relation) and emits a per-element mismatch summary on failure.
    """
    import torch

    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        # ``list`` / ``tuple`` are treated as interchangeable at the container
        # level: torch ops surface multi-tensor returns as ``list`` for
        # ``Tensor[]`` schemas and ``tuple`` for fixed-arity returns.
        assert len(a) == len(b), f"length mismatch {len(a)} vs {len(b)} ({ctx})"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            assert_outputs_equivalent(x, y, atol=atol, rtol=rtol, ctx=f"{ctx}[{i}]")
        return

    if isinstance(a, torch.Tensor):
        a = _to_ndarray(a)
    if isinstance(b, torch.Tensor):
        b = _to_ndarray(b)

    # Only forward tolerances the caller actually specified; leave the
    # rest at numpy's defaults (rtol=1e-7, atol=0).
    np.testing.assert_allclose(
        np.asarray(a),
        np.asarray(b),
        rtol=rtol if rtol is not None else 1e-7,
        atol=atol if atol is not None else 0.0,
        err_msg=ctx,
    )


def run_op_variants(
    pipeline_fn: Callable[[], Any],
    *,
    legacy_ctx: AbstractContextManager[None] | None = None,
    stable_ctx: AbstractContextManager[None] | None = None,
    seed: int = 42,
) -> Any:
    """Run ``pipeline_fn()`` under each provided variant context and assert
    bit-exact equivalence when both are provided.

    The two variants are the same C++ kernels reached via the legacy pybind11
    ``.so`` and the stable-ABI ``torch.ops.<ns>`` surface. If only one
    ``*_ctx`` is provided the helper runs just that one (callers typically
    route through factories that return ``None`` when a variant isn't
    loadable). Raises ``ValueError`` if both are ``None``.

    Raises ``RuntimeError`` when ``legacy_ctx is None`` on a torch that
    supports the stable ABI (see :func:`torch_supports_stable_abi`): the
    legacy surface is expected to be loadable there too, and silently
    skipping it would drop the bit-exact cross-check.

    :param pipeline_fn: Zero-argument callable returning the comparison
        target.
    :param seed: RNG seed applied before each variant's run.
    :return: Stable output if both ran (bit-equivalent to legacy), otherwise
        whichever single variant ran.
    """
    if legacy_ctx is None and stable_ctx is None:
        raise ValueError("run_op_variants requires at least one of legacy_ctx / stable_ctx")

    def _run(ctx: AbstractContextManager[None]) -> Any:
        with ctx:
            _reseed_rngs(seed)
            return pipeline_fn()

    if stable_ctx is None:
        return _run(legacy_ctx)
    if legacy_ctx is None:
        # Gate explicitly so a caller that synthesizes stable_ctx on older
        # torch doesn't get a spurious error.
        if is_torch_available() and torch_supports_stable_abi():
            raise RuntimeError(
                "run_op_variants: legacy_ctx is None on PyTorch supporting the stable ABI; "
                "refusing to run stable-ABI variant only because that would skip the "
                "stable-vs-legacy bit-exact cross-check. Verify the legacy pybind11 ops "
                "module loaded."
            )
        return _run(stable_ctx)

    out_stable = _run(stable_ctx)
    out_legacy = _run(legacy_ctx)

    # Same C++ kernels on both sides; demand bit-exact equivalence.
    assert_outputs_equivalent(out_stable, out_legacy, atol=0, rtol=0, ctx="stable-ABI vs legacy pybind11")
    return out_stable
