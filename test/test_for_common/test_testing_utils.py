#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for environment variable utilities in testing_utils.py."""

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest

from quark.common.utils.import_utils import is_torch_available
from quark.common.utils.testing_utils import (
    _reseed_rngs,
    assert_outputs_equivalent,
    run_op_variants,
    set_environment_variables,
    skip_if_no_gpu,
    slow_test,
    slow_test_if,
    with_env,
)


class TestSetEnvironmentVariables:
    """Tests for set_environment_variables context manager."""

    def test_set_env_vars_basic(self):
        """Test basic environment variable setting and restoration."""
        assert os.getenv("TEST_VAR_1") is None

        with set_environment_variables(TEST_VAR_1="value1"):
            assert os.getenv("TEST_VAR_1") == "value1"

        assert os.getenv("TEST_VAR_1") is None

    def test_set_env_vars_multiple(self):
        """Test setting multiple environment variables."""
        with set_environment_variables(VAR1="val1", VAR2="val2", VAR3="val3"):
            assert os.getenv("VAR1") == "val1"
            assert os.getenv("VAR2") == "val2"
            assert os.getenv("VAR3") == "val3"

        assert os.getenv("VAR1") is None
        assert os.getenv("VAR2") is None
        assert os.getenv("VAR3") is None

    def test_set_env_vars_preserves_existing(self):
        """Test that existing variables are preserved and restored."""
        os.environ["EXISTING_VAR"] = "original"

        with set_environment_variables(EXISTING_VAR="modified"):
            assert os.getenv("EXISTING_VAR") == "modified"

        assert os.getenv("EXISTING_VAR") == "original"

        del os.environ["EXISTING_VAR"]

    def test_set_env_vars_unset_with_none(self):
        """Test unsetting variables with None."""
        os.environ["TO_UNSET"] = "value"

        with set_environment_variables(TO_UNSET=None):
            assert os.getenv("TO_UNSET") is None

        assert os.getenv("TO_UNSET") == "value"

        del os.environ["TO_UNSET"]

    def test_set_env_vars_unset_nonexistent_with_none(self):
        """Test unsetting a variable that doesn't exist."""
        # Ensure variable doesn't exist
        assert os.getenv("NONEXISTENT_VAR") is None

        # Try to unset it (should not raise error)
        with set_environment_variables(NONEXISTENT_VAR=None):
            assert os.getenv("NONEXISTENT_VAR") is None

        # Still shouldn't exist
        assert os.getenv("NONEXISTENT_VAR") is None

    def test_set_env_vars_restore_none(self):
        """Test restoring a variable that was originally None (didn't exist)."""
        # Ensure variable doesn't exist initially
        assert os.getenv("NEW_VAR") is None

        # Set it temporarily
        with set_environment_variables(NEW_VAR="temporary"):
            assert os.getenv("NEW_VAR") == "temporary"

        # Should be removed (restored to None/nonexistent)
        assert os.getenv("NEW_VAR") is None

    def test_set_env_vars_exception_handling(self):
        """Test that variables are restored even on exception."""
        os.environ["EXCEPTION_VAR"] = "original"

        with pytest.raises(ValueError), set_environment_variables(EXCEPTION_VAR="modified"):
            assert os.getenv("EXCEPTION_VAR") == "modified"
            raise ValueError("Test exception")

        assert os.getenv("EXCEPTION_VAR") == "original"

        del os.environ["EXCEPTION_VAR"]

    def test_set_env_vars_exception_with_none(self):
        """Test that unsetting is restored even on exception."""
        os.environ["EXCEPTION_UNSET"] = "original"

        with pytest.raises(RuntimeError), set_environment_variables(EXCEPTION_UNSET=None):
            assert os.getenv("EXCEPTION_UNSET") is None
            raise RuntimeError("Test exception")

        # Should be restored to original value
        assert os.getenv("EXCEPTION_UNSET") == "original"

        del os.environ["EXCEPTION_UNSET"]

    def test_set_env_vars_nested(self):
        """Test nested context managers."""
        with set_environment_variables(OUTER="outer_value"):
            assert os.getenv("OUTER") == "outer_value"

            with set_environment_variables(INNER="inner_value", OUTER="modified"):
                assert os.getenv("OUTER") == "modified"
                assert os.getenv("INNER") == "inner_value"

            # Inner context restored OUTER
            assert os.getenv("OUTER") == "outer_value"
            assert os.getenv("INNER") is None

        assert os.getenv("OUTER") is None

    def test_set_env_vars_mixed_existing_and_new(self):
        """Test setting a mix of existing and new variables."""
        # Set one existing variable
        os.environ["EXISTING"] = "original"

        # Mix existing modification with new variable
        with set_environment_variables(EXISTING="modified", NEW="new_value"):
            assert os.getenv("EXISTING") == "modified"
            assert os.getenv("NEW") == "new_value"

        # Existing should be restored, new should be removed
        assert os.getenv("EXISTING") == "original"
        assert os.getenv("NEW") is None

        del os.environ["EXISTING"]

    def test_set_env_vars_mixed_set_and_unset(self):
        """Test setting some vars and unsetting others in same context."""
        os.environ["TO_UNSET"] = "original"
        os.environ["TO_MODIFY"] = "original"

        with set_environment_variables(TO_UNSET=None, TO_MODIFY="modified", NEW_VAR="new"):
            assert os.getenv("TO_UNSET") is None
            assert os.getenv("TO_MODIFY") == "modified"
            assert os.getenv("NEW_VAR") == "new"

        # Verify restoration
        assert os.getenv("TO_UNSET") == "original"
        assert os.getenv("TO_MODIFY") == "original"
        assert os.getenv("NEW_VAR") is None

        del os.environ["TO_UNSET"]
        del os.environ["TO_MODIFY"]

    def test_empty_context(self):
        """Test context manager with no variables."""
        with set_environment_variables():
            # Should work fine with no variables
            pass


class TestWithEnvDecorator:
    """Tests for with_env decorator."""

    @with_env(DECORATOR_VAR="decorator_value")
    def test_with_env_decorator(self):
        """Test the decorator with single variable."""
        assert os.getenv("DECORATOR_VAR") == "decorator_value"

    @with_env(VAR1="value1", VAR2="value2", VAR3="value3")
    def test_with_env_decorator_multiple(self):
        """Test the decorator with multiple variables."""
        assert os.getenv("VAR1") == "value1"
        assert os.getenv("VAR2") == "value2"
        assert os.getenv("VAR3") == "value3"

    @with_env(UNSET_VAR=None)
    def test_with_env_decorator_unset(self):
        """Test the decorator with None to unset."""
        assert os.getenv("UNSET_VAR") is None

    def test_with_env_decorator_preserves_existing(self):
        """Test that decorator preserves existing variables."""
        os.environ["PRESERVE_ME"] = "original"

        @with_env(PRESERVE_ME="modified")
        def test_func():
            assert os.getenv("PRESERVE_ME") == "modified"
            return "success"

        result = test_func()
        assert result == "success"
        assert os.getenv("PRESERVE_ME") == "original"

        del os.environ["PRESERVE_ME"]

    def test_with_env_decorator_with_args(self):
        """Test decorator on function with arguments."""

        @with_env(TEST_ARG_VAR="env_value")
        def test_func(arg1, arg2, kwarg1=None):
            assert os.getenv("TEST_ARG_VAR") == "env_value"
            return arg1 + arg2 + (kwarg1 or 0)

        result = test_func(1, 2, kwarg1=3)
        assert result == 6
        assert os.getenv("TEST_ARG_VAR") is None

    def test_with_env_decorator_exception(self):
        """Test that decorator restores on exception."""
        os.environ["DECORATOR_EXCEPTION"] = "original"

        @with_env(DECORATOR_EXCEPTION="modified")
        def test_func():
            assert os.getenv("DECORATOR_EXCEPTION") == "modified"
            raise ValueError("Test exception")

        with pytest.raises(ValueError):
            test_func()

        assert os.getenv("DECORATOR_EXCEPTION") == "original"

        del os.environ["DECORATOR_EXCEPTION"]

    def test_decorator_cleanup(self):
        """Verify decorator cleaned up all test variables."""
        assert os.getenv("DECORATOR_VAR") is None
        assert os.getenv("VAR1") is None
        assert os.getenv("VAR2") is None
        assert os.getenv("VAR3") is None
        assert os.getenv("UNSET_VAR") is None
        assert os.getenv("TEST_ARG_VAR") is None

    def test_empty_decorator(self):
        """Test decorator with no variables."""

        @with_env()
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"


class TestEnvVarsFixture:
    """Tests for env_vars pytest fixture."""

    def test_env_vars_fixture(self, env_vars):
        """Test the pytest fixture."""
        with env_vars(FIXTURE_VAR="fixture_value"):
            assert os.getenv("FIXTURE_VAR") == "fixture_value"

    def test_fixture_cleanup(self):
        """Verify fixture cleaned up."""
        assert os.getenv("FIXTURE_VAR") is None


class TestSlowTestDecorator:
    """Tests for slow_test decorator."""

    def test_slow_test_skips_when_not_enabled(self, monkeypatch):
        """Test that slow_test skips when QUARK_TEST_WITH_SLOW is not set."""
        monkeypatch.setenv("QUARK_TEST_WITH_SLOW", "0")
        import importlib

        import quark.common.utils.testing_utils

        importlib.reload(quark.common.utils.testing_utils)
        from quark.common.utils.testing_utils import slow_test

        @slow_test
        def my_slow_test():
            return "executed"

        with pytest.raises(pytest.skip.Exception, match="Skipping slow test"):
            my_slow_test()

    def test_slow_test_runs_when_enabled(self, monkeypatch):
        """Test that slow_test runs when QUARK_TEST_WITH_SLOW=1."""
        monkeypatch.setenv("QUARK_TEST_WITH_SLOW", "1")
        import importlib

        import quark.common.utils.testing_utils

        importlib.reload(quark.common.utils.testing_utils)
        from quark.common.utils.testing_utils import slow_test

        @slow_test
        def my_slow_test():
            return "executed"

        result = my_slow_test()
        assert result == "executed"

    def test_slow_test_sets_marker_attribute(self):
        """Test that slow_test sets the slow_test attribute."""

        @slow_test
        def my_slow_test():
            pass

        assert my_slow_test.__dict__.get("slow_test") is True


class TestSlowTestIfDecorator:
    """Tests for slow_test_if decorator."""

    def test_slow_test_if_true_applies_slow_test(self, monkeypatch):
        """Test that slow_test_if(True) applies slow_test decorator."""
        monkeypatch.setenv("QUARK_TEST_WITH_SLOW", "0")
        import importlib

        import quark.common.utils.testing_utils

        importlib.reload(quark.common.utils.testing_utils)
        from quark.common.utils.testing_utils import slow_test_if

        @slow_test_if(True)
        def my_test():
            return "executed"

        with pytest.raises(pytest.skip.Exception):
            my_test()

    def test_slow_test_if_false_does_not_apply(self):
        """Test that slow_test_if(False) does not apply slow_test decorator."""

        @slow_test_if(False)
        def my_test():
            return "executed"

        result = my_test()
        assert result == "executed"


class TestSkipIfNoGpu:
    """Tests for skip_if_no_gpu decorator."""

    def test_skip_if_no_gpu_skips_when_cuda_unavailable(self):
        """Test that skip_if_no_gpu skips when CUDA is not available."""
        # Patch torch at the function level to avoid global import issues
        with patch("quark.common.utils.testing_utils.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            @skip_if_no_gpu
            def my_gpu_test():
                return "executed"

            with pytest.raises(pytest.skip.Exception, match="Test requires GPU"):
                my_gpu_test()

    def test_skip_if_no_gpu_runs_when_cuda_available(self):
        """Test that skip_if_no_gpu runs when CUDA is available."""
        # Patch torch to simulate GPU available - this will exercise line 320
        with patch("quark.common.utils.testing_utils.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True

            @skip_if_no_gpu
            def my_gpu_test():
                return "executed"

            result = my_gpu_test()
            assert result == "executed"

    def test_skip_if_no_gpu_handles_import_error(self):
        """Test that skip_if_no_gpu handles ImportError gracefully."""
        # Simulate ImportError when accessing torch.cuda - this exercises lines 322-323
        with patch("quark.common.utils.testing_utils.torch") as mock_torch:
            mock_torch.cuda.is_available.side_effect = ImportError("torch not available")

            @skip_if_no_gpu
            def my_gpu_test():
                return "executed"

            # Should run without error (warning logged)
            result = my_gpu_test()
            assert result == "executed"


class TestAddRepoRootToSysPath:
    """Tests for add_repo_root_to_sys_path utility."""

    def test_add_repo_root_to_sys_path_adds_path(self):
        """Test that add_repo_root_to_sys_path adds repo root to sys.path."""
        import sys

        from quark.common.utils.testing_utils import add_repo_root_to_sys_path

        # Record initial sys.path length
        initial_path_len = len(sys.path)

        # Call the function
        add_repo_root_to_sys_path()

        # Verify that a path was added (or already existed)
        # The function is idempotent - it only adds if not present
        assert len(sys.path) >= initial_path_len

        # Verify we can now import from tools directory
        from pathlib import Path

        # Check that repo root is in path
        repo_root = None
        for parent in Path(__file__).resolve().parents:
            if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
                repo_root = parent
                break

        assert repo_root is not None
        assert str(repo_root) in sys.path

    def test_add_repo_root_idempotent(self):
        """Test that add_repo_root_to_sys_path is idempotent."""
        import sys

        from quark.common.utils.testing_utils import add_repo_root_to_sys_path

        # Call multiple times
        add_repo_root_to_sys_path()
        path_after_first = sys.path.copy()

        add_repo_root_to_sys_path()
        path_after_second = sys.path.copy()

        # Path should be the same (no duplicates)
        assert path_after_first == path_after_second


class TestTestCaseClass:
    """Tests for TestCase class."""

    def test_testcase_skips_fast_tests_when_skip_fast_enabled(self, monkeypatch):
        """Test that TestCase skips fast tests when QUARK_TEST_SKIP_FAST=1."""
        monkeypatch.setenv("QUARK_TEST_SKIP_FAST", "1")
        import importlib

        import quark.common.utils.testing_utils

        importlib.reload(quark.common.utils.testing_utils)
        from quark.common.utils.testing_utils import TestCase as ReloadedTestCase

        class MyTestCase(ReloadedTestCase):
            def test_fast(self):
                """A fast test."""
                pass

        test_instance = MyTestCase("test_fast")
        with pytest.raises(unittest.SkipTest, match="test is fast"):
            test_instance.setUp()

    def test_testcase_runs_slow_tests_when_skip_fast_enabled(self, monkeypatch):
        """Test that TestCase runs slow tests when QUARK_TEST_SKIP_FAST=1."""
        monkeypatch.setenv("QUARK_TEST_SKIP_FAST", "1")
        monkeypatch.setenv("QUARK_TEST_WITH_SLOW", "1")
        import importlib

        import quark.common.utils.testing_utils

        importlib.reload(quark.common.utils.testing_utils)
        from quark.common.utils.testing_utils import TestCase as ReloadedTestCase
        from quark.common.utils.testing_utils import slow_test

        class MyTestCase(ReloadedTestCase):
            @slow_test
            def test_slow(self):
                """A slow test."""
                pass

        test_instance = MyTestCase("test_slow")
        # Should not raise
        test_instance.setUp()

    def test_testcase_runs_all_tests_when_skip_fast_disabled(self, monkeypatch):
        """Test that TestCase runs all tests when QUARK_TEST_SKIP_FAST=0."""
        monkeypatch.setenv("QUARK_TEST_SKIP_FAST", "0")
        import importlib

        import quark.common.utils.testing_utils

        importlib.reload(quark.common.utils.testing_utils)
        from quark.common.utils.testing_utils import TestCase as ReloadedTestCase

        class MyTestCase(ReloadedTestCase):
            def test_fast(self):
                """A fast test."""
                pass

        test_instance = MyTestCase("test_fast")
        # Should not raise
        test_instance.setUp()


class TestFindRepoRoot:
    """Tests for find_repo_root utility function."""

    def test_find_repo_root_default(self):
        """Test find_repo_root with default parameters."""
        from quark.common.utils.testing_utils import find_repo_root

        repo_root = find_repo_root()
        assert repo_root is not None
        assert (repo_root / "quark").exists()
        assert (repo_root / "test").exists()
        assert (repo_root / "pyproject.toml").exists() or (repo_root / "setup.py").exists()

    def test_find_repo_root_custom_path(self):
        """Test find_repo_root with custom starting path."""
        from pathlib import Path

        from quark.common.utils.testing_utils import find_repo_root

        custom_start = Path(__file__).resolve()
        repo_root = find_repo_root(custom_start)

        assert repo_root is not None
        assert (repo_root / "quark").exists()

    def test_find_repo_root_string_path(self):
        """Test find_repo_root with string path."""
        from quark.common.utils.testing_utils import find_repo_root

        repo_root = find_repo_root(__file__)

        assert repo_root is not None
        assert (repo_root / "quark").exists()

    def test_find_repo_root_cwd_fallback(self, tmp_path, monkeypatch):
        """When the __file__ walk-up misses, the CWD walk-up resolves the root.

        Simulates a wheel install (no marker above the module) by pointing the
        __file__-derived start at a markerless tree, then running from a tree
        that does carry a marker.
        """
        from quark.common.utils import testing_utils

        markerless = tmp_path / "site-packages" / "quark" / "common"
        markerless.mkdir(parents=True)
        monkeypatch.setattr(testing_utils, "__file__", str(markerless / "testing_utils.py"))

        repo = tmp_path / "checkout"
        repo.mkdir()
        (repo / "pyproject.toml").write_text("")
        monkeypatch.chdir(repo)

        assert testing_utils.find_repo_root() == repo.resolve()


class TestAddRepoRootToSysPathUtil:
    """Tests for add_repo_root_to_sys_path function."""

    def test_add_repo_root_to_sys_path(self):
        """Test that add_repo_root_to_sys_path adds path to sys.path."""
        import sys

        from quark.common.utils.testing_utils import add_repo_root_to_sys_path, find_repo_root

        repo_root = find_repo_root()

        # Verify it's in sys.path (should be added by module-level or previous calls)
        assert str(repo_root) in sys.path

        # Call again to test idempotence
        initial_len = len(sys.path)
        add_repo_root_to_sys_path()

        # Should not add duplicates
        assert len(sys.path) == initial_len
        assert str(repo_root) in sys.path


# Unit tests for the stable-vs-legacy equivalence infra. Today the only
# caller is the ONNX pipeline tests, which don't run in the torch CI job,
# so the torch PR diff-coverage gate was failing at 23%. Covering the
# runner directly under test_for_common/ closes the gap in both jobs
# without depending on either subsystem's dispatch-handle context manager
# being loadable.


@contextmanager
def _noop_ctx():
    # Stand-in for a real subsystem dispatch-handle context manager; no
    # globals to swap in this test.
    yield


def test_assert_outputs_equivalent():
    # Walks every branch in one shot: list/tuple recursion, default-tolerance
    # ndarray (numpy's rtol=1e-7 fallthrough), explicit-tolerance ndarray,
    # explicit bit-exact, single-elem-list-vs-ndarray broadcast (sess.run
    # output vs bare-ndarray golden), scalar (degenerates to a 0-d ndarray
    # compare) - plus the matching mismatch for each.
    assert_outputs_equivalent([np.array([1.0])], [np.array([1.0])])
    assert_outputs_equivalent((1,), (1,))
    assert_outputs_equivalent([1], (1,))
    # Default tolerances fall through to numpy (rtol=1e-7); a 1e-9 drift
    # passes by default but fails when the caller pins atol=rtol=0.
    assert_outputs_equivalent(np.array([1.0]), np.array([1.0 + 1e-9]))
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        assert_outputs_equivalent(np.array([1.0]), np.array([1.0 + 1e-9]), atol=0, rtol=0)
    assert_outputs_equivalent(np.array([1.0]), np.array([1.0 + 1e-9]), atol=1e-6, rtol=0)
    assert_outputs_equivalent([np.array([1.0, 2.0])], np.array([[1.0, 2.0]]))
    assert_outputs_equivalent(7, 7)
    # bfloat16 / float8 tensors lack a native NumPy dtype; the helper
    # promotes them to fp32 (a bit-exact superset) before going through
    # ``np.testing.assert_allclose``.
    if is_torch_available():
        import torch

        bf16 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        assert_outputs_equivalent(bf16, bf16.clone())
        # bf16 -> fp32 promotion is exact, so the ndarrays compare equal.
        assert_outputs_equivalent(bf16, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
        with pytest.raises(AssertionError, match="Mismatch"):
            assert_outputs_equivalent(bf16, torch.tensor([1.0, 2.0, 4.0], dtype=torch.bfloat16))
    for lhs, rhs, msg in [
        ([1, 2], [1], "length mismatch"),
        # Shape-mismatch + scalar-mismatch + numeric-mismatch all funnel
        # through ``assert_allclose``, which raises ``AssertionError`` with
        # a message starting with ``\nNot equal to tolerance`` (numpy's
        # default phrasing). Match a substring common to all three.
        (np.zeros((2,)), np.zeros((3,)), "Not equal to tolerance"),
        (np.array([1.0]), np.array([2.0]), "Not equal to tolerance"),
        (7, 8, "Not equal to tolerance"),
    ]:
        with pytest.raises(AssertionError, match=msg):
            assert_outputs_equivalent(lhs, rhs)


def test_run_op_variants():
    # Exercises every path: both-variants happy return, stable-vs-legacy
    # divergence, the lenient single-variant default (stable_ctx=None
    # degrades to legacy-only on PyTorch < 2.10), and the both-None guard.
    # Also calls _reseed_rngs directly to cover its numpy/random branches.
    _reseed_rngs(0)

    def pipeline() -> np.ndarray:
        return np.array([1, 2, 3])

    # Both variants provided: equivalence asserted, stable output returned.
    out = run_op_variants(pipeline, legacy_ctx=_noop_ctx(), stable_ctx=_noop_ctx())
    np.testing.assert_array_equal(out, np.array([1, 2, 3]))

    counter = {"n": 0}

    def diverging() -> np.ndarray:
        counter["n"] += 1
        return np.array([counter["n"]])

    with pytest.raises(AssertionError, match="stable-ABI vs legacy pybind11"):
        run_op_variants(diverging, legacy_ctx=_noop_ctx(), stable_ctx=_noop_ctx())

    # Lenient default: stable_ctx omitted => legacy-only run, no equivalence
    # assertion, legacy output returned. This is what callers on PyTorch <
    # 2.10 rely on instead of the strict gate that used to live here.
    out = run_op_variants(pipeline, legacy_ctx=_noop_ctx())
    np.testing.assert_array_equal(out, np.array([1, 2, 3]))

    # Stable-only lane errors iff torch supports the stable ABI; patch the
    # capability probe to make this assertion torch-version-independent.
    with (
        patch("quark.common.utils.testing_utils.is_torch_available", return_value=True),
        patch("quark.common.utils.testing_utils.torch_supports_stable_abi", return_value=True),
        pytest.raises(RuntimeError, match="legacy_ctx is None"),
    ):
        run_op_variants(pipeline, stable_ctx=_noop_ctx())

    with patch("quark.common.utils.testing_utils.torch_supports_stable_abi", return_value=False):
        out = run_op_variants(pipeline, stable_ctx=_noop_ctx())
        np.testing.assert_array_equal(out, np.array([1, 2, 3]))

    # Both None is the only invalid case - the helper requires at least one
    # variant to actually run.
    with pytest.raises(ValueError, match="requires at least one of legacy_ctx / stable_ctx"):
        run_op_variants(pipeline, legacy_ctx=None, stable_ctx=None)
