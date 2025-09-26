#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import functools
import importlib.metadata
import os
import platform
import shutil
import sys
import tempfile
import unittest
from typing import Any, Optional, Union

from packaging import version

from .import_utils import is_accelerate_available, is_torch_available

if is_torch_available():  # pragma: no cover
    # Set env var CUDA_VISIBLE_DEVICES="" to force cpu-mode
    import torch

    torch_device: Union[str, torch.device] | None = None
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


def require_torch_cuda(test_case: Any) -> Any:  # pragma: no cover
    """Decorator marking a test that requires CUDA with at least two GPUs and PyTorch."""
    return unittest.skipUnless(
        isinstance(torch_device, torch.device) and torch_device.type == "cuda", "test requires CUDA"
    )(test_case)


def require_torch_multi_gpu(test_case: Any) -> Any:  # pragma: no cover
    """Decorator marking a test that requires CUDA and PyTorch."""
    return unittest.skipUnless(
        isinstance(torch_device, torch.device) and torch_device.type == "cuda" and torch.cuda.device_count() >= 2,
        "test requires CUDA multi-gpu",
    )(test_case)


def require_torch_hip(test_case: Any) -> Any:  # pragma: no cover
    """Decorator marking a test that requires HIP."""
    return unittest.skipUnless(torch.version.hip is not None, "test requires HIP")(test_case)


def require_accelerate(test_case: Any) -> Any:  # pragma: no cover
    """Decorator marking a test that requires Accelerate library."""
    return unittest.skipUnless(is_accelerate_available(), "test requires accelerate")(test_case)


def require_linux(test_case: Any) -> Any:  # pragma: no cover
    """Decorator marking a test that requires Linux."""
    return unittest.skipUnless(platform.system() == "Linux", "test requires Linux")(test_case)


def require_torch_higher_or_equal(min_version: str) -> Any:  # pragma: no cover
    """Decorator marking a test that requires the package `torch` with a version higher or equal than `version`."""

    def decorator(test_case: Any) -> Any:
        return unittest.skipUnless(
            version.parse(torch.__version__) >= version.parse(min_version), f"test requires torch>={min_version}"
        )(test_case)

    return decorator


def use_temporary_directory(func):  # type: ignore
    def wrapper(*args, **kwargs):  # type: ignore
        with tempfile.TemporaryDirectory() as tmpdir:
            result = func(*args, **kwargs, tmpdir=tmpdir)
        return result

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
        importlib.metadata.metadata("amd-quark") is not None
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


def slow(test_case):  # type: ignore[no-untyped-def]
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable to a truthy value to run them.

    """
    run_slow = os.environ.get("RUN_SLOW", "0") == "1"
    return unittest.skipUnless(run_slow, "test is slow")(test_case)
