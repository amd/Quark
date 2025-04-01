#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import unittest
from typing import Any, Union, Optional
import tempfile
import shutil
import functools

from .import_utils import is_torch_available, is_accelerate_available

if is_torch_available():  # pragma: no cover
    # Set env var CUDA_VISIBLE_DEVICES="" to force cpu-mode
    import torch

    torch_device: Optional[Union[str, torch.device]] = None
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


def require_torch_gpu(test_case: Any) -> Any:  # pragma: no cover
    """Decorator marking a test that requires CUDA and PyTorch."""
    return unittest.skipUnless(
        isinstance(torch_device, torch.device) and torch_device.type == "cuda", "test requires CUDA")(test_case)


def require_accelerate(test_case: Any) -> Any:  # pragma: no cover
    """Decorator marking a test that requires Accelerate library."""
    return unittest.skipUnless(is_accelerate_available(), "test requires accelerate")(test_case)


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
