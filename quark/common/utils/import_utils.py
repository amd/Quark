#
# Modifications copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import importlib.metadata
import importlib.util  # type: ignore[attr-defined]
from typing import Any

from packaging import version

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ["_get_tensor_constant_from_node"]


def _parse_version_safe(version_str: str) -> version.Version | None:
    """Return a parsed Version, or None if missing / invalid (e.g. 'N/A')."""
    if not version_str or version_str == "N/A":
        return None
    try:
        return version.parse(version_str)
    except Exception:
        return None


def _version_meets_min(current: str, minimum: str) -> bool:
    """Check if the current version meets the minimum required version.

    Args:
        current: The current version string to check.
        minimum: The minimum required version string.

    Returns:
        bool: True if current version meets or exceeds the minimum version, False otherwise.
             Returns False if the current version cannot be parsed.
    """
    parsed = _parse_version_safe(current)
    if parsed is None:
        return False
    return parsed >= version.parse(minimum)


def _is_package_available(pkg_name: str) -> tuple[bool, str]:  # pragma: no cover
    # This function is licensed under Apache 2.0, Copyright 2022 The HuggingFace Team. All rights reserved.
    # It is unmodified and comes from https://github.com/huggingface/transformers/blob/93352e81f5019abaa52f7bdc2e3284779e864367/src/transformers/utils/import_utils.py#L42.

    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Fallback method: packages that may be importable without metadata.
            if pkg_name in ["torch", "triton", "PIL", "aiter"]:
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    package_version = temp_version
                    package_exists = True
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
        logger.debug(f"Detected {pkg_name} version: {package_version}")

    return package_exists, package_version


_torch_available, _torch_version = _is_package_available("torch")  # pragma: no cover
_accelerate_available, _ = _is_package_available("accelerate")  # pragma: no cover
_transformers_available, _transformers_version = _is_package_available("transformers")  # pragma: no cover
_matplotlib_available, _ = _is_package_available("matplotlib")  # pragma: no cover
_safetensors_available, _ = _is_package_available("safetensors")  # pragma: no cover
_triton_available, _ = _is_package_available("triton")  # pragma: no cover
_gguf_available, _gguf_version = _is_package_available("gguf")  # pragma: no cover
_aiter_available, _aiter_version = _is_package_available("aiter")  # pragma: no cover


_pil_available, _pil_version = _is_package_available("PIL")  # pragma: no cover
_huggingface_hub_available, _huggingface_hub_version = _is_package_available("huggingface_hub")  # pragma: no cover
_datasets_available, _datasets_version = _is_package_available("datasets")  # pragma: no cover
_requests_available, _requests_version = _is_package_available("requests")  # pragma: no cover
_psutil_available, _psutil_version = _is_package_available("psutil")  # pragma: no cover
_is_vllm_available, _ = _is_package_available("vllm")  # pragma: no cover
_compressed_tensors_available, _compressed_tensors_version = _is_package_available(
    "compressed_tensors"
)  # pragma: no cover
_optimum_available, _ = _is_package_available("optimum")  # pragma: no cover
_diffusers_available, _diffusers_version = _is_package_available("diffusers")  # pragma: no cover
_torchao_available, _torchao_version = _is_package_available("torchao")  # pragma: no cover


def is_torch_available() -> bool:  # pragma: no cover
    return _torch_available


def is_torchao_available() -> bool:  # pragma: no cover
    return _torchao_available


def is_vllm_available() -> bool:  # pragma: no cover
    return _is_vllm_available


def is_torch_greater_or_equal_2_5() -> bool:
    return _version_meets_min(_torch_version, "2.5")


TORCH_HIGHER_OR_EQUAL_2_7 = _version_meets_min(_torch_version, "2.7")
TORCH_HIGHER_OR_EQUAL_2_5 = _version_meets_min(_torch_version, "2.5")


def is_torch_greater_or_equal_2_7() -> bool:
    return _version_meets_min(_torch_version, "2.7")


def is_transformers_version_higher_or_equal(target_version: str) -> bool:
    parsed = _parse_version_safe(_transformers_version)
    if parsed is None:
        return False
    return parsed >= version.parse(target_version)


def is_transformers_version_lower(target_version: str) -> bool:
    parsed = _parse_version_safe(_transformers_version)
    if parsed is None:
        return False
    return parsed < version.parse(target_version)


def is_accelerate_available() -> bool:  # pragma: no cover
    return _accelerate_available


def is_transformers_available() -> bool:  # pragma: no cover
    return _transformers_available


def is_matplotlib_available() -> bool:  # pragma: no cover
    return _matplotlib_available


def is_safetensors_available() -> bool:  # pragma: no cover
    return _safetensors_available


def is_triton_available() -> bool:  # pragma: no cover
    return _triton_available


def is_pil_available() -> bool:  # pragma: no cover
    return _pil_available


def is_huggingface_hub_available() -> bool:  # pragma: no cover
    return _huggingface_hub_available


def is_datasets_available() -> bool:  # pragma: no cover
    return _datasets_available


def is_requests_available() -> bool:  # pragma: no cover
    return _requests_available


def is_psutil_available() -> bool:  # pragma: no cover
    return _psutil_available


def is_optimum_available() -> bool:  # pragma: no cover
    return _optimum_available


def is_diffusers_available() -> bool:  # pragma: no cover
    return _diffusers_available


def is_package_lower_or_equal(package_name: str, target_version: str) -> bool:
    _, package_version = _is_package_available(package_name)

    if package_version != "N/A":
        return version.parse(package_version) <= version.parse(target_version)
    else:
        return False


def is_gguf_available_and_minimum_version(minimum_version: str = "0.10.0") -> bool:  # pragma: no cover
    return _gguf_available and _version_meets_min(_gguf_version, minimum_version)


def is_aiter_available() -> bool:  # pragma: no cover
    """
    Check if AMD Aiter is available for native FP8/MXFP4 inference.

    AMD Aiter provides optimized GEMM kernels for quantized inference on AMD GPUs.
    When available, it can be used for native inference mode in QuantLinear layers.

    Returns:
        True if Aiter is installed and importable, False otherwise.
    """
    return _aiter_available


def is_compressed_tensors_available() -> bool:  # pragma: no cover
    return _compressed_tensors_available


# TODO: Remove this wrapper and use torch.export.export directly once we drop torch<=2.10 support.
def export_for_training(
    mod: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
) -> Any:  # pragma: no cover
    import torch

    if is_package_lower_or_equal("torch", "2.10.99"):
        return torch.export.export_for_training(mod=mod, args=args, kwargs=kwargs, dynamic_shapes=dynamic_shapes)
    else:
        return torch.export.export(mod=mod, args=args, kwargs=kwargs, dynamic_shapes=dynamic_shapes)


class UnavailableObject:
    def __init__(self, package_name: str, message: str | None = None):
        # Use object.__setattr__ to bypass our custom __setattr__
        object.__setattr__(self, "_package_name", package_name)
        object.__setattr__(self, "_message", message)

    def _raise_error(self) -> None:
        if self._message:
            error_msg = self._message
        else:
            error_msg = f"'{self._package_name}' is not available. Please make sure it is installed."
        raise ImportError(error_msg)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Raise error when trying to instantiate"""
        self._raise_error()

    def __getattr__(self, name: Any) -> None:
        """Raise error when accessing any attribute/method"""
        self._raise_error()

    def __setattr__(self, name: Any, value: Any) -> None:
        """Raise error when setting any attribute"""
        self._raise_error()

    def __getitem__(self, key: Any) -> None:
        """Raise error when using subscript notation"""
        self._raise_error()


# torch.ao.quantization.pt2e was removed in torch==2.11 and migrated to torchao
# TODO: Remove this conditional import once we drop torch<=2.10 support.
if is_package_lower_or_equal("torch", "2.10.99"):  # pragma: no cover
    from torch.ao.quantization.pt2e.utils import _get_tensor_constant_from_node
elif is_torchao_available():  # pragma: no cover
    from torchao.quantization.pt2e.utils import _get_tensor_constant_from_node  # type: ignore[import-not-found]
else:  # pragma: no cover
    _get_tensor_constant_from_node = UnavailableObject("torchao")  # type: ignore[assignment]
