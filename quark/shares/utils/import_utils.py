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
from typing import Tuple
from packaging import version

from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def _is_package_available(pkg_name: str) -> Tuple[bool, str]:  # pragma: no cover
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
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
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
_transformers_available, _ = _is_package_available("transformers")  # pragma: no cover
_matplotlib_available, _ = _is_package_available("matplotlib")  # pragma: no cover


def is_torch_available() -> bool:  # pragma: no cover
    return _torch_available


def is_torch_greater_or_equal_2_5() -> bool:
    return version.parse(_torch_version) >= version.parse("2.5")


def is_accelerate_available() -> bool:  # pragma: no cover
    return _accelerate_available


def is_transformers_available() -> bool:  # pragma: no cover
    return _transformers_available


def is_matplotlib_available() -> bool:  # pragma: no cover
    return _matplotlib_available
