#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Utilities related to saving files."""

import contextlib
from pathlib import Path
from typing import Any

from quark.common.utils.import_utils import UnavailableObject, is_transformers_available
from quark.common.utils.log import ScreenLogger

if is_transformers_available():
    from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoProcessor, AutoTokenizer
else:  # pragma: no cover
    AutoFeatureExtractor = UnavailableObject("transformers")  # type: ignore[assignment]
    AutoImageProcessor = UnavailableObject("transformers")  # type: ignore[assignment]
    AutoProcessor = UnavailableObject("transformers")  # type: ignore[assignment]
    AutoTokenizer = UnavailableObject("transformers")  # type: ignore[assignment]


logger = ScreenLogger(__name__)


def maybe_load_preprocessors(
    src_name_or_path: str | Path, subfolder: str = "", trust_remote_code: bool = False
) -> list[Any]:
    # Copyright 2022 The HuggingFace Team. All rights reserved.
    #
    # This function is licensed under the Apache License, Version 2.0 (the "License");
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
    """
    Attempt to load available preprocessors from a model directory or repository.

    :param src_name_or_path: The source directory or model identifier from which to load preprocessors.
    :type src_name_or_path: Union[str, Path]
    :param subfolder: Subfolder name where preprocessor files are located, defaults to "".
    :type subfolder: str
    :param trust_remote_code: Whether to allow loading preprocessors that can execute arbitrary code, defaults to False.
    :type trust_remote_code: bool
    :return: List of successfully loaded preprocessors (tokenizer, processor, feature extractor, or image processor).
    :rtype: List
    """
    preprocessors = []
    with contextlib.suppress(Exception):
        preprocessors.append(
            AutoTokenizer.from_pretrained(src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code)  # type: ignore[no-untyped-call]
        )

    with contextlib.suppress(Exception):
        preprocessors.append(
            AutoProcessor.from_pretrained(src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code)  # type: ignore[no-untyped-call]
        )

    with contextlib.suppress(Exception):
        preprocessors.append(
            AutoFeatureExtractor.from_pretrained(  # type: ignore[no-untyped-call]
                src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code
            )  # type: ignore[no-untyped-call]
        )

    with contextlib.suppress(Exception):
        preprocessors.append(
            AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code
            )
        )
    return preprocessors


def maybe_save_preprocessors(
    src_name_or_path: str | Path,
    dest_dir: str | Path,
    src_subfolder: str = "",
    trust_remote_code: bool = False,
) -> None:
    # Copyright 2022 The HuggingFace Team. All rights reserved.
    #
    # This function is licensed under the Apache License, Version 2.0 (the "License");
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
    """
    Save tokenizer, processor, and feature extractor when found in source to destination directory.

    :param src_name_or_path: The source directory or model identifier from which to load the preprocessors.
    :type src_name_or_path: Union[str, Path]
    :param dest_dir: The destination directory to save the preprocessors to.
    :type dest_dir: Union[str, Path]
    :param src_subfolder: Subfolder name where preprocessor files are located in the model directory or Hugging Face Hub repository, defaults to "".
    :type src_subfolder: str
    :param trust_remote_code: Whether to allow saving preprocessors that can execute arbitrary code, defaults to False.
    :type trust_remote_code: bool
    """
    if not isinstance(dest_dir, Path):
        dest_dir = Path(dest_dir)

    dest_dir.mkdir(exist_ok=True)
    for preprocessor in maybe_load_preprocessors(
        src_name_or_path, subfolder=src_subfolder, trust_remote_code=trust_remote_code
    ):
        preprocessor.save_pretrained(dest_dir)
