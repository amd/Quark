#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import functools

from typing import Any


def create_dir(dir_name: str) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def getattr_recursive(obj: Any, attr: str) -> Any:
    """
    Recursive ``getattr``. This is useful e.g. to get the attribute ``"model.layers.0.self_attn.k_proj.weight"`` from a Transformers model.

    :param Any obj: A class instance holding the attribute.
    :param str attr: The attribute that is to be retrieved, e.g. 'attribute1.attribute2'.
    """

    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def setattr_recursive(module: Any, name: str, value: Any) -> None:
    """
    Recursive ``setattr``. This is useful e.g. to set the attribute ``"model.layers.0.self_attn.k_proj.weight"`` from a Transformers model.
    """
    if "." not in name:
        setattr(module, name, value)
    else:
        name, rest = name.split(".", 1)
        setattr_recursive(getattr(module, name), rest, value)
