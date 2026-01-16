#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import functools
import os
from typing import Any

import torch.nn as nn


def get_op_name(root_module: nn.Module, op: nn.Module) -> str:
    # get the name of the op relative to the module
    for name, submodule in root_module.named_modules():
        if submodule is op:
            return name  # type: ignore
    raise ValueError(f"Cannot find op {op} in module {root_module}")


def resolve_star(submodule_names: list[str] | str, model: nn.Module) -> list[str]:
    """
    Resolves e.g. ``["model.layers.*.mlp.experts.*.gate_proj"]``

    to

    [
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.1.gate_proj",
        "model.layers.1.mlp.experts.0.gate_proj",
        "model.layers.1.mlp.experts.1.gate_proj",
        ...,
        "model.layers.30.mlp.experts.0.gate_proj",
        "model.layers.30.mlp.experts.1.gate_proj",
        "model.layers.31.mlp.experts.0.gate_proj",
        "model.layers.31.mlp.experts.1.gate_proj",
    ].

    We expect the parent of the star to be resolved to be an nn.ModuleList.

    This function also supports patterns as ``"model.layers.0.mlp.experts.*"``.
    """
    resolved_submodules = []

    if isinstance(submodule_names, str):
        submodule_names = [submodule_names]

    for submodule_name in submodule_names:
        # Split by '.*.' to process each star recursively.
        parts = submodule_name.split(".*.")

        # Support ending `.*` star.
        parts = parts[:-1] + parts[-1].split(".*")

        if len(parts) > 1:
            remaining = ".*.".join(parts[1:])
            parent_path = parts[0]

            parent_module = getattr_recursive(model, parent_path)

            if isinstance(parent_module, nn.ModuleList):
                length = len(parent_module)
            else:
                length = 0
                # See the comment below.
                while True:
                    if hasattr(parent_module, str(length)):
                        length += 1
                    else:
                        break

            for i in range(length):
                if isinstance(parent_module, nn.ModuleList):
                    submodule = parent_module[i]
                else:
                    # This is of course very weird, but this is unfortunately the approach taken in
                    # quark/experimental/cli/torch_llm/module_replacement/replacement_utils.py for gpt_oss.
                    submodule = getattr(parent_module, str(i))

                subresolved = resolve_star([remaining], submodule)

                for relative_path in subresolved:
                    if relative_path == "":
                        resolved = parent_path + f".{i}" + relative_path
                    else:
                        resolved = parent_path + f".{i}." + relative_path
                    resolved_submodules.append(resolved)
        else:
            resolved_submodules.extend(parts)

    return resolved_submodules


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
