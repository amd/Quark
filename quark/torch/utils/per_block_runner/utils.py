#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Utility helpers: recursive attribute access and auto-detection of decoder layer paths.
"""

import functools
from typing import Any

import torch.nn as nn


def getattr_recursive(obj: Any, attr: str) -> Any:
    """
    Recursive ``getattr``.

    :param Any obj: A class instance holding the attribute.
    :param str attr: Dotted attribute path, e.g. ``"model.layers.0.self_attn.k_proj.weight"``.
    """

    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def infer_decoder_layers_path(model: nn.Module) -> str:
    """
    Auto-detect the dotted path to the top-level ``nn.ModuleList`` that contains the
    decoder blocks (e.g. ``"model.layers"`` or ``"layers"``).

    The function enumerates all ``nn.ModuleList`` submodules, filters out nested ones
    (e.g. expert lists inside a block), and returns the shallowest remaining path.

    :param nn.Module model: Any PyTorch model.
    :return: Dotted path string, or ``""`` if no ``nn.ModuleList`` is found.
    :rtype: str
    """
    modulelist_paths = [name for name, module in model.named_modules() if isinstance(module, nn.ModuleList)]

    def _filter_root_paths(paths: list[str]) -> list[str]:
        sorted_paths = sorted({p for p in paths if p}, key=lambda s: (s.count("."), len(s), s))
        roots: list[str] = []
        for path in sorted_paths:
            if not any(path == r or path.startswith(r + ".") for r in roots):
                roots.append(path)
        return roots

    root_paths = _filter_root_paths(modulelist_paths)
    return root_paths[0] if root_paths else ""
