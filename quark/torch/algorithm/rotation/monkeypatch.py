# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.
#
# This file is from https://github.com/spcl/QuaRot/blob/main/fake_quant/monkeypatch.py
#
# Modifications copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import copy
import functools
import types
from typing import Any, Callable

import torch.nn as nn


def copy_func_with_new_globals(f: Callable[..., Any], globals: dict[str, Any] | None = None) -> Callable[..., Any]:
    """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
    if globals is None:
        globals = f.__globals__
    g = types.FunctionType(
        f.__code__,
        globals,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    g = functools.update_wrapper(g, f)
    g.__module__ = f.__module__
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)  # type: ignore[attr-defined]
    return g


def add_wrapper_after_function_call_in_method(
    module: nn.Module,
    method_name: str,
    function_name: str,
    wrapper_fn: Callable[..., Any],
) -> Any:
    """
    This function adds a wrapper after the output of a function call in the method named `method_name`.
    Only calls directly in the method are affected. Calls by other functions called in the method are not affected.
    """

    original_method = getattr(module, method_name).__func__

    # The `method_name` method (e.g. forward) may be decorated.
    # See e.g. https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py#L223.
    # Here we retried the actual `module.method_name` to access later on its globals.
    if original_method.__closure__ is not None:
        for cell in original_method.__closure__:
            if isinstance(cell.cell_contents, types.FunctionType) and cell.cell_contents.__name__ == method_name:
                original_method = cell.cell_contents
                break
        else:
            raise ValueError(f"Method {method_name} not found in {module}. Please open an issue.")

    method_globals = dict(original_method.__globals__)
    wrapper = wrapper_fn(method_globals[function_name])
    method_globals[function_name] = wrapper
    new_method = copy_func_with_new_globals(original_method, globals=method_globals)
    setattr(module, method_name, new_method.__get__(module))
    return wrapper
