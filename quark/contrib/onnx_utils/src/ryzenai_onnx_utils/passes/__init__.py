# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import glob
import importlib
import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from os.path import basename, dirname, isfile, join
from typing import TypeAlias

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.typing import PassFunction, PassOutputArgs, PatternType, SubPass


def check_replacement_signature(replacement: PassFunction) -> None:
    if not callable(replacement):
        raise TypeError(f"replacement function should be a function, now is {type(replacement)}")
    signature = inspect.signature(replacement)
    if [param.annotation for param in signature.parameters.values()] != [
        onnx.utils.Extractor,
        str,
        list[onnx.NodeProto],
        ryzenai_onnx_utils.ReplaceParams,
    ]:
        raise TypeError(
            f"callable replacement '{signature.parameters.keys()}' has an invalid type. "
            f"Expect {[onnx.utils.Extractor, str, list[onnx.NodeProto], ryzenai_onnx_utils.ReplaceParams]}, "
            f"but got {[param.annotation for param in signature.parameters.values()]}."
        )


def check_pattern_signature(pattern: str) -> None:
    if not callable(pattern):
        assert isinstance(pattern, list | SubPass)
        if isinstance(pattern, list):
            assert all(isinstance(item, str) for item in pattern)
        else:
            assert all(isinstance(item, str) for item in pattern.pattern)
    else:
        signature = inspect.signature(pattern)
        if [param.annotation for param in signature.parameters.values()] != [
            onnx.utils.Extractor,
            str,
            ryzenai_onnx_utils.ReplaceParams,
        ]:
            raise TypeError(
                f"callable PATTERN '{signature.parameters.keys()}' has an invalid type. "
                f"Expected {[onnx.utils.Extractor, str, ryzenai_onnx_utils.ReplaceParams]}, "
                f"but got {[param.annotation for param in signature.parameters.values()]}."
            )


def pass_loader(base_file: str) -> tuple[list[PatternType], list[PassFunction], list[str]]:
    modules = glob.glob(join(dirname(base_file), "*.py"))

    all_files = [basename(f)[:-3] for f in modules if isfile(f) and not basename(f).startswith("_")]

    patterns = []
    replacements = []

    parts = (dirname(base_file)).split(os.sep)
    assert "passes" in parts, f"Unexpected pass path: {dirname(base_file)}."
    index = parts.index("passes")
    passes_path = ".".join(parts[index + 1 :])
    for module in all_files:
        current_pass = importlib.import_module(f"ryzenai_onnx_utils.passes.{passes_path}.{module}")
        pattern = current_pass.PATTERN
        replacement = current_pass.REPLACEMENT
        if isinstance(replacement, list):
            assert isinstance(pattern, list)
            assert len(replacement) == len(pattern)
        else:
            replacement = [replacement]
            pattern = [pattern]
        for p, r in zip(pattern, replacement, strict=True):
            if p:
                check_pattern_signature(p)
            check_replacement_signature(r)
            patterns.append(p)
            replacements.append(r)

    # sort patterns by length to replace the longer ones first
    patterns, replacements = (
        list(t)
        for t in zip(
            *sorted(
                zip(patterns, replacements, strict=True),
                key=lambda p: len(p[0]) if not callable(p[0]) else 0,
                reverse=True,
            ),
            strict=True,
        )
    )

    return patterns, replacements, all_files


def global_pass(
    func: Callable[[onnx.utils.Extractor, str, list[onnx.NodeProto], ryzenai_onnx_utils.ReplaceParams], None],
) -> PassFunction:
    def wrapper(
        extractor: onnx.utils.Extractor,
        pass_id: str,
        subgraph: list[onnx.NodeProto],
        params: ryzenai_onnx_utils.ReplaceParams,
    ) -> PassOutputArgs:
        assert not subgraph
        func(extractor, pass_id, subgraph, params)
        return None, [], None

    return wrapper
