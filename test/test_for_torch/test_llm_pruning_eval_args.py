#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Regression test: eval_model() callers must define every args attribute it reads.

PR #2656 added ``args.use_ppl_eval_model`` / ``args.evaluation_dataset`` /
``args.use_mlperf_rouge`` reads to ``quark.contrib.llm_eval.evaluation.eval_model``
and updated the argparse of ``examples/torch/language_modeling/llm_ptq/quantize_quark.py``,
but missed ``examples/torch/language_modeling/llm_pruning/main.py``. As a result,
both documented llm_pruning recipes crashed at the evaluation stage with
``AttributeError: 'Namespace' object has no attribute 'use_ppl_eval_model'``
and there was no workaround (the flags were not recognized by the parser).

This test statically enforces the eval-arg contract: every ``args.<name>``
hard-read inside ``eval_model`` must be defined by the caller's argparse
(``add_argument`` dest) or assigned on ``args`` at runtime before the
``eval_model`` call. It fails if a future change adds a new ``args`` read to
``eval_model`` without updating the callers.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PY = REPO_ROOT / "quark" / "contrib" / "llm_eval" / "evaluation.py"

# Plain-argparse callers of eval_model(). (The llm_qat / rotation callers build
# their namespace with HuggingFace dataclass parsers, which this static check
# does not parse.)
CALLERS = {
    "llm_pruning/main.py": REPO_ROOT / "examples" / "torch" / "language_modeling" / "llm_pruning" / "main.py",
    "llm_ptq/quantize_quark.py": REPO_ROOT
    / "examples"
    / "torch"
    / "language_modeling"
    / "llm_ptq"
    / "quantize_quark.py",
}


def _eval_model_arg_reads() -> set[str]:
    """Attribute names hard-read as ``args.<name>`` inside eval_model().

    ``getattr(args, "<name>", default)`` reads are intentionally excluded:
    they are safe by construction.
    """
    tree = ast.parse(EVALUATION_PY.read_text())
    func = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "eval_model")
    return {
        node.attr
        for node in ast.walk(func)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args"
    }


def _argparse_dests(source: str) -> set[str]:
    """Namespace attributes populated by add_argument() calls in *source*."""
    dests = set()
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
        ):
            continue
        dest = None
        for kw in node.keywords:
            if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                dest = kw.value.value
        if dest is None and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.startswith("-"):
                dest = first.value.lstrip("-")
        if dest is not None:
            dests.add(dest.replace("-", "_"))
    return dests


def _runtime_arg_assigns(source: str) -> set[str]:
    """Attributes assigned on ``args`` at runtime (``args.<name> = ...``)."""
    assigned = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "args"
        ):
            assigned.add(node.targets[0].attr)
    return assigned


@pytest.mark.parametrize("caller", list(CALLERS.values()), ids=list(CALLERS.keys()))
def test_caller_defines_all_args_read_by_eval_model(caller: Path):
    source = caller.read_text()
    defined = _argparse_dests(source) | _runtime_arg_assigns(source)
    missing = _eval_model_arg_reads() - defined
    assert not missing, (
        f"{caller} does not define the args attributes read by eval_model(): {sorted(missing)}. "
        "Add them to the parser (or assign them before the eval_model() call), "
        "otherwise eval_model() raises AttributeError at runtime."
    )
