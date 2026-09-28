#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Hermetic tests for subset.py's lm-eval-facing helpers (no lm-eval installed).

Covers the pieces that normally touch lm-eval — ``_flatten_task_dict``,
``build_subset_inputs`` — plus ``_lm_eval_version`` and ``build_samples``'s error
paths. A tiny fake Task object (and a fake ``lm_eval.tasks`` module injected into
``sys.modules``) stands in for the real harness, so none of this requires lm-eval
to be importable.
"""

from __future__ import annotations

import importlib.metadata
import sys
import types
from typing import Any

import pytest

from quark.contrib.mince import subset as S


class _FakeConfig:
    def __init__(self, task: str | None) -> None:
        self.task = task


class _FakeTask:
    """Minimal stand-in for an lm-eval Task used by build_subset_inputs."""

    def __init__(self, docs: list[str], task_name: str | None = None) -> None:
        self._docs = docs
        self.config = _FakeConfig(task_name)
        self.fewshot_seed: int | None = None

    def set_fewshot_seed(self, seed: int) -> None:
        self.fewshot_seed = seed

    @property
    def eval_docs(self) -> list[str]:
        return self._docs

    def fewshot_context(self, doc: str, num_fewshot: int) -> str:
        return f"ctx[{num_fewshot}]:{doc}"

    def doc_to_target(self, doc: str) -> str:
        return f"tgt:{doc}"


# ---------------------------------------------------------------------------
# _lm_eval_version
# ---------------------------------------------------------------------------


def test_lm_eval_version_returns_found_version(monkeypatch: pytest.MonkeyPatch) -> None:
    # Independent of whether lm-eval is actually installed (it is not in the
    # contrib test env): stub the metadata lookup so the found-version path is
    # exercised deterministically.
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "9.9.9")
    assert S._lm_eval_version() == "9.9.9"


def test_lm_eval_version_falls_back_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", _raise)
    assert S._lm_eval_version() == "unknown"


# ---------------------------------------------------------------------------
# build_samples — single-file path and unregistered-benchmark error
# ---------------------------------------------------------------------------


def test_build_samples_single_file_flat_indices() -> None:
    artifact = {"benchmark": "gsm8k", "indices": [3, 7, 9], "items": []}
    assert S.build_samples(artifact) == {"gsm8k": [3, 7, 9]}


def test_build_samples_unregistered_benchmark_raises() -> None:
    artifact = {"benchmark": "not_a_real_benchmark", "indices": [0], "items": []}
    with pytest.raises(ValueError, match="no --samples mapping"):
        S.build_samples(artifact)


# ---------------------------------------------------------------------------
# _flatten_task_dict
# ---------------------------------------------------------------------------


def test_flatten_uses_config_task_and_recurses() -> None:
    law = _FakeTask(docs=[], task_name="mmlu_pro_law")
    bio = _FakeTask(docs=[], task_name="mmlu_pro_biology")
    # A group-nested dict, exactly the shape lm-eval's get_task_dict returns.
    nested = {"mmlu_pro": {"a": law, "b": bio}}
    flat = S._flatten_task_dict(nested)
    assert flat == {"mmlu_pro_law": law, "mmlu_pro_biology": bio}


def test_flatten_falls_back_to_key_when_no_config() -> None:
    obj = object()  # no .config attribute -> name defaults to the dict key
    flat = S._flatten_task_dict({"gsm8k": obj})
    assert flat == {"gsm8k": obj}


# ---------------------------------------------------------------------------
# build_subset_inputs — with a fake lm_eval.tasks module
# ---------------------------------------------------------------------------


def _install_fake_lm_eval(monkeypatch: pytest.MonkeyPatch, task_dict: dict[str, Any]) -> None:
    """Inject a fake ``lm_eval.tasks`` so the lazy import inside subset resolves."""
    tasks_mod = types.ModuleType("lm_eval.tasks")
    tasks_mod.TaskManager = lambda *a, **k: object()  # type: ignore[attr-defined]
    tasks_mod.get_task_dict = lambda names, tm: task_dict  # type: ignore[attr-defined]
    pkg = types.ModuleType("lm_eval")
    monkeypatch.setitem(sys.modules, "lm_eval", pkg)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", tasks_mod)


def test_build_subset_inputs_renders_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _FakeTask(docs=["d0", "d1", "d2"], task_name="gsm8k")
    _install_fake_lm_eval(monkeypatch, {"gsm8k": task})

    out = S.build_subset_inputs({"gsm8k": [0, 2]}, num_fewshot=5)

    assert list(out) == ["gsm8k"]
    assert out["gsm8k"] == [
        {"index": 0, "input": "ctx[5]:d0", "target": "tgt:d0"},
        {"index": 2, "input": "ctx[5]:d2", "target": "tgt:d2"},
    ]
    # fewshot seed was applied for reproducibility.
    assert task.fewshot_seed == 1234


def test_build_subset_inputs_missing_task_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    other = _FakeTask(docs=["d0"], task_name="other")
    _install_fake_lm_eval(monkeypatch, {"other": other})
    with pytest.raises(KeyError, match="no task named 'gsm8k'"):
        S.build_subset_inputs({"gsm8k": [0]})


def test_build_subset_inputs_index_out_of_range_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _FakeTask(docs=["d0", "d1"], task_name="gsm8k")
    _install_fake_lm_eval(monkeypatch, {"gsm8k": task})
    with pytest.raises(IndexError, match="out of range"):
        S.build_subset_inputs({"gsm8k": [5]})
