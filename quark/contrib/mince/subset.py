#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Build a reproducible frozen subset and its lm-eval ``--samples`` map.

Two related jobs:

  build_frozen_subset  draw a random size-n subset at a fixed seed from the
                       benchmark's canonical item order and return the artifact
                       dict (indices + enriched per-item ids + freeze-time drift
                       table). The draw depends only on (total_items, seed), so the
                       same subset is reproducible across models and runs.
  build_samples        translate that artifact into lm-eval's ``{task: [idx]}``
                       ``--samples`` map, and build_subset_inputs renders the actual
                       lm-eval prompts for those frozen positions.

Drift math lives in mince_metrics.py; the random-draw primitive in montecarlo.py.
"""

from __future__ import annotations

import importlib.metadata
from datetime import datetime
from typing import Any

import numpy as np

from quark.contrib.mince.config import BenchmarkConfig
from quark.contrib.mince.data_loader import BenchmarkItem, load_benchmark_items
from quark.contrib.mince.mince_metrics import compute_drift_table
from quark.contrib.mince.montecarlo import random_subset


def _lm_eval_version() -> str:
    """Best-effort lm-eval-harness version for artifact provenance.

    The frozen ``indices`` are positions into lm-eval's ``eval_docs``, whose
    ordering can change across lm-eval/task versions, so we record the version
    that produced them. ``freeze.py`` runs off sample logs and does not require
    lm-eval to be importable, so we read distribution metadata (no import) and
    fall back to "unknown" if the package is not installed.
    """
    for dist in ("lm_eval", "lm-eval", "lm-eval-harness"):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def build_item_id(benchmark_name: str, item: BenchmarkItem) -> dict[str, Any]:
    """Build a serializable identifier dict for one BenchmarkItem."""
    if benchmark_name == "ifeval":
        return {"key": item.item_id, "stratum": item.stratum, "prompt": item.text[:120]}
    if benchmark_name == "mmlu":
        subject, doc_id = item.item_id
        return {"subject": subject, "doc_id": doc_id, "question": item.text[:80]}
    if benchmark_name == "mmlu_pro":
        category, doc_id = item.item_id
        return {"category": category, "doc_id": doc_id, "question": item.text[:80]}
    if benchmark_name in ("gsm8k", "commonsense_qa", "piqa"):
        return {"doc_id": item.item_id, "question": item.text[:80]}
    return {"item_id": str(item.item_id)}


def build_frozen_subset(
    benchmark: BenchmarkConfig,
    model_paths: dict[str, str],
    n: int,
    seed: int = 42,
) -> tuple[dict[str, Any], list[BenchmarkItem]]:
    """Load items, draw the subset, and return the artifact dict (indices + items)."""
    items = load_benchmark_items(benchmark, model_paths)
    n_items = len(items)
    if not 1 <= n <= n_items:
        raise ValueError(f"n={n} out of range: must be 1 <= n <= total items ({n_items})")

    rng = np.random.default_rng(seed)
    indices = random_subset(n_items, n, rng, sort=True)

    enriched = []
    for idx in indices:
        info = build_item_id(benchmark.name, items[idx])
        info["index"] = idx
        enriched.append(info)

    return {
        "benchmark": benchmark.name,
        "models": list(model_paths),
        "n": n,
        "total_items": n_items,
        "reduction_pct": round((1 - n / n_items) * 100, 1),
        "seed": seed,
        "strategy": "random",
        "generated": datetime.now().isoformat(),
        "lm_eval_version": _lm_eval_version(),
        "indices": indices,
        "items": enriched,
        "drift": compute_drift_table(benchmark, items, indices, model_paths),
    }, items


# ---------------------------------------------------------------------------
# lm-eval --samples map (used by freeze.py to write subset_samples.json)
# ---------------------------------------------------------------------------

# Benchmarks whose frozen indices map 1:1 onto a single lm-eval task's doc order.
_SINGLE_FILE = {"gsm8k", "ifeval", "commonsense_qa", "piqa"}

# Multi-file task groups: lm-eval expands the group into per-subtask tasks named
# "<prefix><subject>" (e.g. mmlu_astronomy, mmlu_pro_biology). samples must be
# keyed by those subtask names, with indices being each item's within-subtask
# doc_id. The enriched artifact carries that doc_id plus the grouping field below.
_GROUP_SUBTASK = {
    "mmlu": ("mmlu_", "subject"),
    "mmlu_pro": ("mmlu_pro_", "category"),
}


def build_samples(artifact: dict[str, Any]) -> dict[str, list[int]]:
    """Translate a frozen artifact into lm-eval's ``--samples`` ``{task: [idx]}`` map.

    Each value is a list of positional indices into that task's ``eval_docs``
    (see lm-eval's ``Task.doc_iterator``); MINCE stores exactly those positions:

    - single-file tasks (gsm8k, ifeval, commonsense_qa, piqa): the flat ``indices`` list;
    - task groups (mmlu, mmlu_pro): each item's per-subtask ``doc_id``, grouped by
      subject/category into ``{subtask_name: [doc_id, ...]}``.

    Raises:
        ValueError: if the benchmark has no registered mapping (add it to
            ``_SINGLE_FILE`` or ``_GROUP_SUBTASK``).
    """
    benchmark = artifact["benchmark"]
    if benchmark in _SINGLE_FILE:
        return {benchmark: list(artifact["indices"])}

    if benchmark not in _GROUP_SUBTASK:
        raise ValueError(
            f"no --samples mapping for benchmark {benchmark!r}: add it to "
            f"_SINGLE_FILE or _GROUP_SUBTASK in quark/contrib/mince/subset.py"
        )

    prefix, group_key = _GROUP_SUBTASK[benchmark]
    samples: dict[str, list[int]] = {}
    for item in artifact["items"]:
        subtask = f"{prefix}{item[group_key]}"
        samples.setdefault(subtask, []).append(int(item["doc_id"]))
    return samples


def _flatten_task_dict(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Flatten lm-eval's (possibly group-nested) task dict to ``{name: Task}``."""
    flat: dict[str, Any] = {}
    for key, val in task_dict.items():
        if isinstance(val, dict):
            flat.update(_flatten_task_dict(val))
        else:
            name = getattr(getattr(val, "config", None), "task", key)
            flat[name] = val
    return flat


def build_subset_inputs(
    samples: dict[str, list[int]],
    num_fewshot: int = 0,
    fewshot_seed: int = 1234,
    task_manager: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    """Render the actual lm-eval inputs for a frozen subset.

    Takes a ``--samples`` map (``{task: [positional indices]}`` as written by
    freeze.py to ``subset_samples.json``) and, for each task, loads that task from
    lm-eval and renders the prompt for exactly the frozen positions. The indices
    are positions into each task's ``eval_docs`` — the same space lm-eval's
    ``--samples`` selects on — so they line up with the frozen subset.

    Args:
        samples: task -> list of positional indices (the ``subset_samples.json`` map).
        num_fewshot: few-shot examples to include in each rendered prompt. Match the
            value used for the bf16 run (e.g. 0 for gsm8k, 5 for mmlu); 0 renders
            just the question (plus any task description).
        fewshot_seed: seed for few-shot example sampling (lm-eval's default is
            1234), so the rendered context is reproducible.
        task_manager: optional pre-built lm-eval ``TaskManager``.

    Returns:
        ``{task: [{"index", "input", "target"}, ...]}``, preserving the order of the
        indices given. ``input`` is the rendered prompt string; ``target`` is the
        task's gold answer.

    Note:
        ``lm_eval`` is imported lazily — this is the only helper that needs it.
    """
    from lm_eval.tasks import TaskManager, get_task_dict

    tm = task_manager or TaskManager()
    tasks = _flatten_task_dict(get_task_dict(list(samples), tm))

    out: dict[str, list[dict[str, Any]]] = {}
    for task_name, indices in samples.items():
        task = tasks.get(task_name)
        if task is None:
            raise KeyError(f"lm-eval returned no task named {task_name!r} (got {sorted(tasks)})")
        task.set_fewshot_seed(fewshot_seed)
        docs = task.eval_docs
        n = len(docs)
        rows = []
        for i in indices:
            i = int(i)
            if not 0 <= i < n:
                raise IndexError(f"{task_name}: index {i} out of range for eval_docs (n={n})")
            doc = docs[i]
            rows.append(
                {
                    "index": i,
                    "input": task.fewshot_context(doc, num_fewshot),
                    "target": task.doc_to_target(doc),
                }
            )
        out[task_name] = rows
    return out
