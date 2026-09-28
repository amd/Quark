#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Load lm-eval-harness sample logs into a list of per-item results.

`load_benchmark_items` reads each model's sample JSONL(s) for one benchmark and
returns a list of BenchmarkItem in a fixed order, where each item carries its
per-model scores in ``item.model_results[model_name]``.

Each benchmark identifies its items differently:
  - IFEVAL:    single JSONL, id = doc.key, 4 metrics with instruction-level lists
  - MMLU:      one JSONL per subject, id = (subject, doc_id), single acc
  - GSM8K:     single JSONL (2 rows/item), id = doc_id, exact_match metric
  - MMLU-Pro:  one JSONL per category, id = (category, doc_id), exact_match
  - CommonsenseQA: single JSONL, id = doc_id, single acc
  - PIQA:      single JSONL, id = doc_id, acc + acc_norm
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any

from quark.contrib.mince.config import BenchmarkConfig


@dataclass
class BenchmarkItem:
    """One benchmark item with an ID, stratum label, and per-model results."""

    item_id: Any
    # Category label logged in the frozen artifact when the benchmark has natural
    # categories: ifeval instruction type, mmlu subject, mmlu_pro category. Set it
    # only if yours does. Never used for sampling — draws are uniform.
    stratum: str = ""
    text: str = ""
    model_results: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_benchmark_items(
    benchmark: BenchmarkConfig,
    model_paths: dict[str, str],
) -> list[BenchmarkItem]:
    """Load per-sample results for one benchmark across multiple models.

    Args:
        benchmark: The benchmark configuration.
        model_paths: Maps model_name -> directory containing sample JSONL(s).

    Returns:
        List of BenchmarkItem in canonical order, each carrying per-model
        metric values in ``item.model_results[model_name]``.
    """
    if not model_paths:
        raise ValueError("model_paths is empty — at least one model required")

    loader = _LOADERS[benchmark.name]
    items = loader(benchmark, model_paths)

    if benchmark.total_items and len(items) != benchmark.total_items:
        raise ValueError(f"{benchmark.name}: loaded {len(items)} items but config expects {benchmark.total_items}")

    model_names = list(model_paths.keys())
    for i, item in enumerate(items):
        for model_name in model_names:
            if model_name not in item.model_results:
                raise ValueError(
                    f"{benchmark.name}: item {item.item_id!r} (index {i}) has no results for model '{model_name}'"
                )

    return items


# ---------------------------------------------------------------------------
# IFEVAL
# ---------------------------------------------------------------------------


def _load_ifeval(
    benchmark: BenchmarkConfig,
    model_paths: dict[str, str],
) -> list[BenchmarkItem]:
    """Load IFEVAL items from one sample JSONL per model.

    Items are keyed by ``doc["doc"]["key"]`` and carry all four
    instruction-following metrics. The stratum is the first entry of
    ``instruction_id_list``.

    Args:
        benchmark: The IFEVAL benchmark configuration.
        model_paths: Maps model_name -> directory containing the sample JSONL.

    Returns:
        List of BenchmarkItem in the first model's file order, each carrying
        per-model metrics in ``item.model_results[model_name]``.
    """
    first_model = next(iter(model_paths))
    first_path = _find_sample_file(model_paths[first_model], benchmark.sample_glob)

    canonical_keys: list[int] = []
    items_by_key: dict[int, BenchmarkItem] = {}

    with open(first_path) as f:
        for line in f:
            doc = json.loads(line)
            key = doc["doc"]["key"]
            if key in items_by_key:
                raise ValueError(f"{benchmark.name}: duplicate canonical id {key!r} in {first_path}")
            canonical_keys.append(key)
            stratum = doc["doc"]["instruction_id_list"][0]
            items_by_key[key] = BenchmarkItem(
                item_id=key,
                stratum=stratum,
                text=doc["doc"]["prompt"],
            )

    for model_name, model_dir in model_paths.items():
        path = _find_sample_file(model_dir, benchmark.sample_glob)
        with open(path) as f:
            for line in f:
                doc = json.loads(line)
                key = doc["doc"]["key"]
                items_by_key[key].model_results[model_name] = {
                    "prompt_strict": doc["prompt_level_strict_acc"],
                    "prompt_loose": doc["prompt_level_loose_acc"],
                    "inst_strict": doc["inst_level_strict_acc"],
                    "inst_loose": doc["inst_level_loose_acc"],
                }

    return [items_by_key[k] for k in canonical_keys]


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------


def _load_mmlu(
    benchmark: BenchmarkConfig,
    model_paths: dict[str, str],
) -> list[BenchmarkItem]:
    """Load MMLU items from one sample JSONL per subject, per model.

    Items are keyed by ``(subject, doc_id)`` because ``doc_id`` restarts at zero
    in every subject file. The subject, parsed from the filename, doubles as the
    stratum. Single ``acc`` metric.

    Args:
        benchmark: The MMLU benchmark configuration.
        model_paths: Maps model_name -> directory containing the per-subject JSONLs.

    Returns:
        List of BenchmarkItem ordered by sorted filename then row, each carrying
        per-model metrics in ``item.model_results[model_name]``.
    """
    first_model = next(iter(model_paths))
    first_dir = model_paths[first_model]
    sample_files = sorted(glob.glob(os.path.join(first_dir, benchmark.sample_glob)))
    if not sample_files:
        raise FileNotFoundError(f"No MMLU sample files found in {first_dir} matching {benchmark.sample_glob}")

    canonical_ids: list[tuple[str, int]] = []
    items_by_id: dict[tuple[str, int], BenchmarkItem] = {}

    for fpath in sample_files:
        subject = _extract_subject(os.path.basename(fpath), "samples_mmlu_")
        with open(fpath) as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc["doc_id"]
                uid = (subject, doc_id)
                if uid in items_by_id:
                    raise ValueError(f"{benchmark.name}: duplicate canonical id {uid!r} in {fpath}")
                canonical_ids.append(uid)
                items_by_id[uid] = BenchmarkItem(
                    item_id=uid,
                    stratum=subject,
                    text=doc["doc"]["question"],
                )

    for model_name, model_dir in model_paths.items():
        model_files = sorted(glob.glob(os.path.join(model_dir, benchmark.sample_glob)))
        for fpath in model_files:
            subject = _extract_subject(os.path.basename(fpath), "samples_mmlu_")
            with open(fpath) as f:
                for line in f:
                    doc = json.loads(line)
                    uid = (subject, doc["doc_id"])
                    if uid in items_by_id:
                        items_by_id[uid].model_results[model_name] = {
                            "acc": doc["acc"],
                        }

    return [items_by_id[uid] for uid in canonical_ids]


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------


def _load_gsm8k(
    benchmark: BenchmarkConfig,
    model_paths: dict[str, str],
) -> list[BenchmarkItem]:
    """Load GSM8K items from one sample JSONL per model.

    lm-eval logs one row per filter, so GSM8K writes two rows per item; rows are
    kept only when they match ``benchmark.sample_filter``. Items are keyed by
    ``doc_id`` with a single ``exact_match`` metric.

    Args:
        benchmark: The GSM8K benchmark configuration, including ``sample_filter``.
        model_paths: Maps model_name -> directory containing the sample JSONL.

    Returns:
        List of BenchmarkItem in the first model's file order, each carrying
        per-model metrics in ``item.model_results[model_name]``.
    """
    first_model = next(iter(model_paths))
    first_path = _find_sample_file(model_paths[first_model], benchmark.sample_glob)

    sample_filter = benchmark.sample_filter
    canonical_ids: list[int] = []
    items_by_id: dict[int, BenchmarkItem] = {}

    with open(first_path) as f:
        for line in f:
            doc = json.loads(line)
            if sample_filter and doc.get("filter") != sample_filter:
                continue
            doc_id = doc["doc_id"]
            if doc_id in items_by_id:
                raise ValueError(f"{benchmark.name}: duplicate canonical id {doc_id!r} in {first_path}")
            canonical_ids.append(doc_id)
            items_by_id[doc_id] = BenchmarkItem(
                item_id=doc_id,
                text=doc["doc"]["question"],
            )

    for model_name, model_dir in model_paths.items():
        path = _find_sample_file(model_dir, benchmark.sample_glob)
        with open(path) as f:
            for line in f:
                doc = json.loads(line)
                if sample_filter and doc.get("filter") != sample_filter:
                    continue
                doc_id = doc["doc_id"]
                items_by_id[doc_id].model_results[model_name] = {
                    "exact_match": doc.get("exact_match", doc.get("acc", 0.0)),
                }

    return [items_by_id[did] for did in canonical_ids]


# ---------------------------------------------------------------------------
# CommonsenseQA
# ---------------------------------------------------------------------------


def _load_commonsense_qa(
    benchmark: BenchmarkConfig,
    model_paths: dict[str, str],
) -> list[BenchmarkItem]:
    """Load CommonsenseQA items from one sample JSONL per model.

    Items are keyed by ``doc_id`` and carry a single ``acc`` metric. Item text is
    the multiple-choice ``question``. CommonsenseQA declares no filters, so every
    row is one item and no filtering is needed.

    Args:
        benchmark: The CommonsenseQA benchmark configuration.
        model_paths: Maps model_name -> directory containing the sample JSONL.

    Returns:
        List of BenchmarkItem in the first model's file order, each carrying
        per-model metrics in ``item.model_results[model_name]``.
    """
    first_model = next(iter(model_paths))
    first_path = _find_sample_file(model_paths[first_model], benchmark.sample_glob)

    canonical_ids: list[int] = []
    items_by_id: dict[int, BenchmarkItem] = {}

    with open(first_path) as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            if doc_id in items_by_id:
                raise ValueError(f"{benchmark.name}: duplicate canonical id {doc_id!r} in {first_path}")
            canonical_ids.append(doc_id)
            items_by_id[doc_id] = BenchmarkItem(
                item_id=doc_id,
                text=doc["doc"]["question"],
            )

    for model_name, model_dir in model_paths.items():
        path = _find_sample_file(model_dir, benchmark.sample_glob)
        with open(path) as f:
            for line in f:
                doc = json.loads(line)
                items_by_id[doc["doc_id"]].model_results[model_name] = {
                    "acc": doc["acc"],
                }

    return [items_by_id[did] for did in canonical_ids]


# ---------------------------------------------------------------------------
# PIQA
# ---------------------------------------------------------------------------


def _load_piqa(
    benchmark: BenchmarkConfig,
    model_paths: dict[str, str],
) -> list[BenchmarkItem]:
    """Load PIQA items from one sample JSONL per model.

    Items are keyed by ``doc_id`` and carry both ``acc`` and ``acc_norm``. Item
    text comes from the ``goal`` field rather than a question, since PIQA poses a
    goal plus two candidate solutions.

    Args:
        benchmark: The PIQA benchmark configuration.
        model_paths: Maps model_name -> directory containing the sample JSONL.

    Returns:
        List of BenchmarkItem in the first model's file order, each carrying
        per-model metrics in ``item.model_results[model_name]``.
    """
    first_model = next(iter(model_paths))
    first_path = _find_sample_file(model_paths[first_model], benchmark.sample_glob)

    sample_filter = benchmark.sample_filter
    canonical_ids: list[int] = []
    items_by_id: dict[int, BenchmarkItem] = {}

    with open(first_path) as f:
        for line in f:
            doc = json.loads(line)
            if sample_filter and doc.get("filter") != sample_filter:
                continue
            doc_id = doc["doc_id"]
            if doc_id in items_by_id:
                raise ValueError(f"{benchmark.name}: duplicate canonical id {doc_id!r} in {first_path}")
            canonical_ids.append(doc_id)
            items_by_id[doc_id] = BenchmarkItem(
                item_id=doc_id,
                text=doc["doc"]["goal"],
            )

    for model_name, model_dir in model_paths.items():
        path = _find_sample_file(model_dir, benchmark.sample_glob)
        with open(path) as f:
            for line in f:
                doc = json.loads(line)
                if sample_filter and doc.get("filter") != sample_filter:
                    continue
                items_by_id[doc["doc_id"]].model_results[model_name] = {
                    "acc": doc["acc"],
                    "acc_norm": doc["acc_norm"],
                }

    return [items_by_id[did] for did in canonical_ids]


# ---------------------------------------------------------------------------
# MMLU-Pro
# ---------------------------------------------------------------------------


def _load_mmlu_pro(
    benchmark: BenchmarkConfig,
    model_paths: dict[str, str],
) -> list[BenchmarkItem]:
    """Load MMLU-Pro items from one sample JSONL per category, per model.

    Items are keyed by ``(category, doc_id)`` because ``doc_id`` restarts at zero
    in every category file. The category, parsed from the filename, doubles as
    the stratum. Single ``exact_match`` metric.

    Args:
        benchmark: The MMLU-Pro benchmark configuration.
        model_paths: Maps model_name -> directory containing the per-category JSONLs.

    Returns:
        List of BenchmarkItem ordered by sorted filename then row, each carrying
        per-model metrics in ``item.model_results[model_name]``.
    """
    first_dir = model_paths[next(iter(model_paths))]
    sample_files = sorted(glob.glob(os.path.join(first_dir, benchmark.sample_glob)))
    if not sample_files:
        raise FileNotFoundError(f"No MMLU-Pro sample files in {first_dir} matching {benchmark.sample_glob}")

    canonical_ids: list[tuple[str, int]] = []
    items_by_id: dict[tuple[str, int], BenchmarkItem] = {}

    for fpath in sample_files:
        category = _extract_subject(os.path.basename(fpath), "samples_mmlu_pro_")
        with open(fpath) as f:
            for line in f:
                doc = json.loads(line)
                uid = (category, doc["doc_id"])
                if uid in items_by_id:
                    raise ValueError(f"{benchmark.name}: duplicate canonical id {uid!r} in {fpath}")
                canonical_ids.append(uid)
                items_by_id[uid] = BenchmarkItem(
                    item_id=uid,
                    stratum=category,
                    text=doc["doc"].get("question", ""),
                )

    for model_name, model_dir in model_paths.items():
        mfiles = sorted(glob.glob(os.path.join(model_dir, benchmark.sample_glob)))
        for fpath in mfiles:
            category = _extract_subject(os.path.basename(fpath), "samples_mmlu_pro_")
            with open(fpath) as f:
                for line in f:
                    doc = json.loads(line)
                    uid = (category, doc["doc_id"])
                    if uid in items_by_id:
                        items_by_id[uid].model_results[model_name] = {
                            "exact_match": float(doc.get("exact_match", doc.get("acc", 0.0))),
                        }

    return [items_by_id[uid] for uid in canonical_ids]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_subject(filename: str, prefix: str) -> str:
    """Extract the subject/category from a sample filename.

    Strips the ``prefix`` and a trailing ``_YYYY-MM-DDT...jsonl`` timestamp.

    Examples:
        samples_mmlu_abstract_algebra_2026-05-26T21-46-03.jsonl -> abstract_algebra
        samples_mmlu_pro_computer_science_2026-...jsonl         -> computer_science
    """
    name = filename
    if name.startswith(prefix):
        name = name[len(prefix) :]
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1][0:4].isdigit():
        name = parts[0]
    else:
        name = name.replace(".jsonl", "")
    return name


def _find_sample_file(directory: str, pattern: str) -> str:
    """Find a single sample JSONL in a directory matching a glob pattern."""
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        raise FileNotFoundError(f"No sample file matching '{pattern}' in {directory}")
    if len(matches) > 1:
        matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


_LOADERS = {
    "ifeval": _load_ifeval,
    "mmlu": _load_mmlu,
    "gsm8k": _load_gsm8k,
    "mmlu_pro": _load_mmlu_pro,
    "commonsense_qa": _load_commonsense_qa,
    "piqa": _load_piqa,
}
