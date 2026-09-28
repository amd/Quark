#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for mince.data_loader: ID keying, canonical order, and guards."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from quark.contrib.mince.config import BenchmarkConfig
from quark.contrib.mince.data_loader import load_benchmark_items
from quark.contrib.mince.test.utils import Fixture


def test_ifeval_loads_keyed_by_doc_key(ifeval_data: Fixture) -> None:
    """IFEVAL items are keyed by ``doc.key`` and carry all four metrics per model."""
    config, model_paths = ifeval_data
    items = load_benchmark_items(config, model_paths)

    assert len(items) == 8
    # IFEval is keyed by doc.key (1000 + i), preserved in file order.
    assert [it.item_id for it in items] == [1000 + i for i in range(8)]
    assert items[0].stratum == "cat_0"
    for it in items:
        for model in model_paths:
            r = it.model_results[model]
            assert set(r) == {"prompt_strict", "prompt_loose", "inst_strict", "inst_loose"}
            assert len(r["inst_strict"]) == 2


def test_mmlu_keyed_by_subject_doc_id(mmlu_data: Fixture) -> None:
    """MMLU items are keyed by ``(subject, doc_id)`` in sorted-subject-file order."""
    config, model_paths = mmlu_data
    items = load_benchmark_items(config, model_paths)

    assert len(items) == 10
    # Canonical order = sorted subject files, then line order.
    assert items[0].item_id == ("abstract_algebra", 0)
    assert items[-1].item_id == ("world_religions", 4)
    assert all("acc" in it.model_results["modelA"] for it in items)


def test_gsm8k_applies_flexible_extract_filter(gsm8k_data: Fixture) -> None:
    """GSM8K yields one item per doc, dropping rows from the non-matching filter."""
    config, model_paths = gsm8k_data
    items = load_benchmark_items(config, model_paths)

    # 6 docs, two filter rows each -> only flexible-extract kept.
    assert len(items) == 6
    assert [it.item_id for it in items] == list(range(6))
    assert all("exact_match" in it.model_results["modelA"] for it in items)


def test_mmlu_pro_keyed_by_category_doc_id(mmlu_pro_data: Fixture) -> None:
    """MMLU-Pro items are keyed by ``(category, doc_id)`` in sorted-category order."""
    config, model_paths = mmlu_pro_data
    items = load_benchmark_items(config, model_paths)

    assert len(items) == 8
    assert items[0].item_id == ("biology", 0)  # sorted: biology before law
    assert items[-1].item_id == ("law", 3)


def test_commonsense_qa_keyed_by_doc_id(commonsense_qa_data: Fixture) -> None:
    """CommonsenseQA items are keyed by ``doc_id`` and carry ``acc`` for every model."""
    config, model_paths = commonsense_qa_data
    items = load_benchmark_items(config, model_paths)

    assert len(items) == 6
    assert [it.item_id for it in items] == list(range(6))
    assert all("acc" in it.model_results["modelA"] for it in items)
    assert all("acc" in it.model_results["modelB"] for it in items)


def test_piqa_keyed_by_doc_id(piqa_data: Fixture) -> None:
    """PIQA items are keyed by ``doc_id``, take text from ``goal``, and carry both accuracies."""
    config, model_paths = piqa_data
    items = load_benchmark_items(config, model_paths)

    assert len(items) == 6
    assert [it.item_id for it in items] == list(range(6))
    # Question text comes from doc.goal, not doc.question.
    assert items[0].text == "piqa goal0"
    for it in items:
        for model in model_paths:
            assert set(it.model_results[model]) == {"acc", "acc_norm"}


def test_item_count_mismatch_raises(gsm8k_data: Fixture) -> None:
    """A ``total_items`` that disagrees with the loaded count raises."""
    config, model_paths = gsm8k_data
    wrong = BenchmarkConfig(
        name="gsm8k",
        total_items=999,
        metric_names=["exact_match"],
        sample_glob="samples_gsm8k_*.jsonl",
        sample_filter="flexible-extract",
    )
    with pytest.raises(ValueError, match="loaded 6 items"):
        load_benchmark_items(wrong, model_paths)


def test_missing_model_result_raises(tmp_path: Path, gsm8k_data: Fixture) -> None:
    """A model missing results for any item trips the completeness guard."""
    config, model_paths = gsm8k_data
    # Add a second model whose file is missing one doc_id -> completeness guard.
    d = os.path.join(tmp_path, "gsm8k", "modelB")
    os.makedirs(d, exist_ok=True)
    rows = []
    for i in range(5):  # only 5 of 6 docs
        rows.append({"doc_id": i, "filter": "flexible-extract", "exact_match": 1.0, "doc": {"question": f"q{i}"}})
    with open(os.path.join(d, "samples_gsm8k_2026.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    model_paths = {**model_paths, "modelB": d}
    with pytest.raises(ValueError, match="no results for model 'modelB'"):
        load_benchmark_items(config, model_paths)


def test_empty_model_paths_raises(ifeval_data: Fixture) -> None:
    """An empty ``model_paths`` raises instead of returning an empty item list."""
    config, _ = ifeval_data
    with pytest.raises(ValueError, match="model_paths is empty"):
        load_benchmark_items(config, {})
