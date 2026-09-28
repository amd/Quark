#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Shared synthetic-data fixtures for the MINCE test suite.

These build tiny, deterministic lm-eval-style sample JSONLs on disk (in pytest's
tmp_path) so the algorithm can be exercised end-to-end with no model inference,
no GPU, and no dependency on the original paper-study data. Each fixture returns
a ``(config, model_paths)`` pair ready for ``load_benchmark_items``.

Non-pytest helpers (``write_jsonl``, ``ifeval_rows``) and the ``Fixture`` type
alias live in ``utils.py`` and are imported here.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from quark.contrib.mince.config import BenchmarkConfig
from quark.contrib.mince.test.utils import Fixture, ifeval_rows, write_jsonl


@pytest.fixture(autouse=True)
def _cleanup_tmp_path(tmp_path: Path) -> Iterator[None]:
    """Guarantee no sample files are left behind after each test.

    All fixtures below write their synthetic JSONLs under pytest's per-test
    ``tmp_path``. pytest's default retention keeps the last few sessions on disk,
    so we explicitly remove this test's ``tmp_path`` on teardown — pass, fail, or
    error — leaving nothing behind once the run completes.
    """
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# IFEVAL — single JSONL per model, keyed by doc.key, 4 metrics
# ---------------------------------------------------------------------------


@pytest.fixture
def ifeval_data(tmp_path: Path) -> Fixture:
    """Provide an 8-item synthetic IFEVAL benchmark spanning two models.

    Writes one sample JSONL per model, each row carrying all four
    instruction-following metrics, keyed by ``doc.key``.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A ``(config, model_paths)`` pair ready for ``load_benchmark_items``.
    """
    config = BenchmarkConfig(
        name="ifeval",
        total_items=8,
        metric_names=["prompt_strict", "prompt_loose", "inst_strict", "inst_loose"],
        sample_glob="samples_ifeval_*.jsonl",
        candidate_ns=[2, 3, 4, 5, 6],
    )
    model_paths = {}
    for mi, model in enumerate(["modelA", "modelB"]):
        d = os.path.join(tmp_path, "ifeval", model)
        write_jsonl(os.path.join(d, "samples_ifeval_2026-01-01T00-00-00.jsonl"), ifeval_rows(seed=mi))
        model_paths[model] = d
    return config, model_paths


# ---------------------------------------------------------------------------
# MMLU — one JSONL per subject, keyed by (subject, doc_id), single acc
# ---------------------------------------------------------------------------


@pytest.fixture
def mmlu_data(tmp_path: Path) -> Fixture:
    """Provide a 10-item synthetic MMLU benchmark spanning two subjects.

    Writes one sample JSONL per subject for a single model, with ``doc_id``
    restarting at zero in each file so the ``(subject, doc_id)`` keying is
    actually exercised.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A ``(config, model_paths)`` pair ready for ``load_benchmark_items``.
    """
    subjects = {"abstract_algebra": 5, "world_religions": 5}
    config = BenchmarkConfig(
        name="mmlu",
        total_items=sum(subjects.values()),
        metric_names=["acc"],
        sample_glob="samples_mmlu_*_*.jsonl",
        candidate_ns=[3, 5, 7],
    )
    model_paths = {}
    for mi, model in enumerate(["modelA"]):
        d = os.path.join(tmp_path, "mmlu", model)
        for subject, count in subjects.items():
            rows = [
                {
                    "doc_id": j,
                    "doc": {"question": f"{subject} q{j}", "subject": subject},
                    "acc": float((j + mi) % 2),
                }
                for j in range(count)
            ]
            write_jsonl(os.path.join(d, f"samples_mmlu_{subject}_2026-01-01T00-00-00.jsonl"), rows)
        model_paths[model] = d
    return config, model_paths


# ---------------------------------------------------------------------------
# GSM8K — single JSONL, two filter rows per doc_id, keyed by doc_id
# ---------------------------------------------------------------------------


@pytest.fixture
def gsm8k_data(tmp_path: Path) -> Fixture:
    """Provide a 6-item synthetic GSM8K benchmark with two filter rows per item.

    Each item gets both a ``strict-match`` and a ``flexible-extract`` row, so the
    fixture verifies that the loader keeps only rows matching
    ``config.sample_filter``.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A ``(config, model_paths)`` pair ready for ``load_benchmark_items``.
    """
    config = BenchmarkConfig(
        name="gsm8k",
        total_items=6,
        metric_names=["exact_match"],
        sample_glob="samples_gsm8k_*.jsonl",
        candidate_ns=[2, 3, 4],
        sample_filter="flexible-extract",
    )
    model_paths = {}
    for mi, model in enumerate(["modelA"]):
        d = os.path.join(tmp_path, "gsm8k", model)
        rows = []
        for i in range(6):
            base = {"doc_id": i, "doc": {"question": f"gsm q{i}"}}
            # strict-match row (should be ignored by the flexible-extract filter)
            rows.append({**base, "filter": "strict-match", "exact_match": 0.0})
            rows.append({**base, "filter": "flexible-extract", "exact_match": float((i + mi) % 2)})
        write_jsonl(os.path.join(d, "samples_gsm8k_2026-01-01T00-00-00.jsonl"), rows)
        model_paths[model] = d
    return config, model_paths


# ---------------------------------------------------------------------------
# CommonsenseQA — single JSONL per model, keyed by doc_id, single acc
# ---------------------------------------------------------------------------


@pytest.fixture
def commonsense_qa_data(tmp_path: Path) -> Fixture:
    """Provide a 6-item synthetic CommonsenseQA benchmark spanning two models.

    Writes one sample JSONL per model, keyed by ``doc_id`` with a single ``acc``
    metric and the five-way ``choices`` structure lm-eval logs.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A ``(config, model_paths)`` pair ready for ``load_benchmark_items``.
    """
    config = BenchmarkConfig(
        name="commonsense_qa",
        total_items=6,
        metric_names=["acc"],
        sample_glob="samples_commonsense_qa_*.jsonl",
        candidate_ns=[2, 3, 4],
    )
    model_paths = {}
    for mi, model in enumerate(["modelA", "modelB"]):
        d = os.path.join(tmp_path, "commonsense_qa", model)
        rows = [
            {
                "doc_id": i,
                "doc": {
                    "id": f"csqa{i}",
                    "question": f"commonsense q{i}",
                    "question_concept": f"concept{i}",
                    "choices": {"label": ["A", "B", "C", "D", "E"], "text": [f"c{i}{c}" for c in range(5)]},
                    "answerKey": "A",
                },
                "filter": "none",
                "acc": float((i + mi) % 2),
            }
            for i in range(6)
        ]
        write_jsonl(os.path.join(d, "samples_commonsense_qa_2026-01-01T00-00-00.jsonl"), rows)
        model_paths[model] = d
    return config, model_paths


# ---------------------------------------------------------------------------
# PIQA — single JSONL per model, keyed by doc_id, acc + acc_norm
# ---------------------------------------------------------------------------


@pytest.fixture
def piqa_data(tmp_path: Path) -> Fixture:
    """Provide a 6-item synthetic PIQA benchmark spanning two models.

    Writes one sample JSONL per model, keyed by ``doc_id``, carrying both ``acc``
    and ``acc_norm`` and using ``goal``/``sol1``/``sol2`` for the item body.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A ``(config, model_paths)`` pair ready for ``load_benchmark_items``.
    """
    config = BenchmarkConfig(
        name="piqa",
        total_items=6,
        metric_names=["acc", "acc_norm"],
        sample_glob="samples_piqa_*.jsonl",
        candidate_ns=[2, 3, 4],
    )
    model_paths = {}
    for mi, model in enumerate(["modelA", "modelB"]):
        d = os.path.join(tmp_path, "piqa", model)
        rows = [
            {
                "doc_id": i,
                "doc": {
                    "goal": f"piqa goal{i}",
                    "sol1": f"sol1 for {i}",
                    "sol2": f"sol2 for {i}",
                    "label": i % 2,
                },
                # PIQA declares no filter_list, so lm-eval writes the literal "none".
                "filter": "none",
                "acc": float((i + mi) % 2),
                "acc_norm": float((i + mi + 1) % 2),
            }
            for i in range(6)
        ]
        write_jsonl(os.path.join(d, "samples_piqa_2026-01-01T00-00-00.jsonl"), rows)
        model_paths[model] = d
    return config, model_paths


# ---------------------------------------------------------------------------
# MMLU-Pro — one JSONL per category, keyed by (category, doc_id), exact_match
# ---------------------------------------------------------------------------


@pytest.fixture
def mmlu_pro_data(tmp_path: Path) -> Fixture:
    """Provide an 8-item synthetic MMLU-Pro benchmark spanning two categories.

    Writes one sample JSONL per category for a single model, with ``doc_id``
    restarting at zero in each file so the ``(category, doc_id)`` keying is
    actually exercised.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A ``(config, model_paths)`` pair ready for ``load_benchmark_items``.
    """
    categories = {"law": 4, "biology": 4}
    config = BenchmarkConfig(
        name="mmlu_pro",
        total_items=sum(categories.values()),
        metric_names=["exact_match"],
        sample_glob="samples_mmlu_pro_*.jsonl",
        candidate_ns=[3, 5, 7],
    )
    model_paths = {}
    for mi, model in enumerate(["modelA"]):
        d = os.path.join(tmp_path, "mmlu_pro", model)
        for category, count in categories.items():
            rows = [
                {
                    "doc_id": j,
                    "doc": {"question": f"{category} q{j}", "category": category},
                    "exact_match": float((j + mi) % 2),
                }
                for j in range(count)
            ]
            write_jsonl(os.path.join(d, f"samples_mmlu_pro_{category}_2026-01-01T00-00-00.jsonl"), rows)
        model_paths[model] = d
    return config, model_paths
