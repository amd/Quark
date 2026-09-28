#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Hermetic tests for subset.build_samples (no lm-eval / no model).

The live alignment between our stored positions and lm-eval's per-subtask
eval_docs is verified separately in the regression flow; here we only check that
the frozen artifact is translated into the correct ``samples={task: [idx]}``
shape for both single-file tasks and multi-file task groups.
"""

from __future__ import annotations

from quark.contrib.mince.subset import build_frozen_subset
from quark.contrib.mince.subset import build_samples as _build_samples
from quark.contrib.mince.test.utils import Fixture


def test_single_file_uses_flat_indices(gsm8k_data: Fixture) -> None:
    """A single-file task maps to the artifact's flat positions under its own name."""
    config, model_paths = gsm8k_data
    artifact, _ = build_frozen_subset(config, model_paths, n=4, seed=42)

    samples = _build_samples(artifact)

    assert set(samples) == {"gsm8k"}
    assert samples["gsm8k"] == artifact["indices"]
    assert len(samples["gsm8k"]) == 4


def test_ifeval_single_file(ifeval_data: Fixture) -> None:
    """IFEVAL is a single-file task, so samples mirror the artifact indices exactly."""
    config, model_paths = ifeval_data
    artifact, _ = build_frozen_subset(config, model_paths, n=5, seed=1)

    samples = _build_samples(artifact)

    assert set(samples) == {"ifeval"}
    assert samples["ifeval"] == artifact["indices"]


def test_commonsense_qa_single_file(commonsense_qa_data: Fixture) -> None:
    """CommonsenseQA maps to flat indices with ``doc_id``-enriched frozen items."""
    config, model_paths = commonsense_qa_data
    artifact, _ = build_frozen_subset(config, model_paths, n=3, seed=42)

    samples = _build_samples(artifact)

    assert set(samples) == {"commonsense_qa"}
    assert samples["commonsense_qa"] == artifact["indices"]
    # Enriched ids carry doc_id/question, not the stringified fallback.
    assert all("doc_id" in it for it in artifact["items"])


def test_piqa_single_file(piqa_data: Fixture) -> None:
    """PIQA maps to flat indices with ``doc_id``-enriched frozen items."""
    config, model_paths = piqa_data
    artifact, _ = build_frozen_subset(config, model_paths, n=3, seed=42)

    samples = _build_samples(artifact)

    assert set(samples) == {config.name}
    assert samples[config.name] == artifact["indices"]
    # Not the stringified fallback from build_item_id.
    assert all("doc_id" in it for it in artifact["items"])


def test_mmlu_groups_by_subject(mmlu_data: Fixture) -> None:
    """MMLU indices split across per-subject subtasks with no drops or duplicates."""
    config, model_paths = mmlu_data
    artifact, _ = build_frozen_subset(config, model_paths, n=7, seed=42)

    samples = _build_samples(artifact)

    # Keys are lm-eval subtask names, one per subject that appears in the subset.
    assert all(k.startswith("mmlu_") for k in samples)
    subjects = {it["subject"] for it in artifact["items"]}
    assert set(samples) == {f"mmlu_{s}" for s in subjects}

    # Every frozen item's doc_id lands under its subject's subtask, and the total
    # count is preserved (no drops, no dupes).
    assert sum(len(v) for v in samples.values()) == 7
    for it in artifact["items"]:
        assert it["doc_id"] in samples[f"mmlu_{it['subject']}"]


def test_mmlu_pro_groups_by_category(mmlu_pro_data: Fixture) -> None:
    """MMLU-Pro indices split across per-category lm-eval subtasks."""
    config, model_paths = mmlu_pro_data
    artifact, _ = build_frozen_subset(config, model_paths, n=5, seed=7)

    samples = _build_samples(artifact)

    assert all(k.startswith("mmlu_pro_") for k in samples)
    cats = {it["category"] for it in artifact["items"]}
    assert set(samples) == {f"mmlu_pro_{c}" for c in cats}
    assert sum(len(v) for v in samples.values()) == 5
    for it in artifact["items"]:
        assert it["doc_id"] in samples[f"mmlu_pro_{it['category']}"]
