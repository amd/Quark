#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import pytest
import yaml


def _store_with_repair_experience(tmp_path):
    from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore

    store = ExperienceStore(tmp_path / "experience.sqlite")
    store.record_repair_experience(
        record_id="session-1:repair:1",
        source_session_id="session-1",
        context_fingerprint="fingerprint",
        framework="vllm",
        framework_version="abc",
        arch_fingerprint="arch",
        quant_signature="mlp=mxfp4",
        failure_mode="load_run",
        error_signature="v2|AttributeError|||none.to",
        outcome="fixed",
        verification_status="verified",
        payload={
            "approach_summary": "guard optional bias tensors",
            "changed_files": ["vllm/quark_moe.py"],
            "attempts": [],
            "verifier_results": [{"verifier": "load_inference", "passed": True}],
        },
    )
    return store


def test_review_service_builds_proposed_yaml_from_runtime_experience(
    tmp_path,
):
    from quark.experimental.torch.quant_perf.knowledge.review import KnowledgeReviewService

    store = _store_with_repair_experience(tmp_path)
    service = KnowledgeReviewService(store)

    rows = service.list_reviewable(
        session_id="session-1",
        domain="repair",
    )
    candidate = service.build_candidate(rows[0]["experience_key"])

    assert rows[0]["review_decision"] == ""
    assert candidate["status"] == "proposed"
    assert candidate["domain"] == "repair"
    assert candidate["kind"] == "verified_repair_recipe"
    assert "guard optional bias" in candidate["summary"]
    assert candidate["provenance"]["source_session_id"] == "session-1"
    assert candidate["match"]["error_signature"] == ["v2|AttributeError|||none.to"]
    store.close()


def test_review_approval_writes_curated_yaml_to_explicit_repo(tmp_path):
    from quark.experimental.torch.quant_perf.knowledge import CuratedKnowledgeProvider
    from quark.experimental.torch.quant_perf.knowledge.review import KnowledgeReviewService

    store = _store_with_repair_experience(tmp_path)
    service = KnowledgeReviewService(store)
    experience_key = service.list_reviewable()[0]["experience_key"]
    candidate = service.build_candidate(experience_key)
    candidate_path = tmp_path / "candidate.yaml"
    candidate_path.write_text(
        yaml.safe_dump(candidate, sort_keys=False),
        encoding="utf-8",
    )
    repo_root = tmp_path / "Quark"
    records_root = repo_root / "quark" / "experimental" / "torch" / "quant_perf" / "knowledge" / "records"
    records_root.mkdir(parents=True)

    installed = service.approve(
        candidate_path=candidate_path,
        repo_root=repo_root,
    )

    assert installed.is_file()
    assert installed.parent.name == "repair"
    records = CuratedKnowledgeProvider(records_root).items
    assert [record.id for record in records] == [candidate["id"]]
    assert records[0].status == "verified"
    store.close()


def test_rejected_experience_is_excluded_from_default_review_list(tmp_path):
    from quark.experimental.torch.quant_perf.knowledge.review import KnowledgeReviewService

    store = _store_with_repair_experience(tmp_path)
    service = KnowledgeReviewService(store)
    key = service.list_reviewable()[0]["experience_key"]

    service.reject(key, reason="too stack specific")

    assert service.list_reviewable() == []
    assert service.list_reviewable(include_reviewed=True)[0]["review_decision"] == "rejected"
    store.close()


def test_review_approval_reuses_curated_safety_validation(tmp_path):
    from quark.experimental.torch.quant_perf.knowledge.review import KnowledgeReviewService

    store = _store_with_repair_experience(tmp_path)
    service = KnowledgeReviewService(store)
    candidate = service.build_candidate(service.list_reviewable()[0]["experience_key"])
    candidate["guidance"] = ["Apply /tmp/unsafe.patch"]
    candidate_path = tmp_path / "candidate.yaml"
    candidate_path.write_text(
        yaml.safe_dump(candidate, sort_keys=False),
        encoding="utf-8",
    )
    repo_root = tmp_path / "Quark Quant-Perf"
    (repo_root / "quant_perf" / "knowledge" / "records").mkdir(parents=True)

    with pytest.raises(ValueError, match="executable patch"):
        service.approve(
            candidate_path=candidate_path,
            repo_root=repo_root,
        )
    store.close()
