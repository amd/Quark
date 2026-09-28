#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
import sqlite3

import pytest


def make_store(tmp_path):
    from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore

    return ExperienceStore(tmp_path / "experience.sqlite")


def test_signature_migration_preserves_reviews_and_recovers_legacy_evidence(tmp_path, monkeypatch):
    from quark.experimental.torch.quant_perf.knowledge import cli, migration
    from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore

    path = tmp_path / "experience.sqlite"
    store = make_store(tmp_path)
    for index in (1, 2):
        store.record_repair_experience(
            record_id=f"session-1:repair:{index}",
            source_session_id="session-1",
            context_fingerprint="fingerprint",
            framework="vllm",
            framework_version="abc",
            arch_fingerprint="arch",
            quant_signature="fp8",
            failure_mode="load_run",
            error_signature="v2|RuntimeError|linear.py|load_weights|shape [16,32]",
            outcome="fixed",
            verification_status="verified",
            payload={"approach_summary": "preserve scales"},
        )
    store.record_review_decision(experience_key="repair:session-1:repair:1", domain="repair", decision="approved")
    store.close()
    with sqlite3.connect(path) as db:
        for name in ("signature_version", "error_pattern"):
            if name in {row[1] for row in db.execute("PRAGMA table_info(repair_experience)")}:
                db.execute(f"ALTER TABLE repair_experience DROP COLUMN {name}")
        db.execute("UPDATE schema_metadata SET value='2' WHERE name='schema_version'")
        db.execute(
            "UPDATE repair_experience SET error_signature='RuntimeError|linear.py|load_weights|shape [16,32]' WHERE record_id='session-1:repair:1'"
        )
        db.execute(
            "UPDATE repair_experience SET error_signature='RuntimeError: shape [N,N]' WHERE record_id='session-1:repair:2'"
        )

    migrate = migration.migrate_repair_signatures

    def interrupted_migration(db):
        migrate(db)
        raise RuntimeError("migration interrupted")

    with monkeypatch.context() as context:
        context.setattr(migration, "migrate_repair_signatures", interrupted_migration)
        with pytest.raises(RuntimeError, match="migration interrupted"):
            ExperienceStore(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT value FROM schema_metadata WHERE name='schema_version'").fetchone()[0] == "2"
        assert "signature_version" not in {row[1] for row in db.execute("PRAGMA table_info(repair_experience)")}

    with ExperienceStore(path) as store:
        rows = store.list_reviewable_experience(include_reviewed=True)
        assert rows[0]["signature_version"] == 2
        assert rows[0]["error_signature"] == "v2|RuntimeError|linear.py|load_weights|shape [16,32]"
        assert rows[0]["review_decision"] == "approved"
        assert rows[1]["signature_version"] == 1
        assert rows[1]["verification_status"] == "verified"
    backups = list(tmp_path.glob("experience.sqlite.v2.*.bak"))
    assert len(backups) == 2
    with sqlite3.connect(backups[0]) as db:
        assert db.execute("SELECT value FROM schema_metadata WHERE name='schema_version'").fetchone()[0] == "2"

    session = tmp_path / "old-session"
    session.mkdir()
    log = session / "failure.log"
    log.write_text('File "/repos/vllm/linear.py", line 1, in load_weights\nRuntimeError: shape [16,32]')
    (session / "state.json").write_text(
        json.dumps(
            {
                "session_id": "session-1",
                "repair_journey": [{}, {"evidence_paths": [str(log)]}],
            }
        )
    )
    monkeypatch.setattr(cli.config, "experience_store_path", lambda: str(path))
    assert cli.main(["migrate", "--session", str(session)]) == 0
    assert cli.main(["migrate", "--session", str(session)]) == 0
    with ExperienceStore(path) as store:
        rows = store.list_reviewable_experience(include_reviewed=True)
        assert len(rows) == 2
        assert all(row["signature_version"] == 2 for row in rows)
        assert rows[0]["review_decision"] == "approved"
        assert all(row["observation_count"] == 1 for row in rows)


def test_quantization_experience_upserts_same_session_attempt(tmp_path):
    store = make_store(tmp_path)
    payload = {
        "candidate": {"self_attn": "fp8", "mlp": "mxfp4"},
        "real_accuracy_gap": 0.01,
        "eval_profile_hash": "profile",
    }

    store.record_quantization_experience(
        record_id="session-1:accuracy:1",
        source_session_id="session-1",
        context_fingerprint="fingerprint",
        model_arch="qwen3_5_moe",
        framework="vllm",
        gpu_type="mi355x",
        outcome="passed",
        verification_status="verified",
        payload=payload,
    )
    store.record_quantization_experience(
        record_id="session-1:accuracy:1",
        source_session_id="session-1",
        context_fingerprint="fingerprint",
        model_arch="qwen3_5_moe",
        framework="vllm",
        gpu_type="mi355x",
        outcome="passed",
        verification_status="verified",
        payload={**payload, "quantized_throughput": 120.0},
    )

    rows = store.find_quantization_experience(
        model_arch="qwen3_5_moe",
        framework="vllm",
        gpu_type="mi355x",
    )
    assert len(rows) == 1
    assert rows[0]["observation_count"] == 1
    assert rows[0]["payload"]["quantized_throughput"] == 120.0


def test_repair_experience_returns_semantic_guidance_without_patch(tmp_path):
    store = make_store(tmp_path)
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
        },
    )

    guidance = store.find_repair_guidance(
        framework="vllm",
        framework_version="abc",
        arch_fingerprint="arch",
        quant_signature="mlp=mxfp4",
        failure_mode="load_run",
        error_signature="v2|AttributeError|||none.to",
    )

    assert guidance["outcome"] == "fixed"
    assert guidance["payload"]["approach_summary"] == ("guard optional bias tensors")
    assert "patch_path" not in guidance["payload"]


def test_accuracy_repair_requires_matching_full_gate_for_verified_knowledge():
    from quark.experimental.torch.quant_perf.knowledge.terminal_experience import (
        _repair_is_verified,
    )

    journey = {
        "failure_class": "accuracy_gap",
        "quant_signature": "mlp=mxfp4",
        "status": "fixed",
        "promoted": True,
        "verifier_results": [{"verifier": "accuracy", "passed": True}],
    }

    assert _repair_is_verified(journey, {}, "mlp=mxfp4") is False
    assert (
        _repair_is_verified(
            journey,
            {"accuracy_validation": {"passed": True}},
            "mlp=fp8",
        )
        is False
    )
    assert (
        _repair_is_verified(
            journey,
            {"accuracy_validation": {"passed": True}},
            "mlp=mxfp4",
        )
        is True
    )


def test_kernel_experience_keeps_final_decision_and_evidence(tmp_path):
    store = make_store(tmp_path)
    store.record_kernel_optimization_experience(
        record_id="session-1:kernel:gemm",
        source_session_id="session-1",
        context_fingerprint="fingerprint",
        kernel_signature="kernel_gemm",
        bound_type="COMPUTE_BOUND",
        quant_signature="mlp=mxfp4",
        outcome="kept",
        verification_status="verified",
        payload={
            "compiler": "flydsl",
            "e2e_gain": 1.08,
            "accuracy_gap": 0.0,
        },
    )

    rows = store.find_kernel_optimization_experience(
        kernel_signature="kernel_gemm",
        bound_type="COMPUTE_BOUND",
        quant_signature="mlp=mxfp4",
    )

    assert rows[0]["outcome"] == "kept"
    assert rows[0]["payload"]["e2e_gain"] == 1.08


def test_baseline_health_cache_remains_exact_and_operational(tmp_path):
    store = make_store(tmp_path)
    store.record_baseline_health(
        fingerprint="runtime-a",
        framework="vllm",
        framework_commit="abc",
        outcome="failed",
        failure_class="framework",
        cache_policy="hard",
        diagnosis="unsupported architecture",
    )

    assert store.baseline_hard_failure("runtime-a")["diagnosis"] == ("unsupported architecture")
    assert store.baseline_hard_failure("runtime-b") is None


def test_experience_provider_exposes_advisory_records_without_replay(tmp_path):
    from quark.experimental.torch.quant_perf.knowledge import KnowledgeContext, KnowledgeRouter
    from quark.experimental.torch.quant_perf.knowledge.experience import ExperienceKnowledgeProvider

    store = make_store(tmp_path)
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
        },
    )

    bundle = KnowledgeRouter([ExperienceKnowledgeProvider(store)]).query(
        KnowledgeContext(
            domain="repair",
            stage="load",
            framework="vllm",
            framework_version="abc",
            arch_fingerprint="arch",
            quant_signature="mlp=mxfp4",
            failure_class="load_run",
            error_signature="v2|AttributeError|||none.to",
        )
    )

    assert bundle.exact_matches[0].record.kind == "validated_playbook"
    assert "guard optional bias" in bundle.exact_matches[0].record.summary


def test_terminal_recorder_captures_verified_session_experience_once(tmp_path):
    from types import SimpleNamespace

    from quark.experimental.torch.quant_perf.knowledge.terminal_experience import (
        TerminalExperienceRecorder,
    )

    store = make_store(tmp_path)
    state = {
        "session_id": "session-1",
        "accuracy_attempts": [
            {
                "attempt": 1,
                "baseline": 0.8,
                "quantized": 0.79,
                "gap": 0.0125,
                "passed": True,
                "candidate": {"self_attn": "fp8", "mlp": "mxfp4"},
                "profile_hash": "profile",
                "artifacts": {"result": "accuracy.json"},
            }
        ],
        "accuracy_validation": {"fingerprint": "accuracy-fingerprint"},
        "best_candidate": {"mlp": "mxfp4"},
        "repair_journey": [
            {
                "failure_class": "load_run",
                "quant_signature": "mlp=mxfp4",
                "status": "fixed",
                "promoted": True,
                "target_role": "framework",
                "knowledge_ids": ["repair.recipe"],
                "changed_files": ["vllm/quark_moe.py"],
                "verifier_results": [{"verifier": "load_inference", "passed": True}],
                "attempts": [{"tried": "guard optional bias"}],
            }
        ],
        "kernel_journey": [
            {
                "kernel_id": "kernel_gemm",
                "name": "kernel_gemm_0.kd",
                "discovery": {"roofline_bound": "COMPUTE_BOUND"},
                "source_mapping": {"compiler": "flydsl"},
                "backend_attempts": [{"approach_summary": "reuse prepared scales"}],
                "e2e": {
                    "decision": "KEEP",
                    "validated": True,
                    "gain": 1.08,
                    "accuracy_gap": 0.0,
                },
                "outcome": "adopted",
            }
        ],
    }
    spec = SimpleNamespace(
        model_arch="qwen3_5_moe",
        arch_fingerprint="arch",
        framework="vllm",
        framework_version="framework-sha",
        kernel_version="kernel-sha",
        gpu_type="mi355x",
        tp=4,
        quant_strategy=None,
        mxfp4_moe_backend="flydsl",
        mxfp4_gemm_backend="flydsl",
        w4a8_gemm_backend="flydsl",
    )
    recorder = TerminalExperienceRecorder(store)

    first = recorder.capture(spec=spec, state=state)
    second = recorder.capture(spec=spec, state=state)

    assert first.quantization == 1
    assert first.repair == 1
    assert first.kernel_optimization == 1
    assert second == first
    assert (
        len(
            store.find_quantization_experience(
                model_arch="qwen3_5_moe",
                framework="vllm",
                gpu_type="mi355x",
            )
        )
        == 1
    )
    assert (
        store.find_repair_guidance(
            framework="vllm",
            framework_version="framework-sha",
            arch_fingerprint="arch",
            quant_signature="mlp=mxfp4",
            failure_mode="load_run",
            error_signature="",
        )["verification_status"]
        == "verified"
    )
    assert (
        store.find_kernel_optimization_experience(
            kernel_signature="kernel_gemm_0.kd",
            bound_type="COMPUTE_BOUND",
            quant_signature="mlp=mxfp4",
        )[0]["outcome"]
        == "kept"
    )
