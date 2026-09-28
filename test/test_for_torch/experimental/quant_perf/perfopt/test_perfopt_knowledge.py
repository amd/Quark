#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from quark.experimental.torch.quant_perf.session.spec import Spec


def test_kernel_knowledge_combines_seed_and_experience(tmp_path):
    from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore
    from quark.experimental.torch.quant_perf.perfopt.knowledge_context import build_kernel_knowledge

    spec = Spec(
        model_dir="/model",
        base_model="/model",
        framework="vllm",
        gpu_type="mi355x",
        gpu_arch="MI355X",
        isl=1024,
        osl=1024,
        quant_strategy="mxfp4",
        session_dir=str(tmp_path),
    )

    with ExperienceStore(tmp_path / "experience.sqlite") as store:
        store.record_kernel_optimization_experience(
            record_id="session:kernel:1",
            source_session_id="session",
            context_fingerprint="fingerprint",
            kernel_signature="flydsl_gemm",
            bound_type="COMPUTE_BOUND",
            quant_signature="mxfp4",
            outcome="kept",
            verification_status="verified",
            payload={"approach_summary": ("A similar kernel used prewarmed scratch.")},
        )
        bundle, rendered = build_kernel_knowledge(
            experience_store=store,
            state={},
            spec=spec,
            bottleneck={
                "op_name": "flydsl_gemm",
                "roofline_bound": "COMPUTE_BOUND",
                "compiler": "flydsl",
            },
        )

    assert "kernel.flydsl.productionization.v1" in bundle.source_ids
    assert any(item.startswith("experience.kernel.") for item in bundle.source_ids)
    assert "prewarmed scratch" in rendered
    assert (tmp_path / "knowledge" / "query_audit.jsonl").exists()
