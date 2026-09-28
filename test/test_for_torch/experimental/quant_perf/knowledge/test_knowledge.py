#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


def _write_record(root: Path, payload: dict) -> Path:
    path = root / payload["domain"] / f"{payload['id']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _repair_record(**overrides) -> dict:
    record = {
        "schema_version": 1,
        "id": "repair.vllm.optional-moe-bias.v1",
        "domain": "repair",
        "kind": "verified_repair_recipe",
        "status": "verified",
        "applicability": {"framework": ["vllm"]},
        "match": {
            "failure_class": ["load_run"],
            "error_contains": ["'NoneType' object has no attribute 'to'"],
            "keywords": ["quark_moe", "w13_bias"],
        },
        "summary": "Guard optional MoE bias tensors before conversion.",
        "guidance": [
            "Guard each optional bias independently.",
            "Do not synthesize missing bias tensors.",
        ],
        "required_checks": ["load", "inference"],
        "evidence": {"level": "E3", "verifier": "load_inference"},
        "provenance": {
            "kind": "external_commit",
            "repository": "vllm",
            "commit": "a" * 40,
        },
    }
    record.update(overrides)
    return record


def test_curated_provider_matches_structured_context(tmp_path):
    from quark.experimental.torch.quant_perf.knowledge import (
        CuratedKnowledgeProvider,
        KnowledgeContext,
        KnowledgeRouter,
    )

    _write_record(tmp_path, _repair_record())
    bundle = KnowledgeRouter([CuratedKnowledgeProvider(tmp_path)]).query(
        KnowledgeContext(
            domain="repair",
            stage="load",
            framework="vllm",
            failure_class="load_run",
            error_signature=("AttributeError: 'NoneType' object has no attribute 'to'"),
        )
    )

    assert [match.record.id for match in bundle.exact_matches] == ["repair.vllm.optional-moe-bias.v1"]
    assert bundle.required_checks == ["load", "inference"]


def test_curated_provider_rejects_wrong_domain_or_framework(tmp_path):
    from quark.experimental.torch.quant_perf.knowledge import (
        CuratedKnowledgeProvider,
        KnowledgeContext,
    )

    _write_record(tmp_path, _repair_record())
    provider = CuratedKnowledgeProvider(tmp_path)

    assert (
        provider.query(
            KnowledgeContext(
                domain="quantization",
                framework="vllm",
                failure_class="load_run",
                error_signature="'NoneType' object has no attribute 'to'",
            )
        )
        == []
    )
    assert (
        provider.query(
            KnowledgeContext(
                domain="repair",
                framework="atom",
                failure_class="load_run",
                error_signature="'NoneType' object has no attribute 'to'",
            )
        )
        == []
    )


def test_keyword_only_records_are_exact_only_when_text_matches(tmp_path):
    from quark.experimental.torch.quant_perf.knowledge import (
        CuratedKnowledgeProvider,
        KnowledgeContext,
        KnowledgeRouter,
    )

    base = _repair_record(
        domain="quantization",
        kind="quantization_rule",
        status="validated",
        applicability={},
        match={"keywords": ["self_attn", "fp8"]},
        required_checks=["post_export_accuracy"],
    )
    _write_record(
        tmp_path,
        {
            **base,
            "id": "quantization.self-attn-fp8.v1",
        },
    )
    _write_record(
        tmp_path,
        {
            **base,
            "id": "quantization.mlp-mxfp4.v1",
            "match": {"keywords": ["mlp", "mxfp4"]},
        },
    )

    bundle = KnowledgeRouter([CuratedKnowledgeProvider(tmp_path)]).query(
        KnowledgeContext(
            domain="quantization",
            quant_signature="self_attn=fp8",
        )
    )

    assert [match.record.id for match in bundle.exact_matches] == ["quantization.self-attn-fp8.v1"]
    assert any(match.record.id == "quantization.mlp-mxfp4.v1" for match in bundle.priors)


def test_curated_provider_rejects_duplicate_yaml_keys(tmp_path):
    from quark.experimental.torch.quant_perf.knowledge import CuratedKnowledgeProvider

    path = tmp_path / "repair" / "duplicate.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "id: repair.duplicate.v1",
                "id: repair.duplicate.v2",
                "domain: repair",
                "kind: reference",
                "status: validated",
                "applicability: {}",
                "match: {}",
                "summary: duplicate",
                "guidance: []",
                "required_checks: []",
                "evidence: {level: E1}",
                "provenance: {kind: manual}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate YAML key"):
        CuratedKnowledgeProvider(tmp_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"domain": "evaluation"}, "unsupported knowledge domain"),
        (
            {"guidance": ["Apply /tmp/fix.patch to the framework repository."]},
            "executable patch",
        ),
        (
            {
                "provenance": {
                    "kind": "manual",
                    "path": "/home/user/private/source.md",
                }
            },
            "absolute local path",
        ),
    ],
)
def test_curated_provider_rejects_unsafe_or_unsupported_records(
    tmp_path,
    overrides,
    message,
):
    from quark.experimental.torch.quant_perf.knowledge import CuratedKnowledgeProvider

    _write_record(tmp_path, _repair_record(**overrides))

    with pytest.raises(ValueError, match=message):
        CuratedKnowledgeProvider(tmp_path)


def test_renderer_is_structured_and_honors_character_budget(tmp_path):
    from quark.experimental.torch.quant_perf.knowledge import (
        CuratedKnowledgeProvider,
        KnowledgeContext,
        KnowledgeRenderer,
        KnowledgeRouter,
    )

    _write_record(
        tmp_path,
        _repair_record(
            summary="S" * 200,
            guidance=["G" * 200, "H" * 200],
        ),
    )
    bundle = KnowledgeRouter([CuratedKnowledgeProvider(tmp_path)]).query(
        KnowledgeContext(
            domain="repair",
            framework="vllm",
            failure_class="load_run",
            error_signature="'NoneType' object has no attribute 'to'",
        )
    )

    rendered = KnowledgeRenderer(max_chars=320).render(bundle)

    assert len(rendered) <= 320
    assert "Exact matches" in rendered
    assert "repair.vllm.optional-moe-bias.v1" in rendered


def test_shipped_curated_records_are_valid():
    from quark.experimental.torch.quant_perf.knowledge import (
        build_knowledge_router,
    )

    provider = build_knowledge_router().providers[0]
    entries = provider.items

    assert len(entries) >= 20
    assert all(item.provenance.get("kind") for item in entries)
    assert {
        "repair.quark.packed-modules-mapping.v1",
        "repair.vllm.cudagraph-lifecycle.v1",
        "quant.mxfp4.frontier-moe-recipe.v1",
        "kernel.flydsl.productionization.v1",
        "repair.vllm.quark.optional-moe-bias.v1",
    }.issubset({item.id for item in entries})


def test_fused_moe_parent_scale_shape_failure_matches_curated_repair():
    from quark.experimental.torch.quant_perf.knowledge import (
        KnowledgeContext,
        build_knowledge_router,
    )

    bundle = build_knowledge_router().query(
        KnowledgeContext(
            domain="repair",
            framework="vllm",
            model_arch="qwen3_5_moe",
            failure_class="load_run",
            error_signature="RuntimeError: shape '[]' is invalid for input of size 32768",
            quant_signature="mxfp4_fp8",
        )
    )

    assert "repair.vllm.quark.fused-moe-expert-config.v1" in {match.record.id for match in bundle.exact_matches}


@pytest.mark.parametrize(
    "error,framework,failure_class,expected",
    [
        ("wk_weights_proj.input_scale", "vllm", "load_run", True),
        ("wk_weights_proj.weight_scale", "vllm", "load_run", True),
        ("other_proj.weight_scale", "vllm", "load_run", False),
        ("wk_weights_proj.weight_scale", "sglang", "load_run", False),
        ("wk_weights_proj.weight_scale", "vllm", "accuracy_gap", False),
    ],
)
def test_fused_indexer_scale_keyerror_matches_curated_repair(error, framework, failure_class, expected):
    from quark.experimental.torch.quant_perf.knowledge import (
        KnowledgeContext,
        build_knowledge_router,
    )

    bundle = build_knowledge_router().query(
        KnowledgeContext(
            domain="repair",
            framework=framework,
            model_arch="glm_moe_dsa",
            gpu_type="mi355x",
            failure_class=failure_class,
            error_signature=f"KeyError: model.layers.0.self_attn.indexer.{error}",
            quant_signature="self_attn=fp8",
        )
    )

    record_ids = {match.record.id for match in bundle.exact_matches}
    assert ("repair.vllm.quark.fused-indexer-scale-loading.v1" in record_ids) is expected
    if expected:
        assert {"parameter_mapping", "load", "inference", "accuracy"}.issubset(bundle.required_checks)


def test_quark_fp8_per_block_failure_matches_curated_repair():
    from quark.experimental.torch.quant_perf.knowledge import (
        KnowledgeContext,
        build_knowledge_router,
    )

    bundle = build_knowledge_router().query(
        KnowledgeContext(
            domain="repair",
            framework="vllm",
            model_arch="glm_moe_dsa",
            gpu_type="mi355x",
            failure_class="load_run",
            error_signature=(
                "NotImplementedError: No quark compatible scheme was found. "
                "Weight config: {'dtype': 'fp8_e4m3', 'qscheme': 'per_block'}"
            ),
            quant_signature="routed_moe=mxfp4;self_attn=native",
        )
    )

    record_ids = {match.record.id for match in bundle.exact_matches}
    assert "repair.vllm.quark.fp8-per-block-runtime.v1" in record_ids
    assert {"scheme_selection", "block_shape", "load", "inference", "accuracy"}.issubset(bundle.required_checks)


def test_static_fp8_synthetic_weight_recovery_matches_curated_repair():
    from quark.experimental.torch.quant_perf.knowledge import (
        KnowledgeContext,
        build_knowledge_router,
    )

    bundle = build_knowledge_router().query(
        KnowledgeContext(
            domain="repair",
            framework="vllm",
            model_arch="glm_moe_dsa",
            gpu_type="mi355x",
            failure_class="accuracy_gap",
            error_signature="quantized model accuracy gap 0.98",
            quant_signature="self_attn=fp8;mlp=mxfp4",
        )
    )

    record_ids = {match.record.id for match in bundle.exact_matches}
    assert "repair.vllm.quark.synthetic-weight-recovery.v1" in record_ids
    assert {"weight_reference", "load", "inference", "accuracy"}.issubset(bundle.required_checks)


def test_untuned_aiter_a4w4_splitk_matches_curated_repair():
    from quark.experimental.torch.quant_perf.knowledge import (
        KnowledgeContext,
        build_knowledge_router,
    )

    bundle = build_knowledge_router().query(
        KnowledgeContext(
            domain="repair",
            framework="vllm",
            model_arch="glm_moe_dsa",
            gpu_type="mi355x",
            failure_class="accuracy_gap",
            error_signature="garbage outputs from AITER A4W4 split-k",
            quant_signature="mlp=mxfp4;shared_expert=mxfp4",
        )
    )

    record_ids = {match.record.id for match in bundle.exact_matches}
    assert "repair.aiter.untuned-a4w4-splitk.v1" in record_ids
    assert {"kernel_reference", "load", "inference", "accuracy"}.issubset(bundle.required_checks)


def test_session_provider_adds_rejected_attempts_to_bundle():
    from quark.experimental.torch.quant_perf.knowledge import KnowledgeContext, KnowledgeRouter
    from quark.experimental.torch.quant_perf.knowledge.session import SessionKnowledgeProvider

    bundle = KnowledgeRouter(
        [
            SessionKnowledgeProvider(
                {
                    "repair_journey": [
                        {
                            "failure_class": "load_run",
                            "attempts": [
                                {
                                    "tried": "patched registry",
                                    "failed_because": ("shape mismatch remained"),
                                }
                            ],
                        }
                    ]
                }
            )
        ]
    ).query(
        KnowledgeContext(
            domain="repair",
            failure_class="load_run",
        )
    )

    assert bundle.session_dead_ends == [
        {
            "tried": "patched registry",
            "failed_because": "shape mismatch remained",
        }
    ]


def test_knowledge_audit_appends_query_record(tmp_path):
    from quark.experimental.torch.quant_perf.knowledge.audit import append_query_audit
    from quark.experimental.torch.quant_perf.knowledge.types import KnowledgeBundle

    append_query_audit(
        tmp_path,
        consumer="runtime_repair",
        context_hash="ctx",
        bundle=KnowledgeBundle(source_ids=["a", "b"]),
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "knowledge" / "query_audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["consumer"] == "runtime_repair"
    assert rows[0]["knowledge_ids"] == ["a", "b"]
