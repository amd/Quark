#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for direct PTQ artifact validation."""

import asyncio
import sys
from pathlib import Path
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quark.experimental.torch.quant_perf.quantize.direct_ptq import (
    _build_ptq_prompt,
    check_quant_artifacts,
    run_ptq,
)
from quark.experimental.torch.quant_perf.session.spec import Spec, StageError

from .quant_artifact_fixtures import (
    write_minimal_safetensors,
)


def test_success_when_safetensors_and_config_present(tmp_path):
    write_minimal_safetensors(tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text("{}")
    result = check_quant_artifacts(str(tmp_path))
    assert result["status"] == "success"
    assert result["has_safetensors"] and result["has_config"]


def test_empty_safetensors_is_not_a_valid_checkpoint(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "config.json").write_text("{}")

    result = check_quant_artifacts(str(tmp_path))

    assert result["status"] == "failed"
    assert "invalid_safetensors" in result["errors"]


@pytest.mark.parametrize(
    ("directory_exists", "has_safetensors", "has_config"),
    [
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, False, False),
    ],
)
def test_failed_artifact_combinations(
    tmp_path,
    directory_exists,
    has_safetensors,
    has_config,
):
    target = tmp_path / "checkpoint"
    if directory_exists:
        target.mkdir()
    if has_safetensors:
        (target / "model.safetensors").write_bytes(b"")
    if has_config:
        (target / "config.json").write_text("{}")

    result = check_quant_artifacts(str(target))

    assert result["status"] == "failed"
    assert result["has_safetensors"] is has_safetensors
    assert result["has_config"] is has_config


def test_ptq_prompt_keeps_runtime_intent_and_adds_advisory_knowledge(tmp_path):
    from quark.experimental.torch.quant_perf.session.spec import Spec

    spec = Spec(
        model_dir="/models/source",
        base_model="/models/source",
        framework="vllm",
        gpu_type="mi355x",
        gpu_arch="MI355X",
        isl=1024,
        osl=1024,
        quant_strategy="use MXFP4 for routed experts",
        session_dir=str(tmp_path),
    )

    prompt = _build_ptq_prompt(
        spec,
        str(tmp_path),
        knowledge_text="## Retrieved knowledge\n- advisory recipe",
    )

    assert "use MXFP4 for routed experts" in prompt
    assert "## Retrieved knowledge" in prompt
    assert "advisory recipe" in prompt
    assert "verifiers remain authoritative" in prompt
    assert "Do not launch quantization in the background" in prompt
    assert "wait for its foreground process to exit" in prompt


def test_run_ptq_reports_missing_agent_sdk(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    spec = Spec(
        model_dir="/models/source",
        base_model="/models/source",
        framework="vllm",
        gpu_type="mi355x",
        gpu_arch="MI355X",
        isl=128,
        osl=128,
        quant_strategy="FP8",
        session_dir=str(tmp_path),
    )

    with pytest.raises(StageError, match="Quant-Perf optional dependencies") as caught:
        asyncio.run(run_ptq(spec, spec.session_dir, quark_root=str(tmp_path)))

    assert caught.value.code == "missing_agent_sdk"


def test_run_ptq_uses_isolated_quark_workspace_and_experience_kb(
    tmp_path,
    monkeypatch,
):
    captured = {}

    class ClaudeAgentOptions:
        def __init__(self, **kwargs):
            captured["options"] = kwargs

    async def query(**kwargs):
        captured["query"] = kwargs
        if False:
            yield None

    sdk = ModuleType("claude_agent_sdk")
    sdk.ClaudeAgentOptions = ClaudeAgentOptions
    sdk.query = query
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    monkeypatch.setattr(
        "quark.experimental.torch.quant_perf.quantize.direct_ptq.config.build_subprocess_env",
        lambda: {},
    )

    quark_workspace = tmp_path / "quark-worktree"
    quark_workspace.mkdir()
    spec = Spec(
        model_dir="/models/source",
        base_model="/models/source",
        framework="vllm",
        gpu_type="mi355x",
        gpu_arch="MI355X",
        isl=128,
        osl=128,
        quant_strategy="FP8",
        model_arch="qwen3",
        session_dir=str(tmp_path / "session"),
    )

    from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore

    with ExperienceStore(tmp_path / "experience.sqlite") as store:
        store.record_quantization_experience(
            record_id="session-1:accuracy:1",
            source_session_id="session-1",
            context_fingerprint="fingerprint",
            model_arch="qwen3",
            framework="vllm",
            gpu_type="mi355x",
            outcome="passed",
            verification_status="verified",
            payload={
                "candidate": {"mlp": "fp8"},
                "real_accuracy_gap": 0.0,
            },
        )
        asyncio.run(
            run_ptq(
                spec,
                spec.session_dir,
                experience_store=store,
                state={},
                quark_root=str(quark_workspace),
            )
        )

    assert captured["options"]["cwd"] == str(quark_workspace)
    audit = (Path(spec.session_dir) / "knowledge" / "query_audit.jsonl").read_text()
    assert "experience.quantization." in audit
