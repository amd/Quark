#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for perfopt/kernel_workflow.py pure helpers: kernel-name token
extraction and the early-emit disk scan (no SDK / GPU)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from quark.experimental.torch.quant_perf.perfopt import kernel_workflow
from quark.experimental.torch.quant_perf.perfopt.kernel_workflow import (
    _rebase_diff,
    _scan_emit,
    _short_kernel_token,
    apply_kernel_patch,
    cleanup_kernel_workflow_eval_dir,
    gen_workload_spec,
    persist_kernel_workflow_artifacts,
)
from quark.experimental.torch.quant_perf.perfopt.workload_contract import WorkloadContract


def _tracked_kernel_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "aiter"
    kernel = repo / "aiter" / "ops" / "moe" / "kernel.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("def kernel():\n    return 1\n")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "init",
        ],
        check=True,
    )
    return repo, kernel


def _mixed_moe_workload_contract(tmp_path: Path) -> WorkloadContract:
    workload = tmp_path / "workload.json"
    workload.write_text(
        json.dumps(
            {
                "schema": "workload-v1",
                "kernels": [
                    {
                        "name": "mfma_moe2.kd",
                        "cases": [
                            {
                                "dims": [
                                    [32, 2048],
                                    [256, 1024, 1024],
                                    [256, 2048, 256],
                                    [32, 8],
                                ],
                                "dtypes": [
                                    "c10::BFloat16",
                                    "c10::Float4_e2m1fn_x2",
                                    "c10::Float4_e2m1fn_x2",
                                    "float",
                                ],
                                "count": 440,
                                "baseline_latency_ms": 0.0095,
                            }
                        ],
                    }
                ],
            }
        )
    )
    contract = WorkloadContract.from_spec(
        str(workload),
        {
            "compiler": "flydsl",
            "builder_symbol": "compile_mixed_moe_gemm",
            "source_relpath": ("aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py"),
        },
    )
    assert contract is not None
    return contract


def test_short_kernel_token_strips_signature():
    sig = "void vllm::scaled_fp8_quant_kernel_strided_group_shape<c10::BFloat16, true>"
    assert _short_kernel_token(sig) == "scaled_fp8_quant_kernel_strided_group_shape"


def test_short_kernel_token_plain_name():
    assert _short_kernel_token("_fwd_kernel.kd") == "_fwd_kernel.kd"


def test_scan_emit_none_when_empty(tmp_path):
    assert _scan_emit(str(tmp_path)) is None


def test_scan_emit_does_not_borrow_another_candidates_speedup(tmp_path):
    eval_dir = tmp_path / "kw_eval"
    (eval_dir / "round_1" / "engineer_0").mkdir(parents=True)
    # an engineer worker_result with a verified number
    (eval_dir / "round_1" / "engineer_0" / "worker_result.json").write_text(json.dumps({"verified_geomean": 1.34}))
    # a committed current_best.diff (the safe emit target)
    (eval_dir / "current_best.diff").write_text("diff --git a/x b/x\n+opt\n")
    # also an engineer best_patch.diff (lower priority)
    (eval_dir / "round_1" / "engineer_0" / "best_patch.diff").write_text("diff\n")

    emit = _scan_emit(str(eval_dir))
    assert emit is not None
    assert emit["best_patch"].endswith("current_best.diff")
    assert emit["verified_speedup"] is None
    assert emit["micro_speedup_source"] == "unverified_patch"


def test_scan_emit_marginal_speedup_when_number_missing(tmp_path):
    eval_dir = tmp_path / "kw_eval"
    eval_dir.mkdir()
    (eval_dir / "current_best.diff").write_text("diff\n+x\n")
    emit = _scan_emit(str(eval_dir))
    assert emit is not None
    assert emit["verified_speedup"] is None
    assert emit["micro_speedup_source"] == "unverified_patch"


def test_persist_kernel_workflow_artifacts_rewrites_patch_path(tmp_path):
    repo, kernel = _tracked_kernel_repo(tmp_path)
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    patch = eval_dir / "final_patch.diff"
    patch.write_text(
        "diff --git a/home/geak/workspace/kernel.py b/./kernel.py\n"
        "--- a/home/geak/workspace/kernel.py\n"
        "+++ b/./kernel.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def kernel():\n"
        "-    return 1\n"
        "+    return 2\n"
    )
    (eval_dir / "director_validation.json").write_text(json.dumps({"validation_status": "accept"}))
    (eval_dir / "correctness.json").write_text(json.dumps({"success": True}))
    (eval_dir / "benchmark.json").write_text(json.dumps({"speedup": 1.2}))
    run_dir = tmp_path / "session" / "geak" / "kernel"
    report = {
        "best_patch": str(patch),
        "verified_speedup": 1.2,
        "eval_dir": str(eval_dir),
    }

    persisted = persist_kernel_workflow_artifacts(
        report,
        run_dir,
        kernel_src=str(kernel),
        source_repo=str(repo),
    )

    artifacts = run_dir / "artifacts"
    assert persisted["best_patch"] == str(artifacts / "final_patch.diff")
    assert "aiter/ops/moe/kernel.py" in (artifacts / "final_patch.diff").read_text()
    assert (artifacts / "raw_patch.diff").read_text() == patch.read_text()
    assert persisted["patch_changed_files"] == ["aiter/ops/moe/kernel.py"]
    assert persisted["patch_requires_rebuild"] is False
    assert (artifacts / "result.json").is_file()
    assert not (artifacts / "correctness.json").exists()
    assert not (artifacts / "benchmark.json").exists()


def test_persist_kernel_workflow_artifacts_filters_eval_only_patch_files(
    tmp_path,
):
    repo, kernel = _tracked_kernel_repo(tmp_path)
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    patch = eval_dir / "current_best.diff"
    patch.write_text(
        "diff --git a/kernel.py b/kernel.py\n"
        "--- a/kernel.py\n"
        "+++ b/kernel.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def kernel():\n"
        "-    return 1\n"
        "+    return 2\n"
        "diff --git a/.profile_smoke.stale_1/profile_report.txt "
        "b/.profile_smoke.stale_1/profile_report.txt\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/.profile_smoke.stale_1/profile_report.txt\n"
        "@@ -0,0 +1 @@\n"
        "+temporary profile\n"
        "diff --git a/test_harness.py b/test_harness.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/test_harness.py\n"
        "@@ -0,0 +1 @@\n"
        "+temporary harness\n"
    )
    run_dir = tmp_path / "session" / "geak" / "kernel"
    report = {
        "best_patch": str(patch),
        "verified_speedup": 1.2,
        "eval_dir": str(eval_dir),
    }

    persisted = persist_kernel_workflow_artifacts(
        report,
        run_dir,
        kernel_src=str(kernel),
        source_repo=str(repo),
    )

    artifacts = run_dir / "artifacts"
    assert persisted["best_patch"] == str(artifacts / "final_patch.diff")
    final_patch = (artifacts / "final_patch.diff").read_text()
    assert "aiter/ops/moe/kernel.py" in final_patch
    assert "test_harness.py" not in final_patch
    assert ".profile_smoke" not in final_patch
    assert persisted["patch_validation"]["status"] == "accepted"
    assert persisted["patch_validation"]["filtered_artifacts"] == [
        "aiter/ops/moe/.profile_smoke.stale_1/profile_report.txt",
        "aiter/ops/moe/test_harness.py",
    ]
    assert persisted["patch_validation"]["deliverable_changed_files"] == ["aiter/ops/moe/kernel.py"]
    assert persisted["patch_validation"]["rejected_new_files"] == []
    assert (artifacts / "raw_patch.diff").read_text() == patch.read_text()


@pytest.mark.parametrize("hip_file_kind", ["new", "modified", "tracked"])
def test_persist_kernel_workflow_artifacts_filters_generated_compiler_files(tmp_path, hip_file_kind):
    repo = tmp_path / "aiter"
    kernel = repo / "csrc" / "kernels" / "quant_kernels.cu"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("void kernel() { return; }\n")
    if hip_file_kind == "tracked":
        kernel.with_suffix(".hip").write_text(kernel.read_text())
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "init",
        ],
        check=True,
    )
    patch = tmp_path / "candidate.diff"
    patch.write_text(
        "diff --git a/quant_kernels.cu b/quant_kernels.cu\n"
        "--- a/quant_kernels.cu\n"
        "+++ b/quant_kernels.cu\n"
        "@@ -1 +1 @@\n"
        "-void kernel() { return; }\n"
        "+void kernel() { optimized(); }\n"
        "diff --git a/.torch_ext.stale_1/build.ninja "
        "b/.torch_ext.stale_1/build.ninja\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/.torch_ext.stale_1/build.ninja\n"
        "@@ -0,0 +1 @@\n"
        "+generated build\n"
        "diff --git a/quant_harness_pybind.cu "
        "b/quant_harness_pybind.cu\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/quant_harness_pybind.cu\n"
        "@@ -0,0 +1 @@\n"
        "+generated harness\n"
        "diff --git a/harness_pybind_hip.cpp b/harness_pybind_hip.cpp\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/harness_pybind_hip.cpp\n"
        "@@ -0,0 +1 @@\n"
        "+generated hip harness\n"
        "diff --git a/.harness/binding.cpp b/.harness/binding.cpp\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/.harness/binding.cpp\n"
        "@@ -0,0 +1 @@\n"
        "+generated binding\n"
        "diff --git a/.aiter_shadow_meta/csrc/kernels/helper.cu "
        "b/.aiter_shadow_meta/csrc/kernels/helper.cu\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/.aiter_shadow_meta/csrc/kernels/helper.cu\n"
        "@@ -0,0 +1 @@\n"
        "+generated shadow source\n"
        "diff --git a/quant_kernels.hip b/quant_kernels.hip\n"
        + (
            "new file mode 100644\n--- /dev/null\n+++ b/quant_kernels.hip\n@@ -0,0 +1 @@\n+generated hip source\n"
            if hip_file_kind == "new"
            else "--- a/quant_kernels.hip\n+++ b/quant_kernels.hip\n"
            "@@ -1 +1 @@\n-void kernel() { return; }\n+void kernel() { optimized(); }\n"
        )
    )
    run_dir = tmp_path / "session" / "geak" / "quant"

    persisted = persist_kernel_workflow_artifacts(
        {
            "best_patch": str(patch),
            "verified_speedup": 1.1,
        },
        run_dir,
        kernel_src=str(kernel),
        source_repo=str(repo),
    )

    assert persisted["patch_validation"]["status"] == "accepted"
    assert persisted["patch_changed_files"] == ["csrc/kernels/quant_kernels.cu"] + (
        ["csrc/kernels/quant_kernels.hip"] if hip_file_kind == "tracked" else []
    )
    assert persisted["patch_validation"]["filtered_artifacts"] == [
        "csrc/kernels/.torch_ext.stale_1/build.ninja",
        "csrc/kernels/quant_harness_pybind.cu",
        "csrc/kernels/harness_pybind_hip.cpp",
        "csrc/kernels/.harness/binding.cpp",
        "csrc/kernels/.aiter_shadow_meta/csrc/kernels/helper.cu",
    ] + ([] if hip_file_kind == "tracked" else ["csrc/kernels/quant_kernels.hip"])
    final_patch = Path(persisted["best_patch"]).read_text()
    assert "quant_kernels.cu" in final_patch
    assert "build.ninja" not in final_patch
    assert "harness_pybind" not in final_patch
    assert ".harness/binding.cpp" not in final_patch
    assert ".aiter_shadow_meta" not in final_patch
    assert ("quant_kernels.hip" in final_patch) is (hip_file_kind == "tracked")


def test_cleanup_kernel_workflow_eval_dir_only_removes_owned_tmp_paths(tmp_path):
    import tempfile

    owned = Path(tempfile.gettempdir()) / "quark_quant_perf_kw_eval" / "unit-test-owned"
    owned.mkdir(parents=True, exist_ok=True)
    (owned / "result.json").write_text("{}")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()

    cleanup_kernel_workflow_eval_dir(str(owned))
    cleanup_kernel_workflow_eval_dir(str(unrelated))

    assert not owned.exists()
    assert unrelated.exists()


def test_kernel_workflow_eval_dir_is_fresh_for_each_attempt(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        kernel_workflow.tempfile,
        "gettempdir",
        lambda: str(tmp_path),
    )

    first = Path(
        kernel_workflow._create_kernel_workflow_eval_dir(
            "/session/geak/kernel",
        )
    )
    (first / "stale-result.json").write_text("{}")
    second = Path(
        kernel_workflow._create_kernel_workflow_eval_dir(
            "/session/geak/kernel",
        )
    )

    assert first != second
    assert first.parent == tmp_path / "quark_quant_perf_kw_eval"
    assert second.parent == first.parent
    assert list(second.iterdir()) == []


def test_required_workload_rejects_unaligned_geak_metadata(tmp_path):
    contract = _mixed_moe_workload_contract(tmp_path)
    (tmp_path / "baseline_timing.json").write_text(json.dumps({"workload_aligned": False}))

    valid, reason = contract.validate_alignment(str(tmp_path))

    assert valid is False
    assert "workload_aligned=true" in reason


def test_required_workload_rejects_mismatched_case_dimensions(tmp_path):
    workload = tmp_path / "workload.json"
    workload.write_text(
        json.dumps(
            {
                "schema": "workload-v1",
                "kernels": [
                    {
                        "cases": [
                            {
                                "dims": [
                                    [32, 2048],
                                    [256, 1024, 1024],
                                ],
                                "count": 440,
                            }
                        ]
                    }
                ],
            }
        )
    )
    (tmp_path / "baseline_timing.json").write_text(
        json.dumps(
            {
                "workload_aligned": True,
                "test_cases": [
                    {
                        "dims": [
                            [32, 7168],
                            [256, 7168, 256],
                        ],
                        "count": 440,
                    }
                ],
            }
        )
    )

    contract = WorkloadContract.from_spec(str(workload), {})
    assert contract is not None
    valid, reason = contract.validate_alignment(str(tmp_path))

    assert valid is False
    assert "case dimensions/counts" in reason


def test_required_workload_accepts_matching_case_dimensions(tmp_path):
    expected_cases = [
        {
            "dims": [
                [32, 2048],
                [256, 1024, 1024],
                [256, 2048, 256],
                [32, 8],
                [32, 8],
                [256, 1024, 64],
                [256, 2048, 16],
            ],
            "count": 440,
        },
        {
            "dims": [
                [16, 2048],
                [256, 1024, 1024],
                [256, 2048, 256],
                [16, 8],
                [16, 8],
                [256, 1024, 64],
                [256, 2048, 16],
            ],
            "count": 40,
        },
    ]
    measured_cases = [
        {
            "dims": [
                [16, 2048],
                [256, 2048, 256],
                [256, 2048, 16],
            ],
            "count": 40,
        },
        {
            "dims": [
                [32, 2048],
                [256, 2048, 256],
                [256, 2048, 16],
            ],
            "count": 440,
        },
    ]
    workload = tmp_path / "workload.json"
    workload.write_text(
        json.dumps(
            {
                "schema": "workload-v1",
                "kernels": [{"cases": expected_cases}],
            }
        )
    )
    (tmp_path / "baseline_timing.json").write_text(
        json.dumps(
            {
                "workload_aligned": True,
                "test_cases": measured_cases,
            }
        )
    )

    contract = WorkloadContract.from_spec(str(workload), {})
    assert contract is not None
    valid, reason = contract.validate_alignment(str(tmp_path))

    assert valid is True
    assert "confirmed" in reason


def test_required_workload_rejects_baseline_latency_drift(tmp_path):
    workload = tmp_path / "workload.json"
    workload.write_text(
        json.dumps(
            {
                "schema": "workload-v1",
                "kernels": [
                    {
                        "cases": [
                            {
                                "dims": [
                                    [32, 2048],
                                    [256, 2048, 256],
                                ],
                                "count": 440,
                                "baseline_latency_ms": 0.009525,
                            }
                        ]
                    }
                ],
            }
        )
    )
    (tmp_path / "baseline_timing.json").write_text(
        json.dumps(
            {
                "workload_aligned": True,
                "test_cases": [
                    {
                        "dims": [
                            [32, 2048],
                            [256, 2048, 256],
                        ],
                        "count": 440,
                        "latency_ms": 0.017201,
                    }
                ],
            }
        )
    )

    contract = WorkloadContract.from_spec(str(workload), {})
    assert contract is not None
    valid, reason = contract.validate_alignment(str(tmp_path))

    assert valid is False
    assert "baseline latency" in reason


def test_analysis_contract_accepts_matching_mixed_moe_semantics(tmp_path):
    contract = _mixed_moe_workload_contract(tmp_path)
    (tmp_path / "analysis.json").write_text(
        json.dumps(
            {
                "problem": {
                    "model_dim_N": 2048,
                    "inter_dim_K": 512,
                    "experts": 256,
                    "topk": 8,
                    "cases_tokens": [32],
                }
            }
        )
    )

    reason = contract.analysis_violation(str(tmp_path))

    assert reason == ""


def test_analysis_contract_rejects_mixed_moe_semantic_mismatch(tmp_path):
    contract = _mixed_moe_workload_contract(tmp_path)
    (tmp_path / "analysis.json").write_text(
        json.dumps(
            {
                "problem": {
                    "model_dim_N": 7168,
                    "inter_dim_K": 256,
                    "experts": 256,
                    "topk": 8,
                    "cases_tokens": [16, 64, 256, 1024],
                }
            }
        )
    )

    reason = contract.analysis_violation(str(tmp_path))

    assert "model_dim" in reason
    assert "7168" in reason
    assert "2048" in reason


def test_baseline_contract_waits_until_timing_evidence_exists(tmp_path):
    contract = _mixed_moe_workload_contract(tmp_path)
    reason = contract.baseline_violation(str(tmp_path))

    assert reason == ""


def test_baseline_contract_rejects_misaligned_timing_evidence(tmp_path):
    workload = tmp_path / "workload.json"
    workload.write_text(
        json.dumps(
            {
                "schema": "workload-v1",
                "kernels": [
                    {
                        "cases": [
                            {
                                "dims": [[32, 2048]],
                                "count": 440,
                                "baseline_latency_ms": 0.0095,
                            }
                        ]
                    }
                ],
            }
        )
    )
    (tmp_path / "baseline_timing.json").write_text(
        json.dumps(
            {
                "workload_aligned": True,
                "test_cases": [
                    {
                        "dims": [[32, 8, 256]],
                        "count": 440,
                        "latency_ms": 0.039,
                    }
                ],
            }
        )
    )

    contract = WorkloadContract.from_spec(str(workload), {})
    assert contract is not None
    reason = contract.baseline_violation(str(tmp_path))

    assert "dimensions/counts" in reason


@pytest.mark.parametrize("source_layout", ["csrc", "aiter_meta/csrc"])
def test_kernel_workflow_sdk_env_isolates_aiter_candidate(tmp_path, monkeypatch, source_layout):
    monkeypatch.setenv("AMD_LLM_API_KEY", "test-gateway-key")
    source_repo = tmp_path / "aiter"
    (source_repo / "aiter").mkdir(parents=True)
    (source_repo / source_layout).mkdir(parents=True)

    env = kernel_workflow._build_kernel_workflow_sdk_env(
        gpu_id=2,
        geak_root="/opt/geak",
        source_repo=str(source_repo),
        run_dir=str(tmp_path / "run"),
    )

    assert env["ROCR_VISIBLE_DEVICES"] == "2"
    assert env["GEAK_ROOT"] == "/opt/geak"
    assert env["FLYDSL_RUNTIME_ENABLE_CACHE"] == "0"
    assert env.get("AITER_REBUILD") == "2"
    assert env.get("AITER_JIT_DIR") == str(tmp_path / "run" / "build" / "aiter")
    if source_layout == "csrc":
        assert env["AITER_ROOT_DIR"] == str(source_repo.resolve())


def test_kernel_workflow_sdk_env_includes_gateway_auth(monkeypatch):
    monkeypatch.setenv("AMD_LLM_GATEWAY_KEY", "real-gateway-key")
    monkeypatch.delenv("AMD_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_KEY", raising=False)

    env = kernel_workflow._build_kernel_workflow_sdk_env(
        gpu_id=0,
        geak_root="/opt/geak",
    )

    assert env["AMD_LLM_API_KEY"] == "real-gateway-key"
    assert env["GEAK_API_KEY"] == "real-gateway-key"
    assert env["LLM_GATEWAY_KEY"] == "real-gateway-key"
    assert env["ANTHROPIC_API_KEY"] == "dummy"
    assert env["ANTHROPIC_BASE_URL"] == "https://llm-api.amd.com/Anthropic"
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "Ocp-Apim-Subscription-Key: real-gateway-key"


def test_kernel_workflow_git_tracks_source_but_not_generated_files(tmp_path, monkeypatch):
    monkeypatch.setenv("AMD_LLM_API_KEY", "test-gateway-key")
    repo, kernel = _tracked_kernel_repo(tmp_path)
    header = kernel.parent / "include" / "native [1].hip"
    header.parent.mkdir()
    header.write_text("// tracked HIP source\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "header"],
        check=True,
    )
    (kernel.parent / "staged.hip").write_text("// not part of the committed baseline\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.name")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "existing setting")
    env = {
        **os.environ,
        **kernel_workflow._build_kernel_workflow_sdk_env(
            gpu_id=0,
            geak_root="/opt/geak",
            source_repo=str(repo),
            kernel_path=str(kernel.parent),
            run_dir=str(tmp_path / "run"),
        ),
    }
    worker = tmp_path / "worker"
    shutil.copytree(kernel.parent, worker)

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(worker), *args], env=env, capture_output=True, text=True, check=True
        ).stdout.strip()

    git("init", "-q")
    git("add", "-A")
    git("-c", "user.email=t@t", "commit", "-qm", "baseline")
    assert git("ls-files", "-z").rstrip("\0").split("\0") == ["include/native [1].hip", "kernel.py"]
    assert git("config", "user.name") == "existing setting"
    (worker / "kernel.py").write_text("def kernel():\n    return 2\n")
    generated = [
        ".harness_src/kernel.hip",
        ".harness_src/kernel_hip.hip",
        "another_stage/copy.cu",
        "include/native 1.hip",
    ]
    for name in generated:
        path = worker / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// generated during evaluation\n")
    git("add", "-A")
    assert git("diff", "--cached", "--name-only") == "kernel.py"
    assert all((worker / name).is_file() for name in generated)
    patch = tmp_path / "final.diff"
    patch.write_text(git("diff", "HEAD") + "\n")
    prepared = kernel_workflow.prepare_kernel_patch(str(patch), str(kernel), str(repo))
    assert prepared.accepted, prepared.reason
    assert prepared.changed_files == ("aiter/ops/moe/kernel.py",)
    (worker / "helper.h").write_text("// explicitly added new source\n")
    git("add", "-f", "helper.h")
    patch.write_text(git("diff", "HEAD") + "\n")
    rejected = kernel_workflow.prepare_kernel_patch(str(patch), str(kernel), str(repo))
    assert not rejected.accepted
    assert "unknown new files are not allowed" in rejected.reason


def test_run_kernel_workflow_checks_geak_root_before_loading_sdk(monkeypatch, tmp_path):
    monkeypatch.setattr(kernel_workflow.config, "geak_root", lambda: "")
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)

    result = kernel_workflow.run_kernel_workflow(
        kernel_path=str(tmp_path / "kernel.py"),
        task="Optimize the kernel.",
        gpu_id=0,
        run_dir=str(tmp_path / "run"),
    )

    assert result["watchdog_status"] == "kernel_workflow_not_found"
    assert result["error"] == "GEAK_ROOT is not configured."


def test_run_kernel_workflow_reports_missing_agent_sdk(monkeypatch, tmp_path):
    geak_root = tmp_path / "geak"
    workflow_dir = geak_root / "kernel_workflow"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "kernel_workflow.js").write_text("// test workflow\n")
    monkeypatch.setattr(kernel_workflow.config, "geak_root", lambda: str(geak_root))
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)

    result = kernel_workflow.run_kernel_workflow(
        kernel_path=str(tmp_path / "kernel.py"),
        task="Optimize the kernel.",
        gpu_id=0,
        run_dir=str(tmp_path / "run"),
    )

    assert result["watchdog_status"] == "kernel_workflow_dependency_missing"
    assert "amd-quark[quant_perf]" in result["error"]


def test_gen_workload_spec_rejects_cases_without_dimensions(
    tmp_path,
    monkeypatch,
):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "trace.json").write_text("{}")
    parser = tmp_path / "geak" / "e2e_workflow" / "scripts"
    parser.mkdir(parents=True)
    (parser / "parse_profile.py").write_text("# test parser\n")
    monkeypatch.setattr(
        "quark.experimental.torch.quant_perf.perfopt.kernel_workflow.config.geak_root",
        lambda: str(tmp_path / "geak"),
    )

    def fake_run(args, **_kwargs):
        out = Path(args[args.index("--workload-out") + 1])
        out.write_text(
            json.dumps(
                {
                    "schema": "workload-v1",
                    "kernels": [{"cases": [{"dims": [], "dtypes": []}]}],
                }
            )
        )

    monkeypatch.setattr(
        "quark.experimental.torch.quant_perf.perfopt.kernel_workflow.subprocess.run",
        fake_run,
    )

    result = gen_workload_spec(
        str(trace_dir),
        "kernel.kd",
        str(tmp_path / "workload.json"),
    )

    assert result is None


def test_gen_workload_spec_accepts_benchmarkable_case(tmp_path, monkeypatch):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "trace.json").write_text("{}")
    parser = tmp_path / "geak" / "e2e_workflow" / "scripts"
    parser.mkdir(parents=True)
    (parser / "parse_profile.py").write_text("# test parser\n")
    monkeypatch.setattr(
        "quark.experimental.torch.quant_perf.perfopt.kernel_workflow.config.geak_root",
        lambda: str(tmp_path / "geak"),
    )

    def fake_run(args, **_kwargs):
        out = Path(args[args.index("--workload-out") + 1])
        out.write_text(
            json.dumps(
                {
                    "schema": "workload-v1",
                    "kernels": [
                        {
                            "cases": [
                                {
                                    "dims": [[128, 4096]],
                                    "dtypes": ["bf16"],
                                }
                            ]
                        }
                    ],
                }
            )
        )

    monkeypatch.setattr(
        "quark.experimental.torch.quant_perf.perfopt.kernel_workflow.subprocess.run",
        fake_run,
    )
    out_path = tmp_path / "workload.json"

    result = gen_workload_spec(
        str(trace_dir),
        "kernel.kd",
        str(out_path),
    )

    assert result == str(out_path)


# --- patch rebase / apply (early-emit patch -> framework_repo) ---------------

_FP8_DIFF = (
    "diff --git a/home/kewang2/workspace/GEAK/exp/team_fp8/fp8/workspace/common.cu b/./common.cu\n"
    "index 52e159d6..d7db2b3b 100644\n"
    "--- a/home/kewang2/workspace/GEAK/exp/team_fp8/fp8/workspace/common.cu\n"
    "+++ b/./common.cu\n"
    "@@ -2,3 +2,4 @@ void scaled_fp8_quant_kernel_strided_group_shape(\n"
    "     scaled_fp8_conversion_vectorized(a, b, c);\n"
    "+    // optimized vectorized path\n"
    "     return;\n"
    " }\n"
)


def test_rebase_diff_maps_to_repo_dir():
    out = _rebase_diff(_FP8_DIFF, "csrc/quantization/w8a8/fp8")
    assert "--- a/csrc/quantization/w8a8/fp8/common.cu" in out
    assert "+++ b/csrc/quantization/w8a8/fp8/common.cu" in out
    assert "diff --git a/csrc/quantization/w8a8/fp8/common.cu b/csrc/quantization/w8a8/fp8/common.cu" in out


def test_apply_kernel_patch_to_repo(tmp_path, monkeypatch):
    """Rebase a GEAK-workspace-relative diff and apply it to framework_repo."""
    repo = tmp_path / "vllm"
    kdir = repo / "csrc" / "quantization" / "w8a8" / "fp8"
    kdir.mkdir(parents=True)
    kfile = kdir / "common.cu"
    orig = "void k(\n    a();\n    return;\n}\n"
    kfile.write_text(orig)

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True)

    git("init", "-q")
    git("add", "-A")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    # produce a real, valid diff, then rewrite paths to GEAK's workspace format
    kfile.write_text("void k(\n    a();\n    // optimized\n    return;\n}\n")
    real = git("diff").stdout
    kfile.write_text(orig)  # reset; apply_kernel_patch must re-introduce the change
    ws = "home/x/GEAK/exp/team/fp8/workspace"
    mangled = real.replace("a/csrc/quantization/w8a8/fp8/common.cu", f"a/{ws}/common.cu").replace(
        "b/csrc/quantization/w8a8/fp8/common.cu", "b/./common.cu"
    )
    mangled = "\n".join(mangled.splitlines()[1:]) + "\n"
    patch = tmp_path / "current_best.diff"
    patch.write_text(mangled)
    monkeypatch.chdir(tmp_path)

    ok, msg = apply_kernel_patch(
        patch.name,
        str(kfile),
        str(repo),
    )
    assert ok, msg
    assert "// optimized" in kfile.read_text()


def test_apply_kernel_patch_rejects_unauthorized_operations(tmp_path):
    repo = tmp_path / "aiter"
    kdir = repo / "aiter" / "ops" / "moe"
    kdir.mkdir(parents=True)
    kfile = kdir / "kernel.py"
    kfile.write_text("def kernel():\n    return 1\n")
    support = kdir / "support.py"
    support.write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        check=True,
    )
    cases = {
        "new": (
            "diff --git a/home/geak/workspace/new_kernel.py "
            "b/./new_kernel.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/./new_kernel.py\n"
            "@@ -0,0 +1 @@\n"
            "+def new_kernel(): pass\n",
            "unknown new files are not allowed",
        ),
        "delete": (
            "diff --git a/home/geak/workspace/kernel.py b/./kernel.py\n"
            "deleted file mode 100644\n"
            "--- a/home/geak/workspace/kernel.py\n"
            "+++ /dev/null\n",
            "non-modification patch operation",
        ),
        "rename": (
            "diff --git a/home/geak/workspace/kernel.py "
            "b/./renamed.py\n"
            "similarity index 100%\n"
            "rename from kernel.py\n"
            "rename to renamed.py\n",
            "non-modification patch operation",
        ),
        "binary": (
            "diff --git a/home/geak/workspace/kernel.py b/./kernel.py\nGIT binary patch\n",
            "non-modification patch operation",
        ),
        "no_target": (
            "diff --git a/home/geak/workspace/support.py "
            "b/./support.py\n"
            "--- a/home/geak/workspace/support.py\n"
            "+++ b/./support.py\n"
            "@@ -1 +1 @@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n",
            "does not modify target kernel source",
        ),
    }

    for name, (diff, expected) in cases.items():
        patch = tmp_path / f"{name}.diff"
        patch.write_text(diff)
        ok, msg = apply_kernel_patch(
            str(patch),
            str(kfile),
            str(repo),
        )
        assert not ok, name
        assert expected in msg, name
        assert "return 1" in kfile.read_text()
        assert support.read_text() == "VALUE = 1\n"

    assert not (kdir / "renamed.py").exists()


def test_apply_kernel_patch_filters_profile_artifacts(tmp_path):
    repo = tmp_path / "aiter"
    kdir = repo / "aiter" / "ops" / "moe"
    kdir.mkdir(parents=True)
    kfile = kdir / "kernel.py"
    kfile.write_text("def kernel():\n    return 1\n")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "init",
        ],
        check=True,
    )
    patch = tmp_path / "candidate.diff"
    patch.write_text(
        "diff --git a/home/geak/workspace/kernel.py b/./kernel.py\n"
        "--- a/home/geak/workspace/kernel.py\n"
        "+++ b/./kernel.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def kernel():\n"
        "-    return 1\n"
        "+    return 2\n"
        "diff --git a/home/geak/workspace/.profile_smoke.stale_1/"
        "profile_report.txt "
        "b/./.profile_smoke.stale_1/profile_report.txt\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/./.profile_smoke.stale_1/profile_report.txt\n"
        "@@ -0,0 +1 @@\n"
        "+temporary profile\n"
    )

    ok, msg = apply_kernel_patch(str(patch), str(kfile), str(repo))

    assert ok, msg
    assert "return 2" in kfile.read_text()
    assert not (kdir / ".profile_smoke.stale_1").exists()


# --- complete-or-salvage: _scan_final (completion detection) ------------------


@pytest.mark.parametrize("with_workload", [False, True])
def test_workflow_passes_measurement_contract_to_geak(monkeypatch, tmp_path, with_workload):
    from contextlib import asynccontextmanager

    workflow_dir = tmp_path / "geak" / "kernel_workflow"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "kernel_workflow.js").write_text("// test workflow\n")
    contract = _mixed_moe_workload_contract(tmp_path) if with_workload else None
    arguments = {}

    async def query(prompt):
        arguments.update(json.loads(prompt.split("args: ", 1)[1]))

    async def receive_messages():
        yield type("ResultMessage", (), {"is_error": False})()

    @asynccontextmanager
    async def client(**kwargs):
        yield SimpleNamespace(query=query, receive_messages=receive_messages)

    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk",
        SimpleNamespace(ClaudeAgentOptions=lambda **kwargs: kwargs, ClaudeSDKClient=client),
    )
    monkeypatch.setattr(kernel_workflow.config, "geak_root", lambda: str(workflow_dir.parent))
    monkeypatch.setattr(kernel_workflow, "_build_kernel_workflow_sdk_env", lambda **kwargs: {})
    monkeypatch.setattr(kernel_workflow, "_create_kernel_workflow_eval_dir", lambda _: str(tmp_path / "eval"))

    report = kernel_workflow.run_kernel_workflow(
        kernel_path=str(tmp_path),
        task="Optimize kernel.",
        gpu_id=0,
        run_dir=str(tmp_path / "run"),
        workload_contract=contract,
    )

    assert not report["error"]
    if contract is None:
        assert "harness_addendum" not in arguments
    else:
        addendum = Path(arguments["harness_addendum"])
        assert addendum.parent == tmp_path / "run"
        text = addendum.read_text()
        assert contract.prompt_suffix() in text
        assert "GEAK_TIMING_RECEIPT" in text
        assert "start.query()" in text


def test_scan_final_none_when_not_finished(tmp_path):
    from quark.experimental.torch.quant_perf.perfopt.kernel_workflow import _scan_final

    (tmp_path / "round_1" / "engineer_0").mkdir(parents=True)
    (tmp_path / "round_1" / "engineer_0" / "best_patch.diff").write_text("diff\n")
    assert _scan_final(str(tmp_path)) is None  # a candidate is not completion


def test_scan_final_reads_director_validation(tmp_path):
    from quark.experimental.torch.quant_perf.perfopt.kernel_workflow import _scan_final

    fp = tmp_path / "final_patch.diff"
    fp.write_text("diff --git a/common.cu b/common.cu\n")
    (tmp_path / "director_validation.json").write_text(
        json.dumps(
            {
                "validation_status": "accept",
                "director_verified_speedup_geomean": 1.27,
                "correctness": "pass",
                "per_case": [{"name": "case", "baseline_ms": 1.27, "optimized_ms": 1.0}],
                "timing_receipt": {"all_primed": True, "timer_unprimed": False},
                "final_patch": str(fp),
            }
        )
    )
    out = _scan_final(str(tmp_path))
    assert out and out["final"] is True
    assert abs(out["verified_speedup"] - 1.27) < 1e-9
    assert out["best_patch"].endswith("final_patch.diff")


def test_scan_final_waits_for_director_after_final_patch_diff(tmp_path):
    from quark.experimental.torch.quant_perf.perfopt.kernel_workflow import _scan_final

    (tmp_path / "final_patch.diff").write_text("diff --git a/common.cu b/common.cu\n+opt\n")
    out = _scan_final(str(tmp_path))
    assert out is None


@pytest.mark.parametrize(
    "overrides,speedup,status",
    [
        ({}, 1.2, "candidate"),
        ({"per_case": None}, None, "candidate"),
        ({"timing_receipt": None}, None, "candidate"),
        ({"correctness": None}, None, "incomplete"),
        ({"correctness": "fail"}, None, "rejected"),
        ({"timing_receipt": {"all_primed": False, "timer_unprimed": False}}, None, "candidate"),
        ({"per_case": [{"baseline_ms": 1.2, "optimized_ms": float("nan")}]}, None, "candidate"),
        ({"validation_status": "Accepted"}, 1.2, "candidate"),
        ({"validation_status": " ACCEPTED "}, 1.2, "candidate"),
        ({"validation_status": "accept"}, 1.2, "candidate"),
        ({"validation_status": "rejected", "director_verified_speedup_geomean": 0.85}, None, "rejected"),
        ({"validation_status": " Rejected "}, None, "rejected"),
        ({"validation_status": "rejected", "timing_receipt": None}, None, "rejected"),
        ({"validation_status": None}, None, "incomplete"),
        ({"validation_status": "unknown"}, None, "incomplete"),
        ({"validation_status": ["accepted"]}, None, "incomplete"),
        ({"timing_receipt": {"all_primed": True}}, None, "candidate"),
        *[
            ({"timing_receipt": {"all_primed": True, "timer_unprimed": value}}, None, "candidate")
            for value in (True, None, "true", 1)
        ],
    ],
)
def test_final_speedup_requires_device_measurements(tmp_path, overrides, speedup, status):
    from quark.experimental.torch.quant_perf.perfopt.journey import load_geak_resume_state
    from quark.experimental.torch.quant_perf.perfopt.service import OptimizationService

    repo, kernel = _tracked_kernel_repo(tmp_path)
    (tmp_path / "final_patch.diff").write_text(
        "diff --git a/kernel.py b/kernel.py\n--- a/kernel.py\n+++ b/kernel.py\n"
        "@@ -1,2 +1,2 @@\n def kernel():\n-    return 1\n+    return 2\n"
    )
    validation = {
        "validation_status": "flagged",  # a smaller measured gain is still usable
        "correctness": "pass",
        "director_verified_speedup_geomean": 1.2,
        "per_case": [{"name": "case", "baseline_ms": 1.2, "optimized_ms": 1.0}],
        "timing_receipt": {"all_primed": True, "timer_unprimed": False},
    }
    validation.update(overrides)
    (tmp_path / "director_validation.json").write_text(json.dumps(validation))
    report = kernel_workflow._scan_final(str(tmp_path))
    assert report["verified_speedup"] == speedup
    assert OptimizationService._keep(report) is (status == "candidate")
    assert OptimizationService._candidate_status(report) == status
    persisted = persist_kernel_workflow_artifacts(
        report, tmp_path / "run", kernel_src=str(kernel), source_repo=str(repo)
    )
    entry = {
        **persisted,
        "kernel_name": "kernel",
        "kernel_sig": "kernel",
        "status": "candidate",
        "verified_speedup": 99.0,
    }
    _, resumed = load_geak_resume_state(SimpleNamespace(state={"geak_patches": [entry]}))
    assert bool(resumed) is (status == "candidate")
    if resumed:
        assert resumed[0]["verified_speedup"] == speedup
    assert (tmp_path / "run/artifacts/benchmark.json").exists() is (speedup is not None)
    if speedup is not None:
        Path(persisted["best_patch"]).write_text("changed after validation")
        assert not load_geak_resume_state(SimpleNamespace(state={"geak_patches": [entry]}))[1]


@pytest.mark.parametrize(
    "correctness,ending",
    [
        ("pass", "completed"),
        ("fail", "completed"),
        (None, "completed"),
        ("pass", "timeout"),
        (None, "timeout"),
        ("pass", "error"),
        ("pass", "late"),
        ("pass", "updated"),
        ("pass", "failed_direction"),
    ],
)
def test_workflow_preserves_director_correctness_for_retention(monkeypatch, tmp_path, correctness, ending):
    import anyio

    from quark.experimental.torch.quant_perf.perfopt.service import OptimizationService

    repo, kernel = _tracked_kernel_repo(tmp_path)
    geak_root = tmp_path / "geak"
    workflow_dir = geak_root / "kernel_workflow"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "kernel_workflow.js").write_text("// test workflow\n")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    final_patch = eval_dir / "final_patch.diff"
    messages = []
    expected_correctness = {"pass": True, "fail": False}.get(correctness)

    def message(name, **values):
        return type(name, (SimpleNamespace,), {})(**values)

    def write_validation():
        (eval_dir / "director_validation.json").write_text(
            json.dumps(
                {
                    "validation_status": "accepted" if correctness == "pass" else "flagged",
                    "correctness": correctness,
                    "director_verified_speedup_geomean": 1.2,
                    "final_patch": str(final_patch),
                }
            )
        )
        messages.append("validation")

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def query(self, prompt):
            pass

        async def receive_messages(self):
            yield message("TaskStartedMessage", task_id="workflow")
            yield message("ResultMessage", is_error=False)
            if ending == "failed_direction":
                yield message("TaskStartedMessage", task_id="direction")
                yield message("TaskNotificationMessage", task_id="direction", status="failed")
            final_patch.write_text(
                "diff --git a/kernel.py b/kernel.py\n"
                "--- a/kernel.py\n+++ b/kernel.py\n@@ -1,2 +1,2 @@\n"
                " def kernel():\n-    return 1\n+    return 2\n"
            )
            messages.append("report")
            yield SimpleNamespace(content=[])
            if ending != "late" and not (ending == "timeout" and correctness is None):
                write_validation()
            yield SimpleNamespace(content=[])
            if ending == "timeout":
                await anyio.sleep_forever()
            if ending == "error":
                raise RuntimeError("SDK disconnected")
            messages.append("terminal")
            if ending == "updated":
                yield message("TaskUpdatedMessage", task_id="workflow", patch={"status": "completed"})
                await anyio.sleep_forever()
            else:
                yield message("TaskNotificationMessage", task_id="workflow", status="completed")

    monkeypatch.setattr(kernel_workflow.config, "geak_root", lambda: str(geak_root))
    monkeypatch.setattr(kernel_workflow, "_create_kernel_workflow_eval_dir", lambda _: str(eval_dir))
    monkeypatch.setattr(kernel_workflow, "_build_kernel_workflow_sdk_env", lambda **kwargs: {})
    if ending == "late":

        async def finish_validation(_delay):
            write_validation()

        monkeypatch.setattr(anyio, "sleep", finish_validation)
    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk",
        SimpleNamespace(
            ClaudeAgentOptions=lambda **kwargs: kwargs,
            ClaudeSDKClient=FakeClient,
        ),
    )
    report = kernel_workflow.run_kernel_workflow(
        kernel_path=str(kernel.parent),
        task="Optimize kernel.",
        gpu_id=0,
        run_dir=str(tmp_path / "run"),
        hard_timeout_s=0.05 if ending == "timeout" else 2,
    )
    persisted = persist_kernel_workflow_artifacts(
        report,
        tmp_path / "run",
        kernel_src=str(kernel),
        source_repo=str(repo),
    )
    cleanup_kernel_workflow_eval_dir(str(eval_dir))

    if ending in {"completed", "late"}:
        assert "terminal" in messages
    assert persisted["execution_status"] == ({"timeout": "timed_out", "error": "error"}.get(ending, "completed"))
    assert persisted["timed_out"] is (ending == "timeout")
    assert persisted["round_evaluation"]["correctness"]["success"] is expected_correctness
    assert OptimizationService._keep(persisted) is (expected_correctness is True)
    assert OptimizationService._candidate_status(persisted) == (
        "candidate" if expected_correctness is True else "rejected" if expected_correctness is False else "incomplete"
    )
    if ending == "error":
        assert "SDK disconnected" in persisted["error"]
    evidence = json.loads((tmp_path / "run" / "artifacts" / "correctness.json").read_text())
    assert evidence["success"] is expected_correctness
    saved = json.loads((tmp_path / "run" / "artifacts" / "result.json").read_text())
    assert saved["execution_status"] == persisted["execution_status"]
    assert saved["best_patch"] == persisted["best_patch"]


@pytest.mark.parametrize("result", [[], {}, "incomplete", {"correctness": "pass", "final_patch": "missing.diff"}])
def test_scan_final_does_not_attach_unrelated_patch(tmp_path, result):
    (tmp_path / "final_patch.diff").write_text("unrelated patch")
    (tmp_path / "director_validation.json").write_text(json.dumps(result))
    report = kernel_workflow._scan_final(str(tmp_path))
    assert report is None or not report["best_patch"]


def test_scan_final_real_number_not_sentinel_when_no_gain(tmp_path):
    """A completed run that reverted to a bit-identical baseline (Director 1.0x,
    empty patch) must report the REAL 1.0 (so _keep rejects it), not a 1.01
    sentinel that would falsely 'keep' a no-op."""
    from quark.experimental.torch.quant_perf.perfopt.kernel_workflow import _scan_final

    (tmp_path / "final_patch.diff").write_text("")  # 0-byte: bit-identical baseline
    (tmp_path / "director_validation.json").write_text(
        json.dumps(
            {
                "validation_status": "accepted",
                "director_verified_speedup_geomean": 1.0,
                "correctness": "pass",
                "per_case": [{"name": "case", "baseline_ms": 1.0, "optimized_ms": 1.0}],
                "timing_receipt": {"all_primed": True, "timer_unprimed": False},
                "final_patch": str(tmp_path / "final_patch.diff"),
            }
        )
    )
    out = _scan_final(str(tmp_path))
    assert out and out["final"] is True
    assert out["verified_speedup"] == 1.0  # real number, NOT 1.01
    assert out["best_patch"] == ""  # empty patch dropped
