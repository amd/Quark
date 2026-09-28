#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from quark.experimental.torch.quant_perf.reporting.service import (
    _capability_summary,
    build_final_summary,
    build_session_breakdown,
    render_final_markdown,
    render_session_markdown,
    write_final_artifacts,
)
from quark.experimental.torch.quant_perf.session.spec import Checkpoint, DeployPackage, EvalProfile, PerfResult, Spec


def test_kernel_capability_reports_incomplete_attempt():
    summary = _capability_summary({"geak_patches": [{"status": "incomplete"}]})
    assert summary["kernel_optimization"]["status"] == "incomplete"


def make_spec(tmp_path: Path, **overrides) -> Spec:
    defaults = dict(
        model_dir="/models/qwen",
        base_model="/models/qwen",
        framework="vllm",
        gpu_type="mi350x",
        gpu_arch="MI355X",
        isl=1024,
        osl=1024,
        quant_strategy=None,
        session_dir=str(tmp_path),
        layer_precision_candidates=["mxfp4", "fp8", "native"],
        kv_cache_precision_candidates=["native"],
        gsm8k_num_samples=100,
        target_gain=1.55,
        accuracy_gap=0.03,
        bench_concurrency=64,
        framework_repo="/repos/vllm",
        kernel_repo="/repos/aiter",
        mxfp4_moe_backend="flydsl",
        mxfp4_gemm_backend="flydsl",
        w4a8_gemm_backend="flydsl",
        aiter_config_fmoe="/configs/fmoe.csv",
        eval_profile=EvalProfile(
            profile_id="gsm8k-chat-nothink-v1",
            profile_hash="profile-hash",
            model_mode="chat",
            apply_chat_template=True,
            enable_thinking=False,
            detection_reason="chat_template",
        ),
    )
    defaults.update(overrides)
    return Spec(**defaults)


def populated_checkpoint(tmp_path: Path) -> tuple[Spec, Checkpoint]:
    spec = make_spec(tmp_path)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "eval_profile": spec.eval_profile.to_dict(),
            "eval_profile_hash": spec.eval_profile.profile_hash,
            "stage": "perf_failed",
            "quant_ckpt_dir": str(tmp_path / "quant_ckpt"),
            "best_candidate": {
                "linear_attn_mode": "mxfp4",
                "self_attn_mode": "mxfp4",
                "mlp_mode": "mxfp4",
                "kv_cache_mode": "native",
            },
            "mix_precision_search": {
                "schema_version": 1,
                "status": "completed",
                "api": "quark.experimental.torch.mix_precision",
                "config": {"hardware": "mi355"},
                "result": {
                    "best_config": {
                        "mlp_mode": "mxfp4",
                        "kv_cache_mode": "native",
                    },
                    "all_results": [
                        {
                            "config": {
                                "mlp_mode": "mxfp4",
                                "kv_cache_mode": "native",
                            },
                            "metrics": {"gsm8k": 0.84},
                            "relative_change": {"gsm8k": -0.01},
                            "is_valid": True,
                            "rank": 3,
                        }
                    ],
                    "baseline_metrics": {"gsm8k": 0.85},
                    "total_configs_evaluated": 1,
                    "total_configs_available": 4,
                    "search_time_seconds": 12.0,
                    "granularity": "module",
                    "hardware": "mi355",
                },
                "candidate_queue": [
                    {
                        "config": {
                            "mlp_mode": "mxfp4",
                            "kv_cache_mode": "native",
                        },
                        "status": "accepted",
                    }
                ],
                "candidate_cursor": 0,
                "candidate_order_source": ("quark_reverse_evaluation_order"),
            },
            "accuracy_attempts": [
                {
                    "baseline": 0.85,
                    "quantized": 0.84,
                    "gap": (0.85 - 0.84) / 0.85,
                    "passed": True,
                    "artifacts": {
                        "baseline": "/eval/baseline",
                        "quantized": "/eval/quantized",
                    },
                }
            ],
            "performance_measurements": [
                {
                    "role": "quant_only",
                    "baseline_tps": 3200.0,
                    "quantized_tps": 4800.0,
                    "gain": 1.5,
                }
            ],
            "vendor_gemm_tuning": {
                "status": "skipped",
                "reason": "tuner_unavailable",
                "candidates": [],
            },
            "vendor_shape_evidence": {
                "status": "exact",
                "shape_count": 2,
                "selected_signature_count": 2,
            },
            "bottleneck_analysis": {
                "policy_version": 1,
                "requested_mode": "differential",
                "effective_mode": "differential",
                "status": "completed",
                "reason": "",
                "quantized_trace": "/trace/quantized.json.gz",
                "baseline_trace": "/trace/baseline.json.gz",
                "trace_kind": "raw",
                "candidates": [
                    {
                        "rank": 1,
                        "kernel_id": "k1",
                        "name": "mfma_moe1",
                        "kernel_time_us": 100.0,
                        "baseline_time_us": 0.0,
                        "differential_type": "unique_to_quantized",
                    }
                ],
            },
            "kernel_journey": [
                {
                    "kernel_id": "k1",
                    "name": "mfma_moe1",
                    "outcome": "rejected",
                    "source_mapping": {
                        "source_file": "/aiter/mixed_moe.py",
                        "method": "operator_rule",
                        "confidence": "operator_rule",
                        "retryable": False,
                    },
                    "backend_attempts": [
                        {
                            "backend": "geak",
                            "correctness_passed": True,
                            "micro_speedup": 0.99,
                        }
                    ],
                    "e2e": {"decision": "REJECTED", "validated": False},
                }
            ],
            "retain_trials": [
                {
                    "kernel_id": "k1",
                    "accuracy_score": 0.84,
                    "accuracy_gap": (0.85 - 0.84) / 0.85,
                    "throughput_tps": 4790.0,
                    "gain": 4790.0 / 3200.0,
                    "decision": "DROP",
                    "reason": "below_keep_floor",
                }
            ],
            "phase_timeline": [
                {"action": "quantize", "status": "done"},
                {"action": "perfopt", "status": "done"},
            ],
            "baseline_reference": {
                "score": 0.85,
                "profile_hash": "profile-hash",
                "source": "measured",
            },
            "baseline_runtime_health": {
                "status": "healthy",
                "fingerprint": "runtime-fingerprint",
                "source": "measured",
            },
            "performance_runtime": {
                "requested_gpu_memory_utilization": 0.75,
                "effective_gpu_memory_utilization": 0.8,
                "runtime_env": {"VLLM_ROCM_MOE_PADDING": "0"},
            },
            "recovery_attempts": [
                {
                    "stage": "throughput",
                    "role": "baseline",
                    "code": "rocm_moe_padding_oom",
                    "action": "set_env",
                    "outcome": "retry",
                }
            ],
            "repair_journey": [
                {
                    "failure_class": "load_run",
                    "status": "fixed",
                    "target_role": "framework",
                }
            ],
            "change_ledger": [{"intent": "repair", "status": "promoted"}],
            "experience_capture": {
                "quantization": 1,
                "repair": 1,
                "kernel_optimization": 1,
                "reviewable": 3,
            },
            "baseline_gsm8k": 0.85,
            "baseline_tps": 3200.0,
            "quant_tps": 4800.0,
        }
    )
    ckpt.save()
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(exist_ok=True)
    (knowledge_dir / "query_audit.jsonl").write_text(
        json.dumps(
            {
                "consumer": "runtime_repair",
                "knowledge_ids": ["repair.quark.packed-modules-mapping.v1"],
            }
        )
        + "\n"
    )
    (tmp_path / "llm_calls.jsonl").write_text(json.dumps({"call_type": "runtime_repair", "outcome": "verified"}) + "\n")
    return spec, ckpt


def test_build_session_breakdown_is_complete_fact_source(tmp_path):
    spec, ckpt = populated_checkpoint(tmp_path)
    package = DeployPackage(
        status="perf_below_target",
        quant_ckpt_dir=ckpt.state["quant_ckpt_dir"],
        perf=PerfResult(patches=[], gain=1.5),
        message="below target",
    )

    breakdown = build_session_breakdown(spec, ckpt.state, package)

    assert breakdown["schema_version"] == "quark.quant_perf.session_breakdown.v5"
    assert breakdown["session"]["stop_reason"] == "perf_below_target"
    assert breakdown["workload"]["objective"]["target_gain"] == 1.55
    assert breakdown["workload"]["gpu_id"] == 0
    assert breakdown["workload"]["runtime"]["w4a8_gemm_backend"] == "flydsl"
    assert breakdown["workload"]["runtime"]["mxfp4_gemm_backend"] == "flydsl"
    assert breakdown["workload"]["runtime"]["layer_precision_candidates"] == ["mxfp4", "fp8", "native"]
    assert breakdown["workload"]["runtime"]["kv_cache_precision_candidates"] == ["native"]
    assert "layer_modes" not in breakdown["workload"]["runtime"]
    assert "kv_cache_modes" not in breakdown["workload"]["runtime"]
    search = breakdown["quantization_search"]
    assert search["api"] == "quark.experimental.torch.mix_precision"
    assert search["total_configs_evaluated"] == 1
    assert search["total_configs_available"] == 4
    assert search["candidate_cursor"] == 0
    assert search["candidate_order_source"] == "quark_reverse_evaluation_order"
    assert breakdown["bottleneck_analysis"]["requested_mode"] == ("differential")
    assert breakdown["bottleneck_analysis"]["effective_mode"] == ("differential")


def test_reporting_exposes_patch_bundle_artifacts(tmp_path):
    spec, ckpt = populated_checkpoint(tmp_path)
    manifest = tmp_path / "reports" / "patches" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    ckpt.state["patch_bundle"] = {
        "manifest": str(manifest),
        "framework": str(manifest.parent / "framework.diff"),
    }
    ckpt.state["resolved_sources"] = {
        "framework": {"kind": "installed_overlay"},
        "kernel": {"kind": "editable_git"},
    }
    ckpt.state["session_runtime"] = {
        "python_exe": "/session/venv/bin/python",
        "pythonpath_prefixes": ["/session/overlay"],
    }
    package = DeployPackage(
        status="perf_below_target",
        quant_ckpt_dir=ckpt.state["quant_ckpt_dir"],
        perf=PerfResult(patches=[], gain=1.5),
    )

    breakdown = build_session_breakdown(spec, ckpt.state, package)
    final = build_final_summary(breakdown)

    assert breakdown["patch_bundle"]["manifest"] == str(manifest)
    assert breakdown["repositories"]["framework"]["source_kind"] == ("installed_overlay")
    assert breakdown["workload"]["runtime"]["session_runtime"]["python_exe"] == "/session/venv/bin/python"
    assert final["artifacts"]["patch_bundle"]["manifest"] == str(manifest)
    assert breakdown["accuracy"]["attempts"][0]["quantized"] == 0.84
    assert breakdown["accuracy"]["profile"]["profile_hash"] == "profile-hash"
    protocols = breakdown["evaluation_protocols"]
    assert protocols["search_eval"]["authoritative"] is False
    assert protocols["search_eval"]["evaluator"] == "quark_fakequant"
    assert protocols["real_accuracy_gate"]["authoritative"] is True
    assert protocols["real_accuracy_gate"]["num_fewshot"] == 0
    assert protocols["real_accuracy_gate"]["profile_hash"] == "profile-hash"
    assert breakdown["performance"]["measurements"][0]["gain"] == 1.5
    assert breakdown["recovery_details"]["attempt_count"] == 1
    assert breakdown["recovery_details"]["performance_runtime"]["effective_gpu_memory_utilization"] == 0.8
    assert breakdown["recovery_details"]["baseline_reference"]["score"] == 0.85
    assert breakdown["kernel_journey"][0]["kernel_id"] == "k1"
    assert breakdown["performance"]["vendor_gemm_tuning"]["reason"] == "tuner_unavailable"
    assert breakdown["performance"]["vendor_shape_evidence"]["shape_count"] == 2
    assert breakdown["kernel_journey"][0]["e2e"]["decision"] == "REJECTED"
    assert breakdown["repair_journey"][0]["status"] == "fixed"
    assert breakdown["knowledge"]["queries"][0]["consumer"] == "runtime_repair"
    assert breakdown["knowledge"]["experience_capture"]["reviewable"] == 3
    assert breakdown["llm_calls"][0]["outcome"] == "verified"
    assert breakdown["change_ledger"][0]["intent"] == "repair"
    assert breakdown["versions"]["reporting"] == "quark-quant-perf-reporting-2.1.0"
    assert breakdown["versions"]["quant_perf"]
    state_provenance = next(item for item in breakdown["data_provenance"] if item["source"] == "state_json")
    assert state_provenance["snapshot_sha256"]
    assert breakdown["artifact_manifest"]["found_count"] >= 1
    assert "quant_trace" in breakdown["artifact_manifest"]["missing_sources"]
    assert breakdown["source_files"]["state_json"].endswith("state.json")


@pytest.mark.parametrize("status", ["fixed", "not_fixed"])
def test_repair_report_shows_rounds_and_timings(tmp_path, status):
    spec, ckpt = populated_checkpoint(tmp_path)
    ckpt.state["repair_journey"] = [
        {
            "failure_class": "load_run",
            "target_role": "framework",
            "status": status,
            "candidate_id": "candidate-1",
            "rounds": 2,
            "elapsed_seconds": 12.0,
            "attempts": [
                {"round": 1, "generation_seconds": 2.0, "verification_seconds": 3.0},
                {"round": 2, "generation_seconds": 2.0, "verification_seconds": 3.0},
            ],
        }
    ]
    breakdown = build_session_breakdown(spec, ckpt.state, DeployPackage(status="failed"))
    assert breakdown["capability_summary"]["runtime_repair"]["status"] == (
        "completed" if status == "fixed" else "not_fixed"
    )
    markdown = render_session_markdown(breakdown)
    assert "| Rounds | Generation (s) | Verification (s) | Total (s) |" in markdown
    assert "| 2 | 4.00 | 6.00 | 12.00 |" in markdown


def test_final_summary_is_compact_projection(tmp_path):
    spec, ckpt = populated_checkpoint(tmp_path)
    ckpt.state["runtime_inventory"] = {
        "python": "3.12.13",
        "platform": "Linux-test",
        "accelerator": {
            "rocm": "7.2.1",
            "cuda": None,
            "gpu_models": ["AMD Instinct MI355X"],
        },
        "packages": {
            "vllm": {
                "version": "0.24.0",
                "origin": "/runtime/vllm/__init__.py",
                "git_sha": "abc123",
            }
        },
        "tools": {
            "claude": {
                "version": "2.1.212 (Claude Code)",
                "path": "/usr/local/bin/claude",
            }
        },
    }
    package = DeployPackage(status="perf_below_target", perf=PerfResult([], 1.5))
    breakdown = build_session_breakdown(spec, ckpt.state, package)

    final = build_final_summary(breakdown)

    assert final["schema_version"] == "quark.quant_perf.final.v5"
    assert "target" in final["session"]["stop_reason_explanation"]
    assert final["accuracy"]["gap"] == (0.85 - 0.84) / 0.85
    assert final["accuracy"]["profile_hash"] == "profile-hash"
    assert final["accuracy"]["num_fewshot"] == 0
    assert final["accuracy"]["prompting_strategy"] == "cot"
    assert final["accuracy"]["settings_source"] == "quant_perf_profile"
    assert final["accuracy"]["artifacts"]["quantized"] == ("/eval/quantized")
    assert final["quantization"]["candidates"] == [
        {
            "mlp_mode": "mxfp4",
            "kv_cache_mode": "native",
        }
    ]
    assert final["performance"]["final_gain"] == 1.5
    assert final["optimization"]["requested_bottleneck_mode"] == ("differential")
    assert final["optimization"]["effective_bottleneck_mode"] == ("differential")
    assert final["recovery"]["attempt_count"] == 1
    assert final["recovery"]["effective_gpu_memory_utilization"] == 0.8
    assert final["recovery"]["runtime_env"]["VLLM_ROCM_MOE_PADDING"] == "0"
    assert final["optimization"]["patches_retained"] == 0
    assert final["versions"]["runtime"]["packages"]["vllm"]["version"] == "0.24.0"
    assert "kernel_journey" not in final

    markdown = render_session_markdown(breakdown)
    assert "| vllm | `0.24.0` | `/runtime/vllm/__init__.py` | `abc123` |" in markdown
    assert "- ROCm: `7.2.1`" in markdown


def test_accuracy_only_report_marks_performance_not_requested(tmp_path):
    spec = make_spec(
        tmp_path,
        performance_mode="off",
        target_gain=None,
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "done",
            "performance_status": "not_requested",
            "accuracy_attempts": [
                {
                    "baseline": 0.85,
                    "quantized": 0.84,
                    "gap": (0.85 - 0.84) / 0.85,
                    "passed": True,
                }
            ],
        }
    )
    package = DeployPackage(
        status="success",
        quant_ckpt_dir=spec.quant_ckpt_dir,
    )

    breakdown = build_session_breakdown(spec, ckpt.state, package)
    final = build_final_summary(breakdown)

    assert breakdown["performance"]["status"] == "not_requested"
    assert breakdown["performance"]["target_met"] is None
    assert final["performance"]["status"] == "not_requested"
    assert final["performance"]["target_met"] is None
    assert "- Performance: `not_requested`." in render_final_markdown(final)
    assert "- Status: `not_requested`" in render_session_markdown(breakdown)


def test_final_candidate_list_uses_only_the_persisted_candidate_queue(tmp_path):
    spec, ckpt = populated_checkpoint(tmp_path)
    ckpt.state["mix_precision_search"]["candidate_queue"] = []
    package = DeployPackage(status="perf_below_target", perf=PerfResult([], 1.5))

    final = build_final_summary(build_session_breakdown(spec, ckpt.state, package))

    assert final["quantization"]["candidates"] == []


def test_reports_partial_search_timeout_completion(tmp_path):
    spec, ckpt = populated_checkpoint(tmp_path)
    search = ckpt.state["mix_precision_search"]
    search["status"] = "partial_timeout"
    search["termination_reason"] = "search_timeout"
    package = DeployPackage(status="success", quant_ckpt_dir=ckpt.state["quant_ckpt_dir"])

    breakdown = build_session_breakdown(spec, ckpt.state, package)
    final = build_final_summary(breakdown)

    assert breakdown["quantization_search"]["status"] == "partial_timeout"
    assert breakdown["quantization_search"]["termination_reason"] == "search_timeout"
    assert final["quantization"]["search_status"] == "partial_timeout"
    assert final["quantization"]["termination_reason"] == "search_timeout"
    assert "partial_timeout" in render_final_markdown(final)
    assert "search_timeout" in render_session_markdown(breakdown)


def test_reports_include_commands_derived_from_the_persisted_run_spec(tmp_path):
    spec, ckpt = populated_checkpoint(tmp_path)
    spec = replace(
        spec,
        vllm_extra_args=["--trust-remote-code", "--max-num-seqs 64", "--kv-cache-dtype fp8"],
        invocation_argv=[
            "quark-quant-perf",
            "--model",
            spec.model_dir,
            "--gpu-type",
            "mi350x",
        ],
    )
    spec.runtime.runtime_python = "/runtime/bin/python"
    spec.runtime.runtime_env = {"PYTHONPATH": "/runtime/overlay"}
    ckpt.state["invocation"] = {
        "argv": list(spec.invocation_argv),
        "spec": spec.to_dict(),
        "env": {"AITER_FLYDSL_FORCE": "1"},
    }
    ckpt.state["performance_runtime"] = {
        "effective_gpu_memory_utilization": 0.78,
        "runtime_env": {"VLLM_ROCM_MOE_PADDING": "0"},
    }
    ckpt.state["patch_bundle"] = {
        "framework": str(tmp_path / "reports" / "patches" / "framework.diff"),
    }
    package = DeployPackage(status="perf_below_target", perf=PerfResult([], 1.5))

    breakdown = build_session_breakdown(spec, ckpt.state, package)
    final = build_final_summary(breakdown)

    commands = breakdown["reproduction"]["commands"]
    assert commands["original"] == spec.invocation_argv
    assert commands["evaluate"] == [
        "quark-quant-perf",
        "eval",
        "--from-session",
        str(tmp_path.resolve()),
        "--session-dir",
        "<new-eval-session>",
    ]
    assert commands["serve"][:2] == ["/runtime/bin/python", "-c"]
    assert "start_vllm_server" in commands["serve"][2]
    assert commands["benchmark"][:2] == ["/runtime/bin/python", "-c"]
    assert "gpu_memory_utilization=0.78" in commands["benchmark"][2]
    with patch(
        "quark.experimental.torch.quant_perf.evaluation.throughput.measure_throughput", autospec=True
    ) as benchmark:
        benchmark.return_value.to_dict.return_value = {}
        exec(commands["benchmark"][2], {})
    benchmark.assert_called_once()
    assert benchmark.call_args.kwargs["trust_remote_code"] is True
    assert benchmark.call_args.kwargs["max_num_seqs"] == 64
    assert benchmark.call_args.kwargs["kv_cache_dtype"] == "fp8"
    assert breakdown["reproduction"]["environment"]["ROCR_VISIBLE_DEVICES"] == "0"
    assert breakdown["reproduction"]["environment"]["VLLM_ROCM_MOE_PADDING"] == "0"
    assert final["reproduction"] == breakdown["reproduction"]

    markdown = render_final_markdown(final)
    assert "## Reproduce / Serve / Evaluate" in markdown
    assert "quark-quant-perf eval --from-session" in markdown
    assert "<new-eval-session>" in markdown
    assert "Retained patch artifacts:" in markdown
    assert "framework.diff" in markdown


def test_session_report_displays_generated_kernel_artifact(tmp_path):
    spec, ckpt = populated_checkpoint(tmp_path)
    generated_source = tmp_path / "runtime" / "cache" / "triton" / "HASH" / "triton_red_fused_example.source"
    ckpt.state["kernel_journey"] = [
        {
            "kernel_id": "triton_red_fused_example",
            "name": "triton_red_fused_example.kd",
            "outcome": "skipped",
            "source_mapping": {
                "mapping_kind": "generated_artifact",
                "patchable": False,
                "generated_source_file": str(generated_source),
                "method": "torchinductor_generated",
                "confidence": "generated_artifact",
                "retryable": False,
            },
            "backend_attempts": [],
            "e2e": {},
        }
    ]
    package = DeployPackage(
        status="perf_below_target",
        perf=PerfResult([], 1.5),
    )

    markdown = render_session_markdown(build_session_breakdown(spec, ckpt.state, package))

    assert str(generated_source) in markdown
    assert "generated_artifact" in markdown


def test_write_final_artifacts_writes_four_valid_files(tmp_path):
    spec, ckpt = populated_checkpoint(tmp_path)
    package = DeployPackage(status="perf_below_target", perf=PerfResult([], 1.5))

    paths = write_final_artifacts(spec, ckpt.state, package)

    assert set(paths) == {
        "final_json",
        "final_md",
        "session_breakdown_json",
        "session_report_md",
    }
    for path in paths.values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0
    json.loads(Path(paths["final_json"]).read_text())
    breakdown = json.loads(Path(paths["session_breakdown_json"]).read_text())
    assert breakdown["report_artifacts"] == paths
    assert breakdown["reporting"]["status"] == "complete"
    assert breakdown["final"]["deploy_package"]["report_status"] == "complete"
    assert breakdown["final"]["deploy_package"]["report_paths"] == paths
    assert breakdown["phase_timeline"][-1]["action"] == "final"
    assert breakdown["phase_timeline"][-1]["status"] == "complete"
    final_md = Path(paths["final_md"]).read_text()
    session_md = Path(paths["session_report_md"]).read_text()
    assert "# Quark Quant-Perf Final Report" in final_md
    assert "Candidates tried" not in final_md
    assert "## Repository Versions" in final_md
    assert "# Quark Quant-Perf Session Report" in session_md
    assert "Runtime experiences recorded: `3`" in session_md
    assert "Reviewable experiences: `3`" in session_md
    for heading in (
        "## Repositories & Versions",
        "## Quantization Search",
        "## Accuracy",
        "## Performance Results",
        "## Capability Summary",
        "## Kernel Optimization",
        "## Retain Trials",
        "## Gain Attribution",
        "## Phase Timeline",
        "## Source Artifacts",
        "## Data Provenance",
    ):
        assert heading in session_md
    assert "mfma_moe1" in session_md
    assert "below_keep_floor" in session_md
    assert (
        session_md.count(
            "| Kernel | Source | Resolution | Outcome | Execution | Compile | Correctness | Micro speedup | E2E decision |"
        )
        == 1
    )
    assert "operator_rule / operator_rule" in session_md
    assert (
        "| Kernel | Mode | Accuracy gap | TPS | Gain | Incremental | Effective floor | "
        "Initial decision | Final decision | Reason |"
    ) in session_md


def test_write_final_artifacts_includes_kernel_mapping_artifacts(
    tmp_path,
):
    spec, ckpt = populated_checkpoint(tmp_path)
    provenance = tmp_path / "kernel_provenance.json"
    resolution = tmp_path / "kernel_source_resolution.json"
    provenance.write_text('{"schema_version":"p","entries":[]}')
    resolution.write_text('{"schema_version":"r","entries":[]}')
    ckpt.state["kernel_provenance_artifact"] = str(provenance)
    package = DeployPackage(
        status="perf_below_target",
        quant_ckpt_dir=ckpt.state["quant_ckpt_dir"],
        perf=PerfResult(patches=[], gain=1.5),
    )

    paths = write_final_artifacts(spec, ckpt.state, package)
    final = json.loads(Path(paths["final_json"]).read_text())
    breakdown = json.loads(Path(paths["session_breakdown_json"]).read_text())

    assert paths["kernel_provenance_json"] == str(provenance)
    assert paths["kernel_source_resolution_json"] == str(resolution)
    assert final["artifacts"]["kernel_provenance_json"] == str(provenance)
    assert breakdown["report_artifacts"]["kernel_source_resolution_json"] == str(resolution)
    by_source = {row["source"]: row for row in breakdown["data_provenance"]}
    assert by_source["kernel_provenance"]["found"] is True
    assert by_source["kernel_source_resolution"]["found"] is True


def test_final_performance_prefers_final_abba_measurement(tmp_path):
    spec, ckpt = populated_checkpoint(tmp_path)
    ckpt.state["retention_stack"] = [
        {
            "kind": "patch",
            "patch": "/p.diff",
            "active": True,
        }
    ]
    ckpt.state["performance_measurements"].append(
        {
            "attempt": 2,
            "role": "final_abba",
            "mode": "abba",
            "baseline_tps": 100.0,
            "final_tps": 180.0,
            "gain": 1.8,
            "baseline_measurements": [{"median_tps": 99.0}, {"median_tps": 101.0}],
            "final_measurements": [{"median_tps": 179.0}, {"median_tps": 181.0}],
        }
    )
    ckpt.state["retain_trials"].append(
        {
            "kernel_id": "k2",
            "decision": "KEEP",
            "throughput_tps": 170.0,
            "gain": 1.7,
        }
    )
    package = DeployPackage(
        status="success",
        quant_ckpt_dir=ckpt.state["quant_ckpt_dir"],
        perf=PerfResult(patches=[], gain=9.9),
    )

    breakdown = build_session_breakdown(spec, ckpt.state, package)

    assert breakdown["performance"]["final"]["baseline_tps"] == 100.0
    assert breakdown["performance"]["final"]["final_tps"] == 180.0
    assert breakdown["performance"]["final"]["final_gain"] == 1.8
    assert breakdown["performance"]["final"]["quant_only_gain"] == 1.5
    assert breakdown["performance"]["final"]["measurement_role"] == "final_abba"
    assert breakdown["performance"]["final"]["measurement_attempt"] == 2
    assert breakdown["performance"]["validation_policy"]["policy_version"] == 1


def test_final_summary_counts_only_active_retained_patches(tmp_path):
    spec, ckpt = populated_checkpoint(tmp_path)
    retained_patch = tmp_path / "retained.diff"
    retained_patch.write_text("diff --git a/kernel.py b/kernel.py\n")
    ckpt.state["retain_trials"].append(
        {
            "kernel_id": "k-retained",
            "patch": str(retained_patch),
            "decision": "KEEP",
            "reason": "improved_past_keep_floor",
            "active": False,
            "final_decision": "REVERTED",
        }
    )
    ckpt.state["retention_stack"] = [
        {
            "kind": "patch",
            "patch": str(retained_patch),
            "active": False,
        }
    ]
    ckpt.state["performance_measurements"].append(
        {
            "role": "final_abba",
            "baseline_tps": 100.0,
            "final_tps": 200.0,
            "gain": 2.0,
        }
    )
    package = DeployPackage(
        status="perf_below_target",
        quant_ckpt_dir=ckpt.state["quant_ckpt_dir"],
        perf=PerfResult(patches=[], gain=1.5),
        applied_patches=[str(retained_patch)],
    )

    breakdown = build_session_breakdown(spec, ckpt.state, package)
    final = build_final_summary(breakdown)
    markdown = render_session_markdown(breakdown)

    assert breakdown["final"]["retained_patches"] == []
    assert final["optimization"]["patches_retained"] == 0
    assert "| Initial decision | Final decision |" in markdown
    assert "| KEEP | REVERTED |" in markdown
    assert breakdown["final"]["performance"]["baseline_tps"] == 100.0
    assert breakdown["final"]["performance"]["final_tps"] == 200.0
    assert breakdown["final"]["performance"]["final_gain"] == 2.0
    assert breakdown["final"]["performance"]["measurement_role"] == "final_abba"


@patch(
    "quark.experimental.torch.quant_perf.reporting.service.subprocess.run",
    side_effect=subprocess.TimeoutExpired("git", 30),
)
def test_repository_probe_timeout_does_not_break_report(mock_run, tmp_path):
    spec, ckpt = populated_checkpoint(tmp_path)
    framework_repo = tmp_path / "vllm"
    kernel_repo = tmp_path / "aiter"
    framework_repo.mkdir()
    kernel_repo.mkdir()
    spec = replace(
        spec,
        framework_repo=str(framework_repo),
        kernel_repo=str(kernel_repo),
    )

    breakdown = build_session_breakdown(
        spec,
        ckpt.state,
        DeployPackage(status="perf_below_target"),
    )

    assert breakdown["repositories"]["framework"]["path"] == str(framework_repo)
    assert breakdown["repositories"]["framework"]["head"] == ""
    assert breakdown["repositories"]["kernel"]["path"] == str(kernel_repo)


def test_repository_report_uses_retained_branch_sha_after_worktree_removal(
    tmp_path,
):
    repo = tmp_path / "vllm"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    (repo / "module.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "add", "module.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
    source_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "branch", "quark-quant-perf-session"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "quark-quant-perf-session"], cwd=repo, check=True, capture_output=True)
    (repo / "module.py").write_text("VALUE = 2\n")
    subprocess.run(["git", "commit", "-am", "optimized"], cwd=repo, check=True, capture_output=True)
    final_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "checkout", "main"], cwd=repo, check=True, capture_output=True)

    spec, ckpt = populated_checkpoint(tmp_path / "session")
    spec = replace(spec, framework_repo=str(repo))
    ckpt.state["repo_workspaces"] = {
        "framework": {
            "source_repo": str(repo),
            "integration_path": "/tmp/removed/worktree",
            "original_branch": "main",
            "work_branch": "quark-quant-perf-session",
            "base_sha": source_head,
            "final_sha": final_sha,
            "branch_retained": True,
            "status": "removed",
        }
    }
    ckpt.state["cleanup"] = {"status": "complete", "removed": [], "errors": []}

    breakdown = build_session_breakdown(spec, ckpt.state, DeployPackage(status="success"))

    framework = breakdown["repositories"]["framework"]
    assert framework["head"] == final_sha
    assert framework["source_head"] == source_head
    assert framework["branch_retained"] is True
    assert framework["workspace_status"] == "removed"
    assert breakdown["cleanup"]["status"] == "complete"


def test_final_performance_without_final_abba_uses_quant_only_measurement(
    tmp_path,
):
    spec, ckpt = populated_checkpoint(tmp_path)
    ckpt.state["retain_trials"].append(
        {
            "kernel_id": "k-win",
            "throughput_tps": 5100.0,
            "gain": 5100.0 / 3200.0,
            "decision": "KEEP",
            "reason": "improved_past_keep_floor",
        }
    )

    breakdown = build_session_breakdown(
        spec,
        ckpt.state,
        DeployPackage(
            status="success",
            perf=PerfResult(["/p.diff"], 5100.0 / 3200.0),
            applied_patches=["/p.diff"],
        ),
    )

    final_perf = breakdown["final"]["performance"]
    assert final_perf["quantized_tps"] == 4800.0
    assert final_perf["quant_only_gain"] == 1.5
    assert final_perf["final_tps"] == 4800.0
    assert final_perf["final_gain"] == 1.5
    assert final_perf["measurement_role"] == "quant_only"
    assert breakdown["attribution"]["kernel_incremental_multiplier"] == 1.0
