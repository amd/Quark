#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for PerfOpt's resumable per-kernel GEAK ledger
(_load_geak_resume_state), which lets a mid-loop interruption skip kernels
already attempted instead of re-running the whole (multi-hour) loop."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from quark.experimental.torch.quant_perf.orchestration.orchestrator import Orchestrator
from quark.experimental.torch.quant_perf.perfopt.bottleneck_analysis import (
    BottleneckAnalysisResult,
    BottleneckMode,
)
from quark.experimental.torch.quant_perf.perfopt.journey import (
    load_geak_resume_state as _load_geak_resume_state,
)
from quark.experimental.torch.quant_perf.perfopt.journey import (
    record_bottleneck_analysis as _record_bottleneck_analysis,
)
from quark.experimental.torch.quant_perf.perfopt.journey import (
    record_kernel_attempt as _record_kernel_attempt,
)
from quark.experimental.torch.quant_perf.perfopt.keep import make_kernel_id
from quark.experimental.torch.quant_perf.perfopt.service import (
    _resolve_quant_ckpt_dir,
)


def _ckpt(state: dict):
    return SimpleNamespace(state=state)


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _save_candidate(tmp_path, entry):
    patch = _write(tmp_path / "final_patch.diff", "diff\n")
    entry["best_patch"] = str(patch)
    validation = {"correctness": "pass", "validation_status": "accepted"}
    speedup = entry.get("verified_speedup")
    if speedup is not None:
        validation.update(
            director_verified_speedup_geomean=speedup,
            per_case=[{"baseline_ms": speedup, "optimized_ms": 1.0}],
            timing_receipt={"all_primed": True, "timer_unprimed": False},
        )
    _write(
        tmp_path / "result.json",
        json.dumps(
            {
                "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                "round_evaluation": {"correctness": {"result": validation}},
            }
        ),
    )


def test_empty_ledger_is_fresh_run():
    assert _load_geak_resume_state(_ckpt({})) == (set(), [])
    assert _load_geak_resume_state(_ckpt({"geak_patches": []})) == (set(), [])


@pytest.mark.parametrize("legacy", [False, True])
def test_template_kernels_keep_independent_journeys_on_rediscovery(legacy):
    from quark.experimental.torch.quant_perf.perfopt.journey import record_kernel_skip

    names = [f"void ck::kernel_moe<{kind}>(ck::Gemm<{kind}>::Argument)" for kind in ("Scale", "Weight", "Expert")]
    ckpt = SimpleNamespace(state={"kernel_journey": []}, save=lambda: None)
    if legacy:
        ckpt.state["kernel_journey"] = [{"kernel_id": "Argument)", "name": name} for name in names]
    analysis = _analysis([{"op_name": name} for name in names])
    _record_bottleneck_analysis(ckpt, analysis)
    ids = [row["kernel_id"] for row in ckpt.state["bottleneck_analysis"]["candidates"]]
    assert len(set(ids)) == 3
    for index, kernel_id in enumerate(ids):
        record_kernel_skip(ckpt, kernel_id, f"reason {index}")
    _record_bottleneck_analysis(ckpt, analysis)
    assert [(row["name"], row["skip_reason"]) for row in ckpt.state["kernel_journey"]] == [
        (name, f"reason {index}") for index, name in enumerate(names)
    ]


def test_external_checkpoint_path_is_used_for_geak_context():
    spec = SimpleNamespace(quant_ckpt_dir="/session/quant_ckpt")
    ckpt = _ckpt({"quant_ckpt_dir": "/source-session/quant_ckpt"})

    assert _resolve_quant_ckpt_dir(spec, ckpt) == ("/source-session/quant_ckpt")


def test_attempted_sigs_include_kept_and_rejected(tmp_path):
    ledger = {
        "geak_patches": [
            {"kernel_sig": "fused_moe", "status": "rejected", "best_patch": "", "verified_speedup": 0.0},
            {
                "kernel_sig": "topk",
                "status": "candidate",
                "best_patch": "/p/topk.diff",
                "verified_speedup": 1.07,
                "kernel_src": "vllm/x/topk.py",
                "kernel_repo": "/repo/vllm",
            },
        ]
    }
    _save_candidate(tmp_path, ledger["geak_patches"][1])
    attempted, kept = _load_geak_resume_state(_ckpt(ledger))
    # both attempted kernels are skipped on resume (don't re-run the rejected one)
    assert attempted == {"fused_moe", "topk"}
    # only the kept one seeds results, with the keys aggregate_gain/PerfResult +
    # _apply_patches need (incl. kernel_src for patch rebase on resume).
    assert kept == [
        {
            "best_patch": str(tmp_path / "final_patch.diff"),
            "verified_speedup": 1.07,
            "micro_speedup_source": "director",
            "kernel_name": "",
            "kernel_src": "vllm/x/topk.py",
            "kernel_repo": "/repo/vllm",
        }
    ]


def test_kept_without_patch_is_not_seeded():
    # defensive: a candidate entry with no patch path can't be applied, so it must
    # not enter the reconstructed results (would break PerfResult.patches).
    ledger = {"geak_patches": [{"kernel_sig": "k", "status": "candidate", "best_patch": "", "verified_speedup": 1.2}]}
    attempted, kept = _load_geak_resume_state(_ckpt(ledger))
    assert attempted == {"k"}
    assert kept == []


def test_e2e_dropped_patch_is_attempted_but_not_seeded_on_resume():
    state = {
        "geak_patches": [
            {
                "kernel_sig": "mfma_moe1",
                "status": "candidate",
                "best_patch": "/p/mfma.diff",
                "verified_speedup": 1.05,
                "kernel_src": "/aiter/mixed_moe.py",
                "kernel_repo": "/aiter",
            }
        ],
        "kernel_journey": [
            {
                "kernel_id": "mfma_moe1",
                "e2e": {"decision": "DROP", "reason": "below_keep_floor"},
            }
        ],
    }

    attempted, kept = _load_geak_resume_state(_ckpt(state))

    assert attempted == {"mfma_moe1"}
    assert kept == []


def test_e2e_adopted_patch_is_already_in_retained_branch_not_seeded():
    state = {
        "geak_patches": [
            {
                "kernel_sig": "mfma_moe1",
                "status": "candidate",
                "best_patch": "/p/mfma.diff",
                "verified_speedup": 1.05,
                "kernel_src": "/aiter/mixed_moe.py",
                "kernel_repo": "/aiter",
            }
        ],
        "kernel_journey": [
            {
                "kernel_id": "mfma_moe1",
                "e2e": {"decision": "KEEP", "validated": True},
                "outcome": "adopted",
            }
        ],
    }

    attempted, kept = _load_geak_resume_state(_ckpt(state))

    assert attempted == {"mfma_moe1"}
    assert kept == []


def test_e2e_retryable_fault_reuses_patch_without_rerunning_geak(tmp_path):
    state = {
        "geak_patches": [
            {
                "kernel_sig": "kernel_gemm_0.kd",
                "status": "candidate",
                "best_patch": "/p/kernel.diff",
                "verified_speedup": None,
                "micro_speedup_source": "unmeasured_verified_patch",
                "kernel_src": "/aiter/kernel.py",
                "kernel_repo": "/aiter",
            }
        ],
        "kernel_journey": [
            {
                "kernel_id": "kernel_gemm_0.kd",
                "outcome": "deferred",
                "e2e": {"decision": "RETRYABLE_FAULT"},
            }
        ],
    }

    _save_candidate(tmp_path, state["geak_patches"][0])
    attempted, kept = _load_geak_resume_state(_ckpt(state))

    assert attempted == {"kernel_gemm_0.kd"}
    assert kept[0]["best_patch"] == str(tmp_path / "final_patch.diff")
    assert kept[0]["verified_speedup"] is None


def test_retryable_e2e_fault_is_pending_on_terminal_resume():
    state = {
        "kernel_journey": [
            {
                "kernel_id": "kernel_gemm_0.kd",
                "outcome": "deferred",
                "backend_attempts": [{"backend": "geak"}],
                "e2e": {"decision": "RETRYABLE_FAULT"},
            }
        ]
    }

    assert Orchestrator._has_pending_kernel_candidates(state)


def _analysis(candidates: list[dict]) -> BottleneckAnalysisResult:
    return BottleneckAnalysisResult(
        requested_mode=BottleneckMode.DIFFERENTIAL,
        effective_mode="differential",
        status="completed",
        reason="",
        candidates=tuple(candidates),
        quantized_trace="/trace/quantized.json.gz",
        baseline_trace="/trace/baseline.json.gz",
        trace_kind="raw",
    )


def test_record_bottleneck_analysis_initializes_kernel_journey():
    saved = []
    ckpt = SimpleNamespace(
        state={
            "bottleneck_analysis": {},
            "kernel_journey": [],
            "phase_timeline": [],
        },
        save=lambda: saved.append(True),
    )
    bottlenecks = [
        {
            "op_name": "mfma_moe1.kd",
            "kernel_time_us": 777.0,
            "baseline_time_us": 0.0,
            "differential_type": "unique_to_quantized",
            "roofline_bound": "MEMORY_BOUND",
            "parent_op_name": "aiter::fused_moe_",
            "parent_op_names": ["aiter::fused_moe_"],
            "call_count": 10,
            "shape_cases": [[[]]],
            "dtypes": ["BFloat16"],
        }
    ]

    _record_bottleneck_analysis(ckpt, _analysis(bottlenecks))

    candidate = ckpt.state["bottleneck_analysis"]["candidates"][0]
    journey = ckpt.state["kernel_journey"][0]
    assert candidate["rank"] == 1
    assert candidate["kernel_id"] == make_kernel_id("mfma_moe1.kd")
    assert candidate["kernel_time_us"] == 777.0
    assert candidate["parent_op_name"] == "aiter::fused_moe_"
    assert candidate["call_count"] == 10
    assert journey["kernel_id"] == candidate["kernel_id"]
    assert journey["outcome"] == "selected"
    assert journey["backend_attempts"] == []
    assert ckpt.state["phase_timeline"][-1]["action"] == ("bottleneck_analysis")
    assert saved


@pytest.mark.parametrize("status", ["rejected", "incomplete"])
def test_record_kernel_attempt_updates_source_backend_and_outcome(status):
    saved = []
    ckpt = SimpleNamespace(
        state={
            "phase_timeline": [],
            "kernel_journey": [
                {
                    "kernel_id": "mfma_moe1.kd",
                    "name": "mfma_moe1.kd",
                    "backend_attempts": [],
                    "e2e": {},
                    "outcome": "selected",
                }
            ],
        },
        save=lambda: saved.append(True),
    )

    _record_kernel_attempt(
        ckpt,
        kernel_id=make_kernel_id("mfma_moe1.kd"),
        source_file="/aiter/mixed_moe.py",
        source_repo="/aiter",
        source_reason="FlyDSL symbol",
        report={
            "candidate_status": status,
            "execution_status": "timed_out" if status == "incomplete" else "completed",
            "timed_out": status == "incomplete",
            "verified_speedup": 0.99,
            "best_patch": "",
            "knowledge_ids": ["kernel.flydsl.productionization.v1"],
            "round_evaluation": {
                "compilation": {"success": True},
                "correctness": {"success": True},
            },
        },
        kept=False,
    )

    journey = ckpt.state["kernel_journey"][0]
    assert journey["source_mapping"]["source_file"] == "/aiter/mixed_moe.py"
    assert journey["backend_attempts"][0]["backend"] == "geak"
    assert journey["backend_attempts"][0]["attempt"] == 1
    assert journey["backend_attempts"][0]["ts"]
    assert journey["backend_attempts"][0]["compile_passed"] is True
    assert journey["backend_attempts"][0]["correctness_passed"] is True
    assert journey["backend_attempts"][0]["micro_speedup"] == 0.99
    assert journey["backend_attempts"][0]["knowledge_ids"] == ["kernel.flydsl.productionization.v1"]
    assert journey["outcome"] == status
    assert journey["backend_attempts"][0]["timed_out"] is (status == "incomplete")
    assert not Orchestrator._has_pending_kernel_candidates(ckpt.state)
    assert ckpt.state["phase_timeline"][-1]["action"] == "geak"
    assert ckpt.state["phase_timeline"][-1]["status"] == status
    assert saved


def test_record_bottleneck_analysis_preserves_prior_kernel_history():
    ckpt = SimpleNamespace(
        state={
            "bottleneck_analysis": {},
            "phase_timeline": [],
            "kernel_journey": [
                {
                    "kernel_id": "old-kernel",
                    "name": "old-kernel",
                    "backend_attempts": [{"backend": "geak"}],
                    "e2e": {"decision": "REJECTED"},
                    "outcome": "rejected",
                }
            ],
        },
        save=lambda: None,
    )

    _record_bottleneck_analysis(
        ckpt,
        _analysis(
            [
                {
                    "op_name": "new-kernel",
                    "kernel_time_us": 10.0,
                    "baseline_time_us": 0.0,
                    "differential_type": "unique_to_quantized",
                    "roofline_bound": "UNKNOWN",
                }
            ]
        ),
    )

    by_id = {row["kernel_id"]: row for row in ckpt.state["kernel_journey"]}
    assert set(by_id) == {"old-kernel", make_kernel_id("new-kernel")}
    assert by_id["old-kernel"]["backend_attempts"] == [{"backend": "geak"}]


def test_vendor_skip_is_not_pending_after_resolver_upgrade():
    state = {
        "kernel_journey": [
            {
                "kernel_id": "Cijk_vendor",
                "outcome": "skipped",
                "skip_reason": "vendor library kernel",
                "source_mapping": {"retryable": False},
                "backend_attempts": [],
            }
        ]
    }

    assert not Orchestrator._has_pending_kernel_candidates(state)


def test_resume_rebases_retained_kernel_source_to_current_worktree(tmp_path):
    active_repo = tmp_path / "active-aiter"
    active_source = _write(
        active_repo / "aiter" / "ops" / "flydsl" / "kernels" / "kernel.py",
        "def kernel():\n    pass\n",
    )
    state = {
        "geak_patches": [
            {
                "kernel_sig": "kernel",
                "status": "candidate",
                "best_patch": "/patches/kernel.diff",
                "verified_speedup": 1.05,
                "kernel_src": ("/tmp/old-worktree/aiter/ops/flydsl/kernels/kernel.py"),
                "kernel_repo": "/tmp/old-worktree",
            }
        ]
    }

    _save_candidate(tmp_path, state["geak_patches"][0])
    _, kept = _load_geak_resume_state(
        _ckpt(state),
        active_repos=[str(active_repo)],
    )

    assert kept[0]["kernel_src"] == str(active_source)
    assert kept[0]["kernel_repo"] == str(active_repo)
    assert kept[0]["kernel_relpath"] == ("aiter/ops/flydsl/kernels/kernel.py")
