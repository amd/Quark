#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quant-Perf CLI: parse intent, preflight dependencies, and run/report sessions."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

from quark.experimental.torch.quant_perf import config
from quark.experimental.torch.quant_perf.orchestration.spec_factory import build_spec_from_args
from quark.experimental.torch.quant_perf.runtime.backends import (
    configure_runtime_env as _configure_runtime_env,
)
from quark.experimental.torch.quant_perf.session.spec import (
    Checkpoint,
    DeployPackage,
    SessionLock,
    Spec,
    StageError,
)
from quark.experimental.torch.quant_perf.session.state import SessionState


def _resolved_config_summary(spec: Spec) -> dict[str, object]:
    """Return the user-facing choices that materially affect a run."""
    return {
        "model": spec.model_dir,
        "base_model": spec.base_model,
        "framework": spec.framework,
        "gpu": {
            "type": spec.gpu_type,
            "first_device": spec.gpu_id,
            "tensor_parallel_size": spec.tp,
        },
        "workload": {
            "input_tokens": spec.isl,
            "output_tokens": spec.osl,
            "benchmark_concurrency": spec.bench_concurrency,
        },
        "quantization": {
            "path": "direct_ptq" if spec.quant_strategy else "mixed_precision_search",
            "strategy": spec.quant_strategy,
            "layer_precision_candidates": spec.layer_precision_candidates or "auto",
            "kv_cache_precision_candidates": spec.kv_cache_precision_candidates,
            "max_search_candidates": spec.max_search_candidates,
            "search_timeout_s": spec.search_timeout_s,
            "search_gpu_memory_utilization_override": spec.search_gpu_memory_utilization,
            "search_moe_backend": spec.effective_search_moe_backend,
        },
        "evaluation": {
            "search_gsm8k_num_samples": spec.effective_search_gsm8k_num_samples,
            "accuracy_gsm8k_num_samples": spec.gsm8k_num_samples,
            "accuracy_gap": spec.accuracy_gap,
            "inference_moe_backend": spec.effective_inference_moe_backend,
            "kv_cache_dtype": spec.vllm_kv_cache_dtype,
        },
        "optimization": {
            "performance_mode": spec.effective_performance_mode,
            "target_gain": spec.target_gain,
            "geak_direction_budget": spec.geak_direction_budget,
            "workspace_source": spec.workspace_source,
            "tracelens_gpu_arch_json": spec.tracelens_gpu_arch_json or None,
        },
        "session_dir": spec.session_dir,
    }


def _print_resolved_config(spec: Spec) -> None:
    print("[Quark Quant-Perf] Resolved configuration:", file=sys.stderr)
    print(json.dumps(_resolved_config_summary(spec), indent=2), file=sys.stderr)


def _prepend_pythonpath(repo: str) -> None:
    if not repo:
        return
    resolved = str(Path(repo).resolve())
    entries = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    entries = [p for p in entries if str(Path(p).resolve()) != resolved]
    os.environ["PYTHONPATH"] = os.pathsep.join([resolved, *entries])
    sys.path[:] = [
        resolved,
        *[p for p in sys.path if str(Path(p or ".").resolve()) != resolved],
    ]


def _probe_python_import(
    root: str,
    module: str,
    *,
    label: str,
) -> subprocess.CompletedProcess[str]:
    if not root:
        print(
            f"\n[Quark Quant-Perf preflight] FAILED: {label} root is not configured.\n",
            file=sys.stderr,
        )
        raise SystemExit(1)
    resolved = str(Path(root).resolve())
    if not Path(resolved).is_dir():
        print(
            f"\n[Quark Quant-Perf preflight] FAILED: {label} root does not exist: {resolved}\n",
            file=sys.stderr,
        )
        raise SystemExit(1)
    existing = os.environ.get("PYTHONPATH", "")
    env = {
        **os.environ,
        "PYTHONPATH": (f"{resolved}{os.pathsep}{existing}" if existing else resolved),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {module} as imported; print(imported.__file__)",
        ],
        cwd=resolved,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    if result.returncode != 0:
        print(
            f"\n[Quark Quant-Perf preflight] FAILED: cannot import {module} from {resolved}:\n{result.stderr[-800:]}\n",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return result


def _preflight_atom() -> None:
    root = config.atom_root()
    _probe_python_import(
        root,
        "atom.entrypoints.openai_server",
        label="Atom (experimental; set ATOM_ROOT)",
    )


def _preflight_framework_imports(framework_repo: str) -> None:
    """Probe whether the custom vllm at framework_repo can be imported cleanly.

    Runs in a subprocess so the probe cannot pollute the main process's module
    cache.  On failure, prints an actionable fix command and raises SystemExit
    so the user finds out before spending time on quantization.
    """
    # Use PYTHONPATH to put framework_repo first so its vllm takes precedence
    # over any system-installed vllm -- sys.path.insert in a -c probe still
    # lets the system package win when it was already found in sys.path.
    import os as _os
    import subprocess as _sp
    import sys as _sys

    _existing_pypath = _os.environ.get("PYTHONPATH", "")
    probe_env = {
        **_os.environ,
        "PYTHONPATH": f"{framework_repo}:{_existing_pypath}" if _existing_pypath else framework_repo,
    }
    probe = "import vllm.model_executor.layers.quantization"
    result = _sp.run(
        [_sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
        env=probe_env,
    )
    if result.returncode == 0:
        return

    stderr = result.stderr
    # Extract the missing module name from the traceback for a targeted message.
    missing = None
    for line in stderr.splitlines():
        if "ModuleNotFoundError: No module named" in line:
            missing = line.split("No module named")[-1].strip().strip("'\"")
            break
        if "ImportError:" in line:
            missing = line.split("ImportError:")[-1].strip()
            break

    if missing:
        # Map known modules to the pip package that provides them.
        _pkg_map = {
            "compressed_tensors": "compressed-tensors",
        }
        pkg = next(
            (v for k, v in _pkg_map.items() if k in missing),
            missing.split(".")[0].replace("_", "-"),
        )
        print(
            f"\n[Quark Quant-Perf preflight] FAILED: '{framework_repo}' requires "
            f"'{missing}' which is not importable.\n"
            f"  Fix: pip install --upgrade '{pkg}'\n"
            "  Then re-run quark-quant-perf.\n",
            file=_sys.stderr,
        )
    else:
        print(
            f"\n[Quark Quant-Perf preflight] FAILED: importing vllm from '{framework_repo}' "
            f"raised an error:\n{stderr[-800:]}\n",
            file=_sys.stderr,
        )
    raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    from quark.experimental.torch.quant_perf.evaluation.cli import add_eval_profile_args

    p = argparse.ArgumentParser(
        prog="quark-quant-perf",
        description="Quantize an LLM checkpoint and optionally measure or optimize inference performance.",
    )
    # -- required --
    p.add_argument("--model", required=True, dest="model_dir", help="path to the model to quantize")
    p.add_argument("--base-model", help="baseline (unquantized) model; defaults to --model")
    p.add_argument(
        "--framework",
        default="vllm",
        choices=["vllm", "atom"],
        help="inference framework: vllm (supported, default) or atom (experimental; requires ATOM_ROOT)",
    )
    p.add_argument("--gpu-type", default="mi355x", choices=["mi300x", "mi325x", "mi350x", "mi355x"])
    p.add_argument("--isl", type=int, default=1024)
    p.add_argument("--osl", type=int, default=1024)

    # -- quantization control --
    p.add_argument(
        "--quant-strategy",
        default=None,
        help=(
            "direct_ptq: quantization intent as a free-form string — can be a short scheme name "
            "(e.g. 'fp8', 'mxfp4') or a natural-language description "
            "(e.g. 'FP8 weights with INT8 activations, skip embedding layer'). "
            "Passed to Quark's quark-torch-ptq skill which interprets it and selects "
            "the concrete configuration. Omit for mix_precision_search automatic search."
        ),
    )
    p.add_argument("--accuracy-gap", type=float, default=0.02)
    p.add_argument(
        "--performance-mode",
        choices=["off", "measure", "optimize"],
        default=None,
        help=(
            "performance work after the accuracy gate: off skips throughput and "
            "optimization (default), measure records throughput only, and optimize "
            "runs TraceLens/GEAK when the target is missed"
        ),
    )
    p.add_argument(
        "--target-gain",
        type=float,
        default=None,
        help="desired throughput multiplier; supplying this implies --performance-mode optimize",
    )
    p.add_argument("--kv-cache-scheme", default=None, choices=[None, "fp8"])
    p.add_argument("--exclude-layers", nargs="*", default=None)
    p.add_argument(
        "--file2file-export",
        action="store_true",
        help="export mixed-precision candidates shard by shard through Quark's file2file API; "
        "skips calibration-dependent candidates and preserves compatible completed search results. "
        "Enabled automatically for packed MXFP4 source weights unsupported by standard export "
        "or models exceeding the estimated GPU memory budget",
    )
    p.add_argument(
        "--max-search-candidates",
        type=int,
        default=20,
        help="maximum mixed-precision candidates to evaluate; "
        "set 0 to exhaust the generated search space without early stopping",
    )
    p.add_argument(
        "--search-timeout",
        type=float,
        default=43200.0,
        help="wall-clock timeout in seconds for isolated mixed-precision search and export",
    )
    p.add_argument(
        "--baseline-floor",
        type=float,
        default=0.03,
        help="min baseline (unquantized) GSM8K for the run to be meaningful; "
        "below it the base model is broken in the framework and the run aborts "
        "with base_unhealthy. Set 0 to disable.",
    )
    p.add_argument(
        "--fix-base-framework",
        action="store_true",
        help="opt-in: allow bounded runtime recovery and framework repair for an "
        "unhealthy BASE model before aborting (default: fail fast).",
    )
    p.add_argument(
        "--recheck-baseline",
        action="store_true",
        help="ignore an exact cached baseline failure and run the health gate again",
    )
    p.add_argument(
        "--retry-accuracy-gate",
        action="store_true",
        help="reopen a failed session at the real quantized accuracy gate; "
        "if the saved checkpoint is missing, re-export the persisted "
        "search winner without repeating mixed-precision search",
    )
    p.add_argument(
        "--retry-perfopt",
        action="store_true",
        help="reopen a terminal perf_failed session at bottleneck discovery; "
        "requires reusable accuracy and quant-only throughput evidence",
    )
    p.add_argument(
        "--layer-precision-candidates",
        action="extend",
        nargs="+",
        default=None,
        metavar="MODE",
        help="mix_precision_search: quantization modes to search for attention/MLP layers "
        "(e.g. native fp8 ptpc_fp8 mxfp4 mxfp4_fp8). "
        "The native mode is always included as the mixed-precision fallback. "
        "Default (omit): auto = every scheme the GPU supports except mxfp6_e2m3 "
        "(gfx942: native/fp8/ptpc_fp8; gfx950: + mxfp4/mxfp4_fp8).",
    )
    p.add_argument(
        "--kv-cache-precision-candidates",
        action="extend",
        nargs="+",
        default=None,
        metavar="MODE",
        help="mix_precision_search: KV cache quantization modes to search (e.g. native fp8). Defaults to: native fp8",
    )

    # -- accuracy evaluation --
    p.add_argument(
        "--search-gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM GPU memory fraction for mixed-precision search only, in (0, 1]; "
        "overrides shared --gpu-memory-utilization. Defaults to the shared value if supplied, otherwise 0.75. "
        "Leaves headroom for calibration and fake quantization; OOM stops search with advice.",
    )
    p.add_argument(
        "--search-gsm8k-num-samples",
        type=int,
        default=None,
        help="GSM8K samples per mixed-precision search candidate; "
        "defaults to --gsm8k-num-samples for backward compatibility",
    )
    p.add_argument(
        "--gsm8k-num-samples",
        type=int,
        default=1319,
        help="GSM8K samples for the baseline and exported-checkpoint accuracy gate",
    )
    add_eval_profile_args(p)
    p.add_argument("--vllm-extra-arg", dest="vllm_extra_args", action="append", default=[])
    p.add_argument(
        "--search-moe-backend",
        choices=["auto", "triton", "triton_unfused", "aiter", "aiter_mxfp4_bf16", "emulation"],
        default=None,
        help="MoE backend for search only (default: auto, intersect plugin a1/a2 QDQ adapters with vLLM "
        "support for each layer's activation, source weight format and parallel configuration). "
        "Legacy --moe-backend passthrough applies when omitted.",
    )
    p.add_argument(
        "--inference-moe-backend",
        choices=["auto", "triton", "aiter", "aiter_mxfp4_bf16", "emulation"],
        default=None,
        help="MoE backend for real accuracy, throughput, and serving (default: auto). "
        "Auto enables AITER for MXFP4 MoE and architectures requiring it. "
        "Legacy --moe-backend passthrough applies when omitted.",
    )
    p.add_argument(
        "--mxfp4-moe-backend",
        choices=["aiter", "triton", "flydsl"],
        default=os.environ.get("QUARK_QUANT_PERF_MXFP4_MOE_BACKEND", "aiter"),
        help="MXFP4 MoE kernel backend for the real GSM8K gate / throughput "
        "benchmark / serving. 'aiter' (default) = CK kernels; "
        "'triton' = AITER Triton W4A4 kernels (graph-safe, keeps CUDA graphs); "
        "'flydsl' = AITER FlyDSL kernels + CUDA graphs. Only affects "
        "fp4-quantized checkpoints. Defaults to $QUARK_QUANT_PERF_MXFP4_MOE_BACKEND.",
    )
    p.add_argument(
        "--w4a8-gemm-backend",
        choices=["triton", "flydsl"],
        default=os.environ.get("QUARK_QUANT_PERF_W4A8_GEMM_BACKEND", "triton"),
        help="Dense static FP8-per-tensor x MXFP4 GEMM backend. "
        "'triton' preserves the existing path; 'flydsl' uses the "
        "gfx950 AITER FlyDSL kernel for supported layer shapes.",
    )
    p.add_argument(
        "--mxfp4-gemm-backend",
        choices=["triton", "flydsl", "asm"],
        default=os.environ.get("QUARK_QUANT_PERF_MXFP4_GEMM_BACKEND", "triton"),
        help="Dense dynamic MXFP4 x MXFP4 GEMM backend. "
        "'triton' preserves the existing path; 'flydsl' dynamically "
        "quantizes activations and uses the gfx950 FlyDSL A4W4 kernel; "
        "'asm' uses the AITER HIP quantizer and assembly A4W4 GEMM.",
    )
    p.add_argument(
        "--aiter-config-fmoe",
        default=os.environ.get("AITER_CONFIG_FMOE", ""),
        help="AITER tuned FMoE CSV, used by the FlyDSL backend.",
    )

    # -- calibration --
    p.add_argument("--calib-dataset", default="pileval")
    p.add_argument("--num-calib-data", type=int, default=512)
    p.add_argument("--calib-seqlen", type=int, default=512)

    # -- performance optimization control --
    p.add_argument("--top-kernels", type=int, default=5)
    p.add_argument(
        "--bottleneck-mode",
        choices=["differential", "absolute"],
        default="differential",
        help="select quantized-only absolute bottlenecks or compare them against a baseline trace",
    )
    p.add_argument(
        "--tracelens-gpu-arch-json",
        default="",
        help="explicit TraceLens GPU architecture JSON for roofline analysis; "
        "when omitted, resolve the architecture from --gpu-type",
    )
    p.add_argument("--keep-floor", type=float, default=0.01)
    p.add_argument("--framework-repo", default="")
    p.add_argument(
        "--kernel-repo",
        default="",
        help="optional dependency/kernel source repo (for example AITER). "
        "PerfOpt searches and patches it independently of --framework-repo.",
    )
    p.add_argument(
        "--workspace-source",
        choices=["explicit", "auto", "readonly"],
        default="auto",
        help="source workspace policy: explicit uses only supplied Git repos; "
        "auto may discover editable installs or create session-local "
        "package overlays; readonly disables source modification",
    )
    p.add_argument(
        "--bench-concurrency",
        type=int,
        default=64,
        help="concurrent sequences for the orchestrator throughput benchmark (final gain)",
    )

    # -- GEAK control --
    p.add_argument("--geak-model", default=None, help="defaults to config.geak_model()")
    p.add_argument(
        "--geak-direction-budget",
        type=int,
        default=3,
        help="kernel_workflow optimization-direction budget per kernel",
    )
    p.add_argument(
        "--geak-timeout",
        type=float,
        default=5400.0,
        help="per-kernel wall-clock budget (s): run kernel_workflow to completion "
        "within this, else salvage the best result so far",
    )
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument(
        "--server-port",
        type=int,
        default=8080,
        help="OpenAI-compatible server port used by TraceLens/PerfOpt",
    )

    # -- session --
    p.add_argument("--session-dir", default=None, help="defaults to ./runs/<timestamp>")

    return p


build_spec = build_spec_from_args


def _status(argv: list[str]) -> int:
    """`quant_perf status --session <dir>`: read-only poll of progress.json,
    no new protocol needed (IMPL_SPEC §4.8)."""
    from quark.experimental.torch.quant_perf.session.progress import read_progress

    p = argparse.ArgumentParser(prog="quark-quant-perf status")
    p.add_argument("--session", required=True, dest="session_dir")
    args = p.parse_args(argv)

    progress = read_progress(args.session_dir)
    if progress is None:
        print(f"no progress.json found under {args.session_dir}", file=sys.stderr)
        return 1
    print(json.dumps(progress, indent=2))
    return 0


def _partial_spec_from_state(
    session_dir: Path,
    state: SessionState,
) -> Spec:
    data = state.get("run_spec") or {}
    if data:
        data = dict(data)
        data["session_dir"] = str(session_dir)
        return Spec.from_dict(
            data,
            runtime_context=dict(state.get("runtime_context") or {}),
        )
    return Spec(
        model_dir="",
        base_model="",
        framework="",
        gpu_type="",
        gpu_arch="",
        isl=0,
        osl=0,
        quant_strategy=None,
        session_dir=str(session_dir),
    )


def _terminal_report_package(ckpt: Checkpoint) -> DeployPackage | None:
    terminal_stages = {
        "done": "success",
        "perf_failed": "perf_below_target",
        "failed": "accuracy_failed",
        "base_unhealthy": "base_unhealthy",
    }
    stage = ckpt.state.get("terminal_stage") or ckpt.state.get("stage")
    if stage not in terminal_stages:
        return None
    terminal = ckpt.state.get("terminal_result")
    if terminal:
        return DeployPackage.from_dict(terminal)
    return DeployPackage(
        status=terminal_stages[stage],
        quant_ckpt_dir=ckpt.state.get("quant_ckpt_dir") or "",
    )


def _regenerate_final_report(
    session_dir: Path,
    ckpt: Checkpoint,
    package: DeployPackage,
) -> int:
    from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore
    from quark.experimental.torch.quant_perf.knowledge.terminal_experience import (
        TerminalExperienceRecorder,
    )
    from quark.experimental.torch.quant_perf.reporting.service import write_final_artifacts
    from quark.experimental.torch.quant_perf.session.progress import write_progress

    spec = _partial_spec_from_state(session_dir, ckpt.state)
    try:
        with ExperienceStore(config.experience_store_path()) as experience_store:
            summary = TerminalExperienceRecorder(experience_store).capture(
                spec=spec,
                state=ckpt.state,
            )
            summary_data = dataclasses.asdict(summary)
            summary_data["reviewable"] = len(
                experience_store.list_reviewable_experience(session_id=str(ckpt.state.get("session_id") or ""))
            )
        ckpt.state["experience_capture"] = summary_data
        ckpt.save()
    except Exception as exc:
        ckpt.state.setdefault("report_warnings", []).append(f"runtime experience capture failed: {exc}")
        ckpt.save()

    try:
        paths = write_final_artifacts(spec, ckpt.state, package)
    except Exception as exc:
        ckpt.state["reporting"] = {
            "status": "failed",
            "paths": {},
            "error": str(exc),
        }
        ckpt.save()
        write_progress(
            session_dir,
            stage=ckpt.state.get("terminal_stage") or ckpt.state.get("stage"),
            report_status="failed",
            warning=f"final report generation failed: {exc}",
        )
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1

    package.report_status = "complete"
    package.report_paths = dict(paths)
    ckpt.state["reporting"] = {
        "status": "complete",
        "paths": dict(paths),
        "error": "",
    }
    ckpt.state["terminal_result"] = dataclasses.asdict(package)
    ckpt.save()
    write_progress(
        session_dir,
        stage=ckpt.state.get("terminal_stage") or ckpt.state.get("stage"),
        stage_detail="final reports generated",
        report_status="complete",
        reports=dict(paths),
    )
    print(json.dumps({"status": "complete", "paths": paths}, indent=2))
    return 0


def _report(argv: list[str]) -> int:
    """Regenerate FINAL-stage artifacts without running GPU work."""
    p = argparse.ArgumentParser(prog="quark-quant-perf report")
    p.add_argument("--session", required=True, dest="session_dir")
    args = p.parse_args(argv)
    session_dir = Path(args.session_dir).resolve()
    if not (session_dir / "state.json").is_file():
        print(f"no state.json found under {session_dir}", file=sys.stderr)
        return 1

    try:
        with SessionLock(session_dir):
            ckpt = Checkpoint.load(session_dir)
            if ckpt is None:
                print(f"no state.json found under {session_dir}", file=sys.stderr)
                return 1
            package = _terminal_report_package(ckpt)
            if package is None:
                stage = ckpt.state.get("stage") or "unknown"
                print(
                    f"session is not terminal (stage={stage}); use `quark-quant-perf status` while it is running",
                    file=sys.stderr,
                )
                return 1
            return _regenerate_final_report(session_dir, ckpt, package)
    except StageError as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _bottlenecks(argv: list[str]) -> int:
    """Analyze existing session traces without changing session state."""
    from quark.experimental.torch.quant_perf.perfopt.bottleneck_analysis import (
        BottleneckMode,
        analyze_existing_session_bottlenecks,
    )
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    parser = argparse.ArgumentParser(prog="quark-quant-perf bottlenecks")
    parser.add_argument("--session", required=True, dest="session_dir")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in BottleneckMode],
        default=BottleneckMode.DIFFERENTIAL.value,
    )
    parser.add_argument("--top-kernels", type=int, default=10)
    args = parser.parse_args(argv)
    session_dir = Path(args.session_dir).resolve()
    checkpoint = Checkpoint.load(session_dir)
    if checkpoint is None:
        print(
            json.dumps(
                {
                    "status": "unavailable",
                    "reason": f"state.json not found under {session_dir}",
                },
                indent=2,
            )
        )
        return 1
    spec = _partial_spec_from_state(session_dir, checkpoint.state)
    try:
        analysis = analyze_existing_session_bottlenecks(
            session_dir,
            mode=args.mode,
            gpu_arch=spec.gpu_arch,
            top_n=args.top_kernels,
            gpu_arch_json_path=spec.tracelens_gpu_arch_json,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "unavailable",
                    "reason": str(exc),
                },
                indent=2,
            )
        )
        return 1
    print(json.dumps(analysis.to_dict(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "status":
        return _status(argv[1:])
    if argv and argv[0] == "report":
        return _report(argv[1:])
    if argv and argv[0] == "bottlenecks":
        return _bottlenecks(argv[1:])
    if argv and argv[0] == "gc":
        from quark.experimental.torch.quant_perf.workspace.cli import run_gc_command

        return run_gc_command(argv[1:])
    if argv and argv[0] == "backend-probe":
        from quark.experimental.torch.quant_perf.orchestration.backend_probe import (
            run_backend_probe_command,
        )

        return run_backend_probe_command(argv[1:])
    if argv and argv[0] == "eval":
        from quark.experimental.torch.quant_perf.evaluation.cli import run_eval_command

        return run_eval_command(
            argv[1:],
            preflight_framework_imports=_preflight_framework_imports,
        )
    if argv and argv[0] == "knowledge":
        from quark.experimental.torch.quant_perf.knowledge.cli import main as knowledge_main

        return knowledge_main(argv[1:])

    # Deferred: intake, orchestration, experience, and optimization modules
    # transitively import CLI helpers, so importing them above would cycle.
    from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore
    from quark.experimental.torch.quant_perf.orchestration import intake
    from quark.experimental.torch.quant_perf.orchestration.orchestrator import Orchestrator
    from quark.experimental.torch.quant_perf.perfopt.service import OptimizationService

    config.load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.framework == "atom":
        _preflight_atom()
    try:
        spec = intake.build_spec(args)
    except ValueError as exc:
        parser.error(str(exc))
    spec = dataclasses.replace(
        spec,
        invocation_argv=["quark-quant-perf", *argv],
    )
    Path(spec.session_dir).mkdir(parents=True, exist_ok=True)

    _configure_runtime_env(spec)
    _prepend_pythonpath(spec.kernel_repo)
    _print_resolved_config(spec)

    optimization_enabled = spec.effective_performance_mode == "optimize" and (
        spec.workspace_source == "auto" or bool(spec.framework_repo or spec.kernel_repo)
    )
    if spec.framework_repo:
        _preflight_framework_imports(spec.framework_repo)

    with ExperienceStore(config.experience_store_path()) as experience_store:
        perfopt = OptimizationService(experience_store) if optimization_enabled else None
        result = Orchestrator(
            experience_store=experience_store,
            perfopt=perfopt,
        ).run(spec)

    print(json.dumps(dataclasses.asdict(result), indent=2))
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
