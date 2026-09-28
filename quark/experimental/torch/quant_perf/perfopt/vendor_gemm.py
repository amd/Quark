#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Optional deterministic vendor GEMM tuning before source-level GEAK work.

The backend follows the useful parts of Hyperloom's Forge integration:
measured shapes in, isolated artifacts out, runtime environment as a candidate,
and no trust in microbenchmarks until Quark Quant-Perf performs its own accuracy and
production-throughput validation.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.session.spec import Spec

VENDOR_TUNER_CAPABILITY_VERSION = 4
_TUNABLEOP_SHAPE_RE = re.compile(
    r"^(?:nn|nt|tn|tt)_(\d+)_(\d+)_(\d+)(?:_|$)",
    re.IGNORECASE,
)

_FORGE_OPTIONAL_INT_FIELDS = (
    "moe_intermediate_size",
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "num_experts_per_tok",
    "num_selected_experts",
    "top_k",
    "v_head_dim",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
)


@dataclass(frozen=True)
class _VendorLane:
    key: str
    precision: str
    quant_type: str
    tuner: str = ""


_FP8_BLOCKSCALE_LANE = _VendorLane(
    "fp8_blockscale",
    "fp8",
    "blockscale",
)
_A4W4_LANE = _VendorLane(
    "a4w4_blockscale",
    "fp4",
    "mxfp4",
    "a4w4_blockscale",
)
_W4A8_LANE = _VendorLane(
    "w4a8_tunableop",
    "bf16",
    "none",
    "vllm_dense_tunableop",
)
_VLLM_TUNABLEOP_LANE = _VendorLane(
    "vllm_tunableop",
    "bf16",
    "none",
    "vllm_dense_tunableop",
)


def _classify_vendor_lane(
    bottleneck: dict[str, Any],
) -> _VendorLane | None:
    evidence = " ".join(
        str(value)
        for value in (
            bottleneck.get("op_name"),
            bottleneck.get("device_kernel_name"),
            bottleneck.get("parent_op_name"),
            *(bottleneck.get("kernel_names") or []),
            *(bottleneck.get("dtypes") or []),
        )
        if value
    ).lower()
    if any(marker in evidence for marker in ("a8wfp4", "w4a8")):
        return _W4A8_LANE
    if any(
        marker in evidence
        for marker in (
            "f8bs",
            "a8w8",
            "fp8",
            "float8",
            "e4m3",
            "e5m2",
        )
    ):
        return _FP8_BLOCKSCALE_LANE
    if any(marker in evidence for marker in ("a4w4", "mxfp4", "fp4", "float4")):
        return _A4W4_LANE
    return None


def vendor_result_needs_run(
    result: dict[str, Any] | None,
) -> bool:
    if not result:
        return True
    if int(result.get("capability_version") or 0) < (VENDOR_TUNER_CAPABILITY_VERSION):
        return True
    return (
        result.get("status") == "failed" and bool(result.get("retryable")) and int(result.get("attempt_count") or 0) < 2
    )


def _normalize_forge_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    normalized = copy.deepcopy(config)
    changed: list[str] = []
    containers = [("", normalized)]
    for name in ("text_config", "language_config", "llm_config"):
        value = normalized.get(name)
        if isinstance(value, dict):
            containers.append((name, value))
    for prefix, container in containers:
        for field in _FORGE_OPTIONAL_INT_FIELDS:
            if field in container and container[field] is None:
                container[field] = 0
                changed.append(f"{prefix + '.' if prefix else ''}{field}")
    return normalized, changed


@contextmanager
def _forge_model_view(
    model_dir: str | Path,
    run_dir: str | Path,
) -> Iterator[tuple[str, dict[str, Any]]]:
    model_dir = Path(model_dir).resolve()
    run_dir = Path(run_dir).resolve()
    config_path = model_dir / "config.json"
    metadata = {
        "used": False,
        "source_model": str(model_dir),
        "normalized_fields": [],
        "source_config_sha256": "",
    }
    try:
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
    except (OSError, json.JSONDecodeError):
        yield str(model_dir), metadata
        return
    metadata["source_config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    normalized, changed = _normalize_forge_config(config)
    metadata["normalized_fields"] = changed
    if not changed:
        yield str(model_dir), metadata
        return

    view_dir = run_dir / "model_view"
    shutil.rmtree(view_dir, ignore_errors=True)
    view_dir.mkdir(parents=True, exist_ok=True)
    try:
        for source in model_dir.iterdir():
            if source.name == "config.json":
                continue
            os.symlink(
                source,
                view_dir / source.name,
                target_is_directory=source.is_dir(),
            )
        (view_dir / "config.json").write_text(json.dumps(normalized, indent=2, sort_keys=True))
        metadata.update(
            {
                "used": True,
                "overlay_path": str(view_dir),
            }
        )
        yield str(view_dir), metadata
    finally:
        shutil.rmtree(view_dir, ignore_errors=True)


def extract_gemm_shapes(
    bottlenecks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return unique valid M/N/K shapes weighted by profiled kernel time."""
    weights: dict[tuple[int, int, int], float] = {}
    for bottleneck in bottlenecks:
        shapes = list(bottleneck.get("gemm_shapes") or [])
        if not shapes:
            singular = bottleneck.get("gemm_shape") or {}
            if singular:
                shapes = [singular]
        valid: list[tuple[int, int, int]] = []
        for shape in shapes:
            try:
                key = (
                    int(shape["M"]),
                    int(shape["N"]),
                    int(shape["K"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if min(key) > 0:
                valid.append(key)
        if not valid:
            continue
        weight = float(bottleneck.get("kernel_time_us") or 0.0) / len(valid)
        for key in valid:
            weights[key] = weights.get(key, 0.0) + weight
    return [{"M": m, "N": n, "K": k, "weight": weights[(m, n, k)]} for m, n, k in sorted(weights)]


def resolve_forge_command() -> list[str] | None:
    """Resolve the optional Forge GEMM tuner without making it a dependency."""
    override = os.environ.get("QUARK_QUANT_PERF_FORGE_GEMM_TUNE", "").strip()
    if override.lower() in {"0", "off", "false", "disabled"}:
        return None
    if override:
        return [override]
    executable = shutil.which("forge-gemm-tune")
    if executable:
        return [executable]
    try:
        if importlib.util.find_spec("forge_gemm_tune") is not None:
            return [sys.executable, "-m", "forge_gemm_tune.cli"]
    except (ModuleNotFoundError, ValueError):
        pass
    return None


def _model_hidden_size(model_dir: str | Path) -> int | None:
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    candidates = [config]
    if isinstance(config.get("text_config"), dict):
        candidates.append(config["text_config"])
    for candidate in candidates:
        value = candidate.get("hidden_size")
        if isinstance(value, int) and value > 0:
            return value
    return None


def resolve_untuned_csv(
    kernel_repo: str | Path,
    precision: str,
    model_dir: str | Path,
) -> str:
    """Return a non-empty AITER shape CSV matching the model hidden size."""
    names = {
        "mxfp4": "a4w4_blockscale_untuned_gemm.csv",
        "fp4": "a4w4_blockscale_untuned_gemm.csv",
        "fp8": "a8w8_blockscale_untuned_gemm.csv",
    }
    name = names.get(precision)
    if not name or not kernel_repo:
        return ""
    path = Path(kernel_repo) / "aiter" / "configs" / name
    if not path.is_file():
        return ""
    try:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return ""
    if not rows:
        return ""
    hidden = _model_hidden_size(model_dir)
    if hidden is None:
        return str(path)
    for row in rows:
        try:
            if int(float(row.get("K", ""))) == hidden:
                return str(path)
        except (TypeError, ValueError):
            continue
    return ""


def _tunableop_evidence(
    tuner_rows: list[dict[str, Any]],
    best_speedup: float,
) -> str:
    loadable_rows = []
    for row in tuner_rows:
        if row.get("tuner") != "vllm_dense_tunableop" or not row.get("candidate"):
            continue
        artifact = str(row.get("artifact") or "")
        if artifact and Path(artifact).is_file():
            loadable_rows.append(row)
    if not loadable_rows:
        return ""
    measured_rows = [row for row in loadable_rows if int(row.get("total_shapes") or 0) > 0]
    if not measured_rows:
        return "unmeasured"
    if best_speedup <= 1.0 and all(
        int(row.get("improved_shapes") or 0) == 0 and float(row.get("best_micro_speedup") or 1.0) <= 1.0
        for row in measured_rows
    ):
        return "measured_no_gain"
    return "measured"


def _candidate_from_result(
    result: dict[str, Any],
    lane: str = "",
) -> list[dict[str, Any]]:
    if str(result.get("micro_decision") or "").lower() != "candidate":
        return []
    tuner_rows = [row for row in (result.get("tuners_run") or []) if isinstance(row, dict)]
    best_speedup = float(result.get("best_speedup") or 1.0)
    tunableop_evidence = _tunableop_evidence(
        tuner_rows,
        best_speedup,
    )
    if tunableop_evidence == "measured_no_gain":
        return []
    if (
        tuner_rows
        and best_speedup <= 1.0
        and all(int(row.get("improved_shapes") or 0) == 0 for row in tuner_rows)
        and tunableop_evidence != "unmeasured"
    ):
        return []
    runtime_env = result.get("recommended_env") or result.get("extra_envs")
    if not isinstance(runtime_env, dict) or not runtime_env:
        return []
    normalized = {str(key): str(value) for key, value in runtime_env.items() if key and value}
    if not normalized:
        return []
    for key, value in normalized.items():
        if "CONFIG" in key and not Path(value).is_file():
            return []
    artifacts = result.get("artifacts")
    candidate = {
        "name": (f"forge_vendor_gemm:{lane}" if lane else "forge_vendor_gemm"),
        "runtime_env": normalized,
        "artifacts": dict(artifacts) if isinstance(artifacts, dict) else {},
        "micro_speedup": best_speedup,
    }
    if tunableop_evidence == "unmeasured":
        candidate["requires_screen"] = True
    return [candidate]


def _is_tunableop_untuned_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("Validator"):
        return False
    fields = [field.strip() for field in stripped.split(",")]
    if len(fields) < 2 or "TunableOp" not in fields[0] or not fields[1]:
        return False
    dimensions = [int(value) for value in re.findall(r"\d+", fields[1])]
    return sum(value > 0 for value in dimensions) >= 3


def merge_tunableop_untuned_files(base_path: Path) -> int:
    rows: list[str] = []
    seen: set[str] = set()
    pattern = f"{base_path.stem}*{base_path.suffix}"
    for path in sorted(base_path.parent.glob(pattern)):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except OSError:
            continue
        for line in lines:
            row = line.strip()
            if _is_tunableop_untuned_row(row) and row not in seen:
                seen.add(row)
                rows.append(row)
    if not rows:
        return 0
    temporary = base_path.with_suffix(f"{base_path.suffix}.tmp")
    try:
        temporary.write_text("\n".join(rows) + "\n")
        os.replace(temporary, base_path)
    except OSError:
        temporary.unlink(missing_ok=True)
        return 0
    return len(rows)


def filter_tunableop_signatures(
    source: str | Path,
    output: str | Path,
    *,
    shapes: list[dict[str, Any]],
    scaled_only: bool = False,
) -> int:
    """Keep native TunableOp rows matching trace-selected M/N/K shapes."""
    selected = {
        (int(shape["M"]), int(shape["N"]), int(shape["K"]))
        for shape in shapes
        if all(shape.get(key) for key in ("M", "N", "K"))
    }
    if not selected:
        return 0
    source = Path(source)
    output = Path(output)
    kept = []
    try:
        lines = source.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        return 0
    for line in lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        if scaled_only and not fields[0].startswith("ScaledGemmTunableOp"):
            continue
        match = _TUNABLEOP_SHAPE_RE.match(fields[1])
        if not match:
            continue
        shape = tuple(int(value) for value in match.groups())
        if shape not in selected and (shape[1], shape[0], shape[2]) not in selected:
            continue
        kept.append(line.strip())
    if not kept:
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(dict.fromkeys(kept)) + "\n")
    return len(dict.fromkeys(kept))


def _parse_stdout(stdout: str) -> dict[str, Any]:
    match = re.search(
        r"FORGE_GEMM_TUNE_RESULT_BEGIN\s*\n(.*?)\n"
        r"FORGE_GEMM_TUNE_RESULT_END",
        stdout or "",
        re.DOTALL,
    )
    if not match:
        return {}
    try:
        value = json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _run_forge_lane(
    *,
    command: list[str],
    lane: _VendorLane,
    bottlenecks: list[dict[str, Any]],
    spec: Spec,
    lane_dir: Path,
    model_path: str,
    tunableop_input: str,
    timeout_s: int,
) -> dict[str, Any]:
    shutil.rmtree(lane_dir, ignore_errors=True)
    lane_dir.mkdir(parents=True, exist_ok=True)
    shapes = extract_gemm_shapes(bottlenecks)
    shapes_path = lane_dir / "shapes.json"
    if shapes:
        shapes_path.write_text(json.dumps({"shapes": shapes}, indent=2))
    untuned_csv = ""
    if not shapes and not tunableop_input and spec.framework != "vllm":
        untuned_csv = resolve_untuned_csv(
            getattr(spec, "active_kernel_repo", ""),
            lane.precision,
            spec.quant_ckpt_dir,
        )
    cmd = [
        *command,
        "run",
        "--model-path",
        model_path,
        "--framework",
        str(spec.framework),
        "--precision",
        lane.precision,
        "--quant-type",
        lane.quant_type,
        "--gpu-type",
        str(spec.gpu_type),
        "--tp",
        str(spec.tp),
        "--conc",
        str(spec.bench_concurrency),
        "--output-dir",
        str(lane_dir),
        "--timeout",
        str(timeout_s),
        "--global-timeout",
        str(timeout_s),
        "--skip-gpu-check",
        "--gpu-ids",
        ",".join(str(getattr(spec, "gpu_id", 0) + index) for index in range(spec.tp)),
    ]
    if lane.tuner:
        cmd.extend(["--tuner", lane.tuner])
    if tunableop_input:
        cmd.extend(["--tunableop-input", tunableop_input])
    elif shapes:
        cmd.extend(["--shapes-json", str(shapes_path)])
    elif untuned_csv:
        cmd.extend(["--untuned-csv", untuned_csv])
    env = {
        **os.environ,
        **dict(getattr(spec, "runtime_env", {}) or {}),
        "ROCR_VISIBLE_DEVICES": ",".join(str(getattr(spec, "gpu_id", 0) + index) for index in range(spec.tp)),
    }
    try:
        process = subprocess.run(
            cmd,
            cwd=lane_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "lane": lane.key,
            "precision": lane.precision,
            "quant_type": lane.quant_type,
            "tuner": lane.tuner,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "retryable": True,
            "candidates": [],
            "shape_count": len(shapes),
        }

    result_path = lane_dir / "result.json"
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            result = {}
    else:
        result = _parse_stdout(process.stdout)
    if not isinstance(result, dict):
        result = {}
    candidates = _candidate_from_result(result, lane.key)
    return {
        "lane": lane.key,
        "precision": lane.precision,
        "quant_type": lane.quant_type,
        "tuner": lane.tuner,
        "status": ("candidate" if candidates else ("failed" if process.returncode else "no_improvement")),
        "reason": result.get("skip_reason") or result.get("error") or "",
        "retryable": False,
        "candidates": candidates,
        "shape_count": len(shapes),
        "returncode": process.returncode,
        "stdout_tail": (process.stdout or "")[-2000:],
        "stderr_tail": (process.stderr or "")[-2000:],
    }


def run_vendor_gemm_tuning(
    bottlenecks: list[dict[str, Any]],
    spec: Spec,
    run_dir: str | Path,
    *,
    timeout_s: int = 1800,
    attempt_count: int = 1,
) -> dict[str, Any]:
    """Run Forge when available and return untrusted runtime-env candidates."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    command = resolve_forge_command()
    if command is None:
        return {
            "status": "skipped",
            "reason": "tuner_unavailable",
            "candidates": [],
            "lanes": [],
            "retryable": False,
            "attempt_count": attempt_count,
            "capability_version": VENDOR_TUNER_CAPABILITY_VERSION,
        }

    grouped: dict[_VendorLane, list[dict[str, Any]]] = {}
    signature_bottlenecks: list[dict[str, Any]] = []
    tunableop_inputs: set[str] = set()
    lane_results: list[dict[str, Any]] = []
    for bottleneck in bottlenecks:
        lane = _classify_vendor_lane(bottleneck)
        shapes = extract_gemm_shapes([bottleneck])
        uses_tunableop = spec.framework == "vllm" and (
            str(bottleneck.get("op_name") or "").startswith("Cijk")
            or str(bottleneck.get("parent_op_name") or "") in {"aten::mm", "aten::_scaled_mm", "aten::addmm"}
            or lane == _W4A8_LANE
        )
        if uses_tunableop:
            tunableop_input = str(bottleneck.get("tunableop_input") or "")
            if not shapes or not tunableop_input:
                reason = (
                    "missing_exact_gemm_shape_evidence" if not shapes else "missing_trace_selected_tunableop_signature"
                )
                lane_results.append(
                    {
                        "lane": _VLLM_TUNABLEOP_LANE.key,
                        "status": "skipped",
                        "reason": reason,
                        "retryable": False,
                        "candidates": [],
                        "shape_count": len(shapes),
                        "scope": "trace_selected",
                    }
                )
                continue
            signature_bottlenecks.append(bottleneck)
            tunableop_inputs.add(tunableop_input)
            continue
        if lane is None:
            lane_results.append(
                {
                    "lane": "unknown",
                    "status": "skipped",
                    "reason": "ambiguous_precision",
                    "retryable": False,
                    "candidates": [],
                    "shape_count": len(shapes),
                }
            )
            continue
        dense_backend = getattr(
            spec,
            "mxfp4_gemm_backend",
            "triton",
        )
        if lane == _A4W4_LANE and dense_backend in {"flydsl", "asm"}:
            lane_results.append(
                {
                    "lane": lane.key,
                    "status": "skipped",
                    "reason": (
                        "unsupported_flydsl_dense_backend"
                        if dense_backend == "flydsl"
                        else "unsupported_asm_dense_backend"
                    ),
                    "retryable": False,
                    "candidates": [],
                    "shape_count": len(shapes),
                }
            )
            continue
        grouped.setdefault(lane, []).append(bottleneck)

    tunableop_input = ""
    if signature_bottlenecks:
        if len(tunableop_inputs) == 1:
            tunableop_input = next(iter(tunableop_inputs))
            grouped[_VLLM_TUNABLEOP_LANE] = signature_bottlenecks
        else:
            lane_results.append(
                {
                    "lane": _VLLM_TUNABLEOP_LANE.key,
                    "status": "skipped",
                    "reason": "ambiguous_tunableop_shape_evidence",
                    "retryable": False,
                    "candidates": [],
                    "shape_count": 0,
                    "scope": "trace_selected",
                }
            )

    with _forge_model_view(spec.quant_ckpt_dir, run_dir) as (
        forge_model_path,
        model_view,
    ):
        for lane, lane_bottlenecks in grouped.items():
            lane_results.append(
                _run_forge_lane(
                    command=command,
                    lane=lane,
                    bottlenecks=lane_bottlenecks,
                    spec=spec,
                    lane_dir=run_dir / "lanes" / lane.key,
                    model_path=forge_model_path,
                    tunableop_input=(tunableop_input if lane == _VLLM_TUNABLEOP_LANE else ""),
                    timeout_s=timeout_s,
                )
            )
    overlay_path = str(model_view.get("overlay_path") or "")
    model_view["cleaned"] = not overlay_path or not Path(overlay_path).exists()
    (run_dir / "model_view_manifest.json").write_text(json.dumps(model_view, indent=2))

    candidates = [candidate for lane_result in lane_results for candidate in lane_result.get("candidates") or []]
    if candidates:
        status = "candidate"
    elif any(row.get("status") == "failed" for row in lane_results):
        status = "failed"
    elif lane_results and all(row.get("status") == "skipped" for row in lane_results):
        status = "skipped"
    else:
        status = "no_improvement"
    reasons = list(dict.fromkeys(str(row.get("reason") or "") for row in lane_results if row.get("reason")))
    normalized = {
        "status": status,
        "reason": "; ".join(reasons),
        "candidates": candidates,
        "lanes": lane_results,
        "model_view": model_view,
        "backend": "forge",
        "scope": "trace_selected",
        "retryable": any(bool(row.get("retryable")) for row in lane_results),
        "attempt_count": attempt_count,
        "capability_version": VENDOR_TUNER_CAPABILITY_VERSION,
    }
    (run_dir / "quark_quant_perf_result.json").write_text(json.dumps(normalized, indent=2))
    return normalized
