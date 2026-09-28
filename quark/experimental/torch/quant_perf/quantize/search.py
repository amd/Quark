#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark public-API adapter for module-level mixed-precision search.

Quark owns candidate generation, Roofline traversal, fake-quant evaluation,
and winner selection.  Quark Quant-Perf owns durable session state, exported-model
accuracy validation, candidate fallback, and the rest of the managed pipeline.
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import traceback
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from quark.experimental.torch.quant_perf.quantize.artifacts import validate_quant_checkpoint
from quark.experimental.torch.quant_perf.session.progress import write_progress
from quark.experimental.torch.quant_perf.session.spec import Checkpoint, Spec, StageError

logger = logging.getLogger(__name__)

_SEARCH_STATE_VERSION = 2
_CANDIDATE_ORDER_SOURCE = "quark_reverse_evaluation_order"

_GEMMA4_NATIVE_EXCLUDES = (
    "*vision_tower*",
    "*embed_vision*",
    "*audio_tower*",
    "*embed_audio*",
    "*router*",
)

_GPU_TYPE_TO_QUARK_HARDWARE = {
    "mi300x": "mi300",
    "mi325x": "mi325",
    "mi350x": "mi355",
    "mi355x": "mi355",
}

_DEFAULT_LAYER_MODES = {
    "mi300": ["native", "fp8", "ptpc_fp8"],
    "mi325": ["native", "fp8", "ptpc_fp8"],
    "mi355": [
        "native",
        "fp8",
        "ptpc_fp8",
        "mxfp4",
        "mxfp4_fp8",
    ],
}


def _mix_precision_api() -> SimpleNamespace:
    """Load only Quark's supported mix-precision public surface."""
    module = importlib.import_module("quark.experimental.torch.mix_precision")
    missing = [name for name in ("MixPrecisionConfig", "MixPrecisionQuantizer") if not hasattr(module, name)]
    if missing:
        raise ImportError("Quark mix_precision public API is missing: " + ", ".join(missing))
    return SimpleNamespace(
        MixPrecisionConfig=module.MixPrecisionConfig,
        MixPrecisionQuantizer=module.MixPrecisionQuantizer,
    )


def _hardware_target(gpu_type: str) -> str:
    try:
        return _GPU_TYPE_TO_QUARK_HARDWARE[gpu_type.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown gpu_type {gpu_type!r}. Known types: {list(_GPU_TYPE_TO_QUARK_HARDWARE)}") from exc


def _resolve_layer_precision_candidates(
    explicit: list[str] | None,
    hardware: str | object,
) -> list[str]:
    if explicit:
        return list(explicit) if "native" in explicit else ["native", *explicit]
    normalized = str(getattr(hardware, "value", hardware)).lower().rstrip("x")
    if normalized in {"mi350", "mi355"}:
        normalized = "mi355"
    try:
        return list(_DEFAULT_LAYER_MODES[normalized])
    except KeyError as exc:
        raise ValueError(f"Unsupported Quark hardware target {hardware!r}") from exc


def _search_runtime_args(spec: Spec | Any) -> list[str]:
    from quark.experimental.torch.mix_precision.run_helpers import resolve_search_memory_args

    args, memory = resolve_search_memory_args(
        spec.search_vllm_args, getattr(spec, "search_gpu_memory_utilization", None)
    )
    args.append(f"--gpu-memory-utilization={memory}")
    has_max_model_len = any(
        token in {"--max-model-len", "--max_model_len"} or token.startswith(("--max-model-len=", "--max_model_len="))
        for token in args
    )
    profile = getattr(spec, "eval_profile", None)
    max_model_len = getattr(profile, "max_model_len", None)
    if not has_max_model_len and max_model_len:
        args.extend(["--max-model-len", str(max_model_len)])
    return args


def _search_failure(error: Exception) -> StageError:
    """Preserve native errors; add conditional guidance only for an explicit OOM."""
    from quark.experimental.torch.mix_precision.run_helpers import is_out_of_memory

    oom = is_out_of_memory(error)
    message = f"{error}\nSearch stopped."
    if oom:
        message += (
            " Check GPU occupancy and the failing allocation in the traceback. "
            "For calibration/evaluation OOM, a new session with lower --search-gpu-memory-utilization "
            "may leave more room for temporary allocations. This does not reduce model-weight storage."
        )
    return StageError(
        "quantize",
        message,
        code="search_out_of_memory" if oom else "search_execution_failed",
        diagnostic=traceback.format_exc(),
    )


def _effective_exclude_patterns(
    spec: Spec | Any,
) -> list[str] | None:
    patterns = list(spec.exclude_layers or [])
    has_quant_perf_additions = bool(patterns) or spec.model_arch == "gemma4"
    if has_quant_perf_additions and "lm_head" not in patterns:
        patterns.append("lm_head")
    if spec.model_arch == "gemma4":
        for pattern in _GEMMA4_NATIVE_EXCLUDES:
            if pattern not in patterns:
                patterns.append(pattern)
    return patterns or None


def _kv_cache_quant(kv_cache_precision_candidates: list[str]) -> bool:
    modes = set(kv_cache_precision_candidates)
    if modes == {"native"}:
        return False
    if modes == {"native", "fp8"}:
        return True
    if modes == {"fp8"}:
        raise ValueError(
            "Quark's current public mix-precision API cannot express an "
            "fp8-only KV-cache search; use native or native+fp8"
        )
    raise ValueError("Unsupported KV-cache search modes: " + ", ".join(sorted(modes)))


def _build_mix_precision_config_kwargs(spec: Spec | Any) -> dict[str, Any]:
    hardware = _hardware_target(spec.gpu_type)
    search_num_samples = getattr(spec, "search_gsm8k_num_samples", None)
    if search_num_samples is None:
        search_num_samples = spec.gsm8k_num_samples
    return {
        "granularity": "module",
        "hardware": hardware,
        "eval_metrics": ["gsm8k"],
        "eval_threshold": 1.0 + float(spec.accuracy_gap),
        "eval_num_samples": int(search_num_samples),
        "eval_max_new_tokens": 256,
        "num_calib_samples": int(spec.num_calib_data),
        "calib_seq_len": int(spec.calib_seqlen),
        "exclude_patterns": _effective_exclude_patterns(spec),
        "max_configs": (None if int(spec.max_search_candidates) == 0 else int(spec.max_search_candidates)),
        "early_stop": False,
        "search_modes": _resolve_layer_precision_candidates(
            spec.layer_precision_candidates,
            hardware,
        ),
        "kv_cache_quant": _kv_cache_quant(spec.kv_cache_precision_candidates),
        "file2file_quantization": bool(getattr(spec, "file2file_export", False)),
    }


def _flydsl_dense_mxfp4_available() -> bool:
    try:
        import aiter
        from aiter.ops.flydsl.utils import is_flydsl_available
    except (ImportError, AttributeError):
        return False
    return bool(
        is_flydsl_available()
        and hasattr(aiter, "flydsl_gemm_a4w4_dynamic")
        and hasattr(aiter, "prepare_flydsl_gemm_mxfp4_weight")
    )


def _validate_runtime_capabilities(
    spec: Spec | Any,
    config_kwargs: dict[str, Any],
) -> None:
    modes = set(config_kwargs["search_modes"])
    if spec.mxfp4_gemm_backend == "flydsl" and "mxfp4" in modes and not _flydsl_dense_mxfp4_available():
        raise StageError(
            "quantize",
            "FlyDSL dense MXFP4 was requested but the active AITER runtime "
            "does not expose its required A4W4 entry points",
            code="missing_flydsl_dense_mxfp4",
        )


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def _serialize_search_result(result: Any) -> dict[str, Any]:
    return {
        "best_config": _plain(result.best_config),
        "all_results": [
            {
                "config": _plain(row.config),
                "metrics": _plain(row.metrics),
                "relative_change": _plain(row.relative_change),
                "is_valid": bool(row.is_valid),
                "rank": int(row.rank),
            }
            for row in result.all_results
        ],
        "baseline_metrics": _plain(result.baseline_metrics),
        "total_configs_evaluated": int(result.total_configs_evaluated),
        "total_configs_available": int(result.total_configs_available),
        "search_time_seconds": float(result.search_time_seconds),
        "granularity": _plain(result.granularity),
        "hardware": _plain(result.hardware),
    }


def _candidate_queue_from_result(result: Any) -> list[dict[str, Any]]:
    best = _plain(result.best_config)
    if not isinstance(best, dict):
        return []

    rows = [
        {
            "config": _plain(row.config),
            "search_metrics": _plain(row.metrics),
            "relative_change": _plain(row.relative_change),
            "quark_rank": int(row.rank),
            "status": "pending",
        }
        for row in result.all_results
        if row.is_valid and isinstance(_plain(row.config), dict)
    ]
    by_config = {_config_key(row["config"]): row for row in rows}
    best_row = by_config.get(
        _config_key(best),
        {
            "config": best,
            "search_metrics": {},
            "relative_change": {},
            "quark_rank": None,
            "status": "pending",
        },
    )
    queue = [best_row]
    seen = {_config_key(best)}
    for row in reversed(rows):
        key = _config_key(row["config"])
        if key in seen:
            continue
        seen.add(key)
        queue.append(row)
    return queue


def _persist_search_progress(
    spec: Spec | Any,
    ckpt: Checkpoint | Any,
    result: Any,
) -> None:
    search_state = _search_state(ckpt.state)
    search_state.update(
        {
            "schema_version": _SEARCH_STATE_VERSION,
            "status": "running",
            "termination_reason": "",
            "result": _serialize_search_result(result),
            "candidate_queue": _candidate_queue_from_result(result),
            "candidate_cursor": 0,
        }
    )
    ckpt.save()
    write_progress(
        spec.session_dir,
        stage="quantize",
        stage_detail=(
            "Quark public mixed-precision search "
            f"({result.total_configs_evaluated}/{result.total_configs_available} candidates evaluated)"
        ),
    )


def _config_key(config: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(config.items()))


def _new_search_state() -> dict[str, Any]:
    return {
        "schema_version": _SEARCH_STATE_VERSION,
        "status": "pending",
        "api": "quark.experimental.torch.mix_precision",
        "config": {},
        "result": {},
        "candidate_queue": [],
        "candidate_cursor": 0,
        "candidate_order_source": _CANDIDATE_ORDER_SOURCE,
        "termination_reason": "",
    }


def _search_state(state: dict[str, Any]) -> dict[str, Any]:
    current = state.get("mix_precision_search")
    if not isinstance(current, dict):
        current = _new_search_state()
        state["mix_precision_search"] = current
    for key, value in _new_search_state().items():
        current.setdefault(key, value)
    return current


def _current_candidate(
    search_state: dict[str, Any],
) -> dict[str, Any] | None:
    queue = search_state.get("candidate_queue") or []
    cursor = int(search_state.get("candidate_cursor") or 0)
    while 0 <= cursor < len(queue) and isinstance(queue[cursor], dict) and queue[cursor].get("status") == "skipped":
        cursor += 1
        search_state["candidate_cursor"] = cursor
    if cursor < 0 or cursor >= len(queue):
        return None
    entry = queue[cursor]
    return entry if isinstance(entry, dict) else None


def has_resumable_search_candidate(state: dict[str, Any]) -> bool:
    """Whether a completed search can export its current persisted candidate."""
    search_state = state.get("mix_precision_search") or {}
    if search_state.get("status") not in {"completed", "partial_timeout"}:
        return False
    entry = _current_candidate(search_state)
    return entry is not None and isinstance(entry.get("config"), dict)


def _prioritize_minimal_change_candidate(
    search_state: dict[str, Any],
    *,
    reference_config: dict[str, Any],
    start_index: int,
) -> None:
    queue = search_state.get("candidate_queue") or []
    pending = [
        (index, entry)
        for index, entry in enumerate(queue[start_index:], start=start_index)
        if isinstance(entry, dict) and entry.get("status") == "pending" and isinstance(entry.get("config"), dict)
    ]
    if not pending:
        return
    selected_index, _ = min(
        pending,
        key=lambda item: (
            sum(
                reference_config.get(key) != item[1]["config"].get(key)
                for key in reference_config.keys() | item[1]["config"].keys()
            ),
            item[0],
        ),
    )
    queue[start_index], queue[selected_index] = queue[selected_index], queue[start_index]


def advance_search_candidate(
    state: dict[str, Any],
    *,
    attempt: dict[str, Any],
    prefer_minimal_change: bool = False,
) -> dict[str, Any] | None:
    """Reject the exported candidate and return the next pending config.

    ``best_candidate`` continues to describe the checkpoint currently stored
    on disk. It is updated only after the next candidate exports successfully.
    """
    search_state = _search_state(state)
    entry = _current_candidate(search_state)
    if entry is None:
        return None
    rejected_config = entry.get("config")
    entry["status"] = "rejected"
    entry["real_gate"] = _plain(attempt)
    next_cursor = int(search_state.get("candidate_cursor") or 0) + 1
    if prefer_minimal_change and isinstance(rejected_config, dict):
        _prioritize_minimal_change_candidate(
            search_state,
            reference_config=rejected_config,
            start_index=next_cursor,
        )
    search_state["candidate_cursor"] = next_cursor
    next_entry = _current_candidate(search_state)
    if next_entry is None:
        return None
    next_config = next_entry.get("config")
    if isinstance(next_config, dict):
        return dict(next_config)
    return None


def accept_current_search_candidate(
    state: dict[str, Any],
    *,
    attempt: dict[str, Any],
) -> None:
    search_state = _search_state(state)
    entry = _current_candidate(search_state)
    if entry is None:
        return
    entry["status"] = "accepted"
    entry["real_gate"] = _plain(attempt)


def _search_accuracy_gap(
    search_state: dict[str, Any],
    entry: dict[str, Any],
) -> float | None:
    baseline = search_state.get("result", {}).get("baseline_metrics", {}).get("gsm8k")
    quantized = (entry.get("search_metrics") or {}).get("gsm8k")
    if not isinstance(baseline, int | float):
        return None
    if not isinstance(quantized, int | float):
        return None
    return max(
        0.0,
        (float(baseline) - float(quantized))
        / max(
            float(baseline),
            1e-9,
        ),
    )


def _prepare_export_quantizer(
    api: SimpleNamespace,
    search_state: dict[str, Any],
    *,
    model_path: str,
    candidate: dict[str, Any],
    quantizer: Any | None,
    file2file_export: bool = False,
) -> Any:
    if quantizer is None:
        config = api.MixPrecisionConfig(**{**search_state["config"], "file2file_quantization": file2file_export})
        quantizer = api.MixPrecisionQuantizer(config)
    quantizer.model_path = model_path
    quantizer.result = SimpleNamespace(best_config=candidate)
    required = quantizer.prepare_export(model_path, file2file_quantization=file2file_export)
    search_state["export"] = {"file2file": required, "reason": quantizer.export_reason}
    return quantizer


def _empty_all_cuda_caches() -> None:
    """Release vLLM search allocations before loading export weights."""
    import gc

    import torch

    gc.collect()
    for device_index in range(torch.cuda.device_count()):
        with torch.cuda.device(device_index):
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    gc.collect()


def _replace_export_directory(
    staging: Path,
    destination: Path,
) -> None:
    backup = destination.with_name(destination.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.replace(backup)
    try:
        staging.replace(destination)
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        if backup.exists():
            backup.replace(destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def _ensure_fallback_export_capacity(
    search_state: dict[str, Any],
    destination: Path,
) -> None:
    """Keep the rejected export unless an atomic fallback export can fit."""
    cursor = int(search_state.get("candidate_cursor") or 0)
    queue = search_state.get("candidate_queue") or []
    if cursor <= 0 or cursor >= len(queue):
        # A first export has no previous checkpoint to preserve and no reliable
        # output-size estimate. Let the exporter report ENOSPC and clean staging.
        return
    current_entry = queue[cursor]
    previous_entry = queue[cursor - 1]
    if (
        not isinstance(current_entry, dict)
        or current_entry.get("status") != "pending"
        or not isinstance(previous_entry, dict)
        or previous_entry.get("status") != "rejected"
    ):
        return
    if not destination.is_dir():
        return
    try:
        existing_bytes = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
        free_bytes = shutil.disk_usage(destination.parent).free
    except OSError:
        return
    reserve_bytes = max(1024**3, existing_bytes // 100)
    required_bytes = existing_bytes + reserve_bytes
    if free_bytes < required_bytes:
        raise StageError(
            "quantize",
            "insufficient free space for atomic fallback export; "
            f"need about {required_bytes / 1024**3:.1f} GiB but only "
            f"{free_bytes / 1024**3:.1f} GiB is available. "
            "The rejected checkpoint was preserved.",
            code="insufficient_storage_for_candidate_fallback",
        )


def _export_current_candidate(
    spec: Spec | Any,
    ckpt: Checkpoint | Any,
    api: SimpleNamespace,
    *,
    quantizer: Any | None = None,
) -> str:
    search_state = _search_state(ckpt.state)
    entry = _current_candidate(search_state)
    if entry is None:
        raise StageError(
            "quantize",
            "all persisted mixed-precision candidates failed the real accuracy gate",
            code="mix_precision_candidates_exhausted",
        )
    candidate = entry.get("config")
    if not isinstance(candidate, dict):
        raise StageError(
            "quantize",
            "persisted mixed-precision candidate is malformed",
            code="invalid_mix_precision_candidate",
        )

    quantizer = _prepare_export_quantizer(
        api,
        search_state,
        model_path=spec.base_model,
        candidate=candidate,
        quantizer=quantizer,
        file2file_export=bool(getattr(spec, "file2file_export", False)),
    )
    ckpt.save()
    if search_state["export"]["file2file"]:
        queue = search_state["candidate_queue"]
        cursor = int(search_state.get("candidate_cursor") or 0)
        try:
            compatible = quantizer.filter_export_configs([row["config"] for row in queue[cursor:]])
        except RuntimeError as error:
            raise StageError("quantize", str(error), code="no_file2file_candidate") from error
        for row in queue[cursor:]:
            if row["config"] not in compatible:
                row.update(status="skipped", skip_reason="incompatible_with_file2file")
        entry = _current_candidate(search_state)
        assert entry is not None
        candidate = entry["config"]
        quantizer.result = SimpleNamespace(best_config=candidate)
    ckpt.save()
    _empty_all_cuda_caches()
    destination = Path(spec.session_dir) / "quant_ckpt"
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    _ensure_fallback_export_capacity(search_state, destination)
    try:
        if getattr(spec, "file2file_export", False):
            quantizer.export_best(str(staging), file2file_quantization=True)
        else:
            quantizer.export_best(str(staging))
        if not staging.is_dir():
            raise RuntimeError(f"Quark export did not create {staging}")
        validation = validate_quant_checkpoint(staging)
        if validation["status"] != "success":
            raise RuntimeError("Quark export produced an invalid checkpoint: " + ", ".join(validation["errors"]))
        _replace_export_directory(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    entry["status"] = "exported"
    ckpt.state["best_candidate"] = dict(candidate)
    ckpt.state["best_accuracy_gap"] = _search_accuracy_gap(
        search_state,
        entry,
    )
    ckpt.state["quant_ckpt_dir"] = str(destination)
    ckpt.save()
    return str(destination)


def run_module_search(
    spec: Spec,
    ckpt: Checkpoint,
) -> str:
    """Run or resume Quark public-API mixed-precision search and export."""
    runtime_args = _search_runtime_args(spec)
    search_state = _search_state(ckpt.state)
    if (
        search_state.get("status") in {"completed", "partial_timeout"}
        and search_state.get("runtime_args") is not None
        and search_state["runtime_args"] != runtime_args
    ):
        raise StageError(
            "quantize",
            "search backend or runtime arguments changed; use a new session for a new search",
            code="search_runtime_changed",
        )
    api = _mix_precision_api()

    if search_state.get("status") not in {"completed", "partial_timeout"}:
        config_kwargs = _build_mix_precision_config_kwargs(spec)
        _validate_runtime_capabilities(spec, config_kwargs)
        search_state.update(
            {
                "schema_version": _SEARCH_STATE_VERSION,
                "status": "running",
                "api": "quark.experimental.torch.mix_precision",
                "config": _plain(config_kwargs),
                "runtime_args": runtime_args,
                "result": {},
                "candidate_queue": [],
                "candidate_cursor": 0,
                "candidate_order_source": _CANDIDATE_ORDER_SOURCE,
                "termination_reason": "",
            }
        )
        ckpt.save()

        config = api.MixPrecisionConfig(**config_kwargs)
        quantizer = api.MixPrecisionQuantizer(config)
        required = quantizer.prepare_export(spec.base_model)
        search_state["export"] = {"file2file": required, "reason": quantizer.export_reason}
        ckpt.save()
        write_progress(
            spec.session_dir,
            stage="quantize",
            stage_detail="Quark public mixed-precision search",
        )

        previous_plugins = os.environ.get("VLLM_PLUGINS")
        os.environ["VLLM_PLUGINS"] = ""
        try:
            result = quantizer.search(
                spec.base_model,
                runtime_args,
                progress_callback=lambda snapshot: _persist_search_progress(
                    spec,
                    ckpt,
                    snapshot,
                ),
            )
        except Exception as error:
            failure = _search_failure(error)
            search_state.update(status="failed", termination_reason=failure.code)
            ckpt.save()
            raise failure from error
        finally:
            if previous_plugins is None:
                os.environ.pop("VLLM_PLUGINS", None)
            else:
                os.environ["VLLM_PLUGINS"] = previous_plugins
            # Keep the available decision when search returns or raises. Engine
            # startup failures may leave it pending until a worker report arrives.
            # runtime_args above remain the user's resumable request.
            resolution = getattr(quantizer, "search_moe_backend_resolution", None)
            if resolution is not None:
                search_state["moe_backend_resolution"] = _plain(resolution)
                search_state["resolved_runtime_args"] = list(quantizer.search_runtime_args)
                ckpt.save()

        queue = _candidate_queue_from_result(result)
        if not queue:
            raise StageError(
                "quantize",
                "Quark mixed-precision search produced no valid candidate",
                code="mix_precision_search_no_candidate",
            )
        search_state.update(
            {
                "status": "completed",
                "result": _serialize_search_result(result),
                "candidate_queue": queue,
                "candidate_cursor": 0,
                "termination_reason": "",
            }
        )
        ckpt.save()
        return _export_current_candidate(
            spec,
            ckpt,
            api,
            quantizer=quantizer,
        )

    cursor = int(search_state.get("candidate_cursor") or 0)
    queue = search_state.get("candidate_queue") or []
    if cursor > 0 and cursor < len(queue):
        previous_entry = queue[cursor - 1]
        previous_gate = previous_entry.get("real_gate") if isinstance(previous_entry, dict) else None
        if (
            isinstance(previous_entry, dict)
            and previous_entry.get("status") == "rejected"
            and isinstance(previous_entry.get("config"), dict)
            and isinstance(previous_gate, dict)
            and previous_gate.get("reason") == "quantized_accuracy_load_failed"
        ):
            _prioritize_minimal_change_candidate(
                search_state,
                reference_config=previous_entry["config"],
                start_index=cursor,
            )
    return _export_current_candidate(spec, ckpt, api)
