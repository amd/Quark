#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Runtime helpers for mix-precision search scripts (model loading, vLLM engine args, result display)."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

import torch

from .config import ConfigEvalResult, HardwareTarget, SearchResult, get_partition_mode, normalize_quant_mode
from .model_loading import _transformers_output_recorder_compatibility
from .moe_backend import split_moe_backend_args

logger = logging.getLogger(__name__)

_TP_FLAG_ALIASES = ("--tensor-parallel-size", "--tensor_parallel_size", "--tp", "-tp")


@dataclass(frozen=True)
class _HardwareSearchPolicy:
    """Hardware-specific inputs for the shared roofline search algorithm."""

    gpu_type: str
    anchor_mode: str


_HARDWARE_SEARCH_POLICIES: dict[HardwareTarget, _HardwareSearchPolicy] = {
    HardwareTarget.MI300: _HardwareSearchPolicy(gpu_type="mi300x", anchor_mode="ptpc_fp8"),
    HardwareTarget.MI325: _HardwareSearchPolicy(gpu_type="mi325x", anchor_mode="ptpc_fp8"),
    HardwareTarget.MI355: _HardwareSearchPolicy(gpu_type="mi355x", anchor_mode="mxfp4"),
}


def _get_hardware_search_policy(hardware: HardwareTarget | str) -> _HardwareSearchPolicy:
    """Return the roofline policy for a supported hardware target."""
    if isinstance(hardware, str):
        hardware = HardwareTarget(hardware.lower())
    return _HARDWARE_SEARCH_POLICIES[hardware]


def extract_tp_from_vllm_args(vllm_cli_args: list[str]) -> int:
    """Pull the tensor-parallel degree out of vLLM passthrough args.

    Recognises ``--tensor-parallel-size N``, ``--tp N``, ``-tp N`` and
    their ``=``-joined forms. Returns ``1`` when no TP flag is present
    or the value is unparseable — matches vLLM's own default so callers
    that use the result (e.g. the per-op roofline tok/s estimate)
    don't silently inflate their numbers.
    """
    for i, tok in enumerate(vllm_cli_args):
        for alias in _TP_FLAG_ALIASES:
            if tok == alias and i + 1 < len(vllm_cli_args):
                try:
                    return max(int(vllm_cli_args[i + 1]), 1)
                except ValueError:
                    return 1
            if tok.startswith(alias + "="):
                try:
                    return max(int(tok.split("=", 1)[1]), 1)
                except ValueError:
                    return 1
    return 1


def load_transformers_model(
    model_path: str,
    *,
    torch_dtype: str | torch.dtype = "auto",
    device_map: str | None = None,
) -> torch.nn.Module:
    """Load a HuggingFace model via Quark's get_model helper."""
    from quark.torch.utils.llm import create_model_skeleton, get_model

    if device_map == "meta":
        # QConfig probing needs logical layer shapes, not packed checkpoint
        # storage (e.g. DeepSeek V4's FP4 experts inside HF FP8Experts).
        return create_model_skeleton(model_path, trust_remote_code=True, attn_implementation="eager")

    if isinstance(torch_dtype, torch.dtype):
        dtype_map = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }
        data_type = dtype_map.get(torch_dtype)
        if data_type is None:
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")
    else:
        data_type = torch_dtype

    with _transformers_output_recorder_compatibility():
        model, _ = get_model(
            model_path,
            data_type=data_type,
            device=device_map or "cuda",
            multi_gpu=False,
            multi_device=False,
            attn_implementation="eager",
            trust_remote_code=True,
        )
    return model


def _load_vllm_engine_cli_parser() -> Any:
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:
        from vllm.utils import FlexibleArgumentParser

    try:
        from vllm.engine.arg_utils import AsyncEngineArgs as VllmEngineArgs
    except ImportError:
        from vllm.engine.arg_utils import EngineArgs as VllmEngineArgs

    import argparse as _argparse

    parser = FlexibleArgumentParser(add_help=False)
    maybe_parser = VllmEngineArgs.add_cli_args(parser)
    if maybe_parser is not None:
        parser = maybe_parser

    for action in parser._actions:
        if action.dest != "help":
            action.default = _argparse.SUPPRESS
    return parser


def _validate_moe_backend(vllm_cli_args: list[str]) -> list[str]:
    """Validate arguments; search workers negotiate concrete plugin adapters."""
    split_moe_backend_args(vllm_cli_args)
    return list(vllm_cli_args)


def resolve_search_memory_args(args: list[str], override: float | None = None) -> tuple[list[str], float]:
    """Separate search memory from vLLM args: phase override > passthrough > 0.75.

    Search needs headroom outside vLLM's KV budget for fake-quantization and
    calibration. This default does not apply to serving or formal evaluation.
    """
    remaining = []
    memory = "0.75"
    tokens = iter(args)
    for token in tokens:
        flag, separator, value = token.partition("=")
        if flag.replace("_", "-") == "--gpu-memory-utilization":
            memory = value if separator else next(tokens, "")
        else:
            remaining.append(token)
    try:
        utilization = float(memory) if override is None else float(override)
        if not 0 < utilization <= 1:
            raise ValueError
    except ValueError as error:
        raise ValueError("search gpu-memory-utilization must be finite and in (0, 1]") from error
    return remaining, utilization


def is_out_of_memory(error: BaseException) -> bool:
    """Recognize local and RPC-wrapped allocator OOMs, including exception causes."""
    seen: set[int] = set()
    while id(error) not in seen:
        seen.add(id(error))
        text = f"{type(error).__name__}: {error}".lower()
        if isinstance(error, torch.OutOfMemoryError) or any(
            marker in text for marker in ("outofmemoryerror", "out of memory", "hiperroroutofmemory")
        ):
            return True
        cause = error.__cause__
        if cause is None and not error.__suppress_context__:
            cause = error.__context__
        if cause is None:
            break
        error = cause
    return False


def build_vllm_engine_kwargs(vllm_cli_args: list[str], llm_cls: type) -> dict[str, Any]:
    """Parse extra vLLM CLI args and return kwargs suitable for LLM()."""

    vllm_cli_args, memory = resolve_search_memory_args(vllm_cli_args)
    if not vllm_cli_args:
        return {
            "trust_remote_code": True,
            "enforce_eager": True,
            "gpu_memory_utilization": memory,
        }

    vllm_cli_args = _validate_moe_backend(vllm_cli_args)
    parser = _load_vllm_engine_cli_parser()  # type: ignore[no-untyped-call]
    namespace, remaining = parser.parse_known_args(vllm_cli_args)
    if remaining:
        raise ValueError("Unsupported vLLM engine args for LLM(): " + " ".join(remaining))

    kwargs = vars(namespace)
    llm_signature = inspect.signature(llm_cls.__init__)
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in llm_signature.parameters.values())
    if not accepts_var_kwargs:
        valid_keys = {
            name
            for name, param in llm_signature.parameters.items()
            if name != "self" and param.kind != inspect.Parameter.VAR_POSITIONAL
        }
        unsupported = sorted(set(kwargs) - valid_keys)
        if unsupported:
            raise ValueError(
                "Parsed vLLM args are not accepted by LLM(): "
                + ", ".join(f"--{name.replace('_', '-')}" for name in unsupported)
            )

    kwargs["trust_remote_code"] = True
    kwargs["enforce_eager"] = True
    kwargs["gpu_memory_utilization"] = memory
    return kwargs


def extend_vllm_server_cmd(cmd: list[str], vllm_cli_args: list[str]) -> None:
    """Append validated vLLM args and enforce eager execution."""
    cmd.extend(_validate_moe_backend(vllm_cli_args))
    if "--enforce-eager" not in vllm_cli_args:
        cmd.append("--enforce-eager")


# ---------------------------------------------------------------------------
# Result display helpers
# ---------------------------------------------------------------------------


def _format_metric(value: object) -> str:
    if isinstance(value, int | float):
        return f"{value:.4f}"
    return str(value)


def _numeric_metric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _get_result_partitions(config: dict[str, str]) -> list[str]:
    partition_order = [
        "linear_attn",
        "self_attn",
        "dense_mlp",
        "routed_moe",
        "shared_expert",
        "kv_cache",
        "attention",
    ]
    partitions = [p for p in partition_order if f"{p}_mode" in config]
    if "mlp_mode" in config and not any(p in partitions for p in ("dense_mlp", "routed_moe")):
        partitions.insert(2, "mlp")
    return partitions


def _render_results_table_lines(
    sorted_results: list[ConfigEvalResult],
    metric_name: str,
    best_config: dict[str, str] | None,
    partitions: list[str],
) -> list[str]:
    """Render the accuracy results table (one config per row, by rank)."""
    if not sorted_results:
        return []

    rows: list[list[str]] = []
    for res in sorted_results:
        raw = res.metrics.get(metric_name, "")
        metric_str = f"{float(raw):.4f}" if isinstance(raw, int | float) else str(raw)
        modes = [normalize_quant_mode(str(res.config.get(f"{p}_mode", "native"))) for p in partitions]
        is_best = best_config is not None and res.config == best_config
        rows.append([str(res.rank), metric_str, *modes, "YES" if is_best else "NO"])

    headers = ["Rank", metric_name, *partitions, "is_best_config"]
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) + 2 for i, h in enumerate(headers)]

    def fmt_row(cells: list[str]) -> str:
        return "".join(c.ljust(w) for c, w in zip(cells, widths, strict=True))

    sep = "-" * sum(widths)
    return [sep, fmt_row(headers), sep, *[fmt_row(r) for r in rows], sep]


def _render_metric_dot_plot(
    sorted_results: list[ConfigEvalResult],
    baseline_metric: object,
    metric_name: str,
    best_config: dict[str, str] | None,
) -> list[str]:
    numeric_points = []
    for res in sorted_results:
        metric_val = _numeric_metric(res.metrics.get(metric_name))
        if metric_val is None:
            continue
        numeric_points.append(
            {
                "rank": res.rank,
                "metric": metric_val,
                "is_best": best_config is not None and res.config == best_config,
            }
        )

    if not numeric_points:
        return []

    baseline_val = _numeric_metric(baseline_metric)
    y_values = [point["metric"] for point in numeric_points]
    if baseline_val is not None:
        y_values.append(baseline_val)

    y_min = min(y_values)
    y_max = max(y_values)
    if y_max == y_min:
        pad = max(abs(y_max) * 0.01, 0.001)
    else:
        pad = max((y_max - y_min) * 0.1, 0.001)
    y_min -= pad
    y_max += pad

    rows = 10
    cell_width = max(3, len(str(max(point["rank"] for point in numeric_points))) + 1)
    chart_width = len(numeric_points) * cell_width
    x_positions = [idx * cell_width + cell_width // 2 for idx in range(len(numeric_points))]
    grid = [[" " for _ in range(chart_width)] for _ in range(rows)]

    def _metric_to_row(metric_val: float) -> int:
        if y_max == y_min:
            return rows // 2
        scaled = (y_max - metric_val) / (y_max - y_min)
        return min(rows - 1, max(0, round(scaled * (rows - 1))))

    if baseline_val is not None:
        baseline_row = _metric_to_row(baseline_val)
        for idx in range(chart_width):
            grid[baseline_row][idx] = "-"

    for x_pos, point in zip(x_positions, numeric_points, strict=True):
        marker = "@" if point["is_best"] else "o"
        grid[_metric_to_row(point["metric"])][x_pos] = marker

    axis_width = len(f"{y_max:.4f}")
    chart_lines = [f"{metric_name} dot plot by rank (o=config, @=best, -=baseline)"]
    for row_idx, row in enumerate(grid):
        tick_val = y_max - (y_max - y_min) * row_idx / (rows - 1)
        chart_lines.append(f"{tick_val:>{axis_width}.4f} |{''.join(row)}")

    chart_lines.append(f"{'':>{axis_width}} +{'-' * chart_width}")

    rank_chars = [" " for _ in range(chart_width)]
    for x_pos, point in zip(x_positions, numeric_points, strict=True):
        rank_str = str(point["rank"])
        start = min(chart_width - len(rank_str), max(0, x_pos - len(rank_str) // 2))
        for idx, char in enumerate(rank_str):
            rank_chars[start + idx] = char
    chart_lines.append(f"{'rank':>{axis_width}}  {''.join(rank_chars)}")
    return chart_lines


# Batch sizes (continuous-batching concurrency) the roofline is swept over.
# Mirrors the InferenceX 397B MXFP4 sglang test case
# (``qwen3.5-fp4-mi355x-sglang``: conc-start=4, conc-end=256) expanded with
# the default geometric step size of 2.
ROOFLINE_BATCH_SIZES: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256)

# Batch size used for roofline scoring during search (single representative value).
ROOFLINE_SCORE_BATCH: int = 128


def compute_roofline_scores(
    configs: list[Any],
    *,
    model: Any,
    gpu_type: str,
    num_gpus: int,
    exclude_patterns: list[str] | None,
    isl: int = 8192,
    osl: int = 1024,
) -> list[float]:
    """Compute normalized roofline scores [0, 1] for each config.

    Uses ROOFLINE_SCORE_BATCH (currently batch=128) as the representative decode point.
    Score 1.0 = highest tok/s config, 0.0 = lowest. All-equal configs get 1.0.
    Returns a list parallel to ``configs``; returns [0.0, ...] if roofline
    meta cannot be loaded.
    """
    from .perf_roofline import compute_perf_score, load_model_perf_meta_from_model

    meta = load_model_perf_meta_from_model(model, exclude_patterns=exclude_patterns)
    if meta is None:
        logger.warning("roofline scoring: cannot load model meta; scores default to 0")
        return [0.0] * len(configs)

    raw = []
    for cfg in configs:
        score = compute_perf_score(
            meta,
            cfg,
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            batch=ROOFLINE_SCORE_BATCH,
            isl=isl,
            osl=osl,
        )
        raw.append(score.peak_tok_per_sec)

    lo, hi = min(raw), max(raw)
    if hi == lo:
        return [1.0] * len(raw)
    return [(v - lo) / (hi - lo) for v in raw]


def _render_simple_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a left-justified fixed-width table (header + separator lines)."""
    if not rows:
        return []
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) + 2 for i, h in enumerate(headers)]

    def fmt(cells: list[str]) -> str:
        return "".join(c.ljust(w) for c, w in zip(cells, widths, strict=True))

    sep = "-" * sum(widths)
    return [sep, fmt(headers), sep, *[fmt(r) for r in rows], sep]


def compute_and_display_roofline(
    configs: list[Any],
    *,
    model: Any,
    gpu_type: str,
    num_gpus: int,
    exclude_patterns: list[str] | None,
    score_batch: int = ROOFLINE_SCORE_BATCH,
    batch_sizes: tuple[int, ...] = ROOFLINE_BATCH_SIZES,
    isl: int = 8192,
    osl: int = 1024,
) -> list[float]:
    """Compute the per-op decode roofline for every candidate config, log
    summary and operator detail tables, and return normalized [0, 1] scores
    (parallel to ``configs``).

    Table 1 — peak tok/s per config across ``batch_sizes``, rendered by the
              same ``_compute_roofline_tps_by_batch`` / ``_render_roofline_table_lines``
              helpers used at the end of a run (Rank + one column per batch).
    Table 2 — batch-1 prefill op/memory/aggregate-compute projections; report
              only and never used by the search score.
    Table 3 — the batch used for scoring (``score_batch``), with the qconfig
              mode columns plus final/memory/compute tok/s, bound and normalized
              score at that batch (1.0 = fastest).
    Table 4 — per-op decode breakdown for the fastest config at ``score_batch``.

    Returns ``[0.0, ...]`` when the model meta cannot be loaded.
    """
    if not configs:
        return []

    # Wrap the candidate configs as rank-ordered results (rank = idx + 1) so we
    # can reuse the verified tok/s computation + renderer. Metrics are empty —
    # this runs before any eval.
    results = [
        ConfigEvalResult(config=cfg, metrics={}, relative_change={}, is_valid=True, rank=i + 1)
        for i, cfg in enumerate(configs)
    ]

    # Score the roofline over the display batches plus the scoring batch (so the
    # score column is always available even if score_batch ∉ batch_sizes).
    tps_batches = tuple(sorted(set(batch_sizes) | {score_batch}))
    score_details: dict[tuple[int, int], Any] = {}
    prefill_scores: dict[int, Any] = {}
    roofline_tps = _compute_roofline_tps_by_batch(
        results,
        model_path=None,
        gpu_type=gpu_type,
        num_gpus=num_gpus,
        batch_sizes=tps_batches,
        isl=isl,
        osl=osl,
        model=model,
        exclude_patterns=exclude_patterns,
        score_details=score_details,
        prefill_scores=prefill_scores,
    )
    if not roofline_tps:
        logger.warning("roofline scoring: cannot load model meta; scores default to 0")
        return [0.0] * len(configs)

    # ── Table 1: tok/s per config per batch (identical to the end-of-run table) ──
    logger.info("")
    logger.info(
        "Per-op decode roofline tok/s (aggregate-memory fallback; gpu=%s x %d, isl=%d, osl=%d)",
        gpu_type,
        num_gpus,
        isl,
        osl,
    )
    for line in _render_roofline_table_lines(results, roofline_tps, batch_sizes):
        logger.info(line)

    prefill_lines = _render_prefill_roofline_table_lines(results, prefill_scores)
    if prefill_lines:
        logger.info("")
        logger.info(
            "Prefill roofline tok/s (batch=1; gpu=%s x %d, isl=%d, osl=%d; not used for ranking)",
            gpu_type,
            num_gpus,
            isl,
            osl,
        )
        for line in prefill_lines:
            logger.info(line)

    # ── normalized scores at the scoring batch ──
    raw_score = [roofline_tps[i + 1].get(score_batch, 0.0) for i in range(len(configs))]
    lo, hi = min(raw_score), max(raw_score)
    scores = [1.0] * len(raw_score) if hi == lo else [(v - lo) / (hi - lo) for v in raw_score]

    # ── Table 2: qconfig modes + tok/s + normalized score at the scoring batch ──
    partitions = _get_result_partitions(configs[0])
    logger.info("")
    logger.info("Roofline normalized score (scoring batch bs=%d; 1.0 = fastest)", score_batch)
    headers2 = [
        "Rank",
        *partitions,
        f"tok/s@bs{score_batch}",
        "agg-mem",
        "op",
        "op-mem",
        "op-cmp",
        "selected",
        "score",
    ]
    rows2: list[list[str]] = []
    for i, cfg in enumerate(configs):
        detail = score_details.get((i + 1, score_batch))
        rows2.append(
            [
                str(i + 1),
                *[normalize_quant_mode(str(cfg.get(f"{p}_mode", "native"))) for p in partitions],
                f"{raw_score[i]:.1f}",
                f"{detail.aggregate_memory_tok_per_sec:.1f}" if detail is not None else "—",
                f"{detail.op_roofline_tok_per_sec:.1f}" if detail is not None and detail.uses_op_model else "—",
                f"{detail.mem_tok_per_sec:.1f}" if detail is not None else "—",
                (f"{detail.compute_tok_per_sec:.1f}" if detail is not None and detail.compute_tok_per_sec > 0 else "—"),
                str(detail.selected_source) if detail is not None else "unknown",
                f"{scores[i]:.4f}",
            ]
        )
    for line in _render_simple_table(headers2, rows2):
        logger.info(line)

    fastest_index = max(range(len(raw_score)), key=raw_score.__getitem__)
    fastest_detail = score_details.get((fastest_index + 1, score_batch))
    if fastest_detail is not None and fastest_detail.ops:
        logger.info("")
        logger.info(
            "Per-op decode breakdown for fastest config (rank=%d, bs=%d)",
            fastest_index + 1,
            score_batch,
        )
        for line in _render_op_roofline_lines(fastest_detail.ops):
            logger.info(line)

    return scores


def _find_roofline_start_config(
    configs: list[Any],
    scores: list[float],
    anchor_mode: str,
) -> int:
    """Return the hardware anchor, with deterministic user-config fallbacks.

    The preferred anchor quantizes only the routed-MoE partition (or dense MLP
    for a non-MoE model) with ``anchor_mode``. If ``--search_configs`` removes
    that mode, use the fastest available anchor-partition-only candidate. If no
    such candidate exists, start from the lowest roofline score in the actual
    search space instead of silently assuming index zero has the desired semantics.
    """
    if not configs:
        raise ValueError("roofline search requires at least one candidate config")
    if len(configs) != len(scores):
        raise ValueError(f"config/score length mismatch: {len(configs)} configs, {len(scores)} scores")

    if any("routed_moe_mode" in cfg for cfg in configs):
        anchor_partition = "routed_moe"
    elif any("dense_mlp_mode" in cfg for cfg in configs):
        anchor_partition = "dense_mlp"
    else:
        anchor_partition = "mlp"  # Legacy externally supplied configs.

    def is_anchor_only(cfg: Any, mode: str | None = None) -> bool:
        partition_mode = get_partition_mode(cfg, anchor_partition)
        if mode is not None and partition_mode != normalize_quant_mode(mode):
            return False
        if mode is None and partition_mode == "native":
            return False
        anchor_key = f"{anchor_partition}_mode"
        return all(
            key == anchor_key or not key.endswith("_mode") or normalize_quant_mode(str(value)) == "native"
            for key, value in cfg.items()
        )

    preferred = [i for i, cfg in enumerate(configs) if is_anchor_only(cfg, anchor_mode)]
    if preferred:
        return max(preferred, key=lambda i: (scores[i], i))

    anchor_only = [i for i, cfg in enumerate(configs) if is_anchor_only(cfg)]
    if anchor_only:
        fallback = max(anchor_only, key=lambda i: (scores[i], i))
        logger.warning(
            "Preferred roofline anchor (%s=%s, others=native) is unavailable; "
            "using anchor-only candidate idx=%d instead: %s",
            anchor_partition,
            anchor_mode,
            fallback,
            configs[fallback],
        )
        return fallback

    fallback = min(range(len(configs)), key=lambda i: (scores[i], i))
    logger.warning(
        "No feed-forward-only roofline anchor is available; using lowest-score candidate idx=%d: %s",
        fallback,
        configs[fallback],
    )
    return fallback


def _next_roofline_neighbor(
    *,
    scores: list[float],
    visited: set[int],
    current_idx: int,
    move_higher: bool,
) -> int | None:
    """Return the nearest unvisited roofline neighbour in one direction.

    ``(score, original_index)`` forms a total order so equal-score configs are
    still visited deterministically. Moving higher after a valid evaluation is
    deliberately incremental; choosing the maximum score here would skip the
    accuracy frontier that early-stop is supposed to locate.
    """
    if current_idx < 0 or current_idx >= len(scores):
        raise IndexError(f"current_idx {current_idx} is out of range for {len(scores)} roofline scores")

    ordered = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    current_pos = ordered.index(current_idx)
    step = 1 if move_higher else -1
    pos = current_pos + step
    while 0 <= pos < len(ordered):
        candidate_idx = ordered[pos]
        if candidate_idx not in visited:
            return candidate_idx
        pos += step
    return None


def _next_roofline_candidate(
    *,
    scores: list[float],
    visited: set[int],
    current_idx: int,
    move_higher: bool,
    allow_direction_fallback: bool,
) -> int | None:
    """Return the next candidate while preserving locality across direction changes."""
    next_idx = _next_roofline_neighbor(
        scores=scores,
        visited=visited,
        current_idx=current_idx,
        move_higher=move_higher,
    )
    if next_idx is not None or not allow_direction_fallback:
        return next_idx
    return _next_roofline_neighbor(
        scores=scores,
        visited=visited,
        current_idx=current_idx,
        move_higher=not move_higher,
    )


def _should_stop_search_early(
    *,
    early_stop: bool,
    is_valid: bool,
    current_idx: int,
    scores: list[float],
    visited: set[int],
    has_valid_config: bool,
) -> bool:
    """Return whether an evaluated config should terminate the search.

    An invalid config above an already-valid config closes the accuracy
    frontier and can stop immediately. If no valid config has been found yet,
    the search may stop only after exhausting lower roofline neighbours.
    """
    if not early_stop or is_valid:
        return False
    if has_valid_config:
        return True

    current_key = (scores[current_idx], current_idx)
    return not any(index not in visited and (score, index) < current_key for index, score in enumerate(scores))


def _compute_roofline_tps_by_batch(
    sorted_results: list[ConfigEvalResult],
    *,
    model_path: str | None,
    gpu_type: str,
    num_gpus: int,
    batch_sizes: tuple[int, ...],
    isl: int,
    osl: int,
    model: object | None = None,
    exclude_patterns: list[str] | None = None,
    score_details: dict[tuple[int, int], Any] | None = None,
    prefill_scores: dict[int, Any] | None = None,
) -> dict[int, dict[int, float]]:
    """Score each evaluated config with the per-op decode roofline at
    every batch size.

    ``model`` supplies exact Linear shapes, packed expert Parameters and dtype
    information while honouring ``exclude_patterns``. ``model_path`` is kept
    for diagnostics/backward-compatible call signatures; this helper does not
    reload the checkpoint.

    Returns a ``rank -> {batch -> tok/s}`` map; returns ``{}`` when the
    model meta can't be loaded (so the caller can degrade silently and
    omit the table rather than render bogus values).
    """
    from .perf_roofline import (
        compute_perf_score,
        load_model_perf_meta_from_model,
    )

    meta = None
    if model is not None:
        meta = load_model_perf_meta_from_model(model, exclude_patterns=exclude_patterns)
    if meta is None:
        logger.warning(
            "roofline: cannot load model meta (model=%s, path=%s); table omitted.",
            type(model).__name__ if model is not None else None,
            model_path,
        )
        return {}
    logger.debug(
        "roofline meta (native GiB): self_attn=%.2f linear_attn=%.2f mlp_dense=%.2f "
        "moe_expert=%.2f shared_expert=%.2f other=%.2f",
        meta.self_attn_bytes_native / 2**30,
        meta.linear_attn_bytes_native / 2**30,
        meta.mlp_dense_bytes_native / 2**30,
        meta.moe_expert_bytes_native / 2**30,
        meta.shared_expert_bytes_native / 2**30,
        meta.other_bytes_native / 2**30,
    )
    out: dict[int, dict[int, float]] = {}
    for res in sorted_results:
        per_batch: dict[int, float] = {}
        for batch_index, batch in enumerate(batch_sizes):
            score = compute_perf_score(
                meta,
                res.config,
                gpu_type=gpu_type,
                num_gpus=num_gpus,
                batch=batch,
                isl=isl,
                osl=osl,
                include_prefill=prefill_scores is not None and batch_index == 0,
            )
            per_batch[batch] = score.peak_tok_per_sec
            if score_details is not None:
                score_details[(res.rank, batch)] = score
            if prefill_scores is not None and batch_index == 0:
                prefill_scores[res.rank] = score
        out[res.rank] = per_batch
    return out


def _render_prefill_roofline_table_lines(
    sorted_results: list[ConfigEvalResult],
    prefill_scores: dict[int, Any],
) -> list[str]:
    """Render per-config prefill op/memory/aggregate-compute projections."""
    if not sorted_results or not prefill_scores:
        return []
    rows: list[list[str]] = []
    for res in sorted_results:
        score = prefill_scores.get(res.rank)
        rows.append(
            [
                str(res.rank),
                f"{score.prefill_tok_per_sec:.1f}" if score is not None and score.prefill_tok_per_sec > 0 else "—",
                (
                    f"{score.prefill_mem_tok_per_sec:.1f}"
                    if score is not None and score.prefill_mem_tok_per_sec > 0
                    else "—"
                ),
                (
                    f"{score.prefill_compute_tok_per_sec:.1f}"
                    if score is not None and score.prefill_compute_tok_per_sec > 0
                    else "—"
                ),
                str(score.prefill_bound_kind) if score is not None else "unknown",
            ]
        )
    return _render_simple_table(
        ["Rank", "Op tok/s", "Memory-only tok/s", "Aggregate compute tok/s", "Bound"],
        rows,
    )


def _render_op_roofline_lines(ops: tuple[Any, ...], *, limit: int = 20) -> list[str]:
    """Render the highest-time decode operators from an op roofline result."""
    if not ops:
        return []
    rows = [
        [
            str(op.name),
            str(op.op_type),
            str(op.partition),
            str(op.mode),
            f"{op.weight_bpe:g}/{op.input_bpe:g}/{op.input_compute_bpe:g}/{op.output_bpe:g}",
            f"{op.weight_dtype}/{op.input_dtype}/{op.output_dtype}",
            f"{op.flops / 1e9:.2f}",
            f"{op.bytes_moved / 2**30:.3f}",
            f"{op.ai:.2f}",
            str(op.bound),
            f"{op.pct_time * 100:.2f}",
        ]
        for op in sorted(ops, key=lambda item: item.time_s, reverse=True)[:limit]
    ]
    return _render_simple_table(
        [
            "Op",
            "Type",
            "Partition",
            "Mode",
            "W/Aread/Acompute/O BPE",
            "W/Acompute/O dtype",
            "GFLOPs",
            "GiB",
            "FLOP/B",
            "Bound",
            "Time%",
        ],
        rows,
    )


def _render_roofline_table_lines(
    sorted_results: list[ConfigEvalResult],
    roofline_tps: dict[int, dict[int, float]],
    batch_sizes: tuple[int, ...],
) -> list[str]:
    """Render the per-op roofline tok/s table (one config per row, by
    rank), with one column per batch size."""
    if not sorted_results or not roofline_tps:
        return []

    rows: list[list[str]] = []
    for res in sorted_results:
        per_batch = roofline_tps.get(res.rank, {})
        cells = [str(res.rank)]
        for batch in batch_sizes:
            tps = per_batch.get(batch)
            cells.append(f"{tps:.1f}" if isinstance(tps, int | float) and tps > 0 else "—")
        rows.append(cells)

    headers = ["Rank", *[f"bs={b}" for b in batch_sizes]]
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) + 2 for i, h in enumerate(headers)]

    def fmt_row(cells: list[str]) -> str:
        return "".join(c.ljust(w) for c, w in zip(cells, widths, strict=True))

    sep = "-" * sum(widths)
    return [sep, fmt_row(headers), sep, *[fmt_row(r) for r in rows], sep]


def display_results(
    result: SearchResult,
    metric_name: str,
    *,
    model_path: str | None = None,
    gpu_type: str | None = None,
    num_gpus: int = 1,
    batch_sizes: tuple[int, ...] = ROOFLINE_BATCH_SIZES,
    isl: int = 8192,
    osl: int = 1024,
    model: object | None = None,
    exclude_patterns: list[str] | None = None,
) -> None:
    """Log search results: accuracy table, roofline table, and a dot plot.

    When ``gpu_type`` is provided, a separate per-op roofline tok/s table
    is logged after the accuracy table. It is sorted by rank (same order as
    the accuracy table) with one column per batch size.

    Pass ``model`` (the loaded meta model) plus ``exclude_patterns`` for exact
    operator geometry. ``model_path`` is retained for diagnostics only.
    """
    baseline_metric = result.baseline_metrics.get(metric_name, "N/A")
    logger.info(f"Original model {metric_name}: {_format_metric(baseline_metric)}")
    if result.all_results:
        sorted_results = sorted(result.all_results, key=lambda r: r.rank)
        partitions = _get_result_partitions(sorted_results[0].config)
        for line in _render_results_table_lines(
            sorted_results,
            metric_name,
            result.best_config,
            partitions,
        ):
            logger.info(line)

        if gpu_type and (model is not None or model_path):
            prefill_scores: dict[int, Any] = {}
            roofline_tps = _compute_roofline_tps_by_batch(
                sorted_results,
                model_path=model_path,
                gpu_type=gpu_type,
                num_gpus=num_gpus,
                batch_sizes=batch_sizes,
                isl=isl,
                osl=osl,
                model=model,
                exclude_patterns=exclude_patterns,
                prefill_scores=prefill_scores,
            )
            roofline_lines = _render_roofline_table_lines(
                sorted_results,
                roofline_tps,
                batch_sizes,
            )
            if roofline_lines:
                logger.info("")
                logger.info(
                    f"Per-op decode roofline tok/s (aggregate-memory fallback; "
                    f"gpu={gpu_type} x {num_gpus}, isl={isl}, osl={osl})"
                )
                for line in roofline_lines:
                    logger.info(line)
            prefill_lines = _render_prefill_roofline_table_lines(sorted_results, prefill_scores)
            if prefill_lines:
                logger.info("")
                logger.info(
                    f"Prefill roofline tok/s (batch=1; gpu={gpu_type} x {num_gpus}, "
                    f"isl={isl}, osl={osl}; not used for ranking)"
                )
                for line in prefill_lines:
                    logger.info(line)

        logger.info("")
        for line in _render_metric_dot_plot(sorted_results, baseline_metric, metric_name, result.best_config):
            logger.info(line)
