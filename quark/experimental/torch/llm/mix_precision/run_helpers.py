#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Runtime helpers for mix-precision search scripts (model loading, vLLM engine args, result display)."""

from __future__ import annotations

import inspect
import logging
from typing import Any

import torch

from .config import ConfigEvalResult, SearchResult, normalize_quant_mode

logger = logging.getLogger(__name__)

__all__ = [
    "load_transformers_model",
    "build_vllm_engine_kwargs",
    "extend_vllm_server_cmd",
    "display_results",
]


def load_transformers_model(
    model_path: str,
    *,
    torch_dtype: str | torch.dtype = "auto",
    device_map: str | None = None,
) -> torch.nn.Module:
    """Load a HuggingFace model via Quark's get_model helper."""
    from quark.torch.utils.llm import get_model

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


def build_vllm_engine_kwargs(vllm_cli_args: list[str], llm_cls: type) -> dict[str, Any]:
    """Parse extra vLLM CLI args and return kwargs suitable for LLM()."""

    if not vllm_cli_args:
        return {
            "trust_remote_code": True,
            "enforce_eager": True,
        }

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
    return kwargs


def extend_vllm_server_cmd(cmd: list[str], vllm_cli_args: list[str]) -> None:
    """Append vLLM args to a server command, ensuring --enforce-eager is present."""
    cmd.extend(vllm_cli_args)
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
    partition_order = ["linear_attn", "self_attn", "mlp", "kv_cache", "attention"]
    return [p for p in partition_order if f"{p}_mode" in config]


def _render_results_table_lines(
    sorted_results: list[ConfigEvalResult],
    metric_name: str,
    best_config: dict[str, str] | None,
    partitions: list[str],
) -> list[str]:
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


def display_results(result: SearchResult, metric_name: str) -> None:
    """Log search results: summary table and a dot plot."""
    baseline_metric = result.baseline_metrics.get(metric_name, "N/A")
    logger.info(f"Original model {metric_name}: {_format_metric(baseline_metric)}")
    if result.all_results:
        sorted_results = sorted(result.all_results, key=lambda r: r.rank)
        partitions = _get_result_partitions(sorted_results[0].config)
        for line in _render_results_table_lines(sorted_results, metric_name, result.best_config, partitions):
            logger.info(line)
        logger.info("")
        for line in _render_metric_dot_plot(sorted_results, baseline_metric, metric_name, result.best_config):
            logger.info(line)
