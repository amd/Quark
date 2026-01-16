# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import ryzenai_onnx_utils.reports as reports


def configure_parser(subparser: argparse._SubParsersAction[Any]) -> argparse.ArgumentParser:
    report_parser: argparse.ArgumentParser = subparser.add_parser("report")

    report_parser.add_argument("report", type=str, help="Type of report to generate")
    report_parser.add_argument("input_path", type=Path, help="Path to input ONNX model")
    report_parser.add_argument(
        "--output-dir",
        default=Path("."),
        type=Path,
        help="Directory to save output files. Defaults to cwd",
    )
    report_parser.add_argument(
        "--summarize",
        action="store_true",
        help="Print the summarized report data",
    )
    report_parser.add_argument(
        "--comparison-path",
        type=Path,
        help="Path to an ONNX model to compare with",
    )
    report_parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="How far around nodes to search for the adjacency report",
    )

    return report_parser


def main(args: argparse.Namespace) -> None:
    report = args.report
    if report == "dd_offload":
        reports.dd_offload.report(args.input_path, args.comparison_path, args.summarize)
    elif report == "op_adjacency":
        reports.op_adjacency.report(args.input_path, args.depth)
    elif report == "subgraph_heuristic":
        reports.subgraph_heuristic.report(args.input_path, args.output_dir)
    else:
        raise ValueError(f"Unknown report: {report}. Supported reports: dd_offload, op_adjacency, subgraph_heuristic")
