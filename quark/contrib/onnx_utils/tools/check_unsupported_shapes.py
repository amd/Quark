# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import argparse
from pathlib import Path

from ryzenai_onnx_utils.rai_vaiml.dd_txn_to_vaiml_config import (
    VAIMLTemplateConfig,
    convert,
    read_error_file,
)


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gen-config",
        action="store_true",
        help="Generate the config for VAIML auto txn bin generation",
    )
    parser.add_argument(
        "-e",
        "--input-error-log",
        type=Path,
        help="Path to the input error log containing error messages",
    )
    parser.add_argument(
        "-j",
        "--json",
        type=Path,
        help="Path to the output JSON configuration file",
        default="ops-config.json",
    )
    parser.add_argument("--plugin", help="Name of the generated plugin", default="default_plugin")
    parser.add_argument("--model-type", help="Model type for VAIML config", default="llama2-unified")

    return parser


def rai_vaiml(args: argparse.Namespace) -> None:
    print("==== VAIML CONFIG GENERATION ====")
    if args.gen_config:
        vaiml_config = VAIMLTemplateConfig(args.model_type, args.plugin)
        transactions = read_error_file(args.input_error_log, vaiml_config)
        convert(transactions, args.json, vaiml_config)


if __name__ == "__main__":
    parser = configure_parser()
    args = parser.parse_args()
    rai_vaiml(args)
