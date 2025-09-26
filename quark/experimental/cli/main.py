#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
========================================================
quark-cli - AMD Quark Command Line Interface
========================================================

SYNOPSIS
    quark-cli [SUBCOMMAND] [ARGUMENTS ...] ...
    quark-cli onnx-ptq [ARGUMENTS ...] ...
    quark-cli torch-ptq [ARGUMENTS ...] ...

DESCRIPTION
    quark-cli is the main command-line interface to the AMD Quark quantizer.
    List available subcommands with:

    quark-cli -h

    Subcommands are followed by arguments that specify configuration options for a particular quantization recipe.
    Moving the help option, -h, after the subcommand lists available arguments for each subcommand. e.g.

    quark-cli torch-llm-ptq -h    Help menu for torch-llm-ptq subcommand.

    Recipe suggestion for different models may be found in Quark's documentation at https://quark.docs.amd.com/.

REQUIREMENTS
    AMD Quark must already be installed in a Python environment.

    pip3 install amd-quark

    This requires installation of PyTorch for either CPU and GPU.
    See https://quark.docs.amd.com/ for detailed Quark installation instructions.

    Additionally, Quark-CLI has its own dependencies. From a local copy of Quark:

    pip3 install -r quark/experimental/cli/requirements.txt

EXAMPLES
    Replace with input and output directories of your choice:

    MODEL_DIR=dev/models/Llama-3.1-8b/
    OUTPUT_DIR=dev/models_output/

    Simple Torch PTQ Perplexity evaluation:

    quark-cli torch-llm-ptq --model_dir $MODEL_DIR --skip_quantization

    MX Quantization:

    quark-cli torch-llm-ptq --model_dir $MODEL_DIR --output_dir $OUTPUT_DIR --quant_scheme w_mxfp8 --num_calib_data 32 --group_size 32
"""

# Note that this file doesn't have a dash in its name because it needs to be easily importable as a module.
# The dash is retained in the name of the command however, to align with popular tools' name convention.

# Gracefully handle package import, as user may not be aware of dependencies.
try:
    import argparse
    import sys
except ImportError:
    print(
        "AMD Quark CLI dependencies need to be installed with `pip3 install -r quark/experimental/cli/requirements.txt`."
    )
    exit(1)

# Subcommand parsers, defined in separate files.
from quark.experimental.cli import torch_llm_ptq
from quark.experimental.cli.quark_onnx.export_oga import ExportOGA_CLI
from quark.experimental.cli.quark_onnx.export_onnx import ExportONNX_CLI
from quark.experimental.cli.quark_onnx.onnx_prepare_data import ONNXPrepareData_CLI
from quark.experimental.cli.quark_onnx.onnx_ptq import OnnxPTQ_CLI
from quark.experimental.cli.quark_onnx.onnx_ptq_autosearch import OnnxAutoSearch_CLI
from quark.experimental.cli.quark_onnx.onnx_validate import ONNXValidate_CLI


def get_cli_parser() -> argparse.ArgumentParser:
    """
    This function registers all subcommand parsers with the central parser, and return the master parser.
    Subcommand parsers are defined in separate files.
    Subcommand parsers have a consistent structure, set by the BaseQuarkCLICommand class, which they inherit from.
    """

    # Create the top-level parser.
    parser = argparse.ArgumentParser(prog="quark-cli", description="AMD Quark Model Optimizer CLI")
    subparsers = parser.add_subparsers(
        required=True,
        title="subcommands",
        description="Available Quark CLI commands",
        help="-h after command for additional help",
    )

    onnx_ptq_parser = subparsers.add_parser("onnx-ptq", help="ONNX workflow Post-Training Quantization")
    OnnxPTQ_CLI.register_subcommand(onnx_ptq_parser)
    onnx_ptq_parser.set_defaults(func=OnnxPTQ_CLI)

    onnx_autosearch_parser = subparsers.add_parser(
        "onnx-autosearch", help="Automatically sample config space to quantize ONNX models"
    )
    OnnxAutoSearch_CLI.register_subcommand(onnx_autosearch_parser)
    onnx_autosearch_parser.set_defaults(func=OnnxAutoSearch_CLI)

    export_onnx_parser = subparsers.add_parser("export-onnx", help="Create ONNX model")
    ExportONNX_CLI.register_subcommand(export_onnx_parser)
    export_onnx_parser.set_defaults(func=ExportONNX_CLI)

    export_oga_parser = subparsers.add_parser("export-oga", help="Create AWQ-quantized ONNX model")
    ExportOGA_CLI.register_subcommand(export_oga_parser)
    export_oga_parser.set_defaults(func=ExportOGA_CLI)

    onnx_validate_parser = subparsers.add_parser("onnx-validate", help="Test the accuracy of the ONNX model")
    ONNXValidate_CLI.register_subcommand(onnx_validate_parser)
    onnx_validate_parser.set_defaults(func=ONNXValidate_CLI)

    onnx_prepare_data_parser = subparsers.add_parser("onnx-prepare-data", help="Prepare data for ONNX model")
    ONNXPrepareData_CLI.register_subcommand(onnx_prepare_data_parser)
    onnx_prepare_data_parser.set_defaults(func=ONNXPrepareData_CLI)

    torch_llm_ptq_parser = subparsers.add_parser("torch-llm-ptq", help="PyTorch LLM Post-Training Quantization")
    torch_llm_ptq.TorchLLM_PTQ_CLI.register_subcommand(torch_llm_ptq_parser)
    torch_llm_ptq_parser.set_defaults(func=torch_llm_ptq.TorchLLM_PTQ_CLI)

    return parser


def main(raw_args: list[str] | None = None) -> None:
    """
    The entry-point to the Quark CLI. This function globally captures a list of user-provided command-line arguments,
    and feeds those forward into the appropriate subcommand's parser.
    """

    parser = get_cli_parser()
    args, unknown_args = parser.parse_known_args(raw_args)

    assert hasattr(args, "func")  # Check if my_subparser.set_defaults(func=my_function) is set up properly.

    # Instantiate and run the command
    # Assumes register_subcommand uses set_defaults(func=CommandClass)
    command_instance = args.func(parser, args, unknown_args)
    command_instance.run()


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
