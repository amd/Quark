#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Wrapper for the "export-onnx" subcommand.
import argparse
import os

from onnxruntime_genai.models.builder import create_model

from quark.experimental.cli import base_cli


class ExportOGA_CLI(base_cli.BaseQuarkCLICommand):
    """
    This class is the Quark CLI export-oga subcommand.

    It can convert a pytorch model into an ONNX model using ONNX Runtime GenAI which is a variant of the ONNX runtime optimized for generative AI.
    """

    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser):
        parser.add_argument(
            "-m", "--model_path", help="Folder to load PyTorch model and associated files from", required=True
        )
        parser.add_argument(
            "-o", "--output_path", help="Folder to save AWQ-quantized ONNX model and associated files in", required=True
        )

    def run(self):
        args = self.args

        # Create ONNX model
        model_name = None
        input_folder = args.model_path
        output_folder = args.output_path
        precision = "fp32"  # int4 or fp32
        execution_provider = "cpu"
        cache_dir = os.path.join(".", "cache_dir")
        # NOTE export to onnx model
        create_model(model_name, input_folder, output_folder, precision, execution_provider, cache_dir)
