#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Wrapper for the "export-onnx" subcommand.
import argparse
import sys
from pathlib import Path

import timm
import torch

from quark.experimental.cli import base_cli


class ExportONNX_CLI(base_cli.BaseQuarkCLICommand):
    """
    Simple flow for creating onnx versions of TIMM models.
    """

    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser):
        parser.add_argument("--model-name", help="Specify the model to be converted to onnx.", required=True)
        parser.add_argument("--output-dir", help="The directory to save the output.", required=True)

    def run(self):
        args = self.args
        model_name = args.model_name

        model = timm.create_model(model_name, pretrained=True)
        model = model.eval()
        device = torch.device("cpu")

        data_config = timm.data.resolve_model_data_config(
            model=model,
            use_test_size=True,
        )

        batch_size = 1
        torch.manual_seed(42)
        dummy_input = torch.randn((batch_size,) + tuple(data_config["input_size"])).to(device)

        out_dir = Path(args.output_dir)
        if out_dir.exists() is False:
            print(f"output-dir: {out_dir} does not exist.")
            sys.exit(-1)
        elif out_dir.is_dir() is False:
            print(f"output-dir: {out_dir} is not a directory.")
            sys.exit(-1)

        output_path = out_dir / f"{model_name}.onnx"

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            do_constant_folding=True,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            verbose=True,
        )
        print(f"Onnx model is saved at {output_path}")
