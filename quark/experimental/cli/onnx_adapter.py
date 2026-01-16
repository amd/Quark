#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Wrapper for the "onnx-adapter" subcommand.

import argparse

# Gracefully handle imports, as user may not have Quark installed.
try:
    from quark.experimental.cli import base_cli
    from quark.onnx_adapter import Engine, LoadConfigFromFileOrDict
    from quark.shares.utils.log import ScreenLogger

    logger = ScreenLogger(__name__)
except ImportError:
    print(
        "AMD Quark needs to be installed with e.g. `pip3 install amd-quark`. Refer to https://quark.amd.docs.com for more detail."
    )
    exit(1)

# Gracefully handle imports, as user may not be aware of dependencies required.
try:
    import onnx  # noqa: F401
except ImportError:
    print("AMD Quark CLI dependencies need to be installed with `pip3 install -r quark/cli/requirements.txt`.")
    exit(1)


class ONNXAdapter_CLI(base_cli.BaseQuarkCLICommand):
    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("config_file", type=str, help="Input JSON or YAML file path")

    def run(self) -> None:
        args = self.args

        # Fire up the engine and get running
        # TODO: Expose Engine args to CLI
        engine_config = {}
        if args.config_file:
            engine_config = LoadConfigFromFileOrDict(args.config_file).data
        engine = Engine(config=engine_config)
        engine.initialize()
        engine.run()
