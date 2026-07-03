#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Wrapper for the "shapeshifter" subcommand.

import argparse
import sys

# Gracefully handle imports, as user may not have Quark installed.
try:
    from quark.common.utils.log import ScreenLogger
    from quark.experimental.cli import base_cli
    from quark.shapeshifter import Engine, LoadConfigFromFileOrDict

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


class Shapeshifter_CLI(base_cli.BaseQuarkCLICommand):
    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("config_file", type=str, help="Input JSON or YAML file path")

    def run(self) -> None:
        args = self.args

        # Check if user invoked via deprecated 'onnx-adapter' alias
        if len(sys.argv) > 1 and sys.argv[1] == "onnx-adapter":
            logger.warning("The 'onnx-adapter' command is deprecated. Please use 'shapeshifter' instead.")

        # Fire up the engine and get running
        # TODO: Expose Engine args to CLI
        engine_config = {}
        if args.config_file:
            engine_config = LoadConfigFromFileOrDict(args.config_file).data
        engine = Engine(config=engine_config)
        engine.initialize()
        engine.run()
