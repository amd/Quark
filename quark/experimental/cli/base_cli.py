#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# quark-cli Base Class Structure
# This exists to ensure consistent functions are used with all subcommands, and allow them to be auto-discoverable for processing in a loop.

# Gracefully handle package import, as user may not be aware of dependencies.
try:
    import argparse
    from abc import ABC, abstractmethod
    from typing import ClassVar, Optional
except ImportError:
    print(
        "AMD Quark CLI dependencies need to be installed with `pip3 install -r quark/experimental/cli/requirements.txt`."
    )
    exit(1)


class BaseQuarkCLICommand(ABC):
    """
    This class is the standard interface for Quake CLI subcommands.
    To add a subcommand to Quark CLI, create a class that inherits from this class in its own file,
    and then add it to the list of subcommand parsers in main.py.
    """

    allow_unknown_args: ClassVar[bool] = False

    def __init__(self, parser: argparse.ArgumentParser, args: argparse.Namespace, unknown_args: list | None = None):  # type: ignore
        self.args = args
        self.unknown_args = unknown_args

        if unknown_args and not self.allow_unknown_args:
            parser.error(f"Unknown arguments: {unknown_args}")

    @staticmethod
    @abstractmethod
    def register_subcommand(parser: argparse.ArgumentParser) -> None:
        """
        Your subcommand class must implement this function, to which you add all the flags required by your subcommand. e.g.

        ```python
        @staticmethod
        def register_subcommand(parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--model_dir", help="Specify where the model is.")
        ```
        """
        raise NotImplementedError

    @abstractmethod
    def run(self) -> None:
        """
        Your subcommand class must implement this function, where you define what to do when the user executes your subcommand.
        You can access user-provided arguments with `self.args` e.g.

        ```python
        @staticmethod
        def run(self) -> None:
            args = self.args

            print('Loading model from ' + args.model_dir)
        """
        raise NotImplementedError
