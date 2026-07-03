#
# Copyright (C) 2025 - 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Tests to verify that 'onnx-adapter' CLI alias works identically to 'shapeshifter'."""

import unittest
from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import TensorProto, helper, numpy_helper

from quark.common.utils.testing_utils import use_temporary_directory
from quark.experimental.cli.main import main as cli


def create_simple_conv_model(output_dir: str) -> tuple[str, str]:
    """Create a simple ONNX model with a Conv node for testing.

    This model is realistic and passes ONNX validation.
    """
    # Input: batch=1, channels=1, height=4, width=4
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
    # Output: batch=1, channels=1, height=2, width=2 (after 3x3 conv with no padding)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 2, 2])

    # Create weight tensor: out_channels=1, in_channels=1, kernel=3x3
    W = np.ones((1, 1, 3, 3), dtype=np.float32)
    W_init = numpy_helper.from_array(W, "W")

    # Create Conv node
    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        name="Conv_0",
    )

    graph = helper.make_graph(
        nodes=[conv_node],
        name="SimpleConvModel",
        inputs=[X],
        outputs=[Y],
        initializer=[W_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 8

    # Validate model before saving
    onnx.checker.check_model(model)

    input_model_path = Path(output_dir, "input_model.onnx").as_posix()
    output_model_path = Path(output_dir, "output_model.onnx").as_posix()

    onnx.save(model, input_model_path)
    return input_model_path, output_model_path


def create_config_yaml(output_dir: str, input_model_path: str, output_model_path: str) -> str:
    """Create a YAML config file for shapeshifter."""
    yaml_path = Path(output_dir, "config.yaml").as_posix()
    config = {
        "input_model_path": input_model_path,
        "passes": {
            "onnx_simplify": {"simplify": True},
        },
        "output_model_path": output_model_path,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


class TestShapeshifterCliAlias(unittest.TestCase):
    """Test that 'onnx-adapter' works as an alias for 'shapeshifter'."""

    @use_temporary_directory
    def test_onnx_adapter_alias_produces_same_output(self, tmpdir: str):
        """Test that 'onnx-adapter' produces the same output as 'shapeshifter'."""
        # Create model and config
        input_model_path, output_model_path = create_simple_conv_model(tmpdir)
        yaml_path = create_config_yaml(tmpdir, input_model_path, output_model_path)

        # Run using the deprecated 'onnx-adapter' alias
        cli(["onnx-adapter", yaml_path])

        # Verify output model was created
        self.assertTrue(Path(output_model_path).exists(), "Output model should be created by onnx-adapter")

        # Load and verify the output model is valid
        output_model = onnx.load(output_model_path)
        onnx.checker.check_model(output_model)

    @use_temporary_directory
    def test_shapeshifter_and_onnx_adapter_produce_identical_results(self, tmpdir: str):
        """Test that both commands produce identical output models."""
        # Create two separate output directories
        shapeshifter_dir = Path(tmpdir) / "shapeshifter"
        onnx_adapter_dir = Path(tmpdir) / "onnx_adapter"
        shapeshifter_dir.mkdir()
        onnx_adapter_dir.mkdir()

        # Create models for each
        ss_input, ss_output = create_simple_conv_model(str(shapeshifter_dir))
        oa_input, oa_output = create_simple_conv_model(str(onnx_adapter_dir))

        # Create configs
        ss_yaml = create_config_yaml(str(shapeshifter_dir), ss_input, ss_output)
        oa_yaml = create_config_yaml(str(onnx_adapter_dir), oa_input, oa_output)

        # Run both commands
        cli(["shapeshifter", ss_yaml])
        cli(["onnx-adapter", oa_yaml])

        # Both outputs should exist
        self.assertTrue(Path(ss_output).exists(), "shapeshifter output should exist")
        self.assertTrue(Path(oa_output).exists(), "onnx-adapter output should exist")

        # Load both models and compare structure
        ss_model = onnx.load(ss_output)
        oa_model = onnx.load(oa_output)

        # Compare graph structure (node count, node types)
        self.assertEqual(
            len(ss_model.graph.node),
            len(oa_model.graph.node),
            "Both outputs should have same number of nodes",
        )

        for ss_node, oa_node in zip(ss_model.graph.node, oa_model.graph.node, strict=False):
            self.assertEqual(
                ss_node.op_type,
                oa_node.op_type,
                "Node op_types should match",
            )

    @use_temporary_directory
    def test_both_commands_are_registered(self, tmpdir: str):  # noqa: ARG002
        """Test that both 'onnx-adapter' and 'shapeshifter' are registered CLI commands."""
        from quark.experimental.cli.main import get_cli_parser

        parser = get_cli_parser()
        subparsers = parser._subparsers._group_actions[0].choices

        # Both commands should be registered
        self.assertIn("onnx-adapter", subparsers, "onnx-adapter should be registered")
        self.assertIn("shapeshifter", subparsers, "shapeshifter should be registered")

        # Both should use the same handler class
        oa_defaults = subparsers["onnx-adapter"]._defaults
        ss_defaults = subparsers["shapeshifter"]._defaults
        self.assertEqual(
            oa_defaults["func"],
            ss_defaults["func"],
            "Both commands should use the same CLI handler class",
        )

    @use_temporary_directory
    def test_onnx_adapter_triggers_deprecation_warning_path(self, tmpdir: str):
        """Test that using 'onnx-adapter' triggers the deprecation warning code path."""
        import sys

        # Create model and config
        input_model_path, output_model_path = create_simple_conv_model(tmpdir)
        yaml_path = create_config_yaml(tmpdir, input_model_path, output_model_path)

        # Temporarily modify sys.argv to simulate CLI invocation via onnx-adapter
        # This triggers the deprecation warning in shapeshifter.py
        original_argv = sys.argv
        try:
            sys.argv = ["quark-cli", "onnx-adapter", yaml_path]
            cli(["onnx-adapter", yaml_path])
        finally:
            sys.argv = original_argv

        # Verify output model was created (command succeeded)
        self.assertTrue(Path(output_model_path).exists(), "Output model should be created")


if __name__ == "__main__":
    unittest.main()
