#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Integration tests for Shapeshifter PyTorch model support.

Tests PyTorch model loading, transformation, and saving through both
CLI and programmatic API.
"""

import unittest
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from quark.common.utils.testing_utils import use_temporary_directory
from quark.experimental.cli.main import main as cli
from quark.shapeshifter import PytorchModelConfig, RunConfig, shapeshifter


class SimpleModel(nn.Module):
    """Simple test model with Conv2d, Dropout, and ReLU."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.dropout(self.conv(x)))


class TestShapeshifterPyTorchModels(unittest.TestCase):
    """Test suite for PyTorch model support in Shapeshifter."""

    @use_temporary_directory
    def test_pytorch_model_file_based_via_cli(self, tmpdir: str) -> None:
        """Test PyTorch model processing from file via CLI."""
        tmpdir = Path(tmpdir)

        # Save PyTorch model
        model = SimpleModel()
        model_path = tmpdir / "model.pt"
        torch.save(model, model_path)

        # Create config with explicit weights_only=False
        output_path = tmpdir / "output.pt"
        config = {
            "input_model_config": {
                "model_type": "pytorch",
                "input_model_path": str(model_path),
                "weights_only": False,  # Required for PyTorch 2.6+ to load nn.Module
            },
            "passes": {"pytorch_remove_dropout": {}},
            "output_model_path": str(output_path),
        }

        config_path = tmpdir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Run CLI
        cli(["shapeshifter", str(config_path)])

        # Verify output exists
        self.assertTrue(output_path.exists(), "Output model file should exist")

        # Load and verify dropout removed (weights_only=False to load nn.Module)
        output_model = torch.load(output_path, weights_only=False)
        has_dropout = any(isinstance(m, nn.Dropout) for m in output_model.modules())
        self.assertFalse(has_dropout, "Dropout should be removed from output model")

        # Verify other layers still present
        has_conv = any(isinstance(m, nn.Conv2d) for m in output_model.modules())
        has_relu = any(isinstance(m, nn.ReLU) for m in output_model.modules())
        self.assertTrue(has_conv, "Conv2d should still be present")
        self.assertTrue(has_relu, "ReLU should still be present")

    @use_temporary_directory
    def test_pytorch_model_in_memory_programmatic(self, tmpdir: str) -> None:
        """Test PyTorch model processing with in-memory model via programmatic API."""
        tmpdir = Path(tmpdir)

        # Create in-memory model
        model = SimpleModel()

        # Run adapter with in-memory model (no input_model_config needed)
        run_config = RunConfig(passes={"pytorch_remove_dropout": {}})

        output_model = shapeshifter(run_config, model=model)

        # Verify dropout removed
        self.assertIsNotNone(output_model, "Should return transformed model")
        has_dropout = any(isinstance(m, nn.Dropout) for m in output_model.modules())
        self.assertFalse(has_dropout, "Dropout should be removed")

    @use_temporary_directory
    def test_pytorch_model_programmatic_with_config(self, tmpdir: str) -> None:
        """Test PyTorch model with PytorchModelConfig via programmatic API."""
        tmpdir = Path(tmpdir)

        model = SimpleModel()
        model_path = tmpdir / "model.pt"
        torch.save(model, model_path)

        output_path = tmpdir / "output.pt"

        # Use PytorchModelConfig
        model_config = PytorchModelConfig(input_model_path=model_path, weights_only=False, map_location="cpu")

        run_config = RunConfig(
            input_model_config=model_config, passes={"pytorch_remove_dropout": {}}, output_model_path=str(output_path)
        )

        shapeshifter(run_config)

        # Verify output exists
        self.assertTrue(output_path.exists())

        # Verify dropout removed
        output_model = torch.load(output_path, weights_only=False)
        has_dropout = any(isinstance(m, nn.Dropout) for m in output_model.modules())
        self.assertFalse(has_dropout, "Dropout should be removed")

    @use_temporary_directory
    def test_pytorch_model_in_memory_with_save(self, tmpdir: str) -> None:
        """Test PyTorch model processing with in-memory model and file save."""
        tmpdir = Path(tmpdir)

        model = SimpleModel()
        output_path = tmpdir / "output.pt"

        # Run adapter with in-memory model but specify output path
        run_config = RunConfig(passes={"pytorch_remove_dropout": {}}, output_model_path=str(output_path))

        output_model = shapeshifter(run_config, model=model)

        # Should return model
        self.assertIsNotNone(output_model, "Should return transformed model")

        # Verify dropout removed
        has_dropout = any(isinstance(m, nn.Dropout) for m in output_model.modules())
        self.assertFalse(has_dropout, "Dropout should be removed")

    @use_temporary_directory
    def test_mixed_passes_raises_error(self, tmpdir: str) -> None:
        """Test that mixing ONNX and PyTorch passes raises validation error."""
        tmpdir = Path(tmpdir)

        model = SimpleModel()
        model_path = tmpdir / "model.pt"
        torch.save(model, model_path)

        # Write YAML config manually to ensure onnx_simplify comes first
        config_path = tmpdir / "config.yaml"
        with open(config_path, "w") as f:
            f.write(
                f"""input_model_config:
  model_type: pytorch
  input_model_path: {model_path}
  weights_only: false
passes:
  onnx_simplify:
    simplify: true
  pytorch_remove_dropout: {{}}
output_model_path: {tmpdir / "output.pt"}
"""
            )

        # Should raise validation error
        with self.assertRaises(ValueError) as ctx:
            cli(["shapeshifter", str(config_path)])

        # Since onnx_simplify is first, it detects as ONNX, then complains about pytorch pass
        error_msg = str(ctx.exception)
        self.assertTrue(
            "not an ONNX pass" in error_msg or "not a PyTorch pass" in error_msg,
            f"Expected pass type validation error, got: {error_msg}",
        )

    @use_temporary_directory
    def test_pytorch_trace_model_pass(self, tmpdir: str) -> None:
        """Test pytorch_trace_model pass."""
        tmpdir = Path(tmpdir)

        model = SimpleModel()
        model_path = tmpdir / "model.pt"
        torch.save(model, model_path)

        output_path = tmpdir / "output.pt"
        config = {
            "input_model_config": {
                "model_type": "pytorch",
                "input_model_path": str(model_path),
                "weights_only": False,
            },
            "passes": {
                "pytorch_trace_model": {
                    "input_shapes": {"input": [1, 3, 224, 224]},
                    "input_dtypes": {"input": "float32"},
                }
            },
            "output_model_path": str(output_path),
        }

        config_path = tmpdir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        cli(["shapeshifter", str(config_path)])

        # Verify output exists and is traced
        self.assertTrue(output_path.exists())

        # Load traced model - use torch.jit.load for TorchScript models
        traced_model = torch.jit.load(output_path)

        # TorchScript traced models have different type
        self.assertIsInstance(
            traced_model, (torch.jit.ScriptModule, torch.jit.RecursiveScriptModule), "Should be a traced model"
        )

    @use_temporary_directory
    def test_multiple_passes_sequence(self, tmpdir: str) -> None:
        """Test multiple PyTorch passes in sequence."""
        tmpdir = Path(tmpdir)

        model = SimpleModel()
        model_path = tmpdir / "model.pt"
        torch.save(model, model_path)

        output_path = tmpdir / "output.pt"
        config = {
            "input_model_config": {
                "model_type": "pytorch",
                "input_model_path": str(model_path),
                "weights_only": False,
            },
            "passes": {
                "pytorch_remove_dropout": {},
                "pytorch_trace_model": {
                    "input_shapes": {"input": [1, 3, 224, 224]},
                },
            },
            "output_model_path": str(output_path),
        }

        config_path = tmpdir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        cli(["shapeshifter", str(config_path)])

        # Verify output exists
        self.assertTrue(output_path.exists())

        # Load traced model - use torch.jit.load for TorchScript models
        traced_model = torch.jit.load(output_path)
        self.assertIsInstance(
            traced_model, (torch.jit.ScriptModule, torch.jit.RecursiveScriptModule), "Should be traced"
        )


if __name__ == "__main__":
    unittest.main()
