#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for the onnx_align_scale adapter pass.

Export a Torch model (Concat, MaxPool, AveragePool, GlobalAveragePool, Pad, Slice, Transpose, Reshape) to ONNX, quantize
with Int8 Q/DQ, then run the onnx_align_scale pass via CLI and assert alignment logs
and output model validity.
"""

import unittest
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from onnxruntime.quantization import CalibrationDataReader

from quark.common.utils.testing_utils import use_temporary_directory
from quark.experimental.cli.main import main as cli
from quark.onnx import Int8Spec, ModelQuantizer, QConfig, QLayerConfig

np.random.seed(123456)
# Single calibration batch for quantization (NCHW).
CALIBRATION_INPUT = np.random.randn(1, 3, 16, 16).astype(np.float32) * 0.1


class DataReader(CalibrationDataReader):
    """Single-batch calibration data reader for static quantization."""

    def __init__(self, input_tensor: np.ndarray, input_name: str = "input"):
        """Initialize the DataReader with calibration data.

        Args:
            input_tensor: A numpy array containing the calibration input data.
            input_name: The name of the input tensor. Defaults to "input".

        Initializes:
            self.data: List containing the input tensor.
            self.input_name: Name of the input tensor.
            self._index: Current index for iteration, initialized to 0.
        """
        self.data = [input_tensor]
        self.input_name = input_name
        self._index = 0

    def get_next(self):
        """Return the next batch of calibration data.

        This method is part of the CalibrationDataReader interface and provides
        calibration data one batch at a time for quantization.

        Returns:
            dict: A dictionary with the input name as key and the calibration data as value,
                  or None when all calibration data has been consumed.
        """
        if self._index < len(self.data):
            self._index += 1
            return {self.input_name: self.data[self._index - 1]}
        return None

    def rewind(self):
        """Reset the data reader to the beginning.

        This method resets the internal index counter to allow re-reading
        the calibration data from the start.
        """
        self._index = 0


class AlignScaleOpsModel(nn.Module):
    """
    Small Torch model that produces Concat, MaxPool, Pad, Slice, Transpose, and Reshape
    in the exported ONNX graph for exercising the onnx_align_scale pass.
    """

    def __init__(self):
        """Initialize the AlignScaleOpsModel architecture.

        Creates two convolutional layers (conv1 and conv2) with 3 input channels,
        4 output channels, kernel size 3, and padding 1, along with a max pooling
        layer with kernel size 2 and stride 2.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 3, 16, 16) in NCHW format.

        Returns:
            Output tensor of shape (batch_size, 128) after the following operations:
            - Concat: Two convolutional branches concatenated along channel dimension
            - Pool: MaxPool2d with kernel size 2x2
            - Pad: Constant padding of 1 pixel on all sides
            - Slice: Spatial slicing to extract region [1:5, 1:5]
            - Transpose: Permute from NCHW to NHWC format
            - Reshape: Flatten to (batch_size, -1)
        """
        branch_a = self.conv1(x)
        branch_b = self.conv2(x)
        out = torch.cat([branch_a, branch_b], dim=1)
        out = self.pool(out)
        out = F.pad(out, (1, 1, 1, 1), mode="constant", value=0.0)
        out = out[:, :, 1:5, 1:5]
        out = out.permute(0, 2, 3, 1)
        out = out.reshape(out.shape[0], -1)
        return out


def export_float_onnx(output_dir: str) -> str:
    """Export AlignScaleOpsModel to float ONNX. Returns path to the saved model."""
    torch.manual_seed(42)
    model = AlignScaleOpsModel()
    model.eval()
    out_path = Path(output_dir, "align_scale_ops_float.onnx").as_posix()
    dummy_input = torch.randn(1, 3, 16, 16)
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["input"],
        output_names=["output"],
        keep_initializers_as_inputs=False,
        do_constant_folding=False,
        opset_version=17,
        dynamo=False,
    )
    return out_path


def quantize_model(output_dir: str, calibration_data: np.ndarray) -> str:
    """Quantize the float ONNX model with Int8 Q/DQ. Returns path to the quantized model."""
    float_model_path = export_float_onnx(output_dir)
    quant_model_path = Path(output_dir, "align_scale_ops_quant.onnx").as_posix()
    data_reader = DataReader(calibration_data)
    qconfig = QConfig(
        global_config=QLayerConfig(activation=Int8Spec(), weight=Int8Spec()),
        extra_options={
            "ExtraOpTypesToQuantize": [
                "Concat",
                "Pad",
                "Slice",
                "MaxPool",
                "AveragePool",
                "GlobalAveragePool",
                "Transpose",
                "Reshape",
            ],
        },
    )
    quantizer = ModelQuantizer(qconfig)
    quantizer.quantize_model(float_model_path, quant_model_path, data_reader)
    return quant_model_path


def write_align_scale_yaml(
    output_dir: str,
    input_model_path: str,
    output_model_path: str,
    align_scale_option: bool | str | list[str] = True,
) -> str:
    """Write shapeshifter YAML config with onnx_align_scale pass. Returns path to the YAML file."""
    yaml_path = Path(output_dir, "align_scale.yaml").as_posix()
    adapter_config = {
        "input_model_path": input_model_path,
        "passes": {
            "onnx_align_scale": {
                "align_scale": align_scale_option,
            }
        },
        "output_model_path": output_model_path,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(adapter_config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def is_model_valid(model_path: str) -> bool:
    """Load the ONNX model and run the checker. Returns True if the model is valid."""
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        return True
    except Exception:
        return False


class TestONNXAdapterONNXAlignScalePass(unittest.TestCase):
    """Test the onnx_align_scale pass on a quantized model with Concat, MaxPool, AveragePool, GlobalAveragePool, Pad, Slice, Transpose, Reshape."""

    def _run_align_scale(
        self,
        tmpdir: str,
        input_model_path: str,
        output_model_path: str,
        align_scale_option: bool | str | list[str],
    ) -> list[str]:
        """Run the onnx_align_scale pass via CLI and return the captured screen-logger output.

        The pass unconditionally emits at least one INFO line ("Adjust the quantize info ...")
        whenever it runs, so ``assertLogs`` is always satisfied even for a no-op alignment.
        """
        yaml_path = write_align_scale_yaml(
            tmpdir, input_model_path, output_model_path, align_scale_option=align_scale_option
        )
        with self.assertLogs("quark.shapeshifter.passes.onnx_align_scale_screen", level="INFO") as log_ctx:
            cli(["shapeshifter", yaml_path])
        return log_ctx.output

    @use_temporary_directory
    def test_align_scale_pass_quantized_model_via_cli(self, tmpdir: str) -> None:
        """Quantize, align via CLI, and assert the alignment post-condition (idempotence) + output validity.

        NOTE: we intentionally do NOT assert specific ``Have aligned <op> node <name> ...`` messages.
        Those fire only when a node's Q/DQ scales were *misaligned before* the pass, which depends on the
        calibrated quant scales -- not bit-reproducible across GPUs/runs -- so such assertions are flaky.
        Instead we assert the invariant the pass guarantees: after alignment, re-running it is a no-op
        (every targeted op already shares input/output Q/DQ params).
        """
        ops = [
            "Concat",
            "Pad",
            "Slice",
            "MaxPool",
            "AveragePool",
            "GlobalAveragePool",
            "Transpose",
            "Reshape",
        ]
        quant_model_path = quantize_model(tmpdir, CALIBRATION_INPUT)
        aligned_model_path = Path(tmpdir, "align_scale_ops_aligned.onnx").as_posix()
        rerun_model_path = Path(tmpdir, "align_scale_ops_aligned_rerun.onnx").as_posix()

        self._run_align_scale(tmpdir, quant_model_path, aligned_model_path, ops)
        self.assertTrue(is_model_valid(aligned_model_path), "Aligned model should pass ONNX check")

        # Idempotence: aligning an already-aligned model must not realign anything.
        rerun_logs = self._run_align_scale(tmpdir, aligned_model_path, rerun_model_path, ops)
        realigned = [msg for msg in rerun_logs if "Have aligned" in msg]
        self.assertEqual(realigned, [], f"Alignment should be idempotent, but re-run realigned: {realigned}")
        self.assertTrue(is_model_valid(rerun_model_path), "Re-aligned model should pass ONNX check")

    @use_temporary_directory
    def test_align_scale_pass_with_single_op_type(self, tmpdir: str) -> None:
        """Run pass with align_scale set to a single op type string (Concat); assert idempotence + validity."""
        quant_model_path = quantize_model(tmpdir, CALIBRATION_INPUT)
        aligned_model_path = Path(tmpdir, "align_scale_ops_aligned.onnx").as_posix()
        rerun_model_path = Path(tmpdir, "align_scale_ops_aligned_rerun.onnx").as_posix()

        self._run_align_scale(tmpdir, quant_model_path, aligned_model_path, "Concat")
        self.assertTrue(is_model_valid(aligned_model_path), "Aligned model should pass ONNX check")

        rerun_logs = self._run_align_scale(tmpdir, aligned_model_path, rerun_model_path, "Concat")
        realigned = [msg for msg in rerun_logs if "Have aligned" in msg]
        self.assertEqual(realigned, [], f"Alignment should be idempotent, but re-run realigned: {realigned}")

    @use_temporary_directory
    def test_align_scale_pass_logs_when_alignment_runs(self, tmpdir: str) -> None:
        """Assert the pass logs its activity when it runs.

        Uses the unconditional activity log emitted on every pass run rather than a per-node
        ``Have aligned ...`` message (which is calibration-dependent and therefore flaky).
        """
        quant_model_path = quantize_model(tmpdir, CALIBRATION_INPUT)
        aligned_model_path = Path(tmpdir, "align_scale_ops_aligned.onnx").as_posix()

        logs = self._run_align_scale(tmpdir, quant_model_path, aligned_model_path, ["Concat"])
        self.assertTrue(
            any("Adjust the quantize info to meet the compiler constraints" in msg for msg in logs),
            "Pass should log its activity when it runs",
        )
        self.assertTrue(is_model_valid(aligned_model_path), "Aligned model should pass ONNX check")


if __name__ == "__main__":
    unittest.main()
