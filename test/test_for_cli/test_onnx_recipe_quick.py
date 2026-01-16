#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import tempfile

import pytest
from PIL import Image

from quark.experimental.cli.main import main


@pytest.fixture
def setup_environment():
    old_hf_home = os.environ.get("HF_HOME", None)
    tmpdir = tempfile.TemporaryDirectory()
    os.environ["HF_HOME"] = tmpdir.name

    # create fake calib data
    calib_path = f"{tmpdir.name}/calib"
    image_folder = f"{calib_path}/n00000001"
    os.makedirs(image_folder)

    image = Image.new("RGB", (500, 375))
    image.save(f"{image_folder}/placeholder.jpg")

    yield calib_path

    if old_hf_home is None:
        del os.environ["HF_HOME"]
    else:
        os.environ["HF_HOME"] = old_hf_home


def export_onnx(output_dir, model_name):
    main(["export-onnx", "--model-name", model_name, "--output-dir", output_dir])


def test_export_onnx(setup_environment):
    _ = setup_environment
    with tempfile.TemporaryDirectory() as tmpdir:
        export_onnx(output_dir=tmpdir, model_name="test_byobnet.r160_in1k")
        assert "test_byobnet.r160_in1k.onnx" in os.listdir(tmpdir)


configs = [
    "BFP16",
    "MXINT8",
    "S8S8_AAWS",
    "S16S16_MIXED_S8S8",
]


@pytest.mark.parametrize("config_name", configs)
def test_onnx_cli_image_quantization_basic(setup_environment, config_name):
    calib_path = setup_environment
    with tempfile.TemporaryDirectory() as tmpdir:
        model_name = "test_resnet.r160_in1k"
        quantized_model_name = f"{model_name}_quantized.onnx"

        export_onnx(output_dir=tmpdir, model_name=model_name)

        main(
            [
                "onnx-ptq",
                "--model_name",
                model_name,
                "--input_model_path",
                f"{tmpdir}/{model_name}.onnx",
                "--output_model_path",
                f"{tmpdir}/{quantized_model_name}",
                "--calib_data_path",
                calib_path,
                "--config",
                config_name,
            ]
        )

        assert quantized_model_name in os.listdir(tmpdir)
