#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import tempfile

import pytest

from quark.common.utils.testing_utils import set_environment_variables
from quark.experimental.cli.main import main


@pytest.fixture(scope="module", autouse=True)
def _setup_env():
    """Set environment variables for all torch recipe tests."""
    with set_environment_variables(
        CUDA_VISIBLE_DEVICES="0",
        HIP_VISIBLE_DEVICES="0",
        TORCH_COMPILE_DISABLE="1",
        TORCHDYNAMO_DISABLE="1",
    ):
        yield


def test_recipe_w_fp8_a_fp8_kv_fp8():
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.environ.get("CUSTOM_CI_TINY_MODELS_CACHE")
        assert os.path.exists(model_path)
        main(
            [
                "torch-llm-ptq",
                "--model_dir",
                model_path,
                "--output_dir",
                temp_dir,
                "--quant_scheme",
                "fp8",
                "--kv_cache_dtype",
                "fp8",
                "--num_calib_data",
                "128",
                "--skip_evaluation",
            ]
        )

        assert len(os.listdir(temp_dir)) > 0


def test_recipe_w_int4_per_group_sym_awq():
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.environ.get("CUSTOM_CI_TINY_MODELS_CACHE")
        assert os.path.exists(model_path)
        main(
            [
                "torch-llm-ptq",
                "--model_dir",
                model_path,
                "--output_dir",
                temp_dir,
                "--quant_scheme",
                "int4_wo_32",
                "--num_calib_data",
                "128",
                "--quant_algo",
                "awq",
                "--dataset",
                "pileval_for_awq_benchmark",
                "--seq_len",
                "512",
                "--skip_evaluation",
            ]
        )

        assert len(os.listdir(temp_dir)) > 0


def test_recipe_w_int8_a_int8_per_tensor_sym():
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.environ.get("CUSTOM_CI_TINY_MODELS_CACHE")
        assert os.path.exists(model_path)
        main(
            [
                "torch-llm-ptq",
                "--model_dir",
                model_path,
                "--output_dir",
                temp_dir,
                "--quant_scheme",
                "int8",
                "--num_calib_data",
                "128",
                "--device",
                "cuda",
                "--skip_evaluation",
            ]
        )

        assert len(os.listdir(temp_dir)) > 0


def test_recipe_w_uint4_per_group_asym_awq():
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.environ.get("CUSTOM_CI_TINY_MODELS_CACHE")
        assert os.path.exists(model_path)
        main(
            [
                "torch-llm-ptq",
                "--model_dir",
                model_path,
                "--output_dir",
                temp_dir,
                "--quant_scheme",
                "uint4_wo_32",
                "--quant_algo",
                "awq",
                "--num_calib_data",
                "128",
                "--device",
                "cuda",
                "--skip_evaluation",
            ]
        )

        assert len(os.listdir(temp_dir)) > 0
