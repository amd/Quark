#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
from io import StringIO

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from quark.common.utils.log import ScreenLogger
from quark.common.utils.testing_utils import PatchEverywhere, capture_quark_logs, torch_device
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.template import LLMTemplate
from quark.torch.utils.llm import preprocess_for_quantization


def test_logging_levels():
    old_stderr = sys.stderr
    mystderr = StringIO()
    sys.stderr = mystderr

    old_level = ScreenLogger._shared_level
    logger = ScreenLogger(__name__)

    logger.info("hehe")
    logger.debug("hoho")

    assert "hehe" in mystderr.getvalue()

    # See https://docs.python.org/3/library/logging.html#logging-levels.
    if ScreenLogger._shared_level >= 20:
        assert "hoho" not in mystderr.getvalue()

    with PatchEverywhere("QUARK_LOG_LEVEL", "debug", module_name_prefix="quark"):
        logger = ScreenLogger(__name__)

        logger.debug("huhu")
        logger.info("hihi")
        assert "huhu" in mystderr.getvalue()
        assert "hihi" in mystderr.getvalue()

    # Restore global/shared logging level for subsequent tests in same process.
    ScreenLogger.set_shared_level(old_level)
    sys.stderr = old_stderr


def test_check_token_distribution_warning():
    """Test that the warning from _check_token_distribution is displayed for MoE models."""
    config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")
    config.num_hidden_layers = 2

    with torch.device(torch_device):
        model = AutoModelForCausalLM.from_config(config)
    model = model.eval()

    preprocess_for_quantization(model)

    with capture_quark_logs() as captured_output:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")

        # Get fp8 template configuration
        template = LLMTemplate.get("qwen3_moe")
        config = template.get_config(scheme="fp8")

        # Create a minimal dataloader with a single prompt "a"
        text = "a"
        tokenized_outputs = tokenizer(text, return_tensors="pt").to(torch_device)
        calib_dataloader = DataLoader(tokenized_outputs["input_ids"])

        quantizer = ModelQuantizer(config)
        _ = quantizer.quantize_model(model, calib_dataloader)

        # Check that the MoE expert coverage warning is in captured output
        output_content = captured_output.getvalue()
        assert "MoE EXPERT COVERAGE WARNING" in output_content
        assert "received 0 tokens during calibration" in output_content


def test_token_distribution_no_warning():
    """Test that the warning from _check_token_distribution is displayed for MoE models."""

    for model_id, scheme in [("Qwen/Qwen3-8B", "fp8"), ("Qwen/Qwen3-30B-A3B-Instruct-2507", "mxfp4")]:
        config = AutoConfig.from_pretrained(model_id)
        config.num_hidden_layers = 2

        with torch.device(torch_device):
            model = AutoModelForCausalLM.from_config(config)
        model = model.eval()

        with capture_quark_logs() as captured_output:
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            template = LLMTemplate.get(model_type=model.config.model_type)
            config = template.get_config(scheme=scheme)

            # Create a minimal dataloader with a single prompt "a"
            text = "a"
            tokenized_outputs = tokenizer(text, return_tensors="pt").to(torch_device)
            calib_dataloader = DataLoader(tokenized_outputs["input_ids"])

            quantizer = ModelQuantizer(config)
            _ = quantizer.quantize_model(model, calib_dataloader)

            # Check that the MoE expert coverage warning is in captured output
            output_content = captured_output.getvalue()
            assert "MoE EXPERT COVERAGE WARNING" not in output_content
            assert "received 0 tokens during calibration" not in output_content
