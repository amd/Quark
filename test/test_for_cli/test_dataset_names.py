#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import argparse
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoTokenizer

from quark.experimental.cli.quark_onnx.helper_utils import (
    get_calib_dataloader_to_tensor,
    get_pileval,
    get_wikitext2,
)
from quark.experimental.cli.quark_onnx.onnx_validate import TextDataset, evaluate_wikitext_onnx


def test_get_wikitext2() -> None:
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    get_wikitext2(tokenizer=tokenizer, nsamples=1, seqlen=10, device=None)


def test_get_pileval() -> None:
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    result = get_pileval(tokenizer=tokenizer, nsamples=128, seqlen=2048, device=None)
    assert len(result) > 0
    assert "input_ids" in result[0]


def test_get_calib_dataloader_to_tensor_pileval() -> None:
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    get_calib_dataloader_to_tensor(dataset_name="pileval", tokenizer=tokenizer, num_calib_data=1, seqlen=10)


def test_get_calib_dataloader_to_tensor_cnn_dailymail() -> None:
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    get_calib_dataloader_to_tensor(
        dataset_name="abisee/cnn_dailymail", tokenizer=tokenizer, num_calib_data=1, seqlen=10
    )


def test_get_calib_dataloader_to_tensor_wikitext() -> None:
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    get_calib_dataloader_to_tensor(dataset_name="Salesforce/wikitext", tokenizer=tokenizer, num_calib_data=1, seqlen=10)


def test_text_dataset() -> None:
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    with pytest.raises(AttributeError):
        TextDataset(tokenizer=tokenizer, block_size=512)


@patch("quark.experimental.cli.quark_onnx.onnx_validate.load_and_cache_examples", return_value=[])
def test_evaluate_wikitext_onnx(_mock_load_and_cache: MagicMock) -> None:
    args = argparse.Namespace(per_gpu_eval_batch_size=1, block_size=512)
    with pytest.raises(ZeroDivisionError):
        evaluate_wikitext_onnx(args, MagicMock(), MagicMock())


@patch("quark.experimental.cli.torch_llm_ptq.ppl_eval")
@patch("quark.experimental.cli.torch_llm_ptq.export_safetensors")
@patch("quark.experimental.cli.torch_llm_ptq.ModelQuantizer")
@patch("quark.experimental.cli.torch_llm_ptq.LLMTemplate")
@patch("quark.experimental.cli.torch_llm_ptq.get_calib_dataloader")
@patch("quark.experimental.cli.torch_llm_ptq.get_tokenizer")
@patch("quark.experimental.cli.torch_llm_ptq.preprocess_for_quantization")
@patch("quark.experimental.cli.torch_llm_ptq.get_model")
@patch("os.chdir")
@patch("os.makedirs")
def test_torch_llm_ptq_run(
    _mock_makedirs: MagicMock,
    _mock_chdir: MagicMock,
    mock_get_model: MagicMock,
    _mock_preprocess: MagicMock,
    _mock_get_tokenizer: MagicMock,
    _mock_get_calib_dataloader: MagicMock,
    _mock_template: MagicMock,
    mock_quantizer_cls: MagicMock,
    _mock_export: MagicMock,
    _mock_ppl_eval: MagicMock,
) -> None:
    mock_get_model.return_value = (MagicMock(), torch.float16)
    quantizer = MagicMock()
    quantizer.quantize_model.return_value = MagicMock()
    quantizer.freeze.return_value = MagicMock()
    mock_quantizer_cls.return_value = quantizer

    args = argparse.Namespace(
        model_dir="/fake",
        device="cpu",
        multi_device=False,
        dataset="pileval",
        seq_len=512,
        batch_size=1,
        num_calib_data=1,
        quant_scheme="w_uint4_per_group_asym",
        layer_quant_scheme=None,
        kv_cache_dtype=None,
        quant_algo=None,
        exclude_layers=None,
        output_dir="/tmp/test_out",
        skip_evaluation=False,
        evaluation_dataset="wikitext",
        no_trust_remote_code=False,
    )

    from quark.experimental.cli.torch_llm_ptq import TorchLLM_PTQ_CLI

    TorchLLM_PTQ_CLI(parser=MagicMock(), args=args).run()
