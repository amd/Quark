#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Verify that data_preparation functions use fully namespaced HuggingFace dataset identifiers."""

from transformers import AutoTokenizer

from quark.torch.utils.llm.data_preparation import (
    get_calib_dataloader,
    get_calib_dataloader_to_dict,
    get_calib_dataloader_to_tensor,
    get_dataset,
    get_pileval,
    get_trainer_dataset,
    get_wikitext2,
)

TOKENIZER = AutoTokenizer.from_pretrained("facebook/opt-125m")


def test_get_wikitext2() -> None:
    get_wikitext2(tokenizer=TOKENIZER, nsamples=1, seqlen=10, device=None)


def test_get_pileval() -> None:
    result = get_pileval(tokenizer=TOKENIZER, nsamples=128, seqlen=2048, device=None)
    assert result.shape[0] > 0
    assert result.shape[1] == 2048


def test_get_calib_dataloader_to_tensor_pileval() -> None:
    get_calib_dataloader_to_tensor(dataset_name="pileval", tokenizer=TOKENIZER, num_calib_data=1, seqlen=10)


def test_get_calib_dataloader_to_tensor_cnn_dailymail() -> None:
    get_calib_dataloader_to_tensor(
        dataset_name="abisee/cnn_dailymail", tokenizer=TOKENIZER, num_calib_data=1, seqlen=10
    )


def test_get_calib_dataloader_to_tensor_wikitext() -> None:
    get_calib_dataloader_to_tensor(dataset_name="Salesforce/wikitext", tokenizer=TOKENIZER, num_calib_data=1, seqlen=10)


def test_get_calib_dataloader_to_dict_cnn_dailymail() -> None:
    get_calib_dataloader_to_dict(dataset_name="abisee/cnn_dailymail", tokenizer=TOKENIZER, num_calib_data=1, seqlen=10)


def test_get_calib_dataloader_to_dict_wikitext() -> None:
    get_calib_dataloader_to_dict(dataset_name="Salesforce/wikitext", tokenizer=TOKENIZER, num_calib_data=1, seqlen=10)


def test_get_calib_dataloader_dispatches_namespaced() -> None:
    get_calib_dataloader(dataset_name="Salesforce/wikitext", tokenizer=TOKENIZER, num_calib_data=1, seqlen=10)


def test_get_trainer_dataset_wikitext() -> None:
    result = get_trainer_dataset(
        path="Salesforce/wikitext",
        subset="train",
        tokenizer=TOKENIZER,
        max_train_samples=2,
        max_eval_samples=2,
        seqlen=64,
    )
    assert "train_dataset" in result
    assert "eval_dataset" in result


def test_get_dataset_wikitext() -> None:
    dataset = get_dataset(path="Salesforce/wikitext", subset="train", tokenizer=TOKENIZER, seqlen=64)
    assert len(dataset) > 0
