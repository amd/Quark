#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Verify data_preparation functions: namespaced HuggingFace dataset identifiers, plus the
blockwise-tuning pile-10k dataloader and its collate function."""

import pytest
import torch
from transformers import AutoTokenizer

from quark.torch.utils.llm.data_preparation import (
    _collate_blockwise_input_ids,
    get_calib_dataloader,
    get_calib_dataloader_to_dict,
    get_calib_dataloader_to_tensor,
    get_dataset,
    get_pile10k_dataloader,
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


def test_collate_blockwise_input_ids_squeezes_and_stacks() -> None:
    batch = [torch.arange(5).unsqueeze(0), torch.arange(5, 10).unsqueeze(0)]
    stacked = _collate_blockwise_input_ids(batch)
    assert stacked.shape == (2, 5)
    assert torch.equal(stacked[0], torch.arange(5))
    assert torch.equal(stacked[1], torch.arange(5, 10))


def test_collate_blockwise_input_ids_rejects_non_tensor() -> None:
    with pytest.raises(TypeError, match="Expected torch.Tensor"):
        _collate_blockwise_input_ids([torch.arange(5).unsqueeze(0), [1, 2, 3, 4, 5]])  # type: ignore[list-item]


def test_get_pile10k_dataloader() -> None:
    train_loader, val_loader = get_pile10k_dataloader(tokenizer=TOKENIZER, train_size=2, seqlen=8, batch_size=2)
    assert val_loader is None
    batches = list(train_loader)
    assert len(batches) == 1
    assert batches[0].shape == (2, 8)


def test_get_calib_dataloader_to_tensor_drops_remainder_by_default() -> None:
    """Unchanged default: a batch size that doesn't divide num_calib_data drops the remainder."""
    dataloader = get_calib_dataloader_to_tensor(
        dataset_name="Salesforce/wikitext", tokenizer=TOKENIZER, batch_size=2, num_calib_data=5, seqlen=10
    )
    assert sum(batch.shape[0] for batch in dataloader) == 4


def test_get_calib_dataloader_to_tensor_keeps_remainder_when_drop_last_false() -> None:
    """drop_last=False keeps every requested sample, including the shorter last batch."""
    dataloader = get_calib_dataloader_to_tensor(
        dataset_name="Salesforce/wikitext",
        tokenizer=TOKENIZER,
        batch_size=2,
        num_calib_data=5,
        seqlen=10,
        drop_last=False,
    )
    assert sum(batch.shape[0] for batch in dataloader) == 5


def test_get_calib_dataloader_forwards_drop_last() -> None:
    """get_calib_dataloader passes drop_last through to the tensor-dataset builder."""
    dataloader = get_calib_dataloader(
        dataset_name="Salesforce/wikitext",
        tokenizer=TOKENIZER,
        batch_size=2,
        num_calib_data=5,
        seqlen=10,
        drop_last=False,
    )
    assert sum(batch.shape[0] for batch in dataloader) == 5
