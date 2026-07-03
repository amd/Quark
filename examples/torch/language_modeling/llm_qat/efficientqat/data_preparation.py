#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import logging
import os
import random
from typing import TYPE_CHECKING

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def get_c4(
    tokenizer: AutoTokenizer,
    train_size: int = 2048,
    val_size: int = 64,
    seed: int = 0,
    seqlen: int = 2048,
    test_only: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]] | torch.Tensor:
    logger.info("Getting C4 dataset...")
    c4_data_dir = os.environ.get("C4_DATA_DIR")
    if c4_data_dir:
        train_file = os.path.join(c4_data_dir, "c4-train.00000-of-01024.jsonl")
        traindata = load_dataset("json", data_files={"train": train_file}, split="train")
        if test_only:
            val_file = os.path.join(c4_data_dir, "c4-train.00001-of-01024.jsonl")
            valdata = load_dataset("json", data_files={"train": val_file}, split="train")
    else:
        traindata = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        if test_only:
            valdata = load_dataset(
                "allenai/c4",
                data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                split="validation",
            )

    if test_only:
        random.seed(0)
        valenc: list[torch.Tensor] = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            valenc.append(tmp.input_ids[:, i:j])
        return torch.hstack(valenc)

    random.seed(seed)
    train_samples: list[torch.Tensor] = []
    val_samples: list[torch.Tensor] = []
    val_sample_ratio = 0.9

    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        train_samples.append(trainenc.input_ids[:, i:j])

    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata) * val_sample_ratio), len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        val_samples.append(trainenc.input_ids[:, i:j])

    logger.info("Getting C4 dataset finished")
    return train_samples, val_samples


def get_redpajama(
    tokenizer: AutoTokenizer,
    train_size: int,
    val_size: int,
    seed: int,
    seqlen: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    logger.info("Getting RedPajama dataset...")
    traindata = load_dataset(
        "togethercomputer/RedPajama-Data-1T",
        "default",
        split="train[:100]",
    )
    random.seed(seed)
    traindata = traindata.shuffle(seed=seed)

    train_samples: list[torch.Tensor] = []
    val_samples: list[torch.Tensor] = []
    val_sample_ratio = 0.9

    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        train_samples.append(trainenc.input_ids[:, i:j])

    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata) * val_sample_ratio), len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        val_samples.append(trainenc.input_ids[:, i:j])

    logger.info("Getting RedPajama dataset finished")
    return train_samples, val_samples


def get_blockwise_tuning_dataloader(
    name: str,
    tokenizer: AutoTokenizer,
    train_size: int = 4096,
    val_size: int = 64,
    seed: int = 0,
    seqlen: int = 2048,
    test_only: bool = False,
    batch_size: int = 2,
) -> tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
    if "c4" in name:
        train_data, val_data = get_c4(tokenizer, train_size, val_size, seed, seqlen, test_only)
        if test_only:
            raise ValueError("test_only=True is not supported for blockwise tuning dataloader.")
    elif "redpajama" in name:
        train_data, val_data = get_redpajama(tokenizer, train_size, val_size, seed, seqlen)
    else:
        raise NotImplementedError(f"Unsupported dataset: {name}")

    def _collate_input_ids(batch: list[torch.Tensor]) -> torch.Tensor:
        squeezed: list[torch.Tensor] = []
        for t in batch:
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor in blockwise tuning dataloader, got {type(t)}")
            if t.ndim == 2 and t.shape[0] == 1:
                t = t.squeeze(0)
            squeezed.append(t)
        return torch.stack(squeezed, dim=0)

    train_loader: DataLoader[torch.Tensor] = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=_collate_input_ids,
    )
    val_loader: DataLoader[torch.Tensor] = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=_collate_input_ids,
    )
    return train_loader, val_loader
