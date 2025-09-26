#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Union

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(next(iter(self.encodings.values())))


def get_mlperf_data(
    data_path: str,
    tokenizer: AutoTokenizer = None,
    batch_size: int = 1,
    num_calib_data: int = 128,
    seqlen: int = 2048,
    device: str = "cpu",
) -> DataLoader[torch.Tensor]:
    print("mlperf calibration data path: ", data_path)

    with open(data_path, "rb") as fh:
        mlperf_df = pickle.load(fh)

    system_prompt_instruction = mlperf_df["input"].tolist()[:num_calib_data]

    batch_encoded = tokenizer.batch_encode_plus(
        system_prompt_instruction,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seqlen,
    )
    if device:
        batch_encoded = batch_encoded.to(device)

    tokenized_dataset = CustomDataset({"input_ids": batch_encoded["input_ids"]})

    calib_dataloader: DataLoader[list[dict[str, torch.Tensor]]] = DataLoader(
        tokenized_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )  # type: ignore

    return calib_dataloader


def get_mlperf_calib_dataloader(
    dataset_name: str, processor: AutoProcessor = None, **kwargs: Any
) -> Union[DataLoader[torch.Tensor], DataLoader[list[dict[str, torch.Tensor]]], DataLoader[dict[str, torch.Tensor]]]:
    if dataset_name.endswith("pkl"):
        return get_mlperf_data(dataset_name, **kwargs)
    else:
        raise NotImplementedError
