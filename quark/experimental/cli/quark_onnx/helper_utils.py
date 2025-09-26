#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Helper functions taken from examples/onnx required to convert ONNX example workflows into CLI workflows.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
from datasets import load_dataset
from onnxruntime.quantization import CalibrationDataReader
from timm.data import create_transform, resolve_data_config
from timm.models import create_model
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, PreTrainedTokenizer


def get_model_input_name(input_model_path: str) -> str:
    model = onnx.load(input_model_path)
    model_input_name = model.graph.input[0].name
    return model_input_name


def parse_subgraphs_list(exclude_subgraphs: str) -> list[tuple[list[str]]]:
    subgraphs_list = []
    tuples = exclude_subgraphs.split(";")
    for tup in tuples:
        tup = tup.strip()
        pattern = r"\[.*?\]"
        matches = re.findall(pattern, tup)
        assert len(matches) == 2
        start_nodes = matches[0].strip("[").strip("]").split(",")
        start_nodes = [node.strip() for node in start_nodes]
        end_nodes = matches[1].strip("[").strip("]").split(",")
        end_nodes = [node.strip() for node in end_nodes]
        subgraphs_list.append((start_nodes, end_nodes))
    return subgraphs_list


def load_dataloader(
    image_folder: str, model_name: str | None = None, image_dim: int | None = None, batch_size: int | None = None
):
    data_transform = None

    if image_dim is not None:
        data_transform = transforms.Compose(
            [
                transforms.Resize(image_dim),
                transforms.CenterCrop(image_dim),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    if model_name is not None:
        try:
            timm_model = create_model(
                model_name,
                pretrained=False,
            )
            data_config = resolve_data_config(model=timm_model, use_test_size=True)
            data_transform = create_transform(**data_config)
        except Exception:
            pass

    dataset = torchvision.datasets.ImageFolder(image_folder, data_transform)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)


class ImageDataReader(CalibrationDataReader):
    def __init__(
        self,
        calibration_image_folder: str,
        input_name: str,
        model_name: str | None = None,
        image_dim: int | None = None,
        batch_size: int | None = None,
    ):
        self.input_name = input_name
        self.iterator = iter(
            load_dataloader(
                image_folder=calibration_image_folder, model_name=model_name, image_dim=image_dim, batch_size=batch_size
            )
        )

    def get_next(self):
        try:
            image = next(self.iterator)[0]
            data = np.array(image)
            data = np.expand_dims(data, axis=0)
            return {self.input_name: data}
        except Exception:
            return None


class BasicDataReader:
    def __init__(self, dataloader):
        super().__init__()
        self.iterator = iter(dataloader)

    def get_next(self) -> dict:
        try:
            return {"input": next(self.iterator)[0].numpy()}
        except Exception:
            return None


class NpyDataReader(BasicDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str, data_size: int = 100, batch_size: int = 1):
        self.enum_data = None
        # Use inference session to get input shape.
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.nhwc_data_list = []
        self.all_files = os.listdir(calibration_image_folder)
        self.all_files = [item for item in self.all_files if item.endswith(".npy")]
        if data_size > len(self.all_files):
            data_size = len(self.all_files)
        for i in range(data_size):
            one_item_path = os.path.join(calibration_image_folder, f"sample_{i}.npy")
            one_item = np.load(one_item_path)
            self.nhwc_data_list.append(one_item)

        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list])
        return next(self.enum_data, None)

    def get_item(self, idx):
        if idx < self.datasize:
            temp_data = self.nhwc_data_list[idx]
        else:
            pass
        return {self.input_name: temp_data}

    def __getitem__(self, idx):
        return {self.input_name: self.nhwc_data_list[idx]}

    def __len__(
        self,
    ):
        return self.datasize

    def rewind(self):
        self.enum_data = None

    def reset(self):
        self.enum_data = None


def get_pileval(
    tokenizer: PreTrainedTokenizer, nsamples: int, seqlen: int, device: str | None, seed: int = 0
) -> list[dict[str, torch.Tensor]]:
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation", cache_dir="data_cache")
    dataset = dataset.shuffle(seed=seed)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        sample = sample.to(device)
        samples.append(sample)
        n_run += 1
        if n_run == nsamples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // seqlen
    logging.debug(f" * Split into {n_split} blocks")
    traindataset = []
    for i in range(n_split):
        traindataset.append({"input_ids": cat_samples[:, i * seqlen : (i + 1) * seqlen]})
    return traindataset


def get_wikitext2(
    tokenizer: PreTrainedTokenizer, nsamples: int, seqlen: int, device: str | None, seed: int = 0
) -> list[dict[str, torch.Tensor]]:
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir="data_cache")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    trainenc = trainenc.to(device)

    import random

    random.seed(seed)
    torch.random.manual_seed(seed)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset


def get_calib_dataloader_to_list(
    dataset_name: str = "pileval_for_awq_benchmark",
    tokenizer: AutoTokenizer = None,
    batch_size: int | None = None,
    num_calib_data: int = 128,
    seqlen: int = 2048,
    device: str = "cpu",
) -> DataLoader[list[dict[str, torch.Tensor]]]:
    if dataset_name == "pileval_for_awq_benchmark":
        samples = get_pileval(tokenizer, num_calib_data, seqlen, device, seed=42)
    elif dataset_name == "wikitext_for_gptq_benchmark":
        samples = get_wikitext2(tokenizer, num_calib_data, seqlen, device)
    else:
        raise NotImplementedError

    calib_dataloader: DataLoader[list[dict[str, torch.Tensor]]] = DataLoader(
        samples, batch_size=batch_size, shuffle=False
    )  # type: ignore

    return calib_dataloader


def get_calib_dataloader_to_tensor(
    dataset_name: str = "cnn_dailymail",
    tokenizer: AutoTokenizer = None,
    batch_size: int = 1,
    num_calib_data: int = 512,
    seqlen: int = 512,
    device: str | None = None,
) -> DataLoader[torch.Tensor]:
    if dataset_name == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation", cache_dir="data_cache")
        text_data = dataset["text"][:num_calib_data]
    elif dataset_name == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train", cache_dir="data_cache")
        text_data = dataset["article"][:num_calib_data]
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir="data_cache")
        text_data = dataset["text"][:num_calib_data]
    else:
        raise NotImplementedError

    batch_encoded = tokenizer(text_data, return_tensors="pt", padding=True, truncation=True, max_length=seqlen)
    if device:
        batch_encoded = batch_encoded.to(device)
    batch_encoded = batch_encoded["input_ids"]

    calib_dataloader = DataLoader(batch_encoded, batch_size=batch_size, shuffle=False)

    return calib_dataloader


def get_calib_dataloader_to_dict(
    dataset_name: str = "cnn_dailymail",
    tokenizer: AutoTokenizer = None,
    batch_size: int = 1,
    num_calib_data: int = 512,
    seqlen: int = 512,
    device: str | None = None,
) -> DataLoader[dict[str, torch.Tensor]]:
    def make_data_block(
        examples: dict[str, list[str]],
        tokenizer: AutoTokenizer = None,
        prompt_col_name: str = "",
        max_length: int = 512,
    ) -> dict[str, list[list[torch.Tensor]]]:
        res: dict[str, list[list[torch.Tensor]]] = tokenizer(
            examples[prompt_col_name], padding=True, truncation=True, max_length=max_length
        )
        return res

    def my_collate_fn(blocks: list[dict[str, list[list[str]]]]) -> dict[str, torch.Tensor]:
        data_batch = {}
        data_batch["input_ids"] = torch.Tensor([block["input_ids"] for block in blocks])
        if device:
            data_batch["input_ids"] = data_batch["input_ids"].to(device)
        return data_batch

    if dataset_name == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation", cache_dir="data_cache")
        prompt_col_name = "text"
    elif dataset_name == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train", cache_dir="data_cache")
        prompt_col_name = "article"
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir="data_cache")
        prompt_col_name = "text"
    else:
        raise NotImplementedError

    dataset = dataset.select(
        indices=[i for i in range(min(len(dataset), num_calib_data))],
        keep_in_memory=True,
    )
    tokenized_datasets = dataset.map(
        make_data_block,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        remove_columns=dataset.column_names,
        keep_in_memory=True,
        fn_kwargs={"tokenizer": tokenizer, "prompt_col_name": prompt_col_name, "max_length": seqlen},
    )

    calib_dataloader = DataLoader(tokenized_datasets, batch_size=batch_size, collate_fn=my_collate_fn)

    return calib_dataloader


def get_calib_dataloader(
    dataset_name: str, **kwargs: Any
) -> Union[DataLoader[torch.Tensor], DataLoader[list[dict[str, torch.Tensor]]], DataLoader[dict[str, torch.Tensor]]]:
    if dataset_name in ["pileval", "cnn_dailymail"]:
        return get_calib_dataloader_to_tensor(dataset_name, **kwargs)
    elif dataset_name in ["pileval_for_awq_benchmark", "wikitext_for_gptq_benchmark"]:
        return get_calib_dataloader_to_list(dataset_name, **kwargs)
    else:
        raise NotImplementedError


kv_cache_name = [
    "past_key_values.0.key",
    "past_key_values.0.value",
    "past_key_values.1.key",
    "past_key_values.1.value",
    "past_key_values.2.key",
    "past_key_values.2.value",
    "past_key_values.3.key",
    "past_key_values.3.value",
    "past_key_values.4.key",
    "past_key_values.4.value",
    "past_key_values.5.key",
    "past_key_values.5.value",
    "past_key_values.6.key",
    "past_key_values.6.value",
    "past_key_values.7.key",
    "past_key_values.7.value",
    "past_key_values.8.key",
    "past_key_values.8.value",
    "past_key_values.9.key",
    "past_key_values.9.value",
    "past_key_values.10.key",
    "past_key_values.10.value",
    "past_key_values.11.key",
    "past_key_values.11.value",
    "past_key_values.12.key",
    "past_key_values.12.value",
    "past_key_values.13.key",
    "past_key_values.13.value",
    "past_key_values.14.key",
    "past_key_values.14.value",
    "past_key_values.15.key",
    "past_key_values.15.value",
    "past_key_values.16.key",
    "past_key_values.16.value",
    "past_key_values.17.key",
    "past_key_values.17.value",
    "past_key_values.18.key",
    "past_key_values.18.value",
    "past_key_values.19.key",
    "past_key_values.19.value",
    "past_key_values.20.key",
    "past_key_values.20.value",
    "past_key_values.21.key",
    "past_key_values.21.value",
    "past_key_values.22.key",
    "past_key_values.22.value",
    "past_key_values.23.key",
    "past_key_values.23.value",
    "past_key_values.24.key",
    "past_key_values.24.value",
    "past_key_values.25.key",
    "past_key_values.25.value",
    "past_key_values.26.key",
    "past_key_values.26.value",
    "past_key_values.27.key",
    "past_key_values.27.value",
    "past_key_values.28.key",
    "past_key_values.28.value",
    "past_key_values.29.key",
    "past_key_values.29.value",
    "past_key_values.30.key",
    "past_key_values.30.value",
    "past_key_values.31.key",
    "past_key_values.31.value",
]


class LLMDataReader:
    def __init__(self, dataloader, hidden_size, num_head, input_names):
        super().__init__()
        self.iterator = iter(dataloader)
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.input_names = input_names

    def get_next(self) -> dict:
        try:
            inputs = next(self.iterator)

            input_dict = {}

            if "position_ids" in self.input_names:
                input_dict["position_ids"] = torch.arange(inputs.size(1), dtype=torch.long).unsqueeze(0).numpy()

            if "input_ids" in self.input_names:
                input_dict["input_ids"] = inputs[0].numpy().reshape(1, -1)

            if "attention_mask" in self.input_names:
                input_dict["attention_mask"] = np.ones_like(inputs[0].numpy().reshape(1, -1))

            past_seq_len = 1  # For fake usage, this can be set as 1
            if self.num_head == 8:
                cache_shape = [
                    1,
                    8,
                    past_seq_len,
                    self.hidden_size // 32,
                ]  # Shape like [batch, num_head, past_seq_len, hidd_dim // num_head]
            elif self.num_head == 32:
                cache_shape = [1, 32, past_seq_len, self.hidden_size // 32]
            else:
                raise NotImplementedError
            for name in kv_cache_name:
                if name in self.input_names:
                    input_dict[name] = np.ones(cache_shape).astype(np.float32)  # Used only oga model  # no effect
            return input_dict
        except StopIteration:
            return None


def get_calib_dataset(
    input_model_path: str,
    dataset_name: str,
    num_calib_data: int,
    device: str,
    hidden_size: int,
    num_head: int,
    input_names: list[str],
) -> LLMDataReader:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.dirname(input_model_path),
            do_lower_case=False,
            cache_dir=None,
        )

        if tokenizer.pad_token != "<unk>":
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        calib_dataloader = get_calib_dataloader(
            dataset_name="pileval",
            tokenizer=tokenizer,
            batch_size=1,
            seqlen=512,
            device=device,
            num_calib_data=num_calib_data,
        )
        return LLMDataReader(calib_dataloader, hidden_size, num_head, input_names)
    except NotImplementedError:
        return None


def is_number(s):
    return bool(re.match(r"^-?\d+(?:\.\d+)?$", s))


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
