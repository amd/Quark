#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
import json
import os
import sys
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from quark.torch.pruning.config import Config, LayerImportancePruneConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_eval.evaluation import eval_model
from llm_utils.model_preparation import get_model, get_model_type, save_model, set_seed


def get_config(args: argparse.Namespace, model_type: str) -> Config:
    algo_config_file = "models/" + model_type + "/layer_importance_config.json"
    with open(algo_config_file) as file:
        algo_config_info = json.load(file)
    pruning_algo_config = LayerImportancePruneConfig.from_dict(algo_config_info)
    blockwise_tuning_config = None

    pruning_config = Config(algo_config=pruning_algo_config, blockwise_tuning_config=blockwise_tuning_config)
    return pruning_config


def get_wikitext_dataset(model_dir: str, dev: torch.device, **kwargs: Any):
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    seqlen_for_eval = 2048
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen_for_eval
    batch_data = []
    testenc = testenc.to(dev)
    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * seqlen_for_eval) : ((i + 1) * seqlen_for_eval)].to(dev)
        batch_data.append(batch)
    return batch_data


def main(args: argparse.Namespace) -> None:
    # 1. Define original model
    print("\n[INFO]: Loading model ...")
    set_seed(args.seed)
    model, model_dtype = get_model(args.model_dir, args.data_type, args.device, args.multi_gpu)
    model_type = get_model_type(model)

    from quark.shares.utils.log import ScreenLogger

    logger = ScreenLogger(__name__)

    # 2. Define calibration dataloader.
    print("\n[INFO]: Loading dataset ...")
    # When the model is small, accelerate will place it on the last device
    main_device = model.device if args.multi_gpu else args.device
    validation_dataloader = get_wikitext_dataset(args.model_dir, main_device)
    # 3. Pruning
    if not args.skip_pruning:
        # 3-1. Set pruning configuration
        pruning_config = get_config(args, model_type)

        # 3-2. In-place replacement of model modules with pruning versions.
        from quark.torch import ModelPruner

        model_pruner = ModelPruner(pruning_config)
        model = model_pruner.pruning_model(model, validation_dataloader)

        # 4. Export
        if args.save_pruned_model:
            print("\n[INFO]: Save pruned model ...")
            save_model(model, None, args.save_dir)

    # 5. (Optional) Model Evaluation
    if not args.skip_evaluation:
        print("\n[INFO]: Evaluating ...")
        eval_model(args, model, main_device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        help="Specify where the HuggingFace model is. This example support Llama, OPT models",
        required=True,
    )
    parser.add_argument("--device", help="Device for running the pruner", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument(
        "--data_type", help="Datatype of the model", default="auto", choices=["auto", "float16", "bfloat16", "float32"]
    )
    parser.add_argument("--skip_pruning", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")

    parser.add_argument("--save_pruned_model", help="pruned model save", action="store_true")
    parser.add_argument(
        "--save_dir",
        help="Directory to save model parameters as safetensors or pth, in the case when --save_pruned_model is used.",
        default="model_params",
    )

    parser.add_argument("--seed", type=int, help="random seed.", default=42)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--trust_remote_code",
        action="store_true",
        dest="trust_remote_code",
        help="Enable execution of custom model code from the Hub (use only with repositories you fully trust).",
    )
    group.add_argument(
        "--no_trust_remote_code",
        action="store_false",
        dest="trust_remote_code",
        help="Disable execution of custom model code from the Hub (safer, recommended if unsure).",
    )
    parser.set_defaults(trust_remote_code=True)

    args = parser.parse_args()

    main(args)
