#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_utils.export_import_hf_model import import_hf_model

from llm_eval.evaluation import task_eval
from llm_utils.data_preparation import get_loader
from util.api import weight_only_quantize, full_finetune, export_safetensor

def run(args):

    # 1. Define original model
    print("\n[QUARK-INFO]: Loading Model and Tokenizer... ")
    torch_dtype = "auto" if args.skip_finetune else torch.bfloat16
    model_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": args.model_trust_remote_code}
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", **model_kwargs)

    tokenizer_kwargs = {"trust_remote_code": args.model_trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)

    # 2. (Optional) Reload quantized model that is exported by Quark
    if args.model_reload:
        print("\nRestore quantized model from hf_format file ...")
        model = import_hf_model(model, model_info_dir=args.import_model_dir)
        args.skip_quantization = True

    # 3. PTQ and load checkpoint (optional)
    if not args.skip_quantization:
        print("\n[QUARK-INFO]: Quantizing... ")
        calib_loader = get_loader('wikitext', 'wikitext-2-raw-v1', 'test', tokenizer, seqlen=2048, num_batch=1)
        model, quant_config = weight_only_quantize(model, calib_loader, args.quant_scheme, args.group_size)

        if args.quant_resume:
            model_state_file = os.path.join(args.finetune_checkpoint, 'best.pth')
            state = torch.load(model_state_file, weights_only=True, map_location="cuda")
            model.load_state_dict(state['state_dict'])
            print(f"\n[QUARK-INFO]: ReLoaded checkpoint from {model_state_file}")

    # 4. Finetuning
    if not args.skip_finetune:
        print("\n[QUARK-INFO]: Fine-Tuning... ")
        ft_loader = get_loader(args.finetune_dataset, args.finetune_datasubset, 'train', tokenizer,
                               seqlen=args.finetune_seqlen,
                               num_batch=args.finetune_iter,
                               batch_size=args.finetune_batchsize,
                               shuffle=True)
        optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.finetune_lr)
        epoch = args.finetune_epoch

        os.makedirs(args.finetune_checkpoint, exist_ok=True)
        full_finetune(model, tokenizer, ft_loader, optimizer, epoch, args.finetune_checkpoint)

    # 5. Export safetensors
    if args.model_export is not None:
        export_safetensor(model, args.model, quant_config, args.output_dir)

    # 6. Evaluation
    if not args.skip_evaluation:
        model.eval()
        dtype = args.quant_scheme if not args.skip_quantization else str(next(model.parameters()).dtype)
        print(f"\n[QUARK-INFO]: Evaluating ({dtype})... ")
        num_fewshot, apply_chat_template = None, False
        task_eval(model, tokenizer, args.eval_batch_size, args.max_eval_batch_size, args.eval_task, num_fewshot, apply_chat_template)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        help="Specify where the HuggingFace model is.",
                        default="THUDM/chatglm3-6b")
    parser.add_argument("--model_trust_remote_code", action='store_true')

    # Argument for quant
    parser.add_argument("--skip_quantization", action='store_true')
    parser.add_argument('--quant_resume', action='store_true')
    parser.add_argument("--quant_scheme",
                        help="Supported quant_scheme in the script. \
                            If there is no suitable quantization strategy among the options, \
                            users can customize the quantization configuration according to their own needs. \
                            If None, the model will be quantized by float16",
                        default="w_uint4_asym",
                        choices=[
                            "w_uint4_asym", "w_int4_sym"
                        ])
    parser.add_argument("--group_size", help="Group size for per_group quantization.", type=int, default=128)
    parser.add_argument("--kv_cache_dtype", help="KV Cache dtype.", default=None, choices=["fp8", None])

    # Argument for finetune
    parser.add_argument("--skip_finetune", action='store_true')
    parser.add_argument('--finetune_dataset', type=str, default='shibing624/AdvertiseGen')
    parser.add_argument('--finetune_datasubset', type=str, default='default')
    parser.add_argument('--finetune_epoch', type=int, default=10)
    parser.add_argument('--finetune_lr', type=float, default=2e-5)
    parser.add_argument('--finetune_iter', type=int, default=500)
    parser.add_argument('--finetune_seqlen', type=int, default=512)
    parser.add_argument('--finetune_batchsize', type=int, default=8)
    parser.add_argument("--finetune_checkpoint", default="finetune_checkpoint")

    # Argument for export
    parser.add_argument("--model_export", help="Model export format", default=None, action="append", choices=[None, "onnx", "quark_format", "hf_format", "gguf"])
    parser.add_argument("--custom_mode", help="When selecting `--custom_mode awq` or `--custom_mode fp8`, this legacy argument allows to export FP8 and AWQ models in the custom format they were exported with with quark<1.0, with custom config saved in the config.json, and config checkpoint format (AWQ uses `qzeros`, `qweight`, transposed `scales`).", default="quark", type=str, choices=["quark", "awq", "fp8"])
    parser.add_argument("--pack_method", type=str, help="Pack method for awq_export", default="reorder", choices=["order", "reorder"])
    parser.add_argument("--output_dir", default="exported_model")
    parser.add_argument("--weight_matrix_merge", help="Whether to merge weight matrix when dump llm-specific quantized model", action='store_true')
    parser.add_argument("--export_weight_format", type=str, help="Whether to export weights compressed or uncompressed", default="real_quantized", choices=["fake_quantized", "real_quantized"])

    # Argument for reloading
    parser.add_argument("--model_reload", help="safetensors or pth model reload", action="store_true")
    parser.add_argument("--import_file_format", type=str, help="file_format for importing. If you export hf_format, you should use 'hf_format' for reloading.", default="hf_format", choices=["quark_format", "hf_format"])
    parser.add_argument("--import_model_dir", type=str, help="directory of hf or quark model", default=None)

    # Argument for evaluation
    parser.add_argument("--skip_evaluation", action='store_true')
    parser.add_argument("--eval_task", default='wikitext', type=str, metavar="task1,task2", help="Comma-separated list of task names or task groupings to evaluate on.")
    parser.add_argument("--eval_batch_size",
                        type=str,
                        default=1,
                        metavar="auto|auto:N|N",
                        help="Batch size for evaluation. Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.")
    parser.add_argument("--max_eval_batch_size",
                        type=int,
                        default=None,
                        metavar="N",
                        help="Maximal batch size to try with --batch_size auto.")

    args = parser.parse_args()

    msg = '\n'.join([f'{k:<26}: {v}' for k, v in vars(args).items()])
    print(f"\n{msg}")

    run(args)
