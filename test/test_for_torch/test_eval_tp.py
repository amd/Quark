#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
import os
import subprocess
from dataclasses import replace
from typing import Optional, Tuple

import pytest
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.testing_utils import (
    TestCasePlus,
    get_torch_dist_unique_port,
    require_torch_multi_gpu,
)

from quark.shares.utils.import_utils import is_transformers_version_higher_or_equal
from quark.shares.utils.log import ScreenLogger
from quark.shares.utils.testing_utils import torch_device
from quark.torch import ModelImporter, ModelQuantizer, export_safetensors, import_model_from_safetensors
from quark.torch.export.api import _move_quantizer_to_dict
from quark.torch.quantization.config.config import Config, GPTQConfig, QuantizationConfig, QuantizationSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver, PerTensorMinMaxObserver
from quark.torch.utils.device import TPDeviceManager

logger = ScreenLogger(__name__)
MODEL_DIR = "facebook/opt-125m"
torch.manual_seed(42)
INPUT_IDS = torch.randint(0, 1024, (1, 10)).to(torch_device)

GPTQ_CONFIG = GPTQConfig(
    model_decoder_layers="model.decoder.layers",
    inside_layer_modules=[
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_proj",
        "self_attn.out_proj",
        "fc1",
        "fc2",
    ],
)

UINT4_PER_GROUP_ASYM_SPEC = QuantizationSpec(
    dtype=Dtype.uint4,
    observer_cls=PerGroupMinMaxObserver,
    symmetric=False,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    qscheme=QSchemeType.per_group,
    ch_axis=1,
    is_dynamic=False,
    group_size=32,
)


@pytest.mark.skip(reason="Util-support")
def get_model(
    ckpt_path: str,
    data_type: str = "auto",
    device: str = "cuda",
    multi_gpu: bool = False,
    attn_implementation: str = "eager",
) -> tuple[nn.Module, torch.dtype]:
    if multi_gpu:
        device = "auto"
    if data_type == "float16":
        model_dtype = torch.float16
    elif data_type == "bfloat16":
        model_dtype = torch.bfloat16
    elif data_type == "float32":
        model_dtype = torch.float32
    elif data_type == "auto":
        model_dtype = data_type
    else:
        raise ValueError(f"{data_type} not support for current model")
    mllama_list = [
        "Llama-3.2-11B-Vision",
        "Llama-3.2-90B-Vision",
        "Llama-3.2-11B-Vision-Instruct",
        "Llama-3.2-90B-Vision-Instruct",
    ]
    model_name = os.path.basename(os.path.normpath(ckpt_path))
    if model_name in mllama_list:
        from transformers import MllamaForConditionalGeneration

        model = MllamaForConditionalGeneration.from_pretrained(
            ckpt_path,
            device_map=device,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                device_map=device,
                torch_dtype=model_dtype,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            )
        except Exception as e:
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path, device_map=device, torch_dtype=model_dtype, trust_remote_code=True
            )

    # For certain models, the attribute model.config._name_or_path is an empty string; enforce the setting here.
    model.config._name_or_path = ckpt_path

    model.eval()
    model_dtype = next(model.parameters()).dtype

    return model, model_dtype


@pytest.mark.skip(reason="Util-support")
def get_dataloader(model_name="facebook/opt-125m", device=torch_device):
    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"].to(device))
    return calib_dataloader


@pytest.mark.skip(reason="Support")
def quantize_model(quant_config, model_name="facebook/opt-125m", multi_gpu=False, device_map: str | None = "auto"):
    # Get quantizer
    quantizer = ModelQuantizer(quant_config)

    if multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, torch_dtype="auto", trust_remote_code=True
        )
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        model = model.to(torch_device)
    # Get dataloader, if multi_gpu, give the first layer's device
    calib_dataloader = get_dataloader(model_name, model.device)

    quant_model = quantizer.quantize_model(model, calib_dataloader)
    # Inference with quantized model
    for i in calib_dataloader:
        quant_model(i)
    quant_model = quantizer.freeze(quant_model)

    return quant_model


@pytest.mark.skip(reason="Support")
@torch.no_grad()
def ppl_eval(model: nn.Module, testenc: AutoTokenizer, dev: str, file_format: str = "hf_format") -> None:
    if file_format != "onnx_format":
        model.eval()
    # Set sequence length as 2048 for wikitext dataset evaluation
    seqlen_for_eval = 2048
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen_for_eval

    testenc = testenc.to(dev)
    nlls = []

    if file_format == "onnx_format":
        import onnxruntime_genai as og

        params = og.GeneratorParams(model)
        params.try_graph_capture_with_max_batch_size(1)
        search_options = {}
        search_options["max_length"] = seqlen_for_eval + 1
        params.set_search_options(**search_options)

    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * seqlen_for_eval) : ((i + 1) * seqlen_for_eval)].to(dev)
        if file_format == "onnx_format":
            # onnx model logits using oga
            params.input_ids = batch.cpu().numpy()
            generator = og.Generator(model, params)
            generator.compute_logits()
            shift_logits = torch.tensor(generator.get_output("logits")[0][:-1]).to(dev)
        else:
            lm_logits = model(batch)["logits"]
            shift_logits = lm_logits[:, :-1, :].contiguous()

        shift_labels = testenc[:, (i * seqlen_for_eval) : ((i + 1) * seqlen_for_eval)][:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen_for_eval
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen_for_eval))

    return ppl


@pytest.mark.skip(reason="Support")
def test_eval_tp(
    skip_mesh: bool,
    import_dir: str,
    import_file_format: str = "hf_format",
    data_type: str = "auto",
    model_attn_implementation: str = "eager",
) -> None:
    logger.info("\n[TP-INFO]: Loading model ...")

    device = "cpu"

    model, model_dtype = get_model(MODEL_DIR, data_type, device, True, model_attn_implementation)

    if not skip_mesh:
        TPDeviceManager.tp_mesh_init()

    if import_file_format in ["safetensors", "hf_format"]:
        model = import_model_from_safetensors(model, model_dir=import_dir, multi_device=False)
    else:
        importer = ModelImporter(model_info_dir=import_dir, saved_format=import_file_format)
        model = importer.import_model_info(model)

    _move_quantizer_to_dict(model.model)

    if skip_mesh:
        return

    device = TPDeviceManager._device
    tp_mesh = TPDeviceManager._tp_mesh

    logger.info(f"\n[TP-INFO]:TP init ... {device}, {tp_mesh}")
    model.tensor_parallel(tp_mesh)

    logger.info("\n[TP-INFO]: Evaluating ...")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
    )

    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    ppl = ppl_eval(model, testenc, device)
    logger.info(f"\n[TP-INFO] Perplexity: {ppl.item()}")


@pytest.mark.skip(reason="Support")
def test_load_multi_device(working_dir: str, weight_format: str):
    INT8_PER_TENSER_SPEC = QuantizationSpec(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )

    INT8_PER_TENSOR_CONFIG = QuantizationConfig(
        weight=INT8_PER_TENSER_SPEC,
        input_tensors=INT8_PER_TENSER_SPEC,
        output_tensors=INT8_PER_TENSER_SPEC,
        bias=INT8_PER_TENSER_SPEC,
    )
    quant_config = Config(global_quant_config=INT8_PER_TENSOR_CONFIG)

    EXCLUDE_LAYERS = ["lm_head", "*.gate", "*.shared_expert_gate"]
    quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)

    if working_dir is not None:
        with torch.inference_mode():
            quant_model = quantize_model(quant_config, model_name=MODEL_DIR, multi_gpu=False)
            export_safetensors(
                model=quant_model, output_dir=working_dir, weight_format=weight_format, pack_method="reorder"
            )


@pytest.mark.tensor_parallel
class TestTensorParallel(TestCasePlus):
    @require_torch_multi_gpu
    @pytest.mark.require_dual_gpu
    def test_tp(self, num_devices=2):
        num_devices = num_devices if num_devices < torch.cuda.device_count() else torch.cuda.device_count()
        """
        # Move the export to unit_tesh.sh
        ids = ""
        for i in range(num_devices):
            ids = ids + str(i) if len(ids) == 0 else ids + "," + str(i)
        os.system(f"export CUDA_VISIBLE_DEVICES={ids}")
        """
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
        logger.info(f"test_tp CUDA_VISIBLE_DEVICES=${cuda_visible_devices}")

        distributed_args = f"""--nproc_per_node={num_devices}
            --master_port={get_torch_dist_unique_port()}
            {self.test_file_dir}/test_eval_tp.py
        """.split()
        output_dir = self.get_auto_remove_tmp_dir()
        print(f"working dir: {output_dir}")
        test_load_multi_device(output_dir, "real_quantized")
        test_eval_tp(skip_mesh=True, import_dir=output_dir)

        args = f"--output_dir {output_dir}".split()
        cmd = ["torchrun"] + distributed_args + args
        logger.info(f"\n[TP-INFO]: CMD: {cmd}")
        if len(cmd) > 0:
            try:
                result = subprocess.run(cmd, capture_output=True, env=self.get_env(), text=True, check=True)
                logger.info(f"\n[TP-INFO]: Evaluating done: \n{result}")
            except subprocess.CalledProcessError as e:
                raise Exception(f"The following error was captured: {e.stderr}")

        """
        # successful return here == success - any errors would have caused an error in the sub-call
        """


if __name__ == "__main__":
    # The script below is meant to be run under torch.distributed, on a machine with multiple GPUs:
    # CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv ./test_eval_tp.py
    # or
    # PYTHONPATH="src" python -m torch.distributed.run --nproc_per_node 2 ./tests/tp/test_tp.py

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="A test program for tensor parallelism.")

    # Add arguments
    parser.add_argument("-o", "--output_dir", help="Quantized model file dir")

    # Parse arguments
    args = parser.parse_args()

    # TODO: update this test for the latest Transformers version, this test broke for
    # transformers>=4.50.
    # TP handling in Transformers was moved from `PretrainedModel.tensor_parallel` to `integrations/tensor_parallel.py`.
    # See https://github.com/huggingface/transformers/pull/36033.
    # See the Quark issue #2544.
    if not is_transformers_version_higher_or_equal("4.50"):
        test_eval_tp(skip_mesh=False, import_dir=args.output_dir)
