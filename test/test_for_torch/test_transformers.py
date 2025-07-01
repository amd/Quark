#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import huggingface_hub

from packaging import version

"""
Reference reproduction:

python3 quantize_quark.py --model_dir fxmarty/tiny-llama-fast-tokenizer --output_dir llama-tiny-w-int8-per-tensor --quant_scheme w_int8_per_tensor_sym --num_calib_data 128 --model_export quark_format --device cpu --group_size 4 --skip_evaluation

python3 quantize_quark.py --model_dir fxmarty/tiny-llama-fast-tokenizer --output_dir llama-tiny-int4-per-group-sym --quant_scheme w_int4_per_group_sym --num_calib_data 128 --model_export quark_format --device cpu --group_size 4 --skip_evaluation

python3 quantize_quark.py --model_dir fxmarty/tiny-llama-fast-tokenizer --output_dir llama-tiny-w-fp8-a-fp8 --quant_scheme w_fp8_a_fp8 --num_calib_data 128 --model_export quark_format --device cpu --group_size 4 --skip_evaluation

python3 quantize_quark.py --model_dir fxmarty/tiny-llama-fast-tokenizer --output_dir llama-tiny-w-fp8-a-fp8-o-fp8 --quant_scheme w_fp8_a_fp8_o_fp8 --num_calib_data 128 --model_export quark_format --device cpu --group_size 4 --skip_evaluation

python3 quantize_quark.py --model_dir fxmarty/small-llama-testing --output_dir llama-small-int4-per-group-sym-awq --quant_scheme w_int4_per_group_sym --num_calib_data 128 --model_export quark_format --device cpu --group_size 4 --skip_evaluation --quant_algo awq

`amd-quark/llama-tiny-w-int8-b-int8-per-tensor` is custom.

and logits before export are saved identically:
```
    # There is a bug in quark where the model is actually not quantize when calling ModelQuantizer.quantize_model first.
    from quark.torch.quantization.nn.modules.mixin import QuantMixin
    for module in model.modules():
        if isinstance(module, QuantMixin):
            if module._weight_quantizer is not None:
                module._weight_quantizer.disable_observer()
                module._weight_quantizer.enable_fake_quant()

                module.weight = torch.nn.Parameter(module.get_quant_weight(module.weight))
            if module._bias_quantizer is not None:
                module._bias_quantizer.disable_observer()
                module._bias_quantizer.enable_fake_quant()

                module.bias = torch.nn.Parameter(module.get_quant_bias(module.bias))

    inp = tokenizer("Today I am in Paris and I would like to", return_tensors="pt")

    model = model.eval()

    with torch.no_grad():
        logits_ref = model(**inp).logits

    torch.save(logits_ref, f"fxmarty-{args.output_dir}_ref_output.pt")
```
"""


@pytest.mark.parametrize("model_id", [
    pytest.param(model_id, id=model_id) for model_id in ["amd-quark/llama-tiny-w-int8-per-tensor", "amd-quark/llama-tiny-w-fp8-a-fp8-o-fp8", "amd-quark/llama-tiny-w-fp8-a-fp8", "amd-quark/llama-tiny-int4-per-group-sym", "amd-quark/llama-small-int4-per-group-sym-awq", "amd-quark/llama-tiny-w-int8-b-int8-per-tensor"]
])
def test_transformers_load(model_id: str):
    if version.parse(transformers.__version__) < version.parse("4.48"):
        pytest.skip("This test requires Quark support in Transformers")

    # We use attn_implementation="eager" here as the asset reference logits were originally computed without SDPA.
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", attn_implementation="eager")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    inp = tokenizer("Today I am in Paris and I would like to", return_tensors="pt")

    model = model.eval()

    with torch.no_grad():
        logits_reloaded = model(**inp).logits

    ref_filename = model_id.replace("/", "-") + "_ref_output.pt"
    file_path = huggingface_hub.hf_hub_download("amd-quark/quark-assets", ref_filename)
    logits_ref = torch.load(file_path, weights_only=True)

    # We should have an exact match between the reference logits obtained from a model before serialization, and from a model after serialization and reload. However, the reference logits were taken locally and for amd-quark/llama-small-int4-per-group-sym-awq we have a small numerical difference in the CI: maxabsdiff 5.9605e-07, due to the different hardware, although the torch.equal test passes locally.
    # We may want to generate the reference logits on the fly.
    if "awq" in model_id:
        assert (logits_reloaded - logits_ref).abs().max() < 1e-6
    else:
        assert torch.equal(logits_reloaded, logits_ref)
