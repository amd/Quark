import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.shares.utils.log import ScreenLogger
from quark.shares.utils.testing_utils import require_torch_higher_or_equal, slow, torch_device
from quark.testing.common_utils import skip_if_no_gpu
from quark.torch import ModelQuantizer
from quark.torch.quantization import Uint4PerChannelSpec
from quark.torch.quantization.config.config import (
    Config,
    GPTQConfig,
    OCP_MXFP4Spec,
    QuantizationConfig,
    Uint4PerGroupSpec,
)
from quark.torch.utils import getattr_recursive, setattr_recursive


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@skip_if_no_gpu
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("dtype", ["uint4", "mxfp4"])
@pytest.mark.parametrize("qscheme", ["per_group", "per_channel"])
@pytest.mark.parametrize("model_id", ["facebook/opt-125m", "HuggingFaceTB/SmolLM-135M"])
def test_gptq_correctness(act_order: bool, dtype: str, qscheme: str, model_id: str):
    n_layers = 2

    if "llama" in model_id or "SmolLM" in model_id:
        model_decoder_layers = "model.layers"
        inside_layer_modules = [
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
        ]
    else:
        model_decoder_layers = "model.decoder.layers"
        inside_layer_modules = [
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.out_proj",
            "fc1",
            "fc2",
        ]

    if dtype == "uint4":
        if qscheme == "per_group":
            qspec = Uint4PerGroupSpec(
                scale_type="float", ch_axis=1, is_dynamic=False, group_size=32
            ).to_quantization_spec()
        elif qscheme == "per_channel":
            # actorder has no influence in this case.
            qspec = Uint4PerChannelSpec(
                symmetric=False, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=False
            ).to_quantization_spec()
    else:
        if qscheme == "per_group":
            qspec = OCP_MXFP4Spec(is_dynamic=False).to_quantization_spec()
        else:
            pytest.skip("ocp mxfp4 not compatible with per_channel, per_tensor")

    global_quant_config = QuantizationConfig(weight=qspec)

    gptq_config = GPTQConfig(
        model_decoder_layers=model_decoder_layers,
        inside_layer_modules=inside_layer_modules,
        desc_act=act_order,
        block_size=32,
    )

    config_gptq = Config(global_quant_config=global_quant_config, algo_config=[gptq_config])
    config_no_algo = Config(global_quant_config=global_quant_config)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    text = "Hello, how are you?"
    tokenized_inputs = tokenizer(text, return_tensors="pt").to(torch_device)
    calib_dataloader = DataLoader(tokenized_inputs["input_ids"])

    model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
    model = model.eval()

    model2 = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
    model2 = model2.eval()

    # Make the model smaller just to speed up this test.
    decoder_layers = getattr_recursive(model, model_decoder_layers)
    setattr_recursive(model, model_decoder_layers, decoder_layers[:n_layers])
    model.config.num_hidden_layers = n_layers

    decoder_layers = getattr_recursive(model2, model_decoder_layers)
    setattr_recursive(model2, model_decoder_layers, decoder_layers[:n_layers])
    model2.config.num_hidden_layers = n_layers

    with torch.no_grad():
        logits_original = model(**tokenized_inputs).logits

        quantizer = ModelQuantizer(config_no_algo, multi_device=False)
        model = quantizer.quantize_model(model, calib_dataloader)
        logits_rtn = model(**tokenized_inputs).logits

        quantizer = ModelQuantizer(config_gptq, multi_device=False)
        model2 = quantizer.quantize_model(model2, calib_dataloader)
        logits_gptq = model2(**tokenized_inputs).logits

        # ensure gptq is closer to original than RTN
        assert (logits_original - logits_rtn).abs().max().item() > (logits_original - logits_gptq).abs().max().item()
