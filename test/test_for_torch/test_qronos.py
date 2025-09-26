import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from quark.shares.utils.log import ScreenLogger
from quark.shares.utils.testing_utils import require_torch_higher_or_equal, slow, torch_device
from quark.testing.common_utils import skip_if_no_gpu
from quark.torch import ModelQuantizer
from quark.torch.quantization import Uint4PerChannelSpec
from quark.torch.quantization.config.config import (
    Config,
    OCP_MXFP4Spec,
    QronosConfig,
    QuantizationConfig,
    Uint4PerGroupSpec,
)
from quark.torch.utils import getattr_recursive, setattr_recursive

logger = ScreenLogger(__name__)


def get_dataloader(device=torch_device):
    seq_length = 4
    tokenized_outputs = {}
    torch.manual_seed(42)
    tokenized_outputs["input_ids"] = torch.randint(10, (1, seq_length), device=device)
    tokenized_outputs["attention_mask"] = torch.ones((1, seq_length), dtype=torch.int64, device=device)
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"].to(device))
    return calib_dataloader


@require_torch_higher_or_equal("2.6")
@skip_if_no_gpu
@pytest.mark.parametrize("dtype", ["uint4", "mxfp4"])
@pytest.mark.parametrize("qscheme", ["per_group", "per_channel"])
@pytest.mark.parametrize("model_id", ["facebook/opt-125m", "HuggingFaceTB/SmolLM-135M"])
def test_qronos_basic_correctness(dtype: str, qscheme: str, model_id: str):
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
            qspec = Uint4PerChannelSpec(
                symmetric=False, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=False
            ).to_quantization_spec()
    else:
        if qscheme == "per_group":
            qspec = OCP_MXFP4Spec(is_dynamic=False).to_quantization_spec()
        else:
            pytest.skip("ocp mxfp4 not compatible with per_channel, per_tensor")

    global_quant_config = QuantizationConfig(weight=qspec)

    qronos_config = QronosConfig(
        model_decoder_layers=model_decoder_layers, inside_layer_modules=inside_layer_modules, block_size=32
    )

    config = Config(global_quant_config=global_quant_config, algo_config=[qronos_config])
    calib_dataloader = get_dataloader(torch_device)

    model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
    model = model.eval()

    # make the model smaller (2 layers instead of 16) just to speed up test.
    decoder_layers = getattr_recursive(model, model_decoder_layers)
    setattr_recursive(model, model_decoder_layers, decoder_layers[:n_layers])
    model.config.num_hidden_layers = n_layers

    logits_original = model(calib_dataloader.dataset).logits

    with torch.no_grad():
        # apply rtn
        quantizer = ModelQuantizer(config)
        model = quantizer._prepare_model(model)
        logits_rtn = model(calib_dataloader.dataset).logits

        # apply qronos
        model = quantizer._apply_advanced_quant_algo(model, calib_dataloader)
        logits_qronos = model(calib_dataloader.dataset).logits

        # ensure qronos is closer to original than RTN
        assert (logits_original - logits_rtn).abs().max().item() > (logits_original - logits_qronos).abs().max().item()


@slow
@require_torch_higher_or_equal("2.6")
@skip_if_no_gpu
@pytest.mark.parametrize("dtype", ["uint4", "mxfp4"])
@pytest.mark.parametrize("qscheme", ["per_group", "per_channel"])
def test_qronos_correctness(dtype: str, qscheme: str):
    model_id = "meta-Llama/Llama-3.2-3B"

    if dtype == "uint4":
        if qscheme == "per_group":
            qspec = Uint4PerGroupSpec(
                scale_type="float", ch_axis=1, is_dynamic=False, group_size=128
            ).to_quantization_spec()
        elif qscheme == "per_channel":
            qspec = Uint4PerChannelSpec(
                symmetric=False, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=False
            ).to_quantization_spec()
    else:
        if qscheme == "per_group":
            qspec = OCP_MXFP4Spec(is_dynamic=False).to_quantization_spec()
        else:
            pytest.skip("ocp mxfp4 not compatible with per_channel, per_tensor")

    global_quant_config = QuantizationConfig(weight=qspec)

    qronos_config = QronosConfig(
        model_decoder_layers="model.layers",
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
        ],
    )

    config = Config(global_quant_config=global_quant_config, algo_config=[qronos_config])
    calib_dataloader = get_dataloader(torch_device)

    model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
    model = model.eval()
    logits_original = model(calib_dataloader.dataset).logits

    with torch.no_grad():
        # apply rtn
        quantizer = ModelQuantizer(config)
        model = quantizer._prepare_model(model)
        logits_rtn = model(calib_dataloader.dataset).logits

        # apply qronos
        model = quantizer._apply_advanced_quant_algo(model, calib_dataloader)
        logits_qronos = model(calib_dataloader.dataset).logits

        # ensure qronos is closer to original than RTN
        assert (logits_original - logits_rtn).abs().max().item() > (logits_original - logits_qronos).abs().max().item()
