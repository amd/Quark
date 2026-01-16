#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import gc
import math
import time
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.shares.utils.testing_utils import PatchEverywhere, require_torch_higher_or_equal, slow, torch_device
from quark.testing import skip_if_no_gpu
from quark.torch.algorithm.gptq.gptq import fasterquant_inner_graph, record_graphs, replay_fasterquant_inner_graphs
from quark.torch.export.nn.modules import realquantizer
from quark.torch.quantization import (
    FP4PerGroupSpec,
    GPTQConfig,
    OCP_MXFP4Spec,
    QConfig,
    QLayerConfig,
    Uint4PerChannelSpec,
    Uint4PerGroupSpec,
)
from quark.torch.quantization.api import ModelQuantizer
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase, ScaledFakeQuantize


@skip_if_no_gpu
def test_realquantizer():
    torch.manual_seed(42)

    qspec = FP4PerGroupSpec(
        ch_axis=-1, group_size=32, scale_format="e8m0", scale_calculation_mode="even", is_dynamic=True
    ).to_quantization_spec()

    input_quantizer = realquantizer.get_real_quantizer(
        qspec=qspec, quantizer=None, real_quantized=False, float_dtype=torch.bfloat16, device=torch_device
    )

    x = torch.randn(256, 11008, device=torch_device, dtype=torch.bfloat16)

    qdqx = input_quantizer(x)
    assert not torch.isinf(qdqx).any()

    qdqx_2 = input_quantizer(x)
    assert not torch.isinf(qdqx_2).any()

    torch.testing.assert_close(qdqx, qdqx_2)

    g = torch.cuda.CUDAGraph()

    with torch.cuda.graph(g):
        qdqx_graph = input_quantizer(x)
    g.replay()

    assert not torch.isinf(qdqx_graph).any()
    torch.testing.assert_close(qdqx, qdqx_graph)


def run_fasterquant_inner_graphs_eager(inputs: dict[str, Any], columns: int):
    for _ in range(columns // inputs["columns_per_graph"]):
        fasterquant_inner_graph(**inputs)
        inputs["columns_start_idx"] += inputs["columns_per_graph"]


@skip_if_no_gpu
def test_gptq_fasterquant_inner_correctness():
    rows, columns = 256, 4096
    actorder = True
    blocksize = 128
    group_size = 32
    percdamp = 0.01

    qspec = Uint4PerChannelSpec(ch_axis=0, is_dynamic=False).to_quantization_spec()

    W = (torch.rand(rows, columns, device=torch_device) - 0.5) * 0.2

    H = torch.zeros((columns, columns), device=torch_device, dtype=torch.float)

    nsamples = 10

    for _ in range(nsamples):
        inp = torch.rand(1, columns, device=torch_device) - 0.5

        tmp = inp.shape[0]
        inp = inp.t()
        H *= nsamples / (nsamples + tmp)
        nsamples += tmp
        inp = math.sqrt(2 / nsamples) * inp.float()
        H += inp.matmul(inp.t())

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
    else:
        perm = None

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=torch_device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H.contiguous()

    qspec_per_group = Uint4PerGroupSpec(group_size=group_size, ch_axis=1, is_dynamic=False).to_quantization_spec()

    quantizer = FakeQuantizeBase.get_fake_quantize(quant_spec=qspec_per_group, device=torch_device)

    scale = []
    zero = []

    for i in range(0, columns, group_size):
        quantizer_ = copy.deepcopy(quantizer)
        quantizer_.observe(W[:, i : (i + group_size)])

        scale.append(quantizer_.scale)
        zero.append(quantizer_.zero_point)

    scales = torch.cat([s.view(-1, 1) for s in scale], dim=1)
    zero_points = torch.cat([z.view(-1, 1) for z in zero], dim=1)

    quantizer = FakeQuantizeBase.get_fake_quantize(quant_spec=qspec, device=torch_device)

    inputs = {
        "Hinv": Hinv,
        "W": W,
        "Q": torch.zeros_like(W),
        "perm": perm,
        "actorder": actorder,
        "blocksize": blocksize,
        "group_size": group_size,
    }

    inputs["scales"] = scales
    inputs["zero_points"] = zero_points
    inputs["quantizer"] = quantizer
    inputs["columns_start_idx"] = 0
    inputs["columns_per_graph"] = 1024

    ref_inputs = {name: inp.clone() if isinstance(inp, torch.Tensor) else inp for name, inp in inputs.items()}

    run_fasterquant_inner_graphs_eager(inputs, columns=columns)
    res_eager = inputs["Q"]

    inputs = {name: inp.clone() if isinstance(inp, torch.Tensor) else inp for name, inp in ref_inputs.items()}

    run_fasterquant_inner_graphs_eager(inputs, columns=columns)
    res_eager_2 = inputs["Q"]

    assert torch.equal(res_eager, res_eager_2)

    inputs = {name: inp.clone() if isinstance(inp, torch.Tensor) else inp for name, inp in ref_inputs.items()}

    graphs = record_graphs(inputs, columns=columns, device=torch_device)

    for name, val in ref_inputs.items():
        if isinstance(val, torch.Tensor):
            if name != "Hinv":
                assert val.is_contiguous()
                assert inputs[name].is_contiguous()
            assert inputs[name].dtype == val.dtype
            assert inputs[name].shape == val.shape

            inputs[name].copy_(val)
        elif name == "columns_start_idx":
            pass
        elif not isinstance(val, ScaledFakeQuantize):
            assert inputs[name] == val

    res_graph = replay_fasterquant_inner_graphs(graphs, inputs)

    absdiff = (res_graph - res_eager).abs()
    assert torch.equal(res_eager, res_graph), f"Max absdiff: {absdiff.max()}"


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@skip_if_no_gpu
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("dtype", ["uint4", "mxfp4"])
@pytest.mark.parametrize("qscheme", ["per_group", "per_channel"])
def test_gptq_cuda_graph_global_correctness(act_order: bool, dtype: str, qscheme: str):
    model_id = "facebook/opt-125m"
    n_layers = 2

    if dtype == "uint4":
        if qscheme == "per_group":
            qspec = Uint4PerGroupSpec(ch_axis=1, is_dynamic=False, group_size=128).to_quantization_spec()
        elif qscheme == "per_channel":
            # actorder has no influence in this case.
            qspec = Uint4PerChannelSpec(ch_axis=0, is_dynamic=False).to_quantization_spec()
    else:
        if qscheme == "per_group":
            qspec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
        else:
            pytest.skip("ocp mxfp4 not compatible with per_channel, per_tensor")

    global_quant_config = QLayerConfig(weight=qspec)

    gptq_config = GPTQConfig(
        model_decoder_layers="model.decoder.layers",
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.out_proj",
            "fc1",
            "fc2",
        ],
        desc_act=act_order,
    )

    config = QConfig(global_quant_config=global_quant_config, algo_config=[gptq_config])

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    text = "Hello, how are you?"
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"])

    model_graph = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto").to(torch_device)
    model_graph = model_graph.eval()

    # Make the model smaller just to speed up this test.
    model_graph.model.decoder.layers = model_graph.model.decoder.layers[:n_layers]
    model_graph.config.num_hidden_layers = n_layers

    with torch.no_grad():
        # Run GPTQ using CUDA Graph.
        quantizer = ModelQuantizer(config, multi_device=False)
        model_graph = quantizer.quantize_model(model_graph, calib_dataloader)

        # Run GPTQ without using CUDA Graph.
        with PatchEverywhere("QUARK_DISABLE_CUDA_GRAPH", True, module_name_prefix="quark"):
            model_no_graph = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto").to(torch_device)
            model_no_graph = model_no_graph.eval()

            model_no_graph.model.decoder.layers = model_no_graph.model.decoder.layers[:n_layers]
            model_no_graph.config.num_hidden_layers = n_layers

            quantizer = ModelQuantizer(config, multi_device=False)
            model_no_graph = quantizer.quantize_model(model_no_graph, calib_dataloader)

        model_params_no_graph = {name: param for name, param in model_no_graph.named_parameters()}
        model_params_no_graph = model_params_no_graph | {name: param for name, param in model_no_graph.named_buffers()}

        for name, param in model_graph.named_parameters():
            absdiff = (param - model_params_no_graph[name]).abs()
            assert torch.equal(model_params_no_graph[name], param), f"{name} max absdiff: {absdiff.max()}"

        for name, param in model_graph.named_buffers():
            absdiff = (param - model_params_no_graph[name]).abs()
            assert torch.equal(model_params_no_graph[name], param), f"{name} max absdiff: {absdiff.max()}"


@slow
@skip_if_no_gpu
@pytest.mark.parametrize("model_id", ["facebook/opt-6.7b", "meta-llama/Llama-2-70b-chat-hf"])
def test_gptq_cuda_graph_speed(model_id: str):
    if model_id == "facebook/opt-6.7b":
        device_map = None
    else:
        device_map = "auto"

    qspec = Uint4PerGroupSpec(ch_axis=1, is_dynamic=False, group_size=128).to_quantization_spec()

    global_quant_config = QLayerConfig(weight=qspec)

    if model_id == "facebook/opt-6.7b":
        gptq_config = GPTQConfig(
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
    else:
        gptq_config = GPTQConfig(
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

    config = QConfig(global_quant_config=global_quant_config, algo_config=[gptq_config])

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    text = "Hello, how are you?"
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"])

    model_graph = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map=device_map)
    if device_map is None:
        model_graph = model_graph.to(torch_device)

    model_graph = model_graph.eval()

    with torch.no_grad():
        # Run GPTQ using CUDA Graph.
        quantizer = ModelQuantizer(config, multi_device=False)

        start = time.time()
        model_graph = quantizer.quantize_model(model_graph, calib_dataloader)

        time_with_graph = time.time() - start

        del model_graph
        gc.collect()
        torch.cuda.empty_cache()

        # Run GPTQ without using CUDA Graph.
        with PatchEverywhere("QUARK_DISABLE_CUDA_GRAPH", True, module_name_prefix="quark"):
            model_no_graph = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map=device_map)
            if device_map is None:
                model_no_graph = model_no_graph.to(torch_device)

            model_no_graph = model_no_graph.eval()

            quantizer = ModelQuantizer(config, multi_device=False)

            start = time.time()
            model_no_graph = quantizer.quantize_model(model_no_graph, calib_dataloader)

            time_no_graph = time.time() - start

        print(f"time_no_graph: {time_no_graph} s, time_with_graph: {time_with_graph} s")
        assert time_no_graph / time_with_graph > 1.95, (
            f"time_no_graph: {time_no_graph} s, time_with_graph: {time_with_graph} s"
        )
