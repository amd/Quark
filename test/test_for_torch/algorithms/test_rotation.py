#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import datasets
import pytest
import torch
import torch.nn as nn
from packaging import version
from safetensors import safe_open
from testing_algorithms_utils import assert_non_destructive_transform
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.loss.loss_utils import fixed_cross_entropy

from quark.common.utils.import_utils import is_transformers_version_higher_or_equal
from quark.common.utils.testing_utils import (
    FROM_PRETRAINED_KWARGS,
    local_test_only,
    require_torch_cuda,
    require_vllm,
    slow_test,
    torch_device,
)
from quark.torch import ModelQuantizer, export_safetensors, import_model_from_safetensors
from quark.torch.algorithm.rotation.cayley import SGDG
from quark.torch.algorithm.rotation.hadamard import KNOWN_HADAMARD_MATRICES, matmul_hadU
from quark.torch.algorithm.rotation.rotation import RotationLinear, RotationProcessor
from quark.torch.algorithm.rotation.rotation_utils import get_rotation_matrix, rotate_in_channels_, rotate_out_channels_
from quark.torch.algorithm.rotation.training import AdamAndSGDGOptimizer
from quark.torch.quantization import (
    GPTQConfig,
    Int4PerChannelSpec,
    Int8PerChannelSpec,
    Int8PerTensorSpec,
    OCP_MXFP4Spec,
    OnlineRotationConfig,
    QConfig,
    QLayerConfig,
    RotationConfig,
)
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase
from quark.torch.utils import getattr_recursive
from quark.torch.utils.llm import preprocess_for_quantization

if torch.device(torch_device).type == "cpu" or torch.version.cuda is not None:
    ATTN_IMPLEMENTATION = "sdpa"
elif torch.version.hip is not None:
    rocm_version = ".".join(torch.version.hip.split(".")[:2])

    # NOTE: From testing on `rocm/dev-ubuntu-24.04:6.4.4-complete` + `torch==2.9.0+rocm6.4`,
    # the training loss on GPU with the above is much worse than on
    # - CPU device,
    # - or with torch==2.9.0a0+git1c57644 shipped in `rocm/vllm-dev:nightly_main_20251117` (that uses rocm 7),
    # - or with torch nightly that uses rocm 7.
    # It is highly likely that there is a bug in torch SDPA + ROCm 6.4 that we run into in this test if `ATTN_IMPLEMENTATION = "eager"` is not used (later training atol is not met).
    if version.parse(rocm_version) <= version.parse("6.99"):
        ATTN_IMPLEMENTATION = "eager"
    else:
        ATTN_IMPLEMENTATION = "sdpa"
else:
    raise ValueError(f"Unsupported torch_device={torch_device} along torch.version.cuda=None, torch.version.hip=None")


TEST_SHAPES = (
    [512, 1024, 2048, 4096, 14336, 1536]
    + list(KNOWN_HADAMARD_MATRICES.keys())
    + [size * 2 for size in list(KNOWN_HADAMARD_MATRICES.keys())]
)

# scaling_layers taken from quarot_config.json in quark examples.
SCALING_LAYERS_LLAMA = {
    "first_layer": [
        {
            "prev_modules": ["model.embed_tokens"],
            "norm_module": "model.layers.layer_id.input_layernorm",
            "next_modules": [
                "model.layers.layer_id.self_attn.q_proj",
                "model.layers.layer_id.self_attn.k_proj",
                "model.layers.layer_id.self_attn.v_proj",
            ],
        },
        {
            "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
            "norm_module": "model.layers.layer_id.post_attention_layernorm",
            "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"],
        },
    ],
    "middle_layers": [
        {
            "prev_modules": ["model.layers.pre_layer_id.mlp.down_proj"],
            "norm_module": "model.layers.layer_id.input_layernorm",
            "next_modules": [
                "model.layers.layer_id.self_attn.q_proj",
                "model.layers.layer_id.self_attn.k_proj",
                "model.layers.layer_id.self_attn.v_proj",
            ],
        },
        {
            "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
            "norm_module": "model.layers.layer_id.post_attention_layernorm",
            "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"],
        },
    ],
    "last_layer": [
        {
            "prev_modules": ["model.layers.layer_id.mlp.down_proj"],
            "norm_module": "model.norm",
            "next_modules": ["lm_head"],
        }
    ],
}

SCALING_LAYERS_QWEN_MOE = {
    "first_layer": [
        {
            "prev_modules": ["model.embed_tokens"],
            "norm_module": "model.layers.layer_id.input_layernorm",
            "next_modules": [
                "model.layers.layer_id.self_attn.q_proj",
                "model.layers.layer_id.self_attn.k_proj",
                "model.layers.layer_id.self_attn.v_proj",
            ],
        },
        {
            "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
            "norm_module": "model.layers.layer_id.post_attention_layernorm",
            "target_modules": [
                "model.layers.layer_id.mlp.experts.*.up_proj",
                "model.layers.layer_id.mlp.experts.*.gate_proj",
            ],
            "next_modules": [
                "model.layers.layer_id.mlp.experts.*.up_proj",
                "model.layers.layer_id.mlp.experts.*.gate_proj",
                "model.layers.layer_id.mlp.gate.linear"
                if is_transformers_version_higher_or_equal("5.0.0")
                else "model.layers.layer_id.mlp.gate",
            ],
        },
    ],
    "middle_layers": [
        {
            "prev_modules": ["model.layers.pre_layer_id.mlp.experts.*.down_proj"],
            "norm_module": "model.layers.layer_id.input_layernorm",
            "next_modules": [
                "model.layers.layer_id.self_attn.q_proj",
                "model.layers.layer_id.self_attn.k_proj",
                "model.layers.layer_id.self_attn.v_proj",
            ],
        },
        {
            "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
            "norm_module": "model.layers.layer_id.post_attention_layernorm",
            "target_modules": [
                "model.layers.layer_id.mlp.experts.*.up_proj",
                "model.layers.layer_id.mlp.experts.*.gate_proj",
            ],
            "next_modules": [
                "model.layers.layer_id.mlp.experts.*.up_proj",
                "model.layers.layer_id.mlp.experts.*.gate_proj",
                "model.layers.layer_id.mlp.gate.linear"
                if is_transformers_version_higher_or_equal("5.0.0")
                else "model.layers.layer_id.mlp.gate",
            ],
        },
    ],
    "last_layer": [
        {
            "prev_modules": ["model.layers.layer_id.mlp.experts.*.down_proj"],
            "norm_module": "model.norm",
            "target_modules": [],
            "next_modules": ["lm_head"],
        }
    ],
}


SCALING_LAYERS_GPT_OSS_MOE = {
    "first_layer": [
        {
            "prev_modules": ["model.embed_tokens"],
            "norm_module": "model.layers.layer_id.input_layernorm",
            "next_modules": [
                "model.layers.layer_id.self_attn.q_proj",
                "model.layers.layer_id.self_attn.k_proj",
                "model.layers.layer_id.self_attn.v_proj",
            ],
        },
        {
            "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
            "norm_module": "model.layers.layer_id.post_attention_layernorm",
            "target_modules": ["model.layers.layer_id.mlp.experts.*.gate_up_proj"],
            "next_modules": [
                "model.layers.layer_id.mlp.experts.*.gate_up_proj",
                "model.layers.layer_id.mlp.router.linear"
                if is_transformers_version_higher_or_equal("5.0.0")
                else "model.layers.layer_id.mlp.router",
            ],
        },
    ],
    "middle_layers": [
        {
            "prev_modules": ["model.layers.pre_layer_id.mlp.experts.*.down_proj"],
            "norm_module": "model.layers.layer_id.input_layernorm",
            "next_modules": [
                "model.layers.layer_id.self_attn.q_proj",
                "model.layers.layer_id.self_attn.k_proj",
                "model.layers.layer_id.self_attn.v_proj",
            ],
        },
        {
            "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
            "norm_module": "model.layers.layer_id.post_attention_layernorm",
            "target_modules": ["model.layers.layer_id.mlp.experts.*.gate_up_proj"],
            "next_modules": [
                "model.layers.layer_id.mlp.experts.*.gate_up_proj",
                "model.layers.layer_id.mlp.router.linear"
                if is_transformers_version_higher_or_equal("5.0.0")
                else "model.layers.layer_id.mlp.router",
            ],
        },
    ],
    "last_layer": [
        {
            "prev_modules": ["model.layers.layer_id.mlp.experts.*.down_proj"],
            "norm_module": "model.norm",
            "target_modules": [],
            "next_modules": ["lm_head"],
        }
    ],
}


MODEL_TYPE_TO_SCALING_LAYERS = {
    "llama": SCALING_LAYERS_LLAMA,
    "qwen3_moe": SCALING_LAYERS_QWEN_MOE,
    "gpt_oss": SCALING_LAYERS_GPT_OSS_MOE,
}


def run_rotation_training(
    quant_config: QConfig,
    learning_rate: float,
    dtype: str,
    model_id: str,
    online_r1_rotation: bool,
    steps: int = 30,
    n_layers: int | None = None,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation=ATTN_IMPLEMENTATION, **FROM_PRETRAINED_KWARGS
    )
    model = model.eval()

    if n_layers is not None:
        model.model.layers = model.model.layers[:n_layers]
        model.config.num_hidden_layers = n_layers

    model = model.to(torch_device)

    model = model.to(torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    inp = tokenizer(
        "Tokyo, officially the Tokyo Metropolis, is the capital and most populous city in Japan. With a population of over 14 million in the city proper in 2023, it is one of the most populous urban areas in the world. The Greater Tokyo Area, which includes Tokyo and parts of six neighboring prefectures, is the most populous metropolitan area in the world, with 41 million residents as of 2024.",
        return_tensors="pt",
    ).to(torch_device)

    with torch.no_grad():
        reference_output_before_patch = model(**inp).logits

    if model.config.model_type in ["gpt_oss", "qwen3_moe"]:
        preprocess_for_quantization(model)

    with torch.no_grad():
        reference_output = model(**inp).logits

    # TODO: test this elsewhere & remove this check.
    assert torch.allclose(reference_output_before_patch, reference_output, atol=1e-3, rtol=1e-3)

    original_weights = {name: param.clone() for name, param in model.named_parameters()}

    # 4-2. In-place replacement of model modules with quantized versions.
    quantizer = ModelQuantizer(quant_config)
    model = quantizer.quantize_model(model)

    inp_training = copy.deepcopy(inp)
    inp_training["labels"] = inp["input_ids"].clone()
    train_data = datasets.Dataset.from_dict(inp_training)

    model.use_cache = False

    model = model.eval()

    with torch.no_grad():
        output_notrain = model(**inp).logits

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainable_parameters_orthogonal, trainable_parameters_adam = RotationProcessor.get_trainable_parameters(
        model, rotation_config=quant_config.algo_config[0]
    )

    start_rotations = {}

    if quant_config.algo_config[0].r1:
        if not online_r1_rotation:
            start_rotations["shared_r1"] = model.shared_r1_rotation.data.clone()
        else:
            for name, submodule in model.named_modules():
                if (
                    isinstance(submodule, RotationLinear)
                    and submodule.rotation_in is not None
                    and submodule.hint_in == "r1"
                ):
                    start_rotations[name] = submodule.rotation_in.data.clone()

    if quant_config.algo_config[0].r2:
        for name, submodule in model.named_modules():
            if (
                isinstance(submodule, RotationLinear)
                and submodule.rotation_in is not None
                and submodule.hint_in == "r2"
            ):
                start_rotations[name] = submodule.rotation_in.data.clone()

    if quant_config.algo_config[0].r4:
        for name, submodule in model.named_modules():
            if (
                isinstance(submodule, RotationLinear)
                and submodule.rotation_in is not None
                and submodule.hint_in == "r4"
            ):
                start_rotations[name] = submodule.rotation_in.data.clone()

    if len(trainable_parameters_adam) == 0:
        optimizer = SGDG(trainable_parameters_orthogonal, lr=learning_rate, stiefel=True)
    elif len(trainable_parameters_orthogonal) == 0:
        optimizer = torch.optim.Adam(trainable_parameters_adam, lr=1e-2)
    else:
        optimizer = AdamAndSGDGOptimizer(
            sgdg_params=trainable_parameters_orthogonal,
            adam_params=trainable_parameters_adam,
            learning_rate=learning_rate,
            smooth_learning_rate=1e-2,
        )

    fp16 = False
    bf16 = False
    if dtype == "bf16":
        bf16 = True
    elif dtype == "fp16":
        fp16 = True

    with TemporaryDirectory() as tmpdir:
        training_args = TrainingArguments(
            output_dir=tmpdir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            fp16=fp16,
            bf16=bf16,
            log_on_each_node=False,
            logging_steps=1.0,
            learning_rate=learning_rate,
            logging_dir=tmpdir,
            do_eval=False,
            do_train=True,
            gradient_checkpointing=True,
            max_steps=steps,
            lr_scheduler_type="cosine",
            save_strategy="no",
        )

        trainer = Trainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_data,
            data_collator=default_data_collator,
            optimizers=(optimizer, None),
        )

        # Make sure QDQ is enabled during training.
        for name, module in model.named_modules():
            if isinstance(module, FakeQuantizeBase):
                assert module.is_fake_quant_enabled

                if "_weight_quantizer" not in name:
                    assert module.is_dynamic
                    assert module.observer_enabled

        trainer.train()

    model = model.eval()
    return model, start_rotations, quantizer, inp, reference_output, output_notrain, original_weights


def compute_loss(logits, inp):
    labels = inp["input_ids"].clone()
    labels = nn.functional.pad(labels, (0, 1), value=-100)
    labels = labels[..., 1:].contiguous()
    labels = labels.view(-1)

    logits = logits.view(-1, logits.shape[-1])

    logits = logits.float()
    loss = fixed_cross_entropy(logits, labels, num_items_in_batch=labels.numel())

    return loss


@pytest.mark.parametrize("n", TEST_SHAPES)
def test_matmul_hadU_inverse(n: int):
    inp = torch.rand(5, n)

    res_custom = matmul_hadU(inp.clone())

    inp_bis = matmul_hadU(res_custom, inverse=True)

    assert torch.allclose(inp, inp_bis, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("n", TEST_SHAPES)
@pytest.mark.parametrize("shapes", [(4, 5), (3,)])
@pytest.mark.parametrize("inverse", [False, True])
def test_hadamard(n: int, shapes: tuple[int], inverse: bool):
    # Verify the equivalence between `matmul_hadU` and `get_rotation_matrix` + torch.matmul.
    inp = torch.rand(*shapes, n, device=torch_device)

    res_custom = matmul_hadU(inp.clone(), inverse=inverse)

    rotation_matrix = get_rotation_matrix(n, random=False, device=torch_device)
    rotation_matrix = rotation_matrix.to(inp.dtype)

    identity = torch.eye(n).to(torch_device)
    assert torch.allclose(rotation_matrix @ rotation_matrix.T, identity, atol=1e-5, rtol=1e-4)

    if inverse:
        rotation_matrix = rotation_matrix.T

    res_matmul = inp @ rotation_matrix

    assert torch.allclose(res_matmul, res_custom, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("n", TEST_SHAPES)
@pytest.mark.parametrize("inverse", [False, True])
def test_rotate_in_channels(n: int, inverse: bool):
    linear = nn.Linear(n, n, bias=False)
    inp = torch.rand(n, n, device=torch_device)

    res_custom = matmul_hadU(inp.clone(), inverse=inverse)

    rotation_matrix = get_rotation_matrix(n, random=False, device=torch_device)

    if inverse:
        rotation_matrix = rotation_matrix.T

    linear.weight.data = inp.clone()
    rotate_in_channels_(linear, rotation_matrix.to(torch.float64))

    assert torch.allclose(linear.weight.data, res_custom, atol=1e-2, rtol=1e-2)


def test_rotate_out_channels():
    # Create a Linear layer with bias
    in_features = 3
    out_features = 3
    module = nn.Linear(in_features, out_features, bias=True)

    # Initialize rotation matrix as an identity matrix (no rotation)
    rotation = torch.eye(out_features, dtype=torch.float64)

    # Clone original weights and biases for comparison
    original_weight = module.weight.data.clone()
    original_bias = module.bias.data.clone()

    # Apply the rotation
    rotate_out_channels_(module, rotation)

    # Check if weights and biases remain the same (since rotation is identity)
    assert torch.allclose(module.weight.data, original_weight, atol=1e-6), "Weights were incorrectly modified."
    assert torch.allclose(module.bias.data, original_bias, atol=1e-6), "Bias was incorrectly modified."

    # Apply a non-identity rotation matrix
    rotation = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)

    # Manually compute expected rotated weights and biases
    expected_weight = torch.matmul(rotation.T, original_weight.to(dtype=torch.float64))
    expected_bias = torch.matmul(rotation.T, original_bias.to(dtype=torch.float64))

    # Apply the rotation
    rotate_out_channels_(module, rotation)

    # Check if weights and biases match the expected values
    assert torch.allclose(module.weight.data, expected_weight.to(dtype=module.weight.dtype), atol=1e-6), (
        "Weights were not rotated correctly."
    )
    assert torch.allclose(module.bias.data, expected_bias.to(dtype=module.bias.dtype), atol=1e-6), (
        "Bias was not rotated correctly."
    )


@pytest.mark.parametrize("rotation_size", [pytest.param(val, id=f"rotation_size:{val}") for val in [None, 32]])
@pytest.mark.parametrize("r1", [pytest.param(val, id=f"r1:{val}") for val in [True]])
@pytest.mark.parametrize("r2", [pytest.param(val, id=f"r2:{val}") for val in [True]])
@pytest.mark.parametrize("r3", [pytest.param(val, id=f"r3:{val}") for val in [False, True]])
@pytest.mark.parametrize("r4", [pytest.param(val, id=f"r4:{val}") for val in [True]])
@pytest.mark.parametrize(
    "online_r1_rotation", [pytest.param(val, id=f"online_r1_rotation:{val}") for val in [False, True]]
)
@pytest.mark.parametrize("trainable", [pytest.param(val, id=f"trainable:{val}") for val in [False, True]])
@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param(val, id=f"model_id:{val}")
        for val in [
            "HuggingFaceTB/SmolLM-135M",
            "amd-quark/tiny-random-qwen3_moe",
            "optimum-intel-internal-testing/tiny-random-gpt-oss-mxfp4",
        ]
    ],
)
def test_non_destructive_transform(
    rotation_size: int | None,
    r1: bool,
    r2: bool,
    r3: bool,
    r4: bool,
    online_r1_rotation: bool,
    trainable: bool,
    model_id: str,
):
    func_test_non_destructive_transform(
        rotation_size=rotation_size,
        r1=r1,
        r2=r2,
        r3=r3,
        r4=r4,
        online_r1_rotation=online_r1_rotation,
        trainable=trainable,
        model_id=model_id,
    )


def func_test_non_destructive_transform(
    rotation_size: int | None,
    r1: bool,
    r2: bool,
    r3: bool,
    r4: bool,
    online_r1_rotation: bool,
    trainable: bool,
    model_id: str,
):
    if rotation_size == 32 and r3:
        pytest.skip("Custom rotation_size + r3 is not implemented")

    if not r1 and online_r1_rotation:
        pytest.skip("online_r1_rotation=True has no effect when r1=False")

    if trainable and r3:
        pytest.skip(f"r3={r3} not supported with trainable=True")

    config = AutoConfig.from_pretrained(model_id)

    if config.model_type in {"qwen3_moe", "gpt_oss"}:
        mlp = "mlp.experts.*"
    else:
        mlp = "mlp"

    rotation_config = RotationConfig(
        scaling_layers=MODEL_TYPE_TO_SCALING_LAYERS[config.model_type],
        rotation_size=rotation_size,
        r1=r1,
        r2=r2,
        r3=r3,
        r4=r4,
        online_r1_rotation=None if not r1 else online_r1_rotation,
        trainable=trainable,
        mlp=mlp,
    )

    assert_non_destructive_transform(rotation_config, model_id=model_id)


# TODO: test rotation_size once supported.
# We only test the case `r1=True, r2=True, r4=True` so as not to bloat the CI.
@require_torch_cuda
@pytest.mark.parametrize(
    "online_r1_rotation", [pytest.param(val, id=f"online_r1_rotation:{val}") for val in [False, True]]
)
@pytest.mark.parametrize("r1", [pytest.param(val, id=f"r1:{val}") for val in [True]])
@pytest.mark.parametrize("r2", [pytest.param(val, id=f"r2:{val}") for val in [True]])
@pytest.mark.parametrize("r4", [pytest.param(val, id=f"r4:{val}") for val in [True]])
@pytest.mark.parametrize("shared_parallel", [pytest.param(val, id=f"shared_parallel:{val}") for val in [False, True]])
@pytest.mark.parametrize("train_smooth", [pytest.param(val, id=f"train_smooth:{val}") for val in [False, True]])
@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param(val, id=f"model_id:{val}")
        for val in ["HuggingFaceTB/SmolLM-135M", "amd-quark/tiny-random-qwen3_moe"]
    ],
)
def test_trained_rotation_non_destructive(
    online_r1_rotation: bool, r1: bool, r2: bool, r4: bool, shared_parallel: bool, train_smooth: bool, model_id: str
):
    func_test_trained_rotation_non_destructive(
        online_r1_rotation=online_r1_rotation,
        r1=r1,
        r2=r2,
        r4=r4,
        shared_parallel=shared_parallel,
        train_smooth=train_smooth,
        model_id=model_id,
    )


def func_test_trained_rotation_non_destructive(
    online_r1_rotation: bool, r1: bool, r2: bool, r4: bool, shared_parallel: bool, train_smooth: bool, model_id: str
):
    if not r1 and not r2 and not r4:
        pytest.skip("r1=False and r2=False and r4=False is not supported")

    if not r1 and online_r1_rotation:
        pytest.skip("r1=False and online_r1_rotation=True has no effect")

    if (not r1 or not online_r1_rotation) and shared_parallel:
        pytest.skip("r1=False + online_r1_rotation=False + shared_parallel=True has no effect")

    model_config = AutoConfig.from_pretrained(model_id)

    if model_config.model_type in {"qwen3_moe", "gpt_oss"}:
        mlp = "mlp.experts.*"
    else:
        mlp = "mlp"

    if r1 and online_r1_rotation:
        online_config = OnlineRotationConfig(shared_parallel=shared_parallel)
    else:
        online_config = None

    if train_smooth:
        smooth_positions = ["r1", "r2", "r4"]
    else:
        smooth_positions = None

    algo_config = RotationConfig(
        scaling_layers=MODEL_TYPE_TO_SCALING_LAYERS[model_config.model_type],
        rotation_size=None,
        r1=r1,
        r2=r2,
        r3=False,
        r4=r4,
        online_r1_rotation=None if not r1 else online_r1_rotation,
        trainable=True,
        online_config=online_config,
        train_smooth=train_smooth,
        smooth_positions=smooth_positions,
        mlp=mlp,
    )

    # No quantization.
    layer_quant_config = QLayerConfig()

    quant_config_rotation = QConfig(
        global_quant_config=layer_quant_config,
        algo_config=[algo_config],
        exclude=["lm_head", "*.gate", "*.gate.linear"],
    )

    # NOTE: it is not certain bf16 is stable.
    model, _, quantizer, inp, reference_output, _, _ = run_rotation_training(
        quant_config_rotation, learning_rate=1e4, dtype="fp32", model_id=model_id, online_r1_rotation=online_r1_rotation
    )

    model = model.eval()
    with torch.no_grad():
        reference_output_posttrain = model(**inp).logits

    quant_config_quantization = copy.deepcopy(quant_config_rotation)
    quant_config_quantization.algo_config = []

    model = RotationProcessor.post_process_trained_rotation(model=model, quantization_config=quant_config_quantization)

    quantizer = ModelQuantizer(quant_config_quantization)
    with torch.no_grad():
        model = quantizer.quantize_model(model)

    model = model.eval()
    with torch.no_grad():
        candidate_output = model(**inp).logits

    absdiff = (candidate_output - reference_output_posttrain).abs()
    assert absdiff.mean() < 5e-5
    assert absdiff.max() < 1e-3

    # TODO: there is an issue in this test with fusing R1 in case quantization is used, not sure why. This case is tested in test_trained_rotation_correctness anyway.


@local_test_only
@require_torch_cuda
@pytest.mark.parametrize("rotation_size", [pytest.param(val, id=f"rotation_size:{val}") for val in [96, None]])
@pytest.mark.parametrize(
    "online_r1_rotation", [pytest.param(val, id=f"online_r1_rotation:{val}") for val in [False, True]]
)
@pytest.mark.parametrize("r1", [pytest.param(val, id=f"r1:{val}") for val in [False, True]])
@pytest.mark.parametrize("r2", [pytest.param(val, id=f"r2:{val}") for val in [False, True]])
@pytest.mark.parametrize("r4", [pytest.param(val, id=f"r4:{val}") for val in [False, True]])
@pytest.mark.parametrize("act_only", [pytest.param(val, id=f"act_only:{val}") for val in [False, True]])
@pytest.mark.parametrize(
    "weight_format", [pytest.param(val, id=f"weight_format:{val}") for val in ["fake_quantized", "real_quantized"]]
)
@pytest.mark.parametrize("train_smooth", [pytest.param(val, id=f"train_smooth:{val}") for val in [False, True]])
def test_trained_rotation_correctness_complete(
    online_r1_rotation: bool,
    r1: bool,
    r2: bool,
    r4: bool,
    act_only: bool,
    weight_format: str,
    rotation_size: int | None,
    train_smooth: bool,
):
    func_test_trained_rotation_correctness(
        online_r1_rotation=online_r1_rotation,
        r1=r1,
        r2=r2,
        r4=r4,
        act_only=act_only,
        weight_format=weight_format,
        rotation_size=rotation_size,
        train_smooth=train_smooth,
        model_id="HuggingFaceTB/SmolLM-135M",  # TODO: test qwen3_moe.
    )


@pytest.mark.parametrize(
    "online_r1_rotation", [pytest.param(val, id=f"online_r1_rotation:{val}") for val in [False, True]]
)
def test_trained_rotation_correctness_fast(online_r1_rotation: bool):
    func_test_trained_rotation_correctness(
        online_r1_rotation=online_r1_rotation,
        r1=True,
        r2=True,
        r4=True,
        act_only=False,
        weight_format="real_quantized",
        rotation_size=96,
        train_smooth=True,
        model_id="HuggingFaceTB/SmolLM-135M",  # NOTE: qwen3_moe is tested in the slow CI only.
        n_layers=3,
    )


@pytest.mark.parametrize(
    "online_r1_rotation", [pytest.param(val, id=f"online_r1_rotation:{val}") for val in [False, True]]
)
def test_trained_rotation_correctness_moe(online_r1_rotation: bool):
    func_test_trained_rotation_correctness(
        online_r1_rotation=online_r1_rotation,
        r1=True,
        r2=True,
        r4=True,
        act_only=False,
        weight_format="real_quantized",
        rotation_size=64,
        train_smooth=True,
        model_id="amd-quark/tiny-random-qwen3_moe-256",
        improvement_factor=0.97,
        n_layers=3,
    )


@slow_test
@require_torch_cuda
@pytest.mark.parametrize(
    "online_r1_rotation", [pytest.param(val, id=f"online_r1_rotation:{val}") for val in [False, True]]
)
@pytest.mark.parametrize(
    "weight_format", [pytest.param(val, id=f"weight_format:{val}") for val in ["fake_quantized", "real_quantized"]]
)
@pytest.mark.parametrize("train_act_only", [pytest.param(val, id=f"train_act_only:{val}") for val in [True, False]])
def test_trained_rotation_gptq(online_r1_rotation: bool, weight_format: str, train_act_only: bool):
    func_test_trained_rotation_correctness(
        online_r1_rotation=online_r1_rotation,
        r1=True,
        r2=True,
        r4=False,
        rotation_size=96,
        weight_format=weight_format,
        act_only=False,
        quant_algo="gptq",
        train_act_only=train_act_only,
        train_smooth=False,
        model_id="HuggingFaceTB/SmolLM-135M",
    )


def func_test_trained_rotation_correctness(
    online_r1_rotation: bool,
    r1: bool,
    r2: bool,
    r4: bool,
    act_only: bool,
    weight_format: str,
    model_id: str,
    rotation_size: int | None,
    quant_algo: str | None = None,
    train_act_only: bool = False,
    train_smooth: bool = False,
    improvement_factor: int | None = None,
    n_layers: int | None = None,
):
    """
    train_act_only: Whether to do QDQ on activations only during rotation training.
    """
    if not r1 and not r2 and not r4:
        pytest.skip("r1=False and r2=False and r4=False is not supported")

    if not r1 and online_r1_rotation:
        pytest.skip("r1=False and online_r1_rotation=True has no effect")

    algo_config = []
    if quant_algo is not None:
        assert quant_algo == "gptq"
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

        gptq_config = GPTQConfig(
            model_decoder_layers=model_decoder_layers,
            inside_layer_modules=inside_layer_modules,
            block_size=64,
        )
        algo_config = [gptq_config]

    model_config = AutoConfig.from_pretrained(model_id)

    if model_config.model_type in {"qwen3_moe", "gpt_oss"}:
        mlp = "mlp.experts.*"
    else:
        mlp = "mlp"

    if train_smooth:
        smooth_positions = ["r1", "r2", "r4"]
    else:
        smooth_positions = None

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    rotation_config = RotationConfig(
        scaling_layers=MODEL_TYPE_TO_SCALING_LAYERS[model_config.model_type],
        rotation_size=rotation_size,
        r1=r1,
        r2=r2,
        r3=False,
        r4=r4,
        online_r1_rotation=None if not r1 else online_r1_rotation,
        trainable=True,
        train_smooth=train_smooth,
        smooth_positions=smooth_positions,
        mlp=mlp,
    )

    all_algo_config = [rotation_config] + algo_config

    int4_per_channel_sym_spec = Int4PerChannelSpec(
        symmetric=True, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=False
    ).to_quantization_spec()

    int8_per_token_sym_spec = Int8PerChannelSpec(
        symmetric=True, scale_type="float", round_method="half_even", ch_axis=1, is_dynamic=True
    ).to_quantization_spec()

    if act_only:
        layer_quant_config = QLayerConfig(input_tensors=int8_per_token_sym_spec)
    else:
        layer_quant_config = QLayerConfig(weight=int4_per_channel_sym_spec, input_tensors=int8_per_token_sym_spec)

    quant_config_base = QConfig(global_quant_config=layer_quant_config, exclude=["lm_head", "*.gate", "*.gate.linear"])

    quant_config_rotation = copy.deepcopy(quant_config_base)
    quant_config_rotation.algo_config = [rotation_config]

    if train_act_only:
        quant_config_rotation.global_quant_config.weight = None

    quant_config_quantization = copy.deepcopy(quant_config_base)
    quant_config_quantization.algo_config = algo_config

    quant_config_rotation.global_quant_config.input_tensors.is_dynamic = True

    if quant_config_rotation.global_quant_config.weight is not None:
        quant_config_rotation.global_quant_config.weight.is_dynamic = True

    model, start_rotations, quantizer, inp, reference_output, output_notrain, original_weights = run_rotation_training(
        quant_config_rotation,
        learning_rate=1.5,
        dtype="fp32",
        model_id=model_id,
        online_r1_rotation=online_r1_rotation,
        n_layers=n_layers,
    )

    # TODO: we should somehow test that correct STE is applied during training (weight QDQ, activation QDQ).

    model = model.eval()

    with torch.no_grad():
        output_trained = model(**inp).logits

    # Make sure loss decreased.
    # TODO: Use KL-divergence instead of cross entropy here.
    noquant_loss = compute_loss(reference_output, inp)
    notrain_loss = compute_loss(output_notrain, inp)
    train_loss = compute_loss(output_trained, inp)

    if improvement_factor is None:
        if r1:
            improvement_factor = 0.94
        else:
            improvement_factor = 1.0

    # TODO: remove controlflow, not sure why. This is sensitive to `symmetric=False/True` for activations as well.
    if not act_only and not train_act_only:
        if model.config.model_type != "qwen3_moe":
            assert noquant_loss.item() < notrain_loss.item()
            assert train_loss.item() < notrain_loss.item() * improvement_factor

    # Make sure the original weights were not modified during training (e.g. quantized)
    trained_params = dict(model.named_parameters())

    # Make sure the learned rotation is identical for all layers.
    for name, param in model.named_parameters():
        if r1 and not r2 and not r4:
            if not online_r1_rotation:
                assert "rotation_in" not in name and "rotation_out" not in name
        elif not r1 and not r4 and r2:
            assert "rotation_in" not in name

            if "rotation_out" in name:
                submodule_name = ".".join(name.split(".")[:-1])
                submodule = getattr_recursive(model, submodule_name)
                assert submodule.hint_out == "r2"
        elif not r1 and not r2 and r4:
            assert "rotation_out" not in name

            if "rotation_in" in name:
                submodule_name = ".".join(name.split(".")[:-1])
                submodule = getattr_recursive(model, submodule_name)
                assert submodule.hint_in == "r4"

    # In case training is done for offline rotation, LayerNorm weight is fused into linear weights prior to training, so skipping this test.
    # In case `train_smooth=True`, we skip this test as well as the normalization is fused into preceding layer.
    if r1 and online_r1_rotation and not train_smooth:
        for name, param in original_weights.items():
            try:
                assert torch.equal(param, trained_params[name])
            except Exception:
                name = name.replace(".weight", ".linear.weight")
                name = name.replace(".bias", ".linear.bias")

                assert torch.equal(param, trained_params[name])

    end_rotations = {}
    if r1:
        if not online_r1_rotation:
            end_rotations["shared_r1"] = model.shared_r1_rotation.data.clone()
        else:
            for name, submodule in model.named_modules():
                if (
                    isinstance(submodule, RotationLinear)
                    and submodule.rotation_in is not None
                    and submodule.hint_in == "r1"
                ):
                    end_rotations[name] = submodule.rotation_in.data.clone()

    if r2:
        for name, submodule in model.named_modules():
            if (
                isinstance(submodule, RotationLinear)
                and submodule.rotation_in is not None
                and submodule.hint_in == "r2"
            ):
                end_rotations[name] = submodule.rotation_in.data.clone()

    if r4:
        for name, submodule in model.named_modules():
            if (
                isinstance(submodule, RotationLinear)
                and submodule.rotation_in is not None
                and submodule.hint_in == "r4"
            ):
                end_rotations[name] = submodule.rotation_in.data.clone()

    model = RotationProcessor.post_process_trained_rotation(model=model, quantization_config=quant_config_quantization)

    model = model.eval()

    with torch.no_grad():
        output_trained_transformed = model(**inp).logits

    # Make sure `post_process_trained_rotation` did not destroy anything.
    # `post_process_trained_rotation` adds back static weight quantization, but disables fake quantize.
    # Weight quantizatio is handled in the next `quantizer.quantize_model` call.
    assert output_trained_transformed.dtype == output_trained.dtype

    train_loss_after_transform = compute_loss(output_trained_transformed, inp)

    assert torch.allclose(train_loss, train_loss_after_transform, atol=1e-2, rtol=1e-2)

    calibration_dataloader = None
    if quant_algo == "gptq":
        text = "Hello, how are you?"
        tokenized_inputs = tokenizer(text, return_tensors="pt").to(torch_device)
        calibration_dataloader = DataLoader(tokenized_inputs["input_ids"])

    quantizer = ModelQuantizer(quant_config_quantization)
    with torch.no_grad():
        model = quantizer.quantize_model(model, calibration_dataloader)

    model.quant_config.algo_config = all_algo_config

    maxabsdiff = 0
    for name, end_rotation in end_rotations.items():
        start_rotation = start_rotations[name]

        absdiff = (end_rotation - start_rotation).abs()
        maxabsdiff = max(maxabsdiff, absdiff.max().item())

        # Make sure the learned R1 matrix is orthogonal.
        reference_eye = torch.eye(start_rotation.shape[-1], device=start_rotation.device, dtype=start_rotation.dtype)

        tol = 1e-4

        absdiff = (end_rotation @ end_rotation.T - reference_eye).abs()

        assert torch.allclose(end_rotation @ end_rotation.T, reference_eye, atol=tol, rtol=tol)

    # Make sure something was learned.
    if not act_only and not train_act_only:
        assert maxabsdiff > 1e-3
    else:
        assert maxabsdiff > 1e-5

    model = model.eval()

    with torch.no_grad():
        output_after_quant = model(**inp).logits

    # Make sure `quantizer.quantize_model` did not destroy anything.
    assert output_trained_transformed.dtype == output_after_quant.dtype

    train_loss_output_after_quant = compute_loss(output_after_quant, inp)

    # NOTE: In case the rotation matrices are casted to different dtype than training, this can sensibly change. TODO: Verify it is okay to apply rotations in fp16/bf16.
    # This is true only in case RTN (no algo) is used to run quantization.
    # In case `train_act_only=True`, weight quantization is disabled PRIOR TO the `quantizer.quantize_model` call, so skipping this test.
    if quant_algo is None and not train_act_only:
        assert torch.allclose(train_loss_output_after_quant, train_loss_after_transform, atol=1e-2, rtol=1e-2)

    model = quantizer.freeze(model)

    with torch.no_grad():
        output_trained_frozen = model(**inp).logits

    # Make sure `freeze` did not destroy anything.
    assert output_trained_frozen.dtype == output_trained.dtype
    train_loss_after_frozen = compute_loss(output_trained_frozen, inp)

    assert torch.allclose(train_loss_output_after_quant, train_loss_after_frozen, atol=1e-2, rtol=1e-2)

    # Make sure the model generated with trained orthogonal rotations is reloadable and correct.
    with torch.no_grad(), TemporaryDirectory() as tmpdir:
        export_safetensors(model=model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

        config = AutoConfig.from_pretrained(model_id)
        if n_layers is not None:
            config.num_hidden_layers = n_layers

        original_model = AutoModelForCausalLM.from_config(config, attn_implementation=ATTN_IMPLEMENTATION)
        original_model = original_model.to(torch_device)

        q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

        q_model = q_model.eval()
        q_model = q_model.to(torch_device)

        with torch.no_grad():
            output_trained_reloaded = q_model(**inp).logits

        train_loss_after_reload = compute_loss(output_trained_reloaded, inp)

        assert torch.allclose(train_loss_output_after_quant, train_loss_after_reload, atol=1e-2, rtol=1e-2)

        # TODO: verify why no strict torch.equal match here!
        absdiff = (output_trained_transformed - output_trained_reloaded).abs()
        print("max absdiff", absdiff.max())
        print("mean absdiff", absdiff.mean())
        assert torch.allclose(output_after_quant, output_trained_reloaded, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("rotation_size", [pytest.param(val, id=f"rotation_size:{val}") for val in [32, None]])
@pytest.mark.parametrize(
    "online_r1_rotation", [pytest.param(val, id=f"online_r1_rotation:{val}") for val in [False, True]]
)
@pytest.mark.parametrize(
    "weight_format", [pytest.param(val, id=f"weight_format:{val}") for val in ["fake_quantized", "real_quantized"]]
)
@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param(val, id=f"model_id:{val}")
        for val in ["HuggingFaceTB/SmolLM2-360M-Instruct", "amd-quark/tiny-random-qwen3_moe"]
    ],
)
def test_serialization_and_reload(
    rotation_size: int | None,
    online_r1_rotation: bool,
    weight_format: str,
    model_id: str,
):
    # TODO: This test is sometimes failing on Instinct MI300, either hardware bug/deterministic issue or quark bug.
    if torch_device.type == "cuda" and torch.version.hip is not None:
        device_test = torch.device("cpu")
    else:
        device_test = torch_device

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # The prompt "I am having a good time in Chicago, in the US. This is a fairly long text, right? No?" fails with HuggingFaceTB/SmolLM-135M on MI300 (torch==2.8.0+rocm6.4, transformers==4.55.4), but not on CPU nor on H100.
    prompt = "This is a completely different sequence. I don't know if it is long or not, what do you think? And how about this that is even longer? For some reason the absdiff is exactly the same now, how come."
    inp = tokenizer(prompt, return_tensors="pt").to(device_test)

    model_config = AutoConfig.from_pretrained(model_id)

    if model_config.model_type in {"qwen3_moe"}:
        mlp = "mlp.experts.*"
    else:
        mlp = "mlp"

    rotation_config = RotationConfig(
        scaling_layers=MODEL_TYPE_TO_SCALING_LAYERS[model_config.model_type],
        rotation_size=rotation_size,
        r1=True,
        r2=True,
        r3=False,
        r4=True,
        online_r1_rotation=online_r1_rotation,
        mlp=mlp,
    )

    w_int8_spec = Int8PerTensorSpec(
        observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
    ).to_quantization_spec()
    a_int8_spec = Int8PerTensorSpec(
        observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=True
    ).to_quantization_spec()

    algo_config = [rotation_config]

    global_config = QLayerConfig(weight=w_int8_spec, input_tensors=a_int8_spec)
    quant_config = QConfig(
        global_quant_config=global_config, algo_config=algo_config, exclude=["lm_head", "*.gate", "*.gate.linear"]
    )

    with sdpa_kernel(SDPBackend.MATH):
        model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation=ATTN_IMPLEMENTATION)
        model = model.eval()

        # HuggingFaceTB/SmolLM2-360M-Instruct has 32 layers, too slow for a unit test.
        if model.config.model_type == "llama":
            model.model.layers = model.model.layers[:3]
            model.config.num_hidden_layers = 3

        model = model.to(device_test)

        preprocess_for_quantization(model)

        quantizer = ModelQuantizer(quant_config)
        quant_model = quantizer.quantize_model(model)
        quant_model = quantizer.freeze(quant_model)

        quant_model = quant_model.to(device_test)

        with torch.no_grad():
            ref_outputs = quant_model(**inp).logits

        with torch.no_grad(), TemporaryDirectory() as tmpdir:
            export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")

            if Path(tmpdir, "model-00001-of-00002.safetensors").exists():
                checkpoint_name = "model-00001-of-00002.safetensors"
            else:
                checkpoint_name = "model.safetensors"

            found_online = False
            with safe_open(Path(tmpdir, checkpoint_name), framework="pt", device=torch_device.type) as f:
                for key in f.keys():  # noqa
                    if "input_rotation" in key:
                        param = f.get_tensor(key)
                        found_online = True
                        assert param.dtype == torch.int8

                        if rotation_size is not None:
                            assert param.shape[0] == rotation_size

            if online_r1_rotation:
                assert found_online

            config = AutoConfig.from_pretrained(model_id)
            with torch.device(device_test):
                original_model = AutoModelForCausalLM.from_config(config, attn_implementation=ATTN_IMPLEMENTATION)
                if original_model.config.model_type == "llama":
                    original_model.model.layers = original_model.model.layers[:3]
                    original_model.config.num_hidden_layers = 3

            q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

            q_model = q_model.eval()
            q_model = q_model.to(device_test)

            with torch.no_grad():
                outputs = q_model(**inp).logits

            assert ref_outputs.ndim == 3

            for token_id in range(ref_outputs.shape[1]):
                ref_argsort = torch.argsort(ref_outputs[0][token_id], descending=True)
                reload_argsort = torch.argsort(outputs[0][token_id], descending=True)

                absdiff = (ref_outputs[0][token_id] - outputs[0][token_id]).abs()
                print(f"token {token_id} mean absdiff", absdiff.mean())

                # Make sure top token is consistently the same.
                assert torch.equal(ref_argsort[:1], reload_argsort[:1])

                # Make sure top-k is ~~~consistent.
                top_argsort = ref_argsort[:10].tolist()
                top_reload_argsort = reload_argsort[:10].tolist()

                intersection = set(top_argsort).intersection(set(top_reload_argsort))
                assert len(intersection) >= 5

            # This assertion does not pass in some cases, for some prompts and some quantization schemes (OCP MXFP4 & MI300 only so far, seem to be fewer issues on CPU).
            # The reason appear to stem from rounding issues with MX, where in the middle of the model a value gets rounded up instead of down (or convertly), and the eventual output logits are then vastly different.
            # Small models as HuggingFaceTB/SmolLM-135M seem to be more sensitive as well.

            assert torch.allclose(ref_outputs, outputs, atol=1e-3, rtol=1e-3)


def test_scaling_layers():
    scaling_layers = SCALING_LAYERS_LLAMA

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
    model = model.eval()

    model_decoder_layers = "model.layers"

    scaling_layers = RotationProcessor.get_scaling_layers(
        model,
        scaling_layers,
        online_r1_rotation=False,
        r1=True,
        smooth_positions=[],
        model_decoder_layers=model_decoder_layers,
    )

    assert len(scaling_layers[-1]["next_modules"]) == 1
    assert scaling_layers[-1]["next_modules"][0] == "lm_head"

    scaling_layers = SCALING_LAYERS_LLAMA
    scaling_layers = RotationProcessor.get_scaling_layers(
        model,
        scaling_layers,
        online_r1_rotation=True,
        r1=True,
        smooth_positions=[],
        model_decoder_layers=model_decoder_layers,
    )

    assert len(scaling_layers[-1]["next_modules"]) == 2
    assert scaling_layers[-1]["next_modules"][0] == "model.layers.29.mlp.up_proj"
    assert scaling_layers[-1]["next_modules"][1] == "model.layers.29.mlp.gate_proj"


@pytest.mark.parametrize(
    "rotation_size", [pytest.param(val, id=f"rotation_size:{val}") for val in [None]]
)  # TODO: test rotation_size once supported.
@pytest.mark.parametrize("r1", [pytest.param(val, id=f"r1:{val}") for val in [True]])
@pytest.mark.parametrize("r2", [pytest.param(val, id=f"r2:{val}") for val in [True]])
@pytest.mark.parametrize("r4", [pytest.param(val, id=f"r4:{val}") for val in [False, True]])
@pytest.mark.parametrize(
    "online_r1_rotation", [pytest.param(val, id=f"online_r1_rotation:{val}") for val in [False, True]]
)
@pytest.mark.parametrize("shared_parallel", [pytest.param(val, id=f"shared_parallel:{val}") for val in [False, True]])
@pytest.mark.parametrize("train_smooth", [pytest.param(val, id=f"train_smooth:{val}") for val in [False, True]])
def test_get_trainable_parameters(
    r1: bool,
    r2: bool,
    r4: bool,
    online_r1_rotation: bool,
    rotation_size: int | None,
    shared_parallel: bool,
    train_smooth: bool,
):
    model_id = "HuggingFaceTB/SmolLM-135M"

    if not online_r1_rotation and shared_parallel:
        pytest.skip("shared_parallel=True has no effect with online_r1_rotation=False")

    if not r1 and shared_parallel:
        pytest.skip("r1=False and shared_parallel=True has no effect")

    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation=ATTN_IMPLEMENTATION)
    model = model.eval()
    model = model.to(torch_device)

    if r1 and online_r1_rotation:
        online_config = OnlineRotationConfig(shared_parallel=shared_parallel)
    else:
        online_config = None

    if train_smooth:
        smooth_positions = ["r1", "r2", "r4"]
    else:
        smooth_positions = None

    rotation_config = RotationConfig(
        scaling_layers=SCALING_LAYERS_LLAMA,
        rotation_size=rotation_size,
        r1=r1,
        r2=r2,
        r3=False,
        r4=r4,
        online_r1_rotation=None if not r1 else online_r1_rotation,
        trainable=True,
        online_config=online_config,
        train_smooth=train_smooth,
        smooth_positions=smooth_positions,
    )

    int4_per_channel_sym_spec = Int4PerChannelSpec(
        symmetric=True, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=False
    ).to_quantization_spec()

    int8_per_token_sym_spec = Int8PerChannelSpec(
        symmetric=True, scale_type="float", round_method="half_even", ch_axis=1, is_dynamic=True
    ).to_quantization_spec()

    layer_quant_config = QLayerConfig(weight=int4_per_channel_sym_spec, input_tensors=int8_per_token_sym_spec)

    quant_config = QConfig(
        global_quant_config=layer_quant_config,
        algo_config=[rotation_config],
        exclude=["lm_head", "*.gate", "*.gate.linear"],
    )

    # 4-2. In-place replacement of model modules with quantized versions.
    quantizer = ModelQuantizer(quant_config)
    model = quantizer.quantize_model(model)

    trainable_parameters_orthogonal, trainable_parameters_adam = RotationProcessor.get_trainable_parameters(
        model, rotation_config
    )

    hidden_size = model.config.hidden_size
    head_dim = hidden_size // model.config.num_attention_heads
    intermediate_size = model.config.intermediate_size
    num_key_value_heads = model.config.num_key_value_heads

    if train_smooth:
        # Normalization for: qkv, gate_up, o_proj, down_proj.
        expected_norm_trainable = len(model.model.layers) * 4
        expected_norm_params_numel = len(model.model.layers) * (
            hidden_size * 2 + num_key_value_heads * head_dim + intermediate_size
        )

        assert len(trainable_parameters_adam) == expected_norm_trainable

        total_trainable_norm_numel = 0
        for param in trainable_parameters_adam:
            total_trainable_norm_numel += param.numel()

        assert total_trainable_norm_numel == expected_norm_params_numel

    if r1:
        if not online_r1_rotation:
            expected_r1_trainable = 1
            if rotation_size is None:
                expected_r1_trainable_numel = hidden_size * hidden_size
            else:
                expected_r1_trainable_numel = rotation_size * rotation_size
        else:
            if shared_parallel:
                factor = 2
            else:
                factor = 5
            expected_r1_trainable = factor * len(model.model.layers)

            if rotation_size is None:
                expected_r1_trainable_numel = factor * len(model.model.layers) * hidden_size * hidden_size
            else:
                expected_r1_trainable_numel = factor * len(model.model.layers) * rotation_size * rotation_size
    else:
        expected_r1_trainable = 0
        expected_r1_trainable_numel = 0

    if r2:
        expected_r2_trainable = len(model.model.layers)
        if rotation_size is None:
            expected_r2_trainable_numel = len(model.model.layers) * head_dim * head_dim
        else:
            expected_r2_trainable_numel = len(model.model.layers) * rotation_size * rotation_size
    else:
        expected_r2_trainable = 0
        expected_r2_trainable_numel = 0

    if r4:
        expected_r4_trainable = len(model.model.layers)

        if rotation_size is None:
            expected_r4_trainable_numel = len(model.model.layers) * intermediate_size * intermediate_size
        else:
            expected_r4_trainable_numel = len(model.model.layers) * rotation_size * rotation_size
    else:
        expected_r4_trainable = 0
        expected_r4_trainable_numel = 0

    expected_trainable_total = expected_r1_trainable + expected_r2_trainable + expected_r4_trainable

    assert len(trainable_parameters_orthogonal) == expected_trainable_total

    total_trainable_numel = 0
    for param in trainable_parameters_orthogonal:
        total_trainable_numel += param.numel()

    expected_trainable_total_numel = (
        expected_r1_trainable_numel + expected_r2_trainable_numel + expected_r4_trainable_numel
    )
    assert total_trainable_numel == expected_trainable_total_numel


@pytest.mark.parametrize("r1", [pytest.param(val, id=f"r1:{val}") for val in [True]])
@pytest.mark.parametrize("r2", [pytest.param(val, id=f"r2:{val}") for val in [True]])
@pytest.mark.parametrize("r4", [pytest.param(val, id=f"r4:{val}") for val in [True]])
@pytest.mark.parametrize(
    "online_r1_rotation", [pytest.param(val, id=f"online_r1_rotation:{val}") for val in [False, True]]
)
def test_get_online_rotation_layers(r1: bool, r2: bool, r4: bool, online_r1_rotation: bool):
    model_id = "HuggingFaceTB/SmolLM-135M"

    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation=ATTN_IMPLEMENTATION)
    model = model.eval()
    model = model.to(torch_device)

    rotation_config = RotationConfig(
        scaling_layers=SCALING_LAYERS_LLAMA,
        rotation_size=None,
        r1=r1,
        r2=r2,
        r3=False,
        r4=r4,
        online_r1_rotation=None if not r1 else online_r1_rotation,
        trainable=False,
    )

    online_rotation_layers = RotationProcessor.get_online_rotation_layers(rotation_config, model)

    if r1:
        if not online_r1_rotation:
            expected_r1_online = 0
        else:
            expected_r1_online = 5 * len(model.model.layers)
    else:
        expected_r1_online = 0

    expected_r2_online = 0

    if r4:
        expected_r4_online = len(model.model.layers)
    else:
        expected_r4_online = 0

    expected_online_total = expected_r1_online + expected_r2_online + expected_r4_online

    assert len(online_rotation_layers) == expected_online_total


@local_test_only
@require_torch_cuda
@require_vllm
def test_load_vllm_hadamard():
    from vllm import LLM, SamplingParams

    model_id = "HuggingFaceTB/SmolLM-135M"

    mxfp4_input_spec = OCP_MXFP4Spec(is_dynamic=True, ch_axis=-1).to_quantization_spec()
    mxfp4_weight_spec = OCP_MXFP4Spec(is_dynamic=False, ch_axis=-1).to_quantization_spec()

    layer_quant_config = QLayerConfig(weight=mxfp4_weight_spec, input_tensors=mxfp4_input_spec)

    rotation_config = RotationConfig(
        scaling_layers=MODEL_TYPE_TO_SCALING_LAYERS["llama"],
        rotation_size=64,
        r1=True,
        r2=True,
        r3=False,
        r4=False,
        online_r1_rotation=True,
        trainable=False,
        mlp="mlp",
    )
    algo_config = [rotation_config]

    quant_config = QConfig(
        global_quant_config=layer_quant_config,
        algo_config=algo_config,
        exclude=["lm_head", "*.gate", "*.gate.linear"],
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation=ATTN_IMPLEMENTATION)
    model = model.eval()

    model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantizer = ModelQuantizer(quant_config)
    model = quantizer.quantize_model(model)

    with torch.no_grad(), TemporaryDirectory() as tmpdir:
        export_safetensors(model=model, output_dir=tmpdir, weight_format="real_quantized", pack_method="reorder")

        tokenizer.save_pretrained(tmpdir)

        llm = LLM(model=tmpdir)

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            prompt = output.prompt

        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


@local_test_only
@require_torch_cuda
@require_vllm
def test_load_vllm_tuned_orthogonal():
    from vllm import LLM, SamplingParams

    model_id = "HuggingFaceTB/SmolLM-135M"

    mxfp4_input_spec = OCP_MXFP4Spec(is_dynamic=True, ch_axis=-1).to_quantization_spec()
    mxfp4_weight_spec = OCP_MXFP4Spec(is_dynamic=False, ch_axis=-1).to_quantization_spec()

    layer_quant_config = QLayerConfig(weight=mxfp4_weight_spec, input_tensors=mxfp4_input_spec)

    rotation_config = RotationConfig(
        scaling_layers=MODEL_TYPE_TO_SCALING_LAYERS["llama"],
        rotation_size=64,
        r1=True,
        r2=True,
        r3=False,
        r4=False,
        online_r1_rotation=True,
        trainable=True,
        mlp="mlp",
    )
    algo_config = [rotation_config]

    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation=ATTN_IMPLEMENTATION)
    model = model.eval()

    model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quant_config_rotation = QConfig(
        global_quant_config=layer_quant_config,
        algo_config=algo_config,
        exclude=["lm_head", "*.gate", "*.gate.linear"],
    )

    model, _, _, _, _, _, _ = run_rotation_training(
        quant_config_rotation, learning_rate=1.5, dtype="fp32", model_id=model_id, online_r1_rotation=True, steps=5
    )

    model = model.eval()

    quant_config_quantization = copy.deepcopy(quant_config_rotation)
    quant_config_quantization.algo_config = []

    model = RotationProcessor.post_process_trained_rotation(model=model, quantization_config=quant_config_quantization)

    quantizer = ModelQuantizer(quant_config_quantization)

    with torch.no_grad():
        model = quantizer.quantize_model(model)
        model.quant_config.algo_config = quant_config_rotation.algo_config

    model = quantizer.freeze(model)

    with torch.no_grad(), TemporaryDirectory() as tmpdir:
        export_safetensors(model=model, output_dir=tmpdir, weight_format="real_quantized", pack_method="reorder")

        tokenizer.save_pretrained(tmpdir)

        llm = LLM(model=tmpdir)

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            prompt = output.prompt

        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
