# ruff: noqa: E402

import os
import re
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

# TODO: Remove when ROCm 7.2 support is dropped.
# Workaround for segfaults: https://github.com/ROCm/rocm-systems/issues/5071
# The patch https://github.com/ROCm/rocm-systems/pull/4066 is part of ROCm 7.14.
# See quark/torch/algorithm/gptq/gptq.py.
# Must be set before `hsa_init` (ROCm initialization), so this cannot be done via a
# pytest fixture.
_original_hsa_tools_disable_register = os.environ.get("HSA_TOOLS_DISABLE_REGISTER")
os.environ["HSA_TOOLS_DISABLE_REGISTER"] = "1"


@pytest.fixture(autouse=True, scope="module")
def _restore_hsa_tools_disable_register() -> Generator[None, None, None]:
    # Restores the pre-existing environment once this module's tests are done, so the workaround
    # does not leak into later test modules or into subprocesses spawned by later tests.
    yield
    if _original_hsa_tools_disable_register is None:
        os.environ.pop("HSA_TOOLS_DISABLE_REGISTER", None)
    else:
        os.environ["HSA_TOOLS_DISABLE_REGISTER"] = _original_hsa_tools_disable_register


import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.common.utils.testing_utils import require_torch_higher_or_equal, skip_if_no_gpu, torch_device
from quark.torch import ModelQuantizer
from quark.torch.algorithm.common import BaseHessianProcessor
from quark.torch.algorithm.gptq import gptq as gptq_module
from quark.torch.algorithm.gptq.gptq import GptqProcessor, _compute_hinv_cholesky_factor
from quark.torch.quantization import Uint4PerChannelSpec
from quark.torch.quantization.config.config import (
    GPTQConfig,
    OCP_MXFP4Spec,
    QConfig,
    QLayerConfig,
    Uint4PerGroupSpec,
)
from quark.torch.utils import getattr_recursive, setattr_recursive


def test_gptq_hinv_cholesky_factor_matches_original_path_for_spd_hessian():
    H = torch.tensor([[2.0, 0.5], [0.5, 1.5]], dtype=torch.float32)
    damp = torch.tensor(0.01, dtype=H.dtype)

    expected = H.clone()
    diag = torch.arange(expected.shape[0], device=expected.device)
    expected[diag, diag] += damp
    expected = torch.linalg.cholesky(expected)
    expected = torch.cholesky_inverse(expected)
    expected = torch.linalg.cholesky(expected, upper=True)

    actual = _compute_hinv_cholesky_factor(H.clone(), damp)

    torch.testing.assert_close(actual, expected)


def test_gptq_hinv_cholesky_factor_recovers_from_nearly_pd_hessian():
    H = torch.tensor([[1.0, 1.0001], [1.0001, 1.0]], dtype=torch.float32)
    damp = torch.tensor(1e-8, dtype=H.dtype)

    Hinv = _compute_hinv_cholesky_factor(H, damp)

    assert torch.isfinite(Hinv).all()
    assert Hinv.shape == H.shape


def test_gptq_hinv_cholesky_factor_reraises_linalg_error_after_retry_exhaustion(monkeypatch: pytest.MonkeyPatch):
    H = torch.eye(2, dtype=torch.float32)
    damp = torch.tensor(0.01, dtype=H.dtype)
    attempts = 0

    def always_fail(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal attempts
        attempts += 1
        raise torch.linalg.LinAlgError("mock cholesky failure")

    monkeypatch.setattr(torch.linalg, "cholesky", always_fail)

    with pytest.raises(torch.linalg.LinAlgError, match="mock cholesky failure"):
        _compute_hinv_cholesky_factor(H.clone(), damp)

    assert attempts == 6


def test_register_original_weights_skips_modules_without_weight_quantizer() -> None:
    """Only submodules actually being quantized get a weight_orig snapshot.

    Regression test: register_original_weights() used to clone .weight for every
    submodule with one, including unquantized MoE expert Parameters -- wasted
    memory that contributed to OOM at large-expert-count scale.
    """
    layer = torch.nn.Module()
    layer.quantized = torch.nn.Linear(4, 4)
    layer.quantized._weight_quantizer = MagicMock()
    layer.no_quantizer = torch.nn.Linear(4, 4)
    # QuantLinear sets _weight_quantizer to None when there is no weight qspec.
    layer.quantizer_is_none = torch.nn.Linear(4, 4)
    layer.quantizer_is_none._weight_quantizer = None

    BaseHessianProcessor.register_original_weights(layer)

    torch.testing.assert_close(layer.quantized.weight_orig, layer.quantized.weight)
    assert not hasattr(layer.no_quantizer, "weight_orig")
    assert not hasattr(layer.quantizer_is_none, "weight_orig")


@pytest.mark.parametrize("helper", ["register_original_weights", "delete_original_weight_buffer"])
def test_subclass_cannot_override_the_snapshot_helpers(helper: str) -> None:
    """Overriding either helper must fail at class-creation time, not silently win at runtime.

    QronosProcessor used to carry byte-identical copies of both. Overriding
    register_original_weights shadows the base implementation's `_weight_quantizer` guard, so the
    subclass goes back to cloning every weight in the block -- exactly the OOM this PR fixes,
    reintroduced silently. Overriding delete_original_weight_buffer cannot create clones, but it
    can fail to clear the ones taken, so the pair is locked together. Enforced in
    __init_subclass__ because @final is inert here: pyproject disables mypy's "misc" code for
    quark.*, which is what "Cannot override final attribute" is reported under.
    """
    namespace = {helper: staticmethod(lambda layer: None)}
    expected = re.escape(f"overrides BaseHessianProcessor.{helper}, which is final")
    with pytest.raises(TypeError, match=expected):
        type("OverridingProcessor", (BaseHessianProcessor,), namespace)


def test_subclass_may_override_anything_else() -> None:
    """The guard must reject exactly the two locked helpers, not method overrides in general.

    Asserted with the hooks subclasses are actually expected to implement -- an empty subclass
    would also pass under a guard that wrongly rejected every override.
    """
    namespace = {
        "_get_algorithm_instance": lambda self, layer: None,
        "_quantize_layer": lambda self, algo, layer, group_size: None,
        "_collect_statistics": lambda self, *args: None,
    }
    assert type("PlainProcessor", (BaseHessianProcessor,), namespace) is not None


def test_gptq_processor_warns_on_rocm_7_2_hip_graph_instability(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_init(self: GptqProcessor, model: object, quant_algo_config: object, data_loader: object) -> None:
        self.use_cuda_graphs = True

    monkeypatch.setattr(BaseHessianProcessor, "__init__", fake_init)
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    monkeypatch.setenv("HSA_TOOLS_DISABLE_REGISTER", "0")
    warning_mock = MagicMock()
    monkeypatch.setattr(gptq_module.logger, "warning", warning_mock)

    quant_algo_config = MagicMock()
    quant_algo_config.damp_percent = 0.01

    GptqProcessor(model=MagicMock(), quant_algo_config=quant_algo_config, data_loader=MagicMock())

    warning_mock.assert_called_once()
    assert "ROCm 7.2 HIP Graph replay is known to have instabilities" in warning_mock.call_args[0][0]


# For torch requirement, refer to /pull/2529#issuecomment-235620
@require_torch_higher_or_equal("2.6")
@skip_if_no_gpu
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("dtype", ["uint4", "mxfp4"])
@pytest.mark.parametrize("qscheme", ["per_group", "per_channel"])
@pytest.mark.parametrize("model_id", ["facebook/opt-125m", "HuggingFaceTB/SmolLM-135M"])
def test_gptq_correctness(act_order: bool, dtype: str, qscheme: str, model_id: str) -> None:
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
            qspec = Uint4PerGroupSpec(1, 32, scale_type="float", is_dynamic=False).to_quantization_spec()
        elif qscheme == "per_channel":
            # actorder has no influence in this case.
            qspec = Uint4PerChannelSpec(
                0, symmetric=False, scale_type="float", round_method="half_even", is_dynamic=False
            ).to_quantization_spec()
    else:
        if qscheme == "per_group":
            qspec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
        else:
            pytest.skip("ocp mxfp4 not compatible with per_channel, per_tensor")

    global_quant_config = QLayerConfig(weight=qspec)

    gptq_config = GPTQConfig(
        model_decoder_layers=model_decoder_layers,
        inside_layer_modules=inside_layer_modules,
        desc_act=act_order,
        block_size=32,
    )

    config_gptq = QConfig(global_quant_config=global_quant_config, algo_config=[gptq_config])
    config_no_algo = QConfig(global_quant_config=global_quant_config)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    text = "Hello, how are you?"
    tokenized_inputs = tokenizer(text, return_tensors="pt").to(torch_device)
    calib_dataloader = DataLoader(tokenized_inputs["input_ids"])

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto").to(torch_device)
    model = model.eval()

    model2 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto").to(torch_device)
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
