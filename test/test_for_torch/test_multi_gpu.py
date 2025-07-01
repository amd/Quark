#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import pytest

from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import Config, QuantizationSpec, QuantizationConfig, AWQConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType
from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver

from quark.shares.utils.testing_utils import require_torch_cuda, torch_device, require_accelerate

DEFAULT_UINT4_PER_GROUP_ASYM_SPEC = QuantizationSpec(dtype=Dtype.uint4,
                                                     observer_cls=PerGroupMinMaxObserver,
                                                     symmetric=False,
                                                     scale_type=ScaleType.float,
                                                     round_method=RoundType.half_even,
                                                     qscheme=QSchemeType.per_group,
                                                     ch_axis=1,
                                                     is_dynamic=False,
                                                     group_size=128)

DEFAULT_W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=DEFAULT_UINT4_PER_GROUP_ASYM_SPEC)


DEFAULT_AWQ_CONFIG = Config(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG, algo_config=AWQConfig())

def get_dataloader(model_name="facebook/opt-125m", device=torch_device):
    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs['input_ids'].to(device))
    return calib_dataloader


@require_torch_cuda
@require_accelerate
@pytest.mark.accelerate_test
def test_smoke_multi_gpu_load_to_cpu_or_disk():
    model_kwargs = {"torch_dtype": "auto", "max_memory": {0: "0.1GB", "cpu": "100GB"}}
    quantizer = ModelQuantizer(DEFAULT_AWQ_CONFIG)
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", device_map="auto", **model_kwargs, trust_remote_code=True)
    model.eval()
    calib_dataloader = get_dataloader("facebook/opt-125m", model.device)
    try:
        quant_model = quantizer.quantize_model(model, calib_dataloader)
    except MemoryError as e:
        assert "Out of memory. The available GPU memory is insufficient to load the entire model." in str(e)
    else:
        raise ValueError("ValueError of pack is not raised")
