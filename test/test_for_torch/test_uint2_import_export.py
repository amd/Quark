#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""uint2 per-group export/import round-trip, mirroring test_int2_import_export.

Verifies the new packed uint2 dtype: a quantized model exported with
weight_format=real_quantized (4 x uint2 packed per byte) re-imports to a model
whose outputs match the in-memory quantized model (atol 1e-4) — i.e. same results.
"""

import tempfile

import pytest
import torch
from test_hf_export_import import (  # reuse the validated helpers
    INPUT_IDS,
    QPARAMSLINEAR_OVERRIDES_STATE_DICT,
    _fix_loaded_weights_key_mismatch,
    _load_weights_from_safetensors,
    init_model,
    quantize_model,
    torch_device,
)
from transformers import AutoConfig

from quark.torch import export_safetensors, import_model_from_safetensors
from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver


@pytest.mark.parametrize("weight_format", ["real_quantized", "fake_quantized"])
def test_uint2_import_export(weight_format: str):
    model_id = "facebook/opt-125m"
    quant_spec = QTensorConfig(
        dtype=Dtype.uint2,
        qscheme=QSchemeType.per_group,
        observer_cls=PerGroupMinMaxObserver,
        symmetric=False,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
        ch_axis=1,
        group_size=8,
    )
    quant_config = QConfig(global_quant_config=QLayerConfig(weight=quant_spec), exclude=["*lm_head*"])
    quant_model = quantize_model(quant_config, model_name=model_id, multi_gpu=False, device_map=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_safetensors(model=quant_model, output_dir=tmpdir, weight_format=weight_format, pack_method="reorder")
        with torch.no_grad():
            ref_outputs = quant_model(INPUT_IDS).to_tuple()

        for device in ["meta", torch_device]:
            config = AutoConfig.from_pretrained(model_id)
            original_model = init_model(config, device)
            weight_dict = _load_weights_from_safetensors(tmpdir)
            if not QPARAMSLINEAR_OVERRIDES_STATE_DICT:
                weight_dict = _fix_loaded_weights_key_mismatch(
                    weight_dict, weight_format=weight_format, custom_mode="quark"
                )
            q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False).eval()
            q_model_state_dict = q_model.state_dict()

            if weight_format == "real_quantized":
                for key in weight_dict:
                    assert weight_dict[key].dtype == q_model_state_dict[key].dtype
                    assert weight_dict[key].shape == q_model_state_dict[key].shape

            for _, p in q_model.named_parameters():
                assert p.device != "meta"
            for _, b in q_model.named_buffers():
                assert b.device != "meta"
            if device == "meta":
                q_model = q_model.to(torch_device)

            with torch.no_grad():
                outputs = q_model(INPUT_IDS).to_tuple()
            for ref_output, output in zip(ref_outputs[0], outputs[0], strict=True):
                if torch_device.type == "cpu":
                    assert torch.equal(ref_output, output)
                else:
                    assert torch.allclose(output, ref_output, atol=1e-4)


if __name__ == "__main__":
    test_uint2_import_export("real_quantized")
    print("uint2 real_quantized round-trip: PASS (outputs match within atol 1e-4)")
