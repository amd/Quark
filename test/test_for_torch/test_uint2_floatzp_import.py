#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Native import of a packed uint2 model with a FLOAT zero-point.

Builds a Quark-native export by hand (Pack_uint2 weights + fp16 scale + fp16
float zero-point + embedded quantization_config with zero_point_type=float32),
loads it via import_model_from_safetensors, and checks the reconstructed-model
logits match the in-memory model (the weights are exactly on a 4-level grid, so
the float-zp linear dequant (q-1.5)*scale reproduces them within fp16 scale storage).
"""

import json
import os
import tempfile

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from quark.torch import import_model_from_safetensors
from quark.torch.export.nn.modules.realquantizer import StaticScaledRealQuantizer
from quark.torch.quantization.config.config import QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType, ZeroPointType
from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver
from quark.torch.utils import create_pack_method

MODEL = "facebook/opt-125m"
G = 64
LEVELS = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])
INPUT = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])


def _is_quant_linear(n, m):
    return isinstance(m, torch.nn.Linear) and "lm_head" not in n and n != ""


def test_uint2_floatzp_native_import():
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    # snap every quantized linear onto an exact 4-level grid (QAD-like)
    with torch.no_grad():
        for n, mod in m.named_modules():
            if _is_quant_linear(n, mod) and mod.weight.shape[1] % G == 0:
                o, i = mod.weight.shape
                idx = torch.randint(0, 4, (o, i))
                s = torch.rand(o, i // G) * 0.2 + 0.05
                mod.weight.data = (LEVELS[idx] * s.repeat_interleave(G, 1)).float()
        ref = m(INPUT).logits

    pm = create_pack_method("per_group", "uint2")
    sd, qkeys = {}, set()
    orig = m.state_dict()
    with torch.no_grad():
        for n, mod in m.named_modules():
            if not (_is_quant_linear(n, mod) and mod.weight.shape[1] % G == 0):
                continue
            w = mod.weight.data.float()
            o, i = w.shape
            wg = w.view(o, i // G, G)
            maxabs = wg.abs().amax(-1).clamp(min=1e-8)
            idx = (
                (wg.unsqueeze(-1) / maxabs.unsqueeze(-1).unsqueeze(-1) - LEVELS.view(1, 1, 1, 4))
                .abs()
                .argmin(-1)
                .view(o, i)
            )
            scale_q = (2.0 / 3.0) * maxabs
            zp = torch.full((o, i // G), 1.5)
            sd[f"{n}.weight"] = pm.pack(idx.to(torch.int32), True)
            sd[f"{n}.weight_scale"] = scale_q.t().contiguous().to(torch.float16)
            sd[f"{n}.weight_zero_point"] = zp.t().contiguous().to(torch.float16)
            qkeys.add(f"{n}.weight")
            qkeys.add(f"{n}.bias")
            if mod.bias is not None:
                sd[f"{n}.bias"] = mod.bias.data.to(torch.float16)
    for k, v in orig.items():
        if k not in qkeys:
            sd[k] = v.clone()

    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as d:
        save_file(sd, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
        cfg = AutoConfig.from_pretrained(MODEL).to_dict()
        wspec = {
            "dtype": "uint2",
            "qscheme": "per_group",
            "ch_axis": 1,
            "group_size": G,
            "symmetric": False,
            "zero_point_type": "float32",
            "scale_type": "float",
            "is_dynamic": False,
            "round_method": "half_even",
            "observer_cls": "PerGroupMinMaxObserver",
        }
        cfg["quantization_config"] = {
            "quant_method": "quark",
            "global_quant_config": {"weight": wspec, "bias": None, "input_tensors": None, "output_tensors": None},
            "exclude": ["lm_head"],
            "layer_quant_config": {},
            "layer_type_quant_config": {},
            "kv_cache_quant_config": {},
            "quant_mode": "eager_mode",
            "export": {
                "weight_format": "real_quantized",
                "pack_method": "reorder",
                "kv_cache_group": [],
                "min_kv_scale": 0.0,
            },
        }
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        base = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(MODEL))
        q = import_model_from_safetensors(base, model_dir=d, multi_device=False).eval()
        with torch.no_grad():
            out = q(INPUT).logits
        # fp16 scale storage only; logits should match closely
        assert (ref - out).abs().max().item() < 5e-2


def _real_quantizer(zero_point_type: ZeroPointType, out_features: int = 6, n_groups: int = 3):
    """A per-group uint2 real quantizer with buffers laid out as on export."""
    qspec = QTensorConfig(
        dtype=Dtype.uint2,
        qscheme=QSchemeType.per_group,
        observer_cls=PerGroupMinMaxObserver,
        symmetric=False,
        scale_type=ScaleType.float,
        zero_point_type=zero_point_type,
        round_method=RoundType.half_even,
        is_dynamic=False,
        ch_axis=1,
        group_size=G,
    )
    quantizer = StaticScaledRealQuantizer(
        qspec=qspec,
        quantizer=None,
        reorder=True,
        real_quantized=True,
        float_dtype=torch.float16,
        device=torch.device("cpu"),
        scale_shape=(out_features, n_groups),
        zero_point_shape=(out_features, n_groups),
    )
    with torch.no_grad():
        quantizer.scale.copy_(torch.rand(out_features, n_groups, dtype=torch.float16) + 0.1)
        # 1.5 centres the four uint2 levels, as the float-zp dequant above expects.
        quantizer.zero_point.copy_(torch.full((out_features, n_groups), 1.5))
    return quantizer


def test_float_zero_point_is_not_bit_packed() -> None:
    """An integer zero-point is bit-packed next to the weight; a float one must not be.

    ``unpack_params`` reads float zero-points back without bit-unpacking, and packing
    a float tensor crashes outright, so ``pack_zero_point`` has to skip them.
    """
    quantizer = _real_quantizer(ZeroPointType.float32)
    before = quantizer.zero_point.clone()

    quantizer.pack_zero_point()

    assert quantizer.zero_point.is_floating_point(), f"float zero-point was packed to {quantizer.zero_point.dtype}"
    assert torch.equal(quantizer.zero_point, before), "float zero-point must pass through untouched"


def test_float_zero_point_follows_the_per_group_scale_transpose() -> None:
    """A float zero-point must be transposed with the scale, or import misreads it.

    Per-group export stores the scale transposed to ``[n_groups, out_features]``;
    an unpacked zero-point that keeps the original orientation comes back with the
    wrong shape.
    """
    quantizer = _real_quantizer(ZeroPointType.float32)
    out_features, n_groups = quantizer.zero_point.shape

    quantizer.maybe_convert_and_transpose_scale()

    assert quantizer.scale.shape == (n_groups, out_features)
    assert quantizer.zero_point.shape == quantizer.scale.shape, (
        f"zero-point is {tuple(quantizer.zero_point.shape)} but the scale is "
        f"{tuple(quantizer.scale.shape)}; the two must stay aligned"
    )


if __name__ == "__main__":
    test_uint2_floatzp_native_import()
    test_float_zero_point_is_not_bit_packed()
    test_float_zero_point_follows_the_per_group_scale_transpose()
    print("uint2 float-zp native import: PASS")
