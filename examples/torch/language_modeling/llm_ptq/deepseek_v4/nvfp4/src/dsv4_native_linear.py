#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Loads and runs the DeepSeek-V4-Pro checkpoint's model code (inference/model.py).
# Copyright (c) 2023 DeepSeek. Licensed under the MIT License.
#

"""
Load a native DeepSeek-V4-Pro checkpoint into a model that Quark can calibrate.

The Stage 2 calibration needs the model to expose plain ``nn.Linear`` modules so
Quark replaces them with QuantLinear, while keeping the original FP4/FP8 weights
on CPU (the model is far larger than one GPU). This module provides:

* :func:`load_all_weights_native` — read the native FP4/FP8 weights + scales into
  CPU RAM and assign them onto the freshly built model.
* :class:`NativeLinear` — a wrapper around a native FP4/FP8 linear that presents a
  scalar-placeholder ``nn.Linear`` proxy to Quark and dequantizes its stored
  weight to BF16 only transiently inside ``forward`` (keeping GPU memory near-zero
  for inactive blocks).
* :func:`wrap_native_linears` — replace the MoE expert linears (routed + shared)
  with :class:`NativeLinear`.
"""

from __future__ import annotations

import json
import re as _re
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import safe_open

from quark.common.utils.log import ScreenLogger
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize

logger = ScreenLogger(__name__)

# E2M1 FP4 code -> float value LUT (low nibble = first element, little endian).
_FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def load_all_weights_native(model, hf_path):
    """Load native FP4/FP8 weights into CPU RAM."""
    hf_path = Path(hf_path)
    weight_map = json.loads((hf_path / "model.safetensors.index.json").read_text())["weight_map"]
    param_keys = {n for n, _ in model.named_parameters()}
    needed = param_keys | {k for k in weight_map if k.endswith(".scale")}
    files_needed = {hf_path / shard for k, shard in weight_map.items() if k in needed}
    logger.info(f"{len(files_needed)} shard(s) …")
    loaded = {}
    for fp in sorted(files_needed):
        with safe_open(str(fp), framework="pt", device="cpu") as f:
            for name in f.keys():  # noqa: SIM118  (safetensors handle is not iterable; .keys() required)
                if name in needed:
                    t = f.get_tensor(name)
                    if t.dtype == torch.int8:
                        t = t.view(torch.float4_e2m1fn_x2)
                    loaded[name] = t
    for key in list(loaded.keys()):
        if key.endswith(".wo_a.weight"):
            w = loaded[key]
            s_key = key.replace(".weight", ".scale")
            s = loaded.get(s_key)
            if s is not None and w.dtype == torch.float8_e4m3fn:
                w_f = w.to(torch.float32)
                out_b, in_b = s.shape
                s_f = torch.pow(2.0, s.view(torch.uint8).float() - 127.0)
                s_f = s_f.unsqueeze(-1).unsqueeze(-1).expand(out_b, in_b, 128, 128)
                s_f = s_f.permute(0, 2, 1, 3).reshape(out_b * 128, in_b * 128)[: w.shape[0], : w.shape[1]]
                loaded[key] = (w_f * s_f).to(torch.bfloat16)
    assigned = 0
    for full_key, tensor in loaded.items():
        parts = full_key.split(".")
        mod = model
        try:
            for part in parts[:-1]:
                mod = getattr(mod, part)
            attr = parts[-1]
        except AttributeError:
            continue
        _quantized_dtypes = {torch.float4_e2m1fn_x2, torch.float8_e4m3fn, torch.float8_e5m2}
        if hasattr(mod, "_parameters") and attr in mod._parameters:
            existing = mod._parameters[attr]
            if (
                existing is not None
                and not existing.is_meta
                and tensor.is_floating_point()
                and tensor.dtype not in _quantized_dtypes
                and existing.dtype != tensor.dtype
            ):
                tensor = tensor.to(existing.dtype)
            new_param = nn.Parameter(tensor, requires_grad=False)
            if attr == "weight" and "scale" in mod._parameters and mod._parameters["scale"] is not None:
                new_param.scale = mod._parameters["scale"]
            mod._parameters[attr] = new_param
            assigned += 1
        elif hasattr(mod, "_buffers") and attr in mod._buffers:
            mod._buffers[attr] = tensor
            assigned += 1
    for mod_name, mod in model.named_modules():
        params = getattr(mod, "_parameters", {})
        if "weight" not in params:
            continue
        w = params["weight"]
        if w is None:
            continue
        s = params.get("scale")
        if s is not None:
            w.scale = s
        if not hasattr(w, "scale") or w.scale is None:
            mod_full = mod_name + ".scale"
            if mod_full in loaded:
                w.scale = nn.Parameter(loaded[mod_full], requires_grad=False)
    logger.info(f"{assigned}/{len(loaded)} keys assigned")


class NativeLinear(nn.Module):
    """Wraps a native FP4/FP8 checkpoint linear.

    Stores the original quantized weight (cpu_native) and an optional per-group
    scale (cpu_scale) on CPU.  The inner `self.proxy` is a standard nn.Linear
    whose weight is a 1-element scalar placeholder; Quark replaces proxy with
    QuantLinear.  The real `[out, in]` weight tensor is *never* registered as an
    nn.Parameter so lazy_loader will not try to offload/restore it to GPU — only
    the scalar placeholder is in the parameter dict, keeping GPU memory near-zero
    for inactive blocks.

    On forward():
      1. Dequantize cpu_native -> BF16 on the current device (transient).
      2. Swap proxy.weight.data for the BF16 tensor (no extra allocation).
      3. Call proxy.forward(x) -- Quark's QuantLinear runs fake-quant here.
      4. Restore proxy.weight.data to the scalar placeholder so lazy_loader's
         post-hook restores a ~zero-size tensor, not a 16 MB one.
    """

    def __init__(
        self,
        cpu_native: torch.Tensor,
        cpu_scale: torch.Tensor | None,
        out_features: int,
        in_features: int,
        bias: torch.Tensor | None,
        native_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.cpu_native = cpu_native  # stays on CPU, never moved
        self.cpu_scale = cpu_scale  # stays on CPU, may be None
        self.native_dtype = native_dtype
        self.out_features = out_features
        self.in_features = in_features

        # proxy: plain nn.Linear whose weight is a 1-element scalar placeholder.
        # Using a tiny placeholder means lazy_loader only copies 4 bytes per
        # QuantLinear weight to GPU, not the full [out, in] BF16 tensor.
        # Quark will replace proxy with QuantLinear.
        #
        # We create nn.Linear(1, 1) to avoid the expensive kaiming_uniform_ init
        # on a full [out, in] matrix (which for MoE experts is [2048, 4096] x 256 x 3
        # and takes ~30 minutes total).  Then we patch in_features/out_features so
        # Quark's QuantLinear.from_float() reads the correct shape.
        self.proxy = nn.Linear(1, 1, bias=(bias is not None))
        self.proxy.in_features = in_features
        self.proxy.out_features = out_features
        # Replace the [1,1] parameter with a 1-element scalar sentinel.
        self.proxy.weight = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16), requires_grad=False)
        if bias is not None:
            self.proxy.bias = nn.Parameter(bias.to(torch.bfloat16), requires_grad=False)
        else:
            self.proxy.bias = None
        self._weight_materialized = False  # tracks whether proxy.weight has real shape

    def _dequant_to_bf16(self, device: torch.device) -> torch.Tensor:
        w = self.cpu_native
        scale = self.cpu_scale

        if self.native_dtype == torch.float4_e2m1fn_x2:
            raw = w.view(torch.uint8).to(device)
            lut = _FP4_LUT.to(device)
            lo = (raw & 0x0F).long()
            hi = ((raw >> 4) & 0x0F).long()
            wf = torch.stack([lut[lo], lut[hi]], dim=-1).flatten(-2)  # float32
            if scale is not None:
                s = torch.pow(2.0, scale.view(torch.uint8).to(device).float() - 127.0)
                s = s.repeat_interleave(32, dim=-1)
                if s.shape[1] > wf.shape[1]:
                    s = s[:, : wf.shape[1]]
                wf = wf * s
            return wf.to(torch.bfloat16)

        if self.native_dtype == torch.float8_e4m3fn:
            wf = w.to(device).to(torch.float32)
            if scale is not None:
                out_b, in_b = scale.shape
                s = torch.pow(2.0, scale.view(torch.uint8).to(device).float() - 127.0)
                s = s.unsqueeze(-1).unsqueeze(-1).expand(out_b, in_b, 128, 128)
                s = s.permute(0, 2, 1, 3).reshape(out_b * 128, in_b * 128)[: w.shape[0], : w.shape[1]]
                wf = wf * s
            return wf.to(torch.bfloat16)

        # Already BF16 or other float type.
        return w.to(device=device, dtype=torch.bfloat16)

    @property
    def weight(self) -> nn.Parameter:
        """Expose proxy.weight.

        Some attention paths (e.g. wo_a) read .weight directly instead of calling
        forward(). We materialise BF16 on demand if the proxy weight is still a scalar
        placeholder (numel==1) or is a meta tensor.
        """
        w = self.proxy.weight
        # Materialise if: meta device, or scalar sentinel (numel==1 means not yet expanded)
        need_materialize = w.is_meta or w.numel() == 1
        if need_materialize and self.cpu_native is not None and not self.cpu_native.is_meta:
            device = torch.device("cpu") if w.is_meta else w.device
            self.proxy.weight.data = self._dequant_to_bf16(device)
            self._weight_materialized = True
        return self.proxy.weight

    @property
    def bias(self):
        return self.proxy.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        bf16_w = self._dequant_to_bf16(device)
        self.proxy.weight.data = bf16_w
        self._weight_materialized = True

        # Move quantizer buffers to GPU transiently for fake-quant, then move
        # them back to CPU after forward.  lazy_loader only manages _parameters
        # (not _buffers), so without this the observer-expanded scale buffers
        # from 768 QuantLinears × 43 blocks accumulate on GPU and OOM.

        _q_bufs: list[tuple[object, str]] = []
        for qname in ("_weight_quantizer", "_input_quantizer"):
            q = getattr(self.proxy, qname, None)
            if q is None:
                continue
            stages = [q] if isinstance(q, ScaledFakeQuantize) else list(q)
            for stage in stages:
                for sub in stage.modules():
                    for bname, buf in list(sub._buffers.items()):
                        if buf is not None and not buf.is_meta:
                            _q_bufs.append((sub, bname))
                            if buf.device != device:
                                sub._buffers[bname] = buf.to(device, non_blocking=True)
                obs = getattr(stage, "observer", None)
                if obs is not None:
                    for attr in ("amax", "min_val", "max_val"):
                        v = getattr(obs, attr, None)
                        if isinstance(v, torch.Tensor) and v.device != device:
                            setattr(obs, attr, v.to(device, non_blocking=True))

        proxy_input = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
        out = self.proxy(proxy_input)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        del bf16_w

        for sub, bname in _q_bufs:
            buf = sub._buffers[bname]
            if buf is not None and buf.device != torch.device("cpu"):
                sub._buffers[bname] = buf.to("cpu")

        for qname in ("_weight_quantizer", "_input_quantizer"):
            q = getattr(self.proxy, qname, None)
            if q is None:
                continue
            stages = [q] if isinstance(q, ScaledFakeQuantize) else list(q)
            for stage in stages:
                obs = getattr(stage, "observer", None)
                if obs is None:
                    continue
                for attr in ("amax", "min_val", "max_val"):
                    v = getattr(obs, attr, None)
                    if isinstance(v, torch.Tensor) and v.device.type == "cuda":
                        setattr(obs, attr, v.to("cpu"))

        # Restore scalar sentinel so lazy_loader's post-hook snapshots ~4 bytes,
        # not the full [out, in] BF16 tensor (which would refill GPU for all experts).
        self.proxy.weight.data = torch.zeros(1, dtype=torch.bfloat16, device=device)
        self._weight_materialized = False
        return out


# Wrap both routed experts (ffn.experts.<n>.w{1,2,3}, FP4 native) AND the
# shared expert (ffn.shared_experts.w{1,2,3}, FP8 native), so we collect input
# stats for both.
_ROUTED_EXPERT_WRAP_RE = _re.compile(r"\.ffn\.(?:experts\.\d+|shared_experts)\.w[123]$")


def wrap_native_linears(model: nn.Module) -> int:
    """Replace MoE expert linears (w1/w2/w3) with NativeLinear.

    Matches both routed experts (ffn.experts.<n>.w{1,2,3}) and the shared
    expert (ffn.shared_experts.w{1,2,3}).  Other linears (attention, gate,
    lm_head) keep their original DS-V4 ``Linear`` class (which inherits
    ``nn.Module``, not ``nn.Linear``).  Quark only sees ``nn.Linear``
    subclasses, so only the wrapped proxies become QuantLinear — no exclude
    list needed.
    """
    _target_class_names = {"Linear", "ColumnParallelLinear", "RowParallelLinear"}
    count = 0

    for parent_name, parent_mod in list(model.named_modules()):
        for child_name, child_mod in list(parent_mod.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not _ROUTED_EXPERT_WRAP_RE.search(full_name):
                continue
            if type(child_mod).__name__ not in _target_class_names:
                continue
            w_param = child_mod._parameters.get("weight")
            if w_param is None or w_param.is_meta:
                continue
            w = w_param.data
            native_dtype = w.dtype

            scale = getattr(w_param, "scale", None)
            if scale is None:
                s_param = child_mod._parameters.get("scale")
                if s_param is not None and not s_param.is_meta:
                    scale = s_param.data

            bias = child_mod.bias.data if child_mod.bias is not None else None
            out_f, in_f = w.shape[0], w.shape[1] if w.dim() > 1 else 1

            if native_dtype == torch.float4_e2m1fn_x2:
                in_f = w.shape[1] * 2

            cpu_scale = scale.detach().cpu() if scale is not None else None
            wrapper = NativeLinear(w.cpu(), cpu_scale, out_f, in_f, bias, native_dtype)
            setattr(parent_mod, child_name, wrapper)
            count += 1

    return count
