#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unpack Quark signed INT4 AWQ safetensors in memory."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor

_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def sign_extend_int4(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.int8)
    return torch.where(values >= 8, values - 16, values)


def unpack_signed_int4(
    packed: torch.Tensor,
    *,
    pack_reorder: bool,
) -> torch.Tensor:
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    unpacked = (packed.to(torch.int32)[:, :, None] >> shifts) & 0xF
    if pack_reorder:
        order = torch.tensor(_REVERSE_AWQ_PACK_ORDER, dtype=torch.long)
    else:
        order = torch.arange(8, dtype=torch.long)
    unpacked = unpacked[:, :, order].reshape(packed.shape[0], -1)
    return sign_extend_int4(unpacked)


def trim_output_dim_for(name: str) -> int | None:
    if name.endswith(".shared_expert_gate.weight"):
        return 1
    return None


def dequantize_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    *,
    group_size: int,
    pack_reorder: bool,
    out_dtype: torch.dtype,
    trim_output_dim: int | None = None,
) -> torch.Tensor:
    weights = unpack_signed_int4(qweight, pack_reorder=pack_reorder).to(torch.float32)
    zeros = unpack_signed_int4(qzeros, pack_reorder=pack_reorder).to(torch.float32)
    scales = scales.to(torch.float32)

    if trim_output_dim is not None:
        weights = weights[:, :trim_output_dim]
        zeros = zeros[:, :trim_output_dim]
        scales = scales[:, :trim_output_dim]

    scales = scales.repeat_interleave(group_size, dim=0)
    zeros = zeros.repeat_interleave(group_size, dim=0)
    return ((weights - zeros) * scales).T.contiguous().to(out_dtype)


def iter_safetensors(model_dir) -> Iterable:
    from pathlib import Path
    import json

    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        seen: set[str] = set()
        for shard in index["weight_map"].values():
            if shard not in seen:
                seen.add(shard)
                yield model_dir / shard
        return
    yield from sorted(model_dir.glob("*.safetensors"))


class QuarkShardCache:
    """Keep safetensors readers open while lazily loading Quark tensors."""

    def __init__(self, quark_dir) -> None:
        from pathlib import Path
        import json

        self.quark_dir = Path(quark_dir)
        self._readers: dict[Any, Any] = {}
        self._tensor_shard: dict[str, Any] = {}
        self._build_weight_map()

    def _build_weight_map(self) -> None:
        import json

        index_path = self.quark_dir / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for name, shard in index["weight_map"].items():
                self._tensor_shard[name] = self.quark_dir / shard
            return

        for shard in iter_safetensors(self.quark_dir):
            with safe_open(shard, framework="pt", device="cpu") as reader:
                for name in reader.keys():
                    self._tensor_shard[name] = shard

    def _reader(self, shard):
        if shard not in self._readers:
            self._readers[shard] = safe_open(shard, framework="pt", device="cpu")
        return self._readers[shard]

    def get_tensor(self, name: str) -> Tensor:
        shard = self._tensor_shard[name]
        return self._reader(shard).get_tensor(name)

    def keys(self) -> list[str]:
        return list(self._tensor_shard.keys())

    def close(self) -> None:
        self._readers.clear()


def build_quark_tensor_loaders(
    quark_dir,
    *,
    group_size: int,
    pack_reorder: bool,
    out_dtype: torch.dtype,
    max_tensors: int | None = None,
) -> tuple[dict[str, Callable[[], Tensor]], QuarkShardCache]:
    cache = QuarkShardCache(quark_dir)
    loaders: dict[str, Callable[[], Tensor]] = {}
    key_set = set(cache.keys())
    processed = 0

    for name in cache.keys():
        if max_tensors is not None and processed >= max_tensors:
            break
        if name.endswith((".weight_scale", ".weight_zero_point")):
            continue

        scale_name = name.removesuffix(".weight") + ".weight_scale"
        zero_name = name.removesuffix(".weight") + ".weight_zero_point"

        def _load_plain(n: str = name, od: torch.dtype = out_dtype) -> Tensor:
            tensor = cache.get_tensor(n)
            if tensor.is_floating_point():
                return tensor.to(od)
            return tensor

        def _load_quant(
            n: str = name,
            sn: str = scale_name,
            zn: str = zero_name,
            od: torch.dtype = out_dtype,
            gs: int = group_size,
            pr: bool = pack_reorder,
        ) -> Tensor:
            return dequantize_weight(
                cache.get_tensor(n),
                cache.get_tensor(sn),
                cache.get_tensor(zn),
                group_size=gs,
                pack_reorder=pr,
                out_dtype=od,
                trim_output_dim=trim_output_dim_for(n),
            )

        if (
            name.endswith(".weight")
            and scale_name in key_set
            and zero_name in key_set
        ):
            sample = cache.get_tensor(name)
            if sample.dtype == torch.int32:
                loaders[name] = _load_quant
                processed += 1
                continue

        loaders[name] = _load_plain
        processed += 1

    return loaders, cache
