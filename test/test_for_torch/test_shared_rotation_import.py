#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Native import of a shared-rotation ("factored") checkpoint.

A compact online-rotation export stores one rotation matrix per unique
``in_features`` as a top-level ``shared_input_rotation_<size>`` tensor instead of
one ``input_rotation`` per layer -- for a 40-layer model that is 2 matrices
rather than 160. The importer expands those into each rotated layer's
``input_rotation`` buffer so the ordinary per-layer rotation runtime is reused.

These tests pin the parts of that contract that were previously uncovered:

* the shared layout must reload to exactly the same logits as the equivalent
  per-layer layout,
* the expansion must not copy the matrix per layer, and
* a ``config.json`` written while the (since removed) ``shared_input_rotation``
  flag existed must still load.
"""

import json
import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

from quark.torch import import_model_from_safetensors
from quark.torch.export.nn.modules.qparamslinear import QParamsLinearWithRotation
from quark.torch.quantization.config.config import (
    OnlineRotationConfig,
    QConfig,
    QLayerConfig,
    QTensorConfig,
    RotationConfig,
)
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType, ZeroPointType
from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver
from quark.torch.utils import create_pack_method

MODEL = "facebook/opt-125m"
GROUP_SIZE = 64
LEVELS = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])
LEVEL_ZERO_POINT = 1.5
INPUT = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])


def _rotated_linears(model: torch.nn.Module) -> dict[str, torch.nn.Linear]:
    """Quantizable linears whose in_features is a multiple of the group size."""
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name
        and "lm_head" not in name
        and module.weight.shape[1] % GROUP_SIZE == 0
    }


def _orthogonal(size: int, seed: int) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(size, size, generator=torch.Generator().manual_seed(seed)))
    return q.contiguous().to(torch.float16)


def _pack_uint2_weights(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Snap each rotated linear onto the 4-level grid and pack it as affine uint2."""
    packer = create_pack_method("per_group", "uint2")
    state: dict[str, torch.Tensor] = {}
    replaced: set[str] = set()

    with torch.no_grad():
        for name, module in _rotated_linears(model).items():
            weight = module.weight.data.float()
            out_features, in_features = weight.shape
            grouped = weight.view(out_features, in_features // GROUP_SIZE, GROUP_SIZE)
            max_abs = grouped.abs().amax(-1).clamp(min=1e-8)
            index = (
                (grouped.unsqueeze(-1) / max_abs.unsqueeze(-1).unsqueeze(-1) - LEVELS.view(1, 1, 1, 4))
                .abs()
                .argmin(-1)
                .view(out_features, in_features)
            )
            state[f"{name}.weight"] = packer.pack(index.to(torch.int32), True)
            state[f"{name}.weight_scale"] = ((2.0 / 3.0) * max_abs).t().contiguous().to(torch.float16)
            state[f"{name}.weight_zero_point"] = torch.full(
                (in_features // GROUP_SIZE, out_features), LEVEL_ZERO_POINT, dtype=torch.float16
            )
            # A non-trivial prescale keeps the `use_input_prescale` path exercised.
            state[f"{name}.input_prescale"] = torch.full((in_features,), 0.5, dtype=torch.float16)
            replaced.update({f"{name}.weight", f"{name}.bias"})
            if module.bias is not None:
                state[f"{name}.bias"] = module.bias.data.to(torch.float16)

    return state, replaced


def _quantization_config(rotated_names: list[str] | None, legacy_flag: bool) -> dict:
    weight_spec = QTensorConfig(
        dtype=Dtype.uint2,
        observer_cls=PerGroupMinMaxObserver,
        symmetric=False,
        qscheme=QSchemeType.per_group,
        ch_axis=1,
        group_size=GROUP_SIZE,
        is_dynamic=False,
        scale_type=ScaleType.float,
        zero_point_type=ZeroPointType.float32,
        round_method=RoundType.half_even,
    )
    rotation = RotationConfig(
        scaling_layers=None,
        r1=True,
        r2=False,
        r3=False,
        r4=False,
        online_r1_rotation=True,
        trainable=True,
        online_config=OnlineRotationConfig(
            shared_parallel=False,
            online_rotation_layers=rotated_names,
            use_input_prescale=True,
        ),
    )
    config = QConfig(
        global_quant_config=QLayerConfig(weight=weight_spec), algo_config=[rotation], exclude=["lm_head"]
    ).to_dict()
    # `to_dict()` carries the quant spec but not the export block the importer reads.
    config["export"] = {
        "kv_cache_group": [],
        "min_kv_scale": 0.0,
        "pack_method": "reorder",
        "weight_format": "real_quantized",
        "weight_merge_groups": None,
    }

    if legacy_flag:
        # Emulate a checkpoint exported while the flag still existed.
        for algo in config["algo_config"]:
            if algo.get("name") == "rotation":
                algo["online_config"]["shared_input_rotation"] = True

    return config


def _write_export(directory: str, shared: bool, legacy_flag: bool = False, declare_online_layers: bool = True) -> None:
    """Write a uint2 + online-rotation export using either rotation layout."""
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()

    state, replaced = _pack_uint2_weights(model)
    for key, value in model.state_dict().items():
        if key not in replaced:
            state[key] = value.clone()

    rotated = _rotated_linears(model)
    sizes = sorted({module.weight.shape[1] for module in rotated.values()})
    rotations = {size: _orthogonal(size, seed=size) for size in sizes}

    if shared:
        for size, matrix in rotations.items():
            state[f"shared_input_rotation_{size}"] = matrix
    else:
        for name, module in rotated.items():
            # A real per-layer export stores one tensor per layer on disk -- the very
            # duplication the shared layout exists to avoid.
            state[f"{name}.input_rotation"] = rotations[module.weight.shape[1]].clone()

    save_file(state, os.path.join(directory, "model.safetensors"), metadata={"format": "pt"})

    config = AutoConfig.from_pretrained(MODEL).to_dict()
    config["quantization_config"] = _quantization_config(
        sorted(rotated) if declare_online_layers else None, legacy_flag
    )
    with open(os.path.join(directory, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)


def _import(directory: str) -> torch.nn.Module:
    base = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(MODEL))
    return import_model_from_safetensors(base, model_dir=directory, multi_device=False).eval()


def test_shared_rotation_import_matches_per_layer() -> None:
    """One matrix per in_features must reload identically to one per layer."""
    with tempfile.TemporaryDirectory() as shared_dir, tempfile.TemporaryDirectory() as per_layer_dir:
        _write_export(shared_dir, shared=True)
        _write_export(per_layer_dir, shared=False)

        with torch.no_grad():
            shared_logits = _import(shared_dir)(INPUT).logits
            per_layer_logits = _import(per_layer_dir)(INPUT).logits

        assert torch.equal(shared_logits, per_layer_logits), (
            "shared_input_rotation_<size> expansion changed the result relative to "
            f"per-layer input_rotation (max |delta| = {(shared_logits - per_layer_logits).abs().max().item()})"
        )


def test_shared_rotation_import_populates_every_rotated_layer() -> None:
    """Every rotated layer must end up with the matrix for its own in_features."""
    with tempfile.TemporaryDirectory() as directory:
        _write_export(directory, shared=True)
        model = _import(directory)

        rotated = [m for _, m in model.named_modules() if isinstance(m, QParamsLinearWithRotation)]
        assert rotated, "no QParamsLinearWithRotation layers were built from the config"

        for module in rotated:
            matrix = module.transform.rotation_matrix
            assert matrix is not None
            assert matrix.shape == (module.rotation_size, module.rotation_size)
            assert torch.count_nonzero(matrix) > 0, "rotation buffer was left zeroed"


def test_shared_rotation_expansion_does_not_copy_per_layer() -> None:
    """Layers sharing an in_features must share one tensor, not a copy each.

    The matrix is read-only at load time, so cloning it per layer costs one
    ``in_features^2`` allocation per rotated layer -- tens of GB for a large
    model whose export is under a gigabyte.
    """
    with tempfile.TemporaryDirectory() as directory:
        _write_export(directory, shared=True)
        model = _import(directory)

        pointers_by_size: dict[int, set[int]] = {}
        counts_by_size: dict[int, int] = {}
        for _, module in model.named_modules():
            if isinstance(module, QParamsLinearWithRotation):
                size = module.rotation_size
                pointers_by_size.setdefault(size, set()).add(module.transform.rotation_matrix.data_ptr())
                counts_by_size[size] = counts_by_size.get(size, 0) + 1

        shared_sizes = {size: count for size, count in counts_by_size.items() if count > 1}
        assert shared_sizes, "expected at least one in_features used by several rotated layers"

        for size, count in shared_sizes.items():
            assert len(pointers_by_size[size]) == 1, (
                f"in_features={size} is used by {count} layers but produced "
                f"{len(pointers_by_size[size])} distinct rotation tensors; the shared "
                "matrix is being copied per layer"
            )


@pytest.mark.parametrize("legacy_flag", [False, True])
def test_removed_shared_input_rotation_flag_still_loads(legacy_flag: bool) -> None:
    """`online_config.shared_input_rotation` was removed; old exports must still load.

    The importer detects the shared layout from the checkpoint tensor names, so the
    flag never had an effect -- but it was written into ``config.json``, and
    ``OnlineRotationConfig`` rejects unknown keyword arguments.
    """
    with tempfile.TemporaryDirectory() as directory:
        _write_export(directory, shared=True, legacy_flag=legacy_flag)

        with open(os.path.join(directory, "config.json")) as handle:
            written = json.load(handle)
        online = next(
            algo["online_config"]
            for algo in written["quantization_config"]["algo_config"]
            if algo["name"] == "rotation"
        )
        assert ("shared_input_rotation" in online) is legacy_flag

        with torch.no_grad():
            logits = _import(directory)(INPUT).logits
        assert torch.isfinite(logits).all()


def test_undeclared_online_rotation_layers_falls_back_to_template() -> None:
    """Without an explicit layer list the importer falls back to the scaling template.

    A hand-crafted native export has no per-architecture template to fall back on, so
    the fallback must fail loudly rather than silently import the model with no
    rotation applied.
    """
    with tempfile.TemporaryDirectory() as directory:
        _write_export(directory, shared=True, declare_online_layers=False)

        with pytest.raises(ValueError, match="scaling_layers"):
            _import(directory)


if __name__ == "__main__":
    test_shared_rotation_import_matches_per_layer()
    test_shared_rotation_import_populates_every_rotated_layer()
    test_shared_rotation_expansion_does_not_copy_per_layer()
    test_removed_shared_input_rotation_flag_still_loads(True)
    test_undeclared_online_rotation_layers_falls_back_to_template()
    print("shared-rotation import: PASS")
