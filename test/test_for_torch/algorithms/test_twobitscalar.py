#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for the TwoBitScalar 2-bit PTQ algorithm.

These run on CPU with a tiny synthetic model (no HF download / no GPU) and
exercise the processor end-to-end: calibration -> AWQ + SRHT rotation ->
4-level quantization -> undo, plus the SRHT/AWQ sidecar dump used by the
downstream factored exports.
"""

import os
from tempfile import TemporaryDirectory

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open

from quark.experimental.torch.twobitscalar.config import TwoBitScalarConfig
from quark.experimental.torch.twobitscalar.twobitscalar import TwoBitScalarProcessor


class TinyMLP(nn.Module):
    """Two Linear layers with power-of-two in_features (so SRHT applies)."""

    def __init__(self, dim: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _make_dataloader(n_batches: int = 4, batch: int = 32, dim: int = 64):
    torch.manual_seed(0)
    return [torch.randn(batch, dim) for _ in range(n_batches)]


def _base_config(**overrides) -> TwoBitScalarConfig:
    cfg = dict(
        bits=2,
        group_size=64,
        act_scale_alpha=0.5,
        enable_incoherence=True,
        incoherence_seed=1234,
        num_hadamard_passes=1,
        use_lloyd_max_levels=True,
        calib_batches=4,
        min_in_features=16,
        verbose=False,
    )
    cfg.update(overrides)
    return TwoBitScalarConfig(**cfg)


def test_twobitscalar_quantizes_and_is_bounded():
    """Default (undo) path: weights change, reconstruction error is bounded."""
    torch.manual_seed(0)
    model = TinyMLP(dim=64, hidden=128)
    w0_fc1 = model.fc1.weight.detach().clone()

    proc = TwoBitScalarProcessor(model, _base_config(), dataloader=_make_dataloader())
    proc.apply()

    # Both Linears (in_features 64 and 128, both >= min_in_features, pow2) processed.
    assert "fc1" in proc._processed and "fc2" in proc._processed

    # Weights changed (quantization applied) ...
    assert not torch.allclose(model.fc1.weight, w0_fc1)

    # ... but 2-bit quant error stays bounded (rotation makes it near-optimal).
    rel = (model.fc1.weight.float() - w0_fc1.float()).pow(2).mean().sqrt() / w0_fc1.float().pow(2).mean().sqrt()
    assert rel.item() < 0.6, f"reconstruction rel-RMS too high: {rel.item():.3f}"


def test_twobitscalar_calibration_only_does_not_modify_weights():
    """do_quantize=False -> calibration only, weights untouched."""
    torch.manual_seed(0)
    model = TinyMLP()
    w0 = {n: p.detach().clone() for n, p in model.named_parameters()}

    proc = TwoBitScalarProcessor(model, _base_config(do_quantize=False), dataloader=_make_dataloader())
    proc.apply()

    for n, p in model.named_parameters():
        assert torch.allclose(p, w0[n]), f"{n} changed despite do_quantize=False"


def test_twobitscalar_sidecar_dump():
    """Sidecar dump emits the full 5-file set (W_q_levels / group_scale / awq_scale /
    srht_perm / srht_signs) per layer, as required by the factored / shared-rotation
    export."""
    torch.manual_seed(0)
    model = TinyMLP(dim=64, hidden=128)

    with TemporaryDirectory() as tmp:
        proc = TwoBitScalarProcessor(model, _base_config(dump_sidecar_dir=tmp), dataloader=_make_dataloader())
        proc.apply()

        for field in ("W_q_levels", "group_scale", "awq_scale", "srht_perm", "srht_signs"):
            path = os.path.join(tmp, f"{field}.safetensors")
            assert os.path.exists(path), f"missing sidecar file {field}"

        # safetensors safe_open exposes .keys() but is not iterable, so .keys() is required.
        with safe_open(os.path.join(tmp, "srht_perm.safetensors"), framework="pt") as f:
            perms = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
        with safe_open(os.path.join(tmp, "srht_signs.safetensors"), framework="pt") as f:
            signs = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
        with safe_open(os.path.join(tmp, "awq_scale.safetensors"), framework="pt") as f:
            awq = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
        with safe_open(os.path.join(tmp, "W_q_levels.safetensors"), framework="pt") as f:
            wlev = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
        with safe_open(os.path.join(tmp, "group_scale.safetensors"), framework="pt") as f:
            gscale = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118

        assert set(perms) == set(proc._processed)
        assert set(wlev) == set(proc._processed)
        assert set(gscale) == set(proc._processed)
        gsz = int(proc.config.group_size)
        for name in proc._processed:
            lin = getattr(model, name)
            in_dim, out_dim = lin.in_features, lin.out_features
            # perm is a valid permutation of [0, in_dim)
            assert perms[name].numel() == in_dim
            assert torch.equal(perms[name].sort().values, torch.arange(in_dim, dtype=perms[name].dtype))
            # signs are +/- 1
            assert torch.all((signs[name].abs() - 1.0).abs() < 1e-6)
            # awq scale is positive, length in_dim
            assert awq[name].numel() == in_dim
            assert torch.all(awq[name] > 0)

            # --- rotated-domain weight-side sidecar (needed by factored export) ---
            num_g = (in_dim + gsz - 1) // gsz
            assert wlev[name].shape == (out_dim, in_dim)
            assert gscale[name].shape == (out_dim, num_g)
            assert torch.all(gscale[name] > 0)
            # normalized levels: at most 4 distinct (rounded; fp division jitters the
            # inner level across a few ULPs), within [-1, 1]
            assert wlev[name].abs().max() <= 1.0 + 1e-6
            assert torch.unique(torch.round(wlev[name], decimals=4)).numel() <= 4
            # capture invariant: per-(row, group) max|level| == 1 (group_scale = amax)
            for gi in range(num_g):
                s, e = gi * gsz, min((gi + 1) * gsz, in_dim)
                grp_max = wlev[name][:, s:e].abs().amax(dim=1)
                assert torch.all((grp_max - 1.0).abs() < 1e-5)


def test_twobitscalar_sidecar_disabled_for_multipass():
    """Multi-pass Hadamard has >1 perm/sign per layer -> sidecar is disabled (guarded)."""
    torch.manual_seed(0)
    model = TinyMLP()
    with TemporaryDirectory() as tmp:
        proc = TwoBitScalarProcessor(
            model, _base_config(dump_sidecar_dir=tmp, num_hadamard_passes=2), dataloader=_make_dataloader()
        )
        proc.apply()
        # Guarded off: no sidecar files written.
        assert not os.path.exists(os.path.join(tmp, "srht_perm.safetensors"))


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="default"),
        pytest.param({"enable_incoherence": False}, id="no_incoherence"),
        pytest.param({"use_lloyd_max_levels": False}, id="linear_levels"),
        pytest.param({"num_hadamard_passes": 2}, id="multipass_hadamard"),
        pytest.param({"enable_output_rotation": True}, id="output_rotation"),
        pytest.param({"sub_group_size": 16}, id="sub_groups"),
        pytest.param({"fine_scale_grid": True}, id="fine_scale_grid"),
        pytest.param({"adaptive_grid_levels": True}, id="adaptive_grid"),
        pytest.param({"use_hessian_weighted_scale": True}, id="hessian_weighted_scale"),
        pytest.param({"enable_blockwise_hessian": True}, id="blockwise_hessian"),
        pytest.param({"enable_blockwise_hessian": True, "hess_every_n_groups": 2}, id="blockwise_every_2"),
        pytest.param({"enable_blockwise_hessian": True, "group_size": 32}, id="blockwise_g32_damp"),
        pytest.param(
            {"enable_blockwise_hessian": True, "group_size": 8, "hess_min_group_size": 4}, id="blockwise_g8_damp"
        ),
        pytest.param({"debug_dump_per_layer": True, "verbose": True}, id="verbose_debug"),
    ],
)
def test_twobitscalar_config_variants(overrides):
    """Exercise the processor across its config branches (GPTQ, adaptive, blockwise
    Hessian, sub-groups, output rotation, store-quantized, ...): each variant must
    process both Linears and keep the 2-bit reconstruction error bounded."""
    torch.manual_seed(0)
    model = TinyMLP(dim=64, hidden=128)
    w0 = model.fc1.weight.detach().clone()

    proc = TwoBitScalarProcessor(model, _base_config(**overrides), dataloader=_make_dataloader())
    proc.apply()

    assert "fc1" in proc._processed and "fc2" in proc._processed
    assert not torch.allclose(model.fc1.weight, w0)
    assert torch.isfinite(model.fc1.weight).all()
    rel = (model.fc1.weight.float() - w0.float()).pow(2).mean().sqrt() / w0.float().pow(2).mean().sqrt()
    assert rel.item() < 1.0, f"reconstruction rel-RMS too high ({overrides}): {rel.item():.3f}"


class _MockQuantizer:
    """Minimal stand-in for a Quark fake-quantizer (disable_fake_quant/observer)."""

    def __init__(self) -> None:
        self.fake_quant_enabled = True
        self.observer_enabled = True

    def disable_fake_quant(self) -> None:
        self.fake_quant_enabled = False

    def disable_observer(self) -> None:
        self.observer_enabled = False


def test_twobitscalar_disables_framework_quantizers_and_verbose_excluded():
    """Processed layers get their framework quantizers disabled; verbose logs the
    excluded layers (covers _disable_weight_quantizers + verbose-excluded path)."""
    torch.manual_seed(0)
    model = TinyMLP(dim=64, hidden=128)
    for lin in (model.fc1, model.fc2):
        lin._weight_quantizer = _MockQuantizer()
        lin._input_quantizer = _MockQuantizer()
        lin._output_quantizer = _MockQuantizer()

    proc = TwoBitScalarProcessor(
        model, _base_config(verbose=True, exclude_layers=["fc2"]), dataloader=_make_dataloader()
    )
    proc.apply()

    assert "fc1" in proc._processed and "fc2" in proc._skipped_excluded
    # fc1 (processed) -> quantizers disabled; fc2 (excluded) -> untouched
    assert model.fc1._weight_quantizer.fake_quant_enabled is False
    assert model.fc1._input_quantizer.observer_enabled is False
    assert model.fc2._weight_quantizer.fake_quant_enabled is True


def test_twobitscalar_calibration_batch_formats():
    """Calibration accepts tensor, dict, and list-of-dict batches (covers _forward_batch)."""
    torch.manual_seed(0)
    model = TinyMLP(dim=64, hidden=128)
    # mix: a plain tensor, a dict {x: tensor}, and a list of dicts
    dl = [
        torch.randn(8, 64),
        {"x": torch.randn(8, 64)},
        [{"x": torch.randn(8, 64)}, {"x": torch.randn(8, 64)}],
    ]
    proc = TwoBitScalarProcessor(model, _base_config(calib_batches=3), dataloader=dl)
    proc.apply()
    assert "fc1" in proc._processed
    assert torch.isfinite(model.fc1.weight).all()


def test_twobitscalar_include_exclude_filters():
    """include_layers / exclude_layers wildcards select the right Linears."""
    torch.manual_seed(0)
    model = TinyMLP()
    proc = TwoBitScalarProcessor(model, _base_config(exclude_layers=["fc2"]), dataloader=_make_dataloader())
    proc.apply()
    assert "fc1" in proc._processed and "fc2" not in proc._processed

    model2 = TinyMLP()
    proc2 = TwoBitScalarProcessor(model2, _base_config(include_layers=["fc1"]), dataloader=_make_dataloader())
    proc2.apply()
    assert proc2._processed == ["fc1"] or ("fc1" in proc2._processed and "fc2" not in proc2._processed)


def test_snap_to_2bit_standalone():
    """Standalone snap_to_2bit: each row collapses to exactly 4 unique level values."""
    from quark.experimental.torch.twobitscalar.twobitscalar import snap_to_2bit

    torch.manual_seed(0)
    for use_lloyd in (True, False):
        m = TinyMLP(dim=64, hidden=128)
        stats = snap_to_2bit(m, group_size=64, use_lloyd_max=use_lloyd)
        assert "fc1" in stats and "fc2" in stats
        # each per-group set of weights takes at most 4 distinct magnitudes/levels
        w = m.fc1.weight.detach()
        for gi in range(w.shape[1] // 64):
            grp = w[:, gi * 64 : (gi + 1) * 64]
            for r in range(grp.shape[0]):
                assert torch.unique(grp[r]).numel() <= 4


class TinyMLPOdd(nn.Module):
    """Non-power-of-two in_features to exercise the 'skip SRHT rotation' branch,
    plus a tiny layer below min_in_features to exercise the skip filter."""

    def __init__(self) -> None:
        super().__init__()
        self.small = nn.Linear(8, 48, bias=False)  # in=8 < default min_in_features(16) path
        self.odd = nn.Linear(48, 48, bias=False)  # in=48 not power-of-two
        self.tail = nn.Linear(48, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(torch.relu(self.odd(torch.relu(self.small(x)))))


def test_twobitscalar_nonpow2_and_small_layer_paths():
    """Non-pow2 in_features -> rotation-require-pow2 skip branch; small in_features
    -> min_in_features filter branch. Both should still finish without error."""
    torch.manual_seed(0)
    model = TinyMLPOdd()
    dl = [torch.randn(16, 8) for _ in range(4)]
    cfg = _base_config(group_size=16, min_in_features=16)
    proc = TwoBitScalarProcessor(model, cfg, dataloader=dl)
    proc.apply()
    # 'small' (in=8 < 16) is filtered out; 'odd'/'tail' (in=48) processed.
    assert "small" not in proc._processed
    assert torch.isfinite(model.odd.weight).all()


def test_twobitscalar_blockwise_hessian_ineligible_group():
    """hess_min_group_size larger than the group width -> Hessian marked ineligible,
    falls back to plain sub-group quant (covers the not-eligible branch + counter)."""
    torch.manual_seed(0)
    model = TinyMLP(dim=64, hidden=128)
    cfg = _base_config(enable_blockwise_hessian=True, hess_min_group_size=1024, group_size=64)
    proc = TwoBitScalarProcessor(model, cfg, dataloader=_make_dataloader())
    proc.apply()
    assert proc._hess_skipped > 0
    assert torch.isfinite(model.fc1.weight).all()


def test_snap_to_2bit_exclude_patterns():
    """snap_to_2bit honors exclude_patterns (fc2 left untouched)."""
    from quark.experimental.torch.twobitscalar.twobitscalar import snap_to_2bit

    torch.manual_seed(0)
    model = TinyMLP(dim=64, hidden=128)
    w2 = model.fc2.weight.detach().clone()
    stats = snap_to_2bit(model, group_size=64, exclude_patterns=["fc2"])
    assert "fc1" in stats and "fc2" not in stats
    assert torch.allclose(model.fc2.weight, w2)


def test_snap_to_2bit_awq_standalone():
    """Standalone snap_to_2bit_awq: replaces Linears with AWQScaledLinear, forward runs."""
    from quark.experimental.torch.twobitscalar.twobitscalar import AWQScaledLinear, snap_to_2bit_awq

    torch.manual_seed(0)
    model = TinyMLP(dim=64, hidden=128)
    stats = snap_to_2bit_awq(model, dataloader=_make_dataloader(), group_size=64, calib_batches=4)
    assert "fc1" in stats and "fc2" in stats
    assert isinstance(model.fc1, AWQScaledLinear)
    out = model(torch.randn(2, 64))
    assert out.shape == (2, 64) and torch.isfinite(out).all()


class MixMLP(nn.Module):
    """Layers exercising snap edge branches: remainder group (96 % 64 != 0),
    a sub-group-size layer (32 < group 64 -> skipped)."""

    def __init__(self) -> None:
        super().__init__()
        self.rem = nn.Linear(96, 64, bias=False)  # in=96, group=64 -> 1 full group + remainder 32
        self.mid = nn.Linear(64, 32, bias=False)
        self.small = nn.Linear(32, 96, bias=False)  # in=32 < group 64 -> skipped by snap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.small(torch.relu(self.mid(torch.relu(self.rem(x)))))


def test_snap_to_2bit_awq_edge_shapes():
    """Cover snap_to_2bit_awq edge branches: 3D input reshape, >max_rows cap, dict
    batches, remainder group, and the sub-group-size skip."""
    from quark.experimental.torch.twobitscalar.twobitscalar import snap_to_2bit_awq

    torch.manual_seed(0)
    model = MixMLP()
    dl = [
        torch.randn(2, 5, 96),  # 3D -> hook reshapes to 2D
        torch.randn(300, 96),  # > max_rows_per_batch(256) -> row cap
        {"x": torch.randn(8, 96)},  # dict batch path
    ]
    stats = snap_to_2bit_awq(model, dataloader=dl, group_size=64, calib_batches=3, max_rows_per_batch=256)
    assert "rem" in stats  # remainder-group layer processed
    assert "small" not in stats  # in=32 < group 64 -> skipped
    assert torch.isfinite(model.rem.weight).all()


def test_twobitscalar_calibration_3d_and_large_batch():
    """Processor calibration handles 3D inputs and batches larger than max_rows."""
    torch.manual_seed(0)
    model = TinyMLP(dim=64, hidden=128)
    # batch1 (240 rows) exceeds max_samples_per_layer(100) -> first-fill cap;
    # batch2 (60 rows) accumulates but is capped to the remaining need;
    # 3D input exercises the reshape path; >max_rows_per_batch cap also hit.
    dl = [torch.randn(2, 120, 64), torch.randn(60, 64)]
    cfg = _base_config(calib_batches=2, max_rows_per_batch=80, max_samples_per_layer=100)
    proc = TwoBitScalarProcessor(model, cfg, dataloader=dl)
    proc.apply()
    assert "fc1" in proc._processed
    assert torch.isfinite(model.fc1.weight).all()


def test_twobitscalar_config_from_file():
    """The config-driven CLI path: --quant_algo_config_file twobitscalar <json>
    resolves to a TwoBitScalarConfig via load_quant_algo_config_from_file."""
    import json

    from quark.torch.quantization.config.config import load_quant_algo_config_from_file

    cfg_dict = {
        "name": "twobitscalar",
        "bits": 2,
        "group_size": 64,
        "act_scale_alpha": 0.5,
        "enable_incoherence": True,
        "use_lloyd_max_levels": True,
        "exclude_layers": ["*embed_tokens*", "*lm_head*"],
    }
    with TemporaryDirectory() as d:
        path = os.path.join(d, "twobitscalar_config.json")
        with open(path, "w") as f:
            json.dump(cfg_dict, f)
        cfg = load_quant_algo_config_from_file(path)

    assert isinstance(cfg, TwoBitScalarConfig)
    assert cfg.bits == 2
    assert cfg.group_size == 64
    assert cfg.act_scale_alpha == 0.5
    assert cfg.enable_incoherence is True
    assert cfg.use_lloyd_max_levels is True
    assert cfg.exclude_layers == ["*embed_tokens*", "*lm_head*"]


def test_twobitscalar_builtin_algo_config():
    """The built-in per-model-type default: --quant_algo twobitscalar works without
    a config file, resolved via algo_configs.get_algo_config."""
    from quark.torch.quantization.config.algo_configs import get_algo_config, get_supported_algorithm_types

    assert "twobitscalar" in get_supported_algorithm_types()

    cfg = get_algo_config("twobitscalar", "phi3")
    assert isinstance(cfg, TwoBitScalarConfig)
    assert cfg.bits == 2
    assert cfg.group_size == 64
    assert cfg.enable_incoherence is True
    assert cfg.exclude_layers == ["*embed_tokens*", "*lm_head*"]

    # Unknown model type -> no built-in default (None), not an error.
    assert get_algo_config("twobitscalar", "some_unknown_arch") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
