# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""CPU-only smoke tests for Hadamard rotation in the file-to-file quantization flow.

These tests build a tiny llama-like sharded checkpoint on disk and run
``ModelQuantizer.direct_quantize_checkpoint`` end to end with ``device="cpu"`` and a
weight-only integer scheme (no FP8/Triton, so no GPU is required).
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from quark.torch import ModelQuantizer
from quark.torch.algorithm.rotation.hadamard import _get_hadamard_K
from quark.torch.algorithm.rotation.rotation import RotationProcessor
from quark.torch.algorithm.rotation.rotation_utils import HadamardTransform, InputRotationWrapperHadamard
from quark.torch.quantization.config.config import (
    Int8PerTensorSpec,
    QConfig,
    QLayerConfig,
    RotationConfig,
)
from quark.torch.quantization.file2file_rotation import (
    RotationPlan,
    _build_input_rotation_int8,
    _rotate_input_channels_online,
    _text_config_view,
    apply_rotation_to_tensor,
    build_rotation_plan,
)
from quark.torch.quantization.file2file_utils import match_modules

HIDDEN = 16
INTERMEDIATE = 32
NUM_LAYERS = 2
NUM_HEADS = 4


def _online_r1_scaling_layers() -> dict:
    """Online-R1 scaling layers targeting q_proj/k_proj (NOT excluded)."""
    targets = [
        "model.layers.layer_id.self_attn.q_proj",
        "model.layers.layer_id.self_attn.k_proj",
    ]
    return {
        "first_layer": [{"target_modules": targets}],
        "middle_layers": [{"target_modules": targets}],
        "last_layer": [{"target_modules": []}],
    }


def _online_r1_rotation_config() -> RotationConfig:
    return RotationConfig(
        scaling_layers=_online_r1_scaling_layers(),
        r1=True,
        r2=False,
        r3=False,
        r4=False,
        online_r1_rotation=True,
        trainable=False,
        model_decoder_layers="model.layers",
    )


def _build_tiny_checkpoint(model_dir: Path) -> dict[str, torch.Tensor]:
    """Write a tiny llama-like sharded checkpoint to ``model_dir``. Returns the tensors."""
    torch.manual_seed(0)
    tensors: dict[str, torch.Tensor] = {}
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}"
        tensors[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(HIDDEN, HIDDEN)
        tensors[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(HIDDEN, HIDDEN)
        tensors[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(HIDDEN, HIDDEN)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(HIDDEN, HIDDEN)
        tensors[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(INTERMEDIATE, HIDDEN)
        tensors[f"{prefix}.mlp.up_proj.weight"] = torch.randn(INTERMEDIATE, HIDDEN)
        tensors[f"{prefix}.mlp.down_proj.weight"] = torch.randn(HIDDEN, INTERMEDIATE)
        # RMSNorm weights (1-D) — must be copied through untouched.
        tensors[f"{prefix}.input_layernorm.weight"] = torch.ones(HIDDEN)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(HIDDEN)
    tensors["model.embed_tokens.weight"] = torch.randn(32, HIDDEN)
    tensors["model.norm.weight"] = torch.ones(HIDDEN)
    tensors["lm_head.weight"] = torch.randn(32, HIDDEN)

    model_dir.mkdir(parents=True, exist_ok=True)
    shard_name = "model.safetensors"
    save_file(tensors, str(model_dir / shard_name))

    weight_map = dict.fromkeys(tensors, shard_name)
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)

    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_HEADS,
        "torch_dtype": "float16",
        "vocab_size": 32,
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    return tensors


TINY_LLAMA_ID = "amd-quark/tiny-llama-fast-tokenizer"


def _materialize_hub_checkpoint(model_dir: Path) -> dict[str, object]:
    """Write a real tiny llama to ``model_dir`` in the layout the file-to-file flow reads.

    The hub repo ships only ``pytorch_model.bin``, and ``save_pretrained`` omits the shard
    index for a single unsharded file, so both the safetensors conversion and the index are
    written here. Returns the model config (dims differ from the synthetic fixture's).
    """
    model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA_ID)
    model.save_pretrained(model_dir, safe_serialization=True)

    tensors = load_file(str(model_dir / "model.safetensors"))
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": dict.fromkeys(tensors, "model.safetensors")}, f)

    with open(model_dir / "config.json") as f:
        config: dict[str, object] = json.load(f)
    return config


def _weight_only_int8_config(algo_config: list | None) -> QConfig:
    weight_spec = Int8PerTensorSpec(
        observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
    ).to_quantization_spec()
    return QConfig(
        global_quant_config=QLayerConfig(weight=weight_spec),
        exclude=["lm_head"],
        algo_config=algo_config,
    )


def test_file2file_online_r1_end_to_end(tmp_path: Path) -> None:
    """Rotation is applied, buffers are emitted, config is persisted, and the transform
    is numerically identity-preserving (rotated_W @ Hadamard(x) == W @ x)."""
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "quantized"
    original = _build_tiny_checkpoint(model_dir)

    quant_config = _weight_only_int8_config(algo_config=[_online_r1_rotation_config()])
    quantizer = ModelQuantizer(quant_config)
    quantizer.direct_quantize_checkpoint(
        pretrained_model_path=str(model_dir),
        save_path=str(out_dir),
        device="cpu",
    )

    out_tensors = load_file(str(out_dir / "model.safetensors"))

    target = "model.layers.0.self_attn.q_proj.weight"
    non_target = "model.layers.0.self_attn.v_proj.weight"

    # (1) rotation buffer present for online target, int8, ±1, hidden x hidden.
    buf_name = "model.layers.0.self_attn.q_proj.input_rotation"
    assert buf_name in out_tensors
    buf = out_tensors[buf_name]
    assert buf.dtype == torch.int8
    assert buf.shape == (HIDDEN, HIDDEN)
    assert set(torch.unique(buf).tolist()) <= {-1, 1}

    # A non-target attention weight gets no rotation buffer.
    assert "model.layers.0.self_attn.v_proj.input_rotation" not in out_tensors

    # (2) exported config.json carries the rotation algo_config (drives inference reload).
    with open(out_dir / "config.json") as f:
        exported = json.load(f)
    algo = exported["quantization_config"]["algo_config"]
    rot = next(a for a in algo if a["name"] == "rotation")
    assert rot["r1"] is True
    assert rot["online_r1_rotation"] is True
    assert rot["trainable"] is False

    # (2b) online_config.online_rotation_layers is populated (regression guard). Inference
    # (sglang / QParamsLinearWithRotation) reads this list to decide which layers get the
    # activation-side x @ H rotation. If it is null/empty, rotated weights run uncompensated
    # and outputs are silently wrong. It must list the online-target modules (no .weight
    # suffix) that also have an emitted input_rotation buffer.
    online_layers = rot["online_config"]["online_rotation_layers"]
    assert online_layers, "online_rotation_layers must be populated for online R1"
    assert "model.layers.0.self_attn.q_proj" in online_layers
    assert "model.layers.0.self_attn.v_proj" not in online_layers
    # Every listed module must have a matching input_rotation buffer in the shard.
    for module_name in online_layers:
        assert f"{module_name}.input_rotation" in out_tensors

    # (3) numeric identity: reconstruct the activation transform the way inference does,
    # and confirm the rotated weight (dequantized) reproduces the original layer output.
    scale = out_tensors[target + "_scale"]
    q_weight = out_tensors[target].to(torch.float32)
    rotated_dequant = q_weight * scale  # per-tensor symmetric int8 dequant

    hadamard_k, k = _get_hadamard_K(HIDDEN)
    transform = HadamardTransform(rotation_size=HIDDEN, use_matmul_hadU=True, hadamard_K=hadamard_k, K=k)

    x = torch.randn(4, HIDDEN)
    original_out = x @ original[target].T
    rotated_out = transform(x) @ rotated_dequant.T
    # int8 quant introduces error; the rotation itself is identity-preserving. Use a
    # tolerance that comfortably covers int8 round-off but would fail if rotation broke.
    assert (original_out - rotated_out).abs().max().item() < 0.5

    # The non-target weight is only quantized, not rotated: dequant ~= original.
    nt_scale = out_tensors[non_target + "_scale"]
    nt_dequant = out_tensors[non_target].to(torch.float32) * nt_scale
    assert (nt_dequant - original[non_target]).abs().max().item() < 0.1


def test_file2file_rotation_trainable_raises(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _build_tiny_checkpoint(model_dir)

    trainable_cfg = RotationConfig(
        scaling_layers=_online_r1_scaling_layers(),
        r1=True,
        online_r1_rotation=True,
        trainable=True,
        model_decoder_layers="model.layers",
    )
    quant_config = _weight_only_int8_config(algo_config=[trainable_cfg])
    quantizer = ModelQuantizer(quant_config)
    with pytest.raises(NotImplementedError, match="trainable"):
        quantizer.direct_quantize_checkpoint(
            pretrained_model_path=str(model_dir),
            save_path=str(tmp_path / "out"),
            device="cpu",
        )


def test_file2file_rotation_excluded_target_raises(tmp_path: Path) -> None:
    """An online rotation target that also matches an exclude pattern must hard-error."""
    model_dir = tmp_path / "model"
    _build_tiny_checkpoint(model_dir)

    quant_config = _weight_only_int8_config(algo_config=[_online_r1_rotation_config()])
    # Exclude the very layers the rotation targets.
    quant_config.exclude = ["lm_head", "*.self_attn.*"]
    quantizer = ModelQuantizer(quant_config)
    with pytest.raises(ValueError, match="exclude"):
        quantizer.direct_quantize_checkpoint(
            pretrained_model_path=str(model_dir),
            save_path=str(tmp_path / "out"),
            device="cpu",
        )


def test_file2file_rotation_r3_raises(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _build_tiny_checkpoint(model_dir)

    r3_cfg = RotationConfig(
        scaling_layers=_online_r1_scaling_layers(),
        r1=True,
        r3=True,
        online_r1_rotation=True,
        trainable=False,
        model_decoder_layers="model.layers",
    )
    quant_config = _weight_only_int8_config(algo_config=[r3_cfg])
    quantizer = ModelQuantizer(quant_config)
    with pytest.raises(NotImplementedError, match="r3"):
        quantizer.direct_quantize_checkpoint(
            pretrained_model_path=str(model_dir),
            save_path=str(tmp_path / "out"),
            device="cpu",
        )


def test_build_rotation_plan_none_without_rotation_config() -> None:
    quant_config = _weight_only_int8_config(algo_config=None)
    plan = build_rotation_plan(quant_config, {"hidden_size": HIDDEN}, {"model.layers.0.self_attn.q_proj.weight"})
    assert plan is None


def test_apply_rotation_to_tensor_r2_is_output_and_input_channel() -> None:
    """R2 rotates v_proj out-channels and o_proj in-channels; verify identity across the
    v_proj -> o_proj pair (o_proj undoes v_proj's rotation on the head_dim axis)."""
    head_dim = HIDDEN // NUM_HEADS
    r2_cfg = RotationConfig(
        scaling_layers={
            "first_layer": [{"target_modules": []}],
            "middle_layers": [{"target_modules": []}],
            "last_layer": [{"target_modules": []}],
        },
        r1=False,
        r2=True,
        r3=False,
        r4=False,
        online_r1_rotation=None,
        trainable=False,
        model_decoder_layers="model.layers",
    )
    quant_config = _weight_only_int8_config(algo_config=[r2_cfg])
    names = {
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    }
    plan = build_rotation_plan(quant_config, {"hidden_size": HIDDEN, "num_attention_heads": NUM_HEADS}, names)
    assert plan is not None
    assert plan.r2_out == {"model.layers.0.self_attn.v_proj.weight": head_dim}
    assert plan.r2_in == {"model.layers.0.self_attn.o_proj.weight": head_dim}
    # R2 is a pure weight edit with no online targets, so no online_config is attached.
    # Setting one would trip RotationConfig.__post_init__'s "online_config has no effect".
    assert r2_cfg.online_config is None

    torch.manual_seed(1)
    v_w = torch.randn(HIDDEN, HIDDEN)
    o_w = torch.randn(HIDDEN, HIDDEN)
    v_rot, v_extra = apply_rotation_to_tensor("model.layers.0.self_attn.v_proj.weight", v_w, plan)
    o_rot, o_extra = apply_rotation_to_tensor("model.layers.0.self_attn.o_proj.weight", o_w, plan)
    assert v_extra == {} and o_extra == {}

    # v output feeds directly into o input; rotating v's out by R and o's in by R leaves
    # the composed mapping (o_w @ v_w) unchanged.
    x = torch.randn(4, HIDDEN)
    original = (x @ v_w.T) @ o_w.T
    rotated = (x @ v_rot.T) @ o_rot.T
    assert (original - rotated).abs().max().item() < 1e-4


def _r2_only_config() -> RotationConfig:
    """R2-only RotationConfig (empty online scaling layers)."""
    return RotationConfig(
        scaling_layers={
            "first_layer": [{"target_modules": []}],
            "middle_layers": [{"target_modules": []}],
            "last_layer": [{"target_modules": []}],
        },
        r1=False,
        r2=True,
        r3=False,
        r4=False,
        online_r1_rotation=None,
        trainable=False,
        model_decoder_layers="model.layers",
    )


def test_r2_out_bias_hard_errors() -> None:
    """A biased v_proj must hard-error: R2 rotates v_proj's output channels, which changes
    the output basis, so the bias must be rotated too — but the per-tensor file-to-file flow
    can't fold it. Serving a rotated weight with an un-rotated bias silently corrupts output,
    so build_rotation_plan raises instead."""
    quant_config = _weight_only_int8_config(algo_config=[_r2_only_config()])
    weight_names = {
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    }
    bias_names = {"model.layers.0.self_attn.v_proj.bias"}
    cfg = {"hidden_size": HIDDEN, "num_attention_heads": NUM_HEADS}
    with pytest.raises(NotImplementedError, match="requires rotating"):
        build_rotation_plan(quant_config, cfg, weight_names, bias_names)


def test_r2_in_bias_is_allowed() -> None:
    """An o_proj bias must NOT trigger the guard: R2's o_proj rotation is input-channel, which
    does not change the output basis, so the bias passes through correctly unrotated."""
    quant_config = _weight_only_int8_config(algo_config=[_r2_only_config()])
    weight_names = {
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    }
    bias_names = {"model.layers.0.self_attn.o_proj.bias"}  # only o_proj has a bias
    cfg = {"hidden_size": HIDDEN, "num_attention_heads": NUM_HEADS}
    plan = build_rotation_plan(quant_config, cfg, weight_names, bias_names)
    assert plan is not None
    assert plan.r2_out == {"model.layers.0.self_attn.v_proj.weight": HIDDEN // NUM_HEADS}
    assert plan.r2_in == {"model.layers.0.self_attn.o_proj.weight": HIDDEN // NUM_HEADS}


def _r1_and_r2_rotation_config() -> RotationConfig:
    """Online R1 over the attention projections (v_proj included) with R2 also enabled."""
    targets = [f"model.layers.layer_id.self_attn.{p}_proj" for p in ("q", "k", "v")]
    return RotationConfig(
        scaling_layers={
            "first_layer": [{"target_modules": targets}],
            "middle_layers": [{"target_modules": targets}],
            "last_layer": [{"target_modules": []}],
        },
        r1=True,
        r2=True,
        r3=False,
        r4=False,
        online_r1_rotation=True,
        trainable=False,
        model_decoder_layers="model.layers",
    )


def test_v_proj_receives_both_r1_and_r2_rotations() -> None:
    """A tensor targeted by two rotations must receive both, not just the first match.

    ``v_proj`` is the case that matters: online R1 rotates its input channels while R2
    rotates its output channels and compensates on ``o_proj``'s input channels. Applying
    only R1 leaves ``o_proj`` rotated against an unrotated ``v_proj``, which silently
    corrupts the attention value path — no error, wrong numbers.
    """
    quant_config = _weight_only_int8_config(algo_config=[_r1_and_r2_rotation_config()])
    v_name = "model.layers.0.self_attn.v_proj.weight"
    o_name = "model.layers.0.self_attn.o_proj.weight"
    weight_names = {f"model.layers.0.self_attn.{p}_proj.weight" for p in ("q", "k", "v", "o")}

    plan = build_rotation_plan(quant_config, {"hidden_size": HIDDEN, "num_attention_heads": NUM_HEADS}, weight_names)
    assert plan is not None
    # Precondition: the planner really does target v_proj with both rotations.
    assert v_name in plan.online_in and v_name in plan.r2_out

    torch.manual_seed(0)
    w_v, w_o = torch.randn(HIDDEN, HIDDEN), torch.randn(HIDDEN, HIDDEN)
    rotated_v, _ = apply_rotation_to_tensor(v_name, w_v, plan)
    rotated_o, _ = apply_rotation_to_tensor(o_name, w_o, plan)

    # R1 alone would stop here; the R2 output rotation must have been applied on top.
    r1_only, _ = _rotate_input_channels_online(w_v, plan.online_in[v_name], plan)
    assert not torch.allclose(rotated_v, r1_only, atol=1e-6)

    # The R2 pair must cancel across o_proj @ v_proj, leaving only R1's effect on v_proj.
    # If v_proj were missing its R2 half, this deviates by orders of magnitude.
    assert torch.allclose(rotated_o @ rotated_v, w_o @ r1_only, atol=1e-4)


def test_r2_divisibility_guard_raises_on_wrong_head_dim() -> None:
    """If the derived/declared head_dim does not divide the rotated tensor dimension,
    apply_rotation_to_tensor raises a clear, module-named error rather than deferring to
    rotate_with_size's generic 'incompatible' message."""
    # head_dim = hidden // heads = 16 // 4 = 4 divides 16 fine, so force a bad size by
    # crafting a plan whose r2_out rotation_size does not divide the tensor's out dim.
    plan = RotationPlan()
    bad_size = 5  # 16 % 5 != 0
    plan.r2_out["model.layers.0.self_attn.v_proj.weight"] = bad_size
    v_w = torch.randn(HIDDEN, HIDDEN)
    with pytest.raises(ValueError, match="does not divide"):
        apply_rotation_to_tensor("model.layers.0.self_attn.v_proj.weight", v_w, plan)


def test_build_rotation_plan_next_modules_fallback() -> None:
    """A scaling entry with only ``next_modules`` (no ``target_modules``) must resolve the
    same online targets the reload path would, so ``input_rotation`` buffers get written.

    This mirrors RotationProcessor.get_scaling_layers / get_online_rotation_layers: without
    the fallback, file-to-file would write no buffer while reload still expects one.
    """
    next_modules = [
        "model.layers.layer_id.self_attn.q_proj",
        "model.layers.layer_id.self_attn.k_proj",
    ]
    scaling_layers = {
        "first_layer": [{"norm_module": "model.layers.layer_id.input_layernorm", "next_modules": next_modules}],
        "middle_layers": [{"norm_module": "model.layers.layer_id.input_layernorm", "next_modules": next_modules}],
        "last_layer": [{"target_modules": []}],
    }
    rot_cfg = RotationConfig(
        scaling_layers=scaling_layers,
        r1=True,
        online_r1_rotation=True,
        trainable=False,
        model_decoder_layers="model.layers",
    )
    quant_config = _weight_only_int8_config(algo_config=[rot_cfg])
    names = {f"model.layers.{i}.self_attn.{p}.weight" for i in range(NUM_LAYERS) for p in ["q_proj", "k_proj"]}
    plan = build_rotation_plan(quant_config, {"hidden_size": HIDDEN}, names)
    assert plan is not None
    # NUM_LAYERS layers x {q_proj, k_proj} online targets, all via the next_modules fallback.
    assert set(plan.online_in) == {
        f"model.layers.{i}.self_attn.{p}.weight" for i in range(NUM_LAYERS) for p in ["q_proj", "k_proj"]
    }
    assert set(plan.online_in.values()) == {HIDDEN}

    # Applying to a target actually emits the input_rotation buffer.
    _, extra = apply_rotation_to_tensor("model.layers.0.self_attn.q_proj.weight", torch.randn(HIDDEN, HIDDEN), plan)
    assert "model.layers.0.self_attn.q_proj.input_rotation" in extra


def test_build_rotation_plan_empty_plan_raises() -> None:
    """A rotation config that resolves to zero targets must hard-error, not silently return
    None. Otherwise config.json would record rotation while no buffers were written, and
    reload would fail looking for a missing input_rotation."""
    # model_decoder_layers points at a prefix absent from the checkpoint tensor names.
    rot_cfg = RotationConfig(
        scaling_layers=_online_r1_scaling_layers(),
        r1=True,
        online_r1_rotation=True,
        trainable=False,
        model_decoder_layers="does.not.exist",
    )
    quant_config = _weight_only_int8_config(algo_config=[rot_cfg])
    names = {"model.layers.0.self_attn.q_proj.weight"}
    with pytest.raises(ValueError, match="zero target layers"):
        build_rotation_plan(quant_config, {"hidden_size": HIDDEN}, names)


def test_hadamard_cache_reused_across_targets_with_distinct_buffers() -> None:
    """The Hadamard matrix / int8 buffer are cached per rotation_size (built once), yet each
    target still gets an independent buffer (distinct storage, equal values) so save_file
    does not reject shared-storage tensors."""
    rot_cfg = _online_r1_rotation_config()
    quant_config = _weight_only_int8_config(algo_config=[rot_cfg])
    names = {f"model.layers.{i}.self_attn.{p}.weight" for i in range(NUM_LAYERS) for p in ["q_proj", "k_proj"]}
    plan = build_rotation_plan(quant_config, {"hidden_size": HIDDEN}, names)
    assert plan is not None

    torch.manual_seed(2)
    _, e1 = apply_rotation_to_tensor("model.layers.0.self_attn.q_proj.weight", torch.randn(HIDDEN, HIDDEN), plan)
    _, e2 = apply_rotation_to_tensor("model.layers.1.self_attn.k_proj.weight", torch.randn(HIDDEN, HIDDEN), plan)
    b1 = e1["model.layers.0.self_attn.q_proj.input_rotation"]
    b2 = e2["model.layers.1.self_attn.k_proj.input_rotation"]

    # Cache built exactly one entry (all targets share rotation_size == HIDDEN).
    assert set(plan._hadamard_cache) == {HIDDEN}
    assert set(plan._input_rotation_cache) == {HIDDEN}
    # Buffers are byte-identical in value but must not share storage.
    assert torch.equal(b1, b2)
    assert b1.data_ptr() != b2.data_ptr()
    assert b1.dtype == torch.int8 and tuple(b1.shape) == (HIDDEN, HIDDEN)


# ---------------------------------------------------------------------------
# _text_config_view coverage (lines 70, 74-76)
# ---------------------------------------------------------------------------


def test_text_config_view_returns_empty_for_none() -> None:
    assert _text_config_view(None) == {}


def test_text_config_view_returns_empty_for_empty_dict() -> None:
    assert _text_config_view({}) == {}


def test_text_config_view_merges_nested_text_config() -> None:
    """Nested text_config values override root, root keys absent from text_config survive."""
    cfg = {"hidden_size": 1024, "vocab_size": 32000, "text_config": {"hidden_size": 2048, "head_dim": 64}}
    result = _text_config_view(cfg)
    assert result["hidden_size"] == 2048
    assert result["head_dim"] == 64
    assert result["vocab_size"] == 32000


def test_text_config_view_ignores_non_dict_text_config() -> None:
    """A non-dict text_config (e.g. a string class name) is not merged."""
    cfg = {"hidden_size": 512, "text_config": "SomeClassName"}
    result = _text_config_view(cfg)
    assert result["hidden_size"] == 512
    assert "text_config" in result


# ---------------------------------------------------------------------------
# _match_modules coverage (line 167)
# ---------------------------------------------------------------------------


def test_match_modules_glob_pattern() -> None:
    names = {
        "model.layers.0.mlp.experts.0.down_proj",
        "model.layers.0.mlp.experts.1.down_proj",
        "model.layers.0.mlp.gate_proj",
    }
    matched = match_modules("model.layers.0.mlp.experts.*.down_proj", names)
    assert matched == [
        "model.layers.0.mlp.experts.0.down_proj",
        "model.layers.0.mlp.experts.1.down_proj",
    ]


def test_match_modules_plain_name_hit_and_miss() -> None:
    names = {"model.layers.0.self_attn.q_proj"}
    assert match_modules("model.layers.0.self_attn.q_proj", names) == ["model.layers.0.self_attn.q_proj"]
    assert match_modules("model.layers.0.self_attn.k_proj", names) == []


# ---------------------------------------------------------------------------
# R4 plan coverage (lines 210, 354-357, 363-368)
# ---------------------------------------------------------------------------


def _r4_rotation_config() -> RotationConfig:
    return RotationConfig(
        scaling_layers={
            "first_layer": [{"target_modules": []}],
            "middle_layers": [{"target_modules": []}],
            "last_layer": [{"target_modules": []}],
        },
        r1=False,
        r2=False,
        r3=False,
        r4=True,
        online_r1_rotation=None,
        trainable=False,
        model_decoder_layers="model.layers",
    )


def test_build_rotation_plan_r4() -> None:
    """R4 targets down_proj's input channels with intermediate_size as rotation_size."""
    r4_cfg = _r4_rotation_config()
    quant_config = _weight_only_int8_config(algo_config=[r4_cfg])
    names = set()
    for layer in range(NUM_LAYERS):
        names.add(f"model.layers.{layer}.mlp.down_proj.weight")
        names.add(f"model.layers.{layer}.mlp.gate_proj.weight")

    plan = build_rotation_plan(
        quant_config,
        {"hidden_size": HIDDEN, "intermediate_size": INTERMEDIATE},
        names,
    )
    assert plan is not None
    assert len(plan.online_in) == NUM_LAYERS
    for layer in range(NUM_LAYERS):
        key = f"model.layers.{layer}.mlp.down_proj.weight"
        assert key in plan.online_in
        assert plan.online_in[key] == INTERMEDIATE


def test_build_rotation_plan_r4_uses_moe_intermediate_size() -> None:
    """R4 prefers moe_intermediate_size over intermediate_size."""
    r4_cfg = _r4_rotation_config()
    quant_config = _weight_only_int8_config(algo_config=[r4_cfg])
    names = {"model.layers.0.mlp.down_proj.weight"}
    plan = build_rotation_plan(
        quant_config,
        {"hidden_size": HIDDEN, "moe_intermediate_size": 64, "intermediate_size": INTERMEDIATE},
        names,
    )
    assert plan is not None
    assert plan.online_in["model.layers.0.mlp.down_proj.weight"] == 64


def test_build_rotation_plan_r4_explicit_rotation_size() -> None:
    """R4 uses RotationConfig.rotation_size when set, ignoring the model config."""
    r4_cfg = _r4_rotation_config()
    r4_cfg.rotation_size = 8
    quant_config = _weight_only_int8_config(algo_config=[r4_cfg])
    names = {"model.layers.0.mlp.down_proj.weight"}
    plan = build_rotation_plan(quant_config, {"hidden_size": HIDDEN}, names)
    assert plan is not None
    assert plan.online_in["model.layers.0.mlp.down_proj.weight"] == 8


def test_build_rotation_plan_r4_missing_size_raises() -> None:
    """R4 with no rotation_size and no intermediate_size in config must error."""
    r4_cfg = _r4_rotation_config()
    quant_config = _weight_only_int8_config(algo_config=[r4_cfg])
    names = {"model.layers.0.mlp.down_proj.weight"}
    with pytest.raises(ValueError, match="R4 rotation size"):
        build_rotation_plan(quant_config, {"hidden_size": HIDDEN}, names)


def test_build_rotation_plan_r4_excluded_target_raises() -> None:
    """R4 target matching an exclude pattern must hard-error."""
    r4_cfg = _r4_rotation_config()
    quant_config = _weight_only_int8_config(algo_config=[r4_cfg])
    quant_config.exclude = ["lm_head", "*.mlp.*"]
    names = {"model.layers.0.mlp.down_proj.weight"}
    with pytest.raises(ValueError, match="exclude"):
        build_rotation_plan(
            quant_config,
            {"hidden_size": HIDDEN, "intermediate_size": INTERMEDIATE},
            names,
        )


def test_file2file_r4_end_to_end(tmp_path: Path) -> None:
    """R4 rotation end-to-end: down_proj gets rotated, input_rotation buffer emitted.

    Runs against a real llama checkpoint rather than the synthetic fixture: every assertion
    here is structural (buffer presence, dtype, shape, +/-1 values), so it does not depend on
    weight magnitudes the way the numeric-equivalence tests do.
    """
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "quantized"
    config = _materialize_hub_checkpoint(model_dir)
    # R4 rotates down_proj's input channels, so the buffer is intermediate_size wide.
    intermediate = config["intermediate_size"]

    r4_cfg = _r4_rotation_config()
    quant_config = _weight_only_int8_config(algo_config=[r4_cfg])
    quantizer = ModelQuantizer(quant_config)
    quantizer.direct_quantize_checkpoint(
        pretrained_model_path=str(model_dir),
        save_path=str(out_dir),
        device="cpu",
    )

    out_tensors = load_file(str(out_dir / "model.safetensors"))
    buf_name = "model.layers.0.mlp.down_proj.input_rotation"
    assert buf_name in out_tensors
    buf = out_tensors[buf_name]
    assert buf.dtype == torch.int8
    assert buf.shape == (intermediate, intermediate)
    assert set(torch.unique(buf).tolist()) <= {-1, 1}

    # Selectivity: gate_proj is a non-target sibling in the same MLP. A buffer here would mean
    # the R4 pattern over-matched — and an over-matched layer gets no wrapper on reload, so its
    # buffer is orphaned (or worse, its weight was rotated with nothing to undo it).
    assert "model.layers.0.mlp.gate_proj.input_rotation" not in out_tensors


# ---------------------------------------------------------------------------
# Cross-flow equivalence: file-to-file vs the standard graph flow
# ---------------------------------------------------------------------------


class _TinyAttn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k_proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v_proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o_proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)


class _TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = torch.nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = torch.nn.Linear(INTERMEDIATE, HIDDEN, bias=False)


class _TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _TinyAttn()
        self.mlp = _TinyMLP()
        self.input_layernorm = torch.nn.LayerNorm(HIDDEN)
        self.post_attention_layernorm = torch.nn.LayerNorm(HIDDEN)


class _TinyInner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, HIDDEN)
        self.layers = torch.nn.ModuleList([_TinyLayer() for _ in range(NUM_LAYERS)])
        self.norm = torch.nn.LayerNorm(HIDDEN)


class _TinyModel(torch.nn.Module):
    """``nn.Module`` mirroring ``_build_tiny_checkpoint``'s tensor layout, so the graph
    flow's module-walking target resolution can be compared against the file-to-file
    model-free resolution on identical structure."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyInner()
        self.lm_head = torch.nn.Linear(HIDDEN, 32, bias=False)


@pytest.mark.parametrize(
    "rotation_config_fn",
    [
        pytest.param(_online_r1_rotation_config, id="online_r1"),
        pytest.param(_r4_rotation_config, id="r4"),
        # R2 contributes no online targets of its own (it is fully fused into the weights),
        # so enabling it alongside R1 must leave the online target set untouched -- including
        # when R1 targets v_proj, which R2 also rewrites.
        pytest.param(_r1_and_r2_rotation_config, id="online_r1_plus_r2"),
    ],
)
def test_online_rotation_targets_match_graph_flow(rotation_config_fn) -> None:  # type: ignore[no-untyped-def]
    """The file-to-file plan must select exactly the layers the reload path will wrap.

    ``QParamsLinearWithRotation`` decides which layers get the runtime ``x @ H`` rotation
    from ``RotationProcessor.get_online_rotation_layers``. File-to-file resolves the same
    targets model-free (from tensor names) and writes an ``input_rotation`` buffer for each.
    If the two disagree, the checkpoint is broken in one of two silent ways: a buffer with
    no wrapper to consume it, or a wrapper expecting a buffer that was never written.

    The two share the config-level target expansion (``expand_scaling_layer_targets``) but
    not the resolution step, which cannot be shared: the graph flow walks a live
    ``nn.Module``, file-to-file has only tensor names. This pins the resolution step.
    """
    rotation_config = rotation_config_fn()
    quant_config = _weight_only_int8_config(algo_config=[rotation_config])

    # Graph flow: resolve targets by walking a live module.
    graph_targets = sorted(RotationProcessor.get_online_rotation_layers(rotation_config, _TinyModel().eval()))

    # File-to-file: resolve targets from tensor names alone.
    weight_names = {
        name
        for name, tensor in _TinyModel().state_dict().items()
        if name.endswith(".weight") and tensor.ndim >= 2 and "embed" not in name
    }
    plan = build_rotation_plan(
        quant_config,
        {"hidden_size": HIDDEN, "intermediate_size": INTERMEDIATE, "num_attention_heads": NUM_HEADS},
        weight_names,
    )
    assert plan is not None
    file2file_targets = sorted(name.removesuffix(".weight") for name in plan.online_in)

    assert file2file_targets == graph_targets

    # The plan's targets are also what gets persisted for the reload path to read back.
    assert sorted(rotation_config.online_config.online_rotation_layers) == graph_targets


def test_input_rotation_buffer_matches_graph_flow_wrapper() -> None:
    """The persisted ``input_rotation`` buffer must be byte-identical to the one the graph
    flow registers on ``InputRotationWrapperHadamard``.

    Inference reconstructs the activation-side transform from this buffer, so any drift
    between the two producers silently corrupts outputs. Both now call the shared
    ``build_input_rotation_int8``; this test pins that they stay in agreement, and that
    the shared implementation itself still produces a valid Hadamard.
    """
    for rotation_size, in_features in ((HIDDEN, HIDDEN), (INTERMEDIATE, INTERMEDIATE)):
        linear = torch.nn.Linear(in_features, HIDDEN, bias=False)
        hadamard_K, K = _get_hadamard_K(rotation_size)
        wrapper = InputRotationWrapperHadamard(linear, hadamard_K=hadamard_K, K=K, rotation_size=rotation_size)

        file2file_buffer = _build_input_rotation_int8(rotation_size, RotationPlan())

        # Both producers must agree: catches one growing a private copy that drifts.
        assert torch.equal(file2file_buffer, wrapper.input_rotation), (
            f"input_rotation mismatch at rotation_size={rotation_size}"
        )
        assert file2file_buffer.dtype == wrapper.input_rotation.dtype

        # Independent oracle: agreement alone would still hold if the shared builder were
        # corrupted, so verify the buffer is genuinely an orthogonal Hadamard (H @ H.T == n*I)
        # matching scipy's canonical matrix up to the kron construction.
        as_float = file2file_buffer.to(torch.float64)
        assert torch.equal(as_float @ as_float.T, rotation_size * torch.eye(rotation_size, dtype=torch.float64)), (
            f"input_rotation is not an orthogonal Hadamard at rotation_size={rotation_size}"
        )
        assert file2file_buffer[0, 0] == 1, "Hadamard construction must start with +1"


# ---------------------------------------------------------------------------
# Kron expansion path coverage (lines 393-395, 435-441)
# ---------------------------------------------------------------------------
# To trigger the kron expansion, rotation_size must NOT be a power of 2
# and must be a multiple of a known Hadamard size. 12 is in
# KNOWN_HADAMARD_MATRICES; 12 * 2 = 24 triggers the kron path because
# _get_hadamard_K(24) returns a (12, 12) matrix with K=12.


KRON_ROTATION_SIZE = 24


def test_build_input_rotation_int8_kron_expansion() -> None:
    """When rotation_size requires kron expansion, _build_input_rotation_int8 produces
    a correct ±1 int8 matrix of the right size."""
    plan = RotationPlan()
    buf = _build_input_rotation_int8(KRON_ROTATION_SIZE, plan)
    assert buf.dtype == torch.int8
    assert buf.shape == (KRON_ROTATION_SIZE, KRON_ROTATION_SIZE)
    assert set(torch.unique(buf).tolist()) <= {-1, 1}


def test_rotate_input_channels_online_kron_expansion() -> None:
    """When rotation_size < in_features and the base Hadamard needs kron expansion,
    _rotate_input_channels_online must still produce a valid rotation."""
    plan = RotationPlan()
    in_features = KRON_ROTATION_SIZE * 2
    torch.manual_seed(42)
    weight = torch.randn(8, in_features)

    rotated, input_rotation = _rotate_input_channels_online(weight, KRON_ROTATION_SIZE, plan)
    assert rotated.shape == weight.shape
    assert rotated.dtype == weight.dtype
    assert input_rotation.dtype == torch.int8
    assert input_rotation.shape == (KRON_ROTATION_SIZE, KRON_ROTATION_SIZE)


def test_rotate_input_channels_online_kron_full_width() -> None:
    """Kron expansion when rotation_size == in_features (block-diagonal with one block)."""
    plan = RotationPlan()
    torch.manual_seed(42)
    weight = torch.randn(8, KRON_ROTATION_SIZE)

    rotated, input_rotation = _rotate_input_channels_online(weight, KRON_ROTATION_SIZE, plan)
    assert rotated.shape == weight.shape
    assert input_rotation.shape == (KRON_ROTATION_SIZE, KRON_ROTATION_SIZE)
