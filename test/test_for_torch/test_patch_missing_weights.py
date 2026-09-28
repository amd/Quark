#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Unit tests for _find_missing_weights_from_source and the restored-MTP config merging logic in
quark.torch.export.safetensors.

Covers the edge-case branches: tqdm unavailable, HuggingFace model-ID resolution
(success and failure), source checkpoint missing, single-file / sharded source
checkpoints, the ignore-pattern matching rules, and merging restored BF16 keys into
`exclude` / restored FP8 keys into `layer_quant_config` before `save_pretrained` is called.
"""

import json
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from quark.torch.export import safetensors as safetensors_module
from quark.torch.export import utils as export_utils_module
from quark.torch.export.safetensors import (
    _derive_exclude_entries,
    _find_missing_weights_from_source,
    _is_module_parameter_key,
    _load_weights_from_safetensors,
    _merge_exclude_entries_into_quantization_config,
    _merge_layer_quant_config_for_restored_quantized_layers,
    _resolve_source_model_dir,
    export_hf_model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(name_or_path=None, ignore_patterns=None):
    """Return a minimal fake nn.Module with .config._name_or_path set."""
    model = types.SimpleNamespace()
    if name_or_path is not None:
        model.config = types.SimpleNamespace(_name_or_path=name_or_path)
    else:
        model.config = None
    if ignore_patterns is not None:
        model._keys_to_ignore_on_load_unexpected = ignore_patterns
    return model


def _write_single_safetensors(directory: Path, tensors: dict[str, torch.Tensor]) -> None:
    save_file(tensors, str(directory / "model.safetensors"))


def _write_sharded_safetensors(
    directory: Path,
    shards: list[dict[str, torch.Tensor]],
) -> None:
    """Write multiple shard files and a matching index.json."""
    total = 0
    weight_map: dict[str, str] = {}
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(directory / fname))
        for k, t in shard.items():
            weight_map[k] = fname
            total += t.numel() * t.element_size()
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))


# ---------------------------------------------------------------------------
# Tests — early-return guard rails
# ---------------------------------------------------------------------------


def test_no_config_returns_early(tmp_path):
    """_find_missing_weights_from_source must return {} when model.config is None."""
    model = _make_model(name_or_path=None)  # config = None
    assert _find_missing_weights_from_source(model, existing_keys=set()) == {}


def test_empty_name_or_path_returns_early(tmp_path):
    """Empty string _name_or_path must trigger early return with a warning."""
    model = _make_model(name_or_path="")
    assert _find_missing_weights_from_source(model, existing_keys=set()) == {}


def test_source_dir_has_no_safetensors_returns_early(tmp_path):
    """
    When source_model_dir contains no safetensors files, _find_missing_weights_from_source
    must log a warning and return {} without raising.
    """
    source_dir = tmp_path / "source_empty"
    source_dir.mkdir()  # no files

    model = _make_model(name_or_path=str(source_dir))
    assert _find_missing_weights_from_source(model, existing_keys={"weight"}) == {}


# ---------------------------------------------------------------------------
# Tests — HuggingFace model-ID resolution via transformers.utils.cached_file
# ---------------------------------------------------------------------------


def test_model_id_resolved_via_cached_file(tmp_path):
    """
    When _name_or_path is not a local path, _find_missing_weights_from_source must attempt
    to resolve it through transformers.utils.cached_file.
    """
    source_dir = tmp_path / "hf_cache" / "snapshots" / "abc123"
    source_dir.mkdir(parents=True)
    # source has one extra tensor missing from existing_keys
    _write_single_safetensors(source_dir, {"weight": torch.ones(4), "mtp.extra": torch.zeros(2)})

    model = _make_model(name_or_path="SomeOrg/some-model-that-is-not-a-local-path")

    fake_config_path = str(source_dir / "config.json")

    # cached_file is imported inside the function body, so patch at its definition site.
    with patch("transformers.utils.cached_file", return_value=fake_config_path):
        missing = _find_missing_weights_from_source(model, existing_keys={"weight"})

    assert set(missing.keys()) == {"mtp.extra"}
    assert torch.equal(missing["mtp.extra"], torch.zeros(2))


def test_model_id_cached_file_raises_propagates(tmp_path):
    """When cached_file raises (model not in local cache), the error propagates —
    the source checkpoint is expected to be resolvable, so failing loudly is correct.
    """
    model = _make_model(name_or_path="NonExistentOrg/NonExistentModel")

    # cached_file is imported inside the function body, so patch at its definition site.
    with patch("transformers.utils.cached_file", side_effect=OSError("not cached")), pytest.raises(OSError):
        _find_missing_weights_from_source(model, existing_keys={"weight"})


def test_resolve_source_model_dir_without_transformers(tmp_path):
    """A local path resolves without transformers; a model ID gives up instead of raising ImportError.

    Resolving a model ID goes through the transformers cache, which this module treats as optional
    (see the ``is_transformers_available`` guards around its imports).
    """
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    with (
        patch.object(export_utils_module, "is_transformers_available", return_value=False),
        patch("transformers.utils.cached_file") as cached_file,
    ):
        assert _resolve_source_model_dir(_make_model(name_or_path=str(source_dir))) == source_dir
        assert _resolve_source_model_dir(_make_model(name_or_path="SomeOrg/some-model")) is None

    # Reaching transformers at all is the failure mode: the import itself would raise when the
    # package is genuinely absent, which patching cannot reproduce in an env that has it installed.
    cached_file.assert_not_called()


def test_resolve_source_model_dir_without_a_name_or_path():
    """A model that cannot name its source gives up rather than guessing at a directory."""
    assert _resolve_source_model_dir(_make_model(name_or_path=None)) is None


def test_find_missing_weights_without_transformers_says_what_is_missing(tmp_path):
    """The restore still fails loudly without transformers, but names the real problem.

    Dropping weights silently is not an option here, so unlike ``_resolve_source_model_dir`` this
    path raises. What it must not do is surface a bare ``No module named transformers.utils``,
    which reads like a Quark bug rather than a missing optional dependency.
    """
    model = _make_model(name_or_path="SomeOrg/some-model")

    with (
        patch.object(export_utils_module, "is_transformers_available", return_value=False),
        patch("transformers.utils.cached_file") as cached_file,
        pytest.raises(ImportError, match="requires transformers"),
    ):
        _find_missing_weights_from_source(model, existing_keys={"weight"})

    cached_file.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — tqdm unavailable (lines 22-23)
# ---------------------------------------------------------------------------


def test_find_missing_weights_works_without_tqdm(tmp_path, monkeypatch):
    """
    _find_missing_weights_from_source must work correctly even when tqdm is not installed.
    """
    # Simulate tqdm being absent by forcing _tqdm to None in the module
    monkeypatch.setattr(safetensors_module, "_tqdm", None)

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_single_safetensors(source_dir, {"weight": torch.ones(4), "mtp.layer.w": torch.zeros(2)})

    model = _make_model(name_or_path=str(source_dir))
    missing = _find_missing_weights_from_source(model, existing_keys={"weight"})

    assert "mtp.layer.w" in missing


# ---------------------------------------------------------------------------
# Tests — single-file source (happy path)
# ---------------------------------------------------------------------------


def test_missing_mtp_weights_are_found(tmp_path):
    """
    When the source has mtp.* weights absent from existing_keys, they must be returned.
    """
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tensors = {
        "model.weight": torch.ones(4),
        "mtp.0.weight": torch.full((3,), 2.0),
        "mtp.1.bias": torch.zeros(3),
    }
    _write_single_safetensors(source_dir, source_tensors)

    model = _make_model(name_or_path=str(source_dir))
    missing = _find_missing_weights_from_source(model, existing_keys={"model.weight"})

    assert set(missing.keys()) == {"mtp.0.weight", "mtp.1.bias"}
    assert torch.equal(missing["mtp.0.weight"], source_tensors["mtp.0.weight"])


def test_no_missing_weights_returns_empty(tmp_path):
    """When no weights are missing, an empty dict must be returned."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_single_safetensors(source_dir, {"weight": torch.ones(4)})

    model = _make_model(name_or_path=str(source_dir))
    missing = _find_missing_weights_from_source(model, existing_keys={"weight"})

    assert missing == {}


def test_explicit_empty_ignore_patterns_finds_nothing(tmp_path):
    """
    When model._keys_to_ignore_on_load_unexpected == [], no weights should be
    returned even if they are missing from existing_keys.
    """
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_single_safetensors(source_dir, {"weight": torch.ones(4), "mtp.0.w": torch.zeros(2)})

    model = _make_model(name_or_path=str(source_dir), ignore_patterns=[])
    missing = _find_missing_weights_from_source(model, existing_keys={"weight"})

    assert "mtp.0.w" not in missing


# ---------------------------------------------------------------------------
# Tests — sharded source checkpoint
# ---------------------------------------------------------------------------


def test_sharded_source_missing_weights_are_found(tmp_path):
    """_find_missing_weights_from_source works correctly when the source checkpoint is sharded."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_sharded_safetensors(
        source_dir,
        [
            {"shard1.weight": torch.ones(4)},
            {"shard2.weight": torch.ones(4), "mtp.extra": torch.full((2,), 7.0)},
        ],
    )

    model = _make_model(name_or_path=str(source_dir))
    missing = _find_missing_weights_from_source(model, existing_keys={"shard1.weight", "shard2.weight"})

    assert set(missing.keys()) == {"mtp.extra"}
    assert torch.equal(missing["mtp.extra"], torch.full((2,), 7.0))


# ---------------------------------------------------------------------------
# Tests — _derive_exclude_entries
# ---------------------------------------------------------------------------


def test_derive_exclude_entries_strips_weight_and_bias_suffix():
    keys = {"mtp.fc.weight", "mtp.layers.0.self_attn.q_proj.bias"}
    assert _derive_exclude_entries(keys) == {"mtp.fc", "mtp.layers.0.self_attn.q_proj"}


def test_derive_exclude_entries_no_regex_for_mtp_or_model_layers():
    """Every restored key becomes its exact module name -- no `re:`-anchored patterns,
    regardless of whether it lives under `mtp.*` or `model.layers.{N}.*`."""
    keys = {
        "mtp.layers.0.self_attn.q_proj.weight",
        "model.layers.78.self_attn.q_proj.weight",
    }
    entries = _derive_exclude_entries(keys)
    assert entries == {"mtp.layers.0.self_attn.q_proj", "model.layers.78.self_attn.q_proj"}
    assert not any(entry.startswith("re:") for entry in entries)


def test_derive_exclude_entries_key_without_known_suffix_is_unchanged():
    assert _derive_exclude_entries({"mtp.some_buffer"}) == {"mtp.some_buffer"}


# ---------------------------------------------------------------------------
# Tests — _merge_exclude_entries_into_quantization_config
# ---------------------------------------------------------------------------


def _make_model_with_quantization_config(quantization_config):
    model = types.SimpleNamespace()
    model.config = types.SimpleNamespace(quantization_config=quantization_config)
    return model


def test_merge_exclude_noop_when_no_restored_keys():
    quant_config = {"quant_method": "quark", "exclude": ["lm_head"]}
    model = _make_model_with_quantization_config(quant_config)

    _merge_exclude_entries_into_quantization_config(model, set())

    assert quant_config == {"quant_method": "quark", "exclude": ["lm_head"]}


def test_merge_exclude_noop_when_no_quantization_config():
    model = _make_model_with_quantization_config(None)
    # Must not raise even though there is nothing to merge into.
    _merge_exclude_entries_into_quantization_config(model, {"mtp.fc.weight"})


def test_merge_exclude_adds_full_name_entry_to_exclude_field():
    """The restored BF16 MTP weight must be added to quantization_config['exclude'] by its
    full module name (no regex), matching the on-disk quark custom mode."""
    quant_config = {"quant_method": "quark", "exclude": ["lm_head", "model.layers.0.mlp.gate"]}
    model = _make_model_with_quantization_config(quant_config)

    _merge_exclude_entries_into_quantization_config(model, {"mtp.layers.0.mlp.experts.0.gate_proj.weight"})

    assert "mtp.layers.0.mlp.experts.0.gate_proj" in quant_config["exclude"]
    # existing entries are preserved
    assert "lm_head" in quant_config["exclude"] and "model.layers.0.mlp.gate" in quant_config["exclude"]


def test_merge_exclude_uses_ignored_layers_field_for_fp8_custom_mode():
    quant_config = {"quant_method": "fp8", "ignored_layers": ["lm_head"]}
    model = _make_model_with_quantization_config(quant_config)

    _merge_exclude_entries_into_quantization_config(model, {"model.layers.78.self_attn.q_proj.weight"})

    assert "model.layers.78.self_attn.q_proj" in quant_config["ignored_layers"]


def test_merge_exclude_dedup_does_not_duplicate_existing_entry():
    quant_config = {"quant_method": "quark", "exclude": ["lm_head", "mtp.fc"]}
    model = _make_model_with_quantization_config(quant_config)

    _merge_exclude_entries_into_quantization_config(model, {"mtp.fc.weight"})

    assert quant_config["exclude"].count("mtp.fc") == 1


def test_merge_exclude_noop_when_no_recognized_exclude_field():
    """awq's `modules_to_not_convert` field is not a recognized exclude field, so this
    must be a no-op rather than silently adding an `exclude`/`ignored_layers` key."""
    quant_config = {"quant_method": "awq", "modules_to_not_convert": ["lm_head"]}
    model = _make_model_with_quantization_config(quant_config)

    _merge_exclude_entries_into_quantization_config(model, {"mtp.fc.weight"})

    assert quant_config == {"quant_method": "awq", "modules_to_not_convert": ["lm_head"]}


def test_is_module_parameter_key_skips_scale_tensors():
    assert _is_module_parameter_key("mtp.layers.0.self_attn.q_proj.weight")
    assert _is_module_parameter_key("mtp.layers.0.self_attn.q_proj.bias")
    assert not _is_module_parameter_key("mtp.layers.0.self_attn.q_proj.weight_scale_inv")


def test_merge_exclude_skips_weight_scale_inv_companion_tensors():
    quant_config = {"quant_method": "quark", "exclude": ["lm_head"]}
    model = _make_model_with_quantization_config(quant_config)

    _merge_exclude_entries_into_quantization_config(
        model,
        {
            "mtp.layers.0.self_attn.q_proj.weight_scale_inv",
            "mtp.fc.weight",
        },
    )

    assert "mtp.fc" in quant_config["exclude"]
    assert not any("scale_inv" in entry for entry in quant_config["exclude"])


def test_merge_exclude_leaves_the_config_untouched_for_companion_tensors_alone():
    """When every restored key is a companion tensor, ``exclude`` is not touched at all.

    Distinct from the case above, where a real parameter came along for the ride: here there is
    no module to name, so the merge has to bow out rather than write an empty-handed entry.
    """
    quant_config = {"quant_method": "quark", "exclude": ["lm_head"]}
    model = _make_model_with_quantization_config(quant_config)

    _merge_exclude_entries_into_quantization_config(model, {"mtp.layers.0.self_attn.q_proj.weight_scale_inv"})

    assert quant_config["exclude"] == ["lm_head"]


def test_merge_layer_quant_config_noop_when_no_quantization_config():
    model = _make_model_with_quantization_config(None)
    updated_config = _merge_layer_quant_config_for_restored_quantized_layers(
        model,
        {"mtp.layers.0.self_attn.q_proj.weight": torch.zeros((8, 8), dtype=torch.float8_e4m3fn)},
    )
    assert updated_config is None


def test_merge_layer_quant_config_returns_a_new_config_when_there_are_no_quantized_weights():
    quant_config = {"quant_method": "quark", "exclude": ["lm_head"]}
    model = _make_model_with_quantization_config(quant_config)

    updated_config = _merge_layer_quant_config_for_restored_quantized_layers(
        model,
        {"mtp.fc.weight": torch.zeros((8, 8), dtype=torch.bfloat16)},
    )

    assert updated_config == quant_config
    assert updated_config is not quant_config


# ---------------------------------------------------------------------------
# Tests — end-to-end export. No checkpoint is downloaded: the model is built
# from a config declared in code, and the "source checkpoint" is written to tmp_path.
# ---------------------------------------------------------------------------


def _tiny_qwen3_5_moe():
    """Instantiate a real -- but tiny -- Qwen3.5-MoE model, without downloading anything.

    This is the class that declares ``_keys_to_ignore_on_load_unexpected = ["^mtp.*"]`` and never
    builds an MTP block, which is precisely the situation the missing-weights restore exists for.
    """
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

    config = Qwen3_5MoeConfig(
        text_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
            moe_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
        ),
        vision_config=dict(hidden_size=32, intermediate_size=64, depth=2, num_heads=2, out_hidden_size=32),
    )
    return Qwen3_5MoeForConditionalGeneration(config)


def _write_source_with_fp8_mtp(model, source_dir: Path) -> dict[str, torch.Tensor]:
    """Write a source checkpoint holding the model's own weights plus an FP8 MTP block.

    Mirrors a real Qwen3.5-*-FP8 checkpoint: MTP Linear weights are ``float8_e4m3fn`` paired with a
    ``weight_scale_inv``, alongside high-precision MTP tensors such as ``mtp.fc.weight``.
    """
    source_dir.mkdir()
    mtp_tensors = {
        "mtp.fc.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "mtp.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(8, 8).to(torch.float8_e4m3fn),
        "mtp.layers.0.mlp.experts.0.gate_proj.bias": torch.randn(8, dtype=torch.bfloat16),
        "mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": torch.ones(1, dtype=torch.float32),
    }
    own_tensors = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    _write_single_safetensors(source_dir, {**own_tensors, **mtp_tensors})
    model.config._name_or_path = str(source_dir)
    return mtp_tensors


def _attach_fp8_quantizer(model):
    """Attach the quantizer transformers gives any checkpoint carrying an fp8 ``quantization_config``."""
    from transformers import FineGrainedFP8Config
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

    quantizer = FineGrainedFP8HfQuantizer(FineGrainedFP8Config())
    quantizer.pre_quantized = True
    model.hf_quantizer = quantizer
    return quantizer


def _export_and_load(model, tmp_path: Path) -> dict[str, torch.Tensor]:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    export_hf_model(model, export_dir)
    return _load_weights_from_safetensors(str(export_dir))


# The one source shape `_build_exclude_aware_quant_config` knows how to describe: DeepSeek-style
# FP8 with block-quantized weights and runtime-computed activation scales.
_BLOCK_QUANTIZED_FP8 = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}


def _prepare_fp8_source_export(model, tmp_path: Path, source_quantization_config: dict) -> None:
    """Point ``model`` at an FP8 source checkpoint declaring ``source_quantization_config``."""
    source_dir = tmp_path / "source"
    _write_source_with_fp8_mtp(model, source_dir)
    (source_dir / "config.json").write_text(json.dumps({"quantization_config": source_quantization_config}))
    # What the Quark run being exported would have built for itself.
    model.config.quantization_config = {"quant_method": "quark", "exclude": [], "layer_quant_config": {}}
    _attach_fp8_quantizer(model)


def _export_and_load_with_config(model, tmp_path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Export, then read back both the weights and the ``quantization_config`` actually serialized."""
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    export_hf_model(model, export_dir)

    weights = _load_weights_from_safetensors(str(export_dir))
    written_config = json.loads((export_dir / "config.json").read_text())["quantization_config"]
    return weights, written_config


def test_export_describes_fp8_mtp_layers_from_a_real_source_config(tmp_path):
    """Restored FP8 MTP layers are described in the ``config.json`` that lands on disk.

    Nothing is mocked here: the entry is derived by reading the source ``config.json`` and
    inspecting its safetensors, and it is asserted on the serialized output rather than on the
    in-memory dict, so a config that never reaches disk cannot pass.
    """
    model = _tiny_qwen3_5_moe()
    _prepare_fp8_source_export(model, tmp_path, _BLOCK_QUANTIZED_FP8)

    weights, written_config = _export_and_load_with_config(model, tmp_path)

    entry = written_config["layer_quant_config"]["mtp.layers.0.mlp.experts.0.gate_proj"]
    assert entry["weight"]["dtype"] == "fp8_e4m3"
    assert entry["weight"]["qscheme"] == "per_block"
    assert "mtp.layers.0.mlp.experts.0.gate_proj.weight" in weights
    # A described layer is not also excluded; its BF16 sibling is excluded instead of described.
    assert "mtp.layers.0.mlp.experts.0.gate_proj" not in written_config["exclude"]
    assert "mtp.fc" in written_config["exclude"]


def test_export_overwrites_an_existing_layer_quant_config_entry_from_source(tmp_path):
    """A restored layer uses its source checkpoint's description.

    The layer was absent from the live model and therefore was not quantized by the current Quark
    run. Any existing entry describes the requested format, not the restored bytes on disk.
    """
    model = _tiny_qwen3_5_moe()
    _prepare_fp8_source_export(model, tmp_path, _BLOCK_QUANTIZED_FP8)

    existing_entry = {"weight": {"dtype": "fp8_e5m2", "qscheme": "per_tensor"}}
    model.config.quantization_config["layer_quant_config"] = {"mtp.layers.0.mlp.experts.0.gate_proj": existing_entry}

    _, written_config = _export_and_load_with_config(model, tmp_path)

    entry = written_config["layer_quant_config"]["mtp.layers.0.mlp.experts.0.gate_proj"]
    assert entry["weight"]["dtype"] == "fp8_e4m3"
    assert entry["weight"]["qscheme"] == "per_block"


def test_export_describes_mtp_layers_when_layer_quant_config_is_null(tmp_path):
    """A ``layer_quant_config`` of ``None`` is treated as empty rather than crashing the export.

    Quark<1.0 serialized ``"layer_quant_config": null`` (see the legacy branch in
    ``QConfig.from_dict``), so the key can be present while holding ``None``. Merging into that
    value directly would raise before ``save_pretrained`` writes anything.
    """
    model = _tiny_qwen3_5_moe()
    _prepare_fp8_source_export(model, tmp_path, _BLOCK_QUANTIZED_FP8)
    model.config.quantization_config["layer_quant_config"] = None

    _, written_config = _export_and_load_with_config(model, tmp_path)

    entry = written_config["layer_quant_config"]["mtp.layers.0.mlp.experts.0.gate_proj"]
    assert entry["weight"]["dtype"] == "fp8_e4m3"


@pytest.mark.parametrize(
    "source_quantization_config",
    [
        pytest.param(
            {"quant_method": "fp8", "activation_scheme": "static", "weight_block_size": [128, 128]},
            id="static-activations",
        ),
        pytest.param({"quant_method": "fp8", "activation_scheme": "dynamic"}, id="not-block-quantized"),
        pytest.param({"quant_method": "awq", "bits": 4}, id="not-fp8-at-all"),
    ],
)
def test_export_survives_a_source_the_config_builder_cannot_describe(source_quantization_config, tmp_path):
    """A source shape the builder rejects must cost the description, not the whole export.

    ``_build_exclude_aware_quant_config`` raises for anything that is neither a quark source nor
    block-quantized FP8 with dynamic activations. That call happens before ``save_pretrained``, so
    letting the error escape leaves an empty output directory -- discarding a full calibration run
    to gain nothing. Degrading to "weights exported, layers undescribed" is the intended trade-off.
    """
    model = _tiny_qwen3_5_moe()
    _prepare_fp8_source_export(model, tmp_path, source_quantization_config)

    weights, written_config = _export_and_load_with_config(model, tmp_path)

    assert "mtp.layers.0.mlp.experts.0.gate_proj.weight" in weights
    assert "mtp.fc.weight" in weights
    assert not written_config["layer_quant_config"]


@pytest.mark.parametrize(
    "source_config_text",
    [
        pytest.param('{"quantization_config": {"quant_method": "fp8"', id="truncated-json"),
        pytest.param('{"quantization_config": "fp8"}', id="quantization-config-not-a-mapping"),
        pytest.param("", id="empty-file"),
    ],
)
def test_export_survives_an_unreadable_source_config(source_config_text, tmp_path):
    """A source ``config.json`` we cannot parse costs the description, not the export.

    The builder signals "I do not understand this source" with ``ValueError``, but a config file
    can also be truncated or hold a ``quantization_config`` that is not a mapping at all, which
    surfaces as ``JSONDecodeError`` or ``AttributeError`` instead. All of it happens before
    ``save_pretrained``, so none of it may reach the caller.
    """
    model = _tiny_qwen3_5_moe()
    _prepare_fp8_source_export(model, tmp_path, _BLOCK_QUANTIZED_FP8)
    (tmp_path / "source" / "config.json").write_text(source_config_text)

    weights, written_config = _export_and_load_with_config(model, tmp_path)

    assert "mtp.layers.0.mlp.experts.0.gate_proj.weight" in weights
    assert "mtp.fc.weight" in weights
    assert not written_config["layer_quant_config"]


def test_export_excludes_a_restored_layer_the_source_leaves_undescribed(tmp_path):
    """A source offering no scheme for a restored layer still gets that layer into ``exclude``.

    The builder splits the modules it is handed into the ones it can describe and the ones it
    leaves excluded. Reading only the described half would drop the rest from both fields, and
    a loader would then apply this run's global scheme to bytes that do not match it.
    """
    model = _tiny_qwen3_5_moe()
    source_without_a_scheme = {
        "quant_method": "quark",
        "exclude": [],
        "layer_quant_config": {},
        "global_quant_config": None,
    }
    _prepare_fp8_source_export(model, tmp_path, source_without_a_scheme)

    weights, written_config = _export_and_load_with_config(model, tmp_path)

    assert "mtp.layers.0.mlp.experts.0.gate_proj.weight" in weights
    assert "mtp.layers.0.mlp.experts.0.gate_proj" in written_config["exclude"]
    assert not written_config["layer_quant_config"]


def test_export_excludes_a_restored_layer_the_source_itself_excludes(tmp_path):
    """A layer the source already lists as excluded is carried over rather than dropped."""
    model = _tiny_qwen3_5_moe()
    source_excluding_the_layer = {
        "quant_method": "quark",
        "exclude": ["mtp.layers.0.mlp.experts.0.gate_proj"],
        "layer_quant_config": {},
        "global_quant_config": None,
    }
    _prepare_fp8_source_export(model, tmp_path, source_excluding_the_layer)

    weights, written_config = _export_and_load_with_config(model, tmp_path)

    assert "mtp.layers.0.mlp.experts.0.gate_proj.weight" in weights
    assert "mtp.layers.0.mlp.experts.0.gate_proj" in written_config["exclude"]


def test_export_survives_a_source_directory_without_a_config(tmp_path):
    """A source checkpoint holding weights but no ``config.json`` still exports those weights.

    The restore reads the safetensors directly and never needs the config, so a source that has
    been stripped down to its weights must still hand back the MTP block -- only the description
    of it is lost.
    """
    model = _tiny_qwen3_5_moe()
    _prepare_fp8_source_export(model, tmp_path, _BLOCK_QUANTIZED_FP8)
    (tmp_path / "source" / "config.json").unlink()

    weights, written_config = _export_and_load_with_config(model, tmp_path)

    assert "mtp.layers.0.mlp.experts.0.gate_proj.weight" in weights
    assert "mtp.fc.weight" in weights
    assert not written_config["layer_quant_config"]


def test_export_survives_a_source_lookup_that_fails_unexpectedly(tmp_path):
    """A cache lookup that fails in an unforeseen way costs the description, not the export.

    The weights are already restored by the time this runs, so an exception type nobody listed
    must not be the difference between a written checkpoint and an empty directory. Transformers
    does normalize bad model IDs to ``OSError`` today, which is the point of pinning this: the
    guarantee has to survive that changing, not a failure already seen in the wild.

    ``cached_file`` is mocked because "a failure we did not anticipate" has no unmocked form --
    what is under test is the class of failure, not one instance of it. The first call is the
    weight restore and must succeed, so only the second one fails.
    """
    model = _tiny_qwen3_5_moe()
    _prepare_fp8_source_export(model, tmp_path, _BLOCK_QUANTIZED_FP8)
    source_config = tmp_path / "source" / "config.json"
    model.config._name_or_path = "SomeOrg/some-model"

    with patch(
        "transformers.utils.cached_file",
        side_effect=[str(source_config), ValueError("Repo id must be in the form 'namespace/repo_name'")],
    ):
        weights, written_config = _export_and_load_with_config(model, tmp_path)

    assert "mtp.layers.0.mlp.experts.0.gate_proj.weight" in weights
    assert "mtp.fc.weight" in weights
    assert not written_config["layer_quant_config"]


def test_export_leaves_a_layer_quant_config_that_is_not_a_mapping(tmp_path):
    """A ``layer_quant_config`` of the wrong type is reported, not merged into and not overwritten.

    Merging would raise ``AttributeError`` and take the whole export down with it; replacing it
    with a fresh dict would throw away whatever is in there. Neither is worth doing for optional
    metadata, so the export says what it found and moves on.
    """
    model = _tiny_qwen3_5_moe()
    _prepare_fp8_source_export(model, tmp_path, _BLOCK_QUANTIZED_FP8)
    model.config.quantization_config["layer_quant_config"] = ["not a mapping"]

    weights, written_config = _export_and_load_with_config(model, tmp_path)

    assert "mtp.layers.0.mlp.experts.0.gate_proj.weight" in weights
    assert written_config["layer_quant_config"] == ["not a mapping"]


def test_module_named_after_a_scale_is_still_a_module(tmp_path):
    """Only the key's suffix decides, so a module whose name contains "scale" is not mistaken
    for a companion tensor.

    Companion tensors are ``weight_scale``, ``weight_scale_inv``, ``input_scale`` and friends --
    none of them end in ``.weight``. Rejecting anything with "scale" in the name would drop a
    layer like ``rescale_proj`` out of ``exclude``, which is the very bug this reports (#6067).
    """
    assert _is_module_parameter_key("mtp.layers.0.mlp.rescale_proj.weight")
    assert _is_module_parameter_key("mtp.scale_attn.c_proj.bias")
    assert not _is_module_parameter_key("mtp.layers.0.mlp.rescale_proj.weight_scale_inv")


def test_export_restores_mtp_weights_for_fp8_pretrained_model(tmp_path):
    """MTP weights must reach the safetensors file when the source is an FP8 pre-quantized model.

    Such checkpoints carry a ``quantization_config``, so transformers attaches an ``hf_quantizer``,
    and ``save_pretrained`` then rebinds its ``state_dict`` argument from
    ``hf_quantizer.get_state_dict_and_metadata()`` -- discarding whatever the caller passed in.
    Merging the restored tensors into ``state_dict`` is therefore not enough on its own.
    """
    model = _tiny_qwen3_5_moe()
    assert not any(key.startswith("mtp.") for key in model.state_dict()), "transformers should drop mtp.*"

    mtp_tensors = _write_source_with_fp8_mtp(model, tmp_path / "source")
    quantizer = _attach_fp8_quantizer(model)

    exported = _export_and_load(model, tmp_path)

    for name, tensor in mtp_tensors.items():
        assert name in exported, f"{name} is missing from the exported checkpoint"
        assert torch.equal(exported[name].to(torch.float32), tensor.to(torch.float32))
    assert model.hf_quantizer is quantizer, "the quantizer must be re-attached after the save"


def test_export_rejects_restored_packed_compressed_tensors_module(tmp_path):
    """An unsupported packed module aborts the export instead of producing an incomplete checkpoint."""
    model = _tiny_qwen3_5_moe()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    prefix = "mtp.layers.0.mlp.experts.0.gate_proj"
    source_tensors = {
        **{name: tensor.clone() for name, tensor in model.state_dict().items()},
        "mtp.fc.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        f"{prefix}.weight_packed": torch.zeros(8, 4, dtype=torch.uint8),
        f"{prefix}.weight_scale": torch.ones(8, 1, dtype=torch.float32),
        f"{prefix}.weight_shape": torch.tensor([8, 8], dtype=torch.int32),
        f"{prefix}.bias": torch.randn(8, dtype=torch.bfloat16),
    }
    _write_single_safetensors(source_dir, source_tensors)
    (source_dir / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "compressed-tensors", "format": "pack-quantized"}})
    )
    model.config._name_or_path = str(source_dir)
    model.config.quantization_config = {"quant_method": "quark", "exclude": [], "layer_quant_config": {}}
    _attach_fp8_quantizer(model)

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    with pytest.raises(NotImplementedError, match="aborting instead of writing an incomplete checkpoint"):
        export_hf_model(model, export_dir)

    assert not any(export_dir.iterdir())


def test_export_reattaches_the_quantizer_when_the_save_fails(tmp_path):
    """A failed save must not leave the model without its quantizer.

    Detaching the quantizer mutates the caller's model, so the export owes it back even when
    ``save_pretrained`` raises -- otherwise a recoverable error would silently turn the model
    into an unquantized one for whatever runs next.
    """
    model = _tiny_qwen3_5_moe()
    _write_source_with_fp8_mtp(model, tmp_path / "source")
    quantizer = _attach_fp8_quantizer(model)

    def _explode(*args, **kwargs):
        raise RuntimeError("no space left on device")

    model.save_pretrained = _explode

    with pytest.raises(RuntimeError, match="no space left on device"):
        export_hf_model(model, tmp_path / "export")

    assert model.hf_quantizer is quantizer


def test_export_restores_mtp_weights_without_quantizer(tmp_path):
    """The same restore must keep working for BF16 sources, which have no quantizer attached."""
    model = _tiny_qwen3_5_moe()
    mtp_tensors = _write_source_with_fp8_mtp(model, tmp_path / "source")
    assert getattr(model, "hf_quantizer", None) is None

    exported = _export_and_load(model, tmp_path)

    for name in mtp_tensors:
        assert name in exported, f"{name} is missing from the exported checkpoint"


def _write_source_with_fp4_mtp(model, source_dir: Path, scheme: str) -> dict[str, torch.Tensor]:
    """Write a source checkpoint whose MTP block is already quantized to NVFP4 or MXFP4.

    FP4 weights are ``uint8`` with the inner dimension halved, since a byte holds two codes. NVFP4
    pairs them with an fp8 per-group ``weight_scale`` plus a per-tensor ``weight_scale_2``, while
    MXFP4 uses a single e8m0 -- also ``uint8`` -- per-group ``weight_scale``.
    """
    source_dir.mkdir()
    prefix = "mtp.layers.0.mlp.experts.0.gate_proj"
    mtp_tensors = {
        "mtp.fc.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        f"{prefix}.weight": torch.randint(0, 255, (8, 4), dtype=torch.uint8),
    }
    if scheme == "nvfp4":
        mtp_tensors[f"{prefix}.weight_scale"] = torch.ones(8, 1).to(torch.float8_e4m3fn)
        mtp_tensors[f"{prefix}.weight_scale_2"] = torch.ones(1, dtype=torch.float32)
    else:
        mtp_tensors[f"{prefix}.weight_scale"] = torch.full((8, 1), 127, dtype=torch.uint8)

    own_tensors = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    _write_single_safetensors(source_dir, {**own_tensors, **mtp_tensors})
    model.config._name_or_path = str(source_dir)
    return mtp_tensors


@pytest.mark.parametrize("scheme", ["nvfp4", "mxfp4"])
def test_export_restores_mtp_weights_for_fp4_pretrained_model(scheme, tmp_path):
    """An MTP block that is already FP4 in the source must reach the export untouched.

    The restore matches on tensor names and treats anything containing ``scale`` as a companion
    tensor, so it never inspects dtypes -- FP4's packed ``uint8`` weights and the extra
    ``weight_scale_2`` that NVFP4 carries need no special handling. This pins that down, and runs
    through the quantizer bypass so the check covers the whole save path rather than the match alone.
    """
    model = _tiny_qwen3_5_moe()
    mtp_tensors = _write_source_with_fp4_mtp(model, tmp_path / "source", scheme)
    _attach_fp8_quantizer(model)

    exported = _export_and_load(model, tmp_path)

    for name, tensor in mtp_tensors.items():
        assert name in exported, f"{name} is missing from the exported checkpoint"
        assert exported[name].dtype == tensor.dtype, f"{name} changed dtype during export"
        assert exported[name].shape == tensor.shape, f"{name} changed shape during export"
        assert torch.equal(exported[name].to(torch.float32), tensor.to(torch.float32))


def _quark_source_quantization_config(scheme: str) -> dict:
    """The ``quantization_config`` a Quark FP4 export writes to its ``config.json``.

    Built from the scheme Quark itself would have applied, so the fixture cannot drift into a
    shape the builder never sees in production.
    """
    from quark.torch.quantization.config.template import LLMTemplate

    return {
        "quant_method": "quark",
        "global_quant_config": LLMTemplate._SCHEME_COLLECTION.get_scheme(scheme).config.to_dict(),
        "layer_quant_config": {},
        "exclude": [],
    }


@pytest.mark.parametrize("scheme", ["nvfp4", "mxfp4"])
def test_export_describes_fp4_mtp_layers_from_a_real_source_config(scheme, tmp_path):
    """Restored FP4 MTP layers are described in the exported ``config.json``.

    FP4 reaches ``layer_quant_config`` by a different route than FP8: a Quark-exported source
    declares ``quant_method: "quark"``, which short-circuits to the branch that copies the
    source's own description verbatim, while FP8 goes through the branch that hardcodes
    per-block. Passing the FP8 case therefore says nothing about this one.
    """
    model = _tiny_qwen3_5_moe()
    source_dir = tmp_path / "source"
    mtp_tensors = _write_source_with_fp4_mtp(model, source_dir, scheme)
    (source_dir / "config.json").write_text(
        json.dumps({"quantization_config": _quark_source_quantization_config(scheme)})
    )
    model.config.quantization_config = {"quant_method": "quark", "exclude": [], "layer_quant_config": {}}
    _attach_fp8_quantizer(model)

    weights, written_config = _export_and_load_with_config(model, tmp_path)

    entry = written_config["layer_quant_config"]["mtp.layers.0.mlp.experts.0.gate_proj"]
    # NVFP4 describes its weight as two levels -- fp4 groups plus an fp8 per-tensor quantization
    # of the group scales, the checkpoint's `weight_scale_2`. MXFP4 has a single level.
    levels = entry["weight"] if isinstance(entry["weight"], list) else [entry["weight"]]
    assert len(levels) == (2 if scheme == "nvfp4" else 1)
    assert levels[0]["dtype"] == "fp4"
    assert levels[0]["group_size"] == (16 if scheme == "nvfp4" else 32)
    assert levels[0]["scale_format"] == ("float32" if scheme == "nvfp4" else "e8m0")
    for name in mtp_tensors:
        assert name in weights, f"{name} is missing from the exported checkpoint"
    # The BF16 sibling has no scales on disk, so it is excluded rather than described.
    assert "mtp.fc" in written_config["exclude"]
    assert "mtp.fc" not in written_config["layer_quant_config"]


def test_export_leaves_quantizer_that_builds_its_own_state_dict(tmp_path):
    """Quantizers overriding ``get_state_dict_and_metadata`` keep ownership of the export state_dict.

    mxfp4 and torchao rebuild it (extra packed tensors, non-empty metadata), so detaching them would
    corrupt the checkpoint. Giving up the MTP restore is the intended trade-off in that case: the
    export must not swap one broken checkpoint for another. Because those tensors are not exported,
    their descriptions must not be added to the exported quantization config either.
    """
    from transformers import FineGrainedFP8Config
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

    class _StateDictOwningQuantizer(FineGrainedFP8HfQuantizer):
        def get_state_dict_and_metadata(self, model):
            return model.state_dict(), {}

    model = _tiny_qwen3_5_moe()
    _prepare_fp8_source_export(model, tmp_path, _BLOCK_QUANTIZED_FP8)
    mtp_tensors = _load_weights_from_safetensors(str(tmp_path / "source"))
    mtp_tensors = {name: tensor for name, tensor in mtp_tensors.items() if name.startswith("mtp.")}
    quantizer = _StateDictOwningQuantizer(FineGrainedFP8Config())
    quantizer.pre_quantized = True
    model.hf_quantizer = quantizer

    exported, written_config = _export_and_load_with_config(model, tmp_path)

    assert not any(name in exported for name in mtp_tensors)
    assert written_config["exclude"] == []
    assert written_config["layer_quant_config"] == {}
    assert model.config.quantization_config["exclude"] == []
    assert model.config.quantization_config["layer_quant_config"] == {}
    assert model.hf_quantizer is quantizer
