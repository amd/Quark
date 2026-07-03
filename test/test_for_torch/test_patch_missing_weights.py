#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Unit tests for patch_missing_weights in quark.torch.export.safetensors.

Covers the edge-case branches that are not exercised by the main happy-path
tests: tqdm unavailable, HuggingFace model-ID resolution (success and failure),
empty export directory, and the single-file / sharded write-back paths.
"""

import json
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from quark.torch.export import safetensors as safetensors_module
from quark.torch.export.safetensors import patch_missing_weights

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
    """patch_missing_weights must return silently when model.config is None."""
    model = _make_model(name_or_path=None)  # config = None
    # Should not raise
    patch_missing_weights(tmp_path, model)


def test_empty_name_or_path_returns_early(tmp_path):
    """Empty string _name_or_path must trigger early return with a warning."""
    model = _make_model(name_or_path="")
    patch_missing_weights(tmp_path, model)


def test_export_dir_has_no_safetensors_returns_early(tmp_path):
    """
    When export_dir contains neither model.safetensors nor index.json,
    patch_missing_weights must log a warning and return without raising.

    This covers lines 93-94 of safetensors.py.
    """
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_single_safetensors(source_dir, {"weight": torch.ones(4)})

    export_dir = tmp_path / "export_empty"
    export_dir.mkdir()  # deliberately left empty

    model = _make_model(name_or_path=str(source_dir))
    # Must not raise
    patch_missing_weights(export_dir, model)


def test_source_dir_has_no_safetensors_returns_early(tmp_path):
    """
    When source_model_dir contains no safetensors files,
    patch_missing_weights must warn and return without raising.
    """
    source_dir = tmp_path / "source_empty"
    source_dir.mkdir()  # no files

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_single_safetensors(export_dir, {"weight": torch.ones(4)})

    model = _make_model(name_or_path=str(source_dir))
    patch_missing_weights(export_dir, model)


# ---------------------------------------------------------------------------
# Tests — HuggingFace model-ID resolution via transformers.utils.cached_file
# ---------------------------------------------------------------------------


def test_model_id_resolved_via_cached_file(tmp_path):
    """
    When _name_or_path is not a local path, patch_missing_weights must attempt
    to resolve it through transformers.utils.cached_file.

    This covers lines 70-71: successful cached_file lookup.
    """
    source_dir = tmp_path / "hf_cache" / "snapshots" / "abc123"
    source_dir.mkdir(parents=True)
    # source has one extra tensor the export is missing
    _write_single_safetensors(source_dir, {"weight": torch.ones(4), "mtp.extra": torch.zeros(2)})

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_single_safetensors(export_dir, {"weight": torch.ones(4)})

    model = _make_model(name_or_path="SomeOrg/some-model-that-is-not-a-local-path")

    fake_config_path = str(source_dir / "config.json")

    # cached_file is imported inside the function body, so patch at its definition site.
    with patch("transformers.utils.cached_file", return_value=fake_config_path):
        patch_missing_weights(export_dir, model)

    import safetensors

    with safetensors.safe_open(str(export_dir / "model.safetensors"), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    assert "mtp.extra" in keys
    assert "weight" in keys


def test_model_id_cached_file_raises_propagates(tmp_path):
    """When cached_file raises (model not in local cache), the error propagates —
    the source checkpoint is expected to be resolvable, so failing loudly is correct.
    """
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_single_safetensors(export_dir, {"weight": torch.ones(4)})

    model = _make_model(name_or_path="NonExistentOrg/NonExistentModel")

    # cached_file is imported inside the function body, so patch at its definition site.
    with patch("transformers.utils.cached_file", side_effect=OSError("not cached")), pytest.raises(OSError):
        patch_missing_weights(export_dir, model)


# ---------------------------------------------------------------------------
# Tests — tqdm unavailable (lines 22-23)
# ---------------------------------------------------------------------------


def test_patch_missing_weights_works_without_tqdm(tmp_path, monkeypatch):
    """
    patch_missing_weights must work correctly even when tqdm is not installed.

    This covers lines 22-23: the ImportError branch that sets _tqdm = None.
    """
    # Simulate tqdm being absent by forcing _tqdm to None in the module
    monkeypatch.setattr(safetensors_module, "_tqdm", None)

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_single_safetensors(source_dir, {"weight": torch.ones(4), "mtp.layer.w": torch.zeros(2)})

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_single_safetensors(export_dir, {"weight": torch.ones(4)})

    model = _make_model(name_or_path=str(source_dir))
    patch_missing_weights(export_dir, model)

    import safetensors

    with safetensors.safe_open(str(export_dir / "model.safetensors"), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    assert "mtp.layer.w" in keys


# ---------------------------------------------------------------------------
# Tests — single-file write-back (happy path)
# ---------------------------------------------------------------------------


def test_single_file_missing_mtp_weights_are_restored(tmp_path):
    """
    When the source has mtp.* weights absent from the single-file export,
    they must be appended to model.safetensors in-place.
    """
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tensors = {
        "model.weight": torch.ones(4),
        "mtp.0.weight": torch.full((3,), 2.0),
        "mtp.1.bias": torch.zeros(3),
    }
    _write_single_safetensors(source_dir, source_tensors)

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_single_safetensors(export_dir, {"model.weight": torch.ones(4)})

    model = _make_model(name_or_path=str(source_dir))
    patch_missing_weights(export_dir, model)

    import safetensors

    with safetensors.safe_open(str(export_dir / "model.safetensors"), framework="pt", device="cpu") as f:
        result = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118

    assert set(result.keys()) == {"model.weight", "mtp.0.weight", "mtp.1.bias"}
    assert torch.equal(result["mtp.0.weight"], source_tensors["mtp.0.weight"])


def test_no_missing_weights_leaves_file_unchanged(tmp_path):
    """When no weights are missing, the export file must not be modified."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_single_safetensors(source_dir, {"weight": torch.ones(4)})

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_single_safetensors(export_dir, {"weight": torch.ones(4)})

    mtime_before = (export_dir / "model.safetensors").stat().st_mtime
    model = _make_model(name_or_path=str(source_dir))
    patch_missing_weights(export_dir, model)
    mtime_after = (export_dir / "model.safetensors").stat().st_mtime

    assert mtime_before == mtime_after


def test_explicit_empty_ignore_patterns_patches_nothing(tmp_path):
    """
    When model._keys_to_ignore_on_load_unexpected == [], no weights should be
    patched even if they are missing from the export.
    """
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_single_safetensors(source_dir, {"weight": torch.ones(4), "mtp.0.w": torch.zeros(2)})

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_single_safetensors(export_dir, {"weight": torch.ones(4)})

    model = _make_model(name_or_path=str(source_dir), ignore_patterns=[])
    patch_missing_weights(export_dir, model)

    import safetensors

    with safetensors.safe_open(str(export_dir / "model.safetensors"), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    assert "mtp.0.w" not in keys


# ---------------------------------------------------------------------------
# Tests — sharded write-back (happy path)
# ---------------------------------------------------------------------------


def test_sharded_missing_weights_appended_to_last_shard(tmp_path):
    """
    For a sharded export, missing mtp.* weights must be appended to the last
    shard and the index.json weight_map must be updated accordingly.
    """
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_tensors = {
        "shard1.weight": torch.ones(4),
        "shard2.weight": torch.ones(4),
        "mtp.extra": torch.full((2,), 7.0),
    }
    _write_single_safetensors(source_dir, source_tensors)

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_sharded_safetensors(
        export_dir,
        [
            {"shard1.weight": torch.ones(4)},
            {"shard2.weight": torch.ones(4)},
        ],
    )

    model = _make_model(name_or_path=str(source_dir))
    patch_missing_weights(export_dir, model)

    # Verify index updated
    index = json.loads((export_dir / "model.safetensors.index.json").read_text())
    assert "mtp.extra" in index["weight_map"]
    last_shard = sorted(set(index["weight_map"].values()))[-1]
    assert index["weight_map"]["mtp.extra"] == last_shard

    # Verify tensor present in last shard
    import safetensors

    with safetensors.safe_open(str(export_dir / last_shard), framework="pt", device="cpu") as f:
        tensor = f.get_tensor("mtp.extra")
    assert torch.equal(tensor, source_tensors["mtp.extra"])


def test_sharded_index_total_size_updated(tmp_path):
    """total_size in index.json metadata must increase by the size of added tensors."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_single_safetensors(
        source_dir,
        {"weight": torch.ones(4), "mtp.w": torch.ones(4, dtype=torch.float32)},
    )

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_sharded_safetensors(export_dir, [{"weight": torch.ones(4)}])

    index_before = json.loads((export_dir / "model.safetensors.index.json").read_text())
    size_before = int(index_before["metadata"]["total_size"])

    model = _make_model(name_or_path=str(source_dir))
    patch_missing_weights(export_dir, model)

    index_after = json.loads((export_dir / "model.safetensors.index.json").read_text())
    size_after = int(index_after["metadata"]["total_size"])

    # mtp.w is 4 float32 elements = 16 bytes
    assert size_after == size_before + 16


def test_sharded_source_missing_weights_appended(tmp_path):
    """patch_missing_weights works correctly when the source is also sharded."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_sharded_safetensors(
        source_dir,
        [
            {"weight": torch.ones(4)},
            {"mtp.layer": torch.zeros(3)},
        ],
    )

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    _write_single_safetensors(export_dir, {"weight": torch.ones(4)})

    model = _make_model(name_or_path=str(source_dir))
    patch_missing_weights(export_dir, model)

    import safetensors

    with safetensors.safe_open(str(export_dir / "model.safetensors"), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    assert "mtp.layer" in keys
