#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""vLLM integration coverage for MTP weights restored during export."""

import gc
import json

import pytest
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText

from quark.torch.export.safetensors import export_hf_model
from quark.torch.quantization.config.config import QConfig
from quark.torch.quantization.config.template import LLMTemplate

_TINY_QWEN35_REPO = "yujiepan/qwen3.5-tiny-random"
_TINY_QWEN35_REVISION = "3a13bbea23b6df303d83c6127b4a40f1260598d5"


@pytest.mark.needs_real_vllm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM model loading requires a GPU")
def test_exported_restored_mtp_weights_load_in_vllm(tmp_path):
    """Restored BF16 MTP weights must be excluded by module name, as vLLM expects."""
    from vllm.config import (
        CompilationConfig,
        ModelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    source_dir = snapshot_download(
        repo_id=_TINY_QWEN35_REPO,
        revision=_TINY_QWEN35_REVISION,
        allow_patterns=["config.json", "model.safetensors"],
    )
    expected_mtp_fc_weight = load_file(f"{source_dir}/model.safetensors")["mtp.fc.weight"]
    model = AutoModelForImageTextToText.from_pretrained(source_dir, dtype=torch.bfloat16)

    quantization_config = QConfig(
        global_quant_config=LLMTemplate._SCHEME_COLLECTION.get_scheme("mxfp4").config,
        exclude=["lm_head"],
    ).to_dict()
    quantization_config["quant_method"] = "quark"
    quantization_config["export"] = {
        "kv_cache_group": [],
        "min_kv_scale": 0.0,
        "pack_method": "reorder",
        "weight_format": "real_quantized",
        "weight_merge_groups": None,
    }
    model.config.quantization_config = quantization_config

    export_dir = tmp_path / "export"
    export_hf_model(model, export_dir)
    del model

    with open(export_dir / "config.json") as config_file:
        exported_quantization_config = json.load(config_file)["quantization_config"]
    restored_mtp_excludes = [entry for entry in exported_quantization_config["exclude"] if entry.startswith("mtp.")]
    assert "mtp.fc" in restored_mtp_excludes
    assert "mtp.layers.0.mlp.down_proj" in restored_mtp_excludes
    assert all(not entry.endswith((".weight", ".bias")) for entry in restored_mtp_excludes)

    model_config = ModelConfig(
        model=str(export_dir),
        runner="draft",
        dtype="bfloat16",
        quantization="quark",
        skip_tokenizer_init=True,
        hf_overrides={"architectures": ["Qwen3_5MTP"]},
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        compilation_config=CompilationConfig(mode=0),
    )

    loaded_model = None
    try:
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{tmp_path / 'distributed_init'}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)
            loaded_model = DefaultModelLoader(vllm_config.load_config).load_model(vllm_config, model_config)

        assert type(loaded_model).__name__ == "Qwen3_5MTP"
        assert isinstance(loaded_model.model.fc.quant_method, UnquantizedLinearMethod)
        assert loaded_model.model.fc.weight.dtype == torch.bfloat16
        torch.testing.assert_close(loaded_model.model.fc.weight.cpu(), expected_mtp_fc_weight)
    finally:
        del loaded_model
        destroy_model_parallel()
        destroy_distributed_environment()
        gc.collect()
        torch.cuda.empty_cache()
