# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Structural search loading must not require materialized checkpoint weights."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.mark.parametrize("dtype,expected", [("BF16", 128), ("F8_E4M3", 128), ("F32", 256)])
def test_export_memory_estimate_uses_restored_weights(tmp_path, dtype, expected):
    from safetensors.torch import save_file

    from quark.torch.quantization import file2file_utils

    torch_dtype = {"BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32}[dtype]
    save_file({"layer.weight": torch.zeros((8, 8)).to(torch_dtype)}, tmp_path / "model.safetensors")
    assert file2file_utils.estimate_model_weight_bytes(str(tmp_path), element_size=2) == expected


@pytest.mark.parametrize("hub_source", [False, True])
@pytest.mark.parametrize(
    "free_bytes,cached_bytes,dtype,has_weights,required",
    [
        ([255], 0, "bfloat16", True, True),
        ([256], 0, "bfloat16", True, False),
        ([128, 128], 0, "bfloat16", True, False),
        ([200], 56, "bfloat16", True, False),
        ([511], 0, "float32", True, True),
        ([], 0, "bfloat16", True, False),
        ([0], 0, "bfloat16", False, False),
    ],
)
def test_large_model_export_uses_available_memory_before_execution(
    tmp_path, monkeypatch, caplog, hub_source, free_bytes, cached_bytes, dtype, has_weights, required
):
    from safetensors.torch import save_file

    from quark.experimental.torch.mix_precision import quantizer as implementation

    (tmp_path / "config.json").write_text(json.dumps({"dtype": dtype}))
    if has_weights:
        save_file({"layer.weight": torch.zeros((8, 8), dtype=torch.bfloat16)}, tmp_path / "model.safetensors")
    download = MagicMock(return_value=str(tmp_path))
    monkeypatch.setattr(implementation, "snapshot_download", download)
    monkeypatch.setattr(implementation, "hf_hub_download", lambda **_: str(tmp_path / "config.json"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: bool(free_bytes))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(free_bytes))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free_bytes[device], 1024))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 100 + cached_bytes)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 100)
    config = implementation.MixPrecisionConfig()
    quantizer = implementation.MixPrecisionQuantizer(config)
    source = "org/model" if hub_source else str(tmp_path)
    assert quantizer.prepare_export(source) is required
    assert config.file2file_quantization is False
    assert ("memory" in caplog.text) is required
    candidates = [{"mlp_mode": "fp8"}, {"mlp_mode": "ptpc_fp8"}]
    assert quantizer.filter_export_configs(candidates) == (candidates[1:] if required else candidates)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: pytest.fail("Reuse the pre-search budget decision"))
    assert quantizer.prepare_export(source) is required
    assert download.call_count == int(hub_source and bool(free_bytes))


@pytest.mark.parametrize(
    "weight_dtype,scale_dtype,scale_width,shard,required,weight_bytes",
    [
        ("I8", "F8_E8M0", 1, "model.safetensors", True, 2112),
        ("U8", "F8_E8M0", 1, None, True, 2112),
        ("I8", "F32", 1, None, False, 1152),
        ("F8_E4M3", "F8_E8M0", 1, None, False, 1088),
        ("I8", "F8_E8M0", 2, None, False, 1152),
        ("I8", "F8_E8M0", 1, "other.safetensors", False, 2112),
    ],
)
def test_export_requirement_uses_weight_format_without_loading_tensors(
    tmp_path, monkeypatch, caplog, weight_dtype, scale_dtype, scale_width, shard, required, weight_bytes
):
    from quark.experimental.torch.mix_precision import MixPrecisionConfig, MixPrecisionQuantizer
    from quark.torch.quantization import file2file_utils

    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}))
    scale_bytes = 32 * scale_width * (4 if scale_dtype == "F32" else 1)
    header = json.dumps(
        {
            "layers.0.experts.0.w1.weight": {"dtype": weight_dtype, "shape": [32, 16], "data_offsets": [0, 512]},
            "layers.0.experts.0.w1.scale": {
                "dtype": scale_dtype,
                "shape": [32, scale_width],
                "data_offsets": [512, 512 + scale_bytes],
            },
        }
    ).encode()
    header += b" " * (-len(header) % 8)
    (tmp_path / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(512 + scale_bytes))
    if shard:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "layers.0.experts.0.w1.weight": "model.safetensors",
                        "layers.0.experts.0.w1.scale": shard,
                    }
                }
            )
        )
    real_open = file2file_utils.safe_open

    class HeaderOnly:
        def __enter__(self):
            self.context = real_open(tmp_path / "model.safetensors", framework="pt")
            self.handle = self.context.__enter__()
            return self

        def __exit__(self, *args):
            return self.context.__exit__(*args)

        def keys(self):
            return self.handle.keys()

        def get_slice(self, name):
            return self.handle.get_slice(name)

        def get_tensor(self, name):
            pytest.fail("Export routing must not load tensor payloads")

    monkeypatch.setattr(file2file_utils, "safe_open", lambda *args, **kwargs: HeaderOnly())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert file2file_utils.estimate_model_weight_bytes(str(tmp_path)) == weight_bytes
    config = MixPrecisionConfig()
    quantizer = MixPrecisionQuantizer(config)
    assert quantizer.prepare_export(str(tmp_path)) is required
    assert config.file2file_quantization is False
    assert quantizer.config.file2file_quantization is required
    assert ("packed MXFP4" in caplog.text) is required


@pytest.mark.parametrize(
    "method,required", [("fp8", True), ("quark", False), (None, False), ("compressed-tensors", False)]
)
def test_hub_export_preparation_gates_header_inspection_and_caches_result(tmp_path, monkeypatch, method, required):
    from quark.experimental.torch.mix_precision import quantizer as implementation

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"quantization_config": {"quant_method": method}}))
    download = MagicMock(return_value=str(config_path))
    monkeypatch.setattr(implementation, "hf_hub_download", download)
    resolve = MagicMock(return_value=(str(tmp_path), {}))
    monkeypatch.setattr(implementation, "_resolve_file_to_file_source", resolve)
    detect = MagicMock(return_value=True)
    monkeypatch.setattr(implementation, "has_packed_mxfp4_source", detect)
    memory_reason = MagicMock(return_value=None)
    monkeypatch.setattr(implementation, "_file_to_file_memory_reason", memory_reason)
    quantizer = implementation.MixPrecisionQuantizer(implementation.MixPrecisionConfig())
    assert quantizer.prepare_export("org/model") is required
    assert quantizer.prepare_export("org/model") is required
    download.assert_called_once_with(repo_id="org/model", filename="config.json")
    assert detect.call_count == resolve.call_count == int(required)
    assert memory_reason.call_count == int(not required and method != "compressed-tensors")


def test_large_compressed_mxfp4_source_preserves_native_experts(tmp_path, monkeypatch):
    from safetensors.torch import load_file, save_file

    from quark.experimental.torch.mix_precision import MixPrecisionConfig, MixPrecisionQuantizer
    from quark.torch.quantization.config.config import QConfig, QLayerConfig
    from quark.torch.quantization.file2file_quantization import quantize_model_per_safetensor
    from quark.torch.quantization.file2file_utils import estimate_model_weight_bytes

    source = tmp_path / "source"
    source.mkdir()
    weights = {"num_bits": 4, "type": "float", "strategy": "group", "group_size": 32, "symmetric": True}
    source_config = {
        "quant_method": "compressed-tensors",
        "format": "mxfp4-pack-quantized",
        "config_groups": {"group_0": {"weights": weights}},
    }
    (source / "config.json").write_text(json.dumps({"text_config": {"quantization_config": source_config}}))
    packed = torch.full((4, 16), 0x22, dtype=torch.uint8)
    scale = torch.full((4, 1), 127, dtype=torch.uint8)
    dense = torch.ones((4, 32), dtype=torch.bfloat16)
    save_file(
        {"expert.weight_packed": packed, "expert.weight_scale": scale, "dense.weight": dense},
        source / "model.safetensors",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (0, 1024))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: 0)
    assert MixPrecisionQuantizer(MixPrecisionConfig()).prepare_export(str(source))
    assert estimate_model_weight_bytes(str(source)) == 520
    output = tmp_path / "export"
    quantize_model_per_safetensor(
        str(source),
        QConfig(global_quant_config=QLayerConfig(), exclude=["*"]),
        str(output),
        device="cpu",
        keep_excluded_layers_as_original_model_state=True,
    )
    tensors = load_file(output / "model.safetensors")
    assert torch.equal(tensors["expert.weight"], packed)
    assert torch.equal(tensors["expert.weight_scale"], scale)
    assert torch.equal(tensors["dense.weight"], dense)
    config = json.loads((output / "config.json").read_text())["quantization_config"]
    assert config["layer_quant_config"]["expert"]["weight"]["dtype"] == "fp4"
    assert config["exclude"] == ["dense"]


def test_meta_loading_preserves_fp4_expert_logical_shapes(tmp_path):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "DeepseekV4Config"):
        pytest.skip("transformers build without DeepSeek V4")

    from quark.experimental.torch.mix_precision.quantizer import _preprocess_qconfig_model
    from quark.experimental.torch.mix_precision.run_helpers import load_transformers_model
    from quark.experimental.torch.mix_precision.utils import categorize_layers

    config = transformers.DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        vocab_size=128,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=32,
        q_lora_rank=16,
        o_groups=1,
        o_lora_rank=16,
        n_routed_experts=2,
        num_experts_per_tok=1,
        num_nextn_predict_layers=0,
        compress_ratios=[0],
        expert_dtype="fp4",
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "activation_scheme": "dynamic",
            "scale_fmt": "ue8m0",
        },
    )
    config.save_pretrained(tmp_path)

    model = load_transformers_model(str(tmp_path), device_map="meta")
    _preprocess_qconfig_model(model)

    assert all(parameter.is_meta for parameter in model.parameters())
    assert model.config.expert_dtype == "fp4"
    assert model.config.quantization_config["quant_method"] == "fp8"
    experts = model.model.layers[0].mlp.experts
    assert experts.get_submodule("0.gate_proj").weight.shape == (32, 64)
    assert experts.get_submodule("0.up_proj").weight.shape == (32, 64)
    assert experts.get_submodule("0.down_proj").weight.shape == (64, 32)
    expected_projections = {
        f"model.layers.0.mlp.experts.{expert}.{projection}"
        for expert in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    assert expected_projections <= categorize_layers(model)["routed_moe"]


def test_materialized_loading_keeps_existing_loader(monkeypatch):
    from quark.experimental.torch.mix_precision.run_helpers import load_transformers_model
    from quark.torch.utils import llm

    model = object()
    loader = MagicMock(return_value=(model, None))
    monkeypatch.setattr(llm, "get_model", loader)

    assert load_transformers_model("model", torch_dtype=torch.bfloat16, device_map="auto") is model
    loader.assert_called_once_with(
        "model",
        data_type="bfloat16",
        device="auto",
        multi_gpu=False,
        multi_device=False,
        attn_implementation="eager",
        trust_remote_code=True,
    )


def test_file2file_v4_export_applies_saved_modes_to_source_names(monkeypatch, tmp_path):
    if not hasattr(pytest.importorskip("transformers"), "DeepseekV4Config"):
        pytest.skip("transformers build without DeepSeek V4")

    from quark.experimental.torch.mix_precision import quantizer as implementation
    from quark.experimental.torch.mix_precision.config import MixPrecisionConfig, get_layer_config
    from quark.torch.quantization.config.config import QConfig

    source_names = [
        "layers.0.ffn.experts.0.w1.weight",
        "layers.0.ffn.experts.0.w2.weight",
        "layers.0.ffn.experts.0.w3.weight",
        "layers.0.ffn.shared_experts.w1.weight",
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.compressor.wgate.weight",
        "layers.0.attn.indexer.compressor.wgate.weight",
        "layers.0.ffn.gate.weight",
        "head.weight",
        "mtp.0.ffn.experts.0.w1.weight",
    ]
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(source_names, "model.safetensors")})
    )
    fp8 = get_layer_config("ptpc_fp8")
    fp4 = get_layer_config("mxfp4")
    qconfig = QConfig(
        global_quant_config=fp8,
        layer_quant_config={
            "model.layers.*.mlp.experts.*.*_proj": fp4,
            "model.layers.*.mlp.shared_experts.gate_proj": fp4,
        },
        exclude=["model.layers.*.self_attn.q_a_proj", "lm_head"],
    )
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="deepseek_v4")
    exporter = implementation.MixPrecisionQuantizer(MixPrecisionConfig())
    exporter.model_path = str(tmp_path)
    exporter.result = SimpleNamespace(best_config={"routed_moe_mode": "mxfp4", "kv_cache_mode": "native"})
    monkeypatch.setattr(
        implementation, "_resolve_file_to_file_source", lambda _path: (str(tmp_path), {"model_type": "deepseek_v4"})
    )
    monkeypatch.setattr(exporter, "_load_preprocessed_export_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(implementation, "create_qconfig_from_quant_config", lambda **_kwargs: qconfig)
    captured = []
    monkeypatch.setattr(
        implementation,
        "ModelQuantizer",
        lambda config: SimpleNamespace(direct_quantize_checkpoint=lambda **_kwargs: captured.append(config)),
    )
    exporter.export_best(str(tmp_path / "output"), file2file_quantization=True)
    from fnmatch import fnmatch

    result = captured[0]
    for name in source_names[:4]:
        matching = [
            value
            for pattern, value in result.layer_quant_config.items()
            if fnmatch(name.removesuffix(".weight"), pattern)
        ]
        assert matching and all(value == fp4 for value in matching), name
    for name in [source_names[4], *source_names[7:]]:
        assert any(fnmatch(name.removesuffix(".weight"), pattern) for pattern in result.exclude), name
    for name in source_names[5:7]:
        assert not any(fnmatch(name.removesuffix(".weight"), pattern) for pattern in result.exclude)
        matching = [
            value
            for pattern, value in result.layer_quant_config.items()
            if fnmatch(name.removesuffix(".weight"), pattern)
        ]
        assert (matching or [result.global_quant_config]) == [fp8]


@pytest.mark.parametrize("expert_dtype, expected_bits, expected_mode", [("fp4", 4, "mxfp4"), ("fp8", 8, None)])
def test_deepseek_v4_mixed_source_precision(expert_dtype, expected_bits, expected_mode):
    from quark.experimental.torch.mix_precision.moe_backend import resolve_search_moe_backend
    from quark.experimental.torch.mix_precision.quantizer import (
        _resolve_partition_source_weight_bitwidths,
        _resolve_source_weight_mode,
    )

    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="deepseek_v4",
            expert_dtype=expert_dtype,
            quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]},
        )
    )
    partitions = {
        "self_attn": {"model.layers.0.self_attn.q_proj"},
        "routed_moe": {"model.layers.0.mlp.experts.0.gate_proj"},
        "shared_expert": {"model.layers.0.mlp.shared_experts.gate_proj"},
    }
    bitwidths = _resolve_partition_source_weight_bitwidths(model, None, partitions)
    assert bitwidths == {"self_attn": 8, "routed_moe": expected_bits, "shared_expert": 8}
    mode = _resolve_source_weight_mode(model, None)
    assert mode == expected_mode
    args = ["--tensor-parallel-size", "2"]
    resolved, decision = resolve_search_moe_backend(
        args,
        [{"routed_moe_mode": "mxfp4"}],
        bitwidths,
        model_config=model.config,
        source_weight_mode=mode,
    )
    assert resolved == [*args, "--moe-backend=auto"]
    assert decision.requires_weight_requantization is (expert_dtype == "fp8")
