#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Convert Quark AWQ checkpoints into llama.cpp-compatible GGUF."""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from quark.common.utils.log import ScreenLogger
from quark.torch.export.llama_cpp_export.formats import LlamaCppExportFormat, get_export_format
from quark.torch.export.llama_cpp_export.ggml_quantizer import GgmlQuantizer
from quark.torch.export.llama_cpp_export.quark_awq_unpack import (
    build_native_passthrough_loaders,
    build_quark_tensor_loaders,
)
from quark.torch.export.llama_cpp_export.scheme_compat import (
    QuarkNativeScheme,
    validate_native_passthrough,
)

logger = ScreenLogger(__name__)

_COPY_SUFFIXES = {".json", ".jinja", ".txt", ".md", ".model"}
_COPY_NAMES = {".gitattributes", "LICENSE"}
_TOKENIZER_NAMES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


@dataclass
class LlamaCppConvertConfig:
    quark_model_dir: Path
    output_dir: Path
    export_format: str
    name: str
    llama_cpp_dir: Path
    libggml: Path
    tokenizer_source: Path | None = None
    group_size: int | None = None
    pack_method: str | None = None
    native_passthrough: bool | None = None
    split_max_size: str = "8G"
    max_tensors: int | None = None
    dry_run: bool = False
    keep_staging: bool = False


def split_str_to_n_bytes(split_str: str) -> int:
    if split_str.endswith("K"):
        return int(split_str[:-1]) * 1000
    if split_str.endswith("M"):
        return int(split_str[:-1]) * 1000 * 1000
    if split_str.endswith("G"):
        return int(split_str[:-1]) * 1000 * 1000 * 1000
    if split_str.isnumeric():
        return int(split_str)
    raise ValueError(f"Invalid split size: {split_str}")


def _setup_llama_cpp_imports(llama_cpp_dir: Path) -> None:
    gguf_py = llama_cpp_dir / "gguf-py"
    llama_cpp = str(llama_cpp_dir)
    if "NO_LOCAL_GGUF" not in os.environ and gguf_py.exists():
        sys.path.insert(0, str(gguf_py))
    if llama_cpp not in sys.path:
        sys.path.insert(0, llama_cpp)


def _copy_metadata_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if not item.is_file() and not item.is_symlink():
            continue
        if item.name.startswith("model") and item.suffix == ".safetensors":
            continue
        if item.name == "model.safetensors.index.json":
            continue
        if item.name in _COPY_NAMES or item.suffix in _COPY_SUFFIXES:
            shutil.copy2(item, dst / item.name, follow_symlinks=True)


def _copy_tokenizer_files(src: Path, dst: Path) -> None:
    for name in _TOKENIZER_NAMES:
        path = src / name
        if path.exists():
            shutil.copy2(path, dst / name, follow_symlinks=True)


def prepare_staging_dir(
    quark_dir: Path,
    staging_dir: Path,
    tokenizer_source: Path | None,
) -> None:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    _copy_metadata_files(quark_dir, staging_dir)
    if tokenizer_source is not None:
        _copy_tokenizer_files(tokenizer_source, staging_dir)

    config_path = staging_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("quantization_config", None)
    config["torch_dtype"] = config.get("torch_dtype", config.get("dtype", "float16"))
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def convert_quark_checkpoint_to_llama_cpp_gguf(config: LlamaCppConvertConfig) -> Path:
    """Stream a Quark checkpoint into a public llama.cpp GGUF file."""
    fmt = get_export_format(config.export_format)
    _setup_llama_cpp_imports(config.llama_cpp_dir)

    import gguf
    from conversion import get_model_architecture, get_model_class
    from conversion.base import ModelBase, ModelType

    quark_dir = config.quark_model_dir.resolve()
    out_dir = config.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = out_dir / f".{config.name}-llama-cpp-staging"

    native_scheme: QuarkNativeScheme | None = None
    if config.native_passthrough is False:
        native_scheme = None
    elif config.native_passthrough is True or fmt.name in {"q4_0", "q4_1"}:
        try:
            native_scheme = validate_native_passthrough(quark_dir, fmt.name)
        except ValueError:
            if config.native_passthrough is True:
                raise
            native_scheme = None

    if native_scheme is not None:
        group_size = native_scheme.group_size
        pack_method = native_scheme.pack_method
        logger.info(
            "Using native GGUF passthrough (%s, group_size=%d, pack_method=%s)",
            fmt.name,
            group_size,
            pack_method,
        )
    else:
        from quark.torch.export.llama_cpp_export.scheme_compat import read_quark_export_config

        weight_cfg, export_cfg = read_quark_export_config(quark_dir)
        group_size = config.group_size or int(weight_cfg.get("group_size", 128))
        pack_method = config.pack_method or export_cfg.get("pack_method", "reorder")

    prepare_staging_dir(quark_dir, staging_dir, config.tokenizer_source)

    hparams = ModelBase.load_hparams(staging_dir, is_mistral_format=False)
    hparams.pop("quantization_config", None)
    architecture = get_model_architecture(hparams, ModelType.TEXT)
    model_class = get_model_class(architecture)

    outfile = out_dir / f"{config.name}-{fmt.name}-{{ftype}}.gguf"
    quantizer = GgmlQuantizer(config.libggml)
    pack_reorder = pack_method == "reorder"
    use_native = native_scheme is not None

    class QuarkLlamaCppModel(model_class):  # type: ignore[misc, valid-type]
        model_arch = model_class.model_arch
        no_mtp = True
        _shard_cache = None

        def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
            del remote_hf_model_id
            hparams_local = {**self.hparams, **self.hparams.get("text_config", {})}
            key = next(
                (
                    k
                    for k in (
                        "n_layers",
                        "num_hidden_layers",
                        "n_layer",
                        "num_layers",
                    )
                    if k in hparams_local
                ),
                None,
            )
            type(self)._original_block_count = hparams_local.get(key)
            if use_native:
                assert native_scheme is not None
                loaders, cache = build_native_passthrough_loaders(
                    quark_dir,
                    native_scheme,
                    max_tensors=config.max_tensors,
                )
            else:
                loaders, cache = build_quark_tensor_loaders(
                    quark_dir,
                    group_size=group_size,
                    pack_reorder=pack_reorder,
                    out_dtype=torch.float32,
                    max_tensors=config.max_tensors,
                )
            QuarkLlamaCppModel._shard_cache = cache
            filtered: dict[str, Callable[[], Tensor]] = {}
            for item in loaders.items():
                if titem := self.filter_tensors(item):
                    tname, tgen = titem
                    filtered[tname] = tgen
            return filtered

        def dequant_model(self) -> None:
            return

        def prepare_tensors(self) -> None:
            if self.tensor_map.mapping:
                max_name_len = (
                    max(len(s) for _, s in self.tensor_map.mapping.values())
                    + len(".weight,")
                )
            else:
                max_name_len = len("vision_encoder.weight,")

            for name, data_torch in chain(
                self.generate_extra_tensors(),
                self.get_tensors(),
            ):
                if name.endswith(
                    (".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")
                ):
                    continue

                old_dtype = data_torch.dtype
                if data_torch.dtype not in (torch.float16, torch.float32):
                    data_torch = data_torch.to(torch.float32)

                bid = None
                for part in name.split("."):
                    if part.isdecimal():
                        bid = int(part)
                        break

                for new_name, data_torch in self.modify_tensors(data_torch, name, bid):
                    if use_native and data_torch.dtype == torch.uint8:
                        data_qtype = fmt.weight_qtype
                        data = data_torch.numpy()
                        shape = gguf.quant_shape_from_byte_shape(data.shape, data_qtype)
                        shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"
                        logger.info(
                            f"{f'%-{max_name_len}s' % f'{new_name},'} "
                            f"quark-native --> {data_qtype.name}, shape = {shape_str}"
                        )
                        self.gguf_writer.add_tensor(new_name, data, raw_dtype=data_qtype)
                        continue

                    data = data_torch.numpy()
                    n_dims = len(data.shape)
                    data_qtype: gguf.GGMLQuantizationType | bool = (
                        self.tensor_force_quant(new_name, new_name, bid, n_dims)
                    )

                    if n_dims <= 1 or new_name.endswith("_norm.weight"):
                        data_qtype = gguf.GGMLQuantizationType.F32

                    if data_qtype is False and (
                        any(
                            self.match_model_tensor_name(new_name, key, bid)
                            for key in (
                                gguf.MODEL_TENSOR.FFN_GATE_INP,
                                gguf.MODEL_TENSOR.FFN_GATE_INP_SHEXP,
                                gguf.MODEL_TENSOR.POS_EMBD,
                                gguf.MODEL_TENSOR.TOKEN_TYPES,
                                gguf.MODEL_TENSOR.SSM_CONV1D,
                                gguf.MODEL_TENSOR.SHORTCONV_CONV,
                                gguf.MODEL_TENSOR.TIME_MIX_FIRST,
                                gguf.MODEL_TENSOR.TIME_MIX_W1,
                                gguf.MODEL_TENSOR.TIME_MIX_W2,
                                gguf.MODEL_TENSOR.TIME_MIX_DECAY_W1,
                                gguf.MODEL_TENSOR.TIME_MIX_DECAY_W2,
                                gguf.MODEL_TENSOR.TIME_MIX_LERP_FUSED,
                                gguf.MODEL_TENSOR.POSNET_NORM1,
                                gguf.MODEL_TENSOR.POSNET_NORM2,
                                gguf.MODEL_TENSOR.V_ENC_EMBD_POS,
                                gguf.MODEL_TENSOR.A_ENC_EMBD_POS,
                                gguf.MODEL_TENSOR.ALTUP_CORRECT_COEF,
                                gguf.MODEL_TENSOR.ALTUP_PREDICT_COEF,
                                gguf.MODEL_TENSOR.SSM_CONV1D_Q,
                                gguf.MODEL_TENSOR.SSM_CONV1D_K,
                                gguf.MODEL_TENSOR.SSM_CONV1D_V,
                                gguf.MODEL_TENSOR.INDEXER_PROJ,
                            )
                        )
                        or new_name[-7:] not in (".weight", ".lora_a", ".lora_b")
                    ):
                        data_qtype = gguf.GGMLQuantizationType.F32

                    if data_qtype is False and any(
                        self.match_model_tensor_name(new_name, key, bid)
                        for key in (
                            gguf.MODEL_TENSOR.TOKEN_EMBD,
                            gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD,
                            gguf.MODEL_TENSOR.OUTPUT,
                            gguf.MODEL_TENSOR.ALTUP_ROUTER,
                            gguf.MODEL_TENSOR.LAUREL_L,
                            gguf.MODEL_TENSOR.LAUREL_R,
                        )
                    ):
                        data_qtype = gguf.GGMLQuantizationType.F16

                    if isinstance(data_qtype, bool):
                        data_qtype = fmt.weight_qtype

                    try:
                        if use_native and fmt.name in {"q4_0", "q4_1"}:
                            data = gguf.quants.quantize(data, data_qtype)
                        elif fmt.use_libggml:
                            data = quantizer.quantize(data, data_qtype)
                        else:
                            data = gguf.quants.quantize(data, data_qtype)
                    except (gguf.QuantError, ValueError, RuntimeError) as exc:
                        logger.warning("%s, %s", exc, "falling back to F16")
                        data_qtype = gguf.GGMLQuantizationType.F16
                        data = gguf.quants.quantize(data, data_qtype)

                    shape = (
                        gguf.quant_shape_from_byte_shape(data.shape, data_qtype)
                        if data.dtype == np.uint8
                        else data.shape
                    )
                    shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"
                    logger.info(
                        f"{f'%-{max_name_len}s' % f'{new_name},'} "
                        f"{old_dtype} --> {data_qtype.name}, shape = {shape_str}"
                    )
                    self.gguf_writer.add_tensor(new_name, data, raw_dtype=data_qtype)

    model = QuarkLlamaCppModel(
        staging_dir,
        fmt.llama_file_type,
        outfile,
        hparams=hparams,
        eager=True,
        split_max_size=split_str_to_n_bytes(config.split_max_size),
        dry_run=config.dry_run,
        remote_hf_model_id=None,
    )

    try:
        logger.info(
            "Converting Quark checkpoint %s -> llama.cpp %s GGUF",
            quark_dir,
            fmt.name,
        )
        model.write()
    finally:
        if QuarkLlamaCppModel._shard_cache is not None:
            QuarkLlamaCppModel._shard_cache.close()

    if config.keep_staging:
        logger.info("Kept metadata staging dir: %s", staging_dir)
    else:
        shutil.rmtree(staging_dir, ignore_errors=True)

    out_path = model.fname_out
    if isinstance(out_path, Path) and out_path.is_dir():
        shards = sorted(out_path.glob(f"*{fmt.name}*.gguf"))
        return shards[0] if shards else out_path
    return Path(out_path)
