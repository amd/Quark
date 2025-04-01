#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2023-2024 The ggml authors
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from pathlib import Path
import torch
from typing import Sequence, Dict, Tuple, Any, Iterable
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)
try:
    import gguf  # type: ignore
    from gguf.constants import MODEL_ARCH, GGMLQuantizationType  # type: ignore
    from gguf.tensor_mapping import get_tensor_name_map  # type: ignore
except ImportError as e:
    logger.exception(str(e))
    raise ImportError("please install gguf==0.6.0")

from .tensor_convert import convert_from_gguf, build_quant_cfg, gguf_shape
from .utils import inverse_permute


class GGUFModelConverter():

    name_arch_map: Dict[str, MODEL_ARCH] = {"llama": MODEL_ARCH.LLAMA}

    def __init__(self, model_name: str, model_info: Dict[str, Any], param_info: Dict[str, torch.Tensor],
                 gguf_path: Path) -> None:
        self.model_name = model_name
        self.model_arch = self.name_arch_map[model_name]
        self._model_info = model_info
        self._param_info = param_info
        self.hparams = self.load_hparams()
        self.gguf_path = gguf_path
        self.gguf_reader = gguf.gguf_reader.GGUFReader(gguf_path)

    def convert(self) -> None:
        n_block = int(self.find_hparam(["n_layers", "num_hidden_layers", "n_layer"]))
        n_head = int(self.hparams.get("num_attention_heads"))  # type: ignore
        n_kv_head = int(self.hparams.get("num_key_value_heads"))  # type: ignore

        self.name_map = get_tensor_name_map(self.model_arch, n_block)
        self.gguf_name_tensor_map: Dict[str, gguf.gguf_reader.ReaderTensor] = {
            tensor.name: tensor
            for tensor in self.gguf_reader.tensors
        }

        for name, info in GGUFModelConverter.get_name_and_info(self._model_info["structure"]):
            if name not in self.name_map.mapping:
                continue
            gguf_name = self.name_map.get_name(name)
            weight_name = gguf_name + ".weight"
            if weight_name not in self.gguf_name_tensor_map:
                continue
            quark_name = name + ".weight"
            weight_dtype = self._param_info[quark_name].dtype

            gguf_weight_dtype = self.gguf_name_tensor_map[weight_name].tensor_type
            gguf_weight_shape = self.gguf_name_tensor_map[weight_name].shape.tolist()[::-1]

            gguf_store_shape = gguf_shape(gguf_weight_shape, gguf_weight_dtype)
            gguf_weight_data = torch.tensor(self.gguf_name_tensor_map[weight_name].data).reshape(gguf_store_shape)

            if quark_name.endswith("q_proj.weight"):
                gguf_weight_data = inverse_permute(gguf_weight_data, n_head, n_head)
            if quark_name.endswith("k_proj.weight"):
                gguf_weight_data = inverse_permute(gguf_weight_data, n_head, n_kv_head)

            if gguf_weight_dtype in [
                    GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.F64,
                    GGMLQuantizationType.BF16
            ]:
                self._param_info[quark_name] = gguf_weight_data.to(weight_dtype)
            else:
                data, scale, zero_point = convert_from_gguf(gguf_weight_data, gguf_type=gguf_weight_dtype)
                quant_cfg = build_quant_cfg(quark_name)
                info["weight_quant"] = quant_cfg
                info["type"] = "QuantLinear"
                self._param_info[quark_name] = data.to(weight_dtype)
                self._param_info[quant_cfg["scale"]] = scale.to(weight_dtype)  # type: ignore
                self._param_info[quant_cfg["zero_point"]] = zero_point.round().to(torch.int32)  # type: ignore

    def load_hparams(self) -> Dict[str, Any]:
        if self._model_info.get("config", None) is None:
            raise ValueError("Only support hugging face models' gguf-quark convertion")
        return self._model_info["config"]  # type: ignore

    def find_hparam(self, keys: Sequence[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in self.hparams), None)
        if key is not None:
            return self.hparams[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    @staticmethod
    def get_name_and_info(model_info: Dict[str, Any], parent_key: str = "") -> Iterable[Tuple[str, Dict[str, Any]]]:
        for key, value in model_info.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                if value.get("type", None) is not None:
                    yield new_key, value
                else:
                    yield from GGUFModelConverter.get_name_and_info(value, new_key)
            else:
                continue

    @property
    def model_info(self) -> Dict[str, Any]:
        return self._model_info

    @property
    def param_info(self) -> Dict[str, torch.Tensor]:
        return self._param_info
