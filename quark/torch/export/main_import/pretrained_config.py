#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json
from pathlib import Path
from typing import Union, Optional, Dict, Any

CONFIG_NAME = "config.json"


class PretrainedConfig():

    def __init__(self, pretrained_dir: Union[str, Path]) -> None:
        model_info_dir = Path(pretrained_dir)
        config_file_path = model_info_dir / CONFIG_NAME
        with open(config_file_path, "r", encoding="utf-8") as reader:
            text = reader.read()
        self.config_dict = json.loads(text)

    @property
    def quantization_config(self) -> Optional[Dict[str, Any]]:
        if self.config_dict.get("quantization_config", None) is not None:
            return self.config_dict["quantization_config"]  # type: ignore
        return None

    @property
    def quant_method(self) -> Optional[str]:
        if self.quantization_config is None:
            return None
        else:
            return self.quantization_config['quant_method']  # type: ignore

    @property
    def pack_method(self) -> Optional[str]:
        if self.quantization_config is not None and self.quantization_config['quant_method'] == "awq":
            return self.quantization_config["pack_method"]  # type: ignore
        if self.quantization_config is None or self.quantization_config.get("export", None) is None:
            return None
        else:
            return self.quantization_config["export"]['pack_method']  # type: ignore

    # quark has fake_quant and real_quant, fp8 and awq only have fake_quant
    @property
    def weight_format(self) -> Optional[str]:
        if self.quantization_config is not None and self.quantization_config['quant_method'] == "quark":
            return self.quantization_config["export"]['weight_format']  # type: ignore
        return None

    # only fp8 format have kv_layers_name, quark has fake_quant and real_quant, fp8 and awq only have fake
    @property
    def kv_layers_name(self) -> Optional[list[str]]:
        if self.quantization_config is not None and self.quantization_config[
                'quant_method'] == "fp8" and "export" in self.quantization_config:
            return self.quantization_config["export"]["kv_cache_group"]  # type: ignore
        return None
