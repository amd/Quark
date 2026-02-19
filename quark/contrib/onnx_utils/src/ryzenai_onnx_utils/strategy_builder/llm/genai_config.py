# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import json
import logging
import re
from pathlib import Path
from typing import Any

from . import GenaiMode, LlmMode

_logger = logging.getLogger(__name__)

SESSION_OPTIONS = {
    r"amd_options",
    r"compile_fusion_rt",
    r"config_entries",
    r"custom_allocator",
    r"custom_ops_library",
    r"dd_cache",
    r"dd_root",
    r"external_data_file",
    r"fusion_opt.*"
    r"hybrid_dbg.*",
    r"hybrid_opt.*",
    r"model_name",
    r"onnx_custom_ops_const_key",
    r"external_data_blob.*",
}


class GenAiConfig:
    """
    The Config class in onnxruntime_genai does not offer all the API methods we
    need so wrap the file manually.
    """

    def __init__(self, model: str, mode: GenaiMode) -> None:
        self.model = model
        self.mode = mode

        self.config = {"model": {self.model: {"session_options": {"provider_options": []}}}, "search": {}}

        if self.mode == GenaiMode.STANDALONE:
            self._session_options()["provider_options"] = []
            self.set_session_option("custom_ops_library", "/path/to/custom_ops.dll")
            self.set_session_option("custom_allocator", "ryzen_mm")
        else:
            self.set_provider("RyzenAI")

    def finalize(self, model_name, llm_mode, max_seq_len, has_external_data: bool) -> None:
        # arbitrary non-empty text for fusion mode to get the model directory in OGA
        external_data_file = f"{model_name}.pb.bin" if has_external_data else "null"
        self.set_option("external_data_file", external_data_file)
        self.set_filename(f"{model_name}.onnx")

        if llm_mode in [LlmMode.NPU_EAGER, LlmMode.NPU_TOKEN_FUSION, LlmMode.FULL_FUSION_TOKEN]:
            self.set_option("hybrid_opt_token_backend", "npu")

        if llm_mode in [
            LlmMode.NPU_PREFILL_FUSION,
            LlmMode.NPU_TOKEN_FUSION,
            LlmMode.FULL_FUSION_PREFILL,
            LlmMode.FULL_FUSION_TOKEN,
        ]:
            self.set_option("max_length_for_kv_cache", f"{max_seq_len}")
            self.set_search_option("max_length", max_seq_len)

    def set_option(self, option: str, value: Any) -> None:
        if self.mode == GenaiMode.STANDALONE:
            self.set_session_option(option, value)
        else:
            self.set_provider_option("RyzenAI", option, value)

    def set_search_option(self, option: str, value: Any) -> None:
        self.config["search"][option] = value

    def set_filename(self, filename: str) -> None:
        self._config_model()["filename"] = filename

    def set_session_option(self, option: str, value: Any) -> None:
        self._session_options()[option] = value

    def set_provider(self, provider: str):
        # delete existing provider(s) if already present
        self._session_options()["provider_options"] = []
        self._session_options()["provider_options"].append({provider: {}})

    def set_provider_option(self, provider: str, option: str, value: Any) -> None:
        for p in self._session_options()["provider_options"]:
            if provider in p:
                p[provider][option] = value
                return
        raise ValueError(f"Provider {provider} not found in session options")

    def merge(self, other: "GenAiConfig") -> None:
        self._merge(self.config, other.config)

    @staticmethod
    def _merge(a: dict[str, Any], b: dict[str, Any], path=None):
        if path is None:
            path = []
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    GenAiConfig._merge(a[key], b[key], path + [str(key)])
                elif isinstance(a[key], list) and isinstance(b[key], list):
                    if key == "provider_options" and a[key] and b[key]:
                        assert len(a[key]) == len(b[key]) == 1
                        GenAiConfig._merge(a[key][0], b[key][0], path + [str(key)])
                    else:
                        for i in b[key]:
                            if i not in a[key]:
                                a[key].append(i)
                elif key == "external_data_file":
                    if b[key] != "null":
                        if a[key] == "null":
                            a[key] = b[key]
                        else:
                            assert a[key] == b[key], (
                                "Conflict at " + ".".join(path + [str(key)]) + f": {a[key]} != {b[key]}"
                            )
                    # if b[key] == "null", we keep a[key] as is. Either it will
                    # also be "null" or a valid filename
                elif key == "max_length":
                    a[key] = min(a[key], b[key])
                elif key == "filename":
                    # allow potential conflicts with filename assuming it will be corrected later
                    a[key] = b[key]
                elif a[key] != b[key]:
                    raise Exception("Conflict at " + ".".join(path + [str(key)]) + f": {a[key]} != {b[key]}")
            else:
                a[key] = b[key]

    def save(self, input_dir: Path, output_model: Path, dry_run: bool):
        input_path = None
        # prioritize a file named with _cpu or _dml over the base in case multiple are present
        # this helps starting with a good default config file
        for filename in ["genai_config_cpu.json", "genai_config_dml.json", "genai_config.json"]:
            if (input_dir / filename).exists():
                input_path = input_dir / filename
                break

        if input_path is None:
            _logger.warning("GenAI config file not found in %s, skipping generation", input_dir)
            return

        output_path = output_model.parent / f"{output_model.stem}_genai_config.json"

        if dry_run:
            print(f"Update GenAI config from {input_path} to {output_path}")
            return

        with input_path.open("r") as f:
            config: dict[str, Any] = json.load(f)

        # remove existing session options that we will override to prevent false conflicts
        existing_session_options = {
            x
            for x in self._session_options_static(config, self.model)
            if any(re.search(pattern, x) for pattern in SESSION_OPTIONS)
        }
        for option in existing_session_options:
            del self._session_options_static(config, self.model)[option]
        self._session_options_static(config, self.model)["provider_options"] = []

        self._merge(config, self.config)

        with output_path.open("w") as f:
            json.dump(config, f, indent=4)

    def _config_model(self) -> dict[str, Any]:
        return self._config_model_static(self.config, self.model)

    def _session_options(self) -> dict[str, Any]:
        return self._config_model()["session_options"]

    @staticmethod
    def _config_model_static(config: dict[str, Any], model: str) -> dict[str, Any]:
        return config["model"][model]

    @staticmethod
    def _session_options_static(config: dict[str, Any], model: str) -> dict[str, Any]:
        return GenAiConfig._config_model_static(config, model).get("session_options", {})
