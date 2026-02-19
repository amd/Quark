# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import importlib
import logging
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

import ryzenai_onnx_utils.strategy_builder as strategy_builder

if TYPE_CHECKING:
    from ryzenai_onnx_utils.optimize import LlmArgs

from . import SUPPORTED_LLM_OPERATORS, LlmMode, LlmModel, genai_config

_logger = logging.getLogger(__name__)


def _conditional_true(explicit_bool: bool | None, condition: bool) -> bool:
    if explicit_bool is not None:
        return explicit_bool
    return condition


class LlmModelBase:
    model_type = LlmModel.DEFAULT
    default_jit_enabled = True
    default_max_seq_len = 4096

    def __init__(self, args: LlmArgs, llm_mode: LlmMode):
        self.args = args
        self.llm_mode = llm_mode
        if self.args.max_seq_len is None:
            self.args.max_seq_len = self.default_max_seq_len

        self.base_strategy = self._get_empty_strategy()
        self.pass_builder = strategy_builder.PassBuilder()

        self.genai_config = genai_config.GenAiConfig(self.args.genai_config_model, self.args.genai_config)

        # certain model types, e.g. eager, those with LoRA, or offloaded embeddings
        # need to keep external data files
        self.keep_external_data = False

    def save(self, output_strategy_path: Path) -> None:
        if self.llm_mode in {LlmMode.HYBRID, LlmMode.NPU_EAGER}:
            self._eager()
        elif self.llm_mode in {LlmMode.NPU_TOKEN_FUSION, LlmMode.FULL_FUSION_TOKEN}:
            self._token_fusion()
        elif self.llm_mode in {LlmMode.NPU_PREFILL_FUSION, LlmMode.FULL_FUSION_PREFILL}:
            self._prefill_fusion()
        else:
            raise ValueError(f"Unsupported LLM mode {self.llm_mode}")

        self.base_strategy["passes"] = self.pass_builder.finalize()

        self._clamp_dynamic_shape_list()

        with open(output_strategy_path, "w") as f:
            yaml.dump(self.base_strategy, f)

    def generate_genai_config(self, model_name) -> genai_config.GenAiConfig:
        self.genai_config.finalize(model_name, self.llm_mode, self.args.max_seq_len, self.keep_external_data)

        return self.genai_config

    def need_prefill_fusion_backup(self) -> tuple[bool, int]:
        if self.llm_mode != LlmMode.FULL_FUSION_PREFILL:
            return False, 0
        if not self.args.prefill_fusion_eager_fallback:
            return False, 0
        dynamic_shape_list: list[dict[str, Any]] = self._get_attribute("dynamic_shape_list")
        max_seq_len_supported = 0
        for shape_dict in dynamic_shape_list:
            max_seq_len_supported = max(max_seq_len_supported, shape_dict.get("sequence_length_padded", 0))
        return max_seq_len_supported < self.args.max_seq_len, max_seq_len_supported

    def postprocess_attributes(self) -> dict[str, Any]:
        attributes = {}

        need_backup, max_seq_len_supported = self.need_prefill_fusion_backup()
        if need_backup:
            attributes["eager_switch_threshold"] = str(max_seq_len_supported)

        return attributes

    @classmethod
    def matches(cls, _input_path: str) -> bool:
        return False

    def _eager(self) -> None:
        self.keep_external_data = True
        if self.llm_mode == LlmMode.NPU_EAGER:
            self._set_attribute("cast_kv_cache", False)
            self._set_attribute("npu_only", True)

        operators = self._get_eager_operators()

        self._prune_logits(True)

        for operator in operators:
            if operator not in self.args.exclude:
                self.pass_builder.add_step(self._path(operator))

        if self.llm_mode == LlmMode.NPU_EAGER:
            self.pass_builder.add_pass("initializers_to_float16")

        self.pass_builder.add_step(self._path("npu_weights"))

        if self.llm_mode == LlmMode.HYBRID:
            self.pass_builder.add_step(self._path("simplify_casts_eager_hybrid"))
        else:
            self.pass_builder.add_conditional_pass("hybrid_llm_kv_cache_to_bf16", "hybrid_llm_gqo")
            self.pass_builder.add_step(self._path("simplify_casts_eager_npu"))

        self.pass_builder.add_step(self._path("simplify_casts"))
        self.pass_builder.add_step(self._path("cast_logits"))
        self.pass_builder.add_step(self._path("cleanup"))
        self.pass_builder.add_step(self._path("postprocess"))

        self._set_default_jit_attribute(self.default_jit_enabled)

        if self.args.lora:
            self.genai_config.set_option("hybrid_opt_npu_pdi_name", "DPU_4")

        self._set_attribute("xclbins", "")
        self._set_attribute("mladf_version", str(self.args.mladf_version))

        self._override_eager_attributes()

    def _token_fusion(self) -> None:
        self._set_token_fusion_attributes()

        if self.args.dynamic_attention_mask:
            self.pass_builder.add_step(self._path("dynamic_attention_mask"))

        if self._priority_memory():
            self.pass_builder.add_step(self._path("embeddings"))
            self._enable_offload_embedding()

        self.pass_builder.add_step(self._path("dd_token"))
        self.pass_builder.add_step(self._path("simplify_casts"))

        self._override_token_fusion_fuse()

        if self.args.lora:
            self.pass_builder.add_pass("remove_obsolete_lora_inputs")
            self.keep_external_data = True

        self.pass_builder.add_step(self._path("cast_logits"))
        self.pass_builder.add_step(self._path("cleanup"))

        if _conditional_true(self.args.shared_weights, self._priority_memory()) or self.args.lora:
            self.pass_builder.add_step(self._path("postprocess"))

    def _prefill_fusion(self) -> None:
        self._set_prefill_fusion_attributes()

        self.pass_builder.add_step(self._path("pad_input_ids"))

        if self._priority_memory():
            self.pass_builder.add_step(self._path("embeddings"))
            self._enable_offload_embedding()

        self._prune_logits(True)

        if self.llm_mode == LlmMode.FULL_FUSION_PREFILL:
            self.pass_builder.add_pass("hybrid_llm_kv_cache_to_bf16")

        self.pass_builder.add_step(self._path("dd_prefill"))
        self.pass_builder.add_step(self._path("simplify_casts"))
        self.pass_builder.add_step(self._path("fuse_prefill"))
        if self.args.lora:
            self.pass_builder.add_pass("remove_obsolete_lora_inputs")
            self.keep_external_data = True
        # must occur before cast_logits
        self.pass_builder.add_pass("promote_slices")
        self.pass_builder.add_step(self._path("cast_logits"))
        self.pass_builder.add_step(self._path("cleanup"))

    def _prune_logits(self, default_val: str | bool):
        prune_logits = self.args.prune_logits
        if prune_logits is None:
            prune_logits = default_val
        if prune_logits:
            self.pass_builder.add_step(self._path("prune_logits"))
            self._set_attribute("prune_logits", prune_logits)

    def _get_eager_operators(self) -> list[str]:
        operators = set()
        if self.args.include_only:
            operators = set(self.args.include_only)
        else:
            operators = set(SUPPORTED_LLM_OPERATORS)
            for op in ["embeddings", "qmoe", "fuse_qkv_matmul_concat"]:
                if op not in self.args.include:
                    operators.discard(op)
                # operators.discard("fuse_kv")

            if self._priority_memory():
                operators.add("embeddings")

            if self.llm_mode == LlmMode.HYBRID:
                operators.add("fuse_qkv_matmul_concat")

            if _conditional_true(self.args.shared_weights, self._priority_memory()) or self.args.lora:
                operators.discard("fuse_qkv_matmul")

        operators = self._override_eager_operators(operators)

        if "embeddings" in operators:
            self._enable_offload_embedding()
        if "qmoe" in operators:
            self._set_attribute("offload_qmoe", True)
            self.keep_external_data = True

        if _conditional_true(self.args.shared_weights, self._priority_memory()):
            self._set_attribute("enable_model_hash", True)
        if self.args.lora:
            self._set_attribute("lora", True)

        # sort the user input operators based on the supported operators order
        operators = sorted(operators, key=lambda x: SUPPORTED_LLM_OPERATORS.index(x))

        return operators

    def _set_token_fusion_attributes(self) -> None:
        self._set_attribute("mladf_version", str(strategy_builder.MladfVersion.FLAT))
        self._set_attribute("is_llm", True)

        if self.args.dynamic_attention_mask is None:
            # default token fusion to use dynamic attention mask
            self.args.dynamic_attention_mask = True

        if self.args.lora:
            self._set_attribute("xclbins", "stx_LLM_TPS_mladf_2x4x4_bfp16_gemm_flat_rms_silu_mul")
            self._set_attribute("lora", True)
            self._set_attribute("pdi_id", 0)
        elif self.args.dynamic_attention_mask or self.args.ctrl_pkt:
            self._set_attribute("xclbins", "stx_LLM_TPS_10bo_mladf_2x4x4_bfp16_gemm_mlp_flat_rms")
        else:
            self._set_attribute("xclbins", "stx_LLM_TPS_7bo_mladf_2x4x4_bfp16_gemm_mlp_flat_rms")

        if self.args.dynamic_attention_mask:
            self._set_attribute("fuse_qk_mha", False)
            if self.args.lora:
                self._set_attribute(
                    "dynamic_shape_list",
                    [
                        {"attention_mask_padded": 256},
                        {"attention_mask_padded": 384},
                        {"attention_mask_padded": 512},
                        {"attention_mask_padded": 1024},
                        {"attention_mask_padded": 1536},
                        {"attention_mask_padded": 2048},
                        {"attention_mask_padded": 2368},
                        {"attention_mask_padded": 3072},
                        {"attention_mask_padded": 4096},
                    ],
                )
            else:
                self._set_attribute(
                    "dynamic_shape_list",
                    [
                        {"attention_mask_padded": 256},
                        {"attention_mask_padded": 512},
                        {"attention_mask_padded": 1024},
                        {"attention_mask_padded": 2048},
                        {"attention_mask_padded": 4096},
                    ],
                )

        if self.args.ctrl_pkt is None or self.args.ctrl_pkt:
            self.args.ctrl_pkt = True

        self._override_token_fusion_attributes()

        if _conditional_true(self.args.shared_weights, self._priority_memory()):
            # disable model_hash for full fusion: rely on DD const BO sharing instead
            if self.llm_mode != LlmMode.FULL_FUSION_TOKEN:
                self._set_attribute("enable_model_hash", True)
            self._set_attribute("fuse_qk_mha", False)
        if self.args.ctrl_pkt:
            self._set_attribute("enable_ctrl_pkt", True)

    def _override_token_fusion_fuse(self) -> None:
        """
        Allow child classes to override the default token fusion fuse pass
        """
        self.pass_builder.add_step(self._path("fuse_token"))

    def _set_prefill_fusion_attributes(self) -> None:
        self._set_attribute("xclbins", "stx_llama2_mladf_2x4x4_v2_gemmbfp16_silu_mul_mha_rms_rope_unified")
        self._set_attribute("mladf_version", str(self.args.mladf_version))
        # disable model_hash for full fusion: rely on DD const BO sharing instead
        if _conditional_true(
            self.args.shared_weights, self._priority_memory() and self.llm_mode != LlmMode.FULL_FUSION_PREFILL
        ):
            self._set_attribute("enable_model_hash", True)
        # attributes["fuse_qk_mha"] = False
        self._set_attribute("buffer_factor", 2)  # TODO(varunsh): could this be auto-set?
        self._set_attribute("is_ttft", True)
        self._set_attribute("total_seq_len", self.args.max_seq_len)

        if self.args.lora:
            self._set_attribute("lora", True)
            self._set_attribute(
                # some shapes removed for memory
                "dynamic_shape_list",
                [
                    # {"sequence_length_padded": 128},
                    # {"sequence_length_padded": 256},
                    {"sequence_length_padded": 512},
                    # {"sequence_length_padded": 768},
                    {"sequence_length_padded": 1024},
                    # {"sequence_length_padded": 1792},
                    {"sequence_length_padded": 2048},
                    # {"sequence_length_padded": 2304},
                    {"sequence_length_padded": 3072},
                ],
            )
            self._set_attribute("pdi_id", 4)
        else:
            self._set_attribute(
                "dynamic_shape_list",
                [
                    {"sequence_length_padded": 128},
                    {"sequence_length_padded": 256},
                    {"sequence_length_padded": 512},
                    {"sequence_length_padded": 768},
                    {"sequence_length_padded": 1024},
                    {"sequence_length_padded": 1536},
                    {"sequence_length_padded": 1792},
                    {"sequence_length_padded": 2048},
                    {"sequence_length_padded": 2304},
                    {"sequence_length_padded": 3072},
                    {"sequence_length_padded": 4096},
                ],
            )

        self._override_prefill_fusion_attributes()

    def _set_default_jit_attribute(self, default_val: bool):
        if self.args.jit is not None:
            self._set_attribute("gpu_jit", self.args.jit)
            self._set_attribute("npu_jit", self.args.jit)
        else:
            if self.args.priority == strategy_builder.Priority.PERFORMANCE:
                val = False
            elif self._priority_memory():
                val = True
            else:
                val = default_val
            self._set_attribute("gpu_jit", val)
            self._set_attribute("npu_jit", val)

        # for NPU-only modes, explicitly disable GPU JIT to prevent errors
        if self.llm_mode in [
            LlmMode.NPU_EAGER,
            LlmMode.FULL_FUSION_PREFILL,
            LlmMode.FULL_FUSION_TOKEN,
            LlmMode.NPU_TOKEN_FUSION,
        ]:
            self._set_attribute("gpu_jit", False)

    def _override_eager_operators(self, operators: set[str]) -> set[str]:
        """
        Allow child classes to override the default eager operators

        Args:
            operators (set[str]): The default set of operators based on user arguments

        Returns:
            set[str]: The updated set of operators
        """
        return operators

    def _override_eager_attributes(self) -> None:
        """
        Allow child classes to override the default eager attributes
        """
        pass

    def _override_token_fusion_attributes(self) -> None:
        """
        Allow child classes to override the default token fusion attributes
        """
        pass

    def _override_prefill_fusion_attributes(self) -> None:
        """
        Allow child classes to override the default prefill fusion attributes
        """
        pass

    def _enable_offload_embedding(self) -> None:
        self._set_attribute("offload_embedding", True)
        self.genai_config.set_option("hybrid_opt_embedding_mmap", "1")
        self.keep_external_data = True

    def _clamp_dynamic_shape_list(self) -> None:
        try:
            dynamic_shape_list = self._get_attribute("dynamic_shape_list")
        except KeyError:
            dynamic_shape_list = []
        prune_list = []
        for idx, shape_dict in enumerate(dynamic_shape_list):
            for key, value in shape_dict.items():
                if key in ["attention_mask_padded", "sequence_length_padded"] and value > self.args.max_seq_len:
                    prune_list.append(idx)
        if prune_list:
            for key in sorted(prune_list, reverse=True):
                del dynamic_shape_list[key]

    def _set_attribute(self, key, value):
        self.base_strategy["attributes"][key] = value

    def _get_attribute(self, key: str) -> Any:
        return self.base_strategy["attributes"][key]

    def _priority_memory(self) -> bool:
        return self.args.priority == strategy_builder.Priority.MEMORY

    @staticmethod
    def _get_empty_strategy() -> dict[str, Any]:
        return {"passes": [], "attributes": {"domains": "com.ryzenai", "op_namespaces": "hybrid"}}

    @staticmethod
    def _path(key: str) -> Path:
        return Path(f"composition/llm/{key}.yaml")


CUSTOM_MODELS: dict[LlmModel, LlmModelBase] = {}


def register_models():
    global CUSTOM_MODELS
    file_path = Path(__file__)
    for path in file_path.parent.joinpath("models").rglob("*.py"):
        if path.stem.startswith("_"):
            continue
        path = path.relative_to(file_path.parent)
        model_path = path.as_posix().replace("/", ".").replace(".py", "")
        importlib.import_module(f"ryzenai_onnx_utils.strategy_builder.llm.{model_path}")
    model = importlib.import_module("ryzenai_onnx_utils.strategy_builder.llm.models")
    CUSTOM_MODELS = getattr(model, "REGISTRY", {})


def _infer_model_type(input_path: Path) -> LlmModel:
    assert CUSTOM_MODELS, "Custom models not registered yet"
    input_path_str = input_path.as_posix().lower()

    for model_type, model_class in CUSTOM_MODELS.items():
        if model_class.matches(input_path_str):
            return model_type
    return LlmModel.DEFAULT


def _confirm_model_type(model_type: LlmModel, force: bool) -> LlmModel:
    if not force:
        available_models = {m.value for m in LlmModel if m != LlmModel.AUTO}
        prompt = textwrap.dedent(f"""
            Inferring model type as {model_type}.
            Available model types are: {", ".join(sorted(available_models))}.
            Press Enter to confirm or specify a different model type: """)
        response = input(prompt).lower()
        if response:
            while response not in available_models:
                response = input("Invalid model type. Please specify one of the available models: ")
            model_type = LlmModel(response)
    return model_type


def get_builder(args: LlmArgs, llm_mode: LlmMode) -> LlmModelBase:
    if not CUSTOM_MODELS:
        register_models()
    if args.model_type == LlmModel.AUTO:
        model_type = _infer_model_type(args.input_model)
        args.model_type = _confirm_model_type(model_type, args.force)
        _logger.info("Inferred model type as %s", args.model_type)
    if args.model_type in CUSTOM_MODELS:
        return CUSTOM_MODELS[args.model_type](args, llm_mode)
    return LlmModelBase(args, llm_mode)
