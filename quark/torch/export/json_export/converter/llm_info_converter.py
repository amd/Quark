#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import quark.torch.kernel  # noqa
from quark.torch.export.config.config import JsonExporterConfig

QUANTIZE_NONE = ""
QUANTIZE_FP8_E4M3 = "fp8_e4m3"
QUANTIZE_FP8_E5M2 = "fp8_e5m2"
QUANTIZE_INT4_AWQ = "int4_awq"
QUANTIZE_W4A8_AWQ = "w4a8_awq"


class LLMInfoConverter:

    def __init__(self, model_info: Dict[str, Any], params_info: Dict[str, torch.Tensor],
                 config: JsonExporterConfig) -> None:
        self.model_info = model_info
        self.params_info = params_info
        self.quant_type = self._get_quant_type()
        self.config = config

    def _get_quant_type(self) -> str:
        q_proj_info = self.model_info["layers"][0]["self_attn"]["q_proj"]
        w_quantizer = q_proj_info.get("weight_quant_info", None)
        i_quantizer = q_proj_info.get("input_quant_info", None)
        if w_quantizer is None:
            return QUANTIZE_NONE
        else:
            w_dtype = w_quantizer.get("dtype", None)
            w_qscheme = w_quantizer.get("qscheme", None)
            if w_dtype is None:
                return QUANTIZE_NONE
            elif w_dtype == "fp8_e4m3" and w_qscheme == "per_tensor":
                return QUANTIZE_FP8_E4M3
            elif w_dtype == "fp8_e5m2" and w_qscheme == "per_tensor":
                return QUANTIZE_FP8_E5M2
            elif w_dtype == "int4" and w_qscheme == "per_group":
                if i_quantizer is None:
                    return QUANTIZE_INT4_AWQ
                elif i_quantizer.get("dtype", None) == "int8":
                    return QUANTIZE_W4A8_AWQ
                else:
                    raise ValueError("Unsupported quantization configuration to export vllm-adopt format")
            else:
                raise ValueError("Unsupported quantization configuration to export vllm-adopt format")

    def _convert_embed_info(self, info: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        embed_config = {}
        embed_config["weight"] = None if info.get("weight", None) is None else info["weight"]
        return embed_config

    def _convert_layernorm_info(self, info: Dict[str, Union[str, float, None]]) -> Dict[str, Union[str, float, None]]:
        layernorm_config = {}
        layernorm_config["weight"] = None if info.get("weight", None) is None else info["weight"]
        layernorm_config["bias"] = None if info.get("bias", None) is None else info["bias"]
        layernorm_config["layernorm_type"] = None if info.get("type", None) is None else info["type"]
        layernorm_config["eps"] = "" if info.get("eps", None) is None else info["eps"]
        return layernorm_config

    def _get_quant_scale(self, info: Dict[str, Union[str, int, None]]) -> Optional[str]:
        return None if info.get("scale", None) is None else info["scale"]

    def _to_quantized_weight(self, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        weight = weight.to("cpu")
        scale = scale.to("cpu")

        if self.quant_type == QUANTIZE_FP8_E4M3:
            return quark.torch.kernel.quant_fp8_e4m3(weight, scale).view(torch.int8)
        elif self.quant_type == QUANTIZE_FP8_E5M2:
            return quark.torch.kernel.quant_fp8_e5m2(weight, scale).view(torch.int8)
        elif self.quant_type in [QUANTIZE_INT4_AWQ, QUANTIZE_W4A8_AWQ]:

            dim_out = weight.size(0)
            dim_in = weight.size(1)
            group_size = weight.size(1) // scale.size(1)
            tensor_div = weight / scale[:, torch.arange(dim_in) // group_size]
            tensor_quant = torch.round(tensor_div).clamp(-8, 7)
            tensor_int8 = tensor_quant.to(torch.int8)
            tensor_int8 = tensor_int8.T.reshape(dim_in, dim_out // 2, 2)
            low_4_bits = tensor_int8[:, :, 0] & 0x0F
            high_4_bits = tensor_int8[:, :, 1] << 4
            tensor_int4_2 = torch.bitwise_or(low_4_bits, high_4_bits)
            return tensor_int4_2.T.contiguous()
        else:
            raise ValueError(f"Unsupported quantization format {self.quant_type}")

    def _convert_linear_info(self, info: Dict[str, Any]) -> Dict[str, Union[str, int, None]]:
        linear_config = {}
        linear_config["weight"] = None if info.get("weight", None) is None else info["weight"]
        linear_config["bias"] = None if info.get("bias", None) is None else info["bias"]
        if info.get("input_quant_info", None) is not None:
            linear_config["activation_scaling_factor"] = self._get_quant_scale(info["input_quant_info"])
        else:
            linear_config["activation_scaling_factor"] = None

        if info.get("weight_quant_info", None) is not None:
            linear_config["weight_scaling_factor"] = self._get_quant_scale(info["weight_quant_info"])
            linear_config["awq_block_size"] = info["weight_quant_info"]["group_size"]
        else:
            linear_config["weight_scaling_factor"] = None
            linear_config["awq_block_size"] = 0

        if info.get("output_quant_info", None) is not None:
            linear_config["output_scaling_factor"] = self._get_quant_scale(info["output_quant_info"])
        else:
            linear_config["output_scaling_factor"] = None

        if linear_config["weight"] is not None and linear_config["weight_scaling_factor"] is not None:
            linear_weight = self.params_info[linear_config["weight"]]
            linear_scale = self.params_info[linear_config["weight_scaling_factor"]]
            self.params_info[linear_config["weight"]] = self._to_quantized_weight(linear_weight, linear_scale)

        return linear_config

    @staticmethod
    def create_qkv_name(param_name: str) -> str:
        split_name = param_name.split(".")
        split_name[-2] = "qkv"
        qkv_name = ".".join(split_name)
        return qkv_name

    def _build_qkv_weight(self, q_info: Dict[str, Any], k_info: Dict[str, Any], v_info: Dict[str, Any]) -> str:
        q_weight_key = q_info["weight"]
        q_weight = self.params_info[q_weight_key]
        k_weight_key = k_info["weight"]
        k_weight = self.params_info[k_weight_key]
        v_weight_key = v_info["weight"]
        v_weight = self.params_info[v_weight_key]

        qkv_weight = torch.cat((q_weight, k_weight, v_weight))
        qkv_weight_name = LLMInfoConverter.create_qkv_name(q_weight_key)
        self.params_info[qkv_weight_name] = qkv_weight
        del self.params_info[q_weight_key]
        del self.params_info[k_weight_key]
        del self.params_info[v_weight_key]
        return qkv_weight_name

    def _build_qkv_bias(self, q_info: Dict[str, Any], k_info: Dict[str, Any], v_info: Dict[str, Any]) -> Optional[str]:
        q_bias_key = q_info.get("bias", None)
        k_bias_key = k_info.get("bias", None)
        v_bias_key = v_info.get("bias", None)

        if q_bias_key is None:
            assert (k_bias_key is None and v_bias_key is None), "K and V should have valid bias as Q"
            return None
        q_bias = self.params_info[q_bias_key]
        k_bias = self.params_info[k_bias_key]
        v_bias = self.params_info[v_bias_key]

        qkv_bias = torch.cat((q_bias, k_bias, v_bias))
        qkv_bias_name = LLMInfoConverter.create_qkv_name(q_bias_key)
        self.params_info[qkv_bias_name] = qkv_bias

        if q_bias_key is not None:
            del self.params_info[q_bias_key]
        if k_bias_key is not None:
            del self.params_info[k_bias_key]
        if v_bias_key is not None:
            del self.params_info[v_bias_key]
        return qkv_bias_name

    def _build_activation_scaling_factor(self, q_info: Dict[str, Any], k_info: Dict[str, Any],
                                         v_info: Dict[str, Any]) -> Optional[str]:
        if (q_info.get("input_quant_info", None) is None or k_info.get("input_quant_info", None) is None
                or v_info.get("input_quant_info", None) is None):
            return None

        q_input_scale_key = q_info["input_quant_info"]["scale"]
        k_input_scale_key = k_info["input_quant_info"]["scale"]
        v_input_scale_key = v_info["input_quant_info"]["scale"]
        q_input_scale = self.params_info[q_input_scale_key]
        k_input_scale = self.params_info[k_input_scale_key]
        v_input_scale = self.params_info[v_input_scale_key]

        qkv_input_scale = torch.stack([
            q_input_scale,
            k_input_scale,
            v_input_scale,
        ]).max(dim=0).values
        qkv_input_scale_name = LLMInfoConverter.create_qkv_name(q_input_scale_key)
        self.params_info[qkv_input_scale_name] = qkv_input_scale

        del self.params_info[q_input_scale_key]
        del self.params_info[k_input_scale_key]
        del self.params_info[v_input_scale_key]
        return qkv_input_scale_name

    def _build_output_scaling_factor(self, q_info: Dict[str, Any], k_info: Dict[str, Any],
                                     v_info: Dict[str, Any]) -> Optional[str]:
        if (q_info.get("output_quant_info", None) is None or k_info.get("output_quant_info", None) is None
                or v_info.get("output_quant_info", None) is None):
            return None

        q_output_scale_key = q_info["output_quant_info"]["scale"]
        k_output_scale_key = k_info["output_quant_info"]["scale"]
        v_output_scale_key = v_info["output_quant_info"]["scale"]
        q_output_scale = self.params_info[q_output_scale_key]
        k_output_scale = self.params_info[k_output_scale_key]
        v_output_scale = self.params_info[v_output_scale_key]

        qkv_output_scale = torch.stack([
            q_output_scale,
            k_output_scale,
            v_output_scale,
        ]).max(dim=0).values
        qkv_output_scale_name = LLMInfoConverter.create_qkv_name(q_output_scale_key)
        self.params_info[qkv_output_scale_name] = qkv_output_scale

        return qkv_output_scale_name

    def _build_weight_scaling_factor(self, q_info: Dict[str, Any], k_info: Dict[str, Any],
                                     v_info: Dict[str, Any]) -> Optional[str]:
        if (q_info.get("weight_quant_info", None) is None or k_info.get("weight_quant_info", None) is None
                or v_info.get("weight_quant_info", None) is None):
            return None

        q_weight_scale_key = q_info["weight_quant_info"]["scale"]
        k_weight_scale_key = k_info["weight_quant_info"]["scale"]
        v_weight_scale_key = v_info["weight_quant_info"]["scale"]
        q_weight_scale = self.params_info[q_weight_scale_key]
        k_weight_scale = self.params_info[k_weight_scale_key]
        v_weight_scale = self.params_info[v_weight_scale_key]

        if q_weight_scale.numel() != 1:
            qkv_weight_scale = torch.cat((
                q_weight_scale,
                k_weight_scale,
                v_weight_scale,
            ))
        else:
            qkv_weight_scale = torch.stack([
                q_weight_scale,
                k_weight_scale,
                v_weight_scale,
            ], ).max(dim=0).values

        qkv_weight_scale_name = LLMInfoConverter.create_qkv_name(q_weight_scale_key)
        self.params_info[qkv_weight_scale_name] = qkv_weight_scale

        del self.params_info[q_weight_scale_key]
        del self.params_info[k_weight_scale_key]
        del self.params_info[v_weight_scale_key]
        return qkv_weight_scale_name

    def _get_awq_block_size(self, q_info: Dict[str, Any]) -> Optional[int]:
        if q_info.get("weight_quant_info", None) is None:
            return 0
        return q_info["weight_quant_info"]["group_size"]

    def _build_qkv(self, q_info: Dict[str, Any], k_info: Dict[str, Any],
                   v_info: Dict[str, Any]) -> Dict[str, Union[str, int, None]]:
        qkv_info = {}
        qkv_info["weight"] = self._build_qkv_weight(q_info, k_info, v_info)
        qkv_info["bias"] = self._build_qkv_bias(q_info, k_info, v_info)

        qkv_info["activation_scaling_factor"] = self._build_activation_scaling_factor(q_info, k_info, v_info)
        qkv_info["weights_scaling_factor"] = self._build_weight_scaling_factor(q_info, k_info, v_info)
        qkv_info["output_scaling_factor"] = self._build_output_scaling_factor(q_info, k_info, v_info)
        qkv_info["awq_block_size"] = self._get_awq_block_size(q_info)

        if qkv_info["weight"] is not None and qkv_info["weights_scaling_factor"] is not None:
            qkv_weight = self.params_info[qkv_info["weight"]]
            qkv_scale = self.params_info[qkv_info["weights_scaling_factor"]]
            self.params_info[qkv_info["weight"]] = self._to_quantized_weight(qkv_weight, qkv_scale)

        return qkv_info

    def _get_kv_cache_scale(self, q_info: Dict[str, Any], k_info: Dict[str, Any],
                            v_info: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        qkv_output_scales = []
        qkv_output_dtypes = []
        q_output_scale_name = None
        if q_info.get("output_quant_info", None) is not None:
            q_output_scale_name = q_info["output_quant_info"]["scale"]
            qkv_output_scales.append(self.params_info[q_output_scale_name])
            qkv_output_dtypes.append(q_info["output_quant_info"]["dtype"])

        k_output_scale_name = None
        if k_info.get("output_quant_info", None) is not None:
            k_output_scale_name = k_info["output_quant_info"]["scale"]
            qkv_output_scales.append(self.params_info[k_output_scale_name])
            qkv_output_dtypes.append(k_info["output_quant_info"]["dtype"])

        v_output_scale_name = None
        if v_info.get("output_quant_info", None) is not None:
            v_output_scale_name = v_info["output_quant_info"]["scale"]
            qkv_output_scales.append(self.params_info[v_output_scale_name])
            qkv_output_dtypes.append(v_info["output_quant_info"]["dtype"])

        if not qkv_output_scales:
            return None, None

        def create_kv_cache_name(output_scale_name):
            attention_name = output_scale_name.split('.')[:-3]
            attention_name.append("kv_cache_scaling_factor")
            return ".".join(attention_name)

        kv_cache_scale_name = create_kv_cache_name(k_output_scale_name)
        kv_cache_scale = torch.stack(qkv_output_scales).max(dim=0).values
        self.params_info[kv_cache_scale_name] = kv_cache_scale

        kv_cache_dtype = None
        if "fp8_e4m3" in qkv_output_dtypes[0].lower():
            kv_cache_dtype = "fp8_e4m3"
        elif "fp8_e5m2" in qkv_output_dtypes[0].lower():
            kv_cache_dtype = "fp8_e5m2"
        elif "int8" in qkv_output_dtypes[0].lower():
            kv_cache_dtype = "INT8"

        if q_output_scale_name:
            self.params_info.pop(q_output_scale_name, None)
        if k_output_scale_name:
            self.params_info.pop(k_output_scale_name, None)
        if v_output_scale_name:
            self.params_info.pop(v_output_scale_name, None)

        return kv_cache_scale_name, kv_cache_dtype

    def _convert_attention_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        attention_info = {}
        assert info.get("q_proj", None) is not None, "Q project of self-attention module is None."
        assert info.get("k_proj", None) is not None, "K project of self-attention module is None."
        assert info.get("v_proj", None) is not None, "V project of self-attention module is None."

        attention_info["qkv"] = self._build_qkv(info["q_proj"], info["k_proj"], info["v_proj"])
        attention_info["kv_cache_scaling_factor"], attention_info["kv_cache_dtype"] = self._get_kv_cache_scale(
            info["q_proj"], info["k_proj"], info["v_proj"])

        if info.get("o_proj", None) is not None:
            attention_info["dense"] = self._convert_linear_info(info["o_proj"])
        else:
            attention_info["dense"] = None

        return attention_info

    def _convert_mlp_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        mlp_info = {}
        if info.get("gate_proj", None) is not None:
            mlp_info["fc"] = self._convert_linear_info(info["gate_proj"])
        else:
            mlp_info["fc"] = None

        if info.get("up_proj", None) is not None:
            mlp_info["gate"] = self._convert_linear_info(info["up_proj"])
        else:
            mlp_info["gate"] = None

        if info.get("down_proj", None) is not None:
            mlp_info["proj"] = self._convert_linear_info(info["down_proj"])
        else:
            mlp_info["proj"] = None

        if info.get("act_fn", None) is not None:
            mlp_info["hidden_act"] = info["act_fn"]["type"].lower()
        else:
            mlp_info["hidden_act"] = None

        return mlp_info

    def _convert_layers_info(self, info_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        layers_info = []
        for info in info_list:
            layer_info = {}
            layer_info["quantization"] = self.quant_type
            layer_info["decoder_type"] = None if info.get("decoder_type", None) is None else info["decoder_type"]

            if info.get("input_layernorm", None) is not None:
                layer_info["input_layernorm"] = self._convert_layernorm_info(info["input_layernorm"])
            else:
                layer_info["input_layernorm"] = None

            if info.get("mlp_layernorm", None) is not None:
                layer_info["mlp_layernorm"] = self._convert_layernorm_info(info["mlp_layernorm"])
            else:
                layer_info["mlp_layernorm"] = None

            if info.get("self_attn", None) is not None:
                layer_info["attention"] = self._convert_attention_info(info["self_attn"])
            else:
                layer_info["attention"] = None

            if info.get("post_attention_layernorm", None) is not None:
                layer_info["post_layernorm"] = self._convert_layernorm_info(info["post_attention_layernorm"])
            else:
                layer_info["post_layernorm"] = None

            if info.get("mlp", None) is not None:
                layer_info["mlp"] = self._convert_mlp_info(info["mlp"])
            else:
                layer_info["mlp"] = None

            layer_info["num_attention_heads"] = None if info.get("num_attention_heads",
                                                                 None) is None else info["num_attention_heads"]
            layer_info["attention_head_size"] = None if info.get("attention_head_size",
                                                                 None) is None else info["attention_head_size"]
            layer_info["num_kv_heads"] = None if info.get("num_kv_heads", None) is None else info["num_kv_heads"]
            layer_info["max_position_embeddings"] = None if info.get("max_position_embeddings",
                                                                     None) is None else info["max_position_embeddings"]

            layer_info["rotary_pct"] = info["rotary_pct"]
            layer_info["parallel_attention"] = info["parallel_attention"]
            layer_info["apply_residual_connection_post_layernorm"] = info["apply_residual_connection_post_layernorm"]
            layer_info["use_cache"] = info["use_cache"]
            layer_info["rope_ratio"] = info["rope_ratio"]
            layer_info["seq_length"] = info["seq_length"]
            layers_info.append(layer_info)

        return layers_info

    def convert(self) -> Dict[str, Any]:
        model_dict = {}
        model_dict["version"] = self.model_info["version"]
        model_dict["quantization"] = self.quant_type
        model_dict["dtype"] = self.model_info["dtype"]
        model_dict["vocab_size"] = self.model_info["vocab_size"]
        model_dict["rank"] = 0 if self.model_info.get("rank", None) is None else self.model_info["rank"]
        model_dict["tensor_parallel"] = 1 \
            if self.model_info.get("tensor_parallel", None) is None \
            else self.model_info["tensor_parallel"]
        model_dict["pipeline_parallel"] = 1 \
            if self.model_info.get("pipeline_parallel", None) is None \
            else self.model_info["pipeline_parallel"]

        model_dict["vocab_embedding"] = None \
            if self.model_info.get("tokens_embed", None) is None \
            else self._convert_embed_info(self.model_info["tokens_embed"])

        model_dict["positional_embedding"] = None \
            if self.model_info.get("positional_embed", None) is None \
            else self._convert_embed_info(self.model_info["positional_embed"])

        model_dict["ln_embed"] = None \
            if self.model_info.get("ln_embed", None) is None \
            else self._convert_embed_info(self.model_info["ln_embed"])

        model_dict["layers"] = None \
            if self.model_info.get("layers", None) is None \
            else self._convert_layers_info(self.model_info["layers"])

        model_dict["final_layernorm"] = None \
            if self.model_info.get("final_layernorm", None) is None \
            else self._convert_layernorm_info(self.model_info["final_layernorm"])

        model_dict["ln_f"] = None \
            if self.model_info.get("final_norm", None) is None \
            else self._convert_layernorm_info(self.model_info["final_norm"])

        model_dict["lm_head"] = None \
            if self.model_info.get("lm_head", None) is None \
            else self._convert_linear_info(self.model_info["lm_head"])

        model_dict["share_embedding_table"] = False \
            if self.model_info.get("embed_weight_share", None) is None \
            else self.model_info["embed_weight_share"]

        return model_dict
