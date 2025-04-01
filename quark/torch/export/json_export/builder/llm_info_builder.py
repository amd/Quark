#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from abc import abstractmethod
from collections import OrderedDict
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from typing import no_type_check
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize
from quark.torch.export.config.config import JsonExporterConfig
from quark.torch.export.utils import find_patterns_groups
from .llm_info import (CURRENT_VERSION, EmbeddingType, LayerNormType, QuantInfo, EmbeddingInfo, LayerNormInfo,
                       LinearInfo, ActInfo, AttentionInfo, MLPInfo, DecoderInfo, ModelInfo)


class LLMInfoBuilder:

    def __init__(self, model: nn.Module, model_type: str, model_dtype: torch.dtype, config: JsonExporterConfig) -> None:
        self.model = model
        self.decoder_type = model_type
        self.model_dtype = model_dtype
        self.config = config
        self.vocab_size = 0
        self.linear_infos_dict: Dict[str, LinearInfo] = {}
        self.linear_weight_merged_groups: List[List[str]] = []
        if hasattr(self.model, "config") and hasattr(self.model.config, "vocab_size"):
            self.vocab_size = self.model.config.vocab_size
        else:
            raise ValueError("Could not find vocab size config in model")

    @abstractmethod
    def get_flatten_layers(self) -> Dict[str, Any]:
        pass

    def is_embed(self, module: nn.Module) -> bool:
        return 'embedding' in module.__class__.__name__.lower()

    def build_embed_info(self, name: str, module: nn.Module) -> EmbeddingInfo:
        assert self.is_embed(module)
        embed_info = EmbeddingInfo(name=name)
        if "rotary" in module.__class__.__name__.lower():
            embed_info.type = EmbeddingType.rotary.value
            embed_info.weight = None
        else:
            embed_info.type = EmbeddingType.default.value
            embed_info.weight = module.weight.detach().cpu()
        return embed_info

    def is_layernorm(self, module: nn.Module) -> bool:
        module_type_name = module.__class__.__name__
        return "layernorm" in module_type_name.lower() or "rmsnorm" in module_type_name.lower()

    def build_layernorm_info(self, name: str, module: nn.Module) -> LayerNormInfo:
        assert self.is_layernorm(module)
        layernorm_info = LayerNormInfo(name=name, weight=module.weight.detach().cpu())

        if "rmsnorm" in module.__class__.__name__.lower():
            layernorm_info.type = LayerNormType.rms.value

        if hasattr(module, "eps"):
            layernorm_info.eps = module.eps
        elif hasattr(module, "variance_epsilon"):
            layernorm_info.eps = module.variance_epsilon

        if hasattr(module, "bias") and module.bias is not None:
            layernorm_info.bias = module.bias.detach()
        return layernorm_info

    @staticmethod
    def get_scale(quantizer: ScaledFakeQuantize) -> Optional[torch.Tensor]:
        """Returns scale from the quantizer as torch.Tensor."""
        if quantizer is None:
            return None

        if hasattr(quantizer, 'scale') and quantizer.scale is not None and quantizer.scale.numel() > 0:
            return quantizer.scale.detach().cpu()

        return None

    @staticmethod
    def get_zero_point(quantizer: ScaledFakeQuantize) -> Optional[torch.Tensor]:
        """Returns zero point from the quantizer as torch.Tensor."""
        if quantizer is None:
            return None

        if hasattr(quantizer, 'zero_point') and quantizer.zero_point is not None and quantizer.zero_point.numel() > 0:
            return quantizer.zero_point.detach().cpu()

        return None

    def is_linear(self, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear)

    def build_linear_info(self, name: str, module: nn.Module) -> LinearInfo:
        assert self.is_linear(module)
        linear_info = LinearInfo(name=name, weight=module.weight.detach().cpu())
        if hasattr(module, "bias") and module.bias is not None:
            linear_info.bias = module.bias.detach()

        if hasattr(module, "_input_quantizer") and module._input_quantizer is not None:
            quantizer = module._input_quantizer
            quant_info = QuantInfo(name=name + ".input_quant",
                                   dtype=quantizer.dtype.name,
                                   qscheme=quantizer.qscheme.name,
                                   ch_axis=quantizer.ch_axis)
            quant_info.scale = LLMInfoBuilder.get_scale(quantizer)
            quant_info.zero_point = LLMInfoBuilder.get_zero_point(quantizer)
            linear_info.input_quant_info = quant_info

        if hasattr(module, "_weight_quantizer") and module._weight_quantizer is not None:
            quantizer = module._weight_quantizer
            quant_info = QuantInfo(name=name + ".weight_quant",
                                   dtype=quantizer.dtype.name,
                                   qscheme=quantizer.qscheme.name,
                                   ch_axis=quantizer.ch_axis)
            quant_info.scale = LLMInfoBuilder.get_scale(quantizer)
            quant_info.zero_point = LLMInfoBuilder.get_zero_point(quantizer)
            quant_info.group_size = quantizer.group_size if quantizer.group_size is not None else 0
            linear_info.weight_quant_info = quant_info

        if hasattr(module, "_output_quantizer") and module._output_quantizer is not None:
            quantizer = module._output_quantizer
            quant_info = QuantInfo(name=name + ".output_quant",
                                   dtype=quantizer.dtype.name,
                                   qscheme=quantizer.qscheme.name,
                                   ch_axis=quantizer.ch_axis)
            quant_info.scale = LLMInfoBuilder.get_scale(quantizer)
            quant_info.zero_point = LLMInfoBuilder.get_zero_point(quantizer)
            linear_info.output_quant_info = quant_info

        self.linear_infos_dict[name] = linear_info

        return linear_info

    def is_attention(self, module: nn.Module) -> bool:
        return "attention" in module.__class__.__name__.lower()

    def build_attention_info(self, name: str, module: nn.Module) -> AttentionInfo:
        assert self.is_attention(module)
        attention_info = AttentionInfo(name=name)
        for key, mod in module.named_children():
            key_name = name + "." + key
            if self.is_linear(mod):
                if key == "q_proj":
                    attention_info.q_proj = self.build_linear_info(key_name, mod)
                elif key == "k_proj":
                    attention_info.k_proj = self.build_linear_info(key_name, mod)
                elif key == "v_proj":
                    attention_info.v_proj = self.build_linear_info(key_name, mod)
                elif key == "o_proj":
                    attention_info.o_proj = self.build_linear_info(key_name, mod)
                else:
                    raise ValueError(f"Not support {key} key in llama model")
            elif self.is_embed(mod):
                attention_info.emb = self.build_embed_info(key_name, mod)
        return attention_info

    def is_mlp(self, module: nn.Module) -> bool:
        return "mlp" in module.__class__.__name__.lower()

    @abstractmethod
    def build_mlp_info(self, name: str, module: nn.Module) -> MLPInfo:
        pass

    def is_decoder(self, module: nn.Module) -> bool:
        return "decoder" in module.__class__.__name__.lower()

    def build_decoder_info(self, name: str, module: nn.Module) -> DecoderInfo:
        assert self.is_decoder(module)
        decoder_info = DecoderInfo(name=name, decoder_type=self.decoder_type)
        for key, mod in module.named_children():
            key_name = name + "." + key
            if self.is_attention(mod):
                decoder_info.self_attn = self.build_attention_info(key_name, mod)
            elif self.is_mlp(mod):
                decoder_info.mlp = self.build_mlp_info(key_name, mod)
            elif self.is_layernorm(mod):
                if "input" in key.lower():
                    decoder_info.input_layernorm = self.build_layernorm_info(key_name, mod)
                elif "post" in key.lower():
                    decoder_info.post_attention_layernorm = self.build_layernorm_info(key_name, mod)

        decoder_info.num_attention_heads = self.model.config.num_attention_heads
        if (hasattr(decoder_info, "self_attn") and decoder_info.self_attn is not None
                and hasattr(decoder_info.self_attn, "q_proj") and decoder_info.self_attn.q_proj is not None
                and hasattr(decoder_info.self_attn.q_proj, "weight")
                and decoder_info.self_attn.q_proj.weight is not None):
            decoder_info.attention_head_size = decoder_info.self_attn.q_proj.weight.shape[
                0] // decoder_info.num_attention_heads
        decoder_info.num_kv_heads = self.model.config.num_key_value_heads
        decoder_info.max_position_embeddings = self.model.config.max_position_embeddings

        return decoder_info

    def is_decoder_list(self, module: nn.Module) -> bool:
        return (module.__class__.__name__ == "ModuleList")

    def build_decoder_list(self, name: str, module: nn.Module) -> List[DecoderInfo]:
        assert self.is_decoder_list(module)
        decoder_list = []
        index = 0
        for mod in list(module):
            if self.is_decoder(mod):
                mod_name = name + "." + str(index)
                decoder_list.append(self.build_decoder_info(mod_name, mod))
                index = index + 1
        return decoder_list

    @no_type_check
    def merge_scaling_factor(self, linear_group: List[LinearInfo]) -> None:
        weight_quant_or_not = all(linear_info.weight_quant_info is not None for linear_info in linear_group)
        if not weight_quant_or_not:
            return
        per_tensor_or_not = all(linear_info.weight_quant_info.qscheme == "per_tensor" for linear_info in linear_group)
        if not per_tensor_or_not:
            return
        symmetric_quant_or_not = all(
            torch.all(linear_info.weight_quant_info.zero_point == 0) for linear_info in linear_group)
        if not symmetric_quant_or_not:
            return

        weight_quant_name = linear_group[0].weight_quant_info.name
        weight_scale_list = [linear_info.weight_quant_info.scale for linear_info in linear_group]

        group_weight_scale = torch.stack(weight_scale_list, ).max(dim=0).values

        for linear_info in linear_group:
            # inear_info.weight_quant_info.name = weight_quant_name
            linear_info.weight_quant_info.scale = group_weight_scale

        output_quant_or_not = all(linear_info.output_quant_info is not None for linear_info in linear_group)
        if not output_quant_or_not:
            return
        per_tensor_or_not = all(linear_info.output_quant_info.qscheme == "per_tensor" for linear_info in linear_group)
        if not per_tensor_or_not:
            return
        symmetric_quant_or_not = all(
            torch.all(linear_info.output_quant_info.zero_point == 0) for linear_info in linear_group)
        if not symmetric_quant_or_not:
            return

        output_quant_name = linear_group[0].output_quant_info.name
        output_scale_list = [linear_info.output_quant_info.scale for linear_info in linear_group]

        group_output_scale = torch.stack(output_scale_list, ).max(dim=0).values

        for linear_info in linear_group:
            # linear_info.output_quant_info.name = output_quant_name
            linear_info.output_quant_info.scale = group_output_scale

    def virtual_merge_weight_matrix(self) -> None:
        if self.config.weight_merge_groups is None or len(self.config.weight_merge_groups) == 0:
            return
        linear_keys = list(self.linear_infos_dict.keys())
        self.linear_weight_merged_groups = find_patterns_groups(self.config.weight_merge_groups, linear_keys)
        for key_group in self.linear_weight_merged_groups:
            linear_merge_group = [self.linear_infos_dict[key] for key in key_group]
            self.merge_scaling_factor(linear_merge_group)

    def build_model_info(self) -> ModelInfo:
        model_info = ModelInfo(version=CURRENT_VERSION,
                               dtype=str(self.model_dtype).split('.')[1],
                               vocab_size=self.vocab_size)
        flatten_layers = self.get_flatten_layers()

        for name, module in flatten_layers.items():
            if self.is_embed(module):
                if "tokens" in name:
                    model_info.tokens_embed = self.build_embed_info(name, module)
                elif "positional" in name:
                    model_info.positional_embed = self.build_embed_info(name, module)
            elif self.is_layernorm(module):
                model_info.final_norm = self.build_layernorm_info(name, module)
            elif self.is_linear(module):
                model_info.lm_head = self.build_linear_info(name, module)
            elif self.is_decoder_list(module):
                model_info.layers = self.build_decoder_list(name, module)
        self.virtual_merge_weight_matrix()

        return model_info


class LlamaModelInfoBuilder(LLMInfoBuilder):

    def get_flatten_layers(self) -> Dict[str, nn.Module]:
        flatten_layers = OrderedDict()
        for name, module in self.model.named_children():
            if self.is_linear(module):
                flatten_layers[name] = module
            else:
                for n, mod in module.named_children():
                    key = name + "." + n
                    flatten_layers[key] = mod
        return flatten_layers

    def build_mlp_info(self, name: str, module: nn.Module) -> MLPInfo:
        assert self.is_mlp(module)
        mlp_info = MLPInfo(name=name)
        for key, mod in module.named_children():
            key_name = name + "." + key
            if self.is_linear(mod):
                if key == "gate_proj":
                    mlp_info.gate_proj = self.build_linear_info(key_name, mod)
                elif key == "up_proj":
                    mlp_info.up_proj = self.build_linear_info(key_name, mod)
                elif key == "down_proj":
                    mlp_info.down_proj = self.build_linear_info(key_name, mod)
                else:
                    raise ValueError(f"Not support {key} key in llama model")
            else:
                if key == "act_fn":
                    mlp_info.act_fn = ActInfo(name=key_name, type=mod.__class__.__name__)
        return mlp_info


def create_llm_builder(model: nn.Module, model_type: str, model_dtype: torch.dtype,
                       config: JsonExporterConfig) -> LLMInfoBuilder:
    if "llama" in model_type:
        return LlamaModelInfoBuilder(model, model_type, model_dtype, config)
    else:
        raise ValueError(
            f"Not support {model_type} type model when exporting vllm-adopted json-safetensors model currently")
