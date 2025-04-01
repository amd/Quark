#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2023-2024 The ggml authors
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path
from hashlib import sha256
from typing import Any, Callable, ContextManager, Iterator, Sequence, TypeVar, cast, Dict

import torch
from .tensor_convert import convert_to_gguf
from quark.shares.utils.log import ScreenLogger, log_errors
from quark.shares.utils.import_utils import is_transformers_available

if is_transformers_available():
    from transformers import AutoTokenizer

logger = ScreenLogger(__name__)

try:
    import gguf  # type: ignore
except ImportError as e:
    logger.exception(str(e))
    raise ImportError("please install gguf==0.6.0")

from .utils import permute
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class QuantSpec(object):

    def __init__(self,
                 tensor_name: str,
                 tensor: torch.Tensor,
                 scales: torch.Tensor | None = None,
                 zero_points: torch.Tensor | None = None,
                 quant_type: gguf.GGMLQuantizationType = None) -> None:
        self.tensor_name = tensor_name
        self.tensor = tensor
        self.scales = scales
        self.zero_points = zero_points
        self.quant_type = quant_type


class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


AnyModel = TypeVar("AnyModel", bound="type[ModelWriter]")


class ModelWriter(ABC):
    _model_classes: dict[str, type[ModelWriter]] = {}

    def __init__(self,
                 model_name: str,
                 json_path: Path,
                 safetensor_path: Path,
                 tokenizer_dir: Path,
                 fname_out: Path,
                 is_big_endian: bool = False,
                 use_temp_file: bool = False):
        self.model_name = model_name
        with open(json_path, 'r') as f:
            self._model_json: Dict[str, int | str | Any] = json.load(f)
        self.safetensor_path = safetensor_path
        self.tokenizer_dir = tokenizer_dir
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.use_temp_file = use_temp_file
        self.hparams = self.load_hparams()
        self.gguf_writer = gguf.GGUFWriter(fname_out,
                                           gguf.MODEL_ARCH_NAMES[self.model_arch],
                                           endianess=self.endianess,
                                           use_temp_file=self.use_temp_file)
        self.block_count = self.find_hparam(["n_layers", "num_hidden_layers", "n_layer"])

    def parse_quant_info(self) -> Dict[str, Dict[str, str | int]]:
        quant_info = {}

        def traverse_model_json(model_json: Dict[str, Any]) -> None:
            for k, v in model_json.items():
                if not isinstance(v, dict):
                    continue
                if "weight_quant" in v:
                    quant_info[v["weight"]] = v["weight_quant"]
                else:
                    traverse_model_json(v)

        traverse_model_json(self._model_json)
        return quant_info

    def get_quant_type_from_weight_quant(self, weight_quant: Dict[str, str | int]) -> gguf.GGMLQuantizationType:
        if weight_quant["dtype"] == "uint4" and weight_quant["qscheme"] == "per_group" and weight_quant[
                "group_size"] == 32:
            return gguf.GGMLQuantizationType.Q4_1
        else:
            raise Exception("Unsupported quant spec")

    def is_quant_spec_complete(self, quant_spec: QuantSpec) -> bool:
        return quant_spec.tensor is not None and quant_spec.scales is not None and quant_spec.zero_points is not None

    @property
    @abstractmethod
    def model_arch(self) -> gguf.MODEL_ARCH:
        pass

    def find_hparam(self, keys: Sequence[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in self.hparams), None)
        if key is not None:
            return self.hparams[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def set_vocab(self) -> None:
        pass

    def get_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        ctx: ContextManager[Any]
        from safetensors import safe_open
        ctx = cast(ContextManager[Any], safe_open(self.safetensor_path, framework="pt", device="cpu"))

        with ctx as model_part:
            for name in model_part.keys():
                data = model_part.get_tensor(name)
                yield name, data

    def set_gguf_parameters(self) -> None:
        self.gguf_writer.add_name(self.model_name)
        self.gguf_writer.add_block_count(self.block_count)

        if (n_ctx := self.find_hparam(["max_position_embeddings", "n_ctx"], optional=True)) is not None:
            self.gguf_writer.add_context_length(n_ctx)
            logger.info(f"gguf: context length = {n_ctx}")

        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        self.gguf_writer.add_embedding_length(n_embd)
        logger.info(f"gguf: embedding length = {n_embd}")

        if (n_ff := self.find_hparam(["intermediate_size", "n_inner"], optional=True)) is not None:
            self.gguf_writer.add_feed_forward_length(n_ff)
            logger.info(f"gguf: feed forward length = {n_ff}")

        n_head = self.find_hparam(["num_attention_heads", "n_head"])
        self.gguf_writer.add_head_count(n_head)
        logger.info(f"gguf: head count = {n_head}")

        if (n_head_kv := self.hparams.get("num_key_value_heads")) is not None:
            self.gguf_writer.add_head_count_kv(n_head_kv)
            logger.info(f"gguf: key-value head count = {n_head_kv}")

        if (rope_theta := self.hparams.get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base(rope_theta)
            logger.info(f"gguf: rope theta = {rope_theta}")
        if (f_rms_eps := self.hparams.get("rms_norm_eps")) is not None:
            self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
            logger.info(f"gguf: rms norm epsilon = {f_rms_eps}")
        if (f_norm_eps := self.find_hparam(["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"],
                                           optional=True)) is not None:
            self.gguf_writer.add_layer_norm_eps(f_norm_eps)
            logger.info(f"gguf: layer norm epsilon = {f_norm_eps}")
        if (n_experts := self.hparams.get("num_local_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
            logger.info(f"gguf: expert count = {n_experts}")
        if (n_experts_used := self.hparams.get("num_experts_per_tok")) is not None:
            self.gguf_writer.add_expert_used_count(n_experts_used)
            logger.info(f"gguf: experts used count = {n_experts_used}")

        self.gguf_writer.add_file_type(gguf.GGMLQuantizationType.F32)

    @abstractmethod
    def write_tensors(self) -> None:
        pass

    def write(self) -> None:
        self.write_tensors()
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file()
        self.gguf_writer.close()

    def write_vocab(self) -> None:
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.close()

    @staticmethod
    def count_model_parts(dir_model: Path, prefix: str) -> int:
        num_parts = 0
        for filename in os.listdir(dir_model):
            if filename.endswith(prefix):
                num_parts += 1

        return num_parts

    def load_hparams(self) -> Dict[str, int | str | Any]:
        return self._model_json["config"]  # type: ignore[return-value]

    @classmethod
    def register(cls, *names: str) -> Callable[[type[ModelWriter]], type[ModelWriter]]:
        assert names

        def func(modelcls: type[ModelWriter]) -> type[ModelWriter]:
            for name in names:
                cls._model_classes[name] = modelcls
            return modelcls

        return func

    @classmethod
    @log_errors
    def from_model_architecture(cls: type[ModelWriter], arch: str) -> type[ModelWriter]:
        try:
            return cls._model_classes[arch]
        except KeyError as e:
            raise NotImplementedError(f'Architecture {arch!r} not supported!')

    # used for GPT-2 BPE and WordPiece vocabs
    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []

        if not is_transformers_available():  # pragma: no cover
            raise ImportError(
                "The `transformers` library is required to run `ModelWriter.get_vocab_base`, but the library was not found in the current environment. Please install Transformers library (`pip install transformers`)."
            )
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        vocab_size = int(self.hparams.get("vocab_size", len(tokenizer.vocab)))
        assert max(tokenizer.vocab.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.USER_DEFINED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                if tokenizer.added_tokens_decoder[i].special:
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        return tokens, toktypes, tokpre

    # NOTE: this function is generated by convert-hf-to-gguf-update.py
    #       do not modify it manually!
    # ref:  https://github.com/ggerganov/llama.cpp/pull/6920
    def get_vocab_base_pre(self, tokenizer) -> str:  # type: ignore[no-untyped-def]
        # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that
        # is specific for the BPE pre-tokenizer used by the model
        # we will use this unique identifier to write a "tokenizer.ggml.pre" entry in the GGUF file which we can
        # use in llama.cpp to implement the same pre-tokenizer

        chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

        chktok = tokenizer.encode(chktxt)
        chkhsh = sha256(str(chktok).encode()).hexdigest()
        res = None

        # NOTE: if you get an error here, you need to update the convert-hf-to-gguf-update.py script
        #       or pull the latest version of the model from Huggingface
        #       don't edit the hashes manually!
        if chkhsh == "0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5":
            # ref: https://huggingface.co/meta-llama/Meta-Llama-3-8B
            res = "llama-bpe"
        if chkhsh == "049ecf7629871e3041641907f3de7c733e4dbfdc736f57d882ba0b0845599754":
            # ref: https://huggingface.co/deepseek-ai/deepseek-llm-7b-base
            res = "deepseek-llm"
        if chkhsh == "347715f544604f9118bb75ed199f68779f423cabb20db6de6f31b908d04d7821":
            # ref: https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base
            res = "deepseek-coder"
        if chkhsh == "8aeee3860c56296a157a1fe2fad249ec40aa59b1bb5709f4ade11c4e6fe652ed":
            # ref: https://huggingface.co/tiiuae/falcon-7b
            res = "falcon"
        if chkhsh == "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f":
            # ref: https://huggingface.co/BAAI/bge-small-en-v1.5
            res = "bert-bge"
        if chkhsh == "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166":
            # ref: https://huggingface.co/mosaicml/mpt-7b
            res = "mpt"
        if chkhsh == "35d91631860c815f952d711435f48d356ebac988362536bed955d43bfa436e34":
            # ref: https://huggingface.co/bigcode/starcoder2-3b
            res = "starcoder"
        if chkhsh == "3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454":
            # ref: https://huggingface.co/openai-community/gpt2
            res = "gpt-2"
        if chkhsh == "6221ad2852e85ce96f791f476e0b390cf9b474c9e3d1362f53a24a06dc8220ff":
            # ref: https://huggingface.co/smallcloudai/Refact-1_6-base
            res = "refact"
        if chkhsh == "9c2227e4dd922002fb81bde4fc02b0483ca4f12911410dee2255e4987644e3f8":
            # ref: https://huggingface.co/CohereForAI/c4ai-command-r-v01
            res = "command-r"

        if res is None:
            logger.warning("\n")
            logger.warning("**************************************************************************************")
            logger.warning("** WARNING: The BPE pre-tokenizer was not recognized!")
            logger.warning("**          There are 2 possible reasons for this:")
            logger.warning("**          - the model has not been added to convert-hf-to-gguf-update.py yet")
            logger.warning("**          - the pre-tokenization config has changed upstream")
            logger.warning(
                "**          Check your model files and convert-hf-to-gguf-update.py and update them accordingly.")
            logger.warning("** ref:     https://github.com/ggerganov/llama.cpp/pull/6920")
            logger.warning("**")
            logger.warning(f"** chkhsh:  {chkhsh}")
            logger.warning("**************************************************************************************")
            logger.warning("\n")
            raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")

        return res

    def _set_vocab_sentencepiece(self) -> None:
        from sentencepiece import SentencePieceProcessor  # type: ignore

        tokenizer_path = self.tokenizer_dir / 'tokenizer.model'

        tokens: list[bytes | str] = []
        scores: list[float] = []
        toktypes: list[int] = []

        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"File not found: {tokenizer_path}")

        tokenizer = SentencePieceProcessor(str(tokenizer_path))
        vocab_size = int(self.hparams.get('vocab_size', tokenizer.vocab_size()))

        for token_id in range(tokenizer.vocab_size()):
            piece = tokenizer.id_to_piece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.get_score(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.is_unknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.is_control(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.is_unused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.is_byte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        added_tokens_file = self.tokenizer_dir / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)

                for key in added_tokens_json:
                    key = key.encode("utf-8")
                    if key not in tokens:
                        tokens.append(key)
                        scores.append(-1000.0)
                        toktypes.append(SentencePieceTokenTypes.USER_DEFINED)

        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            for i in range(1, pad_count + 1):
                tokens.append(f"[PAD{i}]")
                scores.append(-1000.0)
                toktypes.append(SentencePieceTokenTypes.UNUSED)

        assert len(tokens) == vocab_size

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_string("tokenizer.ggml.pre", "default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.tokenizer_dir, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)


@ModelWriter.register("LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM")
class LlamaModelWriter(ModelWriter):
    model_arch = gguf.MODEL_ARCH.LLAMA

    def set_vocab(self) -> None:
        self._set_vocab_sentencepiece()

        # Apply to CodeLlama only (and ignore for Llama 3 with a vocab size of 128256)
        if self.hparams.get("vocab_size", 32000) == 32016:
            special_vocab = gguf.SpecialVocab(self.tokenizer_dir,
                                              load_merges=False,
                                              special_token_types=['prefix', 'suffix', 'middle', 'eot'])
            special_vocab._set_special_token("prefix", 32007)
            special_vocab._set_special_token("suffix", 32008)
            special_vocab._set_special_token("middle", 32009)
            special_vocab._set_special_token("eot", 32010)
            special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self) -> None:
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.vocab_size", hparams["vocab_size"])
        self.gguf_writer.add_rope_dimension_count(hparams["hidden_size"] //  # type: ignore
                                                  hparams["num_attention_heads"])

        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:  # type: ignore
            if self.hparams["rope_scaling"].get("type") == "linear":  # type: ignore
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_writer.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])  # type: ignore

    # Same as super class, but permuting q_proj, k_proj
    def write_tensors(self) -> None:
        block_count = int(
            self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer"))))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        n_head = int(self.hparams.get("num_attention_heads"))  # type: ignore
        n_kv_head = int(self.hparams.get("num_key_value_heads"))  # type: ignore

        quant_info = self.parse_quant_info()
        quant_spec_map: Dict[str, QuantSpec] = {}
        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
                continue
            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            n_dims = len(data_torch.shape)
            if n_dims == 1:
                data_torch = data_torch.to(torch.float32)

            raw_shape = None
            raw_dtype = None
            if name in quant_info:
                quant_spec_map[name] = QuantSpec(tensor_name=name,
                                                 tensor=data_torch,
                                                 scales=None,
                                                 zero_points=None,
                                                 quant_type=self.get_quant_type_from_weight_quant(quant_info[name]))
                continue
            if name.endswith("_scale"):
                tensor_name = name[:-len("_scale")]
                assert tensor_name in quant_spec_map, f"tensor : {tensor_name} has to be in quant_spec_map"
                quant_spec = quant_spec_map[tensor_name]
                quant_spec.scales = data_torch
                if self.is_quant_spec_complete(quant_spec):
                    data_torch = convert_to_gguf(
                        inpt=quant_spec.tensor,
                        scale=quant_spec.scales,
                        zero_point=quant_spec.zero_points,  # type: ignore
                        gguf_type=quant_spec.quant_type)
                    raw_shape = quant_spec.tensor.shape
                    raw_dtype = quant_spec.quant_type
                    name = quant_spec.tensor_name
                else:
                    continue
            if name.endswith("_zero_point"):
                tensor_name = name[:-len("_zero_point")]
                assert tensor_name in quant_spec_map, f"tensor : {tensor_name} has to be in quant_spec_map"
                quant_spec_map[tensor_name].zero_points = data_torch
                if self.is_quant_spec_complete(quant_spec):
                    data_torch = convert_to_gguf(
                        inpt=quant_spec.tensor,
                        scale=quant_spec.scales,  # type: ignore
                        zero_point=quant_spec.zero_points,  # type: ignore
                        gguf_type=quant_spec.quant_type)
                    raw_shape = quant_spec.tensor.shape
                    raw_dtype = quant_spec.quant_type
                    name = quant_spec.tensor_name
                else:
                    continue

            if name.endswith("q_proj.weight"):
                data_torch = permute(data_torch, n_head, n_head)
            if name.endswith("k_proj.weight"):
                data_torch = permute(data_torch, n_head, n_kv_head)

            data = data_torch.numpy()

            data = data.squeeze()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                raise ValueError(f"Can not map tensor {name!r}")

            self.gguf_writer.add_tensor(new_name, data, raw_shape=raw_shape, raw_dtype=raw_dtype)
