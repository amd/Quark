#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Iterable, Protocol, runtime_checkable
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

import torch
try:
    import gguf  # type: ignore
except ImportError as e:
    logger.error(f"{e}")
    raise ImportError(f"Please install gguf==0.6.0. {str(e)}")

ADDED_TOKENS_FILE = 'added_tokens.json'
FAST_TOKENIZER_FILE = 'tokenizer.json'


@runtime_checkable
class BaseVocab(Protocol):
    tokenizer_model: ClassVar[str]
    name: ClassVar[str]


class NoVocab(BaseVocab):
    tokenizer_model = "no_vocab"
    name = "no_vocab"

    def __repr__(self) -> str:
        return "<NoVocab for a model without integrated vocabulary>"


@runtime_checkable
class Vocab(BaseVocab, Protocol):
    vocab_size: int
    added_tokens_dict: dict[str, int]
    added_tokens_list: list[str]
    fname_tokenizer: Path

    def __init__(self, base_path: Path):
        ...

    def all_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        ...


class BpeVocab(Vocab):
    tokenizer_model = "gpt2"
    name = "bpe"

    def __init__(self, base_path: Path):
        added_tokens: dict[str, int] = {}

        if (fname_tokenizer := base_path / 'vocab.json').exists():
            # "slow" tokenizer
            with open(fname_tokenizer, encoding="utf-8") as f:
                self.vocab = json.load(f)

            try:
                # FIXME: Verify that added tokens here _cannot_ overlap with the main vocab.
                with open(base_path / ADDED_TOKENS_FILE, encoding="utf-8") as f:
                    added_tokens = json.load(f)
            except FileNotFoundError as e:
                logger.exception(str(
                    e))  # TODO: refactor. logger.exception should not be called without actually raising an exception.
                pass
        else:
            # "fast" tokenizer
            fname_tokenizer = base_path / FAST_TOKENIZER_FILE

            # if this fails, FileNotFoundError propagates to caller
            with open(fname_tokenizer, encoding="utf-8") as f:
                tokenizer_json = json.load(f)

            tokenizer_model: dict[str, Any] = tokenizer_json['model']
            if (tokenizer_model['type'] != 'BPE' or tokenizer_model.get('byte_fallback', False)
                    or tokenizer_json['decoder']['type'] != 'ByteLevel'):
                raise FileNotFoundError('Cannot find GPT-2 BPE tokenizer')

            self.vocab = tokenizer_model["vocab"]

            if (added := tokenizer_json.get('added_tokens')) is not None:
                # Added tokens here can be duplicates of the main vocabulary.
                added_tokens = {item['content']: item['id'] for item in added if item['content'] not in self.vocab}

        vocab_size = len(self.vocab)
        expected_ids = list(range(vocab_size, vocab_size + len(added_tokens)))
        actual_ids = sorted(added_tokens.values())
        if expected_ids != actual_ids:
            expected_end_id = vocab_size + len(actual_ids) - 1
            raise ValueError(f"Expected the {len(actual_ids)} added token ID(s) to be sequential in the range "
                             f"{vocab_size} - {expected_end_id}; got {actual_ids}")

        items = sorted(added_tokens.items(), key=lambda text_idx: text_idx[1])
        self.added_tokens_dict = added_tokens
        self.added_tokens_list = [text for (text, idx) in items]
        self.vocab_size_base = vocab_size
        self.vocab_size = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer = fname_tokenizer

    def bpe_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        reverse_vocab = {id: encoded_tok for encoded_tok, id in self.vocab.items()}

        for i, _ in enumerate(self.vocab):
            yield reverse_vocab[i], 0.0, gguf.TokenType.NORMAL

    def added_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        for text in self.added_tokens_list:
            score = -1000.0
            yield text.encode("utf-8"), score, gguf.TokenType.CONTROL

    def all_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        yield from self.bpe_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"<BpeVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"


def permute(weights: torch.Tensor, n_head: int, n_head_kv: int) -> torch.Tensor:
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2,
                            *weights.shape[1:]).swapaxes(1, 2).reshape(weights.shape))


def inverse_permute(weights: torch.Tensor, n_head: int, n_head_kv: int) -> torch.Tensor:
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, weights.shape[0] // n_head // 2, 2,
                            *weights.shape[1:]).swapaxes(1, 2).reshape(weights.shape))
