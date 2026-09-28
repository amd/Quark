#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ._serialization import StrictSchema, atomic_write_json, load_json_object, sha256_json
from .errors import PlanSelectionError, SchemaValidationError
from .mixed_precision_strategy import MixedPrecisionStrategy


class TokenPurpose(StrEnum):
    CALIBRATION = "calibration"
    PPL = "ppl"


@dataclass(frozen=True, slots=True)
class TokenDataset(StrictSchema):
    purpose: TokenPurpose
    dataset: str
    split: str
    revision: str | None
    tokenizer_id: str
    tokenizer_fingerprint: str
    sampling_seed: int | None
    sequences: tuple[tuple[int, ...], ...]
    token_hash: str

    def __post_init__(self) -> None:
        if not self.dataset or not self.split or not self.tokenizer_id or not self.sequences:
            raise SchemaValidationError("Token dataset metadata and sequences must not be empty.")
        if self.revision == "":
            raise SchemaValidationError("Token dataset revision must not be empty.")
        if not self.tokenizer_fingerprint.startswith("sha256:"):
            raise SchemaValidationError("tokenizer_fingerprint must be a SHA-256 value.")
        lengths = {len(sequence) for sequence in self.sequences}
        if len(lengths) != 1 or next(iter(lengths)) <= 1:
            raise SchemaValidationError("Token sequences must have one common length greater than one.")
        if any(type(token) is not int or token < 0 for sequence in self.sequences for token in sequence):
            raise SchemaValidationError("Token ids must be non-negative integers.")
        if self.sampling_seed is not None and (type(self.sampling_seed) is not int or self.sampling_seed < 0):
            raise SchemaValidationError("Token sampling seed must be non-negative.")
        if self.purpose is TokenPurpose.CALIBRATION and self.sampling_seed is None:
            raise SchemaValidationError("Calibration token manifests must record their sampling seed.")
        if self.purpose is TokenPurpose.PPL and self.sampling_seed is not None:
            raise SchemaValidationError("PPL token manifests must not record an unused sampling seed.")
        if self.token_hash != sha256_json(self.semantic_dict()):
            raise SchemaValidationError("Token dataset hash does not match its content.")

    def semantic_dict(self) -> dict[str, object]:
        return {
            "purpose": self.purpose,
            "dataset": self.dataset,
            "split": self.split,
            "revision": self.revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "sampling_seed": self.sampling_seed,
            "sequences": self.sequences,
        }

    @classmethod
    def create(
        cls,
        *,
        purpose: TokenPurpose,
        dataset: str,
        split: str,
        revision: str | None,
        tokenizer_id: str,
        sequences: tuple[tuple[int, ...], ...],
        tokenizer_fingerprint: str,
        sampling_seed: int | None = None,
    ) -> TokenDataset:
        semantic = {
            "purpose": purpose,
            "dataset": dataset,
            "split": split,
            "revision": revision,
            "tokenizer_id": tokenizer_id,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "sampling_seed": sampling_seed,
            "sequences": sequences,
        }
        return cls(token_hash=sha256_json(semantic), **semantic)

    @classmethod
    def load(cls, path: str | Path) -> TokenDataset:
        return cls.from_dict(load_json_object(path))

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())

    def to_dataloader(self, device: str | torch.device) -> DataLoader[dict[str, torch.Tensor]]:
        samples = [
            {"input_ids": torch.tensor(sequence, dtype=torch.long, device=device)} for sequence in self.sequences
        ]
        return DataLoader(samples, batch_size=1, shuffle=False)


def _load_hf_dataset(path: str, name: str | None, split: str, revision: str | None) -> Any:
    try:
        load_dataset = importlib.import_module("datasets").load_dataset
    except (ImportError, AttributeError) as exc:
        raise PlanSelectionError("The datasets package is required to materialize calibration and PPL tokens.") from exc
    kwargs: dict[str, object] = {"split": split, "streaming": True}
    if revision is not None:
        kwargs["revision"] = revision
    return load_dataset(path, name, **kwargs) if name is not None else load_dataset(path, **kwargs)


def fingerprint_tokenizer(tokenizer: Any) -> str:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocab = get_vocab() if callable(get_vocab) else {}
    if not isinstance(vocab, Mapping):
        raise PlanSelectionError("tokenizer.get_vocab() must return a mapping.")
    return sha256_json(
        {
            "vocab": dict(sorted((str(token), int(index)) for token, index in vocab.items())),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        }
    )


def validate_token_datasets(
    calibration_tokens: TokenDataset,
    ppl_tokens: TokenDataset,
    strategy: MixedPrecisionStrategy,
    *,
    tokenizer_fingerprint: str | None = None,
) -> None:
    """Validate materialized token manifests against every Strategy sampling input."""
    calibration = strategy.calibration
    quality_gate = strategy.plan_selection.quality_gate
    if (
        calibration_tokens.purpose is not TokenPurpose.CALIBRATION
        or calibration_tokens.dataset != "mit-han-lab/pile-val-backup"
        or calibration_tokens.split != "validation"
        or calibration_tokens.revision != calibration.revision
        or calibration_tokens.sampling_seed != strategy.seed
        or len(calibration_tokens.sequences) != calibration.num_samples
        or len(calibration_tokens.sequences[0]) != calibration.max_length
    ):
        raise PlanSelectionError(
            "Calibration token manifest does not match Strategy sampling inputs; regenerate the manifest."
        )
    if (
        ppl_tokens.purpose is not TokenPurpose.PPL
        or ppl_tokens.dataset != "Salesforce/wikitext/wikitext-2-raw-v1"
        or ppl_tokens.split != "test"
        or ppl_tokens.revision != quality_gate.revision
        or ppl_tokens.sampling_seed is not None
        or len(ppl_tokens.sequences) != quality_gate.num_chunks
        or len(ppl_tokens.sequences[0]) != quality_gate.max_length
    ):
        raise PlanSelectionError("PPL token manifest does not match Strategy sampling inputs; regenerate the manifest.")
    if (
        calibration_tokens.tokenizer_id != ppl_tokens.tokenizer_id
        or calibration_tokens.tokenizer_fingerprint != ppl_tokens.tokenizer_fingerprint
    ):
        raise PlanSelectionError("Calibration and PPL tokens must use the same tokenizer.")
    if tokenizer_fingerprint is not None and calibration_tokens.tokenizer_fingerprint != tokenizer_fingerprint:
        raise PlanSelectionError("Cached token manifests do not match the current tokenizer.")


def _materialize_sequences(
    rows: Any,
    tokenizer: Any,
    *,
    count: int,
    length: int,
) -> tuple[tuple[int, ...], ...]:
    required = count * length
    tokens: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    for row in rows:
        text = str(row["text"]).strip()
        if not text:
            continue
        encoded = tokenizer.encode(text, add_special_tokens=False)
        tokens.extend(int(token) for token in encoded)
        if isinstance(eos_token_id, int):
            tokens.append(eos_token_id)
        if len(tokens) >= required:
            break
    if len(tokens) < required:
        raise PlanSelectionError(f"Dataset produced {len(tokens)} tokens, but {required} are required.")
    return tuple(tuple(tokens[index * length : (index + 1) * length]) for index in range(count))


def materialize_calibration_tokens(
    tokenizer: Any,
    tokenizer_id: str,
    strategy: MixedPrecisionStrategy,
) -> TokenDataset:
    calibration = strategy.calibration
    dataset = _load_hf_dataset(
        "mit-han-lab/pile-val-backup",
        None,
        "validation",
        calibration.revision,
    )
    dataset = dataset.shuffle(seed=strategy.seed)
    sequences = _materialize_sequences(
        dataset,
        tokenizer,
        count=calibration.num_samples,
        length=calibration.max_length,
    )
    return TokenDataset.create(
        purpose=TokenPurpose.CALIBRATION,
        dataset="mit-han-lab/pile-val-backup",
        split="validation",
        revision=calibration.revision,
        tokenizer_id=tokenizer_id,
        tokenizer_fingerprint=fingerprint_tokenizer(tokenizer),
        sampling_seed=strategy.seed,
        sequences=sequences,
    )


def materialize_ppl_tokens(
    tokenizer: Any,
    tokenizer_id: str,
    strategy: MixedPrecisionStrategy,
) -> TokenDataset:
    quality_gate = strategy.plan_selection.quality_gate
    dataset = _load_hf_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        "test",
        quality_gate.revision,
    )
    sequences = _materialize_sequences(
        dataset,
        tokenizer,
        count=quality_gate.num_chunks,
        length=quality_gate.max_length,
    )
    return TokenDataset.create(
        purpose=TokenPurpose.PPL,
        dataset="Salesforce/wikitext/wikitext-2-raw-v1",
        split="test",
        revision=quality_gate.revision,
        tokenizer_id=tokenizer_id,
        tokenizer_fingerprint=fingerprint_tokenizer(tokenizer),
        sampling_seed=None,
        sequences=sequences,
    )


__all__ = [
    "TokenDataset",
    "TokenPurpose",
    "fingerprint_tokenizer",
    "materialize_calibration_tokens",
    "materialize_ppl_tokens",
    "validate_token_datasets",
]
