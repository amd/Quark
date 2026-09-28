#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""CPU-only tests for generic speculative-decoding domain manifests."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from quark.experimental.speculative_decoding.data import (
    DomainSource,
    build_domain_records,
    load_domain_manifest,
    normalized_sha256,
    parse_domain_manifest,
    provenance_license_summary,
)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_manifest(tmp_path: Path) -> Path:
    _write_jsonl(
        tmp_path / "general.jsonl",
        [
            {"prompt": "Ａ   shared\tprompt"},
            {"instruction": "general only"},
            {"prompt": "not selected by the domain cap"},
        ],
    )
    _write_jsonl(
        tmp_path / "code.jsonl",
        [
            {"conversations": [{"from": "human", "value": "A shared prompt"}]},
            {"prompt": "write a parser"},
            {"prompt": "debug the parser"},
        ],
    )
    manifest = {
        "version": 1,
        "seed": 17,
        "max_samples_per_domain": 2,
        "splits": {"train": 0.5, "validation": 0.5},
        "domains": {
            "general_instruction": {"sources": [{"type": "jsonl", "path": "general.jsonl", "license": "MIT"}]},
            "code": {"sources": [{"type": "jsonl", "path": "code.jsonl", "license": "Apache-2.0"}]},
        },
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def test_manifest_build_is_normalized_deduplicated_capped_and_deterministic(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)

    first = build_domain_records(manifest_path)
    second = build_domain_records(load_domain_manifest(manifest_path))

    assert first == second
    assert len(first) == 4
    assert Counter(row["domain"] for row in first) == {
        "general_instruction": 2,
        "code": 2,
    }
    assert len({row["id"] for row in first}) == len(first)
    assert sum(row["id"] == normalized_sha256("A shared prompt") for row in first) == 1
    prompts = [row["conversations"][0]["content"] for row in first]
    assert "A shared prompt" in prompts
    assert all("\t" not in prompt and "  " not in prompt for prompt in prompts)

    summary = provenance_license_summary(first)
    assert summary["licenses"] == {"Apache-2.0": 2, "MIT": 2}
    assert summary["total_records"] == 4


def test_a_manifest_may_name_its_own_domains(tmp_path: Path) -> None:
    """Domains group and label records, so adding one is a manifest edit.

    They used to be checked against a fixed list, which meant a workload the
    suggested names did not cover needed a library change to describe itself.
    """
    _write_jsonl(tmp_path / "legal.jsonl", [{"prompt": "summarise this clause"}])
    _write_jsonl(tmp_path / "code.jsonl", [{"prompt": "write a parser"}])
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "max_samples_per_domain": 10,
                "domains": {
                    "legal_review": {"sources": [{"type": "jsonl", "path": "legal.jsonl", "license": "MIT"}]},
                    "code": {"sources": [{"type": "jsonl", "path": "code.jsonl", "license": "MIT"}]},
                },
            }
        ),
        encoding="utf-8",
    )

    records = build_domain_records(load_domain_manifest(manifest))

    assert {row["domain"] for row in records} == {"legal_review", "code"}
    assert provenance_license_summary(records)["domains"] == {"legal_review": 1, "code": 1}


def test_domain_cap_samples_across_sources_and_follows_the_seed(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "first.jsonl", [{"prompt": f"first source prompt {index}"} for index in range(20)])
    _write_jsonl(tmp_path / "second.jsonl", [{"prompt": f"second source prompt {index}"} for index in range(20)])

    def manifest_for(seed: int) -> Path:
        path = tmp_path / f"manifest-{seed}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "seed": seed,
                    "max_samples_per_domain": 10,
                    "domains": {
                        "code": {
                            "sources": [
                                {"type": "jsonl", "name": "first", "path": "first.jsonl", "license": "MIT"},
                                {"type": "jsonl", "name": "second", "path": "second.jsonl", "license": "MIT"},
                            ]
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    records = build_domain_records(manifest_for(0))
    sources = Counter(row["provenance"]["source"] for row in records)

    assert len(records) == 10
    # A cap applied while reading would let the first source claim all ten.
    assert sources["first"] > 0
    assert sources["second"] > 0

    assert {row["id"] for row in build_domain_records(manifest_for(0))} == {row["id"] for row in records}
    assert {row["id"] for row in build_domain_records(manifest_for(1))} != {row["id"] for row in records}


def test_domain_cap_does_not_block_later_domains_from_dropped_prompts(tmp_path: Path) -> None:
    shared = [{"prompt": f"shared prompt {index}"} for index in range(6)]
    _write_jsonl(tmp_path / "general.jsonl", shared)
    _write_jsonl(tmp_path / "code.jsonl", shared)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "seed": 3,
                "max_samples_per_domain": 2,
                "domains": {
                    "general_instruction": {"sources": [{"type": "jsonl", "path": "general.jsonl", "license": "MIT"}]},
                    "code": {"sources": [{"type": "jsonl", "path": "code.jsonl", "license": "MIT"}]},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    records = build_domain_records(manifest)

    assert Counter(row["domain"] for row in records) == {"general_instruction": 2, "code": 2}
    assert len({row["id"] for row in records}) == 4


def test_hf_descriptor_uses_injected_loader_without_network(tmp_path: Path) -> None:
    parsed = parse_domain_manifest(
        {
            "version": 1,
            "max_samples_per_domain": 2,
            "domains": {
                "question_answering": {
                    "sources": [
                        {
                            "type": "hf",
                            "dataset": "public/example",
                            "subset": "plain",
                            "split": "train",
                            "revision": "public-revision",
                            "license": "CC-BY-4.0",
                        }
                    ]
                }
            },
        },
        base_dir=tmp_path,
    )
    calls: list[DomainSource] = []

    def local_fixture_loader(source: DomainSource) -> Iterable[Mapping[str, Any]]:
        calls.append(source)
        return [{"question": "first"}, {"question": "second"}]

    rows = build_domain_records(parsed, hf_loader=local_fixture_loader)

    assert len(calls) == 1
    assert calls[0].dataset == "public/example"
    assert [row["provenance"]["source_type"] for row in rows] == ["hf", "hf"]
    assert all(row["provenance"]["revision"] == "public-revision" for row in rows)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (
            {"version": 2, "domains": {}},
            "version",
        ),
        (
            {
                "version": 1,
                "domains": {"private mix": {"sources": [{"type": "jsonl", "path": "a.jsonl", "license": "MIT"}]}},
            },
            "may contain only letters",
        ),
        (
            {
                "version": 1,
                "domains": {
                    "code": {
                        "sources": [{"type": "jsonl", "path": "code.jsonl"}],
                    }
                },
            },
            "license",
        ),
        (
            {
                "version": 1,
                "domains": {
                    "question_answering": {
                        "max_samples": 1,
                        "sources": [{"type": "hf", "license": "CC-BY-4.0"}],
                    }
                },
            },
            "dataset",
        ),
        (
            {
                "version": 1,
                "splits": {"train": 0.8, "validation": 0.3},
                "domains": {
                    "code": {
                        "max_samples": 1,
                        "sources": [{"type": "jsonl", "path": "code.jsonl", "license": "MIT"}],
                    }
                },
            },
            "sum to 1",
        ),
    ],
)
def test_manifest_parsing_rejects_invalid_values(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_domain_manifest(value)
