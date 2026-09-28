#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Deterministic preparation of generic-domain prompt data.

The manifest format intentionally describes data, not a target model. Sources
may be local JSONL files or public Hugging Face datasets. Every accepted prompt
is normalized, identified by a SHA256 digest, and deduplicated globally before
domain caps, deterministic ordering, and stratified splits are applied.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import yaml

# Model-agnostic domain names accepted by a version-1 manifest.
# Suggested names, not a permitted set: a domain is a grouping key and a
# provenance label, so requiring a library edit to add one bought little.
SUGGESTED_DOMAIN_NAMES: Final[tuple[str, ...]] = (
    "general_instruction",
    "code",
    "math_reasoning",
    "question_answering",
    "multilingual",
    "long_context",
    "structured_tool_use",
)

DOMAIN_MANIFEST_VERSION: Final = 1
DEFAULT_PROMPT_FIELDS: Final[tuple[str, ...]] = ("prompt", "instruction", "question", "text")

SourceType = Literal["jsonl", "hf"]
HFDatasetLoader = Callable[["DomainSource"], Iterable[Mapping[str, Any]]]

_WHITESPACE = re.compile(r"\s+")
_DOMAIN_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
# `torchspec_runner` reads the held-out ratio under either spelling and uses
# the remainder for training; nothing else is consumed.
_SPLIT_NAMES = frozenset({"train", "eval", "validation"})
_TOP_LEVEL_KEYS = frozenset({"version", "seed", "max_samples_per_domain", "splits", "domains"})
_DOMAIN_KEYS = frozenset({"sources", "max_samples", "prompt_fields"})
_SOURCE_KEYS = frozenset(
    {
        "type",
        "path",
        "dataset",
        "subset",
        "split",
        "revision",
        "name",
        "license",
        "max_samples",
        "prompt_fields",
    }
)


@dataclass(frozen=True)
class DomainSource:
    """One local JSONL or Hugging Face source in a domain manifest."""

    source_type: SourceType
    license: str
    path: str | None = None
    dataset: str | None = None
    subset: str | None = None
    split: str = "train"
    revision: str | None = None
    name: str | None = None
    max_samples: int | None = None
    prompt_fields: tuple[str, ...] | None = None

    @property
    def identity(self) -> str:
        """Return a stable, relocatable source label for provenance."""
        if self.name:
            return self.name
        if self.dataset:
            return self.dataset
        if self.path:
            digest = hashlib.sha256(self.path.encode("utf-8")).hexdigest()[:12]
            return f"local-jsonl-{digest}"
        return ""


@dataclass(frozen=True)
class DomainDefinition:
    """Sources and limits for one generic domain."""

    name: str
    sources: tuple[DomainSource, ...]
    max_samples: int | None = None
    prompt_fields: tuple[str, ...] = DEFAULT_PROMPT_FIELDS


@dataclass(frozen=True)
class DomainManifest:
    """Validated version-1 generic domain manifest."""

    version: int
    seed: int
    domains: dict[str, DomainDefinition]
    max_samples_per_domain: int | None
    splits: dict[str, float]
    base_dir: Path


def normalize_text(text: str) -> str:
    """Return the stable Unicode and whitespace form used for output and dedup."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def normalized_sha256(text: str) -> str:
    """Return the hex SHA256 digest of :func:`normalize_text`."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _as_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} keys must be strings")
    return cast(Mapping[str, Any], value)


def _check_keys(value: Mapping[str, Any], allowed: frozenset[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


def _optional_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value.strip()


def _integer(value: object, context: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{context} must be greater than zero")
    return value


def _prompt_fields(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty list of field names")
    fields: list[str] = []
    for field in value:
        parsed = _optional_string(field, context)
        if parsed is None:
            raise ValueError(f"{context} entries must be non-empty strings")
        fields.append(parsed)
    return tuple(fields)


def _parse_source(value: object, context: str) -> DomainSource:
    source = _as_mapping(value, context)
    _check_keys(source, _SOURCE_KEYS, context)

    source_type = source.get("type")
    if source_type not in ("jsonl", "hf"):
        raise ValueError(f"{context}.type must be 'jsonl' or 'hf'")
    license_name = _optional_string(source.get("license"), f"{context}.license")
    if license_name is None:
        raise ValueError(f"{context}.license is required")

    path = _optional_string(source.get("path"), f"{context}.path")
    dataset = _optional_string(source.get("dataset"), f"{context}.dataset")
    if source_type == "jsonl":
        if path is None:
            raise ValueError(f"{context}.path is required for a JSONL source")
        if dataset is not None:
            raise ValueError(f"{context}.dataset is only valid for an HF source")
    else:
        if dataset is None:
            raise ValueError(f"{context}.dataset is required for an HF source")
        if path is not None:
            raise ValueError(f"{context}.path is only valid for a JSONL source")

    max_samples_raw = source.get("max_samples")
    max_samples = (
        None if max_samples_raw is None else _integer(max_samples_raw, f"{context}.max_samples", positive=True)
    )
    fields_raw = source.get("prompt_fields")
    fields = None if fields_raw is None else _prompt_fields(fields_raw, f"{context}.prompt_fields")
    split = _optional_string(source.get("split"), f"{context}.split") or "train"

    return DomainSource(
        source_type=cast(SourceType, source_type),
        license=license_name,
        path=path,
        dataset=dataset,
        subset=_optional_string(source.get("subset"), f"{context}.subset"),
        split=split,
        revision=_optional_string(source.get("revision"), f"{context}.revision"),
        name=_optional_string(source.get("name"), f"{context}.name"),
        max_samples=max_samples,
        prompt_fields=fields,
    )


def _parse_splits(value: object) -> dict[str, float]:
    if value is None:
        return {"train": 1.0}
    splits = _as_mapping(value, "manifest.splits")
    if not splits:
        raise ValueError("manifest.splits must not be empty")

    parsed: dict[str, float] = {}
    for name, ratio in splits.items():
        # Only these are read downstream; accepting arbitrary names meant a
        # third split validated cleanly and was then dropped without a word.
        if name not in _SPLIT_NAMES:
            raise ValueError(f"manifest.splits.{name} is not used; choose from {', '.join(sorted(_SPLIT_NAMES))}")
        if isinstance(ratio, bool) or not isinstance(ratio, int | float):
            raise ValueError(f"manifest.splits.{name} must be a number")
        parsed[name] = float(ratio)
        if not math.isfinite(parsed[name]) or parsed[name] <= 0:
            raise ValueError(f"manifest.splits.{name} must be greater than zero")
    if not math.isclose(sum(parsed.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("manifest.splits ratios must sum to 1")
    return parsed


def parse_domain_manifest(value: object, *, base_dir: str | Path = ".") -> DomainManifest:
    """Validate a decoded YAML mapping and return its typed representation."""
    manifest = _as_mapping(value, "domain manifest")
    _check_keys(manifest, _TOP_LEVEL_KEYS, "domain manifest")

    version = _integer(manifest.get("version"), "manifest.version")
    if version != DOMAIN_MANIFEST_VERSION:
        raise ValueError(f"manifest.version must be {DOMAIN_MANIFEST_VERSION}")
    seed = _integer(manifest.get("seed", 0), "manifest.seed")

    global_cap_raw = manifest.get("max_samples_per_domain")
    global_cap = (
        None if global_cap_raw is None else _integer(global_cap_raw, "manifest.max_samples_per_domain", positive=True)
    )

    domain_values = _as_mapping(manifest.get("domains"), "manifest.domains")
    if not domain_values:
        raise ValueError("manifest.domains must not be empty")

    domains: dict[str, DomainDefinition] = {}
    # A domain is a grouping key and a provenance label, so the name only has to
    # be usable as one. Ordering is by name rather than by manifest order so
    # that two manifests listing the same domains produce the same records.
    for name in sorted(domain_values):
        if not isinstance(name, str) or _DOMAIN_NAME.fullmatch(name) is None:
            raise ValueError("manifest.domains names may contain only letters, numbers, '_' and '-'")
        context = f"manifest.domains.{name}"
        domain = _as_mapping(domain_values[name], context)
        _check_keys(domain, _DOMAIN_KEYS, context)
        source_values = domain.get("sources")
        if not isinstance(source_values, list) or not source_values:
            raise ValueError(f"{context}.sources must be a non-empty list")
        sources = tuple(
            _parse_source(source, f"{context}.sources[{index}]") for index, source in enumerate(source_values)
        )
        max_samples_raw = domain.get("max_samples")
        max_samples = (
            None if max_samples_raw is None else _integer(max_samples_raw, f"{context}.max_samples", positive=True)
        )
        fields_raw = domain.get("prompt_fields")
        fields = DEFAULT_PROMPT_FIELDS if fields_raw is None else _prompt_fields(fields_raw, f"{context}.prompt_fields")
        domains[name] = DomainDefinition(
            name=name,
            sources=sources,
            max_samples=max_samples,
            prompt_fields=fields,
        )

    return DomainManifest(
        version=version,
        seed=seed,
        domains=domains,
        max_samples_per_domain=global_cap,
        splits=_parse_splits(manifest.get("splits")),
        base_dir=Path(base_dir).resolve(),
    )


def load_domain_manifest(path: str | Path) -> DomainManifest:
    """Load and validate a generic-domain YAML manifest."""
    manifest_path = Path(path)
    try:
        value = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{manifest_path}: invalid YAML") from exc
    if value is None:
        raise ValueError(f"{manifest_path}: domain manifest is empty")
    return parse_domain_manifest(value, base_dir=manifest_path.parent)


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"domain source not found: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            yield cast(Mapping[str, Any], value)


def _default_hf_loader(source: DomainSource) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Hugging Face sources require the optional 'datasets' package") from exc

    kwargs: dict[str, Any] = {"split": source.split, "streaming": True}
    if source.revision is not None:
        kwargs["revision"] = source.revision
    dataset = load_dataset(source.dataset, source.subset, **kwargs)
    return cast(Iterable[Mapping[str, Any]], dataset)


def _source_rows(
    source: DomainSource,
    *,
    base_dir: Path,
    hf_loader: HFDatasetLoader | None,
) -> Iterable[Mapping[str, Any]]:
    if source.source_type == "hf":
        return (hf_loader or _default_hf_loader)(source)
    assert source.path is not None
    raw_path = Path(source.path).expanduser()
    path = raw_path if raw_path.is_absolute() else base_dir / raw_path
    return _iter_jsonl(path)


def _first_user_prompt(row: Mapping[str, Any], prompt_fields: Sequence[str]) -> str | None:
    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = message.get("role") or message.get("from")
            if str(role).lower() not in ("user", "human"):
                continue
            content = message.get("content")
            if content is None:
                content = message.get("value")
            if content is not None and normalize_text(str(content)):
                return normalize_text(str(content))

    for field in prompt_fields:
        content = row.get(field)
        if content is not None and normalize_text(str(content)):
            return normalize_text(str(content))
    return None


def _record_order_key(record: Mapping[str, Any], *, seed: int, namespace: str) -> tuple[str, str]:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    identity = str(record.get("id") or canonical)
    key = hashlib.sha256(f"{seed}\0{namespace}\0{identity}".encode()).hexdigest()
    return key, canonical


def deterministic_shuffle(
    records: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    namespace: str = "",
) -> list[dict[str, Any]]:
    """Return a stable SHA256-keyed ordering without global random state."""
    copied = [dict(record) for record in records]
    return sorted(copied, key=lambda record: _record_order_key(record, seed=seed, namespace=namespace))


def merge_domain_records(
    records_by_domain: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    seed: int,
    namespace: str = "merge",
) -> list[dict[str, Any]]:
    """Shuffle each domain deterministically, then merge it round-robin.

    Domains are visited in name order, so the merge depends on which domains are
    present rather than on the order a caller happened to build the mapping in.
    """
    ordered = sorted(records_by_domain)
    shuffled = {
        domain: deterministic_shuffle(records_by_domain[domain], seed=seed, namespace=f"{namespace}:{domain}")
        for domain in ordered
    }
    max_length = max((len(records) for records in shuffled.values()), default=0)
    merged: list[dict[str, Any]] = []
    for index in range(max_length):
        for domain in ordered:
            records = shuffled[domain]
            if index < len(records):
                merged.append(records[index])
    return merged


def _effective_domain_cap(
    manifest: DomainManifest,
    domain: DomainDefinition,
    override: int | None,
) -> int:
    configured = domain.max_samples or manifest.max_samples_per_domain
    if configured is None and override is None:
        raise ValueError(
            f"domain {domain.name!r} has no cap; set max_samples, max_samples_per_domain, "
            "or the per_domain_cap argument"
        )
    if configured is None:
        assert override is not None
        return override
    return configured if override is None else min(configured, override)


def _provenance(source: DomainSource, digest: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "source": source.identity,
        "source_type": source.source_type,
        "license": source.license,
        "content_sha256": digest,
    }
    if source.source_type == "hf":
        value["split"] = source.split
        if source.subset is not None:
            value["subset"] = source.subset
        if source.revision is not None:
            value["revision"] = source.revision
    return value


def build_domain_records(
    manifest: DomainManifest | str | Path,
    *,
    per_domain_cap: int | None = None,
    hf_loader: HFDatasetLoader | None = None,
) -> list[dict[str, Any]]:
    """Read, normalize, globally deduplicate, cap, shuffle, and merge prompts.

    ``hf_loader`` is injectable so callers can provide an authenticated loader
    and unit tests can exercise HF descriptors without making network calls.
    """
    parsed = load_domain_manifest(manifest) if isinstance(manifest, str | Path) else manifest
    if per_domain_cap is not None:
        per_domain_cap = _integer(per_domain_cap, "per_domain_cap", positive=True)

    # Only selected prompts join the global dedup set, so a prompt a domain
    # reads but drops at its cap stays available to a later domain.
    selected_ids: set[str] = set()
    records_by_domain: dict[str, list[dict[str, Any]]] = {}
    for domain in parsed.domains.values():
        cap = _effective_domain_cap(parsed, domain, per_domain_cap)
        candidates: list[dict[str, Any]] = []
        seen = set(selected_ids)
        for source in domain.sources:
            # Read up to a full cap per source, then let the deterministic
            # shuffle below decide membership. Truncating during the read would
            # let the first source consume the whole cap and make the manifest
            # seed unable to influence which prompts are selected.
            source_budget = cap if source.max_samples is None else min(cap, source.max_samples)
            accepted_from_source = 0
            prompt_fields = source.prompt_fields or domain.prompt_fields
            for row in _source_rows(source, base_dir=parsed.base_dir, hf_loader=hf_loader):
                if accepted_from_source >= source_budget:
                    break
                prompt = _first_user_prompt(row, prompt_fields)
                if prompt is None:
                    continue
                digest = normalized_sha256(prompt)
                if digest in seen:
                    continue
                seen.add(digest)
                candidates.append(
                    {
                        "id": digest,
                        "domain": domain.name,
                        "conversations": [{"role": "user", "content": prompt}],
                        "provenance": _provenance(source, digest),
                    }
                )
                accepted_from_source += 1
        records = deterministic_shuffle(candidates, seed=parsed.seed, namespace=f"domain-cap:{domain.name}")[:cap]
        selected_ids.update(record["id"] for record in records)
        records_by_domain[domain.name] = records

    return merge_domain_records(records_by_domain, seed=parsed.seed)


def provenance_license_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize record counts by generic domain, source, and declared license."""
    domains: Counter[str] = Counter()
    licenses: Counter[str] = Counter()
    sources: Counter[tuple[str, str, str]] = Counter()
    total = 0

    for record in records:
        total += 1
        domain = str(record.get("domain", ""))
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("every record must contain provenance")
        source = str(provenance.get("source", ""))
        source_type = str(provenance.get("source_type", ""))
        license_name = str(provenance.get("license", ""))
        if not source or not source_type or not license_name:
            raise ValueError("record provenance must include source, source_type, and license")
        domains[domain] += 1
        licenses[license_name] += 1
        sources[(source, source_type, license_name)] += 1

    return {
        "total_records": total,
        "domains": dict(sorted(domains.items())),
        "licenses": dict(sorted(licenses.items())),
        "sources": [
            {
                "source": source,
                "source_type": source_type,
                "license": license_name,
                "records": count,
            }
            for (source, source_type, license_name), count in sorted(sources.items())
        ],
    }


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    """Write deterministic UTF-8 JSONL with sorted object keys."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    return output_path


__all__ = [
    "SUGGESTED_DOMAIN_NAMES",
    "DomainManifest",
    "DomainSource",
    "build_domain_records",
    "load_domain_manifest",
    "normalized_sha256",
    "parse_domain_manifest",
    "provenance_license_summary",
    "write_jsonl",
]
