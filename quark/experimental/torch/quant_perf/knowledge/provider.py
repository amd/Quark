#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import yaml

from .types import (
    KnowledgeContext,
    KnowledgeMatch,
    KnowledgeRecord,
)


class KnowledgeProvider(Protocol):
    def query(self, context: KnowledgeContext) -> list[KnowledgeMatch]:
        """Return knowledge matches for the supplied context."""
        ...


class _StrictSafeLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _reject_duplicate_keys(
    loader: _StrictSafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    """Construct a YAML mapping while rejecting duplicate keys.

    :param loader: Active YAML loader.
    :param node: YAML mapping node.
    :param deep: Whether nested objects should be constructed deeply.
    :return: Constructed mapping.
    :raises ValueError: If the mapping contains a duplicate key.
    """
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _reject_duplicate_keys,
)

_DOMAINS = {"quantization", "repair", "kernel_optimization"}
_KINDS = {
    "reference",
    "diagnostic_playbook",
    "validated_playbook",
    "verified_repair_recipe",
    "quantization_rule",
    "kernel_principle",
    "known_issue",
    "policy",
}
_STATUSES = {"advisory", "validated", "verified", "contradicted", "retired"}


def _as_tuple(value: Any) -> tuple[str, ...]:
    """Normalize an optional scalar or iterable to a string tuple.

    :param value: Optional scalar or iterable.
    :return: Normalized tuple of strings.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _matches(expected: Any, actual: Any) -> bool:
    """Return whether an actual value satisfies an expected value set.

    :param expected: Expected scalar or value collection.
    :param actual: Actual value.
    :return: Whether the value matches.
    """
    choices = _as_tuple(expected)
    if not choices or "*" in choices:
        return True
    actual_text = str(actual or "").strip().lower()
    return any(str(choice).strip().lower() == actual_text for choice in choices)


def _validate_embedded_values(value: Any, *, path: Path) -> None:
    """Reject unsafe references in nested knowledge data.

    :param value: Nested value to inspect.
    :param path: Source file used in validation errors.
    :raises ValueError: If an executable patch or absolute path is found.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if "patch" in key_text or "diff" in key_text:
                raise ValueError(f"executable patch reference in {path}")
            _validate_embedded_values(item, path=path)
        return
    if isinstance(value, list | tuple):
        for item in value:
            _validate_embedded_values(item, path=path)
        return
    if not isinstance(value, str):
        return
    lowered = value.lower()
    if ".patch" in lowered or ".diff" in lowered:
        raise ValueError(f"executable patch reference in {path}")
    if os.path.isabs(value):
        raise ValueError(f"absolute local path in {path}")


def _validate_record_data(raw: dict[str, Any], *, path: Path) -> None:
    """Validate one curated knowledge record.

    :param raw: Parsed record data.
    :param path: Source file used in validation errors.
    :raises ValueError: If controlled fields or embedded values are invalid.
    """
    if str(raw.get("domain") or "") not in _DOMAINS:
        raise ValueError(f"unsupported knowledge domain in {path}")
    if str(raw.get("kind") or "") not in _KINDS:
        raise ValueError(f"unsupported knowledge kind in {path}")
    if str(raw.get("status") or "") not in _STATUSES:
        raise ValueError(f"unsupported knowledge status in {path}")
    _validate_embedded_values(raw, path=path)


class CuratedKnowledgeProvider:
    """Load reviewed knowledge records from domain directories."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.items = self._load()

    def _load(self) -> list[KnowledgeRecord]:
        items: list[KnowledgeRecord] = []
        if not self.root.is_dir():
            return items
        for path in sorted(self.root.rglob("*.yaml")):
            raw = (
                yaml.load(
                    path.read_text(encoding="utf-8"),
                    Loader=_StrictSafeLoader,
                )
                or {}
            )
            required = {
                "id",
                "domain",
                "kind",
                "status",
                "summary",
                "evidence",
                "provenance",
            }
            if int(raw.get("schema_version") or 0) != 1:
                raise ValueError(f"unsupported knowledge schema in {path}")
            missing = sorted(required - set(raw))
            if missing:
                raise ValueError(f"incomplete knowledge record {path}: " + ", ".join(missing))
            _validate_record_data(raw, path=path)
            items.append(
                KnowledgeRecord(
                    id=str(raw["id"]),
                    domain=str(raw["domain"]),
                    kind=str(raw["kind"]),
                    status=str(raw["status"]),
                    applicability=dict(raw.get("applicability") or {}),
                    match=dict(raw.get("match") or {}),
                    summary=str(raw["summary"]),
                    guidance=_as_tuple(raw.get("guidance")),
                    required_checks=_as_tuple(raw.get("required_checks")),
                    evidence=dict(raw["evidence"]),
                    provenance=dict(raw["provenance"]),
                )
            )
        ids = [item.id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate knowledge record id")
        return items

    def query(self, context: KnowledgeContext) -> list[KnowledgeMatch]:
        haystack = " ".join(
            (
                context.error_signature,
                context.error_text,
                context.failure_class,
                context.quant_signature,
                context.model_arch,
                str(context.kernel_context),
            )
        ).lower()
        matches = []
        for item in self.items:
            if item.domain != context.domain:
                continue
            if item.status in {"contradicted", "retired"}:
                continue
            if any(not _matches(expected, context.value(key)) for key, expected in item.applicability.items()):
                continue
            match_constraints = 0
            text_hits = 0
            for key, expected in item.match.items():
                values = _as_tuple(expected)
                if not values:
                    continue
                if key == "keywords":
                    if any(value.lower() in haystack for value in values):
                        text_hits += 1
                    continue
                match_constraints += 1
                if key in {"error_contains", "symptoms"}:
                    if any(value.lower() in haystack for value in values):
                        text_hits += 1
                    else:
                        break
                elif not _matches(expected, context.value(key)):
                    break
                else:
                    text_hits += 1
            else:
                score = float(len(item.applicability) * 10 + match_constraints * 10 + text_hits)
                matches.append(
                    KnowledgeMatch(
                        item=item,
                        match_type=("exact" if match_constraints or text_hits else "related"),
                        applicability_score=score,
                    )
                )
        return sorted(
            matches,
            key=lambda match: (
                match.match_type == "exact",
                match.applicability_score,
                match.item.evidence_level,
                match.item.id,
            ),
            reverse=True,
        )
