#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ._serialization import StrictSchema
from .errors import ModelSourceError, SchemaValidationError


@dataclass(frozen=True, slots=True)
class ResolvedModelSource(StrictSchema):
    requested: str
    revision: str | None
    resolved_path: str
    commit_hash: str | None

    def __post_init__(self) -> None:
        if not self.requested or not self.resolved_path:
            raise SchemaValidationError("Model source paths must not be empty.")
        if self.revision == "" or self.commit_hash == "":
            raise SchemaValidationError("Model source revisions must not be empty.")
        if not Path(self.resolved_path).is_absolute():
            raise SchemaValidationError("Resolved model source must be an absolute path.")
        if self.commit_hash is not None and Path(self.resolved_path).name != self.commit_hash:
            raise SchemaValidationError("Model source commit hash must match the resolved snapshot path.")


def resolve_model_source(model_source: str, revision: str | None = None) -> ResolvedModelSource:
    """Resolve a local checkpoint or Hub model id to one concrete local directory."""
    if not model_source:
        raise SchemaValidationError("Model source must not be empty.")
    local_path = Path(model_source).expanduser()
    if local_path.exists():
        if revision is not None:
            raise ModelSourceError("--model-revision cannot be used with a local checkpoint directory.")
        resolved_path = local_path.resolve()
        if not resolved_path.is_dir():
            raise ModelSourceError(f"Local model source is not a directory: {resolved_path}")
        return ResolvedModelSource(
            requested=model_source,
            revision=None,
            resolved_path=str(resolved_path),
            commit_hash=None,
        )
    if local_path.is_absolute() or model_source.startswith((".", "~")):
        raise ModelSourceError(f"Local model source does not exist: {local_path}")

    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
        snapshot_path = Path(
            huggingface_hub.snapshot_download(
                repo_id=model_source,
                revision=revision,
            )
        ).resolve()
    except Exception as exc:
        raise ModelSourceError(
            f"Could not resolve Hugging Face model {model_source!r} at revision {revision!r}: {exc}"
        ) from exc
    if not snapshot_path.is_dir():
        raise ModelSourceError(f"Resolved Hugging Face snapshot is not a directory: {snapshot_path}")
    if snapshot_path.parent.name != "snapshots" or not snapshot_path.name:
        raise ModelSourceError(f"Hugging Face did not return a commit-specific snapshot path: {snapshot_path}")
    return ResolvedModelSource(
        requested=model_source,
        revision=revision,
        resolved_path=str(snapshot_path),
        commit_hash=snapshot_path.name,
    )


def validate_writable_paths(model_source: ResolvedModelSource, paths: Iterable[str | Path]) -> None:
    """Reject planner outputs that would modify the resolved source checkpoint."""
    source_path = Path(model_source.resolved_path)
    for path in paths:
        resolved_path = Path(path).expanduser().resolve()
        if resolved_path == source_path or source_path in resolved_path.parents:
            raise ModelSourceError(f"Planner output path must be outside the source checkpoint: {resolved_path}")


__all__ = ["ResolvedModelSource", "resolve_model_source", "validate_writable_paths"]
