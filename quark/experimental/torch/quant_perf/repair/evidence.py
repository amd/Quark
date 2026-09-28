#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from .signatures import failure_signature
from .types import FailureEvidence

_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)(?:, in ([^\s]+))?')
_EXCEPTION_RE = re.compile(r"((?:[A-Za-z_][\w.]*)?(?:Error|Exception|Warning)):\s*(.*)")
_WORKER_RE = re.compile(r"^(\([^)]*\)|\[rank[^]]+\]:)\s*")
_LOG_PREFIX_RE = re.compile(r"^(?:ERROR|WARNING|INFO)\s+.*?\[[^\]]+\]\s*")
_GENERIC_WRAPPERS = (
    "see root cause above",
    "background process",
    "failed core proc",
)
_MISSING_FLYDSL_BACKEND_MARKERS = (
    "flydsl",
    "unavailable in the installed aiter build",
)


def _clean_line(line: str) -> str:
    return _LOG_PREFIX_RE.sub("", _WORKER_RE.sub("", line)).strip()


def is_missing_flydsl_backend_failure(error: str) -> bool:
    """Return whether an error reports a missing FlyDSL backend in AITER."""
    normalized = error.lower()
    return all(marker in normalized for marker in _MISSING_FLYDSL_BACKEND_MARKERS)


def extract_failure_evidence(error: object) -> FailureEvidence:
    """Extract one root cause from complete output, keeping worker frames separate."""
    if isinstance(error, FailureEvidence):
        return error
    existing = getattr(error, "evidence", None)
    if isinstance(existing, FailureEvidence):
        return existing
    diagnostic = getattr(error, "diagnostic", "")
    streams = []
    for name in ("stdout", "stderr"):
        value = getattr(error, name, "")
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        if isinstance(value, str) and value:
            streams.append(f"{name}:\n{value}")
    text = "\n".join(streams) if streams else str(diagnostic or error or "")
    if isinstance(error, BaseException) and not streams and not diagnostic and "\n" not in text:
        text = f"{type(error).__name__}: {error}"

    # Only frames in the same worker's current traceback belong to an exception.
    frames: dict[str, tuple[str, int, str, str]] = {}
    pending_source: set[str] = set()
    candidates: list[tuple[bool, str, str, tuple[str, int, str, str]]] = []
    empty_frame = ("", 0, "", "")
    lines = text.splitlines()
    for raw_line in lines:
        worker = _WORKER_RE.match(raw_line)
        origin = worker.group(1) if worker else ""
        line = _clean_line(raw_line)
        if line.startswith("Traceback (most recent call last)"):
            frames.pop(origin, None)
            pending_source.discard(origin)
            continue
        frame = _FRAME_RE.match(line)
        if frame:
            filename, lineno, function = frame.groups()
            frames[origin] = (filename, int(lineno), function or "", "")
            pending_source.add(origin)
            continue
        exception = _EXCEPTION_RE.match(line)
        if exception:
            exception_type, message = exception.groups()
            root = frames.pop(origin, empty_frame)
            pending_source.discard(origin)
            # A warning emitted by warnings.warn has a filename/line prefix,
            # unlike a Warning raised as a traceback's terminating exception.
            if not exception_type.endswith("Warning") or root != empty_frame:
                is_wrapper = any(marker in message.lower() for marker in _GENERIC_WRAPPERS)
                candidates.append((not is_wrapper, exception_type, message, root))
        elif origin in pending_source:
            filename, lineno, function, _ = frames[origin]
            frames[origin] = (filename, lineno, function, line)
            pending_source.discard(origin)

    exception_type = ""
    exception_message = _clean_line(lines[-1]) if lines else ""
    root_file = ""
    root_line = 0
    root_function = ""
    root_source = ""
    if candidates:
        _, exception_type, exception_message, root = max(candidates, key=lambda item: item[0])
        root_file, root_line, root_function, root_source = root

    return FailureEvidence(
        exception_type=exception_type,
        exception_message=exception_message.strip(),
        root_file=root_file,
        root_function=root_function,
        root_line=root_line,
        root_source=root_source,
        full_error=text,
        signature=failure_signature(exception_type, exception_message, root_file, root_function),
    )


def render_failure_evidence(evidence: FailureEvidence, limit: int = 5000) -> str:
    """Render root-first context; display limits never truncate stored evidence."""
    root = f"{evidence.exception_type or 'UnknownFailure'}: {evidence.exception_message}"
    if evidence.root_file:
        root += f"\n{evidence.root_file}:{evidence.root_line} in {evidence.root_function}\n{evidence.root_source}"
    paths = "\nEvidence files:\n" + "\n".join(evidence.evidence_paths) if evidence.evidence_paths else ""
    head = (root + paths)[: max(0, limit)]
    remaining = max(0, limit - len(head) - 20)
    context = evidence.full_error[:remaining] if evidence.exception_type else evidence.full_error[-remaining:]
    return head + ("\nOutput context:\n" + context if remaining else "")


def save_failure_evidence(evidence: FailureEvidence, directory: Path) -> FailureEvidence:
    """Persist full diagnostics separately from state and bounded prompt context."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "failure.log"
    path.write_text(evidence.full_error)
    return replace(evidence, evidence_paths=tuple(dict.fromkeys((*evidence.evidence_paths, str(path)))))
