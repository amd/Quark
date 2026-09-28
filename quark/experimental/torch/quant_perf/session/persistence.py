#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Persistence helpers for Quant-Perf session files."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON through a unique same-directory temporary file.

    Each replacement is atomic for readers. Concurrent writers use independent
    temporary files, with the last completed replacement determining the final
    document.

    :param path: Destination JSON path.
    :param value: JSON-serializable value.
    """
    temporary_path = path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        temporary_path.write_text(json.dumps(value, indent=2))
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
