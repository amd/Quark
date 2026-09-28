#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Scoped Transformers compatibility for mixed-precision model loading."""

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def _transformers_output_recorder_compatibility() -> Iterator[None]:
    """Keep Kimi-K3's old OutputRecorder import working during HF loading.

    Transformers 5.2 moved this class from ``utils.generic`` to
    ``utils.output_capturing`` (https://github.com/huggingface/transformers/pull/43765).
    Kimi-K3's remote code still imports the old path. ``trust_remote_code=True``
    permits executing that code but does not fix the import. Search setup and
    file-to-file export preparation both import it when building a meta model.

    Earlier validation on Transformers 5.14.1 patched this import in a model
    copy; unmodified checkpoints still need the compatibility alias here.
    Reuse the upstream class only during loading and restore the module
    afterward; no recording behavior is replaced.
    """
    from transformers.utils import generic

    if hasattr(generic, "OutputRecorder"):
        yield
        return
    try:
        from transformers.utils.output_capturing import OutputRecorder
    except ModuleNotFoundError as error:
        if error.name != "transformers.utils.output_capturing":
            raise
        yield
        return

    generic.OutputRecorder = OutputRecorder
    try:
        yield
    finally:
        del generic.OutputRecorder
