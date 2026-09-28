#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Native vLLM executors with a bounded cold execution phase."""

from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.uniproc_executor import UniProcExecutor

from .preparation import PreparationExecutorMixin


class PreparedMultiprocExecutor(PreparationExecutorMixin, MultiprocExecutor):
    """Native TP execution, including native asynchronous result handling."""


class PreparedUniProcExecutor(PreparationExecutorMixin, UniProcExecutor):
    """Native single-device execution with the same preparation handover."""
