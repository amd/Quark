#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Bound cold execution without changing the evaluator's requests or outputs."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from concurrent.futures import CancelledError, Future
from pathlib import Path
from threading import Lock
from typing import Any

DEADLINE_ENV = "QUARK_QUANT_PERF_PREPARATION_DEADLINE"
STATUS_ENV = "QUARK_QUANT_PERF_PREPARATION_STATUS"
EVALUATION_RUNTIME_VERSION = 4
logger = logging.getLogger("vllm.quark_preparation")


def prepared_executor_backend(tp: int) -> str:
    executor = "PreparedMultiprocExecutor" if tp > 1 else "PreparedUniProcExecutor"
    return f"quark.experimental.torch.quant_perf.evaluation.vllm_executor.{executor}"


def read_preparation_status(path: Path) -> dict[str, Any] | None:
    try:
        notification = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    if (
        not isinstance(notification, dict)
        or notification.get("status") not in {"ready", "failed"}
        or not isinstance(notification.get("completed_at"), int | float)
        or not math.isfinite(notification["completed_at"])
    ):
        raise ValueError(f"invalid preparation notification: {path}")
    return notification


def preparation_succeeded(path: Path) -> bool:
    notification = read_preparation_status(path)
    return notification is not None and notification["status"] == "ready"


class PreparationExecutorMixin:
    """Use a preparation deadline through the first successful real decode."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        deadline = os.environ.get(DEADLINE_ENV)
        self._preparation_deadline = float(deadline) if deadline else None
        self._preparation_lock = Lock()
        self._preparation_finished = False
        self._logged_rpc_methods: set[str] = set()
        super().__init__(*args, **kwargs)

    def collective_rpc(
        self,
        method: Any,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        **options: Any,
    ) -> Any:
        deadline = self._preparation_deadline
        preparing = deadline is not None
        if deadline is not None:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                error = TimeoutError(f"preparation deadline exceeded before RPC {method}")
                self._finish_preparation(error)
                raise error
        if isinstance(method, str) and method not in self._logged_rpc_methods:
            logger.info("[%s] RPC %s timeout=%ss", "preparation" if preparing else "evaluation", method, timeout)
            self._logged_rpc_methods.add(method)
        try:
            result = super().collective_rpc(method, timeout=timeout, args=args, kwargs=kwargs, **options)
        except Exception as error:
            if preparing:
                self._finish_preparation(error)
            raise
        # vLLM identifies cached requests with output tokens as decode requests.
        # Waiting for their execution also covers chunked prefill and the first
        # sampler invocation. Observe native completion without adding a wait.
        if preparing and method == "execute_model" and any(args[0].scheduled_cached_reqs.num_output_tokens):
            if isinstance(result, Future):
                result.add_done_callback(self._decode_completed)
            else:
                self._finish_preparation()
        return result

    def _decode_completed(self, result: Future[Any]) -> None:
        error = CancelledError("first decode cancelled") if result.cancelled() else result.exception()
        self._finish_preparation(error)

    def _finish_preparation(self, error: BaseException | None = None) -> None:
        with self._preparation_lock:
            if self._preparation_deadline is None or self._preparation_finished:
                return
            completed = time.monotonic()
            if error is None and completed > self._preparation_deadline:
                error = TimeoutError("preparation deadline exceeded during first decode")
            notification = {"status": "failed" if error is not None else "ready", "completed_at": completed}
            if error is not None:
                notification.update(
                    reason=f"{type(error).__name__}: {error}",
                    failure_kind=(
                        "timeout"
                        if isinstance(error, TimeoutError)
                        else "cancelled"
                        if isinstance(error, CancelledError)
                        else "rpc_error"
                    ),
                )
            status_path = Path(os.environ[STATUS_ENV])
            temporary = status_path.with_suffix(".tmp")
            try:
                temporary.write_text(json.dumps(notification))
                temporary.replace(status_path)
            except OSError:
                # Keep the preparation deadline active; the parent watchdog is
                # still authoritative when the notification cannot be written.
                logger.exception("Could not publish preparation outcome")
                return
            self._preparation_finished = True
            if error is None:
                self._preparation_deadline = None
                self._logged_rpc_methods.clear()
                logger.info("[evaluation] First real decode completed; native RPC timeouts restored")
