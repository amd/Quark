#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import logging
import os
from collections.abc import Callable
from functools import wraps
from logging import LogRecord
from typing import Any, TypeVar, cast

_C = TypeVar("_C", bound=Callable[..., Any])  # pragma: no cover

QUARK_LOG_LEVEL = os.environ.get("QUARK_LOG_LEVEL", "info").lower()


class CustomFormatter(logging.Formatter):
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    PURPLE = "\033[35m"
    RESET = "\033[0m"
    default_fmt = "\n[QUARK-%(levelname)s]: %(message)s"
    FORMATS = {
        logging.ERROR: RED + default_fmt + RESET,
        logging.WARNING: YELLOW + default_fmt + RESET,
        logging.INFO: GREEN + default_fmt + RESET,
        logging.DEBUG: BLUE + default_fmt + RESET,
        logging.CRITICAL: PURPLE + default_fmt + RESET,
    }

    def format(self, record: LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class DuplicateFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.msgs: set[str] = set()

    def filter(self, record: LogRecord) -> bool:
        # Default to allowing duplicates unless a caller explicitly disables them.
        allow_duplicate = getattr(record, "allow_duplicate", True)
        if allow_duplicate or record.msg not in self.msgs:
            self.msgs.add(record.msg)
            return True
        return False


class ScreenLogger:
    _shared_level = logging.INFO

    @classmethod
    def set_shared_level(cls, level: int) -> None:
        cls._shared_level = level
        for instance in cls._instances:
            instance.logger.setLevel(level)

    _instances: list[Any] = []  # type List[ScreenLogger]: recored all ScreenLogger instances

    def __init__(self, name: str) -> None:
        self.logger = logging.getLogger(f"{name}_screen")
        self.logger.propagate = False
        # Ensure we don't accumulate duplicate handlers/filters across repeated imports.
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            console_handler = logging.StreamHandler()
            console_formatter = CustomFormatter()
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        if not any(isinstance(f, DuplicateFilter) for f in self.logger.filters):
            self.logger.addFilter(DuplicateFilter())

        self.logger.setLevel(self._shared_level)
        self._instances.append(self)

        if QUARK_LOG_LEVEL != "info":
            if QUARK_LOG_LEVEL == "debug":
                level = logging.DEBUG
            elif QUARK_LOG_LEVEL == "warning":
                level = logging.WARNING
            elif QUARK_LOG_LEVEL == "error":
                level = logging.ERROR
            elif QUARK_LOG_LEVEL == "critical":
                level = logging.CRITICAL
            else:
                raise ValueError(
                    f"Unsupported environment value QUARK_LOG_LEVEL={QUARK_LOG_LEVEL}, only 'debug', 'info', 'warning', 'error', 'critical' are supported."
                )

            self.set_shared_level(level)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        allow_duplicate = True
        if "allow_duplicate" in kwargs:
            allow_duplicate = kwargs["allow_duplicate"]
            kwargs.pop("allow_duplicate")

        self.logger.info(msg, *args, extra={"allow_duplicate": allow_duplicate}, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        allow_duplicate = True
        if "allow_duplicate" in kwargs:
            allow_duplicate = kwargs["allow_duplicate"]
            kwargs.pop("allow_duplicate")

        self.logger.warning(msg, *args, extra={"allow_duplicate": allow_duplicate}, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        allow_duplicate = True
        if "allow_duplicate" in kwargs:
            allow_duplicate = kwargs["allow_duplicate"]
            kwargs.pop("allow_duplicate")

        error_code = None
        if "error_code" in kwargs:
            error_code = kwargs["error_code"]
            kwargs.pop("error_code")
        if error_code is not None:
            msg = f"[Error Code: {error_code}] {msg}"
        self.logger.error(msg, *args, extra={"allow_duplicate": allow_duplicate}, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.exception(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.critical(msg, *args, **kwargs)


logger = ScreenLogger(__name__)


def log_errors(func: _C) -> _C:  # pragma: no cover
    @wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{str(e)}")
            raise

    # Just to please mypy: https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators.
    return cast(_C, wrapper)
