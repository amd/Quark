#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import logging
import os
from typing import Any, Set, TypeVar, Callable, cast, List
from logging import LogRecord

from functools import wraps

_C = TypeVar("_C", bound=Callable[..., Any])  # pragma: no cover


class DebugLogger:

    def __init__(self, name: str, debug_file_dir: str = 'quark_logs') -> None:
        self.logger = logging.getLogger(f'{name}_debug')
        self.logger.setLevel(logging.DEBUG)

        if not os.path.exists(debug_file_dir):
            os.makedirs(debug_file_dir)

        file_handler = logging.FileHandler(os.path.join(debug_file_dir, f'{name}_debug.log'), mode="w")
        file_handler.setLevel(logging.DEBUG)
        self.logger.propagate = False
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(msg, *args, **kwargs)


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
        self.msgs: Set[str] = set()

    def filter(self, record: LogRecord) -> bool:
        allow_duplicate = getattr(record, 'allow_duplicate', False)
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

    _instances: List[Any] = []  # type List[ScreenLogger]: recored all ScreenLogger instances

    def __init__(self, name: str) -> None:
        self.logger = logging.getLogger(f'{name}_screen')
        console_handler = logging.StreamHandler()
        self.logger.propagate = False
        console_formatter = CustomFormatter()
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(self._shared_level)
        self._instances.append(self)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        allow_duplicate = True
        if 'allow_duplicate' in kwargs:
            allow_duplicate = kwargs['allow_duplicate']
            kwargs.pop('allow_duplicate')

        self.logger.info(msg, extra={'allow_duplicate': allow_duplicate}, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        allow_duplicate = True
        if 'allow_duplicate' in kwargs:
            allow_duplicate = kwargs['allow_duplicate']
            kwargs.pop('allow_duplicate')

        self.logger.warning(msg, extra={'allow_duplicate': allow_duplicate}, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        allow_duplicate = True
        if 'allow_duplicate' in kwargs:
            allow_duplicate = kwargs['allow_duplicate']
            kwargs.pop('allow_duplicate')

        error_code = None
        if 'error_code' in kwargs:
            error_code = kwargs['error_code']
            kwargs.pop('error_code')
        if error_code is not None:
            msg = f"[Error Code: {error_code}] {msg}"
        self.logger.error(msg, extra={'allow_duplicate': allow_duplicate}, *args, **kwargs)

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
