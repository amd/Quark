# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import contextlib
import logging.config
import os
import sys

if not any(key.startswith("quark") for key in logging.root.manager.loggerDict):
    import colorlog

    IMPORT_FROM_QUARK = False
else:
    IMPORT_FROM_QUARK = True

from . import matcher
from .typing import ReplaceParams

with contextlib.suppress(ImportError):
    from . import builder


class RelativePathFilter(logging.Filter):
    """
    Filter to modify log records to include the relative path instead of the
    absolute path.
    """

    def filter(self, record):
        pathname = record.pathname
        record.relativepath = None
        abs_sys_paths_map = map(os.path.abspath, sys.path)
        # longer paths first
        abs_sys_paths = sorted(abs_sys_paths_map, key=len, reverse=True)
        for path in abs_sys_paths:
            if not path.endswith(os.sep):
                path += os.sep
            if pathname.startswith(path):
                record.relativepath = os.path.relpath(pathname, path)
                break
        return True


def configure_logging() -> None:
    config = {
        "version": 1,
        "filters": {"relative_path": {"()": RelativePathFilter}},
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "%(asctime)s [%(levelname)s] %(relativepath)s(%(lineno)d): %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "colored": {
                "()": colorlog.ColoredFormatter,
                "format": "%(log_color)s[%(levelname)s]%(reset)s %(filename)s(%(lineno)d): %(message)s",
                "log_colors": {
                    "DEBUG": "blue",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "colored",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "simple",
                "filename": "onnx_utils.log",
                "maxBytes": 1048576,  # 1 MB
                "backupCount": 5,
                "delay": True,
                "filters": ["relative_path"],
            },
        },
        "loggers": {
            "ryzenai_onnx_utils": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(config)


if not IMPORT_FROM_QUARK:
    configure_logging()

__all__ = [
    "ReplaceParams",
]
