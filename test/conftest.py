#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
from packaging import version


def pytest_addoption(parser):
    #  --limit option to limit the number of tests run, which is useful for debugging CI infra on a small subset of tests.
    parser.addoption("--limit", action="store", default=-1, type=int, help="tests limit")


def pytest_collection_modifyitems(session, config, items):
    limit = config.getoption("--limit")
    if limit >= 0:
        items[:] = items[:limit]


torch._dynamo.config.accumulated_cache_size_limit = 1000
if version.parse(torch.__version__) >= version.parse("2.7"):
    torch._dynamo.config.recompile_limit = 1000
else:
    torch._dynamo.config.cache_size_limit = 1000
