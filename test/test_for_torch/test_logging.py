#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
from io import StringIO

from quark.shares.utils.log import ScreenLogger
from quark.shares.utils.testing_utils import PatchEverywhere


def test_logging_levels():
    old_stderr = sys.stderr

    mystderr = StringIO()
    sys.stderr = mystderr

    logger = ScreenLogger(__name__)

    logger.info("hehe")
    logger.debug("hoho")

    assert "hehe" in mystderr.getvalue()

    # See https://docs.python.org/3/library/logging.html#logging-levels.
    if ScreenLogger._shared_level >= 20:
        assert "hoho" not in mystderr.getvalue()

    with PatchEverywhere("QUARK_LOG_LEVEL", "debug", module_name_prefix="quark"):
        logger = ScreenLogger(__name__)

        logger.debug("huhu")
        logger.info("hihi")
        assert "huhu" in mystderr.getvalue()
        assert "hihi" in mystderr.getvalue()

    sys.stderr = old_stderr
