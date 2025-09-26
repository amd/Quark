#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest

from quark.experimental.cli.main import main


def test_invoke_cli_help_arg():
    with pytest.raises(SystemExit):
        main(["-h"])
