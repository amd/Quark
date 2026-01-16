#!/bin/bash

#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Update source/sphinx/requirements.txt from source/sphinx/requirements.in
# TODO: Install latest `pip` when https://github.com/jazzband/pip-tools/issues/2252 is fixed
pip install --upgrade pip==25.2 pip-tools
rm -f source/sphinx/requirements.txt
LC_ALL=C pip-compile source/sphinx/requirements.in

pip install quark-dashboard --no-cache --trusted-host xcoartifactory.xilinx.com -i https://xcoartifactory.xilinx.com/artifactory/api/pypi/uai-pip-local/simple/

# Install updated requirements.txt
pip install -r source/sphinx/requirements.txt
LC_ALL=C sphinx-build --version
