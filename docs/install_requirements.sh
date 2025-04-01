#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Update source/sphinx/requirements.txt from source/sphinx/requirements.in
pip install pip-tools
rm -f source/sphinx/requirements.txt
LC_ALL=C pip-compile source/sphinx/requirements.in

# Install updated requirements.txt
pip install -r source/sphinx/requirements.txt
LC_ALL=C sphinx-build --version
