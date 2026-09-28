#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Packaged YAML recipes for the speculative-decoding pipeline.

This directory holds data, not code. It is a package so that ``find_packages()``
returns it and ``setup.py``'s ``package_data`` entry actually ships the YAML in
the wheel -- ``run.py`` resolves its default ``--config`` relative to this
directory, so a pip-installed user needs the files present.
"""
