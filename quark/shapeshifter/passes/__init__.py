#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Core Shapeshifter passes.

This module automatically discovers and registers all pass classes from the
quark/shapeshifter/passes/ directory.
"""

from quark.shapeshifter.pass_discovery import discover_pass_classes, import_pass_modules, register_pass_classes

# Dynamically discover and register all pass classes
_package_path = __path__  # type: ignore[assignment]
_imported_modules = import_pass_modules(_package_path, __name__, pass_type="pass")
_pass_classes = discover_pass_classes(_imported_modules, pass_type="pass")
__all__ = register_pass_classes(_pass_classes, globals())
