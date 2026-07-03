#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Community-contributed Shapeshifter passes.

This module provides automatic discovery and registration for community passes,
following the same patterns as core passes. All passes are registered to the
global REGISTRY with conflict detection to prevent naming collisions.
"""

from quark.common.utils.log import ScreenLogger
from quark.shapeshifter.pass_discovery import discover_pass_classes, import_pass_modules, register_pass_classes

logger = ScreenLogger(__name__)

# Dynamically discover and register all community pass classes
_package_path = __path__  # type: ignore[assignment]
_imported_modules = import_pass_modules(_package_path, __name__, pass_type="community pass")
_pass_classes = discover_pass_classes(_imported_modules, pass_type="community pass")
__all__ = register_pass_classes(_pass_classes, globals())

logger.info(f"Registered {len(__all__)} community passes: {', '.join(__all__)}")
