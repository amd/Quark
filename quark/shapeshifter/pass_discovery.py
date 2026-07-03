#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Shared utilities for discovering and registering Shapeshifter passes.

This module provides reusable functions for automatic pass discovery that are used
by both core passes (quark/shapeshifter/passes/) and community passes
(quark/contrib/shapeshifter_community_passes/).
"""

import importlib
import inspect
import pkgutil
from typing import Any

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def import_pass_modules(package_path: list[str], package_name: str, pass_type: str = "pass") -> dict[str, Any]:
    """Import all pass modules from the given package path.

    This function discovers all Python modules in the passes directory and imports them.
    Importing the modules triggers the @register_pass decorators, which register
    the pass classes in the global REGISTRY.

    Files starting with "_" are skipped (following Python convention for private modules).
    This includes "__init__.py" and any other files prefixed with underscore.

    Args:
        package_path: The package path list (from __path__).
        package_name: The name of the current package (from __name__).
        pass_type: Type of pass for logging ("pass", "community pass", etc.).

    Returns:
        A dictionary mapping module names to imported module objects.
    """
    imported_modules: dict[str, Any] = {}

    for _importer, modname, ispkg in pkgutil.iter_modules(package_path):
        # Skip package modules (only process .py files) and files starting with "_"
        if not ispkg and not modname.startswith("_"):
            try:
                # Import the module (this will trigger @register_pass decorators)
                module = importlib.import_module(f".{modname}", package=package_name)
                imported_modules[modname] = module
                logger.debug(f"Successfully imported {pass_type} module: {modname}")
            except Exception as e:
                # Log but don't fail if a module can't be imported
                logger.warning(f"Failed to import {pass_type} module '{modname}': {e}")

    return imported_modules


def discover_pass_classes(imported_modules: dict[str, Any], pass_type: str = "pass") -> dict[str, type]:
    """Discover pass classes from imported modules.

    This function inspects imported modules to find classes that are pass classes.
    Only classes ending with "Pass" are included (naming convention).

    Warnings are logged for:
    - Classes ending with "Pass" that are not registered with @register_pass decorator
    - Classes registered in REGISTRY that don't end with "Pass" (these are not included)

    Args:
        imported_modules: Dictionary mapping module names to module objects.
        pass_type: Type of pass for logging ("pass", "community pass", etc.).

    Returns:
        A dictionary mapping class names to class objects for discovered pass classes.
    """
    pass_classes: dict[str, type] = {}

    # Try to get registered classes from REGISTRY for validation
    registered_classes: set[type] | None = None
    try:
        from quark.shapeshifter.pass_base import REGISTRY

        registered_classes = set(REGISTRY.values())
    except ImportError:
        logger.debug("Could not import REGISTRY, will use naming convention only")

    # Inspect each imported module for pass classes
    for modname, module in imported_modules.items():
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Only consider classes defined in this module
            if obj.__module__ != module.__name__:
                continue

            # Only include classes that end with "Pass" (naming convention)
            if name.endswith("Pass"):
                # Check if it's registered
                is_registered = registered_classes is not None and obj in registered_classes

                # Log warning if class follows convention but isn't registered
                if not is_registered and registered_classes is not None:
                    logger.warning(
                        f"Found class '{name}' in module '{modname}' that follows naming "
                        f"convention but is not registered with @register_pass decorator"
                    )

                if name not in pass_classes:
                    pass_classes[name] = obj
                    logger.debug(f"Discovered {pass_type} class: {name} from module {modname}")
            else:
                # Log warning for registered classes that don't follow naming convention
                if registered_classes is not None and obj in registered_classes:
                    logger.warning(
                        f"Found registered {pass_type} class '{name}' in module '{modname}' "
                        f"that does not follow naming convention (should end with 'Pass'). "
                        f"It will not be included in __all__."
                    )

    return pass_classes


def register_pass_classes(pass_classes: dict[str, type], namespace: dict[str, Any]) -> list[str]:
    """Register discovered pass classes in the module namespace.

    Args:
        pass_classes: Dictionary mapping class names to class objects.
        namespace: The module namespace (typically globals()) where classes will be registered.

    Returns:
        A sorted list of pass class names (for __all__).
    """
    for name, cls in pass_classes.items():
        namespace[name] = cls

    return sorted(pass_classes.keys())
