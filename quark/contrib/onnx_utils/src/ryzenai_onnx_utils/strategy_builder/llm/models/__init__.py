# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import LlmModelBase

REGISTRY = {}


def register_model(model_class: LlmModelBase) -> LlmModelBase:
    """Register a model class as a custom model"""
    global REGISTRY
    REGISTRY[model_class.model_type] = model_class
    return model_class


__all__ = ["register_model", "REGISTRY"]
