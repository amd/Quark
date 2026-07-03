#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import inspect
from enum import Enum
from typing import Any

from .data_type import dt_name_map


def is_qtensor_config(obj: object) -> bool:
    return hasattr(obj, "data_type") and hasattr(obj, "__dict__")


def is_qlayer_config(obj: object) -> bool:
    return (
        hasattr(obj, "input_tensors")
        or hasattr(obj, "activation")
        or hasattr(obj, "weight")
        or hasattr(obj, "bias")
        or hasattr(obj, "output_tensors")
    )


def is_algo_config(obj: object) -> bool:
    return callable(getattr(obj, "_get_config", None))


def dump_enum(e: Enum) -> str:
    """
    Serialize an Enum value into a string representation.

    The enum is converted into the format "<EnumClass>.<MemberName>",
    which can later be parsed back into the corresponding Enum.

    :param Enum e: Enum instance to be serialized.
    :return: String representation of the enum.
    """
    return f"{e.__class__.__name__}.{e.name}"


def dump_class_key(k: Any) -> Any:
    """
    Normalize dictionary keys for serialization.

    Class objects are converted to their class names. Other values
    are returned as-is.

    :param Any k: Dictionary key to be normalized.
    :return: Normalized key.
    """
    if k is None:
        return None
    if isinstance(k, type):
        return k.__name__
    return k


def get_explicit_params(obj: object) -> dict[str, Any]:
    """
    Extract explicitly set parameters from an algorithm configuration.

    Parameters whose values differ from their constructor defaults
    are collected and returned.

    :param AlgoConfig obj: Algorithm configuration instance.
    :return: Dictionary of explicitly set parameter names and values.
    """
    sig = inspect.signature(obj.__init__)
    defaults = {
        name: param.default
        for name, param in sig.parameters.items()
        if name != "self" and param.default is not inspect._empty
    }

    explicit = {}
    for name, default_val in defaults.items():
        current_val = getattr(obj, name, None)
        if current_val != default_val:
            explicit[name] = current_val

    return explicit


def dump_algo_config(obj: object) -> dict[str, Any]:
    """
    Serialize an algorithm configuration into a dictionary.

    Only parameters explicitly set by the user are included in the
    output, along with the algorithm name.

    :param AlgoConfig obj: Algorithm configuration instance.
    :return: Dictionary representation of the algorithm configuration.
    """
    d = {"name": obj.name}  # type: ignore
    explicit = get_explicit_params(obj)

    for k, v in explicit.items():
        d[k] = v

    return d


def config_to_dict(obj: Any) -> Any:
    """
    Recursively convert a configuration object into a serializable
    Python dictionary or primitive type.

    This function supports primitive types, enums, lists, dictionaries,
    algorithm configurations, tensor configurations, and general objects
    with __dict__ attributes.

    :param Any obj: Configuration object to be converted.
    :return: Serialized representation of the object.
    """
    if isinstance(obj, int | float | str | bool) or obj is None:
        return obj

    if isinstance(obj, Enum):
        return dump_enum(obj)

    if isinstance(obj, list | tuple):
        return [config_to_dict(x) for x in obj]

    if isinstance(obj, dict):
        return {dump_class_key(k): config_to_dict(v) for k, v in obj.items()}

    if is_algo_config(obj):
        return dump_algo_config(obj)

    if is_qtensor_config(obj):
        res = {}
        for k, v in obj.__dict__.items():
            if k.startswith("_") or callable(v):
                continue
            res[k] = config_to_dict(v)

        dt_cls = getattr(obj, "data_type", None)
        if dt_cls is None:
            sig = inspect.signature(obj.__class__)
            default_dt = sig.parameters.get("data_type", None).default  # type: ignore
            dt_cls = default_dt

        if dt_cls in dt_name_map:
            res["data_type"] = dt_name_map[dt_cls]
        elif type(dt_cls) in dt_name_map:
            res["data_type"] = dt_name_map[type(dt_cls)]
        else:
            res["data_type"] = str(dt_cls)

        return res

    if hasattr(obj, "__dict__"):
        res = {}
        for k, v in obj.__dict__.items():
            if k in ["specific_layer_config", "layer_type_config"] and isinstance(v, dict):
                tmp_list = []
                for v_k, v_v in v.items():
                    if v_k is None:
                        tmp_list.append([v_k, config_to_dict(v_v)])
                    if is_qlayer_config(v_k):
                        tmp_list.append([config_to_dict(v_k), config_to_dict(v_v)])
                res[k] = tmp_list
            else:
                if k.startswith("_") or callable(v) or v is None:
                    continue
                if k == "extra_options" and isinstance(v, dict) and "extra_options" in v:
                    res[k] = config_to_dict(v["extra_options"])
                else:
                    res[k] = config_to_dict(v)
        return res

    return str(obj)
