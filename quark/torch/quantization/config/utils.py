#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from dataclasses import is_dataclass, fields
from typing import Any


def dataclass_pretty_string(dataclass_inst: Any, indent: int = 0) -> str:
    """
    Pretty-prints a dataclass, with line breaks and actually pretty indent (pprint is not).
    """
    if not is_dataclass(dataclass_inst):
        raise RuntimeError(
            "The function `dataclass_pretty_string` is meant to be called on dataclass class instances only.")

    s = f"{dataclass_inst.__class__.__name__}(\n"
    for f in fields(dataclass_inst):
        field_value = getattr(dataclass_inst, f.name)
        if is_dataclass(field_value):
            s += "    " * (indent + 1) + f"{f.name}={dataclass_pretty_string(field_value, indent=indent + 1)},\n"
        elif isinstance(field_value, list) and all(is_dataclass(val) for val in field_value):
            # List of dataclasses.
            s += "    " * (indent + 1) + f"{f.name}=[\n"
            for sub_dataclass in field_value:
                s += "    " * (indent + 2) + f"{dataclass_pretty_string(sub_dataclass, indent=indent + 2)},\n"
            s += "    " * (indent + 1) + "],\n"
        elif isinstance(field_value, str):
            s += "    " * (indent + 1) + f"{f.name}=\"{field_value}\",\n"
        else:
            s += "    " * (indent + 1) + f"{f.name}={field_value},\n"
    s += "    " * indent + ")"
    return s
