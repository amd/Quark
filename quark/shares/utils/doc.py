#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from typing import Any


def add_start_docstring(*docstring_text: Any) -> Any:
    """
    Decorates functions/methods to add text at the beginning of the function existing docstring.
    """

    def docstring_decorator(fn: Any) -> Any:
        original_docstring = fn.__doc__

        if original_docstring is not None:
            fn.__doc__ = original_docstring + "".join(docstring_text)
        else:
            fn.__doc__ = "".join(docstring_text)

        return fn

    return docstring_decorator
