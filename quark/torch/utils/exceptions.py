#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Project-wide common exception definitions.
Covers common scenarios such as data validation, business logic errors, and training loss issues.
"""


class AppError(Exception):
    """
    Base class for all custom exceptions in the project.

    Attributes:
        message (str): Description of the error.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)

    def __str__(self) -> str:
        """Return a string representation of the error."""
        return super().__str__()


# =========================
# Loss-related exceptions
# =========================
class LossError(AppError):
    """
    Raised when an unexpected or invalid loss value is detected during training or evaluation.

    Args:
        message (str): Description of the loss-related error.
    """

    def __init__(self, message: str):
        super().__init__(f"Loss error: {message}")
