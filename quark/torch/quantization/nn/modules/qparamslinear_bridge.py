#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Single source of truth for ``QParamsLinear`` ↔ native-inference state transfer.

This module exists so that **all** ``QParamsLinear`` field-level coupling for
the native inference path lives in one place. If ``QParamsLinear`` adds /
renames / removes a metadata field, only :class:`_QParamsLinearBridge` needs
to be updated; ``NativeInferenceLinear`` consumes the bridge through three
small entry points (``build_from_source`` / ``adopt`` / ``materialize``) and
never touches QPL attributes directly.

Scope (intentionally narrow):

* **In scope** — constructive & adaptive concerns: building a QPL from a
  source ``nn.Linear``, adopting QPL state into a native layer, and
  materializing a fresh QPL from a native layer.
* **Out of scope** — quantization-metadata reading (e.g. resolving dtype
  / qscheme from the weight quantizer). That logic also has to handle
  :class:`~quark.torch.quantization.nn.modules.quantize_linear.QuantLinear`
  and is not a QPL adapter; it lives in
  :mod:`native_inference_linear_common`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from torch import nn

from quark.torch.export.nn.modules.qparamslinear import QParamsLinear

__all__ = ["_QParamsLinearBridge"]


class _QParamsLinearBridge:
    """QPL ↔ native-inference state-transfer adapter.

    The single :data:`_FIELDS` tuple is the source of truth for which
    :class:`QParamsLinear` attributes the native inference path cares
    about — :meth:`adopt` and :meth:`materialize` iterate over it, so a
    new field requires a one-line edit here and propagates to both
    directions automatically.

    Parameters / submodules and plain Python attributes are handled
    uniformly: ``setattr`` on an :class:`torch.nn.Module` routes through
    ``Module.__setattr__`` which auto-registers ``Parameter`` /
    ``Module`` values in the right internal dict, so a single iteration
    works for ``weight``, ``bias``, the quantizer children, and the
    plain metadata fields alike.
    """

    _FIELDS: ClassVar[tuple[str, ...]] = (
        # nn.Linear-style geometry (plain ints).
        "in_features",
        "out_features",
        # Parameters — Module.__setattr__ will register them in
        # ``_parameters``.
        "weight",
        "bias",
        # Submodules — Module.__setattr__ will register them in
        # ``_modules`` so they appear in ``state_dict()``.
        "weight_quantizer",
        "input_quantizer",
        "output_quantizer",
        "bias_quantizer",
        # Plain Python metadata used for export / round-trip.
        "_custom_mode",
        "_quant_config",
        "_quant_dict",
        "algo_config",
    )

    @staticmethod
    def build_from_source(
        linear: nn.Linear,
        *,
        custom_mode: str = "quark",
        pack_method: str | None = "reorder",
        quant_config: Any = None,
        algo_config: Any = None,
    ) -> QParamsLinear:
        """Return *linear* if it's already a :class:`QParamsLinear`, else build one.

        Centralizes the call to ``QParamsLinear.from_module`` so its
        constructor signature is referenced from exactly one place.
        """
        if isinstance(linear, QParamsLinear):
            return linear

        return QParamsLinear.from_module(
            linear=linear,
            custom_mode=custom_mode,
            pack_method=pack_method,
            quant_config=quant_config,
            algo_config=algo_config,
        )

    @classmethod
    def adopt(cls, target: nn.Module, qpl: QParamsLinear) -> None:
        """Copy every :data:`_FIELDS` attribute from *qpl* onto *target* by reference.

        ``target`` must be an initialized :class:`torch.nn.Module` (so
        ``Module.__setattr__`` is in effect and registers parameters /
        submodules). No copies are made — the goal is to share state
        such that ``state_dict()`` round-trips the same buffers under
        the same names as the source ``QParamsLinear``.
        """
        for field in cls._FIELDS:
            setattr(target, field, getattr(qpl, field))

    @classmethod
    def materialize(cls, source: nn.Module) -> QParamsLinear:
        """Build a fresh :class:`QParamsLinear` mirroring *source*'s :data:`_FIELDS`.

        Bypasses ``QParamsLinear.__init__`` (which expects a non-quantized
        ``nn.Linear`` source) by allocating with ``__new__`` and running
        only ``nn.Module.__init__``, then copying every field by
        reference. Used by :meth:`NativeInferenceLinear.to_qparams_linear`
        to rebuild an export-format module after native-inference
        forward.
        """
        qpl = QParamsLinear.__new__(QParamsLinear)
        nn.Module.__init__(qpl)
        for field in cls._FIELDS:
            setattr(qpl, field, getattr(source, field))
        return qpl
