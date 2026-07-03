#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Tests covering empty-input handling in PerTensorPercentileObserver and
PerTensorMSEObserver.

Each test exercises a degenerate but valid input that the public method
accepts without complaint. The implementation should either return sensible
(min, max) tensors or raise a typed ValueError, not crash with a low-level
tensor error from .nonzero().min() on an empty tensor or torch.stack([]).
"""

import pytest
import torch

from quark.torch.quantization import QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import (
    PerTensorMSEObserver,
    PerTensorPercentileObserver,
)


def _percentile_spec(symmetric: bool) -> QTensorConfig:
    return QTensorConfig(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorPercentileObserver,
        symmetric=symmetric,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )


def _mse_spec() -> QTensorConfig:
    return QTensorConfig(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMSEObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )


# ---------------------------------------------------------------------------
# Asymmetric percentile, upper_idx (observer.py:1319)
# ---------------------------------------------------------------------------
def test_percentile_asymmetric_zero_histogram_upper_idx():
    """All-zero histogram makes cumulative_dist NaN; ``>= target`` is all False
    so .nonzero() is empty and .min() raises ``RuntimeError`` instead of the
    method returning a reasonable result or raising a typed error."""
    observer = PerTensorPercentileObserver(_percentile_spec(symmetric=False))
    observer.device = torch.device("cpu")
    histogram = torch.zeros(2048)
    bin_edges = torch.linspace(-1.0, 1.0, 2049)

    result = observer.get_min_max_by_percentile(histogram, bin_edges, percentile=99.0)

    assert isinstance(result, tuple) and len(result) == 2


# ---------------------------------------------------------------------------
# Asymmetric percentile, lower_idx (observer.py:1323)
# ---------------------------------------------------------------------------
def test_percentile_asymmetric_concentrated_histogram_lower_idx():
    """Histogram with all mass in the first bin makes cumulative_dist == 1
    everywhere. The condition ``cumulative_dist <= (1 - target_pct_one_side)``
    can be all-False, leaving .nonzero() empty and .min() crashing."""
    observer = PerTensorPercentileObserver(_percentile_spec(symmetric=False))
    observer.device = torch.device("cpu")
    histogram = torch.zeros(2048)
    histogram[0] = 100.0
    bin_edges = torch.linspace(-1.0, 1.0, 2049)

    result = observer.get_min_max_by_percentile(histogram, bin_edges, percentile=0.0)

    assert isinstance(result, tuple) and len(result) == 2


# ---------------------------------------------------------------------------
# Symmetric percentile, upper_idx (observer.py:1333)
# ---------------------------------------------------------------------------
def test_percentile_symmetric_zero_histogram_upper_idx():
    """All-zero histogram in the symmetric branch produces NaN cumulative_dist.
    ``>= target_pct`` is all False, .nonzero() is empty, .min() crashes."""
    observer = PerTensorPercentileObserver(_percentile_spec(symmetric=True))
    observer.device = torch.device("cpu")
    histogram = torch.zeros(2048)
    bin_edges = torch.linspace(0.0, 1.0, 2049)

    result = observer.get_min_max_by_percentile(histogram, bin_edges, percentile=99.0)

    assert isinstance(result, tuple) and len(result) == 2


# ---------------------------------------------------------------------------
# torch.stack(mses) on empty list (observer.py:1395)
# ---------------------------------------------------------------------------
def test_mse_empty_mses_list_when_start_bin_too_large():
    """Default ``start_bin=2045`` means the MSE loop body never executes for any
    histogram with fewer than ~2046 bins. ``torch.stack([])`` then raises
    ``RuntimeError: stack expects a non-empty TensorList``."""
    observer = PerTensorMSEObserver(_mse_spec())
    observer.device = torch.device("cpu")
    calib_hist = torch.ones(10)
    calib_bin_edges = torch.linspace(0.0, 1.0, 11)

    min_val, max_val = observer.get_min_max_by_mse(calib_hist, calib_bin_edges)

    assert isinstance(min_val, torch.Tensor)
    assert isinstance(max_val, torch.Tensor)


# ---------------------------------------------------------------------------
# Empty/short edges in MSE (observer.py:1371)
# ---------------------------------------------------------------------------
def test_mse_single_edge_produces_empty_centers():
    """When ``edges`` has length 1, both ``edges[1:]`` and ``edges[:-1]`` are
    empty so ``centers`` is empty. The MSE loop body never executes and
    ``torch.stack(mses)`` ultimately crashes; a correct implementation should
    fail fast with a typed error."""
    observer = PerTensorMSEObserver(_mse_spec())
    observer.device = torch.device("cpu")
    calib_hist = torch.tensor([5.0])
    calib_bin_edges = torch.tensor([0.5])

    with pytest.raises(ValueError):
        observer.get_min_max_by_mse(calib_hist, calib_bin_edges, start_bin=0)
