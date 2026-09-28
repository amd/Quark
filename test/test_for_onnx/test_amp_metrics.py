#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import numpy as np
import pytest

from quark.onnx.algorithm.mprecision.metric_funcs import (
    cosine_metric,
    evaluate_fn_adapter,
    kl_divergence_metric,
    l2_metric,
    psnr_metric,
    resolve_metric_fn,
    sqnr_metric,
)


def _make_out(val: float) -> list[list[np.ndarray]]:
    return [[np.array([val], dtype=np.float32)]]


def test_l2_metric_identical_outputs_is_zero():
    out = _make_out(1.0)
    assert l2_metric(out, out) == pytest.approx(0.0)


def test_l2_metric_different_outputs_is_positive():
    # float=[1.0], quant=[2.0] → norm([1.0]-[2.0]) = norm([-1.0]) = 1.0
    float_out = _make_out(1.0)
    quant_out = _make_out(2.0)
    assert l2_metric(float_out, quant_out) == pytest.approx(1.0, abs=1e-6)


def test_l2_metric_multi_element_array():
    # float=[3.0, 4.0], quant=[0.0, 0.0] → norm([3, 4]) = 5.0
    float_out = [[np.array([3.0, 4.0], dtype=np.float32)]]
    quant_out = [[np.array([0.0, 0.0], dtype=np.float32)]]
    assert l2_metric(float_out, quant_out) == pytest.approx(5.0, abs=1e-6)


def test_l2_metric_raises_on_mismatched_sample_count():
    float_out = [_make_out(1.0)[0], _make_out(2.0)[0]]
    quant_out = [_make_out(1.0)[0]]
    with pytest.raises(ValueError, match="same number of samples"):
        l2_metric(float_out, quant_out)


def test_l2_metric_empty_lists_returns_zero():
    assert l2_metric([], []) == pytest.approx(0.0)


def test_evaluate_fn_adapter_higher_is_better_converted():
    # evaluate_fn returns 0.9 for float, 0.7 for quant → distance = 0.2
    def evaluate_fn(outputs):
        return float(outputs[0][0][0])

    adapted = evaluate_fn_adapter(evaluate_fn)
    float_out = _make_out(0.9)
    quant_out = _make_out(0.7)
    result = adapted(float_out, quant_out)
    assert result == pytest.approx(0.2, abs=1e-5)


def test_resolve_metric_fn_defaults_to_l2():
    fn = resolve_metric_fn(None, None)
    out = _make_out(0.0)
    # l2 of identical is 0
    assert fn(out, out) == pytest.approx(0.0)


def test_resolve_metric_fn_raises_when_both_given():
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_metric_fn(lambda a, b: 0.0, lambda x: 0.0)


def test_resolve_metric_fn_uses_distance_fn_when_given():
    def custom(float_out, quant_out):
        return 42.0

    fn = resolve_metric_fn(metric_distance_fn=custom, metric_evaluate_fn=None)
    assert fn(_make_out(1.0), _make_out(2.0)) == pytest.approx(42.0)


def test_resolve_metric_fn_wraps_evaluate_fn():
    fn = resolve_metric_fn(metric_distance_fn=None, metric_evaluate_fn=lambda o: float(o[0][0][0]))
    float_out = _make_out(0.8)
    quant_out = _make_out(0.6)
    assert fn(float_out, quant_out) == pytest.approx(0.2, abs=1e-5)


def test_resolve_metric_fn_selects_kl_by_name():
    fn = resolve_metric_fn(None, None, metric_default="kl")
    out = _make_out(1.0)
    assert fn(out, out) == pytest.approx(0.0, abs=1e-6)


def test_resolve_metric_fn_selects_l2_by_name():
    fn = resolve_metric_fn(None, None, metric_default="l2")
    float_out = _make_out(1.0)
    quant_out = _make_out(2.0)
    assert fn(float_out, quant_out) == pytest.approx(1.0, abs=1e-6)


def test_resolve_metric_fn_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown metric_default"):
        resolve_metric_fn(None, None, metric_default="unknown")


def test_kl_divergence_identical_outputs_is_zero():
    out = [[np.array([0.1, 0.4, 0.3, 0.2], dtype=np.float32)]]
    assert kl_divergence_metric(out, out) == pytest.approx(0.0, abs=1e-6)


def test_kl_divergence_different_outputs_is_positive():
    float_out = [[np.array([1.0, 2.0, 3.0], dtype=np.float32)]]
    quant_out = [[np.array([3.0, 2.0, 1.0], dtype=np.float32)]]
    result = kl_divergence_metric(float_out, quant_out)
    assert result > 0.0


def test_kl_divergence_empty_lists_returns_zero():
    assert kl_divergence_metric([], []) == pytest.approx(0.0)


def test_kl_divergence_raises_on_mismatched_sample_count():
    float_out = [_make_out(1.0)[0], _make_out(2.0)[0]]
    quant_out = [_make_out(1.0)[0]]
    with pytest.raises(ValueError, match="same number of samples"):
        kl_divergence_metric(float_out, quant_out)


def test_kl_divergence_handles_negative_values():
    # Negative values are shifted to non-negative before normalizing — should not raise.
    float_out = [[np.array([-1.0, 0.0, 1.0], dtype=np.float32)]]
    quant_out = [[np.array([-1.0, 0.0, 1.0], dtype=np.float32)]]
    result = kl_divergence_metric(float_out, quant_out)
    assert result == pytest.approx(0.0, abs=1e-6)


# --- cosine_metric ---


def test_cosine_metric_identical_outputs_is_zero():
    out = [[np.array([1.0, 2.0, 3.0], dtype=np.float32)]]
    assert cosine_metric(out, out) == pytest.approx(0.0, abs=1e-6)


def test_cosine_metric_orthogonal_vectors_is_one():
    float_out = [[np.array([1.0, 0.0], dtype=np.float32)]]
    quant_out = [[np.array([0.0, 1.0], dtype=np.float32)]]
    assert cosine_metric(float_out, quant_out) == pytest.approx(1.0, abs=1e-6)


def test_cosine_metric_opposite_vectors_is_two():
    float_out = [[np.array([1.0, 0.0], dtype=np.float32)]]
    quant_out = [[np.array([-1.0, 0.0], dtype=np.float32)]]
    assert cosine_metric(float_out, quant_out) == pytest.approx(2.0, abs=1e-6)


def test_cosine_metric_zero_norm_treated_as_identical():
    float_out = [[np.array([0.0, 0.0], dtype=np.float32)]]
    quant_out = [[np.array([1.0, 0.0], dtype=np.float32)]]
    assert cosine_metric(float_out, quant_out) == pytest.approx(0.0, abs=1e-6)


def test_cosine_metric_empty_lists_returns_zero():
    assert cosine_metric([], []) == pytest.approx(0.0)


def test_cosine_metric_raises_on_mismatched_sample_count():
    with pytest.raises(ValueError, match="same number of samples"):
        cosine_metric([_make_out(1.0)[0], _make_out(2.0)[0]], [_make_out(1.0)[0]])


def test_resolve_metric_fn_selects_cosine_by_name():
    fn = resolve_metric_fn(None, None, metric_default="cosine")
    out = [[np.array([1.0, 2.0], dtype=np.float32)]]
    assert fn(out, out) == pytest.approx(0.0, abs=1e-6)


# --- sqnr_metric ---


def test_sqnr_metric_identical_outputs_is_very_negative():
    # Identical arrays → noise → _EPS → SQNR very large → metric very negative.
    out = [[np.array([1.0, 2.0, 3.0], dtype=np.float32)]]
    result = sqnr_metric(out, out)
    assert result < -100.0


def test_sqnr_metric_different_outputs_is_higher_than_identical():
    float_out = [[np.array([1.0, 2.0, 3.0], dtype=np.float32)]]
    quant_out = [[np.array([1.5, 2.5, 3.5], dtype=np.float32)]]
    identical_score = sqnr_metric(float_out, float_out)
    different_score = sqnr_metric(float_out, quant_out)
    assert different_score > identical_score


def test_sqnr_metric_known_value():
    # float=[2.0], quant=[1.0]: signal_power=4.0, noise_power=1.0
    # SQNR = 10*log10(4) ≈ 6.021 dB → metric ≈ -6.021
    float_out = [[np.array([2.0], dtype=np.float32)]]
    quant_out = [[np.array([1.0], dtype=np.float32)]]
    expected = -10.0 * np.log10(4.0)
    assert sqnr_metric(float_out, quant_out) == pytest.approx(expected, abs=1e-4)


def test_sqnr_metric_empty_lists_returns_zero():
    assert sqnr_metric([], []) == pytest.approx(0.0)


def test_sqnr_metric_raises_on_mismatched_sample_count():
    with pytest.raises(ValueError, match="same number of samples"):
        sqnr_metric([_make_out(1.0)[0], _make_out(2.0)[0]], [_make_out(1.0)[0]])


def test_resolve_metric_fn_selects_sqnr_by_name():
    fn = resolve_metric_fn(None, None, metric_default="sqnr")
    float_out = [[np.array([1.0, 2.0], dtype=np.float32)]]
    quant_out = [[np.array([1.5, 2.5], dtype=np.float32)]]
    # Different arrays → negative SQNR metric; identical → more negative.
    assert fn(float_out, quant_out) > fn(float_out, float_out)


# --- psnr_metric ---


def test_psnr_metric_identical_outputs_is_very_negative():
    # Identical arrays → MSE → _EPS → PSNR very large → metric very negative.
    out = [[np.array([1.0, 2.0, 3.0], dtype=np.float32)]]
    result = psnr_metric(out, out)
    assert result < -100.0


def test_psnr_metric_different_outputs_is_higher_than_identical():
    float_out = [[np.array([1.0, 2.0, 3.0], dtype=np.float32)]]
    quant_out = [[np.array([1.5, 2.5, 3.5], dtype=np.float32)]]
    identical_score = psnr_metric(float_out, float_out)
    different_score = psnr_metric(float_out, quant_out)
    assert different_score > identical_score


def test_psnr_metric_empty_lists_returns_zero():
    assert psnr_metric([], []) == pytest.approx(0.0)


def test_psnr_metric_raises_on_mismatched_sample_count():
    with pytest.raises(ValueError, match="same number of samples"):
        psnr_metric([_make_out(1.0)[0], _make_out(2.0)[0]], [_make_out(1.0)[0]])


def test_resolve_metric_fn_selects_psnr_by_name():
    fn = resolve_metric_fn(None, None, metric_default="psnr")
    float_out = [[np.array([1.0, 2.0], dtype=np.float32)]]
    quant_out = [[np.array([1.5, 2.5], dtype=np.float32)]]
    # Different arrays → negative PSNR metric; identical → more negative.
    assert fn(float_out, quant_out) > fn(float_out, float_out)
