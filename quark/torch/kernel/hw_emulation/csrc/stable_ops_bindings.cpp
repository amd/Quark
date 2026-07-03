//
// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Test harness that exposes every quark::torch_stable wrapper from
// stable_ops.h as an individually callable torch custom op so that
// Python unit tests can exercise each wrapper in isolation.
//
// The ops are registered under the "quark_test_stable_ops" namespace
// and become accessible from Python via torch.ops.quark_test_stable_ops.*.
//
// TODO(bteng): Temporary file. Remove once the wrappers in stable_ops.h
// are exposed directly by upstream PyTorch (torch/csrc/stable/ops.h),
// at which point the accompanying test_stable_ops.py harness is also
// unnecessary. Tracking issue:
// https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark/issues/5317
//

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include "stable_ops.h"

namespace stable_ops = quark::torch_stable;

// TORCH_BOX needs an addressable symbol; the stable_ops.h helpers are inline,
// so wrap each in a free function.

static torch::stable::Tensor wrap_floor(const torch::stable::Tensor& s) {
  return stable_ops::floor(s);
}
static torch::stable::Tensor wrap_ceil(const torch::stable::Tensor& s) {
  return stable_ops::ceil(s);
}
static torch::stable::Tensor wrap_round(const torch::stable::Tensor& s) {
  return stable_ops::round(s);
}
static torch::stable::Tensor wrap_logical_not(const torch::stable::Tensor& s) {
  return stable_ops::logical_not(s);
}

static torch::stable::Tensor wrap_mul(
  const torch::stable::Tensor& a, const torch::stable::Tensor& b
) {
  return stable_ops::mul(a, b);
}
static torch::stable::Tensor wrap_div(
  const torch::stable::Tensor& a, const torch::stable::Tensor& b
) {
  return stable_ops::div(a, b);
}
static torch::stable::Tensor wrap_sub(
  const torch::stable::Tensor& a, const torch::stable::Tensor& b
) {
  return stable_ops::sub(a, b);
}
static torch::stable::Tensor wrap_lt(
  const torch::stable::Tensor& a, const torch::stable::Tensor& b
) {
  return stable_ops::lt(a, b);
}
static torch::stable::Tensor wrap_gt(
  const torch::stable::Tensor& a, const torch::stable::Tensor& b
) {
  return stable_ops::gt(a, b);
}
static torch::stable::Tensor wrap_eq(
  const torch::stable::Tensor& a, const torch::stable::Tensor& b
) {
  return stable_ops::eq(a, b);
}
static torch::stable::Tensor wrap_logical_and(
  const torch::stable::Tensor& a, const torch::stable::Tensor& b
) {
  return stable_ops::logical_and(a, b);
}
static torch::stable::Tensor wrap_expand_as(
  const torch::stable::Tensor& a, const torch::stable::Tensor& b
) {
  return stable_ops::expand_as(a, b);
}

static torch::stable::Tensor wrap_where(
  const torch::stable::Tensor& cond, const torch::stable::Tensor& a,
  const torch::stable::Tensor& b
) {
  return stable_ops::where(cond, a, b);
}

static torch::stable::Tensor wrap_zeros_like(const torch::stable::Tensor& s) {
  return stable_ops::zeros_like(s);
}
static torch::stable::Tensor wrap_full_like(
  const torch::stable::Tensor& s, double v
) {
  return stable_ops::full_like(s, v);
}

STABLE_TORCH_LIBRARY(quark_test_stable_ops, m) {
  m.def("floor(Tensor self) -> Tensor");
  m.def("ceil(Tensor self) -> Tensor");
  m.def("round(Tensor self) -> Tensor");
  m.def("logical_not(Tensor self) -> Tensor");

  m.def("mul(Tensor self, Tensor other) -> Tensor");
  m.def("div(Tensor self, Tensor other) -> Tensor");
  m.def("sub(Tensor self, Tensor other) -> Tensor");
  m.def("lt(Tensor self, Tensor other) -> Tensor");
  m.def("gt(Tensor self, Tensor other) -> Tensor");
  m.def("eq(Tensor self, Tensor other) -> Tensor");
  m.def("logical_and(Tensor self, Tensor other) -> Tensor");
  m.def("expand_as(Tensor self, Tensor other) -> Tensor");

  m.def("where(Tensor condition, Tensor self, Tensor other) -> Tensor");

  m.def("zeros_like(Tensor self) -> Tensor");
  m.def("full_like(Tensor self, float fill_value) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(quark_test_stable_ops, CompositeExplicitAutograd, m) {
  m.impl("floor", TORCH_BOX(&wrap_floor));
  m.impl("ceil", TORCH_BOX(&wrap_ceil));
  m.impl("round", TORCH_BOX(&wrap_round));
  m.impl("logical_not", TORCH_BOX(&wrap_logical_not));

  m.impl("mul", TORCH_BOX(&wrap_mul));
  m.impl("div", TORCH_BOX(&wrap_div));
  m.impl("sub", TORCH_BOX(&wrap_sub));
  m.impl("lt", TORCH_BOX(&wrap_lt));
  m.impl("gt", TORCH_BOX(&wrap_gt));
  m.impl("eq", TORCH_BOX(&wrap_eq));
  m.impl("logical_and", TORCH_BOX(&wrap_logical_and));
  m.impl("expand_as", TORCH_BOX(&wrap_expand_as));

  m.impl("where", TORCH_BOX(&wrap_where));

  m.impl("zeros_like", TORCH_BOX(&wrap_zeros_like));
  m.impl("full_like", TORCH_BOX(&wrap_full_like));
}
