//
// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stable ABI wrappers for ATen ops not yet provided by torch/csrc/stable/ops.h.
// Each function calls the ATen dispatcher via the stable C ABI, following
// the same pattern used by upstream PyTorch wrappers (e.g. clone, transpose).
//
// Minimum compatible version: PyTorch 2.10 (requires torch_call_dispatcher
// with TORCH_ABI_VERSION).
//
// TODO(bteng): Temporary header. Remove these wrappers and switch call
// sites to the upstream API once PyTorch exposes these ops directly in
// torch/csrc/stable/ops.h (either via our contribution or upstream's).
// Tracking issue:
// https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark/issues/5317
//

#pragma once

#include <torch/csrc/stable/ops.h>

#include <array>

namespace quark {
namespace torch_stable {

// =========================================================================
// Unary element-wise ops  (Tensor -> Tensor)
// =========================================================================

inline ::torch::stable::Tensor floor(const ::torch::stable::Tensor& self) {
  std::array<StableIValue, 1> stack{::torch::stable::detail::from(self)};
  TORCH_ERROR_CODE_CHECK(
    torch_call_dispatcher("aten::floor", "", stack.data(), TORCH_ABI_VERSION)
  );
  return ::torch::stable::detail::to<::torch::stable::Tensor>(stack[0]);
}

inline ::torch::stable::Tensor ceil(const ::torch::stable::Tensor& self) {
  std::array<StableIValue, 1> stack{::torch::stable::detail::from(self)};
  TORCH_ERROR_CODE_CHECK(
    torch_call_dispatcher("aten::ceil", "", stack.data(), TORCH_ABI_VERSION)
  );
  return ::torch::stable::detail::to<::torch::stable::Tensor>(stack[0]);
}

inline ::torch::stable::Tensor round(const ::torch::stable::Tensor& self) {
  std::array<StableIValue, 1> stack{::torch::stable::detail::from(self)};
  TORCH_ERROR_CODE_CHECK(
    torch_call_dispatcher("aten::round", "", stack.data(), TORCH_ABI_VERSION)
  );
  return ::torch::stable::detail::to<::torch::stable::Tensor>(stack[0]);
}

inline ::torch::stable::Tensor logical_not(
  const ::torch::stable::Tensor& self
) {
  std::array<StableIValue, 1> stack{::torch::stable::detail::from(self)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
    "aten::logical_not", "", stack.data(), TORCH_ABI_VERSION
  ));
  return ::torch::stable::detail::to<::torch::stable::Tensor>(stack[0]);
}

// =========================================================================
// Binary element-wise ops  (Tensor, Tensor -> Tensor)
// =========================================================================

inline ::torch::stable::Tensor mul(
  const ::torch::stable::Tensor& self, const ::torch::stable::Tensor& other
) {
  std::array<StableIValue, 2> stack{
    ::torch::stable::detail::from(self), ::torch::stable::detail::from(other)
  };
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
    "aten::mul", "Tensor", stack.data(), TORCH_ABI_VERSION
  ));
  return ::torch::stable::detail::to<::torch::stable::Tensor>(stack[0]);
}

inline ::torch::stable::Tensor div(
  const ::torch::stable::Tensor& self, const ::torch::stable::Tensor& other
) {
  std::array<StableIValue, 2> stack{
    ::torch::stable::detail::from(self), ::torch::stable::detail::from(other)
  };
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
    "aten::div", "Tensor", stack.data(), TORCH_ABI_VERSION
  ));
  return ::torch::stable::detail::to<::torch::stable::Tensor>(stack[0]);
}

inline ::torch::stable::Tensor lt(
  const ::torch::stable::Tensor& self, const ::torch::stable::Tensor& other
) {
  std::array<StableIValue, 2> stack{
    ::torch::stable::detail::from(self), ::torch::stable::detail::from(other)
  };
  TORCH_ERROR_CODE_CHECK(
    torch_call_dispatcher("aten::lt", "Tensor", stack.data(), TORCH_ABI_VERSION)
  );
  return ::torch::stable::detail::to<::torch::stable::Tensor>(stack[0]);
}

inline ::torch::stable::Tensor gt(
  const ::torch::stable::Tensor& self, const ::torch::stable::Tensor& other
) {
  std::array<StableIValue, 2> stack{
    ::torch::stable::detail::from(self), ::torch::stable::detail::from(other)
  };
  TORCH_ERROR_CODE_CHECK(
    torch_call_dispatcher("aten::gt", "Tensor", stack.data(), TORCH_ABI_VERSION)
  );
  return ::torch::stable::detail::to<::torch::stable::Tensor>(stack[0]);
}

inline ::torch::stable::Tensor eq(
  const ::torch::stable::Tensor& self, const ::torch::stable::Tensor& other
) {
  std::array<StableIValue, 2> stack{
    ::torch::stable::detail::from(self), ::torch::stable::detail::from(other)
  };
  TORCH_ERROR_CODE_CHECK(
    torch_call_dispatcher("aten::eq", "Tensor", stack.data(), TORCH_ABI_VERSION)
  );
  return ::torch::stable::detail::to<::torch::stable::Tensor>(stack[0]);
}

inline ::torch::stable::Tensor logical_and(
  const ::torch::stable::Tensor& self, const ::torch::stable::Tensor& other
) {
  std::array<StableIValue, 2> stack{
    ::torch::stable::detail::from(self), ::torch::stable::detail::from(other)
  };
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
    "aten::logical_and", "", stack.data(), TORCH_ABI_VERSION
  ));
  return ::torch::stable::detail::to<::torch::stable::Tensor>(stack[0]);
}

inline ::torch::stable::Tensor expand_as(
  const ::torch::stable::Tensor& self, const ::torch::stable::Tensor& other
) {
  std::array<StableIValue, 2> stack{
    ::torch::stable::detail::from(self), ::torch::stable::detail::from(other)
  };
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
    "aten::expand_as", "", stack.data(), TORCH_ABI_VERSION
  ));
  return ::torch::stable::detail::to<::torch::stable::Tensor>(stack[0]);
}

// =========================================================================
// Ternary ops
// =========================================================================

inline ::torch::stable::Tensor where(
  const ::torch::stable::Tensor& condition, const ::torch::stable::Tensor& self,
  const ::torch::stable::Tensor& other
) {
  std::array<StableIValue, 3> stack{
    ::torch::stable::detail::from(condition),
    ::torch::stable::detail::from(self), ::torch::stable::detail::from(other)
  };
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
    "aten::where", "self", stack.data(), TORCH_ABI_VERSION
  ));
  return ::torch::stable::detail::to<::torch::stable::Tensor>(stack[0]);
}

// =========================================================================
// sub  –  delegates to the existing stable subtract() C-shim wrapper
//         (Scalar alpha cannot be passed through the generic dispatcher)
// =========================================================================

inline ::torch::stable::Tensor sub(
  const ::torch::stable::Tensor& self, const ::torch::stable::Tensor& other,
  double alpha = 1.0
) {
  return ::torch::stable::subtract(self, other, alpha);
}

// =========================================================================
// Factory ops
// =========================================================================

/// zeros_like – all kwargs default to None (inherits from self).
inline ::torch::stable::Tensor zeros_like(const ::torch::stable::Tensor& self) {
  std::array<StableIValue, 6> stack{
    ::torch::stable::detail::from(self),
    ::torch::stable::detail::from(std::nullopt),
    ::torch::stable::detail::from(std::nullopt),
    ::torch::stable::detail::from(std::nullopt),
    ::torch::stable::detail::from(std::nullopt),
    ::torch::stable::detail::from(std::nullopt)
  };
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
    "aten::zeros_like", "", stack.data(), TORCH_ABI_VERSION
  ));
  return ::torch::stable::detail::to<::torch::stable::Tensor>(stack[0]);
}

/// full_like – composed from empty_like + fill_ because Scalar has no
/// StableIValue specialisation and therefore cannot traverse the dispatcher.
inline ::torch::stable::Tensor full_like(
  const ::torch::stable::Tensor& self, double fill_value
) {
  auto result = ::torch::stable::empty_like(self);
  ::torch::stable::fill_(result, fill_value);
  return result;
}

}  // namespace torch_stable
}  // namespace quark
