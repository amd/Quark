// Copyright (c) 2026 Advanced Micro Devices, Inc.

#pragma once

#include <cstddef>
#include <iostream>
#include <ryzenai/onnx_utils/string.hpp>
#include <stdexcept>

namespace ryzenai::onnx_utils {

namespace detail {
// used for hashing strings for switch statements
// https://gist.github.com/ruby0x1/81308642d0325fd386237cfa3b44785c
constexpr uint64_t val_64_const = 0xcbf29ce484222325;
constexpr uint64_t prime_64_const = 0x100000001b3;

// hash string at compile time based on FNV-1a algorithm
constexpr uint64_t hash_string(
  const char* str, uint64_t value = val_64_const
) noexcept {
  for (; *str != '\0'; ++str) {
    value = (value ^ static_cast<uint64_t>(static_cast<uint8_t>(*str))) *
            prime_64_const;
  }
  return value;
}
}  // namespace detail

/// @brief Supported MLADF versions
class MladfVersion {
 public:
  enum Value : uint8_t {
    v1,
    v2,
    aie4_v1,
    flat,
    unknown,
  };

  /// Constructs a new MladfVersion object
  constexpr MladfVersion() = default;
  /**
   * @brief Constructs a new MladfVersion object
   *
   * @param value string to identify the initial value of the new MladfVersion
   */
  constexpr explicit MladfVersion(const char* value)
    : value_(mapStrToType(value)) {}

  /**
   * @brief Constructs a new MladfVersion object
   *
   * @param value MladfVersion to identify the initial value of the new
   * MladfVersion
   */
  // NOLINTNEXTLINE(google-explicit-constructor, hicpp-explicit-conversions)
  constexpr MladfVersion(MladfVersion::Value value) : value_(value) {}

  /// Implicit conversion between the MladfVersion class and its internal value
  // NOLINTNEXTLINE(google-explicit-constructor, hicpp-explicit-conversions)
  constexpr operator Value() const { return value_; }

  /**
   * @brief Print support for the MladfVersion class
   *
   * @param os stream to print to
   * @param value MladfVersion instance to print
   * @return std::ostream&
   */
  friend std::ostream& operator<<(std::ostream& os, const MladfVersion& value);

  /**
   * @brief Given an enum, return a string corresponding to the version.
   *
   * @return const char*
   */
  [[nodiscard]] constexpr const char* raw() const {
    switch (value_) {
      case MladfVersion::v1:
        return "v1";
      case MladfVersion::v2:
        return "v2";
      case MladfVersion::aie4_v1:
        return "aie4_v1";
      case MladfVersion::flat:
        return "flat";
      default:
        std::cerr << "MladfVersion::raw() unknown value_: "
                  << static_cast<int>(value_) << std::endl;
        throw std::invalid_argument("Unknown MladfVersion passed");
    }
  }

  [[nodiscard]] std::string str() const { return std::string{raw()}; }

  bool anyOf(std::initializer_list<uint8_t> versions) const {
    for (const auto& v : versions) {
      if (static_cast<uint8_t>(value_) == v) {
        return true;
      }
    }
    return false;
  }

 private:
  constexpr MladfVersion::Value static mapStrToType(const char* value) {
    switch (detail::hash_string(value)) {
      case detail::hash_string("v1"):
        return MladfVersion::v1;
      case detail::hash_string("v2"):
        return MladfVersion::v2;
      case detail::hash_string("aie4_v1"):
        return MladfVersion::aie4_v1;
      case detail::hash_string("flat"):
        return MladfVersion::flat;
      default:
        throw std::invalid_argument(
          "Unknown MladfVersion passed: " + std::string(value)
        );
    }
  }

  Value value_ = Value::unknown;
};

inline std::ostream& operator<<(std::ostream& os, const MladfVersion& obj) {
  os << "MladfVersion: " << obj.raw();
  return os;
}

}  // namespace ryzenai::onnx_utils
