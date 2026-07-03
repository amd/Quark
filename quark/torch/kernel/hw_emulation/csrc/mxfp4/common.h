// Torch-facing helpers shared by the MXFP4 host wrappers.
//
// This header only exposes pieces that talk to libtorch: the include
// umbrella for the stable headeronly ScalarType / Exception facilities and
// the `TORCH_CHECK_*` validation macros built on top.

#pragma once

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#define TORCH_CHECK_SHAPES(__x, __dim_x, __y, __dim_y, __scale_y) \
  STD_TORCH_CHECK(                                                \
    (__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y,       \
    #__x " and " #__y " have incompatible shapes"                 \
  )
#define TORCH_CHECK_DTYPE(__x, __dtype)                            \
  STD_TORCH_CHECK(                                                 \
    (__x).scalar_type() == torch::headeronly::ScalarType::__dtype, \
    #__x " is incorrect datatype, must be " #__dtype               \
  )
