#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cstdint>

#include "device_guard.h"
#include "gpu_stream.h"
#include "mxfp4/common.h"
#include "mxfp4/fake_kernels.cuh"
#include "mxfp4/mxfp4_format.cuh"

using torch::stable::accelerator::DeviceGuard;

namespace quark {
namespace hw_emulation {
namespace {

// `qdq_mxfp4_kernel` requires `numel` to be divisible by `block_size`. Pick
// the largest valid block size from {128, 64}.
constexpr int kBlockSizeLarge = 128;
constexpr int kBlockSizeSmall = 64;

// Validate `a`, pick a block size, and launch `qdq_mxfp4_kernel`. The kernel
// reads from `a.data_ptr()` and writes to `out_ptr` (which may alias for the
// in-place variant).
void launch_qdq_mxfp4(
  const torch::stable::Tensor& a, int64_t group_size, void* out_ptr
) {
  quark::DeviceGuard guard(a);
  int numel = a.numel();
  int block_size;

  if (numel % kBlockSizeLarge == 0) {
    block_size = kBlockSizeLarge;
  } else if (numel % kBlockSizeSmall == 0) {
    block_size = kBlockSizeSmall;
  } else {
    STD_TORCH_CHECK(
      false,
      "Expected qdq_mxfp4 input number of elements to be a multiple of 64, but "
      "it is not!"
    );
  }

  STD_TORCH_CHECK(
    group_size == MXFP4_GROUP_SIZE, "Expected group_size=32 in qdq_mxfp4!"
  );
  STD_TORCH_CHECK(
    a.is_contiguous(), "Expected qdq_mxfp4 input to be contiguous!"
  );

  dim3 dimGrid(numel / block_size, 1, 1);
  dim3 dimBlock(block_size, 1, 1);  // < 1024: we are good!

  const cudaStream_t stream = getCurrentStream();

  if (a.scalar_type() == torch::headeronly::ScalarType::Half) {
    qdq_mxfp4_kernel<
      __half, FLOAT16_EXP_BITS, FLOAT16_MANTISSA_BITS, FLOAT16_EXP_BIAS,
      FLOAT16_VAL_TO_ADD, FLOAT16_SIGN_EXPONENT_MASK>
      <<<dimGrid, dimBlock, 0, stream>>>(
        (__half*)a.data_ptr(), (__half*)out_ptr
      );
  } else if (a.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    qdq_mxfp4_kernel<
      __nv_bfloat16, BFLOAT16_EXP_BITS, BFLOAT16_MANTISSA_BITS,
      BFLOAT16_EXP_BIAS, BFLOAT16_VAL_TO_ADD, BFLOAT16_SIGN_EXPONENT_MASK>
      <<<dimGrid, dimBlock, 0, stream>>>(
        (__nv_bfloat16*)a.data_ptr(), (__nv_bfloat16*)out_ptr
      );
  } else {
    STD_TORCH_CHECK(false, "Wrong input dtype in qdq_mxfp4!");
  }
}

}  // namespace

void qdq_mxfp4_inplace_impl(torch::stable::Tensor& a, int64_t group_size) {
  launch_qdq_mxfp4(a, group_size, a.data_ptr());
}

torch::stable::Tensor qdq_mxfp4_impl(
  const torch::stable::Tensor& a, int64_t group_size
) {
  torch::stable::Tensor out = torch::stable::empty_like(a);
  launch_qdq_mxfp4(a, group_size, out.data_ptr());
  return out;
}

}  // namespace hw_emulation
}  // namespace quark
