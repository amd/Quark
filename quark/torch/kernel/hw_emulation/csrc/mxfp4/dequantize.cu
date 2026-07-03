#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cstdint>
#include <limits>

#include "device_guard.h"
#include "gpu_stream.h"
#include "mxfp4/common.h"
#include "mxfp4/dequantize_kernels.cuh"
#include "mxfp4/mxfp4_format.cuh"

using torch::stable::accelerator::DeviceGuard;

namespace quark {
namespace hw_emulation {

void dq_uint8_mxfp4_to_half_impl(
  const torch::stable::Tensor& inp, const torch::stable::Tensor& scales,
  torch::stable::Tensor& out, int64_t group_size
) {
  quark::DeviceGuard guard(inp);
  STD_TORCH_CHECK(
    inp.get_device() == scales.get_device(),
    "Expected inp and scales to be on the same device"
  );
  STD_TORCH_CHECK(
    inp.get_device() == out.get_device(),
    "Expected inp and out to be on the same device"
  );

  // Each thread produces OUTPUTS_PER_THREAD elements, so `numel` must be
  // divisible by `OUTPUTS_PER_THREAD * block_size`. Pick the largest valid
  // block size from {128, 64}.
  constexpr int kBlockSizeLarge = 128;
  constexpr int kBlockSizeSmall = 64;

  int64_t numel = out.numel();
  int block_size;

  if (numel % (OUTPUTS_PER_THREAD * kBlockSizeLarge) == 0) {
    block_size = kBlockSizeLarge;
  } else if (numel % (OUTPUTS_PER_THREAD * kBlockSizeSmall) == 0) {
    block_size = kBlockSizeSmall;
  } else {
    STD_TORCH_CHECK(
      false,
      "Expected dq_uint8_mxfp4_to_half output number of elements to be a "
      "multiple of OUTPUTS_PER_THREAD * 64 = 512."
    );
  }

  int64_t grid_size = numel / (OUTPUTS_PER_THREAD * block_size);

  STD_TORCH_CHECK(
    grid_size <= static_cast<int64_t>(std::numeric_limits<int>::max()),
    "Grid size exceeds CUDA maximum grid dimension"
  );

  dim3 dimGrid(grid_size, 1, 1);
  dim3 dimBlock(block_size, 1, 1);  // < 1024: we are good!

  STD_TORCH_CHECK(
    group_size == MXFP4_GROUP_SIZE,
    "Expected group_size=32 in dq_uint8_mxfp4_to_half!"
  );
  STD_TORCH_CHECK(
    inp.is_contiguous(),
    "Expected dq_uint8_mxfp4_to_half input to be contiguous!"
  );

  const cudaStream_t stream = getCurrentStream();

  if (out.scalar_type() == torch::headeronly::ScalarType::Half) {
    if (scales.scalar_type() == torch::headeronly::ScalarType::Half) {
      dq_uint8_mxfp4_to_half_kernel<
        __half, __half, FLOAT16_EXP_BITS, FLOAT16_MANTISSA_BITS,
        FLOAT16_EXP_BIAS><<<dimGrid, dimBlock, 0, stream>>>(
        (uint8_t*)inp.data_ptr(), (__half*)scales.data_ptr(),
        (__half*)out.data_ptr()
      );
    } else if (scales.scalar_type() == torch::headeronly::ScalarType::Byte) {
      dq_uint8_mxfp4_to_half_kernel<
        __half, uint8_t, FLOAT16_EXP_BITS, FLOAT16_MANTISSA_BITS,
        FLOAT16_EXP_BIAS><<<dimGrid, dimBlock, 0, stream>>>(
        (uint8_t*)inp.data_ptr(), (uint8_t*)scales.data_ptr(),
        (__half*)out.data_ptr()
      );
    } else {
      STD_TORCH_CHECK(false, "Wrong scale dtype in dq_uint8_mxfp4_to_half!");
    }
  } else if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    if (scales.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
      dq_uint8_mxfp4_to_half_kernel<
        __nv_bfloat16, __nv_bfloat16, BFLOAT16_EXP_BITS, BFLOAT16_MANTISSA_BITS,
        BFLOAT16_EXP_BIAS><<<dimGrid, dimBlock, 0, stream>>>(
        (uint8_t*)inp.data_ptr(), (__nv_bfloat16*)scales.data_ptr(),
        (__nv_bfloat16*)out.data_ptr()
      );
    } else if (scales.scalar_type() == torch::headeronly::ScalarType::Byte) {
      dq_uint8_mxfp4_to_half_kernel<
        __nv_bfloat16, uint8_t, BFLOAT16_EXP_BITS, BFLOAT16_MANTISSA_BITS,
        BFLOAT16_EXP_BIAS><<<dimGrid, dimBlock, 0, stream>>>(
        (uint8_t*)inp.data_ptr(), (uint8_t*)scales.data_ptr(),
        (__nv_bfloat16*)out.data_ptr()
      );
    } else {
      STD_TORCH_CHECK(false, "Wrong scale dtype in dq_uint8_mxfp4_to_half!");
    }
  } else {
    STD_TORCH_CHECK(false, "Wrong output dtype in dq_uint8_mxfp4_to_half!");
  }
}

}  // namespace hw_emulation
}  // namespace quark
