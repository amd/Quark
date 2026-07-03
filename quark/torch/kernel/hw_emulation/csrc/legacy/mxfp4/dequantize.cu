#include <limits>

#include "legacy/mxfp4/dequantize.h"
#include "mxfp4/dequantize_kernels.cuh"
#include "mxfp4/mxfp4_format.cuh"

void dq_uint8_mxfp4_to_half(
  torch::Tensor inp, torch::Tensor scales, torch::Tensor out, int group_size
) {
  at::DeviceGuard device_guard(inp.device());
  TORCH_CHECK(
    inp.device() == scales.device(),
    "Expected inp and scales to be on the same device"
  );
  TORCH_CHECK(
    inp.device() == out.device(),
    "Expected inp and out to be on the same device"
  );
  int64_t numel = out.numel();
  int block_size;

  // Each thread produces OUTPUTS_PER_THREAD elements, so `numel` must be
  // divisible by `OUTPUTS_PER_THREAD * block_size`. Pick the largest valid
  // block size from {128, 64}.
  constexpr int kBlockSizeLarge = 128;
  constexpr int kBlockSizeSmall = 64;

  if (numel % (OUTPUTS_PER_THREAD * kBlockSizeLarge) == 0) {
    block_size = kBlockSizeLarge;
  } else if (numel % (OUTPUTS_PER_THREAD * kBlockSizeSmall) == 0) {
    block_size = kBlockSizeSmall;
  } else {
    TORCH_CHECK(
      false, "The number of output elements should be a multiple of 64."
    );
  }

  int64_t grid_size = numel / (OUTPUTS_PER_THREAD * block_size);

  TORCH_CHECK(
    grid_size <= static_cast<int64_t>(std::numeric_limits<int>::max()),
    "Grid size exceeds CUDA maximum grid dimension"
  );

  dim3 dimGrid(grid_size, 1, 1);
  dim3 dimBlock(block_size, 1, 1);  // < 1024: we are good!

  TORCH_CHECK(
    group_size == MXFP4_GROUP_SIZE,
    "Expected group_size=32 in dq_uint8_mxfp4_to_half!"
  );
  TORCH_CHECK(
    inp.is_contiguous(),
    "Expected dq_uint8_mxfp4_to_half input to be contiguous!"
  );

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (out.scalar_type() == at::ScalarType::Half) {
    if (scales.scalar_type() == at::ScalarType::Half) {
      dq_uint8_mxfp4_to_half_kernel<
        __half, __half, FLOAT16_EXP_BITS, FLOAT16_MANTISSA_BITS,
        FLOAT16_EXP_BIAS><<<dimGrid, dimBlock, 0, stream>>>(
        (uint8_t*)inp.data_ptr(), (__half*)scales.data_ptr(),
        (__half*)out.data_ptr()
      );
    } else if (scales.scalar_type() == at::ScalarType::Byte) {
      dq_uint8_mxfp4_to_half_kernel<
        __half, uint8_t, FLOAT16_EXP_BITS, FLOAT16_MANTISSA_BITS,
        FLOAT16_EXP_BIAS><<<dimGrid, dimBlock, 0, stream>>>(
        (uint8_t*)inp.data_ptr(), (uint8_t*)scales.data_ptr(),
        (__half*)out.data_ptr()
      );
    } else {
      TORCH_CHECK(false, "Wrong scale dtype in dq_uint8_mxfp4_to_half!");
    }
  } else if (out.scalar_type() == at::ScalarType::BFloat16) {
    if (scales.scalar_type() == at::ScalarType::BFloat16) {
      dq_uint8_mxfp4_to_half_kernel<
        __nv_bfloat16, __nv_bfloat16, BFLOAT16_EXP_BITS, BFLOAT16_MANTISSA_BITS,
        BFLOAT16_EXP_BIAS><<<dimGrid, dimBlock, 0, stream>>>(
        (uint8_t*)inp.data_ptr(), (__nv_bfloat16*)scales.data_ptr(),
        (__nv_bfloat16*)out.data_ptr()
      );
    } else if (scales.scalar_type() == at::ScalarType::Byte) {
      dq_uint8_mxfp4_to_half_kernel<
        __nv_bfloat16, uint8_t, BFLOAT16_EXP_BITS, BFLOAT16_MANTISSA_BITS,
        BFLOAT16_EXP_BIAS><<<dimGrid, dimBlock, 0, stream>>>(
        (uint8_t*)inp.data_ptr(), (uint8_t*)scales.data_ptr(),
        (__nv_bfloat16*)out.data_ptr()
      );
    } else {
      TORCH_CHECK(false, "Wrong scale dtype in dq_uint8_mxfp4_to_half!");
    }
  } else {
    TORCH_CHECK(false, "Wrong output dtype in dq_uint8_mxfp4_to_half!");
  }
}
