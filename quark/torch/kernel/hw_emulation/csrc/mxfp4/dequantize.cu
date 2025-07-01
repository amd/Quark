#include "common.h"
#include "dequantize.h"

template<typename float_type, uint32_t half_exp_bits, uint32_t half_mantissa_bits, uint32_t half_exp_bias>
__device__ float_type upcast_fp4_to_fp16_or_bf16(uint8_t val) {
    // Takes one float4 values represented as b0000xxxx,
    // and converts it to the corresponding float16 value.

    bool sign = val >> 3;

    uint8_t exp = (val >> 1) & 3;
    uint8_t new_mantissa = val & 1;

    // if exp == 0 and new_mantissa == 0:
    //     new_exp = 0
    // else:
    //     new_exp = exp - FLOAT4_EXP_BIAS + FLOAT16_EXP_BIAS

    // int8_t works with float16, but may overflow with bfloat16.
    int16_t new_exp = exp - FLOAT4_EXP_BIAS + half_exp_bias;

    // Cast b0000 to 0. in fp16/bf16.
    new_exp = new_exp * (exp > 0 || new_mantissa > 0);

    // Cast b0001 to 0.5 in fp16/bf16.
    new_mantissa = new_mantissa && (exp > 0);

    uint16_t qdq_val = (sign << 15) + (new_exp << half_mantissa_bits) + (new_mantissa << (half_mantissa_bits - 1));
    float_type result = *(float_type*)(&qdq_val);
    return result;
}


template<typename float_type, typename scale_type, uint32_t half_exp_bits, uint32_t half_mantissa_bits, uint32_t half_exp_bias>
__global__ void dq_uint8_mxfp4_to_half_kernel(uint8_t* inp, scale_type* scales, float_type* out) {
    // One thread handles 8 output values.
    // Thus, 4 threads handle one group.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float_type out_thread[8];
    uint8_t elems[4];

    reinterpret_cast<float*>(elems)[0] = reinterpret_cast<float*>(inp)[idx];

    float_type scale_half = e8m0_to_half<float_type, scale_type>(scales[idx / 4]);

    for (int i = 0; i < 4; i++) {
        uint8_t elem = elems[i];

        // Tensor packed as [elem0, elem1], but the logical order is [elem1, elem0].
        uint8_t elem0 = elem >> 4;
        uint8_t elem1 = elem & 0xF;

        float_type elem0_half = upcast_fp4_to_fp16_or_bf16<float_type, half_exp_bits, half_mantissa_bits, half_exp_bias>(elem0);
        float_type elem1_half = upcast_fp4_to_fp16_or_bf16<float_type, half_exp_bits, half_mantissa_bits, half_exp_bias>(elem1);

        // Tensor packed as [elem0, elem1], but the logical order is [elem1, elem0].
        // TODO: We could probably use half2 dtype here.
        out_thread[2 * i + 1] = __hmul(elem0_half, scale_half);
        out_thread[2 * i] = __hmul(elem1_half, scale_half);
    }

    // Maps to a global_store_dwordx4 (4 * 4 = 16 bytes = 8 half)
    reinterpret_cast<double2*>(out)[idx] = reinterpret_cast<double2*>(out_thread)[0];
}

void dq_uint8_mxfp4_to_half(torch::Tensor inp, torch::Tensor scales, torch::Tensor out, int group_size) {
    int numel = out.numel();
    int block_size;

    if (numel % (8 * 128) == 0) {
        block_size = 128;
    }
    else if (numel % (8 * 64) == 0) {
        block_size = 64;
    }
    else {
        TORCH_CHECK(false, "The number of output elements should be a multiple of 64.");
    }
    dim3 dimGrid(numel / (8 * block_size), 1, 1);
    dim3 dimBlock(block_size, 1, 1); // < 1024: we are good!

    TORCH_CHECK(numel % (8 * block_size) == 0, "Expected dq_uint8_mxfp4_to_half input number of elements to be a multiple of 512, but it is not!");
    TORCH_CHECK(group_size == 32, "Expected group_size=32 in dq_uint8_mxfp4_to_half!");
    TORCH_CHECK(inp.is_contiguous(), "Expected dq_uint8_mxfp4_to_half input to be contiguous!");

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (out.scalar_type() == at::ScalarType::Half) {
        if (scales.scalar_type() == at::ScalarType::Half) {
            dq_uint8_mxfp4_to_half_kernel<__half, __half, FLOAT16_EXP_BITS, FLOAT16_MANTISSA_BITS, FLOAT16_EXP_BIAS><<<dimGrid, dimBlock, 0, stream>>>(
                (uint8_t*) inp.data_ptr(),
                (__half*) scales.data_ptr(),
                (__half*) out.data_ptr()
            );
        }
        else if (scales.scalar_type() == at::ScalarType::Byte) {
            dq_uint8_mxfp4_to_half_kernel<__half, uint8_t, FLOAT16_EXP_BITS, FLOAT16_MANTISSA_BITS, FLOAT16_EXP_BIAS><<<dimGrid, dimBlock, 0, stream>>>(
                (uint8_t*) inp.data_ptr(),
                (uint8_t*) scales.data_ptr(),
                (__half*) out.data_ptr()
            );
        }
        else {
            TORCH_CHECK(false, "Wrong scale dtype in dq_uint8_mxfp4_to_half!");
        }
    }
    else if (out.scalar_type() == at::ScalarType::BFloat16) {
        if (scales.scalar_type() == at::ScalarType::BFloat16) {
            dq_uint8_mxfp4_to_half_kernel<__nv_bfloat16, __nv_bfloat16, BFLOAT16_EXP_BITS, BFLOAT16_MANTISSA_BITS, BFLOAT16_EXP_BIAS><<<dimGrid, dimBlock, 0, stream>>>(
                (uint8_t*) inp.data_ptr(),
                (__nv_bfloat16*) scales.data_ptr(),
                (__nv_bfloat16*) out.data_ptr()
            );
        }
        else if (scales.scalar_type() == at::ScalarType::Byte) {
            dq_uint8_mxfp4_to_half_kernel<__nv_bfloat16, uint8_t, BFLOAT16_EXP_BITS, BFLOAT16_MANTISSA_BITS, BFLOAT16_EXP_BIAS><<<dimGrid, dimBlock, 0, stream>>>(
                (uint8_t*) inp.data_ptr(),
                (uint8_t*) scales.data_ptr(),
                (__nv_bfloat16*) out.data_ptr()
            );
        }
        else {
            TORCH_CHECK(false, "Wrong scale dtype in dq_uint8_mxfp4_to_half!");
        }
    }
    else {
        TORCH_CHECK(false, "Wrong output dtype in dq_uint8_mxfp4_to_half!");
    }
}
