#include "common.h"
#include "fake.h"

template<typename float_type, uint32_t half_exp_bits, uint32_t half_mantissa_bits, uint32_t half_exp_bias>
__device__ float_type fp16_to_fp4_simulate(float_type* val) {
    // Casts an fp16 input to the restricted values of float4_e2m1,
    // that is to say [0., 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0].

    uint16_t val_view = *(uint16_t*)val;

    uint16_t exp = val_view >> half_mantissa_bits;
    exp = exp & ((1 << half_exp_bits) - 1);

    bool sign = (val_view >> (half_mantissa_bits + half_exp_bits)) & 1;

    bool mantissa_last = (val_view >> (half_mantissa_bits - 1)) & 1;

    int16_t exp_unbias = exp - half_exp_bias;
    int16_t new_exp = exp_unbias + FLOAT4_EXP_BIAS;

    int16_t exp_shift = (new_exp <= 0) * (1 - new_exp);

    // Typically 9.
    // Take the min to prevent overflow on `uint16_t half`. This is the case for very small values,
    // correctly mapped to `round_close`.
    uint16_t tail_bits = min(16, half_mantissa_bits - FLOAT4_MANTISSA_BITS + exp_shift);

    uint16_t mantissa_plus_one = val_view & ((1 << (half_mantissa_bits + 1)) - 1);

    uint16_t half = 1 << (tail_bits - 1);

    uint16_t tail = mantissa_plus_one & ((1 << tail_bits) - 1);

    bool round_close = (tail < half);  // round towards 0
    bool round_away = (tail > half);  // round away from 0
    bool tie = tail == half;

    uint16_t new_mantissa;

    bool new_mantissa_close = 0;
    uint16_t new_exp_close = 0;

    bool new_mantissa_away = 0;
    uint16_t new_exp_away = 0;

    uint16_t new_exp_tie = 0;

    // # 1. round down
    // if new_exp == 0: # case [0.5, 0.749999]
    //     new_mantissa = 0
    // elif new_exp < 0:  # case [0, 0.24999]
    //     new_mantissa = 0
    // else:
    //     new_mantissa = mantissa_last

    new_mantissa_close = (new_exp > 0) * mantissa_last;
    new_exp_close = exp;

    // # 2. round up
    // if new_exp <= 0:  # case [0.250001, 0.499999] and [0.75001, 0.99999]
    //     new_mantissa = 0
    //     new_exp += 1
    // elif mantissa_last == 0:
    //     new_mantissa = 1
    // else:
    //     new_mantissa = 0
    //     new_exp += 1

    new_mantissa_away = (new_exp > 0) && (mantissa_last == 0);
    new_exp_away = exp + ((new_exp <= 0) || (mantissa_last == 1));

    // # 3. tie
    // 0.25 -> 0. (handled by `exp > (half_exp_bias - 2)`)
    // 0.75 -> 1.
    // 1.25 -> 1.
    // 1.75 -> 2.
    // 2.5 -> 2.
    // 3.5 -> 4.
    // 5. -> 4.
    new_exp_tie = (exp > (half_exp_bias - 2)) * (exp + (mantissa_last == 1));

    // # Gather round up, round down and tie.
    new_exp = round_away * new_exp_away + round_close * new_exp_close + tie * new_exp_tie;
    new_mantissa = round_away * new_mantissa_away + round_close * new_mantissa_close;

    // if new_exp > 3:
    //     new_mantissa = 1
    new_mantissa = new_mantissa + (new_exp > (2 + half_exp_bias)) * (new_mantissa == 0);

    // Clamp the exponent to acceptable values.
    new_exp = (new_exp >= (half_exp_bias - 2)) * max((half_exp_bias - 2), min(new_exp, half_exp_bias + 2));

    uint16_t qdq_val = (sign << 15) + (new_exp << half_mantissa_bits) + (new_mantissa << (half_mantissa_bits - 1));
    float_type result = *(float_type*)(&qdq_val);
    return result;
}

template<typename float_type, uint32_t half_exp_bits, uint32_t half_mantissa_bits, uint32_t half_exp_bias, uint16_t val_to_add, uint16_t sign_exponent_mask>
__global__ void qdq_mxfp4_kernel(float_type* inp, float_type* out) {
    // Each thread handles one element.

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float_type elem = inp[idx];
    float_type block_max = __habs(elem);

    // Compute the max 32 lanes by 32 lanes.
    // Each thread handles a single value, thus applying `shfl_xor` 5 times.
    // (Max over 2**5 = 32 values).
    for (int i = 1; i < 32; i*=2) {
        block_max = __hmax(block_max, __habs(shfl_xor_bf16_or_half(block_max, i)));
    }

    // TODO: fix as well in quantize kernel.
    // Apply rounding strategy to block_max.
    // cannot take the address of an rvalue so need this intermediate `block_max_uint` variable?
    uint16_t block_max_uint = (*(uint16_t*)(&block_max) + val_to_add) & sign_exponent_mask;

    block_max = *(float_type*)(&block_max_uint);

    uint8_t scale_exp = max(
        0,
        FLOAT8_E8M0_MAX_EXP + min(bf16_or_half2int_rn<float_type>(hfloor(hlog2(block_max))) - 2, FLOAT8_E8M0_MAX_EXP)
    );
    float_type scale = float_to_bf16_or_half<float_type>(powf(2.0, scale_exp - FLOAT8_E8M0_MAX_EXP));

    elem = __hdiv(elem, scale);

    float_type elem_fp4 = fp16_to_fp4_simulate<float_type, half_exp_bits, half_mantissa_bits, half_exp_bias>(&elem);

    out[idx] = __hmul(elem_fp4, scale);
}

void qdq_mxfp4_(torch::Tensor a, int group_size) {
    int numel = a.numel();

    dim3 dimGrid(numel / 128, 1, 1);
    dim3 dimBlock(128, 1, 1); // < 1024: we are good!

    TORCH_CHECK(numel % 128 == 0, "Expected qdq_mxfp4_ input number of elements to be a multiple of 128, but it is not!");
    TORCH_CHECK(group_size == 32, "Expected group_size=32 in qdq_mxfp4_!");
    TORCH_CHECK(a.is_contiguous(), "Expected qdq_mxfp4_ input to be contiguous!");

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (a.scalar_type() == at::ScalarType::Half) {
        qdq_mxfp4_kernel<__half, FLOAT16_EXP_BITS, FLOAT16_MANTISSA_BITS, FLOAT16_EXP_BIAS, FLOAT16_VAL_TO_ADD, FLOAT16_SIGN_EXPONENT_MASK><<<dimGrid, dimBlock, 0, stream>>>((__half*) a.data_ptr(), (__half*) a.data_ptr());
    }
    else if (a.scalar_type() == at::ScalarType::BFloat16) {
        qdq_mxfp4_kernel<__nv_bfloat16, BFLOAT16_EXP_BITS, BFLOAT16_MANTISSA_BITS, BFLOAT16_EXP_BIAS, BFLOAT16_VAL_TO_ADD, BFLOAT16_SIGN_EXPONENT_MASK><<<dimGrid, dimBlock, 0, stream>>>((__nv_bfloat16*) a.data_ptr(), (__nv_bfloat16*) a.data_ptr());
    }
    else {
        TORCH_CHECK(false, "Wrong input dtype in qdq_mxfp4!");
    }
}

torch::Tensor qdq_mxfp4(torch::Tensor a, int group_size) {
    int numel = a.numel();

    dim3 dimGrid(numel / 128, 1, 1);
    dim3 dimBlock(128, 1, 1); // < 1024: we are good!

    TORCH_CHECK(numel % 128 == 0, "Expected qdq_mxfp4 input number of elements to be a multiple of 128, but it is not!");
    TORCH_CHECK(group_size == 32, "Expected group_size=32 in qdq_mxfp4!");
    TORCH_CHECK(a.is_contiguous(), "Expected qdq_mxfp4 input to be contiguous!");

    torch::Tensor out = at::empty(a.sizes(), a.options());

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (a.scalar_type() == at::ScalarType::Half) {
        qdq_mxfp4_kernel<__half, FLOAT16_EXP_BITS, FLOAT16_MANTISSA_BITS, FLOAT16_EXP_BIAS, FLOAT16_VAL_TO_ADD, FLOAT16_SIGN_EXPONENT_MASK><<<dimGrid, dimBlock, 0, stream>>>((__half*) a.data_ptr(), (__half*) out.data_ptr());
    }
    else if (a.scalar_type() == at::ScalarType::BFloat16) {
        qdq_mxfp4_kernel<__nv_bfloat16, BFLOAT16_EXP_BITS, BFLOAT16_MANTISSA_BITS, BFLOAT16_EXP_BIAS, BFLOAT16_VAL_TO_ADD, BFLOAT16_SIGN_EXPONENT_MASK><<<dimGrid, dimBlock, 0, stream>>>((__nv_bfloat16*) a.data_ptr(), (__nv_bfloat16*) out.data_ptr());
    }
    else {
        TORCH_CHECK(false, "Wrong input dtype in qdq_mxfp4!");
    }

    return out;
}
