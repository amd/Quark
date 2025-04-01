//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdint>

namespace vai_q {

typedef union value_convert {
  std::uint32_t u;
  std::int32_t i;
  float f;
} value_convert_t;

static inline std::uint32_t f_to_u(float data) {
  value_convert_t vc{};
  vc.f = data;
  return vc.u;
}

static inline float u_to_f(std::uint32_t data) {
  value_convert_t vc{};
  vc.u = data;
  return vc.f;
}
static inline std::int32_t f_to_i(float data) {
  value_convert_t vc{};
  vc.f = data;
  return vc.i;
}

static inline float i_to_f(std::int32_t data) {
  value_convert_t vc{};
  vc.i = data;
  return vc.f;
}

inline float float2bfloat_cpu(const float x, std::string str = "false") {
  std::uint32_t itmp = f_to_u(x);           // float32 bitwise to int32
  if ((itmp & 0x00008000) == 0x00008000) {  // half even
    if ((itmp & 0xFFFF) > 0x00008000 ||
        (((itmp & 0xFFFF) == 0x00008000) && (itmp & 0x10000) == 0x10000)) {
      itmp += 0x10000;
    }
  }
  itmp &= 0xFFFF0000;
  return u_to_f(itmp);  // int32 bitwise to float32
}

static inline int expo( float v ) {
    return ( f_to_i( v ) >> 23 ) & 0xFF;
}

static inline float rnd( float v, int m ) {
    int sh = m - expo( v );
    float x = i_to_f(( f_to_i( v ) & ( 0x1FF << 23 )) + ( sh << 23 ));
    return ( v + x ) - x;
}

static inline float add32( float a, float b ) {
    int m = std::max( expo( a ), expo( b ));
    return rnd( a, m ) + rnd( b, m );
}

static inline float mac32( float a, float b, float c ) {
    float d = (float) a * (float) b;
    int m = std::max( expo( a ) + expo( b ) - 127, expo( c ));
    return rnd( c, m ) + rnd( d, m );
}

static inline float addmac32( float a, float b, float c, float d ) {
    float e = (float) a * (float) b;
    int m = std::max( std::max( expo( a ) + expo( b ) - 127, expo( c )), expo( d ));
    return rnd( c, m ) + rnd( d, m ) + rnd( e, m );
}

// msc32 = c - a * b
float msc32( float a, float b, float c ) {
    float a0 = float2bfloat_cpu( a );
    float a1 = float2bfloat_cpu( a - (float) a0 );                    
    float a2 = float2bfloat_cpu( a - (float) a0 - (float) a1 );       
    float b0 = float2bfloat_cpu( -b );
    float b1 = float2bfloat_cpu( -b - (float) b0 );
    float b2 = float2bfloat_cpu( -b - (float) b0 - (float) b1 );
    float terms = mac32( a0, b1, mac32( a1, b0, mac32( a2, b0, mac32( a1, b1, mac32( a0, b2, mac32( a1, b2, mac32( a2, b1, a2 * b2 )))))));
    return addmac32( a0, b0, terms, c );
}

#if 0
// This is a vanila variance implementation
void calculate_mean_var(const float* pinput,
                        int64_t batch, int64_t channel, int64_t size,
                        std::vector<float>& means,
                        std::vector<float>& variances) {
   for (auto i = 0; i < batch * channel; i ++) {
       // calc mean
       float mean_sum = 0;
       for (auto j = 0; j < size; j ++) {
           const float* pdata = pinput + i * size + j;

           float input = float2bfloat_cpu(*pdata);

           mean_sum += input;
       }
       float mean = float2bfloat_cpu(mean_sum * (float)(1.0 / size));
       means.push_back(mean);

       // calc var
       float square_diff_sum = 0;
       for (auto j = 0; j < size; j ++) {
           const float* pdata = pinput + i * size + j;

           float input = float2bfloat_cpu(*pdata);

           square_diff_sum += pow((input - mean), 2);
       }
       float variance = float2bfloat_cpu(square_diff_sum * (float)(1.0 / size));
       variances.push_back(variance);
    }
}
#else
// This is a variance implementation referenced from VART
void calculate_mean_var(const float* pinput,
                        int64_t batch, int64_t channel, int64_t size,
                        std::vector<float>& means,
                        std::vector<float>& variances) {
   for (auto i = 0; i < batch * channel; i ++) {
       float mean_sum = 0;
       float square_mean_sum = 0;

       for (auto j = 0; j < size; j ++) {
           const float* pdata = pinput + i * size + j;

           float input = float2bfloat_cpu(*pdata);

           mean_sum += input;
           square_mean_sum += std::pow(input, 2);
       }

       float mean = mean_sum * (float)(1.0 / size);
       means.push_back(float2bfloat_cpu(mean));
       float square_mean = square_mean_sum * (float)(1.0 / size);
       float variance = msc32(mean, mean, square_mean);
       variances.push_back(variance);
    }
}
#endif

void instance_normalization(const float* pinput,
                        int64_t batch, int64_t channel, int64_t size,
                        std::vector<float>& means,
                        std::vector<float>& variances,
                        std::vector<float>& gamma,
                        std::vector<float>& beta,
                        float epsilon,
                        float* poutput) {
   for (auto i = 0; i < batch * channel; i ++) {
       for (auto j = 0; j < size; j ++) {
           const float* pdata = pinput + i * size + j;

           float mean = means[i];
           float variance = variances[i];

           float inv_stdev = float2bfloat_cpu(powf(variance + epsilon, -0.5));  // inverse of sqrt
           float mean_scaled = float2bfloat_cpu(mean * inv_stdev);

           float input = float2bfloat_cpu(*pdata);
           float input_scaled = float2bfloat_cpu(input * inv_stdev);

           float temp = float2bfloat_cpu(input_scaled - mean_scaled);
           temp = gamma[i % channel] * temp + beta[i % channel];

           *(poutput + i * size + j) = float2bfloat_cpu(temp);
       }
   }
}

}  // namespace vai_q
