/* Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved. */
// Define a Structured Buffer for input bfloat16 data and Read Write StructuredBuffer for output float16 data
StructuredBuffer<uint> inputBuffer : register(t0); // Input bfloat16 data
RWStructuredBuffer<uint> outputBuffer : register(u0); // Output float16 data

// Thread group dimensions
[numthreads(256, 1, 1)]
void BFloat16ToFloat16CS(uint3 id : SV_DispatchThreadID)
{
    // Fetch the bfloat16 value from the input buffer using the current thread ID
    uint index = id.x;
    // uint bfloat16_value = inputBuffer[index];
	uint bfloat16_value = inputBuffer[index] & 0xFFFF;  // Extract lower 16 bits

    // Extract the sign, exponent, and mantissa from the bfloat16 (IEEE 754 format)
    uint sign     = (bfloat16_value >> 15) & 0x1;  // 1-bit sign
    uint exponent = (bfloat16_value >> 7) & 0xFF;  // 8-bit exponent
    uint mantissa = (bfloat16_value >> 0) & 0x7F;  // 7-bit mantissa

    // Convert bfloat16 exponent (8 bits) to float16 exponent (5 bits)
    // Adjust bias from bfloat16 (127) to float16 (15)
    int new_exponent = exponent - 127 + 15;

    // Handle exponent underflow, overflow, and normal cases
    if (new_exponent <= 0)
    {
        // Exponent underflow, denormalize result
        new_exponent = 0;
        mantissa = 0;
    }
    else if (new_exponent >= 31)
    {
        // Exponent overflow, set to infinity
        new_exponent = 31;
        mantissa = 0;
    }

    // Construct the float16 result (1 sign bit, 5 exponent bits, 10 mantissa bits)
    uint float16_value = (sign << 15) | (new_exponent << 10) | (mantissa << 3);  // Shift mantissa to 10 bits

    // Write the result to the output buffer
    outputBuffer[index] = float16_value;
}
