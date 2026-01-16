## Usage

Please refer to https://quark.docs.amd.com/latest/pytorch/example_quark_torch_llm_ptq.html.

# Quantization schemes examples (`--quant_scheme`)

The previously supported `--quant_scheme` arguments have been renamed in Quark 0.11. The previous names are replaced as follows:

- `w_int4_per_group_sym` to `int4_wo_<group_size>` (e.g., `int4_wo_32`, `int4_wo_64`, `int4_wo_128`)
- `w_uint4_per_group_asym` to `uint4_wo_<group_size>` (e.g., `uint4_wo_32`, `uint4_wo_64`, `uint4_wo_128`)
- `w_int8_a_int8_per_tensor_sym` to `int8`
- `w_fp8_a_fp8` to `fp8`
- `w_mxfp4_a_mxfp4` to `mxfp4`
- `w_mxfp6_e3m2_a_mxfp6_e3m2` to `mxfp6_e3m2`
- `w_mxfp6_e2m3_a_mxfp6_e2m3` to `mxfp6_e2m3`
- `w_bfp16_a_bfp16` to `bfp16`
- `w_mx6_a_mx6` to `mx6`

See more details at https://quark.docs.amd.com/latest/pytorch/user_guide_config_for_llm.html, and using `python quantize_quark.py --help`.
