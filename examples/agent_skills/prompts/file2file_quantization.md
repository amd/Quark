# Prompt Examples: File2File Quantization

Two prompts for the same goal — quantize a large sharded safetensors checkpoint to FP8 without
loading the full model — at different autonomy levels. They map to the `execution_mode` field of
`session_context.json`.

## Interactive (`interactive_execute`)

I have a DeepSeek-V3 checkpoint at `/models/DeepSeek-V3` that is too large to load whole. Quantize
it to FP8 using file2file quantization and save the result to `/output/DeepSeek-V3-fp8`.

Requirements:

- show me the checkpoint inspection results and recommend an adaptation path before writing any script
- run the minimum-scale experiment first and show me the validation result
- do not start the full file2file run without my confirmation

## YOLO mode (`batch_execute`)

Quantize the safetensors checkpoint at `/models/DeepSeek-V3` to FP8 using file2file quantization
in YOLO mode — pick the best adaptation path automatically and run end to end without asking.

Requirements:

- output directory: `/output/DeepSeek-V3-fp8`, quant scheme: `w_fp8_a_fp8`, device: `cuda:0`
- run the minimum-scale experiment as a mandatory gate before the full run
- summarize the adaptation path taken, validation results, and any risks at the end
