# Prompt Examples: Torch LLM PTQ

Two prompts for the same goal — quantize `Qwen/Qwen3-8B` to FP8 — at different autonomy levels. They map to the `execution_mode` field of `session_context.json`.

## Interactive (`interactive_execute`)

Help me quantize `Qwen/Qwen3-8B` with Quark using an FP8 plan.

Requirements:

- output directory: `./output/qwen3-8b-fp8`
- show the quantization decision table before writing the plan
- do not run heavy steps without my confirmation

## YOLO mode (`batch_execute`)

Quantize `Qwen/Qwen3-8B` with Quark to FP8 in YOLO mode — pick sensible defaults and run end to end without asking.

Requirements:

- output directory: `./output/qwen3-8b-fp8`
- choose the recommended FP8 scheme automatically
- summarize the chosen plan, results, and any risks at the end
