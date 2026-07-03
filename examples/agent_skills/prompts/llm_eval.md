# Prompt Examples: LLM Accuracy Evaluation

Three prompts for common eval scenarios, from a simple single-model run to a quantized-vs-baseline comparison.

## Basic benchmark run

Evaluate `Qwen/Qwen3-8B` on gsm8k.

- model is already downloaded at `/shareddata/Qwen/Qwen3-8B`
- use vLLM, show me the evaluation plan before running anything

## Quantized model accuracy check

I just quantized `Qwen/Qwen3-8B` to MXFP4 with Quark. Check whether it loses accuracy on gsm8k compared to the base model.

- quantized model: `/shareddata/amd/Qwen3-8B-mxfp4`
- base model: `/shareddata/Qwen/Qwen3-8B`
- run both, produce a side-by-side accuracy comparison

## Multi-benchmark evaluation

Evaluate `/shareddata/amd/Llama-3.1-8B-fp8` on gsm8k, mmlu, and arc_challenge.

- use vLLM
- pull reference scores from the model card or paper
- flag any benchmark where accuracy drops more than 5% from the reference
