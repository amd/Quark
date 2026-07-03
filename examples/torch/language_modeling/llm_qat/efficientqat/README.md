# EfficientQAT LLM Example

This directory provides a Quark-based LLM quantization example, with a primary focus on the
Blockwise Joint Tuning algorithm in `quark/torch/algorithm/blockwise_joint_tuning`.

The main entry is `main.py`, and dataset preparation logic is in `data_preparation.py`.

## About Blockwise Joint Tuning

Compared with standard blockwise tuning, blockwise **joint** tuning optimizes two parameter groups
at the same time for each decoder block:

- model weights
- quantization parameters (for example, `scale`/`zero_point`)

High-level flow:

1. Keep a full-precision reference model and a quantized model.
2. Process decoder blocks one by one to control memory and keep optimization local.
3. For each block, use the reference block output as target, then minimize MSE between quantized
   output and reference output.
4. Update weight params and quant params with separate learning rates/schedulers.

In practice, this helps recover quantization accuracy more effectively than tuning only one parameter
type, especially for low-bit weight-only settings.

## Quick Start

Run from the repository root:

```bash
cd examples/torch/language_modeling/llm_qat/efficientqat
MODEL_DIR=/path/to/your/model \
python main.py \
  --model "${MODEL_DIR}" \
  --output_dir ./output/run_int4_bjt \
  --quant_scheme w_int4_asym \
  --group_size 128 \
  --do_ptq \
  --do_blockwise_joint_tuning \
  --blockwise_tuning_dataset c4 \
  --blockwise_tuning_train_size 32 \
  --blockwise_tuning_epochs 2 \
  --blockwise_quant_lr 1e-4 \
  --blockwise_weight_lr 1e-5 \
  --do_evaluation
```

## Common Commands

PTQ only:

```bash
python main.py \
  --model "${MODEL_DIR}" \
  --output_dir ./output/ptq_int4 \
  --quant_scheme w_int4_asym \
  --group_size 128 \
  --do_ptq \
  --do_evaluation
```

Evaluation only:

```bash
python main.py \
  --model "${MODEL_DIR}" \
  --output_dir ./output/eval_only \
  --do_evaluation
```

## Key Arguments

- `--quant_scheme`: `w_int4_asym`
- `--group_size`: group size for weight-only quantization (commonly `64` or `128`)
- `--do_ptq`: enable PTQ before later stages
- `--do_blockwise_joint_tuning`: enable blockwise joint tuning
- `--blockwise_tuning_dataset`: `c4` or `redpajama`
- `--do_evaluation`: run PPL evaluation on WikiText-2

## Dataset Notes

- Set `C4_DATA_DIR` if you want to use local C4 files.
- If local paths are not provided, datasets are loaded from Hugging Face.
- `blockwise_tuning_dataset` controls which calibration dataset is used.

## Benchmark Results

| Model | Quant Config | WikiText-2 PPL |
| ------ | ------ | ------ |
| Llama2-7b | without quantization | 5.47 |
| Llama2-7b | w_int4_asym with ptq| 5.77 |
| Llama2-7b | w_int4_asym with blockwise joint tuning| 5.59 |
