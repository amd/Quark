# QAD / QAT + QLoRA Fine-Tuning with Quark (`fine_tune.py`)

This example provides a single `fine_tune.py` script to run **QAD**, **QAT**, and optional **QLoRA** **after** **Quark PTQ**.

---

## 1. Overview

### 1.1 QAD (Quantization-Aware Distillation)

QAD combines **knowledge distillation (KD)** with a **quantized** network: a full- (or high-) precision **teacher** typically guides a **student** that has fake-quant or low-bit ops inserted, with losses such as KL divergence to match output distributions—helping **recover accuracy** in low-bit or MXFP4 settings, and often more **stable** and flexible on data and pipelines than QAT alone. There is a large body of work using KD to recover accuracy after PTQ, in both industry and research.

**In this repo:** `QADTrainer` trains the PTQ student, attaches a full-precision teacher, and uses temperature-scaled KL distillation on logits (see `quark.torch.algorithm.qad_trainer`).

### 1.2 QAT (Quantization-Aware Training)

QAT **simulates quantization** (fake quant) during finetuning so that weights and activations match low-bit behavior at deploy time; at the end, fake quant is converted to real quantized operators. Versus PTQ only, QAT can achieve **higher fidelity** at the same bit width, with higher training and tuning cost.

**In this repo:** on top of a Quark-PTQ model, it uses the Hugging Face `Trainer` with a **label-supervised** loss (e.g. cross-entropy) and **does not load a teacher** (unlike the distillation objective in QAD).

### 1.3 QLoRA (Quantized LoRA)

[QLoRA](https://arxiv.org/abs/2305.14314) trains only **low-rank adapters (LoRA)** on top of a **frozen quantized backbone**, greatly reducing memory. This example swaps Quark’s `QuantLinear` for a LoRA-capable `QLoRaQuantLinear`, trains LoRA only, then **merges** back into the quantized linear layer.

---

## 2. End-to-end steps

**Pipeline:** load the LLM to be quantized and set quant/training options → run Quark **PTQ** with calibration (same idea as `quantize_quark.py`) → switch trainable parts and the `Trainer` for your mode (QAT vs QAD, with or without QLoRA) → train on a dataset, then `freeze` and save. Align **PTQ** with `examples/torch/language_modeling/llm_ptq/quantize_quark.py`.

### 2.1 Load the model and quantization options

```python
finetune_args, data_args, training_args, quant_args = parser.parse_args_into_dataclasses()
args = vars(quant_args)  # shared kwargs passed to helpers throughout
model, _ = get_model(quant_args.model_dir, **args)
tokenizer = get_tokenizer(quant_args.model_dir, **args)
```

### 2.2 PTQ (aligned with `quantize_quark.py`)

Build a `calib_dataloader`, call `preprocess_for_quantization`, then `ModelQuantizer.quantize_model` to finish PTQ:

```python
calib_dataloader = get_calib_dataloader(quant_args.calib_dataset, processor, tokenizer,**args)
preprocess_for_quantization(model)
quant_config = LLMTemplate.get(model_type).get_config(scheme=quant_args.quant_scheme, **args)
quantizer = ModelQuantizer(quant_config, quant_args.multi_device)
model = quantizer.quantize_model(model, calib_dataloader)
```

### 2.3 Training setup, `Trainer`, and model changes

The same `fine_tune.py` is a **4-in-1** script: `--training_mode` switches among `qad` / `qad_qlora` / `qat` / `qat_qlora`. QAD uses `QADTrainer`and a teacher; QAT uses `Trainer`; with `qlora` you replace `QuantLinear` with `QLoRaQuantLinear`, etc.:

```python
ft_mode = finetune_args.training_mode
use_qlora = ft_mode in ("qad_qlora", "qat_qlora")
use_qad_trainer = ft_mode in ("qad", "qad_qlora")

if use_qlora:
    replace_quant_linears_with_qlora(model)
    mark_only_qlora_adapter_as_trainable(model)
    disable_adapters(model, disable=False)
else:
    mark_only_quant_linear_as_trainable(model)

data_module = make_supervised_data_module(dataset=train_dataset, tokenizer=tokenizer)
if use_qad_trainer:
    teacher_llm = AutoModelForCausalLM.from_pretrained(quant_args.model_dir)
    model.teacher = teacher_llm
    trainer = QADTrainer(model=model, processing_class=tokenizer, args=training_args, **args)
else:
    trainer = Trainer(model=model, processing_class=tokenizer, **args)
```

### 2.4 Train and save

After `trainer.train()`, merge QLoRA adapters into `linear.weight` when applicable; then optionally `eval_model` and `export_safetensors`.

```python
trainer.train()
if use_qlora:
    merge_qlora_adapters_into_weights(trainer.model)
if getattr(model, "teacher", None) is not None:
    model.teacher = None
model = quantizer.freeze(model)
eval_model(quant_args, model, **args)
export_safetensors(model=model, output_dir=export_model_output_dir, custom_mode="quark", **args)
```

---

## 3. Experiments and evaluation

Evaluations below were run in this **Docker** image, with the test stack preinstalled:

- **Image:** `rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210`

We report **GSM8K**, **MMLU**, and **PPL** on **Qwen3-4B-Instruct-2507**, **Qwen3-14B**, and **Qwen3-0.6B**. In the table, compared to the **PTQ** column, lower PPL or higher GSM8K/MMLU is better.

### Evaluation commands (vLLM + `lm_eval`)

#### (1) Environment

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_DISABLE_COMPILE_CACHE=1
export CUDA_VISIBLE_DEVICES=0
```

#### (2) Start vLLM

(use the same `$MODEL` for `lm_eval` below)

```bash
export MODEL="/shareddata/Qwen/Qwen3-4B-Instruct-2507"
vllm serve "$MODEL" --tensor-parallel-size 1
```

#### (3) GSM8K

(in a new terminal, after vLLM is ready)

```bash
lm_eval --model local-completions \
  --model_args "model=$MODEL,base_url=http://localhost:8000/v1/completions,num_concurrent=256,max_retries=10,max_gen_toks=2048,tokenized_requests=False,tokenizer_backend=None" \
  --tasks gsm8k --num_fewshot 5 --batch_size auto
```

#### (4) MMLU

```bash
lm_eval --model local-completions \
  --model_args "model=$MODEL,base_url=http://localhost:8000/v1/completions,max_retries=10,max_gen_toks=2048,tokenized_requests=False" \
  --tasks mmlu --num_fewshot 5 --batch_size auto
```

### Result table

| Model                      | Metric                  | No quant        | PTQ             | QAD             | QAD + QLoRA     | QAT             | QAT + QLoRA     |
| -------------------------- | ----------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| **Qwen3-4B-Instruct-2507** |                         |                 |                 |                 |                 |                 |                 |
| 4B                         | PPL (↓ better)          | 10.0597         | 11.5434         | 11.0840         | 11.4123         | 10.1521         | 10.9307         |
| 4B                         | GSM8K (↑) strict / full | 0.8734 / 0.8810 | 0.7293 / 0.7718 | 0.8196 / 0.8196 | 0.7354 / 0.7756 | 0.6497 / 0.7695 | 0.6823 / 0.7521 |
| 4B                         | MMLU (↑)                | 0.7262          | 0.6663          | 0.6775          | 0.6703          | 0.6765          | 0.6701          |
| **Qwen3-14B**              |                         |                 |                 |                 |                 |                 |                 |
| 14B                        | PPL (↓)                 | 8.6419          | 9.9697          | 8.9843          | 9.8224          | 8.2677          | 9.2027          |
| 14B                        | GSM8K (↑) strict / full | 0.8749 / 0.9204 | 0.8825 / 0.8908 | 0.8908 / 0.8893 | 0.8863 / 0.8999 | 0.8302 / 0.8302 | 0.8870 / 0.8863 |
| 14B                        | MMLU (↑)                | 0.7874          | 0.7609          | 0.7587          | 0.7546          | 0.7530          | 0.7573          |
| **Qwen3-0.6B**             |                         |                 |                 |                 |                 |                 |                 |
| 0.6B                       | PPL (↓)                 | 20.9617         | 34.9021         | 26.4511         | 30.2052         | 25.4680         | 28.2416         |
| 0.6B                       | GSM8K (↑) strict / full | 0.4208 / 0.4177 | 0.0690 / 0.0523 | 0.1774 / 0.1774 | 0.1289 / 0.1175 | 0.1789 / 0.1759 | 0.1160 / 0.1145 |
| 0.6B                       | MMLU (↑)                | 0.4737          | 0.3568          | 0.3815          | 0.3677          | 0.3829          | 0.3796          |

Across the experiments in this example, **QAD often does better than QAT overall** (the gap vs. PTQ and on task metrics is most visible in the table above: check each model and metric). **QLoRA** can **save a lot of memory** during training, but **converges more slowly** in general, and final quality is **sensitive to hyperparameters** (learning rate, number of steps, LoRA rank, etc.)—tune to your budget and target.

---

## 4. Quick start

Set `model_dir`, `cache_dir`, and other paths to your **actual** model and cache locations.

```bash
# e.g. cd to this example inside your Quark clone
cd examples/torch/language_modeling/llm_qat/training_with_qad_qlora

export CUDA_VISIBLE_DEVICES=0

python fine_tune.py \
  --model_dir {MODEL_ZOO_PATH}/Qwen/Qwen3-0.6B \
  --cache_dir {DATA_PATH}/hugging_face_data \
  --quant_scheme mxfp4 \
  --num_calibration_examples 128 \
  --train_size 1000 \
  --eval_size 100 \
  --learning_rate 5e-5 \
  --lr_scheduler_type linear \
  --train_dataset Daring-Anteater \
  --max_steps 100 \
  --save_strategy no \
  --export_model_output_dir qad_Qwen3-0.6B \
  --training_mode qad
```

For all options, see `fine_tune.py` (e.g. `python fine_tune.py --help`).

---
