# 2-bit LoRA+KD (with shared-rotation / factored export)

End-to-end recipe to recover accuracy on a 2-bit (TwoBitScalar) student by adding
**rank-N LoRA adapters** and distilling from a BF16 teacher (KL/JSD KD), then
exporting in the compact **factored (shared-rotation)** format. This is the
LoRA+KD pipeline used for **Phi-4 (14B)** and **QwQ-32B**.

Pipeline (fully self-contained — every step runs from PTQ outputs + the adapter):

```text
2-bit PTQ student (+ SRHT/AWQ sidecar) → LoRA+KD training → eval → structured `newexport` → factored export → eval factored
  (TwoBitScalar, --twobitscalar_dump_sidecar)  (train_lora_kd.py)      (export_structured_newexport.py)  (convert_*_to_factored.py)  (eval_factored_int16_acts.py)
                                            ↑ teacher reasoning traces (qad/generate_reasoning_traces.py)
```

Run from the repo root with `QUARK_ROOT` exported (locates the reasoning-trace JSONL):

```bash
export QUARK_ROOT=/workspaces/Quark
cd "$QUARK_ROOT"
```

---

## Step 0 — Generate teacher reasoning traces

Same generator as the QAD recipe (`../../llm_qad/generate_reasoning_traces.py`); the task mix
loads them by a per-model task name (file looked up under `$QUARK_ROOT`):

| Model | task name | file |
|-------|-----------|------|
| Phi-4 | `phi4_bf16_traces` | `phi4_bf16_reasoning_traces.jsonl` |
| QwQ-32B | `reasoning_jsonl` | `qwq32b_reasoning_generated.jsonl` |

```bash
# QwQ-32B (8-way shard, ~10k traces), then merge
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python ../../llm_qad/generate_reasoning_traces.py \
    --gpu_id $g --num_shards 8 --source openmath --model_dir /amd_models/Qwen/QwQ-32B \
    --max_per_source 5000 --output /tmp/qwq_traces_$g.jsonl & done
for g in 4 5 6 7; do CUDA_VISIBLE_DEVICES=$g python ../../llm_qad/generate_reasoning_traces.py \
    --gpu_id $g --num_shards 8 --source orca_math --model_dir /amd_models/Qwen/QwQ-32B \
    --max_per_source 5000 --output /tmp/qwq_traces_$g.jsonl & done
wait; cat /tmp/qwq_traces_*.jsonl > "$QUARK_ROOT/qwq32b_reasoning_generated.jsonl"
```

(For Phi-4 swap `--model_dir /amd_models/microsoft/phi-4` and output
`phi4_bf16_reasoning_traces.jsonl`.) Omit the trace task from `--task_include` to skip.

## Step 1 — 2-bit PTQ student (with sidecar)

Produce the TwoBitScalar 2-bit student. **Emit the SRHT/AWQ sidecar** — it is
required later for the factored export. TwoBitScalar is config-driven; see
[`../../../experimental/two_bits/README.md`](../../../experimental/two_bits/README.md). Use a config that also sets
`dump_sidecar_dir`:

> **Use linear (uniform) levels — `use_lloyd_max_levels: false`.** The factored /
> Quark-native export (Steps 5–7) needs the 4 levels to be evenly spaced
> `{-1, -1/3, 1/3, 1}` so they map losslessly to an affine uint2 quant
> (`zero_point=1.5`, `scale=(2/3)·group_scale`). Non-uniform Lloyd-Max levels
> (`use_lloyd_max_levels: true`) cannot be expressed as affine uint2 and would make
> the packed export lossy.

```bash
cat > /tmp/twobitscalar_sidecar.json <<'JSON'
{
  "name": "twobitscalar",
  "bits": 2, "group_size": 64, "act_scale_alpha": 0.5,
  "enable_incoherence": true, "use_lloyd_max_levels": false,
  "exclude_layers": ["*embed_tokens*", "*lm_head*"],
  "dump_sidecar_dir": "/tmp/qwq_32b_linear_g64_sidecar"
}
JSON

python ../../llm_ptq/quantize_quark.py \
  --model_dir /amd_models/Qwen/QwQ-32B \
  --quant_scheme bfp16 --quant_algo twobitscalar \
  --quant_algo_config_file twobitscalar /tmp/twobitscalar_sidecar.json \
  --exclude_layers "*embed_tokens*" "*lm_head*" \
  --export_weight_format fake_quantized --model_export hf_format \
  --output_dir /tmp/qwq_32b_lloyd_max_g64
```

## Step 2 — LoRA+KD training

`train_lora_kd.py` attaches rank-N LoRA adapters (`peft`) to the 2-bit student and
distills from the BF16 teacher. Two training backends (same adapters/result):

- **default** — the built-in manual KD loop
- **`--use_qad_trainer`** — Quark's `QADTrainer` KD loop

> ⚠️ **`--lora_target_modules` must match the architecture's projection layout**, or
> `peft` silently attaches LoRA only to the names that happen to exist (giving a broken
> partial adapter that also can't be exported to the factored format):
>
> - **Phi-4 (Phi-3 arch, *fused*):** `--lora_target_modules qkv_proj,o_proj,gate_up_proj,down_proj`
> - **QwQ-32B (Qwen2 arch, *split*):** default is correct
>   (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`).

### QwQ-32B (v13 config) — multi-GPU

```bash
TASKS="copa_sg,race,siqa,record,reasoning_jsonl,qasc,multirc,sciq,openthoughts,piqa,copa,cb,rte,wic,slimpajama,triviaqa,swag,commonsenseqa"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train_lora_kd.py \
  --student_model_dir /tmp/qwq_32b_lloyd_max_g64 \
  --teacher_model_dir /amd_models/Qwen/QwQ-32B \
  --output_dir /tmp/qwq_32b_lora_kd_v13 \
  --multi_gpu --gradient_checkpointing auto \
  --lora_r 64 --lora_alpha 128 \
  --dataset taskmix --task_include "$TASKS" --use_slimpajama \
  --max_raw_slimpajama 50000 --max_train_samples 90000 --seq_len 1024 \
  --batch_size 1 --grad_accum_steps 4 \
  --num_steps 5000 --warmup_steps 400 --lr 1e-4 \
  --kd_temperature 1.2 --kd_alpha 0.85 --kd_loss_type jsd \
  --save_lora_adapters
# add --use_qad_trainer to train through Quark's QADTrainer instead of the manual loop
```

### Phi-4 (14B) — multi-GPU

```bash
TASKS="copa_sg,race,siqa,record,phi4_bf16_traces,qasc,multirc,sciq,openthoughts,piqa,copa,cb,rte,wic,slimpajama,triviaqa,swag,commonsenseqa"
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_lora_kd.py \
  --student_model_dir /tmp/phi4_lloyd_max_g64 \
  --teacher_model_dir /amd_models/microsoft/phi-4 \
  --output_dir /tmp/phi4_lorakd_v16 \
  --multi_gpu --gradient_checkpointing auto \
  --lora_r 64 --lora_alpha 128 \
  --lora_target_modules qkv_proj,o_proj,gate_up_proj,down_proj \
  --dataset taskmix --task_include "$TASKS" --use_slimpajama \
  --max_raw_slimpajama 50000 --max_train_samples 90000 --seq_len 1024 \
  --batch_size 1 --grad_accum_steps 4 \
  --num_steps 8000 --warmup_steps 500 --lr 1e-4 \
  --kd_temperature 1.2 --kd_alpha 0.85 --kd_loss_type jsd \
  --save_lora_adapters
# add --use_qad_trainer to train through Quark's QADTrainer instead of the manual loop
```

Verified equivalence (Phi-4, 8000 steps, fused targets): manual vs `--use_qad_trainer`
factored-export WikiText-2 PPL **10.347 vs 10.308** (same band). Reference shipped
v16 factored: **10.236**.

## Step 3 — Evaluation

```bash
# WikiText-2 PPL + 5-task downstream on the adapter checkpoint (base + LoRA)
CUDA_VISIBLE_DEVICES=0 python eval_lora_kd.py \
  --base_model_dir /tmp/phi4_lloyd_max_g64 \
  --adapter_dir    /tmp/phi4_lorakd_v16/lora_adapters \
  --eval_ppl --eval_tasks arc_challenge,hellaswag,winogrande,boolq,openbookqa --num_fewshot 0
# QwQ-32B: --base_model_dir /tmp/qwq_32b_lloyd_max_g64 --adapter_dir /tmp/qwq_32b_lora_kd_v13/lora_adapters
```

## Step 4 — Structured `newexport` (from adapter + sidecar)

Build the self-loading **structured** checkpoint directly from PTQ outputs + the
LoRA adapter (no merge, no dependence on any pre-built artifact). It keeps the 2-bit
weights packed (`linear.packed_levels`/`group_scale`), LoRA separate
(`lora_A`/`lora_B`), and the SRHT/AWQ params as small vectors
(`pre_linear.awq_s_vec`/`srht_perm`/`srht_sign`).

```bash
# Phi-4
python factored_export/export_structured_newexport.py \
  --student_dir /tmp/phi4_lloyd_max_g64 \
  --sidecar_dir /tmp/phi4_linear_g64_sidecar \
  --adapter     /tmp/phi4_lorakd_v16/lora_adapters/adapter_model.safetensors \
  --output_dir  /tmp/phi4_lora_kd_newexport \
  --lora_rank 64 --lora_alpha 128 --group_size 64
```

Inputs:

- `--student_dir` — the 2-bit PTQ student (embed / norms / lm_head; **lm_head stays bf16**)
- `--sidecar_dir` — `W_q_levels` + `group_scale` + `awq_scale` + `srht_perm`/`srht_signs` from **Step 1**
- `--adapter` — the trained `peft` LoRA adapter (must target the fused projections, see Step 2)

**Quark-native weight path (`--quark_pack`).** Add `--quark_pack` to store the 2-bit
weight the way Quark's own export does — packed by `quark...Pack_uint2` with a
per-group `weight_scale` and a **float** `weight_zero_point` — and load it through
Quark's `Pack_uint2` + `dequantize` kernels (via `modeling_phi4_structured_quark`).
The uniform Lloyd-Max levels `{-1,-1/3,1/3,1}` become an affine uint2 quant with
`zero_point=1.5`, `scale=(2/3)*group_scale`, so the codebook is reproduced exactly.
Verified lossless: PPL **10.309** and 5-task downstream match the custom pack within
noise. (Requires the uint2 + `float-zp` core from PR2.) The SRHT/AWQ `pre_linear`
and LoRA remain custom — Quark's native format cannot express those.

```bash
python factored_export/export_structured_newexport.py \
  --student_dir /tmp/phi4_lloyd_max_g64 --sidecar_dir /tmp/phi4_linear_g64_sidecar \
  --adapter /tmp/phi4_lorakd_v16/lora_adapters/adapter_model.safetensors \
  --output_dir /tmp/phi4_lora_kd_newexport_quark --quark_pack
```

> QwQ-32B (Qwen2 arch) needs a `modeling_qwq32b_structured_lora.py` + matching
> `--arch_class`/`--modeling_file`; only the QwQ *factored* modeling ships today.
> Until that lands, produce the QwQ structured checkpoint via the equivalent Qwen2
> exporter, then use the same converter below.

## Step 5 — Factored (shared-rotation) export

The factored export replaces the 160 per-projection `srht_perm`/`srht_sign` with a
**single shared rotation matrix per unique `in_features`** (`shared_R_<dim>`), keeping
per-projection `awq_s_vec` + the packed 2-bit `linear` + separate LoRA. Mathematically
identical to the structured `newexport`; ~7 GB (Phi-4).

```bash
# Phi-4
python factored_export/convert_v16_to_factored.py \
  --src /tmp/phi4_lora_kd_newexport \
  --sidecar /tmp/phi4_linear_g64_sidecar \
  --out /tmp/phi4_lora_kd_factored
# QwQ-32B: convert_qwq32b_to_factored.py with --sidecar /tmp/qwq_32b_linear_g64_sidecar
```

If the source was built with `--quark_pack`, pass `--quark_pack` here too: the
2-bit `linear` tensors pass through unchanged and the emitted model uses the
Quark-backed factored modeling (`modeling_phi4_factored_quark`, Pack_uint2 +
`dequantize`). Verified factored PPL **10.3075** == custom-pack **10.3076**.

```bash
python factored_export/convert_v16_to_factored.py \
  --src /tmp/phi4_lora_kd_newexport_quark \
  --sidecar /tmp/phi4_linear_g64_sidecar \
  --out /tmp/phi4_lora_kd_factored_quark --quark_pack
```

## Step 6 — Evaluate the factored export

```bash
CUDA_VISIBLE_DEVICES=0 python factored_export/eval_factored_int16_acts.py \
  --model_dir /tmp/phi4_lora_kd_factored --act_bits 0 --ppl_only   # bf16 baseline
# --act_bits 16 applies INT16-activation fake-quant (NPU-style); drop --ppl_only for 5-task downstream
```

Verified (Phi-4, this pipeline): structured `newexport` PPL **10.310** →
factored PPL **10.308** (lossless conversion, Δ ≈ 0.001); on par with the shipped
v16 factored (**10.236**). The factored output is byte-structurally identical
(885 tensors, same keys/shapes) to `phi-4-2bit-lora-kd-linear-v16-factored`.

## Step 7 — Quark-native export (loads via `import_model_from_safetensors`)

A fully **Quark-native** variant of the shared-rotation export: the quantized base
loads through Quark's own importer (no `trust_remote_code`), and LoRA is applied on
top as a standard **`peft`** adapter. Layout:

- `{layer}.weight` uint2 (Quark `Pack_uint2`, per-group) + `{layer}.weight_scale`
  (fp16, transposed) + `{layer}.weight_zero_point` (fp16 = 1.5, float zero-point)
- `{layer}.input_prescale` (fp16 = `1/awq_s_vec`) applied before the rotation
- top-level `shared_input_rotation_<in>` (fp16 SRHT matrix, one per `in_features`)
- `config.json.quantization_config` carries a `RotationConfig` whose
  `online_config.online_rotation_layers` lists the rotated layers (model-agnostic)

Requires the Quark-core "shared / fp16 / per-channel `prescale` rotation import"
changes. Generic (works for any model with a TwoBitScalar sidecar).

```bash
python factored_export/export_quark_native.py \
  --student_dir /tmp/phi4_lloyd_max_g64 \
  --sidecar_dir /tmp/phi4_linear_g64_sidecar \
  --output_dir  /tmp/phi4_quark_native

# load: native import + peft LoRA on top
python - <<'PYEOF'
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from quark.torch import import_model_from_safetensors
from peft import PeftModel
D="/tmp/phi4_quark_native"
model=AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(D), torch_dtype=torch.float16)
model=import_model_from_safetensors(model, D).to("cuda").eval()
model=PeftModel.from_pretrained(model, "/tmp/phi4_lorakd_v16/lora_adapters")  # trained adapter
PYEOF
```

Verified (Phi-4): **6.6 GB**, WikiText-2 PPL **10.3079** — identical to the custom
factored export (10.3076), loaded entirely through Quark's native pipeline + `peft`.
