# OGA 2-bit models: translate + INT16 activation Q-DQ

End-to-end reference for taking a **standard-Quark uint2** LLM to an ONNX Runtime
`MatMulNBits` model with **INT16 activation Q-DQ** (W2A16), suitable for AMD Ryzen AI
NPU-transformer deployment.

```text
Quark QAD (uint2 export)  ->  OGA builder (MatMulNBits, bits=2)  ->  add_int16_qdq.py  ->  run / eval
        HF safetensors            model.onnx (2-bit weights)         + INT16 activations      ORT
```

## Contents

| File | Purpose |
|------|---------|
| `add_int16_qdq.py`        | Standalone tool: add static INT16 activation Q-DQ on an OGA-translated 2-bit ONNX model. |
| `add_int16_qdq.ipynb`     | Notebook walkthrough of the Q-DQ step (before/after node summary + optional PPL compare). |
| `translate_2bit_oga.ipynb`| Notebook walkthrough of the OGA translation step (QAD + LoRA+KD). |

## Prerequisites

- A standard-Quark **uint2** model (weight `dtype=uint2`, per-group `scale`, float `zero_point`).
  See the [2-bit QAD example](../README_QAD_2BIT.md) for producing one.
- **onnxruntime-genai** with 2-bit support (`--precision int2`): branch `2bit_upstream_qad`
  for plain QAD uint2 models, or `2bit_upstream_lora_kd` for LoRA+KD models (factored
  online rotation + PEFT LoRA adapters baked into the graph).
- **onnxruntime >= 1.27** (float zero-point 2-bit `MatMulNBits`; 12-input GroupQueryAttention).
- For the `quark` backend: `amd-quark` installed (`joblib`, `onnxruntime`), plus `transformers`,
  `datasets` for calibration.

## Step 1 - Translate the uint2 model with OGA

Use the onnxruntime-genai model builder to translate the Quark uint2 checkpoint into an ONNX
`MatMulNBits` (bits=2, float zero-point) model:

```bash
python -m onnxruntime_genai.models.builder \
    -i /path/to/quark_uint2_model \
    -o /path/to/oga_out \
    -p int2 -e cpu \
    -c /path/to/cache
```

The output `oga_out/` contains `model.onnx` (+ `model.onnx.data`), `genai_config.json`, and the
tokenizer files.

**LoRA+KD models** (branch `2bit_upstream_lora_kd`) translate with the same command. Their
`model.safetensors` carries a per-projection `input_prescale` vector plus one shared
`shared_input_rotation_<in>` matrix per input size, and a separate PEFT adapter under
`lora_adapters/`. The builder bakes these into the graph: a factored online rotation
`(x * input_prescale) @ shared_input_rotation` feeds each `MatMulNBits`, and the LoRA delta
`(lora_B @ lora_A @ x) * scaling` is added to the projection output. The rotation matrix is
emitted once per input size and shared by every projection that uses it.

## Step 2 - Add INT16 activation Q-DQ

```bash
# Quark backend (default, recommended - Ryzen AI ready)
python add_int16_qdq.py -i /path/to/oga_out -o /path/to/w2a16_out

# AMD Ryzen AI NPU-transformer mode (power-of-two INT16 scales)
python add_int16_qdq.py -i /path/to/oga_out -o /path/to/w2a16_out --npu-transformer

# ORT backend (fallback, no Quark dependency)
python add_int16_qdq.py -i /path/to/oga_out -o /path/to/w2a16_out --backend ort
```

What it does:

- **Activations -> UINT16, weights -> UINT8**, `quant_format=QDQ`, calibrated on WikiText-2.
- `MatMulNBits`, `GatherBlockQuantized`, and `GroupQueryAttention` are **excluded** from
  `op_types_to_quantize`. Excluding `MatMulNBits` keeps its 2-bit weight untouched; the Q/DQ pair
  inserted on the **preceding op's output** still covers the activation feeding `MatMulNBits`.
- For **LoRA+KD** models the graph also carries the online rotation (`Mul` + `MatMul`) and the
  LoRA path (`MatMul` + `Add`). These are ordinary float ops and are quantized normally, so the
  activation feeding `MatMulNBits` (the rotation `MatMul` output) still gets a Q/DQ pair.
- Copies `genai_config.json` + tokenizer into the output so it stays a runnable OGA model.

### Backends

| Backend | API | Notes |
|---------|-----|-------|
| `quark` (default) | `quark.onnx.quantize_static` | Superset of the ORT quantizer; adds Ryzen AI NPU-transformer mode, power-of-two / INT16 calibration, and algorithms (CLE/AdaQuant/...). Runs with `optimize_model=False` (the ORT post-quant shape-inference reload otherwise fails on the GQA/`MatMulNBits` graph). |
| `ort` | `onnxruntime.quantization.quantize_static` | Simpler fallback, no Quark dependency, free (non-power-of-two) scales. |

### CLI reference

```text
-i, --input-model   OGA output dir (model.onnx + genai_config.json) or a model.onnx path
-o, --output-dir    Where to write the INT16-QDQ model + copied aux files
    --backend       quark (default) | ort
    --tokenizer     HF tokenizer id/path for calibration (default: the input model dir's own tokenizer)
    --calib-dataset / --calib-subset / --calib-split   calibration dataset (default: wikitext / wikitext-2-raw-v1 / train)
    --calib-samples number of calibration samples (default: 64)
    --seq-len       max tokens per calibration sample (default: 1024)
    --npu-transformer   [quark] AMD Ryzen AI NPU-transformer mode (power-of-two INT16 scales)
    --power-of-two      [quark] power-of-two INT16 calibration (MinMSE)
```

## Step 3 - Run / evaluate

The output directory is a self-contained OGA model, so it runs with onnxruntime-genai or raw
onnxruntime. Perplexity can be compared against the pre-Q-DQ model to quantify the INT16 activation
cost (see `add_int16_qdq.ipynb`).

## Results

End-to-end on `phi-4` QAD v6 (standard-Quark uint2), full 40-layer model, WikiText-2 (test),
seq_len 2048, ONNX Runtime CPU:

| Model | Weights | Activations | WikiText-2 PPL |
|-------|---------|-------------|----------------|
| OGA translated (pre-Q-DQ) | 2-bit (`MatMulNBits`) | float | 12.70 |
| + INT16 Q-DQ (`--backend quark`) | 2-bit (`MatMulNBits`) | UINT16 | 12.74 |

INT16 activation Q-DQ adds negligible degradation (+0.04 PPL, +0.35%). The quantized graph keeps
all 200 `MatMulNBits` nodes and adds 442 `QuantizeLinear` / 525 `DequantizeLinear` (UINT16 activation
scales, UINT8 weight scales).

### LoRA+KD (factored rotation + baked LoRA)

Same setup on a `phi-4` LoRA+KD uint2 export (branch `2bit_upstream_lora_kd`), full 40-layer
model, WikiText-2 (test), seq_len 2048, ONNX Runtime CPU:

| Model | Weights | Activations | WikiText-2 PPL |
|-------|---------|-------------|----------------|
| OGA translated (pre-Q-DQ) | 2-bit (`MatMulNBits`) | float | 8.23 |
| + INT16 Q-DQ (`--backend quark`) | 2-bit (`MatMulNBits`) | UINT16 | 8.26 |

INT16 activation Q-DQ again adds negligible degradation (+0.04 PPL, +0.5%). All 280
`MatMulNBits` nodes are preserved. This model produces more QDQ pairs than the QAD one
(2002 `QuantizeLinear` / 2927 `DequantizeLinear`) because the baked online rotation
(`Mul` + `MatMul`) and LoRA (`MatMul` + `Add`) ops are quantized alongside the projections.
(Calibration: 16 WikiText-2 train samples, seq_len 512.)
