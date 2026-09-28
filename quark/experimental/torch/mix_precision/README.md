# Mixed Precision Quantization - Design Document

## 1. Design Goals

### 1.1 Primary Goals

1. **Automatic Optimization**: Automatically find optimal quantization configurations without manual tuning
2. **Accuracy Preservation**: Maintain model accuracy within user-specified thresholds
3. **Hardware Awareness**: Generate only valid configurations for target hardware
4. **Extensibility**: Support multiple search granularities and evaluation metrics

### 1.2 Design Principles

- **Separation of Concerns**: Each module handles a single responsibility
- **Configuration-Driven**: All behavior controlled through configuration objects
- **Lazy Evaluation**: Only load data and create objects when needed
- **Fail-Safe Search**: Early stopping when accuracy threshold exceeded

---

## 2. Architecture

### 2.1 Module Dependency Graph

```text
                    ┌─────────────────────┐
                    │   MixPrecisionConfig │ ◄── User Configuration
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ MixPrecisionQuantizer│ ◄── Main Entry Point
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
   │ConfigSearcher│    │   Switcher   │    │  Evaluation  │
   │ (searcher.py)│    │(switcher.py) │    │(external)    │
   └──────┬──────┘    └──────┬───────┘    └──────────────┘
          │                  │
          │                  ▼
          │           ┌─────────────┐
          │           │    Utils    │
          │           │  (utils.py) │
          └──────────►└─────────────┘
```

### 2.2 Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| `MixPrecisionQuantizer` | quantizer.py | Orchestrates search workflow |
| `MixPrecisionConfig` | config.py | Holds all configuration parameters |
| `ConfigSearcher` | searcher.py | Generates and ranks configurations |
| `Switcher` | switcher.py | Applies quantization to model |
| `Utils` | utils.py | Layer categorization and pattern matching |
| `Evaluation` | (external) | Model quality evaluation |

---

## 3. Core Design Decisions

### 3.1 Search Granularity

**Decision**: Support three granularity levels with MODULE as initial implementation.

| Granularity | Scope | Config Space | Status |
|-------------|-------|--------------|--------|
| MODULE | All layers share same config per partition | O(modes^partitions) | ✅ Implemented |
| DECODER_LAYER | Per-layer or layer-group config | O(modes^partitions × layers) | 🔲 Planned |
| LINEAR_LAYER | Per-linear-layer config | O(modes^linear_layers) | 🔲 Planned |

**Rationale**: MODULE granularity provides fast search with reasonable accuracy. Finer granularities increase search space exponentially.

### 3.2 Layer Partitioning

**Decision**: Partition layers by module type. The core partitions are
`self_attn` and `dense_mlp`; hybrid and MoE models add `linear_attn`,
`routed_moe`, and `shared_expert` when present.

```text
self_attn:     q_proj, k_proj, v_proj, o_proj (attention projections)
linear_attn:   linear attention projections (Qwen 3.5, etc.)
dense_mlp:     dense gate_proj, up_proj, down_proj feed-forward layers
routed_moe:    routed expert projections / fused expert containers
shared_expert: always-active shared experts in MoE models
```

**Rationale**:

- Attention and MLP have different sensitivity to quantization
- Module-type partitions balance configuration space vs. flexibility
- Pattern-based matching handles different model architectures
- `shared_expert` is a dependent partition: its mode is constrained to
  native or the current `routed_moe` mode, so it does not add an
  independent Cartesian dimension to the search space
- Keeping `dense_mlp` separate from `routed_moe` gives prequantized expert
  weights their own source-precision floor without restricting BF16 dense MLPs

### 3.3 Roofline-Guided Search

**Decision**: Generate valid configurations with the sensitivity-weighted
precision constraints, then evaluate them using a hardware-aware decode
Roofline order.

```python
score = Σ(sensitivity[partition] × precision_score[mode])
```

The search starts from a hardware-specific routed-MoE-only anchor when routed
experts are present, otherwise from a dense-MLP-only anchor:

- MI300/MI325: ``routed_moe=ptpc_fp8`` or ``dense_mlp=ptpc_fp8``
- MI355: ``routed_moe=mxfp4`` or ``dense_mlp=mxfp4``

The evaluation engine preserves vLLM's default ``auto`` MoE backend selection.
Explicit ``auto``, standard ``triton``, ``emulation``, ``aiter``, and
``aiter_mxfp4_bf16`` selections are accepted. For prequantized
``AITER_MXFP4_BF16`` sources, Quark retains the packed source weights and
injects target activation QDQ between the two AITER MoE stages. AITER one-stage
execution fails closed when target a2 QDQ is required. ``triton_unfused`` and
prequantized Quark W4A8 MoE sources that require inverse conversion remain out
of scope.

The performance score is bottom-up: exact Linear shapes from the meta-device
model feed GEMM formulas, routed experts feed a gated FusedMoE formula, and
full-attention layers feed SDPA. Each operator contributes the slower of its
compute and HBM times. This per-op result is the primary ranking throughput;
the aggregate weight/KV memory formula is retained as a diagnostic and fallback
when operator geometry is unavailable.

Roofline ranking is decode-only and uses the already-loaded Transformers meta
model; it does not import vLLM or predict runtime projection fusion. HF
projections are scored as separate logical GEMMs. Config-invariant Conv1D,
normalization/gating are omitted. Full-attention layers include SDPA, while
hybrid models also include an ideal Gated-Delta decode/prefill core with BF16
Q/K/V and FP32 recurrent-state traffic. Excluded modules retain their loaded
native dtype, including excluded routed-expert weights inside FusedMoE.

For op traffic, each module's effective ``QLayerConfig`` supplies independent
weight, input-compute, and output dtypes. HBM input traffic follows Hyperloom's
ideal policy and reads the source activation once at a minimum of bf16. Thus
W4A4O16 records ``W/Aread/Acompute/O = 0.5/2/0.5/2`` BPE while W4A8O16 records
``0.5/2/1/2``. It does not additionally charge the input-quant kernel's
low-precision buffer write/read. The legacy mode-to-weight-BPE map is used by
the aggregate-memory fallback only.

The default estimate uses ``ISL=8192`` and ``OSL=1024``. A separate batch-1
prefill report is printed after the decode table with per-op, memory-only, and
aggregate compute-only tok/s. Prefill values are diagnostic and never change
candidate ordering.

A valid candidate moves to its adjacent higher Roofline neighbour. An invalid
anchor moves to its adjacent lower neighbour. Early stopping occurs once the
adjacent valid/invalid accuracy frontier is known.

``max_configs`` limits quantization/evaluation attempts, not Roofline
pre-ranking. All generated candidates are scored first so the hardware anchor
and neighbour order are selected from the complete search space. During a full
sweep, exhausting the preferred direction continues at the nearest unvisited
neighbour in the opposite direction.

**Rationale**:

- The precision score constrains sensible mode combinations.
- The Roofline score directly represents the performance objective.
- Adjacent navigation does not skip untested configurations at the accuracy frontier.

### 3.4 Precision Hierarchy Constraint

**Decision**: Enforce that higher-sensitivity partitions have >= precision than lower-sensitivity ones.

```text
If sensitivity(self_attn) > sensitivity(dense_mlp):
    precision(self_attn) >= precision(dense_mlp)
```

**Rationale**: Invalid to quantize sensitive layers more aggressively than less sensitive ones.

### 3.5 Evaluation Strategy

**Decision**: Use external evaluation module (`quark.contrib.llm_eval.evaluation`).

**Rationale**:

- Reuse existing, well-tested evaluation code
- Consistent evaluation across Quark projects
- Easy to extend with new metrics

---

## 4. Data Flow

Export routing is resolved before search and when resuming a completed search.
File-to-file export is enabled explicitly, for unsupported packed MXFP4 source
weights, or when `2 × restored weight bytes` exceeds available memory across
visible GPUs. The estimate reads safetensors headers, expands compressed weights
to the model compute dtype, and budgets for resident weights plus a working copy.
Available memory includes this process's unused PyTorch cache; CPU RAM is not
counted. The memory rule applies only with GPUs and safetensors. For
compressed-tensors sources, the wrapper supports `mxfp4-pack-quantized`, preserving
native experts without dequantization. Automatic routing warns and removes
calibration-dependent candidates; resumed search results retain their
configurations, scores, and order.

### 4.1 Search Workflow

```text
┌──────────────────────────────────────────────────────────────────┐
│                        Search Workflow                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Initialize                                                   │
│     ├─ Validate MixPrecisionConfig                              │
│     ├─ Create ConfigSearcher with hardware constraints           │
│     └─ Prepare evaluation data (tokenize wikitext)               │
│                                                                  │
│  2. Baseline Evaluation                                          │
│     └─ ppl_eval(model) → baseline_metrics                        │
│                                                                  │
│  3. Configuration Generation                                     │
│     ├─ Generate all mode combinations                            │
│     ├─ Filter by hardware support                                │
│     ├─ Filter by constraints (hierarchy, not-all-bf16)           │
│     └─ Sort by score (conservative → aggressive)                 │
│                                                                  │
│  4. Roofline Search Loop                                        │
│     ├─ apply_quant_config(model, config)                         │
│     ├─ ppl_eval(model) → metrics                                 │
│     ├─ Check: metrics within threshold?                          │
│     │   ├─ Yes: update best_config, move to adjacent higher score│
│     │   └─ No: move lower or stop at the known frontier          │
│     └─ reset_to_bf16(model)                                      │
│                                                                  │
│  5. Return SearchResult                                          │
│     └─ best_config, all_results, baseline_metrics, stats         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Quantization Application Flow

```text
┌─────────────────────────────────────────────────────────────────┐
│                   apply_quant_config()                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: model, QuantConfig                                      │
│                                                                 │
│  1. Check: needs_calibration(config)?                           │
│     │                                                           │
│     ├─ Static modes (fp8, mxfp4_fp8):                          │
│     │   └─ apply_static_config()                                │
│     │       ├─ create_qconfig_from_quant_config()               │
│     │       ├─ get_calib_dataloader()                           │
│     │       └─ ModelQuantizer.quantize_model()                  │
│     │                                                           │
│     └─ Dynamic modes (ptpc_fp8, mxfp4, mxfp6_e2m3):            │
│         └─ apply_dynamic_config()                               │
│             ├─ categorize_layers()                              │
│             │    → {self_attn, dense_mlp, routed_moe}           │
│             └─ For each layer:                                  │
│                 ├─ get_layer_mode() → mode                      │
│                 └─ QuantLinear.from_float() or init_quantizer() │
│                                                                 │
│  Output: quantized model                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Key Data Structures

### 5.1 QuantConfig

Type alias for quantization configuration dictionary.

```python
QuantConfig = dict[str, str]

# Structure:
{
    "self_attn_mode": "fp8" | "ptpc_fp8" | "mxfp4" | ...,
    "dense_mlp_mode": "native" | "fp8" | ...,
    "routed_moe_mode": "native" | "mxfp4" | ...,
    "shared_expert_mode": "native" | <same as routed_moe_mode>,  # MoE only, dependent
    "kv_cache_mode": "bf16" | "fp8",
    "attention_mode": "bf16",
}
```

### 5.2 Configuration Score

Used for ranking configurations in search.

```python
# Precision scores (higher = more conservative)
PRECISION_SCORES = {
    "bf16": 10,
    "ptpc_fp8": 9,
    "fp8": 8,
    "mxfp6_e2m3": 6,
    "mxfp4_fp8": 4,
    "mxfp4": 2,
}

# Default sensitivity (higher = more sensitive to quantization)
DEFAULT_PARTITION_SENSITIVITY = {
    "linear_attn": 3,
    "self_attn": 3,
    "dense_mlp": 1,
    "routed_moe": 1,
    "shared_expert": 1,  # inherits routed_moe sensitivity when present
}
```

### 5.3 Layer Categories

Output of `categorize_layers()` function.

```python
# Structure:
{
    "self_attn": {"layer.0.self_attn.q_proj", "layer.0.self_attn.k_proj", ...},
    "dense_mlp": {"layer.0.mlp.gate_proj", "layer.0.mlp.up_proj", ...},
    "routed_moe": {"layer.1.mlp.experts.0.gate_proj", ...},
    "shared_expert": {"layer.0.mlp.shared_experts.gate_proj", ...},  # MoE only
}
```

---

## 6. Extension Points

### 6.1 Adding New Hardware Target

1. Add enum value to `HardwareTarget`
2. Add supported modes to `HARDWARE_SCHEMES`

```python
# In config.py
class HardwareTarget(Enum):
    MI400 = "mi400"  # New hardware

HARDWARE_SCHEMES[HardwareTarget.MI400] = ["bf16", "fp8", "new_mode"]
```

### 6.2 Adding New Quantization Mode

1. Add mode to type alias and constants
2. Add scheme mapping in `get_layer_config()`
3. Classify as static or dynamic in switcher.py

```python
# In config.py
QuantMode = Literal[..., "new_mode"]
ALL_QUANT_MODES.append("new_mode")
PRECISION_SCORES["new_mode"] = 5

def get_layer_config(mode):
    if mode == "new_mode":
        return NewModeScheme().config
```

### 6.3 Adding New Search Granularity

1. Add enum value to `SearchGranularity`
2. Create new `*SearchConfig` dataclass
3. Implement search logic in `MixPrecisionQuantizer`
4. Update `validate()` to allow new granularity

### 6.4 Adding New Model Architecture

Add pattern mapping to `LAYER_PATTERNS` in utils.py:

```python
LAYER_PATTERNS["new_model"] = {
    "self_attn": ["*custom_attn*", "*q_proj*", ...],
    "dense_mlp": ["*custom_ffn*", ...],
    "routed_moe": ["*experts*", ...],  # optional, MoE models
    "shared_expert": ["*shared_expert*", ...],  # optional, MoE models
}
```

---

## 7. Constraints and Limitations

### 7.1 Current Constraints

| Constraint | Description |
|------------|-------------|
| Not-all-bf16 | At least one partition must be quantized |
| Precision hierarchy | self_attn.precision >= dense_mlp/routed_moe precision according to sensitivity |
| Shared-expert dependency | shared_expert mode must be native or equal to routed_moe mode (not an independent search dimension) |
| Hardware modes | Only modes supported by target hardware |

### 7.2 Current Limitations

| Limitation | Impact | Future Work |
|------------|--------|-------------|
| MODULE granularity only | Cannot optimize per-layer | Implement DECODER_LAYER, LINEAR_LAYER |
| PPL metric only in loop | GSM8K evaluated separately | Integrate GSM8K return values |
| Sequential search | Slow for large config space | Parallel evaluation |
| No sensitivity analysis | Uses fixed sensitivity weights | Gradient/Hessian-based sensitivity |

---

## 8. Future Design Considerations

### 8.1 DECODER_LAYER Granularity

**Design Approach**:

- Group layers by position (early, middle, late) or sensitivity
- Each group can have different quantization config
- Config space: O(modes^partitions × groups)

```python
@dataclass
class DecoderLayerSearchConfig:
    layer_groups: list[list[int]]  # e.g., [[0-10], [11-20], [21-31]]
    sensitivity_method: str = "gradient"  # or "activation"
```

### 8.2 LINEAR_LAYER Granularity

**Design Approach** (Reference: NVIDIA Model-Optimizer):

- Compute sensitivity score per linear layer
- Rank layers by sensitivity
- Assign modes based on ranking

```python
@dataclass
class LinearLayerSearchConfig:
    sensitivity_method: str = "hessian"  # or "weight", "activation"
    top_k_sensitive: int = None  # Keep top-k in high precision
    candidate_modes: list[str] = ["bf16", "fp8", "mxfp4"]
```

### 8.3 Multi-Objective Optimization

**Design Approach**:

- Consider both accuracy and latency
- Pareto-optimal configuration selection

```python
@dataclass
class MultiObjectiveConfig:
    accuracy_weight: float = 0.7
    latency_weight: float = 0.3
    latency_estimator: Callable  # Model latency estimation
```
