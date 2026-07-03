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

**Decision**: Use two partitions: `self_attn` and `mlp`.

```text
self_attn: q_proj, k_proj, v_proj, o_proj (attention projections)
mlp:       gate_proj, up_proj, down_proj, experts (feed-forward layers)
```

**Rationale**:

- Attention and MLP have different sensitivity to quantization
- Two partitions balance configuration space vs. flexibility
- Pattern-based matching handles different model architectures

### 3.3 Prior-Driven Search

**Decision**: Sort configurations by sensitivity-weighted precision score.

```python
score = Σ(sensitivity[partition] × precision_score[mode])
```

**Rationale**:

- Higher score = more conservative (higher precision)
- Search from conservative to aggressive
- Stop when threshold exceeded (most aggressive valid config found)

### 3.4 Precision Hierarchy Constraint

**Decision**: Enforce that higher-sensitivity partitions have >= precision than lower-sensitivity ones.

```text
If sensitivity(self_attn) > sensitivity(mlp):
    precision(self_attn) >= precision(mlp)
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
│  4. Search Loop (for each config)                                │
│     ├─ apply_quant_config(model, config)                         │
│     ├─ ppl_eval(model) → metrics                                 │
│     ├─ Check: metrics within threshold?                          │
│     │   ├─ Yes: update best_config, continue                     │
│     │   └─ No: early_stop? break : continue                      │
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
│             ├─ categorize_layers() → {self_attn, mlp}           │
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
    "mlp_mode": "bf16" | "fp8" | ...,
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
    "self_attn": 3,
    "mlp": 1,
}
```

### 5.3 Layer Categories

Output of `categorize_layers()` function.

```python
# Structure:
{
    "self_attn": {"layer.0.self_attn.q_proj", "layer.0.self_attn.k_proj", ...},
    "mlp": {"layer.0.mlp.gate_proj", "layer.0.mlp.up_proj", ...},
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
    "mlp": ["*custom_ffn*", ...],
}
```

---

## 7. Constraints and Limitations

### 7.1 Current Constraints

| Constraint | Description |
|------------|-------------|
| Not-all-bf16 | At least one partition must be quantized |
| Precision hierarchy | self_attn.precision >= mlp.precision (if sensitivity[self_attn] > sensitivity[mlp]) |
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
