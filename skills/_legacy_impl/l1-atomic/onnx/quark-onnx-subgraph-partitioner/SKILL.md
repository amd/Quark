---
name: quark-onnx-subgraph-partitioner
description: >
  Partition an ONNX model graph into named functional subgraphs and emit a
  subgraph_partition.json file. Use when the user wants to understand a model's
  high-level structure, document its architectural blocks, or prepare a
  partition for downstream workflows (mixed-precision quantization, layer-wise
  profiling, partial deployment, graph visualization). Triggers on "partition
  my ONNX model", "generate a subgraph JSON", "which nodes belong to the
  attention blocks", "show me the model architecture", "split the model into
  blocks", "group layers for AMP", or any request that needs a named block
  decomposition of an ONNX graph. Handles any ONNX architecture: CNNs
  (ResNet, EfficientNet, YOLOv5/v8, DenseNet, MobileNet), Transformers
  (BERT, ViT, Swin, LLMs), multi-modal (CLIP, BLIP), detection models
  with FPN/PAN necks, BEV perception models (BEVFormer), and custom or
  unfamiliar architectures identified from node-name prefixes and op sequences.
layer: l1-atomic
primary_artifact: subgraph_partition.json
source_knowledge:
  - quark/onnx/algorithm/mprecision/subgraph_parser.py
  - quark/onnx/algorithm/mprecision/mprecision_config.py
  - quark/onnx/algorithm/mprecision/auto_mixprecision.py
  - quark/onnx/quantization/quant_utils.py
  - docs/source/onnx/accuracy_algorithms/amp.rst
---

# quark-onnx-subgraph-partitioner

## Purpose

Produce a `subgraph_partition.json` that groups ONNX graph nodes into named
functional blocks (stem, backbone stage, attention layer, FFN, FPN neck, etc.).
The partition can serve as input to AMP sensitivity analysis, layer-wise
profiling, partial deployment, or graph documentation.

Two helper scripts live alongside this file:

```bash
SKILL_DIR=skills/_legacy_impl/l1-atomic/onnx/quark-onnx-subgraph-partitioner
```

## Inputs

- Model path — a single `.onnx` file supplied as `$ARGUMENTS`
- Optional architecture hints from the user (layer count, naming convention,
  block granularity preference)

## Outputs: subgraph_partition.json

Schema: [`subgraph_partition.schema.json`](../../shared/contracts/subgraph_partition.schema.json)

```json
{
  "quantized":     false,
  "num_subgraphs": 4,
  "subgraphs": [
    {
      "name": "backbone_stem",
      "description": "Initial stride-2 Conv → SiLU activation, produces P1 features",
      "start_nodes": ["/model.0/conv/Conv"],
      "end_nodes":   ["/model.0/act/Mul"]
    },
    {
      "name": "attention_layer_0",
      "description": "Full encoder layer 0: self-attention + FFN + LayerNorm + residuals",
      "start_nodes": ["/encoder/layer.0/attention/self/query/MatMul"],
      "end_nodes":   ["/encoder/layer.0/output/LayerNorm"]
    }
  ]
}
```

Every entry requires all four fields: `name`, `description`, `start_nodes`,
`end_nodes`. **Never include `resolved_nodes`** — node resolution happens at
runtime. `num_subgraphs` must equal `len(subgraphs)`; set it last.

## Analysis Procedure

### Step 1 — Load and inspect the model

```bash
python3 "$SKILL_DIR/inspect_model.py" <MODEL_PATH>
```

For quantized models (detected automatically via Q/DQ op presence), Q/DQ
wrapper nodes are suppressed so structural landmarks remain visible; Q/DQ
counts are reported separately.

**Text `.onnxtxt`** — use `grep` instead:

```bash
grep -n 'op_type:\|  name:\|input:\|output:' <MODEL_PATH> | head -400
```

**Quantized model rules:** use compute nodes (`Conv`, `MatMul`, `Add`, …) as
boundaries — never Q/DQ nodes. BFS from a compute node captures surrounding
Q/DQ wrappers automatically.

### Step 2 — Identify structural landmarks

| Signal | Likely boundary |
|--------|-----------------|
| First `Conv` consuming the graph input | Backbone / model stem |
| Repeated `/model.N/` or `/layer.N/` prefix groups | Stage or layer repetition |
| `MaxPool` or strided `Conv` (stride 2) | Resolution downsampling |
| `Resize` (upsample) + `Concat` on lateral path | FPN / PAN top-down merge |
| `Split` → N parallel `Conv` → `Concat` + 1×1 `Conv` | C2f / CSP bottleneck |
| `GlobalAveragePool` → FC → activation → `Mul` | SE block |
| `MatMul`×3 → `Softmax` → `MatMul` (QKV pattern) | Self-attention block |
| `LayerNormalization` or `InstanceNormalization` | Transformer / diffusion layer |
| `MatMul`/`Gemm` → activation → `MatMul`/`Gemm` (hidden × 4) | FFN / MLP block |
| `Add` immediately after attention + FFN | Residual add (layer end) |
| `GridSample` with sampling offsets | Deformable attention |
| `Gather` on index 0 → `LayerNorm` → `Gemm` | CLS token / pooler / head |

### Step 3 — Assign blocks

| Block type | Description |
|------------|-------------|
| `Stem` | Initial convs before first downsampling |
| `BackboneStage` | Stage at one resolution (N residual / bottleneck blocks) |
| `SEBlock` | Squeeze-and-Excitation (GAP → FC → activation → Mul) |
| `PatchEmbedding` | ViT patch projection (Conv with patch-stride kernel) |
| `SelfAttentionBlock` | Multi-head self-attention (QKV + Softmax + out proj) |
| `FFNBlock` | Feed-forward network (Linear → activation → Linear) |
| `TransformerLayer` | Full encoder/decoder layer (attn + FFN + norms + residuals) |
| `SPPF` | Spatial Pyramid Pooling Fast |
| `FPNNeck` | Feature Pyramid Network (lateral convs + upsample + merge) |
| `PANNeck` | Path Aggregation Network (down-path convs + concat + merge) |
| `DetectionHead` | Conv layers + decode logic for bbox / class output |
| `ClassificationHead` | MLP predicting class logits |

**Granularity rules:**

- No single subgraph > ~25 % of total nodes; split if needed.
- No trivial single-activation subgraphs; minimum unit is one compute op.
- Split every repeated layer individually (12 transformer layers → 12 entries).
- Merge inseparable op chains (Conv + BN + ReLU with no branch points).
- Assign `Concat` / `Add` / `Resize` to the subgraph that *produces* the feature map.

### Step 4 — Choose start and end nodes

**`start_nodes`**: first node(s) receiving data from outside the block. Only
list a node if it is **not already reachable** from another listed start node —
downstream-reachable additions are redundant. When a block has multiple
independent entry paths (e.g., parallel detection-head branches), list all.

**`end_nodes`**: last node(s) whose output feeds the next block.

**Residual fork hazard.** BFS stops when it *visits* an end node, but still
enqueues all consumers of every *preceding* node. If any node before the end
node fans out to an external consumer (a residual skip, a lateral FPN branch),
BFS cascades through the rest of the model. Fix: move `end_nodes` back to the
**last node whose output has no external consumers**, i.e. the node immediately
before the fork. Check 5 in `validate_partition.py` catches this automatically
by flagging subgraphs that resolve to > 25 % of total nodes.

Copy all node names verbatim from the Step 1 listing — never reconstruct them.

### Step 5 — Validate the partition

```bash
python3 "$SKILL_DIR/validate_partition.py" <MODEL_PATH> <DRAFT_JSON>
```

Runs five checks (see `validate_partition.py` docstring for details). Fix all
`ERROR:` lines before proceeding. Common fixes:

- Replace `Constant` start nodes with the `Resize` / `Conv` that consumes them.
- Replace initializer names with the node that reads them.
- Adjust `end_nodes` so all end nodes are reachable from `start_nodes`.
- Move `end_nodes` one step earlier to avoid residual-fork BFS explosion.

Also verify **coverage**: confirm the union of resolved subgraphs covers the
model's quantizable ops (`Conv` / `MatMul` / `Gemm`). Explain any ops that land
in `__ungrouped__`.

### Step 6 — Emit the JSON file

Write `subgraph_partition.json` beside the model file (or in the working
directory). Before writing, verify:

- All four fields on every entry; no duplicate `name` values.
- `num_subgraphs` equals `len(subgraphs)` (count programmatically).
- All Step 5 validation checks pass with zero errors.

### Step 6 — Summary table

Always finish with:

| # | Name | Block type | Start node | End node | Est. nodes |
|---|------|------------|-----------|---------|-----------|

Include an **Architectural note**: overall architecture family, number and type
of repeated blocks, and any non-obvious design choices visible in the graph.

## Rules

- **All four fields required** on every entry; omitting `description` is not acceptable.
- **Never fabricate node names.** Copy from Step 1 listing. If unconfirmable,
  omit rather than guess.
- **Never use Constant / ConstantOfShape** as start/end node (shape side-branches).
- **Never use Q/DQ wrapper nodes** as start/end node; always anchor on compute nodes.
- **Never use initializer or graph-input names** as node names.
- **Never use an empty string** as a node name.
- **Never include `resolved_nodes` / `nodes`** in the output.
- **Split every repeated layer** separately (N layers → N entries).
- **Cover the whole model** and explain any `__ungrouped__` nodes.
- **Run Step 5 validation** and fix all errors before writing the file.
- **Always write the file to disk** and report its absolute path.

## Interaction Flow

1. Confirm the model path from `$ARGUMENTS`; resolve to absolute path.
2. Run Step 1 — print the node listing.
3. Run Steps 2–4 — identify landmarks, assign blocks, select node names. Ask
   one focused question if granularity is ambiguous.
4. Run Step 5 — fix all validation errors before proceeding.
5. Run Step 6 — write `subgraph_partition.json`; confirm zero errors.
6. Run Step 7 — print summary table and architectural note.
7. Confirm output path and mention typical next steps:
   - *AMP*: `AutoMixprecisionConfig(subgraph_json="<path>")`
   - *Profiling*: iterate over `resolved_nodes` per block
   - *Documentation*: open alongside Netron

## Recovery

| Failure | Action |
|---------|--------|
| `onnx` not installed | Run `pip install onnx` and retry Step 1 |
| Model > 2 GB, Python OOM | Print only `n.name` and `n.op_type`; skip weight shapes |
| All node names are anonymous (`/Add_7`) | Use op-type sequences and output shapes; use index ranges as block names |
| Validation: node not found | Re-check Step 1 listing; never guess — omit if unconfirmable |
| Validation: name is an initializer | Use the graph node that *reads* the initializer instead |
| Validation: start node is `Constant` | Replace with the data-path node consuming it |
| Validation: start/end is a Q/DQ wrapper | Replace with the adjacent compute node |
| Validation: end not reachable from start | Add the branching node to `start_nodes`, or adjust `end_nodes` |
| Check 5: subgraph resolves to > 25 % | Residual-fork hazard; move `end_nodes` one step earlier to the DQL before the fork |
