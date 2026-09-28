---
name: quark-torch-shrink-model
description: >
  Shrink a HuggingFace safetensors model to 1 hidden layer for fast debugging without loading
  the full model into memory.
layer: l1-atomic
primary_artifact: shrink_result.md
source_knowledge:
  - skills/_legacy_impl/l1-atomic/torch/quark-torch-shrink-model/shrink_model.py
---

# quark-torch-shrink-model

## Purpose

Produce a minimal 1-layer copy of any HuggingFace safetensors model for debugging Quark
workflows. The tool reads `model.safetensors.index.json` to determine layer structure and
rewrites only the necessary shards — the full model is never loaded into memory.

Non-layer weights (embeddings, final norm, lm_head) are always preserved so the output
is a structurally valid model that can be loaded with `transformers`.

## Inputs

- `source_model_directory` — local directory containing safetensors files and `config.json`
- `destination_model_directory` — where to write the shrunk model
- `test_mode` (optional) — if the user wants only JSON files without tensor data, for structural validation

## How to invoke

### Understand the user's intent first

Ask (or infer from context):

1. **Source path**: where is the model? (required)
2. **Destination path**: where to save the shrunk model? (required)
3. **Test mode?**: does the user just want to validate the structure without writing tensors?
   - "quick check", "just test", "no tensors", "only json" → use `--test`
   - "real model", "load and run", "actual weights" → full mode (no `--test`)

### CLI invocation

The script lives in the skill directory and is invoked directly by path:

```bash
SKILL_DIR="skills/_legacy_impl/l1-atomic/torch/quark-torch-shrink-model"

# Full shrink (writes real safetensors shards)
python "$SKILL_DIR/shrink_model.py" \
    --src /path/to/source/model \
    --dst /path/to/output/tiny_model

# Test mode (JSON only, no safetensors — fast structural validation)
python "$SKILL_DIR/shrink_model.py" \
    --src /path/to/source/model \
    --dst /path/to/output/tiny_model \
    --test
```

### Python API

```python
import sys
from pathlib import Path

skill_dir = Path("skills/_legacy_impl/l1-atomic/torch/quark-torch-shrink-model")
sys.path.insert(0, str(skill_dir))
from shrink_model import shrink_model

shrink_model(
    source_model_directory=Path("/path/to/source/model"),
    destination_model_directory=Path("/path/to/output/tiny_model"),
    test_mode=False,  # set True for JSON-only structural validation
)
```

## Supported architectures

The tool auto-detects the layer naming convention from the weight keys:

| Pattern | Architectures |
|---------|--------------|
| `model.layers.N.` | LLaMA, Qwen, Mistral, Gemma, DeepSeek-V3/R1 |
| `layers.N.` (no prefix) | DeepSeek-V4 |
| `transformer.h.N.` | GPT-2, Falcon |
| `model.blocks.N.` | MPT |
| `model.transformer.layer.N.` | BERT-style |

If the user's model uses a different pattern, add a new entry to `_LAYER_INDEX_PATTERNS`
in `skills/_legacy_impl/l1-atomic/torch/quark-torch-shrink-model/shrink_model.py`.

## Output: shrink_result.md

After running, produce a brief report:

```markdown
## Shrink Result

- **Source**: /path/to/source/model
- **Destination**: /path/to/output/tiny_model
- **Mode**: full / test
- **Layers detected**: 80 (0 ... 79)
- **Layer kept**: 0 → remapped to 0
- **Keys kept**: 12 / 723
- **config.json**: num_hidden_layers 80 → 1
- **Status**: success
```

If the run fails, include the error and the most likely fix:

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `Could not detect any layer indices in weight_map` | Unsupported key naming | Add pattern to `_LAYER_INDEX_PATTERNS` |
| `Neither model.safetensors.index.json nor model.safetensors found in` | Wrong source path | Verify `--src` points to the model directory |
| `ImportError: safetensors is required: pip install safetensors` | safetensors not installed | `pip install safetensors` |
