# llama.cpp GGUF Export

Independent export path for Quark AWQ checkpoints targeting **public
llama.cpp GGUF quant types**.

This module is separate from:

- ``export_safetensors`` (Quark/AWQ safetensors export)
- ``export_gguf`` (legacy narrow Quark-native GGUF export)

## Flow

```text
Quark AWQ safetensors
  -> in-memory signed INT4 unpack
  -> libggml quantizer (llama.cpp)
  -> GGUF writer (llama.cpp conversion/)
```

No FP16/BF16 Hugging Face checkpoint is written to disk.

## API

```python
from quark.torch import export_llama_cpp_gguf, list_llama_cpp_export_formats

print(list_llama_cpp_export_formats())
export_llama_cpp_gguf(
    quark_model_dir="/path/to/quark-awq",
    output_dir="/path/to/out",
    export_format="q4_k_m",
    tokenizer_source="/path/to/base-tokenizer",
    llama_cpp_dir="/path/to/llama.cpp",
)
```

## CLI

```bash
python examples/torch/language_modeling/llm_ptq/export_llama_cpp_gguf.py \
  --model /path/to/quark-awq \
  --output-dir /path/to/out \
  --format q4_k_m \
  --tokenizer-source /path/to/tokenizer
```

## Requirements

- ``gguf>=0.10.0``
- Built llama.cpp with ``libggml.so``
- llama.cpp tree with ``gguf-py`` and ``conversion/`` (plus local tensor-mapping
  patches for models such as Qwen3.5 MoE)

## Tests

```bash
LLAMA_CPP_DIR=/path/to/llama.cpp \
pytest test/test_for_torch/test_llama_cpp_export.py -v
```

The parametrized roundtrip tests exercise each libggml-backed export format
against the local llama.cpp build.
