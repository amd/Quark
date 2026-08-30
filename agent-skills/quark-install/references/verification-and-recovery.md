# Verification and recovery

## Required checks

Run checks with the exact Python executable used for installation.

```bash
python -c "import quark; print(quark.__version__)"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip); print(torch.cuda.is_available(), torch.cuda.device_count())"
```

For a PyTorch flow, optionally force first-run kernel compilation when requested:

```bash
python -c "import quark.torch.kernel; print('Quark Torch kernels OK')"
```

For an ONNX flow:

```bash
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
python -c "import quark.onnx; print('quark.onnx OK')"
python -c "import quark.onnx.operators.custom_ops; print('Quark ONNX custom ops OK')"
```

The custom-op check may compile code on first import. Treat it as a mutation/cost-bearing step and confirm first.

## Common recovery paths

| Evidence | Likely cause | Next action |
|---|---|---|
| `No module named quark` | Wrong environment or install failed | Compare `which python` with `python -m pip --version`; reinstall only after confirmation |
| `torch.cuda.is_available()` is false on a requested GPU | CPU wheel or backend mismatch | Inspect `torch.version.cuda` and `torch.version.hip`; select a release-matched backend wheel |
| Torch kernel import cannot compile | Compiler/toolkit unavailable | Check `g++`, `hipcc`/`nvcc`, `ROCM_PATH`/`CUDA_HOME`; consider a compatible pre-built wheel |
| ONNX provider missing | Wrong ORT variant or unsupported backend | Inspect installed `onnxruntime*` distributions and current provider support |
| ONNX custom-op symbol/ABI error | ORT changed after compilation | Confirm compatible ORT pin, clear only the identified build cache, and rebuild after approval |
| Python is outside the supported range | Unsupported dependency set | Create a fresh Python 3.11-3.13 environment; do not force incompatible pins into the old one |

When recovery would replace packages, preserve a snapshot first:

```bash
python -m pip freeze > quark-environment-before.txt
```
