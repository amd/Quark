# Installation options

Use this as a planning checklist, then verify current commands against the official installation guide for the Quark release the user will run.

## Environment prerequisites

- Quark 0.12 documentation supports Python 3.11, 3.12, and 3.13; Python 3.14 is not supported.
- PyTorch 2.2 or later is required by the release documentation. Accelerator-specific combinations are narrower, so consult the release's `tools/ci/install_torch.sh` when working from a Quark checkout.
- The universal Quark wheel compiles optional fast kernels or ONNX custom operators on first import. Linux needs a C++ compiler such as `g++`; GPU compilation also needs `hipcc` or `nvcc` and its toolkit path.
- Pre-built Quark wheels require a compatible Python, PyTorch, operating system, and accelerator combination.

## Quark package source

| Choice | When to use | Command pattern |
|---|---|---|
| Universal PyPI wheel | Default and widest compatibility | `pip install amd-quark` |
| Pre-built CPU | Matching supported Python and PyTorch; avoid first-run compile | `pip install amd-quark --extra-index-url https://pypi.amd.com/quark/cpu/simple` |
| Pre-built CUDA 12.8 | Matching CUDA/PyTorch environment | `pip install amd-quark --extra-index-url https://pypi.amd.com/quark/cu128/simple` |
| Pre-built ROCm 7.1 | Linux and matching ROCm/PyTorch environment | `pip install amd-quark --extra-index-url https://pypi.amd.com/quark/rocm71/simple` |
| Pre-built ROCm 7.2 | Linux and matching ROCm/PyTorch environment | `pip install amd-quark --extra-index-url https://pypi.amd.com/quark/rocm72/simple` |

These are release-0.12-era options, not a permanent compatibility promise. If a requested combination is absent from the current official selector, use the universal wheel or stop and explain the gap.

## PyTorch backend

Use the [PyTorch installation selector](https://pytorch.org/get-started/locally/) or Quark's release-matched CI matrix. The command pattern is:

```bash
# ROCm: use the confirmed rocmX.Y tag
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocmX.Y

# CUDA: use the confirmed cuXYZ tag
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXYZ

# Explicit CPU environment
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Never substitute a remembered tag for a checked compatibility matrix.

## ONNX Runtime

Quark 0.12 documentation requires ONNX Runtime `>=1.22.2,<=1.25.1`.

```bash
# CPU, including the documented ROCm 7.x fallback
pip install "onnxruntime>=1.22.2,<=1.25.1"

# CUDA when the current compatibility guidance supports it
pip install "onnxruntime-gpu>=1.22.2,<=1.25.1"
```

Before changing variants, inspect `python -m pip list` and require approval for removing conflicting `onnxruntime*` packages.
