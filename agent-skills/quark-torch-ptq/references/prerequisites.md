# PyTorch PTQ prerequisites

Use one Python environment with a supported AMD Quark release and a PyTorch build matching the accelerator.

## Inspect

```bash
python --version
python -m pip show amd-quark torch transformers
python -c "import quark, torch; print(quark.__version__, torch.__version__); print(torch.version.cuda, torch.version.hip); print(torch.cuda.is_available(), torch.cuda.device_count())"
```

For GPU use, a false `torch.cuda.is_available()` or a backend mismatch is a blocker. AMD ROCm builds still expose devices through PyTorch's `torch.cuda` API; confirm `torch.version.hip` for ROCm.

## Install planning

Use the release-matched [Quark installation guide](https://quark.docs.amd.com/latest/install.html). Confirm before changing packages. The default Quark distribution is `amd-quark`; GPU PyTorch must come from the correct backend index.

The LLM PTQ example may require additional packages such as Transformers, Accelerate, Datasets, Evaluate, GGUF, and lm-eval. Use the `requirements.txt` shipped beside the release-matched example instead of copying an unpinned package list from another release.

## Source of runnable examples

Use one of:

- `examples/torch/language_modeling/llm_ptq` in a Quark checkout pinned to the installed version;
- the examples bundle distributed with the same Quark release.

Record `git rev-parse HEAD` or the release archive version in the run manifest.
