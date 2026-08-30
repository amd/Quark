---
name: quark-install
description: Install, verify, or repair AMD Quark environments for PyTorch or ONNX workflows. Use for amd-quark setup, Python and accelerator compatibility checks, ROCm/CUDA/CPU package selection, Quark import failures, ONNX Runtime provider selection, or first-run kernel and custom-operator compilation. Inspect first and require confirmation before changing packages or system dependencies.
---

# Install AMD Quark

Prepare a reproducible Quark environment without guessing the user's accelerator, Python environment, or package source. This skill is self-contained: do not delegate its steps to another skill.

## Workflow

1. Clarify the intended flow: PyTorch, ONNX-to-ONNX, or both.
2. Collect facts before proposing commands:

   ```bash
   python scripts/collect_environment.py --output env_context.json
   ```

   Also ask which Python environment may be modified and whether CPU fallback is acceptable.
3. Read [references/install-options.md](references/install-options.md). Recheck the release-matched [official installation guide](https://quark.docs.amd.com/latest/install.html) when generating commands; package and accelerator matrices change.
4. Present one installation plan containing:
   - environment name and Python executable;
   - PyTorch or ONNX Runtime variant and index;
   - Quark wheel source;
   - compiler or first-import requirements;
   - exact install and verification commands.
5. Get explicit confirmation before installing, uninstalling, upgrading, compiling, or modifying system packages.
6. Execute only the approved commands. Capture command output and package versions.
7. Run the relevant verification checks from [references/verification-and-recovery.md](references/verification-and-recovery.md).
8. Write `quark_install_result.json` with status, versions, backend, commands run, and per-check results.

## Decision rules

- Prefer the universal PyPI wheel when compatibility is uncertain. Use a pre-built AMD-index wheel only after confirming its Python, PyTorch, OS, and accelerator match.
- Install a GPU-enabled PyTorch build from the backend-specific PyTorch index. Do not use bare `pip install torch` for an intended GPU environment.
- Treat absent GPU tools as `unknown`, not automatically as CPU-only. Ask before choosing CPU.
- Never mix a CUDA PyTorch build with a ROCm environment, or the reverse.
- Install exactly one ONNX Runtime variant. On current ROCm 7.x guidance, ONNX Runtime may use the CPU package; explain that trade-off instead of inventing an unsupported ROCm wheel.
- Do not claim success from `pip` alone. Imports and backend/provider checks must pass.
- Do not silently repair an existing environment. Show the proposed uninstall/reinstall sequence and its impact first.

## Result contract

Record enough evidence to reproduce or diagnose the environment:

```json
{
  "status": "ok",
  "python": "3.13.0",
  "environment": "/path/to/python",
  "accelerator": "amd-rocm",
  "packages": {"amd-quark": "0.12.0", "torch": "..."},
  "verification": {
    "quark_import": "pass",
    "torch_backend": "pass",
    "torch_kernel": "not-run",
    "onnx_custom_ops": "not-requested"
  },
  "commands": []
}
```

Use `failed` when a required check fails and `partial` when optional compilation or a requested backend check remains unverified. Include the exact failing command and error excerpt.

## Stop conditions

Stop before mutation when the active environment is ambiguous, the requested backend conflicts with detected packages, required credentials are missing, or a wheel compatibility cannot be established. Return the evidence collected and the smallest question or manual check needed to proceed.
