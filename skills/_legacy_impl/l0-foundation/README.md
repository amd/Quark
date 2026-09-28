# L0 Foundation

This layer is reserved for environment sensing and safety checks such as platform detection, Quark version discovery, path validation, and workspace preflight checks.

Available L0 skills:

- `quark-env-preflight`: normalize OS, Python, container, accelerator, and toolchain facts before downstream install or planning work starts.
- `quark-workspace-validate`: validate repo shape, local paths, model references, and output locations before downstream skills depend on them.

These skills act as shared preconditions for `quark-torch-router`, `quark-torch-install`, `quark-install`, and `quark-torch-model-intake`.

L0 skills may update `session_context.json` with raw facts and `open_questions`, but they must not choose install commands, PTQ schemes, or workflow branches.
