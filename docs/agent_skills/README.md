# Quark Agent Skills Documentation

## Current state

`skills/` is the canonical, product-owned skill tree. `.claude/skills` and `.agents/skills` contain real-file discovery adapters that forward directly to `skills/`; neither is a second authored source tree.

The public surface contains 19 top-level `quark-*` entries: three shared, eight Torch, and eight
ONNX. Three entries are self-contained:

- `quark-install`
- `quark-torch-ptq`
- `quark-torch-quant-perf`

The other 16 are transitional stubs whose implementations remain under `skills/_legacy_impl/`;
they are not portable without that tree.

`quark-install` and `quark-torch-ptq` can be copied independently. Automatic search also works
from a copied `quark-torch-quant-perf` entry with `amd-quark[quant_perf]`, but still needs an
external accelerator and vLLM stack. Fixed `--quant-strategy` requires a discoverable Quark
checkout that contains `quark-torch-ptq`; set `QUARK_ROOT` only when automatic discovery fails
or when explicitly selecting a checkout.

## Documents

- [Architecture and naming](architecture.md)
- [Interaction contract](interaction-contract.md)
- [Artifact contracts](artifact-contracts.md)
- [Skill format contract](skill-format-contract.md)
- [Governance and validation](governance.md)
- [Public skills catalog](skills_catalog.md)
