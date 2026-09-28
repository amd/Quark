# Governance And Validation

## Evidence boundary

Repository-relative `source_knowledge`, local `Provenance` sections, and the files they identify
are the baseline for maintainer claims. For an installed Quark version, public CLI help,
distribution metadata, imported public API, and observed runtime results are valid evidence.

External web pages and out-of-tree repositories may provide leads, but cannot replace repository
provenance or installed public CLI/API evidence. If evidence is unavailable, mark the claim
unverified rather than filling the gap from memory.

## Validation by profile

- **Self-contained:** validate the top-level body, local links and contracts, executable helpers, declared outputs, checkpoints, evals, and explicit runtime integrations against current evidence.
- **Transitional:** validate the public stub's name, routing description, and delegation path, then validate the delegated body as the authority for flow, artifacts, checkpoints, and recovery; run the legacy mechanical validator on that body.

Profile rules come from [skill-format-contract.md](skill-format-contract.md); artifact resolution
comes from [artifact-contracts.md](artifact-contracts.md).

## Internal maintainer tools

The following seven tools live under `skills/_legacy_impl/meta/`. They are internal and have no
public top-level discovery entry. “Read-only” means they do not modify inspected skills or upstream
material; they may write a validation report.

| Scope | Tool | Authority |
|---|---|---|
| Shared | [`quark-skill-creator`](../../skills/_legacy_impl/meta/shared/quark-skill-creator/SKILL.md) | Write-capable authoring tool that scaffolds or restructures skills and runs validation after explicit approval. |
| Torch | [`quark-torch-doc-drift-check`](../../skills/_legacy_impl/meta/torch/quark-torch-doc-drift-check/SKILL.md) | Read-only fact-check of user-facing Torch guidance against repository and installed evidence. |
| Torch | [`quark-torch-skill-sync`](../../skills/_legacy_impl/meta/torch/quark-torch-skill-sync/SKILL.md) | Audit and classify Torch drift, and apply targeted skill or contract fixes only after confirmation. |
| Torch | [`quark-torch-eval-runner`](../../skills/_legacy_impl/meta/torch/quark-torch-eval-runner/SKILL.md) | Read-only manual evaluation of Torch routing, planning, artifacts, and recovery. |
| ONNX | [`quark-onnx-doc-drift-check`](../../skills/_legacy_impl/meta/onnx/quark-onnx-doc-drift-check/SKILL.md) | Read-only fact-check of user-facing ONNX guidance against repository and installed evidence. |
| ONNX | [`quark-onnx-skill-sync`](../../skills/_legacy_impl/meta/onnx/quark-onnx-skill-sync/SKILL.md) | Audit and classify ONNX drift, and apply targeted skill or contract fixes only after confirmation. |
| ONNX | [`quark-onnx-eval-runner`](../../skills/_legacy_impl/meta/onnx/quark-onnx-eval-runner/SKILL.md) | Read-only manual evaluation of ONNX routing, planning, artifacts, recovery, and backend isolation. |

Skill-sync may edit approved skill-system targets, but not the Quark source, examples, or
documentation being used as upstream evidence.

## Release checks

1. Confirm the current tree, documentation index, catalog, and discovery aliases agree.
2. Run profile-appropriate mechanical, semantic, contract, link, and runtime checks for every affected entry.
3. Run the relevant read-only drift check and routing, planning, artifact, and recovery evaluations; resolve approved findings through the matching sync tool.
4. Require Markdown lint and local-link checks to pass.
5. Publish only claims supported by the merged repository state or observed installed-version evidence, and record blockers or unverified claims.
