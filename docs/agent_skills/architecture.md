# Architecture And Naming

This document defines responsibility layers, names, placement, and the migration boundary. See
[README.md](README.md) for the current inventory and migration status.

## Responsibility layers

- **L0 foundation:** collect and normalize environment or workspace facts without choosing a workflow or mutating the target.
- **L1 atomic:** perform one reusable action with one primary result and bounded recovery.
- **L2 workflow:** coordinate multiple actions, artifacts, branches, and user checkpoints end to end.
- **L3 recipe:** apply one or more workflows to a named model, deployment, customer, or hardware context.
- **Meta:** maintain, validate, evaluate, or synchronize skills rather than serving the user data path.

These labels describe functional responsibility. A self-contained public entry does not need
legacy `layer` frontmatter or the legacy body template.

## Naming

- Use lowercase kebab-case and prefix every skill with `quark-`.
- Use `quark-<verb>` for backend-agnostic shared skills and `quark-<backend>-<verb>` for
  backend-specific skills.
- Keep the backend infix aligned with its scope: `torch`, `onnx`, or no infix for `shared`.
- Prefer stable, goal-oriented verbs or action-bearing nouns; do not expose temporary
  implementation structure in a public name.
- Layer labels need not appear in public names. Internal meta names conventionally use `-sync`,
  `-eval-`, or `-drift-`.

## Placement

Public discovery entries are flat:

```text
skills/<skill-name>/SKILL.md
```

Delegated and internal implementations are grouped by responsibility and scope:

```text
skills/_legacy_impl/<layer>/<scope>/<implementation-name>/SKILL.md
```

`<scope>` is `shared`, `torch`, or `onnx`, including under `meta/`. A backend-specific
implementation name must match its parent scope.

## Migration boundary

A self-contained entry owns its public instructions and any directly needed references,
contracts, helpers, evals, ownership, and license material inside `skills/<skill-name>/`; its
top-level body does not delegate.

A transitional entry keeps a thin public routing stub while its authoritative body remains under
`_legacy_impl/`. It is repository-coupled until those instructions and owned resources move into
the public directory. Migration moves ownership instead of duplicating two authoritative bodies;
after the move, remove the delegation. Format authority is defined in
[skill-format-contract.md](skill-format-contract.md), and artifact ownership in
[artifact-contracts.md](artifact-contracts.md).
