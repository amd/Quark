# Legacy Agent Skill Implementations

`skills/_legacy_impl/` temporarily stores delegated implementations and internal maintainer tools created before public skills became self-contained.
Canonical user-facing entries live under `skills/`, while `.claude/skills` and `.agents/skills` are discovery aliases only.

## New skills

**Every new public skill must be self-contained under `skills/<skill-name>/`.**
Do not add a new public stub that delegates to `_legacy_impl/`, and do not place a new user-facing implementation in this directory.
The existing Quant-Perf fixed-strategy delegation is a documented grandfathered integration, not a pattern for new skills.
The legacy `quark-skill-creator` scaffolder targets the old delegated layout and must not be used for a new public skill until it supports the self-contained profile.

A self-contained skill owns all instructions and directly required references, contracts, helpers, evals, ownership, and license material in its public directory.
It must not load its primary instruction body or runtime resources from another skill or from `_legacy_impl/`.
Any required installed package, CLI, external tool, model, or source checkout must be declared explicitly rather than hidden behind a repository-relative dependency.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the required creation and validation flow.

## Existing contents

The current public tree contains three self-contained entries and 16 transitional stubs.
Each transitional stub owns its public name and routing description, while its delegated body under this directory remains authoritative for execution, artifacts, checkpoints, and recovery.
Internal routing, planning, governance, templates, and shared contracts also remain here until their owners are migrated or retired.

```text
skills/
├── <self-contained-skill>/
├── <transitional-skill>/SKILL.md
└── _legacy_impl/
    ├── <layer>/<scope>/<implementation>/
    ├── meta/<scope>/<maintainer-tool>/
    └── shared/{contracts,templates}/
```

`<scope>` is `shared`, `torch`, or `onnx`; `<layer>` is `l0-foundation`, `l1-atomic`, `l2-workflows`, or `l3-recipes`.
Meta tools are grouped under `meta/<scope>/`.

## Maintenance rules

- Make the smallest necessary fix in an existing delegated implementation.
- Do not create new dependencies on `_legacy_impl/`.
- Keep the public stub name, routing description, and delegation target valid.
- Treat schemas under `shared/contracts/` as transitional contracts for delegated and internal legacy implementations.
- Use repo-root-relative paths for provenance and source knowledge.
- Prefer migration over expansion when a legacy skill needs substantial work.

## Migration

To migrate a public skill, move its complete instruction body and every directly required runtime resource into `skills/<skill-name>/`.
Replace the public stub with the self-contained body, update local links and contracts, test the copied directory outside `_legacy_impl/`, and remove the old implementation only after no references remain.
Do not leave two authoritative copies.

Current inventory and copy rules are in [`skills/README.md`](../README.md).
Architecture, contracts, catalog, and governance are in [`docs/agent_skills/`](../../docs/agent_skills/README.md).
