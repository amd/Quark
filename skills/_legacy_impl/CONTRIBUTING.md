# Contributing an Agent Skill

## Required rule for new skills

**Every new public skill must be self-contained in `skills/<skill-name>/`.**
Do not create a new delegated stub, add a new implementation under `_legacy_impl/`, or use another skill as the primary runtime instruction source.
The existing Quant-Perf fixed-strategy delegation is a grandfathered integration and is not a template for new skills.
The current legacy `quark-skill-creator` scaffolder generates the old delegated layout, so do not use it for a new public skill until it supports this self-contained flow.

A self-contained skill must keep its complete public instructions and every directly required reference, contract, helper, eval, ownership file, and license file in its own directory.
Installed packages, public APIs or CLIs, models, hardware, and external tools may be explicit inputs, but the skill must not hide a dependency on another skill or a repository-relative legacy body.

## Placement and naming

Create the skill only in the canonical tree:

```text
skills/<skill-name>/
├── SKILL.md
└── references/, scripts/, evals/, ...   # only when required
```

Do not write through `.claude/skills` or `.agents/skills`; both are discovery aliases to `skills/`.

Use lowercase kebab-case:

- Shared: `quark-<verb>`
- PyTorch: `quark-torch-<verb>`
- ONNX: `quark-onnx-<verb>`

Choose a stable user goal rather than an implementation detail for the name.

## Public entry content

Start `SKILL.md` with a stable name and a routing-focused description:

```yaml
---
name: quark-<scope-and-goal>
description: <when to use this skill, its boundary, and important exclusions>
---
```

The instruction body must state:

- purpose and supported inputs
- outputs and side effects
- interaction flow and explicit safety checkpoints
- runtime requirements and authoritative public APIs or CLIs
- bounded recovery behavior

Keep detailed procedures in local `references/` files and executable helpers in local `scripts/`.
Keep runtime schemas beside the skill that owns them.
Repository-relative provenance may identify source facts for maintainers, but it must not become an undeclared runtime dependency.

## Self-contained verification

Before review:

1. Copy only `skills/<skill-name>/` to a temporary location outside the Quark skill tree.
2. Verify that every local link and directly required file resolves inside the copied directory.
3. Confirm that the instructions never load another skill or `_legacy_impl/`.
4. Exercise the narrowest representative workflow with the declared dependencies; if required hardware or services are unavailable, record the blocker and provide a reproducible test for CI or a suitably equipped maintainer.
5. Verify produced artifacts against locally owned contracts.
6. Run the routing, planning, artifact, and recovery evals that the available environment supports, and record any blocked category.
7. Update `skills/README.md` and `docs/agent_skills/skills_catalog.md`.
8. Run repository lint and relevant tests.

Do not call a skill self-contained when the copied directory silently depends on files elsewhere in this repository.

## Maintaining transitional skills

Existing transitional entries may still delegate to:

```text
skills/_legacy_impl/<layer>/<scope>/<implementation-name>/SKILL.md
```

The public stub owns the stable name, routing description, and delegation path.
The delegated body owns execution, artifacts, checkpoints, and recovery.
Make bounded fixes in place, but do not add new legacy dependencies; substantial changes should migrate the skill to the self-contained profile.

The legacy template is [`shared/templates/skill-template.md`](shared/templates/skill-template.md).
The legacy validator applies to delegated and internal legacy bodies:

```bash
python skills/_legacy_impl/meta/shared/quark-skill-creator/scripts/validate_skill.py \
  skills/_legacy_impl/<layer>/<scope>/<implementation-name>
```

Some grandfathered implementations may report existing findings; record them and avoid expanding the debt.

## References

- [Current skill status and copy rules](../README.md)
- [Legacy directory purpose and migration rules](README.md)
- [Architecture and naming](../../docs/agent_skills/architecture.md)
- [Interaction and safety contract](../../docs/agent_skills/interaction-contract.md)
- [Artifact contracts](../../docs/agent_skills/artifact-contracts.md)
- [Skill format contract](../../docs/agent_skills/skill-format-contract.md)
- [Governance and validation](../../docs/agent_skills/governance.md)
