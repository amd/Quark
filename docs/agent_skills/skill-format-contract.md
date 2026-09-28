# Skill Format Contract

## Public entry

Every public entry lives at `skills/<skill-name>/SKILL.md`. Its YAML frontmatter provides at least:

```yaml
---
name: <stable-kebab-case-name>
description: <trigger-oriented responsibility and boundary>
---
```

The public file determines whether the self-contained or transitional profile applies. Naming and
placement rules are in [architecture.md](architecture.md).

## Self-contained profile

- The top-level body is authoritative for interaction flow, checkpoints, outputs, recovery, and links.
- Directly owned references, contracts, helpers, evals, ownership, and license material stay in the public entry directory as needed.
- The primary instruction body does not delegate to `_legacy_impl/`.
- The entry may define sections and checkpoints suited to its own workflow; the legacy template and validator are not its format.

## Transitional profile

- The top-level stub owns the stable public `name`, routing `description`, and delegation path.
- The delegated body under `skills/_legacy_impl/<layer>/<scope>/<implementation-name>/SKILL.md` owns the executable flow, artifacts, checkpoints, and recovery.
- A stub is not made self-contained by copying legacy content selectively; migration follows the ownership boundary in [architecture.md](architecture.md).

## Mechanical validation

The [legacy validator](../../skills/_legacy_impl/meta/shared/quark-skill-creator/scripts/validate_skill.py)
applies to delegated and internal legacy bodies. It requires these five
frontmatter fields:

- `name`
- `description`
- `layer`
- `primary_artifact`
- `source_knowledge`

It parses frontmatter as a YAML mapping and requires those keys to be present.
When values use their expected types, it checks the name against
`^[a-z][a-z0-9-]{0,63}$`, restricts `layer` to `l0-foundation`, `l1-atomic`,
`l2-workflows`, `l3-recipes`, or `meta`, and requires these five body sections:

- `## Purpose`
- `## Inputs`
- `## Outputs`
- `## Interaction Flow`
- `## Recovery`

For string values, the validator limits `description` to 100 words and rejects
unresolved primary-artifact placeholders. For a `source_knowledge` list, each
item must be a string and a repo-root-relative path; a missing known-upstream
path warns, while other invalid or missing paths block. It limits the body to
315 lines. Noncanonical artifact extensions and angle brackets in descriptions
warn. Exit codes are 0 for pass, 1 for warnings only, and 2 for blocking
findings.

The current validator does not reject every wrong value type: type-specific
checks are skipped when fields such as `name` or `layer` are not strings, or
when `source_knowledge` is not a list. Treat this as a mechanical validation
gap and catch it during semantic review until the validator is strengthened.

`backend` is a legacy template convention that should match the parent scope,
but the validator does not currently require or check it.

Some grandfathered delegated bodies may still fail these checks; record those findings as
migration debt rather than applying the legacy format to a public stub or self-contained entry.
Mechanical validation does not replace semantic, link, contract, or runtime review.
