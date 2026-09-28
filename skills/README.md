# Quark Agent Skills

This directory is the canonical, product-owned source for Quark agent skills.

- Each `quark-*` directory is a user-facing skill entry.
- `.agents/skills` contains real-file adapters to this directory for Cursor and Codex discovery.
- `.claude/skills` contains real-file adapters to this directory for Claude Code discovery.
- `_legacy_impl/` temporarily contains the layered implementations that the
  remaining entries still delegate to.

To register a new public skill, add its complete canonical entry at `skills/<skill-name>/SKILL.md` and matching adapter files under `.agents/skills/<skill-name>/SKILL.md` and `.claude/skills/<skill-name>/SKILL.md`. Each adapter repeats only the canonical `name` and `description`, then forwards directly to the canonical entry; do not copy the instruction body or resources.

The adapters are repository discovery entries, not standalone distributions. `quark-skills install` copies the canonical skill tree instead.

## Install from the Python package

After installing `amd-quark`, copy the bundled skill tree into the current
project for the Agent you use:

```bash
quark-skills install --agent claude-code
quark-skills install --agent cursor
quark-skills install --agent codex
quark-skills install --agent all
```

Pass `--target <path>` to install into another project. Existing Quark entries
are not replaced unless `--force` is supplied; unrelated skills are preserved.

## Layout contract

A migrated skill keeps its instructions, references, contracts, executable
helpers, evaluation cases, ownership, and license information in one directory.
Its primary runtime instructions do not delegate to `_legacy_impl/`; any
mode-specific external skill or source-checkout integration must be explicit.
Repository-relative paths recorded as
`source_knowledge` or under a `Provenance` heading are there for maintainers and
drift checks; they are not runtime dependencies.

## Migration status

Skills are being made self-contained one at a time, and only a migrated skill is
safe to copy out of this repository.

| Skill | Status |
|---|---|
| [`quark-torch-ptq`](quark-torch-ptq/SKILL.md) | Self-contained. Copy it into a project with a compatible `amd-quark` installed and it runs a full PyTorch / Hugging Face LLM PTQ on its own. |
| [`quark-torch-quant-perf`](quark-torch-quant-perf/SKILL.md) | Self-contained entry. A copied directory supports automatic search with `amd-quark[quant_perf]` plus a compatible accelerator/vLLM runtime; fixed `--quant-strategy` also requires a Quark checkout containing `quark-torch-ptq`. |
| [`quark-install`](quark-install/SKILL.md) | Self-contained. Copy it into a project to install or verify `amd-quark` without a Quark source checkout or another skill. |
| every other `quark-*` entry | Transitional. Each is a thin entry that reads its instructions from `_legacy_impl/`, so it breaks once separated from this repository. |

`_legacy_impl/` is removed after the last entry has been migrated.
