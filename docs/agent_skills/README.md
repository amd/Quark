# Documentation Index

## Architecture

- [architecture.md](architecture.md) — Layer admission rules, naming conventions, directory structure, and MVP release gate

## Contracts

How skills interact with each other and with the user. Read these before writing a new skill.

- [interaction-contract.md](interaction-contract.md) — The five-stage skeleton (Intake, Route, Plan, Confirm, Execute) every user-facing skill follows
- [artifact-contracts.md](artifact-contracts.md) — The canonical handoff artifacts between skills, with producers and consumers
- [skill-format-contract.md](skill-format-contract.md) — Required `SKILL.md` frontmatter fields and sections

JSON schemas for the artifacts live under [`.claude/skills-impl/shared/contracts/`](../../.claude/skills-impl/shared/contracts/).

## Governance

- [governance.md](governance.md) — How the system stays correct: the governance loop, the Quark-only validation boundary, and the eval upstream evidence rule

## Skills Catalog

- [skills_catalog.md](skills_catalog.md) — Brief introduction to every user-facing Quark skill (grouped by backend and layer), plus notes and tips on what to watch out for when invoking them
