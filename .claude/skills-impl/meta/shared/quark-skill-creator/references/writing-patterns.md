# Writing Patterns

Concrete formats authors can copy when drafting a SKILL.md body. Adapted from the Claude Code skill-creator and grounded in this project's existing skills (`quark-torch-skill-sync`, `quark-torch-doc-drift-check`, `quark-torch-llm-ptq-workflow`, etc.).

## 1. Default to no comments; explain *why*, not *what*

The model is smart. Only add context it does not already have. Challenge each paragraph: "would a future reader of this skill be confused without it?" If no, delete it. Preserve the **why** behind a rule (a past incident, a stable invariant, an upstream constraint) — that is what survives across edits.

Bad — verbose, narrates what the reader can already see:
> PDF (Portable Document Format) files are a common format. To extract text you need a library. There are many libraries…

Good — terse, names the choice and the reason:
> Use `pdfplumber`. Falls back to `pdf2image` + `pytesseract` when the input is scanned (no embedded text layer).

## 2. Imperative voice, third-person frontmatter

Skill bodies talk to the executing agent ("Use X. Read Y. Do not Z."). Frontmatter `description` talks *about* the skill in third person ("Quantize a Torch LLM…", not "I will quantize…" or "You can use this to quantize…").

## 3. Set the right freedom level

| Freedom | When | Example |
|---------|------|---------|
| High — text instructions | multiple valid approaches, context-dependent | code review guidance |
| Medium — pseudocode / templates | preferred pattern, acceptable variation | report generation |
| Low — specific scripts | fragile operations, consistency critical | database migrations, CI install commands |

In this repo: PTQ scheme selection sits at *medium* (decision tables); installation steps sit at *low* (shipped scripts under `tools/ci/`); user-facing routing sits at *high* (Claude reads the description and decides).

## 4. Template pattern — name the output structure exactly

```markdown
## Report structure

Use this exact template:

# Skill Sync Report

## Summary
- Checked: N skills
- Mechanical drift: N findings
- Breaking drift: N findings

## Breaking Drift
### <skill>: <one-line summary>
- **Skill says**: ...
- **Source says**: ...
- **Impact**: ...
- **Fix**: ...
```

## 5. Examples pattern — input → output pairs

```markdown
## Layer assignment examples

**Example 1**
Input: "validates that the user's model path resolves and contains config.json"
Output: `l0-foundation` — pure check, no installs, no PTQ decisions.

**Example 2**
Input: "picks fp8 vs int4 based on model size and accelerator"
Output: `l1-atomic` — single responsibility, one primary artifact (`quant_plan.json`).
```

## 6. Workflow pattern — checklist + steps

For multi-step skills, lead with a checklist the agent can tick through, then define each step. The checklist makes progress visible in the transcript.

```markdown
## PTQ workflow

Track progress through this checklist:

- [ ] Step 1: Validate environment
- [ ] Step 2: Inspect model
- [ ] Step 3: Plan quantization
- [ ] Step 4: Confirm with user
- [ ] Step 5: Execute or summarize

**Step 1: Validate environment**
Run `quark-env-preflight`. Halt if no GPU is detected and `--cpu-only` was not requested.
```

## 7. Conditional pattern — branch on intent

```markdown
## Decide what to do

Look at the user's intent:

**Wants quantization end-to-end?** → Run the PTQ workflow.
**Just wants a plan?** → Stop after `quark-torch-quant-plan` and emit `quant_plan.json`.
**Reports a crash or stack trace?** → Hand off to `quark-torch-debug`.
```

## 8. Feedback loop pattern — validate → fix → re-validate

```markdown
1. Generate the artifact.
2. **Validate immediately**: `python scripts/validate_skill.py <skill-dir>`.
3. If validation fails:
   - Read the error message
   - Apply the smallest fix
   - Re-validate
4. **Only proceed when validation passes.** Do not commit a half-validated skill.
```

## 9. Anti-patterns specific to this project

- **Inventing artifact names.** The eight canonical artifacts in `docs/agent_skills/artifact-contracts.md` are the contract surface. New artifacts require a schema in `shared/contracts/` and a producer/consumer entry — never a unilateral addition.
- **Time-sensitive instructions.** "If you're on Quark 0.10 or earlier…" rots. Use a "current method" + collapsed "deprecated" section instead, and lean on `quark-torch-doc-drift-check` to catch when the current method moves.
- **Inconsistent terminology.** Pick one term and stick with it: `artifact` (not "output file"/"artefact"), `scheme` (not "method"/"mode"), `layer` (not "tier"/"level").
- **Vague names.** `quark-helper`, `quark-utils`, `quark-tools` will be rejected. Names align to user goals.
- **Skipping `## Recovery`.** Every skill must say what to do when it cannot continue. `## Recovery` returning "TODO" fails review.

## 10. Length discipline

If the SKILL.md body is creeping past ~250 lines:

1. **Move detail into `references/`.** A skill body should read like a one-page brief; references are for the appendix.
2. **Move repeatable mechanics into `scripts/`.** If you find yourself describing five Python lines, ship a script that does them.
3. **Move long subagent instructions into `agents/`.** If a sub-agent prompt is more than ~10 lines, give it its own file and reference it.

The 315-line cap is a hard ceiling, not a target. Aim for 150–250.
