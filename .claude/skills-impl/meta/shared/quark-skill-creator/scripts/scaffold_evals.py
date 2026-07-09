#!/usr/bin/env python3
"""Scaffold an evals/evals.json file for a Quark skill.

Produces a starter file with one placeholder eval per applicable category
(routing / planning / artifact / recovery). The author replaces TODO fields
with realistic prompts and discriminating expectations, then hands the skill
+ evals to quark-eval-runner.

Usage:
    python3 scaffold_evals.py <skill-dir> [--categories routing planning ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VALID_CATEGORIES = ("routing", "planning", "artifact", "recovery")
REPO_ROOT = Path(__file__).resolve().parents[5]


def load_skill_name(skill_dir: Path) -> str:
    """Read the ``name`` field from a skill's SKILL.md frontmatter.

    Args:
        skill_dir: Path to the skill directory containing SKILL.md.

    Returns:
        The skill name string, used to populate ``skill_name`` and
        ``expected_skill`` in the scaffolded evals.

    Raises:
        SystemExit: If SKILL.md is missing or has no ``name`` line.
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        raise SystemExit(f"error: {skill_md} not found")
    text = skill_md.read_text(encoding="utf-8")
    m = re.search(r"^name:\s*(\S+)", text, flags=re.MULTILINE)
    if not m:
        raise SystemExit(f"error: could not find 'name' in {skill_md} frontmatter")
    return m.group(1)


def starter_eval(eval_id: int, category: str, skill_name: str) -> dict:
    """Build a placeholder eval dict for one ``(id, category)`` slot.

    The author replaces every ``TODO`` with a realistic prompt and a set of
    discriminating expectations before handing the file to ``quark-eval-runner``.

    Args:
        eval_id: Unique integer id within the evals.json file.
        category: One of ``routing``, ``planning``, ``artifact``, ``recovery``.
        skill_name: The skill's frontmatter ``name`` (used to populate the
            routing eval's ``expected_skill`` and the routing expectation).

    Returns:
        A dict matching the per-eval schema documented in
        ``references/evals-schema.md``, with category-specific defaults
        for ``expectations``, ``input_artifacts``, and ``expected_skill``.
    """
    base = {
        "id": eval_id,
        "category": category,
        "name": f"TODO-{category}-case",
        "prompt": "TODO: a realistic user-style prompt that exercises this skill.",
        "expectations": [
            "TODO: an objectively verifiable expectation about the run.",
        ],
    }
    if category == "routing":
        base["expected_skill"] = skill_name
        base["expectations"] = [f"Claude invokes {skill_name} (or its workflow stub) first."]
    elif category == "planning":
        base["input_artifacts"] = ["evals/inputs/TODO.input_artifact.json"]
    elif category == "artifact":
        base["input_artifacts"] = ["evals/inputs/TODO.input_artifact.json"]
        base["expectations"] = [
            "Output validates against shared/contracts/TODO.schema.json.",
        ]
    elif category == "recovery":
        base["expectations"] = [
            "Diagnosis names the root cause.",
            "Fix is concrete (a command or a config change), not 'check the logs'.",
        ]
    return base


def main(argv: list[str]) -> int:
    """Entry point: scaffold ``evals/evals.json`` for one skill.

    Reads the skill's ``name`` from its SKILL.md, builds one starter eval
    per requested category, and writes the result to
    ``<skill_dir>/evals/evals.json``. Also creates ``evals/inputs/`` with
    a ``.gitkeep`` so the directory is tracked in git.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        Exit code: 0 on success, 2 if the target ``evals.json`` already
        exists without ``--force`` or if the skill directory is invalid.
    """
    p = argparse.ArgumentParser(description="Scaffold evals/evals.json for a Quark skill.")
    p.add_argument("skill_dir", help="Path to the skill directory (containing SKILL.md)")
    p.add_argument(
        "--categories",
        nargs="+",
        choices=VALID_CATEGORIES,
        default=list(VALID_CATEGORIES),
        help="Eval categories to scaffold (default: all four)",
    )
    p.add_argument("--force", action="store_true", help="Overwrite an existing evals/evals.json")
    args = p.parse_args(argv)

    skill_dir = Path(args.skill_dir).resolve()
    if not skill_dir.is_dir():
        print(f"error: {skill_dir} is not a directory", file=sys.stderr)
        return 2

    skill_name = load_skill_name(skill_dir)
    try:
        skill_path_rel = skill_dir.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        skill_path_rel = str(skill_dir)

    evals_dir = skill_dir / "evals"
    evals_file = evals_dir / "evals.json"
    if evals_file.exists() and not args.force:
        print(f"error: {evals_file} already exists; pass --force to overwrite", file=sys.stderr)
        return 2

    evals = [starter_eval(i + 1, cat, skill_name) for i, cat in enumerate(args.categories)]
    payload = {
        "skill_name": skill_name,
        "skill_path": skill_path_rel,
        "evals": evals,
    }

    evals_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = evals_dir / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    (inputs_dir / ".gitkeep").touch()

    evals_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Scaffolded: {evals_file.relative_to(REPO_ROOT) if evals_file.is_relative_to(REPO_ROOT) else evals_file}")
    print()
    print("Next steps:")
    print("  - Replace each TODO with a realistic prompt and discriminating expectations.")
    print("  - Drop input fixtures into evals/inputs/ and reference them from `input_artifacts`.")
    print("  - Hand the skill + evals to quark-eval-runner; iterate on failures.")
    print("  - See references/evals-schema.md for the schema and writing-good-expectations guidance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
