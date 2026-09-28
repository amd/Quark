#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Install the Agent Skills bundled with AMD Quark."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from quark import __version__

_AGENT_SKILLS_DIRS = {
    "claude-code": Path(".claude") / "skills",
    "cursor": Path(".agents") / "skills",
    "codex": Path(".agents") / "skills",
}


class SkillsInstallError(RuntimeError):
    """Raised when bundled Agent Skills cannot be installed safely."""


def _bundled_skills_root() -> Path:
    return Path(__file__).resolve().parent / "_bundled_skills"


def _selected_destinations(agent: str, target: Path) -> dict[Path, tuple[str, ...]]:
    selected_agents = tuple(_AGENT_SKILLS_DIRS) if agent == "all" else (agent,)
    if any(name not in _AGENT_SKILLS_DIRS for name in selected_agents):
        raise SkillsInstallError(f"unsupported agent: {agent}")

    destinations: dict[Path, list[str]] = {}
    for name in selected_agents:
        destination = target / _AGENT_SKILLS_DIRS[name]
        destinations.setdefault(destination, []).append(name)
    return {path: tuple(names) for path, names in destinations.items()}


def _exists(path: Path) -> bool:
    """Return whether a path exists, including a broken symbolic link."""
    return path.exists() or path.is_symlink()


def _remove(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def _copy_entry(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, copy_function=shutil.copy2)
    else:
        shutil.copy2(source, destination)


def install_skills(
    *,
    agent: str,
    target: Path,
    force: bool = False,
    bundle_root: Path | None = None,
) -> dict[Path, tuple[str, ...]]:
    """Copy the complete bundled skill tree into the selected Agent directories."""
    target = target.expanduser().resolve()
    if not target.is_dir():
        raise SkillsInstallError(f"target is not an existing directory: {target}")

    bundle_root = _bundled_skills_root() if bundle_root is None else bundle_root.resolve()
    if not bundle_root.is_dir():
        raise SkillsInstallError(f"bundled skills are missing: {bundle_root}")

    entries = sorted(bundle_root.iterdir(), key=lambda path: path.name)
    if not entries:
        raise SkillsInstallError(f"bundled skills are empty: {bundle_root}")

    destinations = _selected_destinations(agent, target)
    invalid_roots = [
        destination
        for destination in destinations
        if _exists(destination) and (not destination.exists() or not destination.is_dir())
    ]
    if invalid_roots:
        paths = "\n".join(f"  {path}" for path in invalid_roots)
        raise SkillsInstallError(f"skills destination is not a directory:\n{paths}")

    conflicts = [
        destination / entry.name
        for destination in destinations
        for entry in entries
        if _exists(destination / entry.name)
    ]
    if conflicts and not force:
        paths = "\n".join(f"  {path}" for path in conflicts)
        raise SkillsInstallError(
            f"refusing to overwrite existing bundled entries:\n{paths}\nRun again with --force to replace them."
        )

    try:
        for destination in destinations:
            destination.mkdir(parents=True, exist_ok=True)
            for entry in entries:
                installed_entry = destination / entry.name
                if _exists(installed_entry):
                    _remove(installed_entry)
                _copy_entry(entry, installed_entry)
    except OSError as error:
        raise SkillsInstallError(f"failed to install skills: {error}") from error

    return destinations


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quark-skills", description="Install Agent Skills bundled with AMD Quark.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    install_parser = subparsers.add_parser("install", help="install the bundled skills into an Agent project")
    install_parser.add_argument("--agent", required=True, choices=(*_AGENT_SKILLS_DIRS, "all"))
    install_parser.add_argument(
        "--target",
        type=Path,
        default=Path.cwd(),
        help="target project directory (default: current working directory)",
    )
    install_parser.add_argument("--force", action="store_true", help="replace existing Quark skill entries")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``quark-skills`` command."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        destinations = install_skills(agent=args.agent, target=args.target, force=args.force)
    except SkillsInstallError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Installed AMD Quark Agent Skills {__version__}:")
    for destination, agents in destinations.items():
        print(f"  {', '.join(agents)} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
