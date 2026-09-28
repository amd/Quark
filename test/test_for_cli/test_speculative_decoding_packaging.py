#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Packaging, shell, and disclosure guards for the EAGLE-3 runner assets."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "experimental" / "speculative_decoding"

# Assets the installed CLI must be able to resolve from any working directory.
REQUIRED_ASSETS = (
    "eagle3/common/run_all.sh",
    "eagle3/common/scripts/00_setup.sh",
    "eagle3/common/docker/Dockerfile.rocm",
    "eagle3/common/scripts/bench.py",
    "eagle3/common/scripts/finalize_onpolicy_data.py",
    "eagle3/common/python/sitecustomize.py",
    "eagle3/qwen3_8b_quick_start/run.sh",
    "eagle3/qwen3_8b_quick_start/configs/qwen3_8b_eagle3.yaml",
    "eagle3/minimax_m3_best_recipe/run.sh",
    "eagle3/minimax_m3_best_recipe/large_model_profile.yaml",
    "eagle3/minimax_m3_best_recipe/domain_manifest.smoke.yaml",
)

# Directory and file names produced by a run; none may reach a distribution.
GENERATED_OUTPUTS = ("TorchSpec", "models", "data", "outputs", "training", "runtime", "release", "cache", "logs")


def _shell_scripts() -> list[Path]:
    return sorted(EXAMPLE_ROOT.rglob("*.sh"))


def test_every_runner_shell_script_parses() -> None:
    scripts = _shell_scripts()
    # The canonical runner plus its phase scripts, and one wrapper per profile.
    assert len(scripts) >= 10, scripts

    result = subprocess.run(
        ["bash", "-n", *(str(script) for script in scripts)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_runner_reports_success_after_a_completed_full_run() -> None:
    """A trailing `test && cmd` would make the full profile exit non-zero.

    Bash gives a script the exit status of its last command, so a false test in
    final position turns a completed pipeline into a failure.
    """
    runner = EXAMPLE_ROOT / "eagle3" / "common" / "run_all.sh"
    body = [
        line.strip()
        for line in runner.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not body[-1].startswith("["), body[-1]

    # Same shape as the runner's tail, to pin the behaviour rather than the text.
    probe = "\n".join(["set -euo pipefail", "PROFILE=full", "echo done", '[ "$PROFILE" = quick ] && echo hint'])
    assert subprocess.run(["bash", "-c", probe], capture_output=True).returncode == 1
    fixed = "\n".join(
        ["set -euo pipefail", "PROFILE=full", "echo done", 'if [ "$PROFILE" = quick ]; then echo hint; fi']
    )
    assert subprocess.run(["bash", "-c", fixed], capture_output=True).returncode == 0


def test_repository_wide_disclosure_scan() -> None:
    """No published EAGLE-3 surface may carry internal recipe values or local paths.

    ``[2, 30, 57]`` is listed for a different reason than the rest. It is the
    published EAGLE-3 low/middle/high rule evaluated at 60 layers, not an
    internal value, and ``make_draft_config.py`` is expected to produce it for a
    target of that depth. It stays banned as a literal so shipped configuration
    keeps expressing the rule relative to target depth instead of pinning ids
    that silently become wrong for the next target.
    """
    scanned = [
        *(REPO_ROOT / "quark" / "experimental" / "speculative_decoding").rglob("*"),
        *EXAMPLE_ROOT.rglob("*"),
        REPO_ROOT / "docs" / "source" / "eagle3.rst",
        REPO_ROOT / "docs" / "source" / "eagle3_quick_start.rst",
        REPO_ROOT / "docs" / "source" / "eagle3_best_recipe.rst",
        REPO_ROOT / "docs" / "source" / "speculative_decoding.rst",
    ]
    forbidden = (
        "/home/larryli2",
        "onpolicy_mixed.jsonl",
        "/vocab/d2t.pt",
        "draft_vocab_size: 32000",
        "[2, 30, 57]",
        "noise_floor_al: 0.02",
        "base-chat : code/tech",
        "smci350",
    )

    offenders: list[str] = []
    for path in scanned:
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        offenders.extend(f"{path}: {value}" for value in forbidden if value in text)

    assert not offenders, offenders


def test_runner_writes_nothing_into_the_packaged_asset_tree() -> None:
    """Run outputs stay out of the tree by default rather than by exclusion.

    ``EX_DIR`` and ``CACHE_DIR`` used to default beside the scripts, so every
    output directory had to be enumerated in ``MANIFEST.in`` and ``setup.py``
    for each profile -- an enumeration that a new profile silently outgrew.
    Keeping the defaults outside the tree is what makes those lists unnecessary,
    so it is the property worth pinning.
    """
    common = EXAMPLE_ROOT / "eagle3" / "common" / "scripts" / "_common.sh"
    text = common.read_text(encoding="utf-8")

    assert ': "${EX_DIR:=$PWD/ckpts/eagle3-$RUNNER_PROFILE}"' in text
    assert ': "${CACHE_DIR:=${XDG_CACHE_HOME:-$HOME/.cache}/amd-quark/eagle3}"' in text
    # A profile wrapper that re-pointed either one would reintroduce the problem.
    for profile in ("qwen3_8b_quick_start", "minimax_m3_best_recipe"):
        wrapper = (EXAMPLE_ROOT / "eagle3" / profile / "run.sh").read_text(encoding="utf-8")
        assert "EX_DIR=" not in wrapper, profile
        assert "CACHE_DIR=" not in wrapper, profile


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires a POSIX build environment")
def test_sdist_and_wheel_ship_runner_assets_without_run_outputs(tmp_path: Path) -> None:
    """Build from a checkout polluted with run outputs, then install and resolve assets.

    Outputs default outside this tree now, so a current checkout cannot be
    polluted this way. A checkout that ran an older build still can, and its
    leftovers share extensions with real assets, so the exclusions have to hold
    at every depth rather than at the paths one profile happens to use.
    """
    source = tmp_path / "src"
    shutil.copytree(
        REPO_ROOT,
        source,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "build", "dist"),
    )

    example_root = source / "examples" / "experimental" / "speculative_decoding"
    for depth in (example_root, example_root / "eagle3" / "common", example_root / "eagle3" / "qwen3_8b_quick_start"):
        for name in GENERATED_OUTPUTS:
            (depth / name).mkdir(parents=True, exist_ok=True)
            for leaked in ("leaked.json", "leaked.yaml", "leaked.jsonl"):
                (depth / name / leaked).write_text("RUN_OUTPUT_SENTINEL\n", encoding="utf-8")
        (depth / "report.json").write_text("RUN_OUTPUT_SENTINEL\n", encoding="utf-8")

    dist = tmp_path / "dist"
    env = {**os.environ, "QUARK_ACCELERATOR": "cpu"}
    for command in (
        [sys.executable, "setup.py", "-q", "sdist", "--dist-dir", str(dist)],
        [sys.executable, "setup.py", "-q", "build_py"],
        [sys.executable, "setup.py", "-q", "bdist_wheel", "--skip-build", "--dist-dir", str(dist)],
    ):
        result = subprocess.run(command, cwd=source, env=env, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr[-4000:]

    with tarfile.open(next(dist.glob("*.tar.gz"))) as archive:
        sdist_names = archive.getnames()
    assert not [name for name in sdist_names if name.rsplit("/", 1)[-1].startswith("leaked.")]
    assert not [name for name in sdist_names if name.endswith("/report.json")]
    # A wheel built from the sdist has to find the assets there, so they must
    # survive the exclusions above rather than only being copied from a checkout.
    prefix = "examples/experimental/speculative_decoding/"
    shipped = {name.split(prefix, 1)[1] for name in sdist_names if prefix in name}
    assert not set(REQUIRED_ASSETS) - shipped

    install_root = tmp_path / "install"
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--target", str(install_root)]
        + [str(next(dist.glob("*.whl")))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert install.returncode == 0, install.stderr[-4000:]

    assets = install_root / "quark" / "experimental" / "speculative_decoding" / "_torchspec_assets"
    missing = [name for name in REQUIRED_ASSETS if not (assets / name).is_file()]
    assert not missing, missing
    assert not [name for name in GENERATED_OUTPUTS if (assets / name).exists()]
    assert not list(assets.rglob("report.json"))

    # The installed package must resolve its own assets from an unrelated CWD.
    resolved = subprocess.run(
        [
            sys.executable,
            "-c",
            "from quark.experimental.speculative_decoding.torchspec_runner import find_runner_assets;"
            "print(find_runner_assets())",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(install_root), sysconfig.get_paths()["purelib"]]),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert resolved.returncode == 0, resolved.stderr[-4000:]
    assert resolved.stdout.strip() == str(assets)
