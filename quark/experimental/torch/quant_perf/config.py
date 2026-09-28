#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quant-Perf configuration & secret resolution.

Single source of truth for API keys, gateway URLs, and external tool roots.

- Secrets may live in the Quark checkout's gitignored `.env`.
- `load_env()` is called once at CLI entry (via python-dotenv).
- `resolve_api_key()` is the ONLY read point for the LLM gateway key; the three
  consumers (LLMClient / GEAK subprocess / Quark llm-ptq skill via
  claude_agent_sdk) all go through `build_subprocess_env()` or this resolver.
- Never hard-code keys; never log key values.
"""

from __future__ import annotations

import os
from pathlib import Path


class ConfigError(RuntimeError):
    """Raised when a required config value is missing."""


_PACKAGE_DIR = Path(__file__).resolve().parent


def _find_quark_source_root() -> Path | None:
    """Locate the containing Quark source checkout.

    :return: Resolved checkout root, or ``None`` for a wheel installation.
    """
    candidates = [*_PACKAGE_DIR.parents, Path.cwd(), *Path.cwd().parents]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "quark").is_dir()
            and (candidate / ".claude" / "skills" / "quark-torch-ptq" / "SKILL.md").is_file()
        ):
            return candidate
    return None


_SOURCE_ROOT = _find_quark_source_root()


def load_env(dotenv_path: str | os.PathLike[str] | None = None) -> None:
    """Load `.env` into os.environ once (idempotent). Call at CLI entry.

    Values already in the real environment win over `.env` (override=False),
    so CI/production injection takes precedence over a local `.env`.
    """
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ConfigError("python-dotenv not installed; run `pip install -e '.[quant_perf]'`") from exc

    path = Path(dotenv_path) if dotenv_path else (_SOURCE_ROOT / ".env" if _SOURCE_ROOT is not None else None)
    if path is not None and path.exists():
        load_dotenv(path, override=False)


# --- LLM gateway ---------------------------------------------------------


def resolve_api_key() -> str:
    """Return the AMD LLM gateway key, or raise with a fix hint.

    Priority: AMD_LLM_API_KEY -> AMD_LLM_GATEWAY_KEY -> LLM_GATEWAY_KEY
    (all env / .env-loaded).
    """
    key = os.getenv("AMD_LLM_API_KEY") or os.getenv("AMD_LLM_GATEWAY_KEY") or os.getenv("LLM_GATEWAY_KEY")
    if not key:
        raise ConfigError(
            "LLM gateway key not found. Export AMD_LLM_API_KEY, AMD_LLM_GATEWAY_KEY, or LLM_GATEWAY_KEY; "
            "alternatively place one in the Quark source checkout's .env file."
        )
    return key


def resolve_user() -> str:
    """Gateway user header (ntid@amd.com). Falls back to $USER@amd.com."""
    user = os.getenv("AMD_LLM_USER")
    if user:
        return user
    login = os.getenv("USER", "unknown")
    return login if "@" in login else f"{login}@amd.com"


def base_url() -> str:
    """Return the configured AMD LLM gateway base URL.

    :return: Gateway base URL.
    """
    return os.getenv("AMD_LLM_BASE_URL", "https://llm-api.amd.com/Anthropic")


def decision_model() -> str:
    """Lightweight model for bounded decision points."""
    return os.getenv("QUARK_QUANT_PERF_LLM_DECISION_MODEL", "claude-haiku-4-5")


def codegen_model() -> str:
    """High-quality model for repair code generation."""
    return os.getenv("QUARK_QUANT_PERF_LLM_CODEGEN_MODEL", "claude-opus-4-8")


def geak_model() -> str:
    """Return the model used by GEAK subprocesses.

    :return: Configured model name.
    """
    return os.getenv("GEAK_MODEL", "claude-opus-4-8")


# --- External tool roots --------------------------------------------------


def quark_root() -> str:
    """Return the configured or discovered Quark source checkout.

    :return: Quark source checkout path.
    :raises ConfigError: If no source checkout can be resolved.
    """
    root = os.getenv("QUARK_ROOT")
    if root:
        return root
    if _SOURCE_ROOT is not None:
        return str(_SOURCE_ROOT)
    raise ConfigError(
        "Direct PTQ requires a Quark source checkout containing the "
        "quark-torch-ptq skill. Set QUARK_ROOT to that checkout."
    )


def atom_root() -> str:
    """Return the optional ATOM source root.

    :return: ATOM source path, or an empty string when unset.
    """
    return os.getenv("ATOM_ROOT", "")


def geak_root() -> str:
    """Return the optional GEAK source root.

    :return: GEAK source path, or an empty string when unset.
    """
    return os.getenv("GEAK_ROOT", "")


def claude_bin() -> str:
    """Absolute path to the system `claude` CLI the Agent SDK should drive.

    The claude_agent_sdk otherwise prefers its own bundled CLI, which may lag
    the Workflow-tool features kernel_workflow needs; point it at the on-PATH
    binary. Returns "" when none is found (SDK falls back to its bundle)."""
    import shutil

    return os.getenv("CLAUDE_BIN") or shutil.which("claude") or ""


def experience_store_path() -> str:
    """Return the cross-session runtime experience database path."""
    return os.getenv(
        "QUARK_QUANT_PERF_EXPERIENCE_STORE_PATH",
        str(Path.home() / ".cache" / "amd-quark" / "quant_perf" / "experience.sqlite"),
    )


# --- Subprocess env (GEAK / Quark llm-ptq skill) -------------------------

# Aliases various downstream tools read the same gateway key under.
_KEY_ALIASES = (
    "AMD_LLM_API_KEY",
    "GEAK_API_KEY",
    "LLM_GATEWAY_KEY",
)


def build_subprocess_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Env dict for child processes (GEAK, Quark skill via claude_agent_sdk).

    Fans the resolved key out to every alias the child might read, and sets the
    gateway user. `setdefault` never clobbers an explicitly-provided value.
    Design ref: Hyperloom cli.py:1924 alias fan-out (concept only; reimplemented).

    The `claude` CLI (spawned by claude_agent_sdk for the Quark llm-ptq skill)
    is a separate consumer with a different auth contract: it talks to the AMD
    gateway via a custom header, not a plain bearer key. ANTHROPIC_API_KEY must
    be the literal placeholder "dummy" and the real key goes in
    ANTHROPIC_CUSTOM_HEADERS -- these two are force-set (not setdefault)
    because a stale ambient value (e.g. the calling process's own unrelated
    ANTHROPIC_API_KEY="dummy"/ANTHROPIC_CUSTOM_HEADERS) must not silently
    survive and carry the wrong key into this subprocess. Verified against
    ai-assists-skills/.claude/skills/claude-code-amd-setup/SKILL.md and GEAK's
    own amd_claude.py, both of which use this exact pattern.
    """
    env = dict(base if base is not None else os.environ)
    key = resolve_api_key()
    for alias in _KEY_ALIASES:
        env.setdefault(alias, key)
    env.setdefault("AMD_LLM_USER", resolve_user())
    env.setdefault("GEAK_USER", resolve_user())
    env.setdefault("AMD_LLM_BASE_URL", base_url())
    env.setdefault("ANTHROPIC_BASE_URL", base_url())
    env["ANTHROPIC_API_KEY"] = "dummy"
    env["ANTHROPIC_CUSTOM_HEADERS"] = f"Ocp-Apim-Subscription-Key: {key}"
    return env
