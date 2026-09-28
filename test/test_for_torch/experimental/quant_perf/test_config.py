#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for quark.experimental.torch.quant_perf.config's secret-fan-out contract, especially the
claude CLI's header-based auth (distinct from the plain-key fan-out used by
GEAK/other subprocess consumers)."""

from __future__ import annotations

from pathlib import Path

from quark.experimental.torch.quant_perf import config


def test_build_subprocess_env_fans_out_plain_key_aliases(monkeypatch):
    monkeypatch.setenv("AMD_LLM_API_KEY", "real-gateway-key")
    env = config.build_subprocess_env(base={})
    assert env["AMD_LLM_API_KEY"] == "real-gateway-key"
    assert env["GEAK_API_KEY"] == "real-gateway-key"
    assert env["LLM_GATEWAY_KEY"] == "real-gateway-key"


def test_resolve_api_key_accepts_amd_gateway_key_alias(monkeypatch):
    monkeypatch.delenv("AMD_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_KEY", raising=False)
    monkeypatch.setenv("AMD_LLM_GATEWAY_KEY", "real-gateway-key")

    assert config.resolve_api_key() == "real-gateway-key"


def test_build_subprocess_env_sets_dummy_key_and_custom_header_for_claude_cli(monkeypatch):
    """The `claude` CLI spawned by claude_agent_sdk authenticates to the AMD
    gateway via Ocp-Apim-Subscription-Key, not a plain ANTHROPIC_API_KEY --
    verified against claude-code-amd-setup's SKILL.md and GEAK's own
    amd_claude.py, both of which use this exact pattern."""
    monkeypatch.setenv("AMD_LLM_API_KEY", "real-gateway-key")
    env = config.build_subprocess_env(base={})
    assert env["ANTHROPIC_API_KEY"] == "dummy"
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "Ocp-Apim-Subscription-Key: real-gateway-key"
    assert env["ANTHROPIC_BASE_URL"] == config.base_url()


def test_build_subprocess_env_overrides_stale_ambient_anthropic_vars(monkeypatch):
    """A stale ambient ANTHROPIC_API_KEY/ANTHROPIC_CUSTOM_HEADERS (e.g. from
    the calling process's own unrelated Claude Code session) must not survive
    -- unlike the plain-key aliases, these two are force-set, not setdefault,
    because there is only one correct value for this subprocess's contract."""
    monkeypatch.setenv("AMD_LLM_API_KEY", "real-gateway-key")
    stale = {
        "ANTHROPIC_API_KEY": "dummy",
        "ANTHROPIC_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: some-other-stale-key",
    }
    env = config.build_subprocess_env(base=stale)
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "Ocp-Apim-Subscription-Key: real-gateway-key"


def test_external_tool_roots_have_no_vendored_defaults(monkeypatch):
    monkeypatch.delenv("ATOM_ROOT", raising=False)
    monkeypatch.delenv("GEAK_ROOT", raising=False)

    assert config.atom_root() == ""
    assert config.geak_root() == ""


def test_quark_root_discovers_current_source_checkout(monkeypatch):
    monkeypatch.delenv("QUARK_ROOT", raising=False)

    root = config.quark_root()

    assert (Path(root) / "pyproject.toml").is_file()
