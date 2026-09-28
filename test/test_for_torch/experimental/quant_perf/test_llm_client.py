#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
from types import SimpleNamespace

import quark.experimental.torch.quant_perf.llm.client as llm_client


class _FakeAnthropicClient:
    instances: list["_FakeAnthropicClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = SimpleNamespace(create=lambda **_kwargs: None)
        self.instances.append(self)


def test_get_client_reuses_matching_credentials_and_rebuilds_after_rotation(monkeypatch) -> None:
    credentials = {
        "key": "first-key",
        "user": "first-user@amd.com",
        "base_url": "https://gateway.example/Anthropic",
    }
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=_FakeAnthropicClient))
    monkeypatch.setattr(llm_client.config, "resolve_api_key", lambda: credentials["key"])
    monkeypatch.setattr(llm_client.config, "resolve_user", lambda: credentials["user"])
    monkeypatch.setattr(llm_client.config, "base_url", lambda: credentials["base_url"])
    monkeypatch.setattr(llm_client, "_client", None)
    monkeypatch.setattr(llm_client, "_client_identity", None, raising=False)
    _FakeAnthropicClient.instances.clear()

    first = llm_client._get_client()
    assert llm_client._get_client() is first

    credentials["key"] = "second-key"
    second = llm_client._get_client()

    assert second is not first
    assert len(_FakeAnthropicClient.instances) == 2
    assert second.kwargs["default_headers"]["Ocp-Apim-Subscription-Key"] == "second-key"


def test_direct_api_call_uses_client_and_throttles_repeated_tag(monkeypatch) -> None:
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text="decision")])

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(llm_client, "_get_client", lambda: fake_client)
    monkeypatch.setattr(llm_client, "_throttle", llm_client._Throttle(cooldown_s=60.0, max_per_round=3))

    result = llm_client.direct_api_call(tag="route", system="system", user="user", round_id=1)
    throttled = llm_client.direct_api_call(
        tag="route",
        system="system",
        user="user",
        round_id=1,
        default="fallback",
    )

    assert result == "decision"
    assert throttled == "fallback"
    assert len(calls) == 1
