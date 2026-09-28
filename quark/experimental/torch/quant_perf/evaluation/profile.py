#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Deterministic GSM8K profile generation for Quark Quant-Perf's real accuracy gate."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections.abc import Callable
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.session.spec import EvalProfile

EVAL_PROFILE_RESOLVER_VERSION = 2


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest() if value else ""


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def eval_profile_input_hash(
    model_ref: str,
    *,
    discovery: str,
    allow_llm: bool,
    overrides: dict[str, Any],
) -> str:
    """Hash only the inputs that can change profile resolution."""
    return _canonical_hash(
        {
            "model_ref": model_ref,
            "discovery": discovery,
            "allow_llm": allow_llm,
            "overrides": overrides,
        }
    )


def load_eval_policy() -> dict[str, Any]:
    path = files("quark.experimental.torch.quant_perf.evaluation").joinpath("policies/gsm8k.json")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("evaluation policy must be a JSON object")
    return value


def _local_model_card(model_ref: str) -> tuple[str, str]:
    path = Path(model_ref) / "README.md"
    if not path.is_file():
        return "", ""
    try:
        return path.read_text(), str(path)
    except OSError:
        return "", ""


def _default_online_model_card(
    model_ref: str,
) -> tuple[str, str]:
    repo_id = model_ref
    local = Path(model_ref)
    if local.is_dir():
        parts = local.parts
        if len(parts) < 2:
            return "", ""
        repo_id = "/".join(parts[-2:])
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo_id, filename="README.md")
        return (
            Path(path).read_text(),
            f"https://huggingface.co/{repo_id}/raw/main/README.md",
        )
    except Exception:
        return "", ""


def _linked_arxiv_url(text: str) -> str:
    match = re.search(
        r"https?://arxiv\.org/(?:abs|html)/([0-9.]+)",
        text,
    )
    return f"https://arxiv.org/html/{match.group(1)}" if match else ""


def _default_paper_loader(url: str) -> str:
    if not url:
        return ""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _parse_model_card_command(text: str) -> dict[str, Any]:
    if "lm_eval" not in text and "lm-eval" not in text:
        return {}
    values: dict[str, Any] = {}
    task = re.search(r"--tasks(?:=|\s+)([A-Za-z0-9_,.-]+)", text)
    if task:
        values["task"] = task.group(1).split(",", 1)[0]
    fewshot = re.search(
        r"--num[_-]fewshot(?:=|\s+)(\d+)",
        text,
    )
    if fewshot:
        values["num_fewshot"] = int(fewshot.group(1))
    if "cot" in str(values.get("task", "")).lower():
        values["prompting_strategy"] = "cot"
    return values


def _parse_llm_evidence(
    output: str,
    source_text: str,
) -> dict[str, Any]:
    try:
        row = json.loads(output)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(row, dict):
        return {}
    evidence = row.get("evidence_text")
    if not isinstance(evidence, str) or not evidence:
        return {}
    if evidence not in source_text:
        return {}
    values: dict[str, Any] = {}
    task = row.get("task")
    if isinstance(task, str) and task:
        values["task"] = task
    fewshot = row.get("num_fewshot")
    if isinstance(fewshot, int) and fewshot >= 0:
        values["num_fewshot"] = fewshot
    strategy = row.get("prompting_strategy")
    if strategy in {"cot", "direct", "task_default"}:
        values["prompting_strategy"] = strategy
    return values


def _relevant_model_card_text(text: str) -> str:
    lines = text.splitlines()
    keywords = (
        "gsm8k",
        "evaluation",
        "benchmark",
        "few-shot",
        "few shot",
        "chain-of-thought",
        "thinking",
        "lm_eval",
        "lm-eval",
    )
    selected: set[int] = set()
    for index, line in enumerate(lines):
        lowered = line.lower()
        if any(keyword in lowered for keyword in keywords):
            selected.update(range(max(0, index - 2), min(len(lines), index + 3)))
    if not selected:
        return ""
    return "\n".join(lines[index] for index in sorted(selected))[:12000]


def _default_llm_extractor(
    source_text: str,
    context: dict[str, Any],
    *,
    session_dir: Path | None,
) -> str:
    from quark.experimental.torch.quant_perf import config
    from quark.experimental.torch.quant_perf.llm.audit import append_llm_call
    from quark.experimental.torch.quant_perf.llm.client import direct_api_call

    system = (
        "Extract GSM8K evaluation settings from the supplied source. "
        "Return JSON only with optional task, num_fewshot, "
        "prompting_strategy, and required evidence_text copied exactly "
        "from the source. Use null for unsupported fields. Never infer "
        "settings that are not explicitly stated."
    )
    user = json.dumps(
        {
            "context": context,
            "source": source_text,
        },
        sort_keys=True,
    )
    output = direct_api_call(
        tag="eval_profile_extract",
        system=system,
        user=user,
        max_tokens=768,
        default="",
    )
    if session_dir is not None:
        append_llm_call(
            session_dir,
            call_type="eval_profile_extraction",
            model=config.decision_model(),
            round_id=0,
            prompt=user,
            output=output,
            outcome="returned" if output else "fallback",
            knowledge_ids=["evaluation.quark.llm-eval-policy.v1"],
        )
    return output


def _load_json(model_ref: str, filename: str) -> dict[str, Any]:
    local = Path(model_ref) / filename
    if local.exists():
        try:
            return json.loads(local.read_text())
        except Exception:
            return {}
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=model_ref, filename=filename)
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def _load_text(model_ref: str, filename: str) -> str:
    local = Path(model_ref) / filename
    if local.is_file():
        try:
            return local.read_text()
        except OSError:
            return ""
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=model_ref, filename=filename)).read_text()
    except Exception:
        return ""


def build_eval_profile(
    model_ref: str,
    *,
    config_json: dict[str, Any] | None = None,
    tokenizer_config: dict[str, Any] | None = None,
) -> EvalProfile:
    """Build a versioned profile from local or cached model metadata only."""
    config = config_json if config_json is not None else _load_json(model_ref, "config.json")
    tokenizer = tokenizer_config if tokenizer_config is not None else _load_json(model_ref, "tokenizer_config.json")
    raw_text_config = config.get("text_config")
    text_config = raw_text_config if isinstance(raw_text_config, dict) else {}
    chat_template = (
        tokenizer.get("chat_template")
        or tokenizer.get("default_chat_template")
        or _load_text(model_ref, "chat_template.jinja")
    )
    name = " ".join(
        str(value)
        for value in (
            model_ref,
            config.get("_name_or_path"),
            tokenizer.get("name_or_path"),
            config.get("model_type"),
        )
        if value
    )

    if chat_template:
        model_mode = "chat"
        reason = "chat_template"
    elif re.search(r"(?:-instruct|-chat|-it)(?:\b|$)", name, re.I):
        model_mode = "chat"
        reason = "model_name"
    else:
        model_mode = "base"
        reason = "fallback_base"

    thinking = (
        False
        if (
            model_mode == "chat"
            and (
                "enable_thinking" in str(chat_template or "")
                or re.search(r"(?:qwen3|qwq|deepseek.*-r1|\br1\b)", name, re.I)
            )
        )
        else None
    )

    configured_max = config.get("max_position_embeddings") or text_config.get("max_position_embeddings")
    try:
        configured_max = int(configured_max)
    except (TypeError, ValueError):
        configured_max = 8192
    max_model_len = min(8192, configured_max) if configured_max > 0 else 8192

    mode = "nothink" if thinking is False else "default"
    return EvalProfile(
        profile_id=f"gsm8k-{model_mode}-{mode}-v1",
        profile_hash="",
        model_mode=model_mode,
        apply_chat_template=bool(chat_template),
        enable_thinking=thinking,
        detection_reason=reason,
        max_model_len=max_model_len,
    ).with_computed_hash()


def resolve_eval_profile(
    model_ref: str,
    *,
    config_json: dict[str, Any] | None = None,
    tokenizer_config: dict[str, Any] | None = None,
    discovery: str = "local",
    allow_llm: bool = True,
    overrides: dict[str, Any] | None = None,
    llm_extractor: Callable[[str, dict[str, Any]], str] | None = None,
    online_loader: Callable[[str], tuple[str, str]] | None = None,
    paper_loader: Callable[[str], str] | None = None,
    artifact_dir: str | Path | None = None,
) -> EvalProfile:
    """Resolve one real-evaluation profile before GPU work.

    V2 deliberately keeps source discovery narrow: local metadata plus the
    packaged Quant-Perf policy default. Structured model-card and optional LLM
    evidence are added by later resolver stages without changing this public
    contract.
    """
    if discovery not in {"local", "online", "off"}:
        raise ValueError("eval discovery must be one of: local, online, off")
    base = build_eval_profile(
        model_ref,
        config_json=config_json,
        tokenizer_config=tokenizer_config,
    )
    policy = load_eval_policy()
    gsm8k = (policy.get("benchmarks") or {}).get("gsm8k") or {}
    policy_id = str(policy["policy_id"])
    policy_sha256 = _canonical_hash(policy)
    values: dict[str, Any] = {
        "schema_version": 2,
        "policy_version": "quark-quant-perf-gsm8k-profile-v2",
        "profile_id": (f"gsm8k-{base.model_mode}-{'nothink' if base.enable_thinking is False else 'default'}-v2"),
        "task": str(gsm8k.get("task") or "gsm8k"),
        "benchmark": "gsm8k",
        "request_type": str(gsm8k.get("request_type") or "generate_until"),
        "num_fewshot": int(gsm8k.get("num_fewshot") or 0),
        "prompting_strategy": str(gsm8k.get("prompting_strategy") or "cot"),
        "metric": str(gsm8k.get("metric") or "exact_match,flexible-extract"),
        "gen_kwargs": dict(gsm8k.get("gen_kwargs") or {"temperature": 0, "top_p": 1}),
        "evaluation_purpose": "quality_gate",
        "settings_source": "quark_policy_default",
        "source_reference": policy_id,
    }
    card_text = ""
    card_path = ""
    online_text = ""
    online_reference = ""
    paper_text = ""
    paper_reference = ""
    evidence: list[dict[str, Any]] = []
    if discovery != "off":
        card_text, card_path = _local_model_card(model_ref)
        card_values = _parse_model_card_command(card_text)
        if card_values:
            values.update(card_values)
            values["settings_source"] = "local_model_card"
            values["source_reference"] = card_path
            evidence.append(
                {
                    "source": card_path,
                    "source_sha256": _sha256_text(card_text),
                    "method": "structured_command",
                    "values": card_values,
                }
            )
        if not card_values and discovery == "online":
            model_card_loader = online_loader or _default_online_model_card
            online_text, online_reference = model_card_loader(model_ref)
            online_values = _parse_model_card_command(online_text)
            if online_values:
                values.update(online_values)
                values["settings_source"] = "online_model_card"
                values["source_reference"] = online_reference
                evidence.append(
                    {
                        "source": online_reference,
                        "source_sha256": _sha256_text(online_text),
                        "method": "structured_command",
                        "values": online_values,
                    }
                )
                card_values = online_values
            if not card_values:
                paper_reference = _linked_arxiv_url(online_text or card_text)
                if paper_reference:
                    paper_fetcher = paper_loader or _default_paper_loader
                    paper_text = paper_fetcher(paper_reference)
        llm_source = paper_text or card_text or online_text
        llm_reference = paper_reference or card_path or online_reference
        if not card_values and allow_llm and llm_source:
            relevant = _relevant_model_card_text(llm_source)
            extractor = llm_extractor
            artifact_root = Path(artifact_dir) if artifact_dir else None
            session_dir = (
                artifact_root.parents[1] if artifact_root is not None and len(artifact_root.parents) > 1 else None
            )
            output = (
                extractor(
                    relevant,
                    {
                        "benchmark": "gsm8k",
                        "model_ref": model_ref,
                    },
                )
                if extractor is not None
                else _default_llm_extractor(
                    relevant,
                    {
                        "benchmark": "gsm8k",
                        "model_ref": model_ref,
                    },
                    session_dir=session_dir,
                )
            )
            llm_values = _parse_llm_evidence(
                output,
                relevant,
            )
            if llm_values:
                values.update(llm_values)
                if paper_reference:
                    values["settings_source"] = "online_paper_llm"
                elif card_path:
                    values["settings_source"] = "local_model_card_llm"
                else:
                    values["settings_source"] = "online_model_card_llm"
                values["source_reference"] = llm_reference
                evidence.append(
                    {
                        "source": llm_reference,
                        "source_sha256": _sha256_text(llm_source),
                        "method": "llm_extraction",
                        "values": llm_values,
                    }
                )
    if overrides:
        override_values = {key: value for key, value in overrides.items() if value is not None}
        if override_values:
            values.update(override_values)
            values["settings_source"] = "user"
            values["source_reference"] = "Quark Quant-Perf CLI override"

    final_thinking = values.get(
        "enable_thinking",
        base.enable_thinking,
    )
    thinking_label = "think" if final_thinking is True else "nothink" if final_thinking is False else "default"
    values["profile_id"] = f"gsm8k-{base.model_mode}-{thinking_label}-v2"
    profile = replace(base, **values).with_computed_hash()
    sources = {
        "model_ref": model_ref,
        "discovery": discovery,
        "local_model_card": {
            "reference": card_path or None,
            "sha256": _sha256_text(card_text),
        },
        "online_model_card": {
            "reference": online_reference or None,
            "sha256": _sha256_text(online_text),
        },
        "online_paper": {
            "reference": paper_reference or None,
            "sha256": _sha256_text(paper_text),
        },
        "quark_policy": {
            "id": policy_id,
            "sha256": policy_sha256,
        },
    }
    evidence_document = {
        "schema_version": 1,
        "sources": sources,
        "evidence": evidence,
    }
    profile = replace(
        profile,
        evidence_hash=_canonical_hash(evidence_document),
    )
    if artifact_dir:
        root = Path(artifact_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "sources.json").write_text(json.dumps(sources, indent=2, sort_keys=True))
        (root / "evidence.json").write_text(
            json.dumps(
                evidence_document,
                indent=2,
                sort_keys=True,
            )
        )
        (root / "resolved_profile.json").write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
    return profile
