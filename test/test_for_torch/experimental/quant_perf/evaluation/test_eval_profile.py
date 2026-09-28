#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for deterministic GSM8K evaluation profiles."""

import hashlib
import json
from dataclasses import replace

from quark.experimental.torch.quant_perf.evaluation import profile as eval_profile
from quark.experimental.torch.quant_perf.evaluation.profile import build_eval_profile


def _write_model(tmp_path, config, tokenizer_config=None):
    (tmp_path / "config.json").write_text(json.dumps(config))
    if tokenizer_config is not None:
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))


def test_chat_thinking_model_gets_nothink_profile(tmp_path):
    _write_model(
        tmp_path,
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "max_position_embeddings": 40960,
        },
        {"chat_template": ("{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}")},
    )

    profile = build_eval_profile(str(tmp_path))

    assert profile.model_mode == "chat"
    assert profile.apply_chat_template is True
    assert profile.enable_thinking is False
    assert profile.max_model_len == 8192
    assert profile.profile_id == "gsm8k-chat-nothink-v1"


def test_base_model_omits_chat_and_thinking_settings(tmp_path):
    _write_model(
        tmp_path,
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "max_position_embeddings": 4096,
        },
        {},
    )

    profile = build_eval_profile(str(tmp_path))

    assert profile.model_mode == "base"
    assert profile.apply_chat_template is False
    assert profile.enable_thinking is None
    assert profile.max_model_len == 4096
    assert profile.profile_id == "gsm8k-base-default-v1"


def test_conditional_generation_architecture_is_base_without_template(tmp_path):
    _write_model(
        tmp_path,
        {
            "model_type": "vision_language",
            "architectures": ["VisionLanguageForConditionalGeneration"],
        },
        {},
    )

    profile = build_eval_profile(str(tmp_path))

    assert profile.model_mode == "base"
    assert profile.apply_chat_template is False
    assert profile.detection_reason == "fallback_base"


def test_chat_named_model_without_template_does_not_apply_chat_template(tmp_path):
    _write_model(
        tmp_path,
        {
            "_name_or_path": "org/model-it",
            "model_type": "vision_language",
            "architectures": ["VisionLanguageForConditionalGeneration"],
        },
        {},
    )

    profile = build_eval_profile(str(tmp_path))

    assert profile.model_mode == "chat"
    assert profile.apply_chat_template is False
    assert profile.detection_reason == "model_name"


def test_standalone_chat_template_is_detected(tmp_path):
    _write_model(
        tmp_path,
        {
            "_name_or_path": "org/model-it",
            "model_type": "vision_language",
            "architectures": ["VisionLanguageForConditionalGeneration"],
        },
        {},
    )
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")

    profile = build_eval_profile(str(tmp_path))

    assert profile.model_mode == "chat"
    assert profile.apply_chat_template is True
    assert profile.detection_reason == "chat_template"


def test_profile_hash_is_stable_and_changes_with_settings(tmp_path):
    _write_model(
        tmp_path,
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
        {},
    )

    first = build_eval_profile(str(tmp_path))
    second = build_eval_profile(str(tmp_path))
    changed = replace(first, max_gen_toks=2048).with_computed_hash()

    assert first.profile_hash == second.profile_hash
    assert first.profile_hash != changed.profile_hash


def test_resolved_profile_uses_packaged_gsm8k_policy_default(tmp_path):
    _write_model(
        tmp_path,
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "max_position_embeddings": 4096,
        },
        {},
    )

    resolver = getattr(eval_profile, "resolve_eval_profile", None)
    assert resolver is not None

    profile = resolver(str(tmp_path), discovery="local", allow_llm=False)

    assert profile.task == "gsm8k"
    assert profile.num_fewshot == 5
    assert profile.prompting_strategy == "cot"
    assert profile.settings_source == "quark_policy_default"
    assert profile.source_reference == "quark-quant-perf-gsm8k-v2"
    assert profile.policy_version == "quark-quant-perf-gsm8k-profile-v2"
    assert profile.profile_hash
    assert profile.evidence_hash


def test_empty_override_values_do_not_claim_user_source(tmp_path):
    _write_model(
        tmp_path,
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
        },
        {},
    )

    profile = eval_profile.resolve_eval_profile(
        str(tmp_path),
        discovery="off",
        allow_llm=False,
        overrides={
            "task": None,
            "num_fewshot": None,
            "prompting_strategy": None,
            "enable_thinking": None,
            "max_gen_toks": None,
        },
    )

    assert profile.settings_source == "quark_policy_default"


def test_resolved_profile_user_overrides_win(tmp_path):
    _write_model(
        tmp_path,
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        },
        {"chat_template": "{{ messages }}"},
    )

    resolver = getattr(eval_profile, "resolve_eval_profile", None)
    assert resolver is not None

    profile = resolver(
        str(tmp_path),
        discovery="off",
        allow_llm=False,
        overrides={
            "task": "gsm8k_cot_zeroshot",
            "num_fewshot": 0,
            "prompting_strategy": "cot",
            "enable_thinking": False,
            "max_gen_toks": 768,
        },
    )

    assert profile.task == "gsm8k_cot_zeroshot"
    assert profile.num_fewshot == 0
    assert profile.apply_chat_template is True
    assert profile.enable_thinking is False
    assert profile.max_gen_toks == 768
    assert profile.settings_source == "user"


def test_profile_id_reflects_final_thinking_override(tmp_path):
    _write_model(
        tmp_path,
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        },
        {"chat_template": ("{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}")},
    )

    profile = eval_profile.resolve_eval_profile(
        str(tmp_path),
        discovery="off",
        allow_llm=False,
        overrides={"enable_thinking": True},
    )

    assert profile.enable_thinking is True
    assert profile.profile_id == "gsm8k-chat-think-v2"


def test_local_model_card_command_overrides_policy_default(tmp_path):
    _write_model(
        tmp_path,
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        },
        {"chat_template": "{{ messages }}"},
    )
    (tmp_path / "README.md").write_text(
        """
```bash
lm_eval --tasks gsm8k_cot_zeroshot --num_fewshot 0 \
  --apply_chat_template
```
"""
    )

    profile = eval_profile.resolve_eval_profile(
        str(tmp_path),
        discovery="local",
        allow_llm=False,
    )

    assert profile.task == "gsm8k_cot_zeroshot"
    assert profile.num_fewshot == 0
    assert profile.prompting_strategy == "cot"
    assert profile.settings_source == "local_model_card"
    assert profile.source_reference.endswith("README.md")


def test_evidence_hash_addresses_persisted_source_document(tmp_path):
    _write_model(
        tmp_path,
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        },
        {"chat_template": "{{ messages }}"},
    )
    readme = tmp_path / "README.md"
    readme.write_text("lm_eval --tasks gsm8k --num_fewshot 5\n")
    first_dir = tmp_path / "first"
    first = eval_profile.resolve_eval_profile(
        str(tmp_path),
        discovery="local",
        allow_llm=False,
        artifact_dir=first_dir,
    )
    first_document = json.loads((first_dir / "evidence.json").read_text())
    expected_hash = hashlib.sha256(
        json.dumps(
            first_document,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    readme.write_text("# Updated documentation\nlm_eval --tasks gsm8k --num_fewshot 5\n")
    second = eval_profile.resolve_eval_profile(
        str(tmp_path),
        discovery="local",
        allow_llm=False,
        artifact_dir=tmp_path / "second",
    )

    assert first.evidence_hash == expected_hash
    assert first_document["sources"]["quark_policy"]["id"] == "quark-quant-perf-gsm8k-v2"
    assert "quark_skill" not in first_document["sources"]
    assert first.profile_hash == second.profile_hash
    assert first.evidence_hash != second.evidence_hash


def test_recomputing_profile_hash_preserves_resolved_evidence_hash(tmp_path):
    _write_model(
        tmp_path,
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        },
        {"chat_template": "{{ messages }}"},
    )
    (tmp_path / "README.md").write_text("lm_eval --tasks gsm8k --num_fewshot 5\n")
    profile = eval_profile.resolve_eval_profile(
        str(tmp_path),
        discovery="local",
        allow_llm=False,
        artifact_dir=tmp_path / "profile",
    )

    refreshed = profile.with_computed_hash()

    assert refreshed.evidence_hash == profile.evidence_hash


def test_llm_evidence_is_used_only_when_grounded_in_source(tmp_path):
    _write_model(
        tmp_path,
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
        },
        {},
    )
    evidence_text = "GSM8K is evaluated using 4-shot chain-of-thought."
    (tmp_path / "README.md").write_text(evidence_text)

    profile = eval_profile.resolve_eval_profile(
        str(tmp_path),
        discovery="local",
        allow_llm=True,
        llm_extractor=lambda _text, _context: json.dumps(
            {
                "task": "gsm8k",
                "num_fewshot": 4,
                "prompting_strategy": "cot",
                "evidence_text": evidence_text,
            }
        ),
    )

    assert profile.num_fewshot == 4
    assert profile.prompting_strategy == "cot"
    assert profile.settings_source == "local_model_card_llm"


def test_ungrounded_llm_evidence_falls_back_to_policy_default(tmp_path):
    _write_model(
        tmp_path,
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
        },
        {},
    )
    (tmp_path / "README.md").write_text("No benchmark details.")

    profile = eval_profile.resolve_eval_profile(
        str(tmp_path),
        discovery="local",
        allow_llm=True,
        llm_extractor=lambda _text, _context: json.dumps(
            {
                "task": "gsm8k",
                "num_fewshot": 2,
                "prompting_strategy": "direct",
                "evidence_text": "This sentence is not in the model card.",
            }
        ),
    )

    assert profile.num_fewshot == 5
    assert profile.prompting_strategy == "cot"
    assert profile.settings_source == "quark_policy_default"


def test_online_discovery_uses_bounded_official_model_card_loader(tmp_path):
    _write_model(
        tmp_path,
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
        },
        {},
    )
    calls = []

    profile = eval_profile.resolve_eval_profile(
        str(tmp_path),
        discovery="online",
        allow_llm=False,
        online_loader=lambda model_ref: (
            calls.append(model_ref)
            or (
                "lm_eval --tasks gsm8k --num_fewshot 3",
                "https://huggingface.co/org/model/raw/main/README.md",
            )
        ),
    )

    assert calls == [str(tmp_path)]
    assert profile.num_fewshot == 3
    assert profile.settings_source == "online_model_card"


def test_local_discovery_never_calls_online_loader(tmp_path):
    _write_model(
        tmp_path,
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
        },
        {},
    )
    calls = []

    profile = eval_profile.resolve_eval_profile(
        str(tmp_path),
        discovery="local",
        allow_llm=False,
        online_loader=lambda model_ref: calls.append(model_ref) or ("", ""),
    )

    assert calls == []
    assert profile.settings_source == "quark_policy_default"


def test_quant_perf_eval_policy_is_self_contained():
    loader = getattr(eval_profile, "load_eval_policy", None)
    assert loader is not None

    policy = loader()

    assert policy["policy_id"] == "quark-quant-perf-gsm8k-v2"
    assert "source" not in policy
    assert policy["benchmarks"]["gsm8k"]["num_fewshot"] == 5
    assert policy["benchmarks"]["gsm8k"]["prompting_strategy"] == "cot"


def test_online_discovery_can_extract_from_explicit_linked_paper(
    tmp_path,
):
    _write_model(
        tmp_path,
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
        },
        {},
    )
    paper_text = "GSM8K uses 4-shot chain-of-thought evaluation."
    paper_calls = []

    profile = eval_profile.resolve_eval_profile(
        str(tmp_path),
        discovery="online",
        allow_llm=True,
        online_loader=lambda _model_ref: (
            "Technical report: https://arxiv.org/abs/2601.01234",
            "https://huggingface.co/org/model/raw/main/README.md",
        ),
        paper_loader=lambda url: paper_calls.append(url) or paper_text,
        llm_extractor=lambda text, _context: json.dumps(
            {
                "task": "gsm8k",
                "num_fewshot": 4,
                "prompting_strategy": "cot",
                "evidence_text": paper_text,
            }
        ),
    )

    assert paper_calls == ["https://arxiv.org/html/2601.01234"]
    assert profile.num_fewshot == 4
    assert profile.settings_source == "online_paper_llm"
    assert profile.source_reference == ("https://arxiv.org/html/2601.01234")
