#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import hashlib
import importlib.util
import json
import runpy
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

from quark.experimental.speculative_decoding import run as run_module
from quark.experimental.speculative_decoding import setup as eagle3_setup
from quark.experimental.speculative_decoding import torchspec_runner


def _example_assets() -> Path:
    return Path(__file__).resolve().parents[2] / "examples" / "experimental" / "speculative_decoding"


def test_packaged_recipe_alias_is_cwd_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = run_module.load_config(
        "recipes/eagle3_default.yaml",
        [
            "model.target_model_path=alias/model",
            "training.output_dir=custom-output",
            "training.num_epochs=3",
        ],
    )

    assert cfg["execution"]["backend"] == "torchspec"
    assert cfg["model"]["target_model_path"] == "alias/model"
    assert cfg["training"]["output_dir"] == "custom-output"
    assert cfg["training"]["num_epochs"] == 3


def test_missing_non_packaged_recipe_is_not_reinterpreted(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="recipe not found"):
        run_module.load_config(str(tmp_path / "missing.yaml"), [])


def test_explicit_dotlist_wins_over_base_model_alias() -> None:
    cfg = run_module.load_config(
        str(Path(run_module.__file__).parent / "recipes" / "eagle3_default.yaml"),
        [
            "model.target_model_path=from-alias",
            "model.target_model_path=explicit-wins",
        ],
    )
    assert cfg["model"]["target_model_path"] == "explicit-wins"


def test_runner_environment_maps_public_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUARK_EAGLE3_CACHE", str(tmp_path / "cache"))
    cfg = run_module.load_config(
        "recipes/eagle3_default.yaml",
        [
            f"training.output_dir={tmp_path / 'run'}",
            "training.num_epochs=4",
            "data.num_prompts=12345",
            "benchmark.rounds=5",
        ],
    )

    env, report = torchspec_runner.build_runner_environment(cfg)

    assert env["BASE_MODEL"] == "Qwen/Qwen3-8B"
    assert env["EPOCHS"] == "4"
    assert env["NUM_PROMPTS"] == "12345"
    assert env["BENCH_ROUNDS"] == "5"
    assert env["MIN_SPEEDUP"] == "1.25"
    assert env["MIN_SERVED_AL"] == "2.35"
    assert env["RUNNER_PROFILE"] == "qwen3_8b_quick_start"
    assert env["TARGET_TP_SIZE"] == "1"
    assert env["target_tp_size"] == "1"
    assert env["CHAT_TEMPLATE"] == "qwen"
    assert env["WORLD_SIZE"] == "8"
    assert env["INFERENCE_NUM_GPUS"] == "4"
    assert env["TRAINING_NUM_GPUS"] == "4"
    assert env["CONFIG_DIR"].endswith("eagle3/qwen3_8b_quick_start/configs")
    assert env["TRAIN_CONFIG"].endswith("eagle3/qwen3_8b_quick_start/configs/qwen3_8b_eagle3.yaml")
    assert env["DRAFT_CONFIG_SOURCE"].endswith("eagle3/qwen3_8b_quick_start/configs/qwen3_8b_eagle3_draft.json")
    assert env["EX_DIR"] == str((tmp_path / "run").resolve())
    assert env["DATA_DIR"].startswith(str((tmp_path / "cache" / "data").resolve()))
    assert report == (tmp_path / "run" / "report.json").resolve()


def test_profiles_without_a_manifest_keep_their_existing_data_cache_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Qwen must not lose its validated on-policy dataset to an unrelated key."""
    monkeypatch.setenv("QUARK_EAGLE3_CACHE", str(tmp_path / "cache"))
    settings = torchspec_runner.resolve_torchspec_settings(run_module.load_config("recipes/eagle3_default.yaml", []))

    without_manifest = torchspec_runner._data_cache_path(tmp_path, "Qwen/Qwen3-8B", settings, "qwen", None)
    identity = {
        "base_model": "Qwen/Qwen3-8B",
        "prompt_dataset": settings.prompt_dataset,
        "num_prompts": settings.num_prompts,
        "eval_size": settings.eval_size,
        "temperature": 0,
        "max_tokens": settings.generation_max_tokens,
        "generation_reserve_ratio": 1.10,
        "chat_template": "qwen",
    }
    expected = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    assert without_manifest.name == expected

    manifest = _example_assets() / "eagle3" / "minimax_m3_best_recipe" / "domain_manifest.smoke.yaml"
    with_manifest = torchspec_runner._data_cache_path(tmp_path, "Qwen/Qwen3-8B", settings, "qwen", manifest)
    assert with_manifest.name != without_manifest.name


def test_quick_profile_uses_small_plumbing_defaults() -> None:
    cfg = run_module.load_config("recipes/eagle3_default.yaml", ["execution.profile=quick"])
    settings = torchspec_runner.resolve_torchspec_settings(cfg)
    assert settings.num_prompts == 2_000
    assert settings.eval_size == 64
    assert settings.benchmark_prompts == 16
    assert settings.benchmark_rounds == 1
    env, _ = torchspec_runner.build_runner_environment(cfg)
    assert env["EPOCHS"] == "1"
    assert env["GENERATION_MAX_TOKENS"] == "4096"


def test_package_import_does_not_eagerly_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import quark.experimental.speculative_decoding; raise SystemExit('torch' in sys.modules)",
        ],
        check=False,
    )
    assert result.returncode == 0


def test_backend_dispatch_keeps_native_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    native = Mock(return_value={"backend": "native"})
    monkeypatch.setattr(run_module, "_run_native_from_config", native)
    cfg = {"execution": {"backend": "native"}}
    assert run_module.run_from_config(cfg) == {"backend": "native"}
    native.assert_called_once_with(cfg)


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown execution.backend"):
        run_module.run_from_config({"execution": {"backend": "mystery"}})


def test_unprepared_environment_fails_before_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = run_module.load_config(
        "recipes/eagle3_default.yaml",
        [f"execution.cache_dir={tmp_path}", f"training.output_dir={tmp_path / 'run'}"],
    )
    monkeypatch.setattr(torchspec_runner, "_environment_ready", lambda *_args: False)
    with pytest.raises(RuntimeError, match="speculative_decoding.setup"):
        torchspec_runner.run_torchspec_from_config(cfg)


def test_completed_quality_gate_miss_returns_advisory_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    report_path = output_dir / "report.json"
    output_dir.mkdir()
    data_dir.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "median_speedup": 1.20,
                "served_al": 2.20,
                "min_speedup": 1.25,
                "min_served_al": 2.35,
                "draft_path": str(output_dir / "release" / "draft_hf"),
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "execution": {"backend": "torchspec", "profile": "full"},
        "model": {"target_model_path": "Qwen/Qwen3-8B"},
        "training": {"output_dir": str(output_dir)},
    }
    env = {
        "CACHE_DIR": str(cache_dir),
        "EX_DIR": str(output_dir),
        "DATA_DIR": str(data_dir),
        "RUNNER_PROFILE": "qwen3_8b_quick_start",
        "NUM_PROMPTS": "150000",
        "EPOCHS": "2",
    }
    launch = Mock()
    monkeypatch.setattr(torchspec_runner, "build_runner_environment", lambda _cfg: (env, report_path))
    monkeypatch.setattr(torchspec_runner, "_environment_ready", lambda *_args: True)
    monkeypatch.setattr(torchspec_runner, "find_runner_assets", lambda: tmp_path / "assets")
    monkeypatch.setattr(torchspec_runner.subprocess, "run", launch)

    report = torchspec_runner.run_torchspec_from_config(cfg)

    assert report["status"] == "failed"
    assert report["report_path"] == str(report_path)
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "failed"
    assert launch.call_args.kwargs["check"] is True


def test_source_checkout_runner_assets_are_discoverable() -> None:
    assets = torchspec_runner.find_runner_assets()
    common = assets / "eagle3" / "common"
    assert (common / "run_all.sh").is_file()
    assert (common / "scripts" / "00_setup.sh").is_file()
    assert (common / "docker" / "Dockerfile.rocm").is_file()
    train_script = (common / "scripts" / "train.sh").read_text(encoding="utf-8")
    assert "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1" in train_script


def test_every_profile_is_reachable_through_its_own_wrapper() -> None:
    """Dispatch happens in Python, so each profile needs a wrapper to dispatch to.

    A shell dispatcher at the example root used to re-implement the profile-name
    aliasing that ``_runner_profile_name`` performs, which is one definition too
    many for a name that decides which assets a run uses.
    """
    assets = _example_assets()

    for profile in (torchspec_runner._QWEN_RUNNER_PROFILE, torchspec_runner._LARGE_MODEL_RUNNER_PROFILE):
        wrapper = assets / "eagle3" / profile / "run.sh"
        assert wrapper.is_file(), profile
        assert not wrapper.is_symlink()
        assert "common/run_all.sh" in wrapper.read_text(encoding="utf-8"), profile

    # Nothing above the profile directories orchestrates a run any more.
    assert not (assets / "run_all.sh").exists()
    assert not (assets / "scripts").exists()
    assert not (assets / "configs").exists()
    assert not (assets / "docker").exists()


def test_response_parsers_are_reserved_for_deployment_shaped_serves() -> None:
    common = _example_assets() / "eagle3" / "common"
    shared = (common / "scripts" / "_common.sh").read_text(encoding="utf-8")
    runner = (common / "run_all.sh").read_text(encoding="utf-8")

    # Reasoning and tool-call parsers move text out of `content`, so on-policy
    # capture in Phase 1 must not see them.
    assert "reasoning-parser" not in shared.split("SERVE_PARSER_ARGS")[0]
    assert "--reasoning-parser minimax_m3" in shared
    assert "SERVE_PARSER_ARGS" in shared

    phase_one, phase_four = runner.split("Phase 4: baseline + spec serves")
    assert "ENABLE_RESPONSE_PARSERS" not in phase_one
    assert "export ENABLE_RESPONSE_PARSERS=1" in phase_four

    for script in ("serve_target.sh", "serve_spec.sh"):
        text = (common / "scripts" / script).read_text(encoding="utf-8")
        assert '"${ENABLE_RESPONSE_PARSERS:-0}" = "1"' in text


def _run_finalizer(tmp_path: Path, rows: list[dict[str, Any]], *, eval_size: int, seed: int = 0) -> dict[str, Any]:
    script = _example_assets() / "eagle3" / "common" / "scripts" / "finalize_onpolicy_data.py"
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "all.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--train",
            str(train),
            "--eval",
            str(evaluation),
            "--manifest-out",
            str(tmp_path / "manifest.json"),
            "--requested",
            str(len(rows)),
            "--eval-size",
            str(eval_size),
            "--generation-max-tokens",
            "128",
            "--seed",
            str(seed),
        ],
        check=True,
        capture_output=True,
    )

    def read(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    return {"train": read(train), "eval": read(evaluation)}


def test_finalizer_splits_are_deterministic_disjoint_and_domain_stratified(tmp_path: Path) -> None:
    """The train/eval split runs here, after generation, not over raw prompts.

    Determinism lets a rerun reuse a cached dataset, disjointness keeps the
    held-out rows honest, and the quota is proportional so no domain is
    evaluated out of proportion to its share of the data.
    """
    rows = [
        {
            "domain": domain,
            "conversations": [
                {"role": "user", "content": f"{domain} question {index}"},
                {"role": "assistant", "content": f"{domain} answer {index}"},
            ],
        }
        for domain, count in (("code", 12), ("math_reasoning", 8))
        for index in range(count)
    ]

    first = _run_finalizer(tmp_path / "a", rows, eval_size=5)
    second = _run_finalizer(tmp_path / "b", list(reversed(rows)), eval_size=5)

    assert first == second
    assert len(first["eval"]) == 5
    assert len(first["train"]) == 15

    def texts(split: str) -> set[str]:
        return {row["conversations"][0]["content"] for row in first[split]}

    assert texts("train").isdisjoint(texts("eval"))
    assert texts("train") | texts("eval") == {row["conversations"][0]["content"] for row in rows}
    # 12:8 of 5 held-out rows, allocated by largest remainder.
    assert Counter(row["domain"] for row in first["eval"]) == {"code": 3, "math_reasoning": 2}


def test_finalizer_rejects_generations_with_detached_reasoning(tmp_path: Path) -> None:
    script = _example_assets() / "eagle3" / "common" / "scripts" / "finalize_onpolicy_data.py"
    source = tmp_path / "all.jsonl"
    source.write_text(
        json.dumps(
            {
                "conversations": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "answer", "thinking": "hidden reasoning"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--train",
            str(tmp_path / "train.jsonl"),
            "--eval",
            str(tmp_path / "eval.jsonl"),
            "--manifest-out",
            str(tmp_path / "manifest.json"),
            "--requested",
            "1",
            "--eval-size",
            "1",
            "--generation-max-tokens",
            "128",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "reasoning outside assistant content" in result.stderr


def test_minimax_profile_defers_system_rendering_to_public_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: dict[str, object] = {}

    class Registry:
        @staticmethod
        def get_all_template_names() -> list[str]:
            # Mirrors the real registry, which lists a template once registered.
            # sitecustomize re-reads this to confirm the registration took.
            return list(registered)

        @staticmethod
        def register(name: str, template: object) -> None:
            registered[name] = template

    template_module = types.ModuleType("torchspec.data.template")
    template_module.TEMPLATE_REGISTRY = Registry()
    template_module.ChatTemplate = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "torchspec", types.ModuleType("torchspec"))
    monkeypatch.setitem(sys.modules, "torchspec.data", types.ModuleType("torchspec.data"))
    monkeypatch.setitem(sys.modules, "torchspec.data.template", template_module)
    monkeypatch.setenv("TORCHSPEC_CHAT_TEMPLATE_PROFILE", "minimax-m3")

    sitecustomize = _example_assets() / "eagle3" / "common" / "python" / "sitecustomize.py"
    runpy.run_path(str(sitecustomize))

    template = registered["minimax-m3"]
    assert isinstance(template, dict)
    assert template["system_prompt"] is None
    assert template["parser_type"] == "general"
    assert template["assistant_header"] == "]~b]ai\n"
    assert template["end_of_turn_token"] == "[e~["


def test_public_large_model_adapter_maps_profile_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUARK_EAGLE3_CACHE", str(tmp_path / "cache"))
    profile_dir = _example_assets() / "eagle3" / "minimax_m3_best_recipe"
    adapter = yaml.safe_load((profile_dir / "minimax_m3_mxfp4_adapter.yaml").read_text(encoding="utf-8"))
    cfg = {
        "execution": {"backend": "torchspec", "profile": "quick"},
        "model": {
            "target_model_path": adapter["model"]["repository"],
            "embedding_key": adapter["model"]["embedding_key"],
            "lm_head_key": adapter["model"]["lm_head_key"],
            "norm_key": adapter["model"]["norm_key"],
        },
        "data": {"chat_template": adapter["model"]["chat_template"]},
        "training": {"output_dir": str(tmp_path / "large-run"), "num_gpus_per_node": 8},
        "inference": {
            "target_tp_size": adapter["runner"]["target_tp_size"],
            "env": adapter["runtime"]["environment"],
        },
        "quant": {"target": adapter["model"]["quantization"]},
    }

    env, _ = torchspec_runner.build_runner_environment(cfg)

    assert env["RUNNER_PROFILE"] == "minimax_m3_best_recipe"
    assert env["BASE_MODEL"] == "amd/MiniMax-M3-MXFP4"
    assert env["TARGET_TP_SIZE"] == "4"
    assert env["CHAT_TEMPLATE"] == "minimax-m3"
    assert env["DRAFT_CONFIG_SOURCE"] == ""
    assert env["TARGET_EMBEDDING_KEY"] == "language_model.model.embed_tokens.weight"
    assert env["TARGET_LM_HEAD_KEY"] == "language_model.lm_head.weight"
    assert env["TARGET_NORM_KEY"] == "language_model.model.norm.weight"
    assert env["VLLM_ROCM_USE_AITER"] == "1"
    assert env["VLLM_ROCM_USE_AITER_MOE"] == "1"
    for key in ("CONFIG_DIR", "TRAIN_CONFIG", "DOMAIN_MANIFEST"):
        assert Path(env[key]).exists()
    # Nothing downstream reads these, so exporting them only made the recipe
    # look wired up. Asserting their absence keeps them from drifting back.
    for key in ("MODEL_ADAPTER", "RUNNER_PROFILE_FILE", "TARGET_QUANTIZATION"):
        assert key not in env, key


def test_large_model_defaults_are_loaded_from_profile_assets(tmp_path: Path) -> None:
    profile_dir = tmp_path / "eagle3" / "minimax_m3_best_recipe"
    config_dir = profile_dir / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "train.yaml").write_text("model: {}\n", encoding="utf-8")
    (profile_dir / "domain.yaml").write_text("version: 1\ndomains: {}\n", encoding="utf-8")
    (profile_dir / "adapter.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "model": {
                    "repository": "public/example-model",
                    "quantization": "mxfp4",
                    "chat_template": "public-template",
                    "embedding_key": "text.embed.weight",
                    "lm_head_key": "text.head.weight",
                    "norm_key": "text.norm.weight",
                },
                "runner": {
                    "profile": "minimax_m3_best_recipe",
                    "target_tp_size": 2,
                },
                "runtime": {
                    "environment": {
                        "VLLM_ROCM_USE_AITER_MOE": "1",
                        "PUBLIC_TEST_ENV": "enabled",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    profile_manifest = {
        "schema_version": 1,
        "runner_profile": "minimax_m3_best_recipe",
        "model_adapter": "adapter.yaml",
        "assets": {
            "config_dir": "configs",
            "train_config": "configs/train.yaml",
        },
        "topology": {
            "world_size": 6,
            "target_tp_size": 2,
            "inference_num_gpus": 2,
            "training_num_gpus": 4,
        },
        "data_manifests": {
            "smoke": "domain.yaml",
            "full_template": "domain.full.yaml",
        },
    }
    manifest_path = profile_dir / "large_model_profile.yaml"
    manifest_path.write_text(yaml.safe_dump(profile_manifest, sort_keys=False), encoding="utf-8")

    profile = torchspec_runner.resolve_runner_asset_profile(
        {"execution": {"runner_profile": "minimax_m3_best_recipe"}},
        asset_root=tmp_path,
    )

    assert profile.default_base_model == "public/example-model"
    assert profile.default_chat_template == "public-template"
    assert profile.default_target_tp_size == 2
    assert profile.default_world_size == 6
    assert profile.default_inference_num_gpus == 2
    assert profile.default_training_num_gpus == 4
    assert dict(profile.runtime_env)["PUBLIC_TEST_ENV"] == "enabled"

    profile_manifest["assets"]["train_config"] = "../outside.yaml"
    manifest_path.write_text(yaml.safe_dump(profile_manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="must not escape"):
        torchspec_runner.resolve_runner_asset_profile(
            {"execution": {"runner_profile": "minimax_m3_best_recipe"}},
            asset_root=tmp_path,
        )


def test_large_model_smoke_manifest_materializes_generic_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QUARK_EAGLE3_CACHE", str(tmp_path / "cache"))
    cfg = {
        "execution": {"backend": "torchspec", "profile": "quick", "runner_profile": "minimax_m3"},
        "model": {"target_model_path": "amd/MiniMax-M3-MXFP4"},
        "training": {"output_dir": str(tmp_path / "run"), "num_gpus_per_node": 8},
    }
    env, _ = torchspec_runner.build_runner_environment(cfg)
    Path(env["DATA_DIR"]).mkdir(parents=True)

    torchspec_runner._prepare_domain_manifest_prompts(env)

    prompts = [
        json.loads(line) for line in (Path(env["DATA_DIR"]) / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(prompts) == 15
    assert {row["domain"] for row in prompts} == {
        "general_instruction",
        "code",
        "math_reasoning",
        "question_answering",
        "multilingual",
    }
    assert env["NUM_PROMPTS"] == "15"
    assert env["EVAL_N"] == "3"
    assert env["DOMAIN_MANIFEST_ACTIVE"] == "1"
    assert (Path(env["DATA_DIR"]) / "prompt_provenance.json").is_file()


def test_cached_dataset_reports_its_own_row_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reported counts must describe the cached rows, not the profile defaults.

    A cached dataset is reached whenever the manifest and workload identity are
    unchanged, and it need not hold the number of prompts the profile requests.
    """
    monkeypatch.setenv("QUARK_EAGLE3_CACHE", str(tmp_path / "cache"))
    cfg = {
        "execution": {"backend": "torchspec", "profile": "quick", "runner_profile": "minimax_m3"},
        "model": {"target_model_path": "amd/MiniMax-M3-MXFP4"},
        "training": {"output_dir": str(tmp_path / "run"), "num_gpus_per_node": 8},
    }
    env, _ = torchspec_runner.build_runner_environment(cfg)
    data_dir = Path(env["DATA_DIR"])
    data_dir.mkdir(parents=True)
    (data_dir / "onpolicy_train.jsonl").write_text("{}\n" * 7, encoding="utf-8")
    (data_dir / "onpolicy_eval.jsonl").write_text("{}\n" * 2, encoding="utf-8")

    torchspec_runner._prepare_domain_manifest_prompts(env)

    assert env["NUM_PROMPTS"] == "9"
    assert env["EVAL_N"] == "2"
    assert env["DOMAIN_MANIFEST_ACTIVE"] == "1"
    # The cached data is reused as-is, so nothing is re-materialized.
    assert not (data_dir / "prompts.jsonl").exists()


def test_large_model_recipe_pins_no_target_specific_layer_ids() -> None:
    """Layer ids stay derived from target depth so they cannot go stale.

    The rule itself is covered against ``make_draft_config.py``, which is what
    the runner actually executes.
    """
    config = run_module.load_config("recipes/eagle3_mxfp4_moe.yaml", [])
    assert "aux_hidden_layers" not in config["eagle"]


def test_an_explicit_gate_survives_every_profile_and_workload() -> None:
    """A gate the recipe asks for has to reach the benchmark, not be zeroed.

    The runner used to pass ``benchmark.min_speedup`` through unconditionally
    and the shell then overwrote it for smoke runs and for every non-Qwen
    profile, so a large-model recipe that wanted a gate silently got none.
    """
    gated = {
        "benchmark": {
            "min_speedup": 1.4,
            "min_served_al": 2.6,
            "quick_min_speedup": 1.1,
            "quick_min_served_al": 2.0,
        }
    }

    for profile, expected in (("full", (1.4, 2.6)), ("quick", (1.1, 2.0))):
        cfg = {"execution": {"profile": profile}, **gated}
        settings = torchspec_runner.resolve_torchspec_settings(cfg)
        assert (settings.min_speedup, settings.min_served_al) == expected, profile

    # A smoke run defaults to no gate rather than inheriting the full one.
    smoke = torchspec_runner.resolve_torchspec_settings(
        {"execution": {"profile": "quick"}, "benchmark": {"min_speedup": 1.4, "min_served_al": 2.6}}
    )
    assert (smoke.min_speedup, smoke.min_served_al) == (0.0, 0.0)

    runner = (_example_assets() / "eagle3" / "common" / "run_all.sh").read_text(encoding="utf-8")
    # The shell computes a default and defers to an explicit value; assigning
    # first and overwriting after is what discarded it.
    assert 'MIN_SPEEDUP="${MIN_SPEEDUP:-$_default_min_speedup}"' in runner
    assert "MIN_SPEEDUP=0" not in runner


def test_large_model_full_profile_benchmarks_on_qwen_footing() -> None:
    """A published large-model number needs the same sample size as Qwen's."""
    quick = torchspec_runner.resolve_torchspec_settings(
        run_module.load_config("recipes/eagle3_mxfp4_moe.yaml", ["execution.profile=quick"])
    )
    full = torchspec_runner.resolve_torchspec_settings(
        run_module.load_config("recipes/eagle3_mxfp4_moe.yaml", ["execution.profile=full"])
    )
    qwen_full = torchspec_runner.resolve_torchspec_settings(run_module.load_config("recipes/eagle3_default.yaml", []))

    assert (full.benchmark_prompts, full.benchmark_rounds) == (
        qwen_full.benchmark_prompts,
        qwen_full.benchmark_rounds,
    )
    # The smoke profile stays cheap.
    assert (quick.benchmark_prompts, quick.benchmark_rounds) == (8, 1)


def test_large_model_extraction_engine_leaves_headroom_for_training_ranks() -> None:
    """Extraction and training share the node, so the engine must not hoard memory."""
    config = yaml.safe_load(
        (_example_assets() / "eagle3" / "minimax_m3_best_recipe" / "configs" / "large_model_eagle3.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["inference"]["vllm"]["mem_fraction_static"] <= 0.5


def test_large_model_extraction_engine_pins_a_block_compatible_attention_backend() -> None:
    """The in-process extraction engine ignores serve flags and the env var.

    A sparse-attention target pins the KV block size to 128. AITER Flash
    Attention only offers kernel block sizes 16 and 32, so leaving the backend
    to runtime selection fails engine start-up with "No common block size".
    """
    config = yaml.safe_load(
        (_example_assets() / "eagle3" / "minimax_m3_best_recipe" / "configs" / "large_model_eagle3.yaml").read_text(
            encoding="utf-8"
        )
    )
    extra = config["inference"]["vllm"]["extra_args"]

    assert extra["attention_backend"] == "TRITON_ATTN"
    assert extra["block_size"] == 128
    # The MoE fast path is a separate choice and stays on AITER.
    assert extra["moe_backend"] == "aiter"

    serve_args = (_example_assets() / "eagle3" / "common" / "scripts" / "_common.sh").read_text(encoding="utf-8")
    assert "--attention-backend TRITON_ATTN" in serve_args
    assert "--block-size 128" in serve_args


def test_large_model_profile_contains_only_generic_asset_references() -> None:
    profile_dir = _example_assets() / "eagle3" / "minimax_m3_best_recipe"
    profile = yaml.safe_load((profile_dir / "large_model_profile.yaml").read_text(encoding="utf-8"))
    adapter = yaml.safe_load((profile_dir / profile["model_adapter"]).read_text(encoding="utf-8"))

    assert profile["runner_profile"] == "minimax_m3_best_recipe"
    assert adapter["model"]["repository"] == "amd/MiniMax-M3-MXFP4"
    assert profile["data_manifests"] == {
        "smoke": "domain_manifest.smoke.yaml",
        "full_template": "domain_manifest.full.example.yaml",
    }
    # Every reference resolves: the loader rejects unknown fields, so a stale
    # entry cannot linger here as a path that points at nothing.
    for reference in (*profile["assets"].values(), profile["model_adapter"], *profile["data_manifests"].values()):
        assert (profile_dir / reference).exists(), reference


def test_public_large_model_assets_exclude_legacy_reference_values() -> None:
    root = Path(__file__).resolve().parents[2]
    profile_dir = _example_assets() / "eagle3" / "minimax_m3_best_recipe"
    paths = [
        root / "quark" / "experimental" / "speculative_decoding" / "recipes" / "eagle3_mxfp4_moe.yaml",
        root / "docs" / "source" / "eagle3_best_recipe.rst",
        *(path for path in profile_dir.rglob("*") if path.is_file()),
    ]
    public_text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = (
        "/home/larryli2",
        "onpolicy_mixed",
        "draft_vocab_size: 32000",
        "aux_hidden_layers: [2, 30, 57]",
        "/vocab/d2t.pt",
        "base-chat : code/tech",
        "noise_floor_al",
        "gpu_memory_utilization: 0.88",
    )

    for value in forbidden:
        assert value not in public_text


def test_large_model_report_redacts_machine_identifiers(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "run"
    data_dir = tmp_path / "data"
    (cache_dir / "manifests").mkdir(parents=True)
    output_dir.mkdir()
    data_dir.mkdir()
    setup_manifest = cache_dir / "manifests" / "setup-MiniMax-M3-MXFP4.json"
    setup_manifest.write_text(
        json.dumps(
            {
                "hostname": "private-host",
                "model_config": "/private/model/config.json",
                "platform": "Linux",
                "gpu_inventory": {
                    "card0": {
                        "Card Series": "AMD Instinct MI350X",
                        "GFX Version": "gfx950",
                        "Node ID": "3",
                        "GUID": "private-guid-0",
                    },
                    "card1": {
                        "Card Series": "AMD Instinct MI350X",
                        "GFX Version": "gfx950",
                        "Node ID": "5",
                        "GUID": "private-guid-1",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "execution": {"runner_profile": "minimax_m3_best_recipe", "cache_dir": "/private/cache"},
        "model": {"target_model_path": "amd/MiniMax-M3-MXFP4"},
        "training": {"output_dir": "/private/output"},
    }
    (data_dir / "prompt_provenance.json").write_text(
        json.dumps({"manifest": "/private/data/domain_manifest.yaml", "total_records": 15}),
        encoding="utf-8",
    )
    report = {"draft_path": "/private/output/release/draft_hf"}

    torchspec_runner._augment_report(
        report,
        cfg,
        cache_dir,
        output_dir,
        data_dir,
        "amd/MiniMax-M3-MXFP4",
    )

    published = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert published["draft_path"] == "release/draft_hf"
    assert published["report_path"] == "report.json"
    assert "output_dir" not in published
    assert published["resolved_recipe"]["execution"].get("cache_dir") is None
    assert published["resolved_recipe"]["training"]["output_dir"] == "<run-output>"
    environment = published["environment"]
    assert "hostname" not in environment
    assert "model_config" not in environment
    assert environment["gpu_inventory"] == {
        "count": 2,
        "models": ["AMD Instinct MI350X"],
        "gfx_versions": ["gfx950"],
    }
    assert published["data_provenance"]["manifest"] == "domain_manifest.yaml"
    report_text = (output_dir / "report.json").read_text(encoding="utf-8")
    assert "private-guid" not in report_text
    assert "/private/" not in report_text


def test_report_returned_to_the_caller_keeps_the_real_paths(tmp_path: Path) -> None:
    """Redaction is for the file; the caller still has to locate the draft.

    ``run_torchspec_from_config`` returns this dict and logs ``draft_path`` on
    success, so redacting in place told the operator the draft was at a relative
    path with no run directory attached.
    """
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "run"
    data_dir = tmp_path / "data"
    (cache_dir / "manifests").mkdir(parents=True)
    output_dir.mkdir()
    data_dir.mkdir()
    cfg = {
        "execution": {"runner_profile": "minimax_m3_best_recipe", "cache_dir": "/private/cache"},
        "model": {"target_model_path": "amd/MiniMax-M3-MXFP4"},
        "training": {"output_dir": str(output_dir)},
    }
    draft = str(output_dir / "release" / "draft_hf")
    report = {"draft_path": draft, "report_path": str(output_dir / "report.json")}

    torchspec_runner._augment_report(report, cfg, cache_dir, output_dir, data_dir, "amd/MiniMax-M3-MXFP4")

    assert report["draft_path"] == draft
    assert report["report_path"] == str(output_dir / "report.json")
    assert report["output_dir"] == str(output_dir)
    assert report["resolved_recipe"]["training"]["output_dir"] == str(output_dir)
    # The file beside the run is still the publishable one.
    published = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert published["draft_path"] == "release/draft_hf"


def test_serve_credentials_never_reach_a_report_on_any_profile(tmp_path: Path) -> None:
    """Path redaction is a publishability tradeoff; a credential leak is not.

    The remaining redactions apply only to the published large-model profile,
    so the default profile has to drop the serve environment and endpoint on
    its own rather than inheriting that behaviour.
    """
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    cfg = {
        "execution": {"runner_profile": "qwen3_8b_quick_start"},
        "model": {"target_model_path": "Qwen/Qwen3-8B"},
        "training": {"output_dir": str(output_dir)},
        "inference": {
            "target_endpoint": "http://internal-host:8000",
            "env": {"HF_TOKEN": "hf_secret_value", "OPENAI_API_KEY": "sk-secret"},
            "tp_size": 1,
        },
    }
    report: dict[str, object] = {}

    torchspec_runner._augment_report(report, cfg, tmp_path / "cache", output_dir, tmp_path / "data", "Qwen/Qwen3-8B")

    inference = report["resolved_recipe"]["inference"]
    assert "env" not in inference
    assert "target_endpoint" not in inference
    # Unrelated serving settings still survive.
    assert inference["tp_size"] == 1
    report_text = (output_dir / "report.json").read_text(encoding="utf-8")
    assert "hf_secret_value" not in report_text
    assert "sk-secret" not in report_text
    assert "internal-host" not in report_text


def test_large_model_report_drops_absolute_paths_and_serve_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The published report must survive the real load path, not just _augment_report."""
    output_dir = tmp_path / "run"
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    output_dir.mkdir()
    data_dir.mkdir()
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps({"status": "passed", "median_speedup": 1.4, "served_al": 2.6, "draft_path": str(output_dir)}),
        encoding="utf-8",
    )
    cfg = {
        "execution": {"backend": "torchspec", "runner_profile": "minimax_m3_best_recipe"},
        "model": {"target_model_path": "amd/MiniMax-M3-MXFP4"},
        "training": {"output_dir": str(output_dir)},
        "inference": {"env": {"PRIVATE_TOKEN": "secret-value"}, "target_endpoint": "http://private:8000"},
    }
    env = {
        "CACHE_DIR": str(cache_dir),
        "EX_DIR": str(output_dir),
        "DATA_DIR": str(data_dir),
        "RUNNER_PROFILE": "minimax_m3_best_recipe",
        "NUM_PROMPTS": "15",
        "EPOCHS": "1",
    }
    monkeypatch.setattr(torchspec_runner, "build_runner_environment", lambda _cfg: (env, report_path))
    monkeypatch.setattr(torchspec_runner, "_environment_ready", lambda *_args: True)
    monkeypatch.setattr(torchspec_runner, "find_runner_assets", lambda: tmp_path / "assets")
    monkeypatch.setattr(torchspec_runner.subprocess, "run", Mock())

    report = torchspec_runner.run_torchspec_from_config(cfg)

    published = json.loads(report_path.read_text(encoding="utf-8"))
    assert published["report_path"] == "report.json"
    serialized = report_path.read_text(encoding="utf-8")
    assert "secret-value" not in serialized
    assert "http://private:8000" not in serialized
    assert str(tmp_path) not in serialized
    # The caller still needs the run directory it just filled.
    assert report["output_dir"] == str(output_dir)
    assert "secret-value" not in json.dumps(report["resolved_recipe"])


def test_documented_qwen_cli_commands_are_unchanged() -> None:
    docs = Path(__file__).resolve().parents[2] / "docs" / "source" / "eagle3_quick_start.rst"
    normalized = " ".join(docs.read_text(encoding="utf-8").replace("\\\n", " ").split())
    commands = (
        "python3 -m quark.experimental.speculative_decoding.setup --base_model Qwen/Qwen3-8B",
        "python3 -m quark.experimental.speculative_decoding.run --base_model Qwen/Qwen3-8B",
        "python3 -m quark.experimental.speculative_decoding.run --config recipes/eagle3_default.yaml "
        "--base_model Qwen/Qwen3-8B training.output_dir=ckpts/qwen3-8b-eagle3 training.num_epochs=2",
    )
    for command in commands:
        assert command in normalized


def test_setup_passes_pinned_cache_to_idempotent_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assets = tmp_path / "assets"
    relative = Path("eagle3") / "common" / "scripts" / "00_setup.sh"
    # The stub mirrors the packaged layout, and the assertion below ties that
    # layout back to the checkout: a stub alone would keep passing after the
    # real script moved.
    assert (_example_assets() / relative).is_file()
    script = assets / relative
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    run = Mock()
    monkeypatch.setattr(eagle3_setup, "find_runner_assets", lambda: assets)
    monkeypatch.setattr(eagle3_setup.subprocess, "run", run)
    monkeypatch.setattr(eagle3_setup, "_write_manifest", lambda *_args: manifest)

    result = eagle3_setup.prepare_environment(
        "Qwen/Qwen3-8B",
        cache_dir=str(tmp_path / "cache"),
        image="test-image",
        torchspec_commit="abc123",
    )

    assert result == manifest
    command = run.call_args.args[0]
    env = run.call_args.kwargs["env"]
    assert command == ["bash", str(script), "--base_model", "Qwen/Qwen3-8B"]
    assert env["TORCHSPEC_COMMIT"] == "abc123"
    assert env["IMG"] == "test-image"
    assert env["MODELS"] == str((tmp_path / "cache" / "models").resolve())


def _load_benchmark_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "experimental"
        / "speculative_decoding"
        / "eagle3"
        / "common"
        / "scripts"
        / "bench.py"
    )
    spec = importlib.util.spec_from_file_location("quark_eagle3_bench_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_writes_median_gate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = _load_benchmark_module()
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"conversations": [{"role": "user", "content": "hello"}]}) + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    baseline = iter([100.0, 102.0, 98.0])
    speculative = iter([132.0, 137.7, 132.3])

    def fake_run(port, _model, _prompts, _max_tokens):
        tps = next(baseline if port == 8000 else speculative)
        return {"tokens": 512, "seconds": 512 / tps, "tokens_per_second": tps}

    monkeypatch.setattr(bench, "warmup", lambda *_args: None)
    monkeypatch.setattr(bench, "run", fake_run)
    monkeypatch.setattr(bench, "scrape_al", lambda _root: 2.58)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench.py",
            "--eval",
            str(eval_path),
            "--rounds",
            "3",
            "--json-out",
            str(report_path),
            "--min-speedup",
            "1.30",
            "--min-served-al",
            "2.50",
        ],
    )

    bench.main()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["median_speedup"] >= 1.30
    assert report["served_al"] == 2.58
    assert len(report["rounds"]) == 3


def test_benchmark_gate_failure_is_advisory_and_preserves_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bench = _load_benchmark_module()
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"conversations": [{"role": "user", "content": "hello"}]}) + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "failed.json"
    monkeypatch.setattr(bench, "warmup", lambda *_args: None)
    monkeypatch.setattr(
        bench,
        "run",
        lambda *_args: {"tokens": 512, "seconds": 5.12, "tokens_per_second": 100.0},
    )
    monkeypatch.setattr(bench, "scrape_al", lambda _root: 1.5)
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench.py", "--eval", str(eval_path), "--rounds", "1", "--json-out", str(report_path)],
    )

    bench.main()
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "failed"
    assert "continuing without a process error" in capsys.readouterr().out


def test_sequential_benchmark_combines_baseline_and_spec_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = _load_benchmark_module()
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"conversations": [{"role": "user", "content": "hello"}]}) + "\n",
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.json"
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(bench, "warmup", lambda *_args: None)
    monkeypatch.setattr(
        bench,
        "run",
        lambda port, *_args: {
            "tokens": 512,
            "seconds": 5.12 if port == 8000 else 4.0,
            "tokens_per_second": 100.0 if port == 8000 else 128.0,
        },
    )
    monkeypatch.setattr(bench, "scrape_al", lambda _root: 2.1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench.py",
            "--mode",
            "baseline",
            "--eval",
            str(eval_path),
            "--rounds",
            "2",
            "--json-out",
            str(baseline_path),
        ],
    )
    bench.main()
    assert len(json.loads(baseline_path.read_text(encoding="utf-8"))["baseline_rounds"]) == 2

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench.py",
            "--mode",
            "speculative",
            "--baseline-json",
            str(baseline_path),
            "--eval",
            str(eval_path),
            "--rounds",
            "2",
            "--min-speedup",
            "0",
            "--min-served-al",
            "0",
            "--json-out",
            str(report_path),
        ],
    )
    bench.main()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["median_speedup"] == pytest.approx(1.28)
    assert report["served_al"] == 2.1


def test_resolve_export_checkpoint_prefers_best_marker(tmp_path: Path) -> None:
    common_sh = _example_assets() / "eagle3" / "common" / "scripts" / "_common.sh"
    ckpt_dir = tmp_path / "checkpoints"
    (ckpt_dir / "iter_0005001" / "model").mkdir(parents=True)
    (ckpt_dir / "iter_0009359" / "model").mkdir(parents=True)
    (ckpt_dir / "best_checkpointed_iteration.txt").write_text("5001\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", "-c", f'source "{common_sh}"; resolve_export_checkpoint "{ckpt_dir}"'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip().endswith("iter_0005001")


def test_resolve_export_checkpoint_falls_back_to_latest(tmp_path: Path) -> None:
    common_sh = _example_assets() / "eagle3" / "common" / "scripts" / "_common.sh"
    ckpt_dir = tmp_path / "checkpoints"
    (ckpt_dir / "iter_0005001" / "model").mkdir(parents=True)
    (ckpt_dir / "iter_0009359" / "model").mkdir(parents=True)

    result = subprocess.run(
        ["bash", "-c", f'source "{common_sh}"; resolve_export_checkpoint "{ckpt_dir}"'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip().endswith("iter_0009359")


def test_resolve_export_checkpoint_skips_incomplete_best_marker(tmp_path: Path) -> None:
    common_sh = _example_assets() / "eagle3" / "common" / "scripts" / "_common.sh"
    ckpt_dir = tmp_path / "checkpoints"
    (ckpt_dir / "iter_0005001").mkdir(parents=True)
    (ckpt_dir / "iter_0009359" / "model").mkdir(parents=True)
    (ckpt_dir / "best_checkpointed_iteration.txt").write_text("5001\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", "-c", f'source "{common_sh}"; resolve_export_checkpoint "{ckpt_dir}"'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip().endswith("iter_0009359")
    assert "incomplete iter_0005001" in result.stderr


def test_runner_input_identity_changes_with_data_bytes(tmp_path: Path) -> None:
    common_sh = _example_assets() / "eagle3" / "common" / "scripts" / "_common.sh"
    config = tmp_path / "active.yaml"
    train = tmp_path / "train.jsonl"
    config.write_text("seed: 42\n", encoding="utf-8")
    train.write_text('{"prompt":"first"}\n', encoding="utf-8")

    def identity() -> str:
        # Not a login shell: the sourced helpers need nothing from a profile,
        # and a profile that greets the operator would land on stdout ahead of
        # the digest.
        result = subprocess.run(
            ["bash", "-c", f'source "{common_sh}"; runner_input_sha256 "{config}" "{train}"'],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    first = identity()
    train.write_text('{"prompt":"second"}\n', encoding="utf-8")
    second = identity()

    assert len(first) == 64
    assert first != second


def test_benchmark_gate_reports_when_only_served_al_is_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bench = _load_benchmark_module()
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"conversations": [{"role": "user", "content": "hello"}]}) + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "failed_al.json"
    monkeypatch.setattr(bench, "warmup", lambda *_args: None)
    monkeypatch.setattr(
        bench,
        "run",
        lambda port, *_args: {
            "tokens": 512,
            "seconds": 512 / (135.0 if port == 8100 else 100.0),
            "tokens_per_second": 135.0 if port == 8100 else 100.0,
        },
    )
    monkeypatch.setattr(bench, "scrape_al", lambda _root: 2.44)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench.py",
            "--eval",
            str(eval_path),
            "--rounds",
            "1",
            "--json-out",
            str(report_path),
            "--min-speedup",
            "1.30",
            "--min-served-al",
            "2.50",
        ],
    )

    bench.main()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["median_speedup"] >= 1.30
    assert report["served_al"] == pytest.approx(2.44)
    captured = capsys.readouterr()
    assert "served AL 2.440 < 2.50" in captured.out
    assert "best marked checkpoint" in captured.out
    assert "continuing without a process error" in captured.out


def test_benchmark_gate_reports_when_only_speedup_is_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bench = _load_benchmark_module()
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"conversations": [{"role": "user", "content": "hello"}]}) + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "failed_speedup.json"
    monkeypatch.setattr(bench, "warmup", lambda *_args: None)
    monkeypatch.setattr(
        bench,
        "run",
        lambda port, *_args: {
            "tokens": 512,
            "seconds": 512 / (125.0 if port == 8100 else 100.0),
            "tokens_per_second": 125.0 if port == 8100 else 100.0,
        },
    )
    monkeypatch.setattr(bench, "scrape_al", lambda _root: 2.58)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench.py",
            "--eval",
            str(eval_path),
            "--rounds",
            "1",
            "--json-out",
            str(report_path),
            "--min-speedup",
            "1.30",
            "--min-served-al",
            "2.50",
        ],
    )

    bench.main()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["served_al"] == pytest.approx(2.58)
    assert report["median_speedup"] < 1.30
    captured = capsys.readouterr()
    assert "median speedup 1.250x < 1.30x" in captured.out
    assert "continuing without a process error" in captured.out
