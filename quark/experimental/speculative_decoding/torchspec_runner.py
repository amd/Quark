#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Bridge the Quark CLI to the validated multi-GPU TorchSpec pipeline.

The implementation deliberately keeps Docker/TorchSpec out of Quark's Python
dependency graph.  The one-time setup command prepares those external
dependencies, while this module translates a Quark YAML recipe into the
environment consumed by the packaged end-to-end runner.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from quark.experimental.speculative_decoding.config import TorchSpecRunConfig
from quark.experimental.speculative_decoding.data.domain_manifest import (
    build_domain_records,
    load_domain_manifest,
    provenance_license_summary,
    write_jsonl,
)
from quark.experimental.speculative_decoding.utils.logging import get_logger

logger = get_logger(__name__)

_EXAMPLE_REL = Path("examples") / "experimental" / "speculative_decoding"
_INSTALLED_ASSET_DIR = "_torchspec_assets"
_QWEN_RUNNER_PROFILE = "qwen3_8b_quick_start"
_LARGE_MODEL_RUNNER_PROFILE = "minimax_m3_best_recipe"
_RUNNER_SCHEMA_VERSION = 1
_PROFILE_KEYS = frozenset({"schema_version", "runner_profile", "model_adapter", "assets", "topology", "data_manifests"})
_PROFILE_ASSET_KEYS = frozenset({"config_dir", "train_config"})
_PROFILE_TOPOLOGY_KEYS = frozenset({"world_size", "target_tp_size", "inference_num_gpus", "training_num_gpus"})
_PROFILE_MANIFEST_KEYS = frozenset({"smoke", "full_template"})
_ADAPTER_KEYS = frozenset({"schema_version", "model", "runner", "runtime"})
_ADAPTER_MODEL_KEYS = frozenset(
    {"repository", "trust_remote_code", "quantization", "chat_template", "embedding_key", "lm_head_key", "norm_key"}
)
_ADAPTER_RUNNER_KEYS = frozenset({"profile", "target_tp_size"})
_ADAPTER_RUNTIME_KEYS = frozenset({"environment"})
_RUNNER_PROFILE_ALIASES = {
    "default": _QWEN_RUNNER_PROFILE,
    "qwen": _QWEN_RUNNER_PROFILE,
    "qwen3_8b": _QWEN_RUNNER_PROFILE,
    _QWEN_RUNNER_PROFILE: _QWEN_RUNNER_PROFILE,
    "large_model": _LARGE_MODEL_RUNNER_PROFILE,
    "minimax": _LARGE_MODEL_RUNNER_PROFILE,
    "minimax_m3": _LARGE_MODEL_RUNNER_PROFILE,
    _LARGE_MODEL_RUNNER_PROFILE: _LARGE_MODEL_RUNNER_PROFILE,
}


@dataclass(frozen=True)
class RunnerAssetProfile:
    """Resolved, packaged assets and public runtime defaults for one target family."""

    name: str
    profile_dir: Path
    config_dir: Path
    train_config: Path
    draft_config: Path | None
    model_adapter: Path | None
    profile_manifest: Path | None
    domain_manifest: Path | None
    default_chat_template: str
    default_target_tp_size: int
    default_base_model: str
    default_embedding_key: str
    default_lm_head_key: str
    default_norm_key: str
    default_world_size: int = 8
    default_inference_num_gpus: int = 4
    default_training_num_gpus: int = 4
    default_max_model_len: int = 8192
    runtime_env: tuple[tuple[str, str], ...] = ()


def _model_name(model: str) -> str:
    return Path(model.rstrip("/")).name


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-")
    return value or "model"


def find_runner_assets() -> Path:
    """Return the runner asset root in a checkout or installed wheel."""
    package_dir = Path(__file__).resolve().parent
    source_tree = package_dir.parents[2] / _EXAMPLE_REL
    installed = package_dir / _INSTALLED_ASSET_DIR
    for candidate in (source_tree, installed):
        common = candidate / "eagle3" / "common"
        if (common / "run_all.sh").is_file() and (common / "scripts" / "00_setup.sh").is_file():
            return candidate
    raise FileNotFoundError(
        "TorchSpec runner assets are missing. Reinstall amd-quark from a wheel that includes "
        "quark.experimental.speculative_decoding runner assets."
    )


def _runner_profile_name(cfg: dict[str, Any]) -> str:
    execution = cfg.get("execution", {})
    requested = execution.get("runner_profile") or execution.get("asset_profile") or cfg.get("runner_profile")
    if requested is None:
        target = str(cfg.get("model", {}).get("target_model_path", "")).lower()
        requested = _LARGE_MODEL_RUNNER_PROFILE if "minimax-m3" in target else _QWEN_RUNNER_PROFILE
    normalized = str(requested).strip().lower().replace("-", "_")
    try:
        return _RUNNER_PROFILE_ALIASES[normalized]
    except KeyError as error:
        known = ", ".join(sorted({_QWEN_RUNNER_PROFILE, _LARGE_MODEL_RUNNER_PROFILE}))
        raise ValueError(f"unknown EAGLE-3 runner profile {requested!r}; expected one of: {known}") from error


def _yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a YAML mapping: {path}")
    return value


def _nested_mapping(value: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"{label}.{key} must be a mapping")
    return nested


def _check_keys(value: Mapping[str, Any], allowed: frozenset[str], label: str) -> None:
    """Reject fields nothing reads, the way the domain manifest loader does.

    Without this a key can sit in a profile or adapter looking like a setting
    while having no effect, which is how several accumulated here before.
    """
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


def _check_schema_version(value: dict[str, Any], label: str) -> None:
    version = value.get("schema_version")
    if version != _RUNNER_SCHEMA_VERSION:
        raise ValueError(f"{label}.schema_version must be {_RUNNER_SCHEMA_VERSION}, got {version!r}")


def _required_string(value: dict[str, Any], key: str, label: str) -> str:
    resolved = value.get(key)
    if not isinstance(resolved, str) or not resolved.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return resolved.strip()


def _profile_asset_path(profile_dir: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative to the runner profile")
    root = profile_dir.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must not escape the runner profile: {value}") from error
    return resolved


def resolve_runner_asset_profile(
    cfg: dict[str, Any],
    *,
    asset_root: Path | None = None,
) -> RunnerAssetProfile:
    """Resolve profile-specific assets without changing the Qwen CLI defaults."""
    root = asset_root or find_runner_assets()
    profile_name = _runner_profile_name(cfg)
    profile_dir = root / "eagle3" / profile_name
    config_dir = profile_dir / "configs"

    if profile_name == _QWEN_RUNNER_PROFILE:
        profile = RunnerAssetProfile(
            name=profile_name,
            profile_dir=profile_dir,
            config_dir=config_dir,
            train_config=config_dir / "qwen3_8b_eagle3.yaml",
            draft_config=config_dir / "qwen3_8b_eagle3_draft.json",
            model_adapter=None,
            profile_manifest=None,
            domain_manifest=None,
            default_chat_template="qwen",
            default_target_tp_size=1,
            default_base_model="Qwen/Qwen3-8B",
            default_embedding_key="model.embed_tokens.weight",
            default_lm_head_key="lm_head.weight",
            default_norm_key="model.norm.weight",
            runtime_env=(
                ("VLLM_ROCM_USE_AITER", "1"),
                ("VLLM_ROCM_USE_AITER_MOE", "0"),
                ("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "0"),
                ("VLLM_ATTENTION_BACKEND", "TRITON_ATTN"),
                ("VLLM_USE_BREAKABLE_CUDAGRAPH", "0"),
                ("MC_STORE_MEMCPY", "0"),
            ),
        )
    else:
        profile_manifest = profile_dir / "large_model_profile.yaml"
        profile_values = _yaml_mapping(profile_manifest, "runner profile")
        _check_schema_version(profile_values, "runner profile")
        _check_keys(profile_values, _PROFILE_KEYS, "runner profile")
        if profile_values.get("runner_profile") != profile_name:
            raise ValueError(
                f"runner profile name mismatch in {profile_manifest}: "
                f"expected {profile_name!r}, got {profile_values.get('runner_profile')!r}"
            )
        assets = _nested_mapping(profile_values, "assets", "runner profile")
        _check_keys(assets, _PROFILE_ASSET_KEYS, "runner profile.assets")
        topology = _nested_mapping(profile_values, "topology", "runner profile")
        _check_keys(topology, _PROFILE_TOPOLOGY_KEYS, "runner profile.topology")
        manifests = _nested_mapping(profile_values, "data_manifests", "runner profile")
        _check_keys(manifests, _PROFILE_MANIFEST_KEYS, "runner profile.data_manifests")
        model_adapter = _profile_asset_path(
            profile_dir,
            profile_values.get("model_adapter"),
            "runner profile.model_adapter",
        )
        adapter_values = _yaml_mapping(model_adapter, "model adapter")
        _check_schema_version(adapter_values, "model adapter")
        _check_keys(adapter_values, _ADAPTER_KEYS, "model adapter")
        model = _nested_mapping(adapter_values, "model", "model adapter")
        _check_keys(model, _ADAPTER_MODEL_KEYS, "model adapter.model")
        runner = _nested_mapping(adapter_values, "runner", "model adapter")
        _check_keys(runner, _ADAPTER_RUNNER_KEYS, "model adapter.runner")
        runtime = _nested_mapping(adapter_values, "runtime", "model adapter")
        _check_keys(runtime, _ADAPTER_RUNTIME_KEYS, "model adapter.runtime")
        if runner.get("profile") != profile_name:
            raise ValueError(f"model adapter runner.profile must be {profile_name!r}, got {runner.get('profile')!r}")
        # Part of the adapter contract, and checked for that reason alone: the
        # recipe's own `quant.target` is what the pipeline acts on.
        _required_string(model, "quantization", "model adapter")
        adapter_tp_size = _positive_int("model adapter.runner.target_tp_size", runner.get("target_tp_size"))
        topology_tp_size = _positive_int(
            "runner profile.topology.target_tp_size",
            topology.get("target_tp_size"),
        )
        if adapter_tp_size != topology_tp_size:
            raise ValueError(
                "model adapter and runner profile target_tp_size values must match: "
                f"{adapter_tp_size} != {topology_tp_size}"
            )
        runtime_environment = runtime.get("environment", {})
        if not isinstance(runtime_environment, dict):
            raise ValueError("model adapter.runtime.environment must be a mapping")
        merged_runtime_environment = {
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_ROCM_USE_AITER_MOE": "0",
            "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "0",
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
            "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
            "MC_STORE_MEMCPY": "0",
            **{str(key): str(value) for key, value in runtime_environment.items()},
        }
        config_dir = _profile_asset_path(profile_dir, assets.get("config_dir"), "runner profile.assets.config_dir")
        profile = RunnerAssetProfile(
            name=profile_name,
            profile_dir=profile_dir,
            config_dir=config_dir,
            train_config=_profile_asset_path(
                profile_dir,
                assets.get("train_config"),
                "runner profile.assets.train_config",
            ),
            draft_config=None,
            model_adapter=model_adapter,
            profile_manifest=profile_manifest,
            domain_manifest=_profile_asset_path(
                profile_dir,
                manifests.get("smoke"),
                "runner profile.data_manifests.smoke",
            ),
            default_chat_template=_required_string(model, "chat_template", "model adapter"),
            default_target_tp_size=adapter_tp_size,
            default_base_model=_required_string(model, "repository", "model adapter"),
            default_embedding_key=_required_string(model, "embedding_key", "model adapter"),
            default_lm_head_key=_required_string(model, "lm_head_key", "model adapter"),
            default_norm_key=_required_string(model, "norm_key", "model adapter"),
            default_world_size=_positive_int("runner profile.topology.world_size", topology.get("world_size")),
            default_inference_num_gpus=_positive_int(
                "runner profile.topology.inference_num_gpus",
                topology.get("inference_num_gpus"),
            ),
            default_training_num_gpus=_positive_int(
                "runner profile.topology.training_num_gpus",
                topology.get("training_num_gpus"),
            ),
            runtime_env=tuple(merged_runtime_environment.items()),
        )

    required = [profile.profile_dir, profile.config_dir, profile.train_config]
    required.extend(
        path
        for path in (profile.draft_config, profile.model_adapter, profile.profile_manifest, profile.domain_manifest)
        if path
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"runner profile {profile.name!r} is missing packaged assets: {', '.join(missing)}")
    return profile


def resolve_torchspec_settings(cfg: dict[str, Any]) -> TorchSpecRunConfig:
    """Normalize recipe sections into the validated runner settings."""
    execution = cfg.get("execution", {})
    data = cfg.get("data", {})
    benchmark = cfg.get("benchmark", {})
    profile = str(execution.get("profile", "full"))
    quick = profile == "quick"
    return TorchSpecRunConfig(
        profile=profile,
        cache_dir=str(execution.get("cache_dir", "~/.cache/amd-quark/eagle3")),
        image=str(execution.get("image", "quark-specdec-rocm:latest")),
        prompt_dataset=str(data.get("prompt_dataset", "allenai/tulu-3-sft-mixture")),
        num_prompts=int(data.get("quick_num_prompts", 2_000) if quick else data.get("num_prompts", 150_000)),
        eval_size=int(data.get("quick_eval_size", 64) if quick else data.get("eval_size", 256)),
        generation_max_tokens=int(data.get("generation_max_tokens", 4096)),
        benchmark_prompts=int(benchmark.get("quick_num_prompts", 16) if quick else benchmark.get("num_prompts", 40)),
        benchmark_rounds=int(benchmark.get("quick_rounds", 1) if quick else benchmark.get("rounds", 3)),
        # Gates follow the same quick_* shape as the prompt and round counts
        # above: a smoke run defaults to no gate, and a recipe that wants one
        # anyway says so rather than having the runner discard it.
        min_speedup=float(benchmark.get("quick_min_speedup", 0.0) if quick else benchmark.get("min_speedup", 1.25)),
        min_served_al=float(
            benchmark.get("quick_min_served_al", 0.0) if quick else benchmark.get("min_served_al", 2.35)
        ),
    )


def setup_hint(base_model: str) -> str:
    return f"python3 -m quark.experimental.speculative_decoding.setup --base_model {base_model}"


def _domain_manifest_fingerprint(path: Path) -> str:
    """Hash a manifest and its local JSONL inputs for cache isolation."""
    digest = hashlib.sha256(path.read_bytes())
    manifest = load_domain_manifest(path)
    for domain in manifest.domains.values():
        for source in domain.sources:
            if source.source_type != "jsonl" or source.path is None:
                continue
            source_path = Path(source.path).expanduser()
            if not source_path.is_absolute():
                source_path = manifest.base_dir / source_path
            digest.update(source.identity.encode("utf-8"))
            digest.update(source_path.read_bytes())
    return digest.hexdigest()


def _data_cache_path(
    cache_dir: Path,
    base_model: str,
    settings: TorchSpecRunConfig,
    chat_template: str,
    domain_manifest: Path | None = None,
) -> Path:
    identity: dict[str, Any] = {
        "base_model": base_model,
        "prompt_dataset": settings.prompt_dataset,
        "num_prompts": settings.num_prompts,
        "eval_size": settings.eval_size,
        "temperature": 0,
        "max_tokens": settings.generation_max_tokens,
        "generation_reserve_ratio": 1.10,
        "chat_template": chat_template,
    }
    # Added only when a manifest drives the prompts, so profiles without one
    # keep the identity — and therefore the cached dataset — they already had.
    if domain_manifest:
        identity["domain_manifest_sha256"] = _domain_manifest_fingerprint(domain_manifest)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / "data" / _slug(base_model) / digest


def _environment_ready(cache_dir: Path, image: str, base_model: str) -> bool:
    torchspec = cache_dir / "TorchSpec"
    model = cache_dir / "models" / _model_name(base_model) / "config.json"
    if not (torchspec / ".git").is_dir() or not model.is_file():
        return False
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _public_gpu_summary(inventory: object) -> dict[str, object] | None:
    """Reduce rocm-smi inventory to non-identifying model/count metadata."""
    if not isinstance(inventory, dict):
        return None
    cards = [card for card in inventory.values() if isinstance(card, dict)]
    if not cards:
        return None
    models = sorted({str(card["Card Series"]) for card in cards if card.get("Card Series")})
    gfx_versions = sorted({str(card["GFX Version"]) for card in cards if card.get("GFX Version")})
    return {
        "count": len(cards),
        "models": models,
        "gfx_versions": gfx_versions,
    }


def _publishable_report(report: dict[str, Any], large_model: bool) -> dict[str, Any]:
    """Return the copy that goes to disk, without machine-local detail.

    Only the file is redacted. The caller keeps the original so it can still
    tell an operator where the draft actually landed.
    """
    published = deepcopy(report)
    if not large_model:
        return published

    published["draft_path"] = "release/draft_hf"
    published["report_path"] = "report.json"
    published.pop("output_dir", None)
    recipe = published.get("resolved_recipe")
    if isinstance(recipe, dict):
        if isinstance(recipe.get("execution"), dict):
            recipe["execution"].pop("cache_dir", None)
        if isinstance(recipe.get("training"), dict):
            recipe["training"]["output_dir"] = "<run-output>"
        if isinstance(recipe.get("data"), dict) and recipe["data"].get("domain_manifest"):
            recipe["data"]["domain_manifest"] = Path(str(recipe["data"]["domain_manifest"])).name
    environment = published.get("environment")
    if isinstance(environment, dict):
        environment.pop("hostname", None)
        environment.pop("model_config", None)
        gpu_summary = _public_gpu_summary(environment.get("gpu_inventory"))
        if gpu_summary is None:
            environment.pop("gpu_inventory", None)
        else:
            environment["gpu_inventory"] = gpu_summary
    provenance = published.get("data_provenance")
    if isinstance(provenance, dict) and provenance.get("manifest"):
        provenance["manifest"] = Path(str(provenance["manifest"])).name
    return published


def _augment_report(
    report: dict[str, Any],
    cfg: dict[str, Any],
    cache_dir: Path,
    output_dir: Path,
    data_dir: Path,
    base_model: str,
) -> None:
    active_config = output_dir / "runtime" / "active_config.yaml"
    setup_manifest = cache_dir / "manifests" / f"setup-{_model_name(base_model)}.json"
    large_model = _runner_profile_name(cfg) == _LARGE_MODEL_RUNNER_PROFILE
    report["base_model"] = base_model
    report["output_dir"] = str(output_dir)
    report["active_config_sha256"] = _file_sha256(active_config) if active_config.is_file() else None
    resolved_recipe = deepcopy(cfg)
    # A user-supplied serve environment can hold credentials, and an endpoint
    # names a host. Neither is worth keeping even in memory, so this is not
    # conditioned on the profile: the redactions in `_publishable_report` trade
    # local convenience for publishability, but these two would be a credential
    # leak in a file the operator is invited to share.
    if isinstance(resolved_recipe.get("inference"), dict):
        resolved_recipe["inference"].pop("target_endpoint", None)
        resolved_recipe["inference"].pop("env", None)
    report["resolved_recipe"] = resolved_recipe
    if setup_manifest.is_file():
        with setup_manifest.open(encoding="utf-8") as f:
            report["environment"] = json.load(f)
    provenance = data_dir / "prompt_provenance.json"
    if large_model and provenance.is_file():
        report["data_provenance"] = json.loads(provenance.read_text(encoding="utf-8"))
    published = _publishable_report(report, large_model)
    tmp = output_dir / "report.json.tmp"
    tmp.write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output_dir / "report.json")


def _first_defined(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}") from error
    if resolved < 1:
        raise ValueError(f"{name} must be >= 1, got {resolved}")
    return resolved


def _profile_reference(profile: RunnerAssetProfile, value: Any, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return _profile_asset_path(profile.profile_dir, str(path), "profile-relative reference")


def build_runner_environment(cfg: dict[str, Any]) -> tuple[dict[str, str], Path]:
    """Build the subprocess environment without launching the expensive run."""
    settings = resolve_torchspec_settings(cfg)
    asset_profile = resolve_runner_asset_profile(cfg)
    execution = cfg.get("execution", {})
    model_cfg = cfg.get("model", {})
    training = cfg.get("training", {})
    data = cfg.get("data", {})
    inference = cfg.get("inference", {})
    benchmark = cfg.get("benchmark", {})

    base_model = str(model_cfg.get("target_model_path", asset_profile.default_base_model))
    output_dir = Path(str(training.get("output_dir", "ckpts/eagle3"))).expanduser().resolve()
    cache_dir = Path(os.environ.get("QUARK_EAGLE3_CACHE", settings.cache_dir)).expanduser().resolve()
    chat_template = str(data.get("chat_template") or asset_profile.default_chat_template)
    domain_manifest = _profile_reference(
        asset_profile,
        data.get("domain_manifest"),
        default=asset_profile.domain_manifest,
    )
    data_dir = _data_cache_path(cache_dir, base_model, settings, chat_template, domain_manifest)
    report_path = output_dir / "report.json"
    epochs = (
        int(training.get("quick_num_epochs", 1)) if settings.profile == "quick" else int(training.get("num_epochs", 2))
    )

    target_tp_size = _positive_int(
        "target_tp_size",
        _first_defined(
            execution.get("target_tp_size"),
            inference.get("target_tp_size"),
            inference.get("tp_size"),
            cfg.get("target_tp_size"),
            asset_profile.default_target_tp_size,
        ),
    )
    world_size = _positive_int(
        "world_size",
        _first_defined(
            execution.get("world_size"),
            training.get("world_size"),
            training.get("num_gpus_per_node"),
            asset_profile.default_world_size,
        ),
    )
    if target_tp_size > world_size or world_size % target_tp_size:
        raise ValueError(f"target_tp_size={target_tp_size} must divide world_size={world_size} and cannot exceed it")
    inference_num_gpus = _positive_int(
        "inference_num_gpus",
        _first_defined(
            execution.get("inference_num_gpus"),
            inference.get("inference_num_gpus"),
            asset_profile.default_inference_num_gpus,
        ),
    )
    training_num_gpus = _positive_int(
        "training_num_gpus",
        _first_defined(
            execution.get("training_num_gpus"),
            training.get("training_num_gpus"),
            training.get("training_num_gpus_per_node"),
            asset_profile.default_training_num_gpus,
        ),
    )
    target_max_model_len = _positive_int(
        "target_max_model_len",
        _first_defined(
            execution.get("target_max_model_len"),
            inference.get("max_model_len"),
            data.get("max_seq_length"),
            asset_profile.default_max_model_len,
        ),
    )
    draft_config = (
        asset_profile.draft_config
        if asset_profile.draft_config is not None and _model_name(base_model) == "Qwen3-8B"
        else None
    )

    env = os.environ.copy()
    env.update(
        {
            "BASE_MODEL": base_model,
            "EX_DIR": str(output_dir),
            "CACHE_DIR": str(cache_dir),
            "TORCHSPEC_DIR": str(cache_dir / "TorchSpec"),
            "MODELS": str(cache_dir / "models"),
            "DATA_DIR": str(data_dir),
            "IMG": settings.image,
            "PROFILE": settings.profile,
            "RUNNER_PROFILE": asset_profile.name,
            "CONFIG_DIR": str(asset_profile.config_dir),
            "TRAIN_CONFIG": str(asset_profile.train_config),
            "DRAFT_CONFIG_SOURCE": str(draft_config) if draft_config else "",
            "TARGET_TP_SIZE": str(target_tp_size),
            "target_tp_size": str(target_tp_size),
            "WORLD_SIZE": str(world_size),
            "INFERENCE_NUM_GPUS": str(inference_num_gpus),
            "TRAINING_NUM_GPUS": str(training_num_gpus),
            "TARGET_MAX_MODEL_LEN": str(target_max_model_len),
            "NUM_PROMPTS": str(settings.num_prompts),
            "EPOCHS": str(epochs),
            "EVAL_N": str(settings.eval_size),
            "GENERATION_MAX_TOKENS": str(settings.generation_max_tokens),
            "BENCH_N": str(settings.benchmark_prompts),
            "BENCH_ROUNDS": str(settings.benchmark_rounds),
            "MIN_SPEEDUP": str(settings.min_speedup),
            "MIN_SERVED_AL": str(settings.min_served_al),
            "PROMPT_DATASET": settings.prompt_dataset,
            "CHAT_TEMPLATE": chat_template,
            "TARGET_EMBEDDING_KEY": str(model_cfg.get("embedding_key") or asset_profile.default_embedding_key),
            "TARGET_LM_HEAD_KEY": str(model_cfg.get("lm_head_key") or asset_profile.default_lm_head_key),
            "TARGET_NORM_KEY": str(model_cfg.get("norm_key") or asset_profile.default_norm_key),
            "NST": str(int(benchmark.get("num_speculative_tokens", 3))),
            "REPORT_PATH": str(report_path),
        }
    )
    # Serving knobs are defaults, not overrides: `_common.sh` writes them as
    # `: "${VAR:=...}"` and honours the caller's shell, so overwriting them here
    # would make the same variable behave differently depending on which entry
    # point the operator used. Recipe values are applied first so they win over
    # the profile, and an exported value wins over both.
    runtime_env = inference.get("env", {})
    if isinstance(runtime_env, dict):
        for key, value in runtime_env.items():
            if value is not None:
                env.setdefault(str(key), str(value))
    for key, value in asset_profile.runtime_env:
        env.setdefault(key, value)
    if domain_manifest is not None:
        env["DOMAIN_MANIFEST"] = str(domain_manifest)
    return env, report_path


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _prepare_domain_manifest_prompts(env: dict[str, str]) -> None:
    """Materialize generic manifest prompts before entering the Docker runner."""
    manifest_path = Path(env["DOMAIN_MANIFEST"])
    data_dir = Path(env["DATA_DIR"])
    train_path = data_dir / "onpolicy_train.jsonl"
    eval_path = data_dir / "onpolicy_eval.jsonl"
    if train_path.is_file() and eval_path.is_file():
        # Reported counts have to describe the cached rows that will actually be
        # trained on. Returning early without them leaves the profile defaults
        # in place, and every downstream log then names a prompt count the run
        # never uses.
        train_rows = _count_lines(train_path)
        eval_rows = _count_lines(eval_path)
        env["NUM_PROMPTS"] = str(train_rows + eval_rows)
        env["EVAL_N"] = str(eval_rows)
        env["DOMAIN_MANIFEST_ACTIVE"] = "1"
        return

    manifest = load_domain_manifest(manifest_path)
    records = build_domain_records(manifest)
    if len(records) < 2:
        raise ValueError(f"domain manifest must produce at least two unique prompts: {manifest_path}")

    write_jsonl(data_dir / "prompts.jsonl", records)
    summary = {
        "manifest": str(manifest_path),
        "manifest_sha256": _domain_manifest_fingerprint(manifest_path),
        **provenance_license_summary(records),
    }
    (data_dir / "prompt_provenance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    eval_ratio = manifest.splits.get("eval", manifest.splits.get("validation", 0.0))
    eval_size = max(1, min(len(records) - 1, round(len(records) * eval_ratio)))
    env["NUM_PROMPTS"] = str(len(records))
    env["EVAL_N"] = str(eval_size)
    env["DOMAIN_MANIFEST_ACTIVE"] = "1"


def _load_and_augment_report(
    report_path: Path,
    cfg: dict[str, Any],
    cache_dir: Path,
    output_dir: Path,
    data_dir: Path,
    base_model: str,
) -> dict[str, Any]:
    """Load the runner report and attach reproducibility metadata."""
    if not report_path.is_file():
        raise RuntimeError(f"TorchSpec pipeline completed without the expected report: {report_path}")
    with report_path.open(encoding="utf-8") as f:
        report: dict[str, Any] = json.load(f)
    report["report_path"] = str(report_path)
    _augment_report(report, cfg, cache_dir, output_dir, data_dir, base_model)
    return report


def run_torchspec_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run data generation, 8-GPU training, export, and speedup validation."""
    settings = resolve_torchspec_settings(cfg)
    if settings.profile != "full":
        logger.warning(
            "profile=%s is a plumbing check and is not expected to meet the production speedup gate",
            settings.profile,
        )

    base_model = str(cfg.get("model", {}).get("target_model_path", "Qwen/Qwen3-8B"))
    env, report_path = build_runner_environment(cfg)
    cache_dir = Path(env["CACHE_DIR"])
    if not _environment_ready(cache_dir, settings.image, base_model):
        raise RuntimeError(
            f"EAGLE-3 environment is not prepared for {base_model!r}. Run this once first:\n  {setup_hint(base_model)}"
        )

    assets = find_runner_assets()
    output_dir = Path(env["EX_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(env["DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    if env.get("DOMAIN_MANIFEST"):
        _prepare_domain_manifest_prompts(env)

    logger.info(
        "starting TorchSpec pipeline: runner_profile=%s target=%s output=%s prompts=%s epochs=%s",
        env["RUNNER_PROFILE"],
        base_model,
        output_dir,
        env["NUM_PROMPTS"],
        env["EPOCHS"],
    )
    # Dispatch straight to the resolved profile's wrapper. A shell dispatcher
    # above the profiles would have to re-implement the name aliasing that
    # `_runner_profile_name` already performed, and the two would drift.
    subprocess.run(
        ["bash", str(assets / "eagle3" / env["RUNNER_PROFILE"] / "run.sh"), "--base_model", base_model],
        check=True,
        env=env,
    )
    report = _load_and_augment_report(
        report_path,
        cfg,
        cache_dir,
        output_dir,
        Path(env["DATA_DIR"]),
        base_model,
    )
    if report.get("status") == "failed":
        served_al = report.get("served_al")
        served_al_text = "unavailable" if served_al is None else f"{float(served_al):.3f}"
        logger.warning(
            "EAGLE-3 draft completed but missed the advisory quality gate: "
            "speedup=%.3fx (required %.2fx), served_AL=%s (required %.2f); report=%s",
            float(report["median_speedup"]),
            float(report["min_speedup"]),
            served_al_text,
            float(report["min_served_al"]),
            report_path,
        )
    else:
        logger.info(
            "validated EAGLE-3 draft: speedup=%.3fx served_AL=%.3f draft=%s",
            float(report["median_speedup"]),
            float(report["served_al"]),
            report["draft_path"],
        )
    return report
