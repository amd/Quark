"""vLLM plugin for Quark online quant via ``additional_config``.

It patches ``EngineArgs.create_model_config`` to set ``quantization`` and
``hf_overrides`` before ``ModelConfig`` is created.
Disable with ``QUARK_DISABLE_VLLM_PLUGIN=1``.
"""

import logging
import os
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from .hf_quantization_configs import online_quant_config_to_quark, online_quant_overrides

logger = logging.getLogger("quark.online_quantization")

_PATCHED_FLAG = "_quark_online_quant_patched"
_ADDITIONAL_CONFIG_KEY = "online_quant_config"


def _disabled() -> bool:
    """Return whether the Quark vLLM plugin is disabled.

    Returns:
        True if ``QUARK_DISABLE_VLLM_PLUGIN`` is set to a truthy value,
        False otherwise.
    """
    return os.environ.get("QUARK_DISABLE_VLLM_PLUGIN", "0") not in ("0", "", "false", "False")


def _is_truthy_env(var_name: str, default: str = "0") -> bool:
    """Return whether an environment variable is set to a truthy value.

    Args:
        var_name: Name of the environment variable to check.
        default: Value to use when the environment variable is unset.

    Returns:
        True if the resolved value is not ``"0"``, ``""``, ``"false"``, or
        ``"False"``.
    """
    return os.environ.get(var_name, default) not in ("0", "", "false", "False")


def _atom_plugin_active() -> bool:
    """Return whether ATOM vLLM plugin is enabled in this process.

    Purpose: when ATOM has already taken over the vLLM backend, Quark's
    additional-config bridge should stay out of the way to avoid conflicting
    rewrites of quantization behavior.
    """
    if _is_truthy_env("ATOM_DISABLE_VLLM_PLUGIN"):
        return False

    allowed_plugins = os.environ.get("VLLM_PLUGINS")
    if allowed_plugins is not None:
        selected = {name.strip() for name in allowed_plugins.split(",")}
        if "atom" not in selected and "atom_model_registry" not in selected:
            return False

    # Prefer runtime truth from vLLM if available: ATOM platform actually
    # activated means ATOM has taken over backend execution.
    try:
        from vllm import platforms as vllm_platforms

        current_platform = getattr(vllm_platforms, "current_platform", None)
        if current_platform is not None:
            platform_mod = type(current_platform).__module__
            if platform_mod.startswith("atom.plugin.vllm"):
                return True
    except Exception:
        pass

    try:
        platform_plugins = {ep.name for ep in entry_points(group="vllm.platform_plugins")}
        general_plugins = {ep.name for ep in entry_points(group="vllm.general_plugins")}
    except Exception:
        return False

    return "atom" in platform_plugins or "atom_model_registry" in general_plugins


def _has_explicit_hf_overrides(engine_args: Any) -> bool:
    """True if the user already passed a non-empty ``--hf-overrides``.

    The field defaults to ``{}`` (a callable is also possible), so anything
    truthy means the user set it and we must not clobber it.
    """
    return bool(getattr(engine_args, "hf_overrides", None))


def _online_quant_config_from_additional_config(
    additional_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Extract online quantization config from vLLM ``additional_config``.

    Args:
        additional_config: vLLM additional configuration dictionary, or None.

    Returns:
        The online quantization configuration dictionary if present, otherwise
        None.
    """
    if not additional_config:
        return None
    return additional_config.get(_ADDITIONAL_CONFIG_KEY)


def _build_hf_overrides_from_online_quant_config(online_quant_cfg: dict[str, Any]) -> Callable[[Any], Any]:
    """Build a HuggingFace overrides callable from online quantization config.

    Args:
        online_quant_cfg: ATOM-style online quantization configuration.

    Returns:
        A callable that applies Quark online quantization overrides.
    """
    quark_online = online_quant_config_to_quark(online_quant_cfg)
    return online_quant_overrides(quark_online)


def _should_bridge_additional_config(engine_args: Any, online_quant_cfg: dict[str, Any] | None) -> bool:
    """Return whether ``online_quant_config`` should be
    bridged into Quark's ``hf_overrides`` path.

    Rules:
    - no online config -> do nothing;
    - online config + no explicit quantization -> auto-set quark_online;
    - online config + explicit quark_online -> bridge;
    - online config + explicit non-Quark quantization -> ignore bridge.
    """
    if online_quant_cfg is None:
        return False
    if _atom_plugin_active():
        logger.info("Quark online quant: detected active ATOM vLLM plugin; skipping Quark additional_config bridge.")
        return False

    current_quant = getattr(engine_args, "quantization", None)
    if not current_quant:
        engine_args.quantization = "quark_online"
        logger.info(
            "Quark online quant: auto-set quantization=quark_online from additional_config.online_quant_config."
        )
    elif current_quant != "quark_online":
        logger.info(
            "Quark online quant: additional_config.online_quant_config was "
            "provided, but explicit quantization=%s is not quark_online; "
            "skipping Quark online bridge.",
            current_quant,
        )
        return False
    return True


def _maybe_inject_hf_overrides(engine_args: Any) -> None:
    """Handle ``additional_config.online_quant_config`` before
    ``ModelConfig`` construction:
    1) auto-select ``quantization=quark_online`` when absent;
    2) auto-build ``hf_overrides`` when absent.
    """
    additional_config = getattr(engine_args, "additional_config", None)
    online_quant_cfg = _online_quant_config_from_additional_config(additional_config)
    if not _should_bridge_additional_config(engine_args, online_quant_cfg):
        return
    if _has_explicit_hf_overrides(engine_args):
        return

    engine_args.hf_overrides = _build_hf_overrides_from_online_quant_config(online_quant_cfg)
    logger.info("Quark online quant: built hf_overrides from additional_config.online_quant_config.")


def register() -> None:
    """Entry point target. Idempotent across the multiple plugin loads that
    vLLM performs in spawn-based multi-process engines."""
    if _disabled():
        logger.info("Quark vLLM plugin disabled via QUARK_DISABLE_VLLM_PLUGIN.")
        return

    from vllm.engine.arg_utils import EngineArgs

    orig_create_model_config = EngineArgs.create_model_config
    if getattr(orig_create_model_config, _PATCHED_FLAG, False):
        return

    def patched_create_model_config(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            _maybe_inject_hf_overrides(self)
        except Exception:
            logger.exception(
                "Quark online quant plugin failed to apply additional_config; falling back to default vLLM behavior."
            )
        return orig_create_model_config(self, *args, **kwargs)

    setattr(patched_create_model_config, _PATCHED_FLAG, True)
    EngineArgs.create_model_config = patched_create_model_config
    logger.info("Quark vLLM online-quant plugin registered.")
