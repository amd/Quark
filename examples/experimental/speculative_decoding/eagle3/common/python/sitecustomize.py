#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Register public target chat delimiters before TorchSpec workers start."""

from __future__ import annotations

import os


def _register_minimax_m3() -> None:
    from torchspec.data.template import TEMPLATE_REGISTRY, ChatTemplate

    if "minimax-m3" in TEMPLATE_REGISTRY.get_all_template_names():
        return
    # These delimiters are published in the target's chat_template.jinja.
    TEMPLATE_REGISTRY.register(
        name="minimax-m3",
        template=ChatTemplate(
            assistant_header="]~b]ai\n",
            user_header="]~b]user\n",
            # Let the public tokenizer's chat_template.jinja inject its own
            # versioned default system and thinking instructions. Supplying a
            # generic system message here would create train/serve template drift.
            system_prompt=None,
            end_of_turn_token="[e~[",
            parser_type="general",
            image_placeholder="]<]image[>[",
        ),
    )


_REGISTRARS = {"minimax-m3": _register_minimax_m3}


def _register_requested_template() -> None:
    """Register the template this run needs, and fail loudly if it cannot.

    Nothing imports this module by name: it loads only because `train.sh` puts
    its directory on PYTHONPATH and `site.py` picks it up. Move or rename that
    directory and `site.py` skips it silently, the template is never
    registered, and training proceeds against delimiters that do not match what
    the target is served with -- visible only as a lower acceptance length with
    no error anywhere. Raising here turns that into a start-up failure.
    """
    requested = os.environ.get("TORCHSPEC_CHAT_TEMPLATE_PROFILE")
    register = _REGISTRARS.get(requested or "")
    if register is None:
        return
    register()
    from torchspec.data.template import TEMPLATE_REGISTRY

    if requested not in TEMPLATE_REGISTRY.get_all_template_names():
        raise RuntimeError(
            f"chat template {requested!r} is not registered after sitecustomize ran; "
            "training would use delimiters that do not match how the target is served"
        )


_register_requested_template()
