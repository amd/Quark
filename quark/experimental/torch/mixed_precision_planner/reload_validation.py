#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import argparse
from pathlib import Path

from quark.torch import import_model_from_safetensors

from .data import TokenDataset
from .export import ReloadMeasurement
from .ppl import evaluate_ppl
from .qconfig_builder import qconfig_semantic_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reload and validate an exported mixed-precision model.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = import_model_from_safetensors(
        model=None,
        model_dir=args.model_dir,
        multi_device=False,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
        device=args.device,
        multi_gpu=False,
    )
    tokens = TokenDataset.load(args.tokens)
    reloaded_hf_ppl = evaluate_ppl(model, tokens, args.device)
    result = ReloadMeasurement(
        reloaded_hf_ppl=reloaded_hf_ppl,
        ppl_token_hash=tokens.token_hash,
        qconfig_hash=qconfig_semantic_hash(model.quant_config),
        meta_parameters=sum(parameter.device.type == "meta" for parameter in model.parameters()),
        meta_buffers=sum(buffer.device.type == "meta" for buffer in model.buffers()),
    )
    result.save(args.output)


if __name__ == "__main__":
    main()
