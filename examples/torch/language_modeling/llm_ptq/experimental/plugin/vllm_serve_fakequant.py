#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Serve Quark fakequant models with vLLM."""

import os
import sys

import uvloop
import vllm
from common import find_repo_root
from packaging import version
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.platforms import current_platform

vllm_version = version.parse(vllm.__version__)
if vllm_version <= version.parse("0.11.0"):
    from vllm.executor.ray_distributed_executor import RayDistributedExecutor  # noqa: E402
    from vllm.utils import FlexibleArgumentParser  # noqa: E402
else:
    from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402
    from vllm.v1.executor.ray_executor import RayDistributedExecutor  # noqa: E402

additional_env_vars = {
    "QUANT_DATASET",
    "QUANT_CALIB_SIZE",
    "QUANT_CALIB_SEQLEN",
    "QUANT_CFG",
}

if hasattr(RayDistributedExecutor, "ADDITIONAL_ENV_VARS"):
    # vLLM <= 0.16 copies custom env vars via the executor class attribute.
    RayDistributedExecutor.ADDITIONAL_ENV_VARS.update(additional_env_vars)
else:
    # vLLM >= 0.19 copies them from current_platform.additional_env_vars.
    merged_env_vars = set(getattr(current_platform, "additional_env_vars", []))
    merged_env_vars.update(additional_env_vars)
    current_platform.additional_env_vars = sorted(merged_env_vars)


def main() -> None:
    # Create parser that handles both quant and serve arguments
    parser = FlexibleArgumentParser(description="vLLM model server with Quark quantization support")
    parser.add_argument("model", type=str, help="The path or name of the model to serve")
    parser.add_argument(
        "--quant-cfg",
        type=str,
        default=None,
        help="Quark quantization config, which can be a JSON file path, scheme name, or inline JSON payload.",
    )
    parser.add_argument(
        "--quant-dataset",
        type=str,
        default="cnn_dailymail",
        help="Calibration dataset name",
    )
    parser.add_argument(
        "--quant-calib-size",
        type=str,
        default="512",
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--quant-calib-seqlen",
        type=str,
        default="",
        help="Calibration sequence length for get_calib_dataloader (e.g. 2048). Empty = Quark default (512 for pileval).",
    )
    parser = make_arg_parser(parser)

    # Ensure workers can import our custom worker module when using spawn
    repo_root = find_repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + f"{repo_root}"

    # Default to our Quark FakeQuantWorker if user doesn't specify a worker class
    parser.set_defaults(worker_cls="quark.experimental.plugin.fakequant_worker.QuarkFakeQuantWorker")

    # Parse arguments
    args = parser.parse_args()

    # Set quantization env vars from CLI args so distributed workers receive them.
    if args.quant_cfg is not None:
        os.environ["QUANT_CFG"] = args.quant_cfg
    else:
        os.environ.pop("QUANT_CFG", None)
    os.environ["QUANT_DATASET"] = args.quant_dataset
    os.environ["QUANT_CALIB_SIZE"] = str(args.quant_calib_size)
    if str(args.quant_calib_seqlen).strip():
        os.environ["QUANT_CALIB_SEQLEN"] = str(int(args.quant_calib_seqlen))
    else:
        os.environ.pop("QUANT_CALIB_SEQLEN", None)

    # Quark fake-quant uses observers that are incompatible with torch.compile + CUDA graph.
    # Enforce eager to avoid dynamo graph breaks during warmup.
    if args.quant_cfg is not None:
        args.enforce_eager = True

    # Run the server
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
