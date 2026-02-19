# Copyright (c) 2025 Advanced Micro Devices, Inc.

import argparse
import enum
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import ryzenai_onnx_utils
import ryzenai_onnx_utils.lora as lora_processor
import ryzenai_onnx_utils.partitioner as partitioner
import ryzenai_onnx_utils.postprocess as postprocesser
import ryzenai_onnx_utils.preprocess as preprocesser
import ryzenai_onnx_utils.strategy_builder as strategy_builder
import ryzenai_onnx_utils.strategy_builder.llm as llm_strategy_builder

_logger = logging.getLogger(__name__)


CPU_EAGER = "cpu_eager"
GPU_EAGER = "gpu_eager"
NPU_EAGER = "npu_eager"
NPU_FUSION = "npu_fusion"


class Phase(enum.StrEnum):
    # strenum because the value is used in logging messages
    PREFILL = "prefill"
    TOKEN = "token"


# TODO(varunsh): detect when JIT should be auto-disabled


class OptimizeArgs:
    def __init__(self, args: argparse.Namespace):
        self.input_model: Path = args.input_model
        self.output_model: Path = args.output_model
        self.force: bool = args.force
        self.dry_run: bool = args.dry_run
        self.clean: bool = not args.no_clean
        self.include: list[str] = args.include
        self.include_only: list[str] = args.include_only
        self.exclude: list[str] = args.exclude

        self.external_data_extension = args.external_data_extension


class LlmArgs(OptimizeArgs):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.prefill: str = args.prefill
        self.token: str = args.token
        self.priority: strategy_builder.Priority = args.priority
        self.genai_config: llm_strategy_builder.GenaiMode = args.genai_config
        self.genai_config_model: str = args.genai_config_model
        self.jit: bool | None = args.jit
        self.lora: bool | None = args.lora
        self.max_seq_len: int | None = args.max_seq_len
        self.mladf_version: strategy_builder.MladfVersion = args.mladf_version
        self.model_type: llm_strategy_builder.LlmModel = args.model_type
        self.prune_logits: bool | None = args.prune_logits
        self.shared_weights: bool | None = args.shared_weights
        self.dynamic_attention_mask: bool | None = args.dynamic_attention_mask
        self.ctrl_pkt: bool | None = args.ctrl_pkt
        self.prefill_fusion_eager_fallback: bool | None = args.prefill_fusion_eager_fallback
        self.lora_adapter: Path = args.lora_adapter
        self.lora_scaling_factor: float = args.lora_scaling_factor
        self.lora_name: str = args.lora_name

        self._validate()

    def uses_gpu(self) -> bool:
        return self.prefill == GPU_EAGER or self.token == GPU_EAGER

    def uses_npu(self) -> bool:
        return self.prefill in {NPU_EAGER, NPU_FUSION} or self.token in {NPU_EAGER, NPU_FUSION}

    def _validate(self):
        if self.prefill_fusion_eager_fallback is None:
            # disable for LoRA by default but enable otherwise
            self.prefill_fusion_eager_fallback = not self.lora

        if self.lora:
            if self.lora_adapter is None and self.lora_scaling_factor is None and self.lora_name is None:
                _logger.warning(
                    "Skipping LoRA adapter generation since no paths or names were provided via lora options"
                )
            else:
                if self.lora_adapter is None or self.lora_name is None:
                    _logger.error(
                        "Both --lora-adapter and --lora-name must be provided to enable LoRA adapter generation"
                    )
                    sys.exit(1)
                if len(self.lora_adapter) != len(self.lora_name):
                    _logger.error("The number of LoRA adapter paths and names must be the same")
                    sys.exit(1)
                if self.lora_scaling_factor is not None and len(self.lora_adapter) != len(self.lora_scaling_factor):
                    _logger.error("The number of LoRA adapter paths and scaling factors must be the same")
                    sys.exit(1)

                if self.lora_scaling_factor is None:
                    self.lora_scaling_factor = [1.0] * len(self.lora_adapter)

    def __str__(self):
        return (
            f"LlmArgs(input_model={self.input_model}, output_model={self.output_model}, "
            f"prefill={self.prefill}, token={self.token}, priority={self.priority}, "
            f"jit={self.jit}, lora={self.lora})"
        )


# TODO(varunsh): this should be updated to inherit parser arguments from the others somehow
def configure_parser(subparser: argparse._SubParsersAction):
    optimize_parser = subparser.add_parser("optimize")

    optimize_parser.add_argument(
        "--input-model",
        help="Input model path",
        type=Path,
        required=True,
    )

    optimize_parser.add_argument(
        "--output-model",
        help="Output model path",
        type=Path,
        required=True,
    )

    optimize_parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all models, including temporary models, even if they already exist",
    )

    optimize_parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep temporary models / files after execution",
    )

    optimize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the optimization process without making any changes and print the commands that would be executed",
    )

    optimize_subparsers = optimize_parser.add_subparsers(dest="optimize_subparser")
    llm_parser = optimize_subparsers.add_parser("llm")

    llm_parser.add_argument(
        "--prefill",
        help="Prefill phase execution mode",
        type=str,
        choices=[CPU_EAGER, NPU_EAGER, NPU_FUSION],
        required=True,
    )

    llm_parser.add_argument(
        "--token",
        help="Token phase execution mode",
        type=str,
        choices=[CPU_EAGER, GPU_EAGER, NPU_EAGER, NPU_FUSION],
        required=True,
    )

    llm_parser.add_argument(
        "--model-type",
        help="Type of model to optimize",
        type=llm_strategy_builder.LlmModel,
        default=llm_strategy_builder.LlmModel.AUTO,
        choices=[e.value for e in llm_strategy_builder.LlmModel],
    )

    llm_parser.add_argument(
        "--priority",
        choices=[e.value for e in strategy_builder.Priority],
        default=strategy_builder.Priority.DEFAULT,
        type=strategy_builder.Priority,
        help="Prioritize performance or memory optimizations, where applicable",
    )

    llm_parser.add_argument(
        "--max-seq-len",
        type=int,
        help="Set the maximum sequence length for models using NPU fusion",
    )

    llm_parser.add_argument(
        "--mladf-version",
        type=strategy_builder.MladfVersion,
        choices=[e.value for e in strategy_builder.MladfVersion],
        default=strategy_builder.MladfVersion.AIE2_V2,
        help="Set the mladf_version for the model.",
    )

    llm_parser.add_argument(
        "--include",
        action="extend",
        nargs="+",
        choices=llm_strategy_builder.SUPPORTED_LLM_OPERATORS,
        default=[],
        help="Force include specific operators in the optimization process",
    )

    llm_parser.add_argument(
        "--include-only",
        action="extend",
        nargs="+",
        choices=llm_strategy_builder.SUPPORTED_LLM_OPERATORS,
        default=[],
        help="Include only specific operators in the optimization process",
    )

    llm_parser.add_argument(
        "--exclude",
        action="extend",
        nargs="+",
        choices=llm_strategy_builder.SUPPORTED_LLM_OPERATORS,
        default=[],
        help="Exclude specific operators from the optimization process",
    )

    llm_parser.add_argument(
        "--genai-config",
        help="GenAI configuration mode: either a standalone custom ops library or RyzenAI EP",
        choices=[e.value for e in llm_strategy_builder.GenaiMode],
        type=llm_strategy_builder.GenaiMode,
        default=llm_strategy_builder.GenaiMode.EP,
    )

    llm_parser.add_argument(
        "--genai-config-model",
        help="If there are multiple models in the GenAI configuration, specify which section to update",
        default="decoder",
    )

    llm_parser.add_argument(
        "--lora-adapter",
        help="Path to the LoRA adapter",
        action="extend",
        nargs="+",
        type=Path,
    )

    llm_parser.add_argument(
        "--lora-scaling-factor",
        help="Scaling factor for LoRA weights",
        action="extend",
        nargs="+",
        type=float,
    )

    llm_parser.add_argument(
        "--lora-name",
        help="Name of the target LoRA adapter",
        action="extend",
        nargs="+",
        type=str,
    )

    llm_parser.add_argument(
        "--jit",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable JIT compilation for NPU and GPU, where applicable",
    )

    llm_parser.add_argument(
        "--lora",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable LoRA, where applicable",
    )

    llm_parser.add_argument(
        "--prune-logits",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable pruning of logits, where applicable",
    )

    llm_parser.add_argument(
        "--shared-weights",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable shared weights, where applicable",
    )

    llm_parser.add_argument(
        "--dynamic-attention-mask",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable dynamic attention mask, where applicable",
    )

    llm_parser.add_argument(
        "--ctrl-pkt",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable control packets, where applicable",
    )

    llm_parser.add_argument(
        "--prefill-fusion-eager-fallback",
        action=argparse.BooleanOptionalAction,
        help="For prefill npu_fusion models, enable/disable fallback to eager execution for unsupported sequence lengths",
    )

    return optimize_parser


def _check_output(args: LlmArgs, path: Path, prompt: str, force_default: str) -> bool:
    if not path.exists():
        return False

    # no issues if the directory is empty
    if path.is_dir() and not any(path.iterdir()):
        return False

    answer = ""
    if args.force:
        answer = force_default.lower()

    prompt = f"{prompt}. Enter Y to delete, C to continue to reuse existing, N to exit: "
    while answer not in ["y", "n", "c"]:
        answer = input(prompt).lower()

    if answer == "n":
        sys.exit(0)
    elif answer == "y":
        if path.suffix == ".onnx":
            _delete_model(args, path)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        return False
    _logger.info("Skipping processing for existing file %s", path)
    return True


def check_output_cache(args: LlmArgs, cache_path: Path, force_default: str = "y") -> bool:
    prompt = f"Output cache {cache_path} already exists"
    return _check_output(args, cache_path, prompt, force_default)


def check_output_model(args: LlmArgs, model_path: Path, force_default: str = "y") -> None:
    prompt = f"Output model {model_path} already exists"
    return _check_output(args, model_path, prompt, force_default)


DEFAULT_DD_CACHE_PREFIX = "cache"


def partition(args: LlmArgs, strategy: str):
    if check_output_model(args, args.output_model):
        return

    output_model_name = Path(args.output_model).stem
    output_model_dir = Path(args.output_model).parent

    cmd = [
        "partition",
        args.input_model.as_posix(),
        output_model_dir.as_posix(),
        strategy,
        "--model-name",
        output_model_name,
        "--save-as-external",
    ]
    if args.force:
        cmd.append("--force")
    if output_model_name.startswith("tmp_"):
        cmd.extend(["--dd-files-path", f"{DEFAULT_DD_CACHE_PREFIX}_{output_model_name}"])
    if args.dry_run:
        print("onnx_utils " + " ".join(cmd))
        return

    parser = partitioner.get_parser()
    new_args = parser.parse_args(cmd)

    _logger.debug("Calling partitioner with args: %s", new_args)
    partitioner.partition_main(new_args)


def preprocess(cmd: list[str], dry_run: bool):
    parser = partitioner.get_parser()

    if dry_run:
        print("onnx_utils " + " ".join(cmd))
        return

    new_args = parser.parse_args(cmd)
    _logger.debug("Calling preprocess with args: %s", new_args)
    preprocesser.main(new_args)


def lora_process(args: LlmArgs, header_path: Path, mode: str):
    parser = partitioner.get_parser()
    for lora_adapter, lora_name, lora_scaling_factor in zip(
        args.lora_adapter, args.lora_name, args.lora_scaling_factor, strict=False
    ):
        cmd = [
            "lora-process",
            "--lora-file",
            lora_adapter.as_posix(),
            "--lora-name",
            lora_name,
            "--base-model-header",
            header_path.as_posix(),
            "--scaling-factor",
            str(lora_scaling_factor),
            "--mode",
            mode,
            "--output-dir",
            args.output_model.parent.as_posix(),
        ]
        if args.dry_run:
            print("onnx_utils " + " ".join(cmd))
            continue

        new_args = parser.parse_args(cmd)
        _logger.debug("Calling lora_process with args: %s", new_args)
        lora_processor.main(new_args)


def generate_lora_bins(args: LlmArgs, prefill_model: Path, token_model: Path):
    if args.lora and args.lora_adapter:
        _logger.info("Generating LoRA adapters for optimized model")
        prefill_external_data = prefill_model.with_suffix(".pb.bin")
        if prefill_external_data.exists():
            lora_process(args, prefill_external_data, "prefill")
        else:
            _logger.warning(
                "Prefill external data file %s does not exist, skipping LoRA generation for prefill",
                prefill_external_data,
            )
        token_external_data = token_model.with_suffix(".pb.bin")
        if token_external_data.exists():
            lora_process(args, token_external_data, "token")
        else:
            _logger.warning(
                "Token external data file %s does not exist, skipping LoRA generation for token", token_external_data
            )


def llm_preprocess_optimize(args: LlmArgs, output_model: Path):
    if check_output_model(args, output_model, "c"):
        return
    _logger.info("Optimizing model %s with ORT first", args.input_model)
    cmd = [
        "preprocess",
        args.input_model.as_posix(),
        output_model.as_posix(),
        "--save-as-external",
        "--fixed-shapes",
        "batch_size=1",
        "graphs=1",
        "sequence_length=''",
        "total_sequence_length=''",
        "past_sequence_length=''",
        "--optimize",
        "llm",
    ]

    preprocess(cmd, args.dry_run)


def llm_preprocess_shapes(
    input_model: Path, output_model: Path, sequence_length: int | None, max_seq_len: int, dry_run: bool
):
    sequence_length = "" if sequence_length is None else str(sequence_length)
    cmd = [
        "preprocess",
        input_model.as_posix(),
        output_model.as_posix(),
        "--save-as-external",
        "--fixed-shapes",
        "batch_size=1",
        "graphs=1",
        f"sequence_length={sequence_length}",
        f"total_sequence_length={max_seq_len}",
        f"past_sequence_length={max_seq_len}",
    ]
    preprocess(cmd, dry_run)


def llm_postprocess(
    reference_model: Path,
    prefill_model: Path,
    token_model: Path,
    output_model: Path,
    attributes: dict[str, Any],
    dry_run: bool,
):
    cmd = [
        "postprocess",
        "combine_llm",
        "--input-path",
        reference_model.as_posix(),
        prefill_model.as_posix(),
        token_model.as_posix(),
        "--output-path",
        output_model.as_posix(),
    ]

    if attributes:
        cmd.extend(["--attribute"])
        for key, value in attributes.items():
            cmd.extend([f"{key}={value}"])

    if dry_run:
        print("onnx_utils " + " ".join(cmd))
        return

    parser = partitioner.get_parser()
    new_args = parser.parse_args(cmd)

    _logger.debug("Calling postprocess with args: %s", new_args)
    postprocesser.main(new_args)


def llm_fusion_preprocess(args: LlmArgs, phase: Phase):
    if check_output_model(args, args.output_model):
        return
    sequence_length = None if phase == Phase.PREFILL else 1
    llm_preprocess_shapes(args.input_model, args.output_model, sequence_length, args.max_seq_len, args.dry_run)


def llm_gpu_eager(args: LlmArgs) -> llm_strategy_builder.LlmModelBase:
    # currently, gpu eager is same as hybrid until the custom ops can better
    # support not having NPU weights
    return llm_hybrid(args)


def llm_hybrid(args: LlmArgs) -> llm_strategy_builder.LlmModelBase:
    output_strategy = args.output_model.parent / f"{args.output_model.stem}_strategy.yaml"
    builder = llm_strategy_builder.get_builder(args, llm_strategy_builder.LlmMode.HYBRID)
    builder.save(output_strategy)
    partition(args, output_strategy.as_posix())
    return builder


def llm_npu_eager(args: LlmArgs) -> llm_strategy_builder.LlmModelBase:
    output_strategy = args.output_model.parent / f"{args.output_model.stem}_strategy.yaml"
    builder = llm_strategy_builder.get_builder(args, llm_strategy_builder.LlmMode.NPU_EAGER)
    builder.save(output_strategy)
    partition(args, output_strategy.as_posix())
    return builder


def _delete_model(args: LlmArgs, model: Path):
    if args.dry_run:
        print(f"Delete model {model} and associated files, if they exist")
        return
    ryzenai_onnx_utils.matcher.delete_model(model, args.external_data_extension)
    shutil.rmtree(model.parent / f"{DEFAULT_DD_CACHE_PREFIX}_{model.stem}", ignore_errors=True)


def _copytree(src_path: Path, dst_path: Path, dry_run: bool):
    src_path = src_path.as_posix()
    dst_path = dst_path.as_posix()
    if dry_run:
        print(f"Copy from {src_path} to {dst_path}")
        return
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)


def _copy_cache(src_path: Path, dst_path: Path, dry_run: bool):
    _copytree(src_path, dst_path, dry_run)
    if not dry_run:
        # the metajson files have the cache directory in the file names. If
        # we're copying to a different directory, we need to update those names
        meta_files = dst_path.glob("*.json")
        for meta_file in meta_files:
            with meta_file.open("r") as f:
                content = f.read()
            content = content.replace(src_path.name, dst_path.name)
            content = json.loads(content)
            with meta_file.open("w") as f:
                json.dump(content, f, indent=2)


def _rename_external_data_file(args: LlmArgs, src_model: Path, dst_model: Path):
    if args.dry_run:
        print(f"Rename external data file {src_model.stem} to {dst_model.stem}")
        return
    ryzenai_onnx_utils.matcher.rename_external_data_file(src_model, dst_model)


def _llm_npu_fusion(args: LlmArgs, phase: Phase, strategy: str):
    final_model = args.output_model
    if check_output_model(args, final_model):
        return

    if args.max_seq_len is None:
        # arbitrarily use full fusion as the model type. It doesn't matter to get
        # this value
        builder = llm_strategy_builder.get_builder(args, llm_strategy_builder.LlmMode.FULL_FUSION_PREFILL)
        args.max_seq_len = builder.default_max_seq_len

    label = "dynamic" if phase == Phase.PREFILL else "static"
    _logger.info(f"Preprocessing temporary model with {label} shapes for {phase.value} phase npu fusion")
    output_dir = args.output_model.parent
    prompt_size = "dynamic" if phase == Phase.PREFILL else "1"
    tmp_preprocess_model = output_dir / f"tmp_model_{args.max_seq_len}_prompt_{prompt_size}.onnx"
    args.output_model = tmp_preprocess_model
    llm_fusion_preprocess(args, phase)

    _logger.info("Partitioning temporary model for %s phase npu fusion", phase.value)
    args.input_model = tmp_preprocess_model
    args.output_model = final_model
    partition(args, strategy)

    if args.clean:
        _delete_model(args, tmp_preprocess_model)


def _llm_npu_token_fusion(args: LlmArgs, llm_mode: llm_strategy_builder.LlmMode) -> llm_strategy_builder.LlmModelBase:
    output_strategy = args.output_model.parent / f"{args.output_model.stem}_strategy.yaml"
    builder = llm_strategy_builder.get_builder(args, llm_mode)
    builder.save(output_strategy)
    _llm_npu_fusion(args, Phase.TOKEN, output_strategy.as_posix())
    return builder


def _llm_npu_prefill_fusion(args: LlmArgs, llm_mode: llm_strategy_builder.LlmMode) -> llm_strategy_builder.LlmModelBase:
    output_strategy = args.output_model.parent / f"{args.output_model.stem}_strategy.yaml"
    builder = llm_strategy_builder.get_builder(args, llm_mode)
    builder.save(output_strategy)
    _llm_npu_fusion(args, Phase.PREFILL, output_strategy.as_posix())
    return builder


def llm_token_fusion(args: LlmArgs) -> llm_strategy_builder.GenAiConfig:
    global_output_model = args.output_model
    output_dir = args.output_model.parent

    assert args.input_model.is_absolute()
    assert args.output_model.is_absolute()

    _logger.info("Partitioning model for prefill phase npu eager")
    tmp_prefill_model = output_dir / "tmp_model_npu_eager.onnx"
    args.output_model = tmp_prefill_model
    prefill_builder = llm_npu_eager(args)
    prefill_config = prefill_builder.generate_genai_config(args.output_model.stem)

    tmp_token_model = output_dir / "tmp_model_npu_token_fusion.onnx"
    args.output_model = tmp_token_model
    token_builder = _llm_npu_token_fusion(args, llm_strategy_builder.LlmMode.NPU_TOKEN_FUSION)
    token_config = token_builder.generate_genai_config(args.output_model.stem)

    _logger.info("Preparing NPU eager prefill + NPU fusion token model")
    postprocess_attributes = token_builder.postprocess_attributes()
    postprocess_attributes["suffix"] = "_token"
    llm_postprocess(
        tmp_prefill_model, tmp_prefill_model, tmp_token_model, global_output_model, postprocess_attributes, args.dry_run
    )

    output_cache = (
        f"{DEFAULT_DD_CACHE_PREFIX}_{global_output_model.stem}"
        if global_output_model.stem.startswith("tmp_")
        else DEFAULT_DD_CACHE_PREFIX
    )
    _copy_cache(
        output_dir / f"{DEFAULT_DD_CACHE_PREFIX}_{tmp_token_model.stem}", output_dir / output_cache, args.dry_run
    )

    generate_lora_bins(args, tmp_prefill_model, tmp_token_model)

    _rename_external_data_file(args, tmp_prefill_model, global_output_model)
    if args.clean:
        _delete_model(args, tmp_prefill_model)
        _delete_model(args, tmp_token_model)

    prefill_config.merge(token_config)
    prefill_config.set_filename(global_output_model.name)
    prefill_config.set_option("external_data_file", f"{global_output_model.stem}.pb.bin")
    return prefill_config


def llm_prefill_fusion(args: LlmArgs) -> llm_strategy_builder.GenAiConfig:
    global_output_model = args.output_model
    output_dir = args.output_model.parent

    assert args.input_model.is_absolute()
    assert args.output_model.is_absolute()

    _logger.info("Partitioning model for token phase gpu eager")
    tmp_token_model = output_dir / "tmp_model_gpu_eager.onnx"
    args.output_model = tmp_token_model
    token_builder = llm_gpu_eager(args)
    token_config = token_builder.generate_genai_config(args.output_model.stem)

    tmp_prefill_model = output_dir / "tmp_model_npu_prefill_fusion.onnx"
    args.output_model = tmp_prefill_model
    prefill_builder = _llm_npu_prefill_fusion(args, llm_strategy_builder.LlmMode.NPU_PREFILL_FUSION)
    prefill_config = prefill_builder.generate_genai_config(args.output_model.stem)

    _logger.info("Preparing NPU fusion prefill + GPU eager token model")
    postprocess_attributes = prefill_builder.postprocess_attributes()
    postprocess_attributes["suffix"] = "_prefill"
    llm_postprocess(
        tmp_token_model, tmp_prefill_model, tmp_token_model, global_output_model, postprocess_attributes, args.dry_run
    )

    generate_lora_bins(args, tmp_prefill_model, tmp_token_model)

    _rename_external_data_file(args, tmp_token_model, global_output_model)
    _copy_cache(
        output_dir / f"{DEFAULT_DD_CACHE_PREFIX}_{tmp_prefill_model.stem}",
        output_dir / DEFAULT_DD_CACHE_PREFIX,
        args.dry_run,
    )
    if args.clean:
        _delete_model(args, tmp_prefill_model)
        _delete_model(args, tmp_token_model)

    prefill_config.merge(token_config)
    prefill_config.set_filename(global_output_model.name)
    prefill_config.set_option("external_data_file", f"{global_output_model.stem}.pb.bin")
    return prefill_config


def llm_full_fusion(args: LlmArgs) -> llm_strategy_builder.GenAiConfig:
    global_input_model = args.input_model
    global_output_model = args.output_model
    output_dir = args.output_model.parent

    assert args.input_model.is_absolute()
    assert args.output_model.is_absolute()

    _logger.info("Partitioning model for reference inputs")
    tmp_reference_model = output_dir / "tmp_model_full_reference.onnx"
    args.output_model = tmp_reference_model
    llm_npu_eager(args)

    tmp_prefill_model = output_dir / "tmp_model_npu_prefill_fusion.onnx"
    args.input_model = global_input_model
    args.output_model = tmp_prefill_model
    prefill_builder = _llm_npu_prefill_fusion(args, llm_strategy_builder.LlmMode.FULL_FUSION_PREFILL)
    # see note below about external data
    prefill_builder.keep_external_data = False
    prefill_config = prefill_builder.generate_genai_config(args.output_model.stem)

    tmp_token_model = output_dir / "tmp_model_npu_full_token_fusion.onnx"
    args.input_model = global_input_model
    args.output_model = tmp_token_model

    # if prefill fusion cannot support all required sequence lengths, fallback
    # to npu eager for larger lengths
    need_backup, _ = prefill_builder.need_prefill_fusion_backup()
    if need_backup:
        token_config = llm_token_fusion(args)

        _rename_external_data_file(args, tmp_token_model, global_output_model)
        token_config.set_option("external_data_file", f"{global_output_model.stem}.pb.bin")
    else:
        token_builder = _llm_npu_token_fusion(args, llm_strategy_builder.LlmMode.FULL_FUSION_TOKEN)
        token_config = token_builder.generate_genai_config(args.output_model.stem)

    # for now LoRA has no fallback to eager option so these two models are fine
    generate_lora_bins(args, tmp_prefill_model, tmp_token_model)

    _copy_cache(
        output_dir / f"{DEFAULT_DD_CACHE_PREFIX}_{tmp_prefill_model.stem}",
        output_dir / DEFAULT_DD_CACHE_PREFIX,
        args.dry_run,
    )
    _copy_cache(
        output_dir / f"{DEFAULT_DD_CACHE_PREFIX}_{tmp_token_model.stem}",
        output_dir / DEFAULT_DD_CACHE_PREFIX,
        args.dry_run,
    )

    # this relies on the assumption that the external data for both prefill and
    # token models are the same. Arbitrarily only keep the token model's
    if token_builder.keep_external_data:
        _rename_external_data_file(args, tmp_token_model, global_output_model)

    _logger.info("Preparing NPU fusion prefill + NPU fusion token model")
    postprocess_attributes = prefill_builder.postprocess_attributes()
    postprocess_attributes["suffix"] = "_full"
    llm_postprocess(
        tmp_reference_model,
        tmp_prefill_model,
        tmp_token_model,
        global_output_model,
        postprocess_attributes,
        args.dry_run,
    )

    if args.clean:
        _delete_model(args, tmp_reference_model)
        _delete_model(args, tmp_prefill_model)
        _delete_model(args, tmp_token_model)

    # TODO: the external data in the eager mode is not correct

    prefill_config.merge(token_config)
    prefill_config.set_filename(global_output_model.name)
    return prefill_config


def llm(args: LlmArgs):
    global_output_model = args.output_model
    if global_output_model.parent == Path.cwd():
        _logger.error("Output model cannot be in the current working directory")
        sys.exit(-1)

    if check_output_model(args, global_output_model):
        return

    if check_output_cache(args, global_output_model.parent / DEFAULT_DD_CACHE_PREFIX):
        return

    global_input_model = args.input_model

    optimized_model = global_output_model.parent / f"optimized_{global_input_model.name}"
    llm_preprocess_optimize(args, optimized_model)
    args.input_model = optimized_model

    builder: llm_strategy_builder.LlmModelBase | None = None
    genai_config: llm_strategy_builder.GenAiConfig | None = None

    if args.prefill == CPU_EAGER:
        _logger.error("CPU eager prefill is not supported yet")
    elif args.prefill == NPU_EAGER:
        if args.token == CPU_EAGER:
            _logger.error("NPU eager prefill + CPU eager token is not supported yet")
        elif args.token == GPU_EAGER:
            _logger.info("Preparing NPU eager prefill + GPU eager token (hybrid) model")
            builder = llm_hybrid(args)
            generate_lora_bins(args, global_output_model, global_output_model)
        elif args.token == NPU_EAGER:
            _logger.info("Preparing NPU eager prefill + NPU eager token (NPU-only eager) model")
            builder = llm_npu_eager(args)
            generate_lora_bins(args, global_output_model, global_output_model)
        elif args.token == NPU_FUSION:
            _logger.info("Preparing NPU eager prefill + NPU fusion token (token fusion) model")
            genai_config = llm_token_fusion(args)
        else:
            raise ValueError(f"Unexpected token phase {args.token} for NPU eager prefill")
    elif args.prefill == NPU_FUSION:
        if args.token == CPU_EAGER:
            _logger.error("NPU fusion prefill + CPU eager token is not supported yet")
        elif args.token == GPU_EAGER:
            genai_config = llm_prefill_fusion(args)
        elif args.token == NPU_EAGER:
            _logger.error("NPU fusion prefill + NPU eager token is not supported yet")
        elif args.token == NPU_FUSION:
            genai_config = llm_full_fusion(args)
        else:
            raise ValueError(f"Unexpected token phase {args.token} for NPU fusion prefill")
    else:
        raise ValueError(f"Unexpected prefill phase {args.prefill}")

    if builder is not None:
        genai_config = builder.generate_genai_config(global_output_model.stem)

    if genai_config is not None:
        genai_config.save(global_input_model.parent, global_output_model, args.dry_run)

        _logger.info("Optimized LLM model saved to %s", global_output_model)


def main(args):
    args.input_model = Path(args.input_model).resolve()
    args.output_model = Path(args.output_model).resolve()
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    if args.optimize_subparser == "llm":
        llm_args = LlmArgs(args)
        llm(llm_args)
    else:
        raise ValueError(f"Unexpected optimize subparser {args.optimize_subparser} passed")
