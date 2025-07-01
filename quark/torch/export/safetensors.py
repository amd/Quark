#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path
from typing import List, Optional, Dict, Any, Union, TYPE_CHECKING
import torch
import json
from torch import nn
from quark.shares.utils.log import ScreenLogger
from quark.shares.utils.import_utils import is_transformers_available, is_accelerate_available, is_safetensors_available
from tqdm import tqdm

if TYPE_CHECKING and is_transformers_available():
    from transformers import PreTrainedModel, PreTrainedTokenizer

if is_accelerate_available():
    from accelerate.utils.modeling import find_tied_parameters, named_module_tensors

if is_safetensors_available():
    from safetensors.torch import load_file

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
logger = ScreenLogger(__name__)


def export_hf_model(model: "PreTrainedModel",
                    export_dir: Union[str, Path],
                    tokenizer: Optional["PreTrainedTokenizer"] = None) -> None:
    '''
    This function is used to export models in Hugging Face safetensors format.
    '''

    logger.info("Start exporting huggingface_format quantized model ...")
    # Save model to safetensors.
    model.save_pretrained(export_dir)

    # Optionally, save the tokenizer from the original model.
    if tokenizer is not None:
        tokenizer.save_pretrained(export_dir)

    logger.info(f"hf_format quantized model exported to {export_dir} successfully.")


def import_hf_model(
        model_importer: "ModelImporter",  # type: ignore [name-defined]
        model: nn.Module,
        model_info_dir: str) -> nn.Module:
    '''
    Load the model file, perform preprocessing and post-processing, load weights into the model.
    '''
    if not is_safetensors_available():
        raise ImportError(
            "The function `import_hf_model` requires the package `safetensors` to be installed, but it was not found. Please install `safetensors`."
        )
    checkpoint_weights = _load_weights_from_safetensors(model_info_dir)

    model_config = model_importer.get_model_config()
    model = model_importer._build_model(model, model_config, checkpoint_weights)

    if is_accelerate_available():
        _untie_parameters(model, checkpoint_weights)
    # The module here is qparamlinear, the float module has been removed, the internal weight is already a quantized dtype like fp8 and is assigned to each GPU or meta by device.
    model_state_dict = model.state_dict()

    # In case we are loading the quantized weights into a model that is not on meta device,
    # we re-use the original device the weights were placed on, as `assign=True` is used later.
    # This is helpful e.g. in case the original model was dispatched to multiple
    # devices ahead of time with `accelerate`.
    for name, param in model_state_dict.items():
        if name not in checkpoint_weights:
            raise ValueError(f"The loaded checkpoint misses the key {name} present in the model weights.")

        if param.device.type != "meta":
            checkpoint_weights[name] = checkpoint_weights[name].to(param.device)
    if model_importer.multi_device is True:
        named_modules = dict(model.named_modules(remove_duplicate=False))
        for name, module in tqdm(named_modules.items()):
            # deivce must be meta and can only get the lowest granularity mods.
            if hasattr(module, "_hf_hook") and module._hf_hook.offload is True:
                hook = module._hf_hook
                # all_modules = dict(named_module_tensors(module, include_buffers=False, recurse=True, remove_non_persistent=True))
                # "weight"
                weight_modules = dict(
                    named_module_tensors(module, include_buffers=False, recurse=False, remove_non_persistent=True))

                # If meta, send the value of checkpoint_weight to the hook's "weights_map" and convert the param to meta.
                # unquantized mods like "lm_head", "DeepseekV3RMSNorm" can be done directly like this.
                # Like qparamlinear, weight is handled the same way, but scale, zero should be sent directly to execution_device,
                # "weight_map" doesn't support increasing KV.
                prefix = hook.weights_map.prefix
                weight_keys = []
                for weight_name, _ in weight_modules.items():
                    # There are QParamsLinear, RMS, lmhead.
                    # unquantized and quantized linear, "weight", "bias"(maybe it's something else.).
                    # full_name is "model.layers.3.mlp.experts.3.up_proj.weight" or "*.e_score_correction_bias" and so on.
                    full_name = prefix + weight_name
                    weight_keys.append(full_name)
                    hook.weights_map[weight_name].data = checkpoint_weights[full_name]
                    # can't del checkpoint_weights[full_name], should move to meta
                    checkpoint_weights[full_name] = checkpoint_weights[full_name].to("meta")

                for checkpoint_weights_name, _ in checkpoint_weights.items():  # TODO: Reduced complexity
                    if checkpoint_weights_name.startswith(prefix):
                        if checkpoint_weights_name not in weight_keys:  # is scale or zero
                            # how to add kv into weights_map? For OffloadedWeightsLoader and PrefixedDataset, it is not possible to add a k and v.
                            # So scale and zero are buffers that we put directly on the execution_device, while weight is handled by the hook.
                            checkpoint_weights[checkpoint_weights_name] = checkpoint_weights[
                                checkpoint_weights_name].to(hook.execution_device)
            torch.cuda.empty_cache()

    model.load_state_dict(checkpoint_weights, assign=True)
    model = model_importer._convert_model(model, model_config, model_state_dict)

    logger.info("hf_format quantized model imported successfully.")
    return model  # type: ignore [no-any-return]


def _load_weights_from_safetensors(model_info_dir: str) -> Dict[str, torch.Tensor]:
    '''
    Load the state dict from safetensor file with safetensors.torch.load_file, possibly from multiple safetensors files in case of sharded model.
    '''
    model_state_dict: Dict[str, torch.Tensor] = {}
    safetensors_dir = Path(model_info_dir)
    safetensors_path = safetensors_dir / SAFE_WEIGHTS_NAME
    safetensors_index_path = safetensors_dir / SAFE_WEIGHTS_INDEX_NAME
    if safetensors_path.exists():
        # In this case, the weights are in a single `model.safetensors` file.
        model_state_dict = load_file(str(safetensors_path))
    elif safetensors_index_path.exists():
        # In this case, the weights are split in several `.safetensors` files.
        with open(str(safetensors_index_path), "r") as file:
            safetensors_indices = json.load(file)
        safetensors_files = [value for _, value in safetensors_indices["weight_map"].items()]
        safetensors_files = list(set(safetensors_files))
        for filename in safetensors_files:
            filepath = safetensors_dir / filename
            model_state_dict.update(load_file(str(filepath)))
    else:
        raise FileNotFoundError(
            f"Neither {str(safetensors_path)} nor {str(safetensors_index_path)} were found. Please check that the model path specified {str(safetensors_dir)} is correct."
        )
    return model_state_dict


def _untie_parameters(model: nn.Module, model_state_dict: Dict[str, Any]) -> None:
    '''
    Some parameters share weights, such as embedding and lm_head, and when exporting with `PretrainedModel.save_pretrained`
    only one of them will be exported, so need to copy the parameters.
    '''
    # TODO: Only embedding for now, need to solve other cases, such as encoder-decoder tied
    tied_param_groups = find_tied_parameters(model)
    if len(tied_param_groups) > 0:
        if len(tied_param_groups) > 1 or "lm_head.weight" not in tied_param_groups[0]:
            raise ValueError(
                f"Your have tied_param_groups: {tied_param_groups}, temporarily does not support the case where tied_param is not 'lm_head and embedding'"
            )
        missing_key: List[str] = []
        tied_param_value: Optional[torch.Tensor] = None
        for tied_param_name in tied_param_groups[0]:
            if tied_param_name in model_state_dict.keys():
                tied_param_value = model_state_dict[tied_param_name]
            else:
                missing_key.append(tied_param_name)
        if tied_param_value is not None:
            for tied_param_key in missing_key:
                model_state_dict[tied_param_key] = tied_param_value
        else:
            raise ValueError("Cannot assign a value to tied_params because tied_param_value is None")
