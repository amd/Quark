#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn
import os
import re
import json
from operator import add, mul
from typing import Tuple, Dict, List, Any, Type, Union, cast

from quark.shares.utils.log import ScreenLogger
from quark.torch.quantization.config.config import Config, RotationConfig, SmoothQuantConfig, AWQConfig, AlgoConfig

logger = ScreenLogger(__name__)

MODULES_WITH_CONST_PARAM: List[Type[nn.Module]] = [nn.Linear, nn.LayerNorm, nn.Embedding]

# torch.compile used to use `nn.Module` subclasses in its fx node metadata `source_fn_stack` for torch 2.3, but is using
# functional calls in torch 2.5.
CONST_PARAM_SOURCE_FN = MODULES_WITH_CONST_PARAM + [
    torch.nn.functional.embedding, torch.nn.functional.layer_norm, torch.nn.functional.linear
]


def is_auto_config_needed(config: Config) -> Tuple[int, int, bool]:
    smooth_position = -1  # Position of smooth config in pre-optimization, -1 if not found. We cannot use bool here because pre-optizization is a list related to orders. If we want to do smooth then do rotation, it will be like smooth_position=0 and rotation_position=1
    rotation_position = -1  # Position of rotation config in pre-optimization, -1 if not found. We cannot use bool here because pre-optizization is a list related to orders. If we want to do smooth then do rotation, it will be like smooth_position=0 and rotation_position=1
    is_awq_needed = False  # Here we use bool because

    # Iterate over pre-quant optimization configs
    for i, pre_optimization in enumerate(config.pre_quant_opt_config):
        if isinstance(pre_optimization, RotationConfig) and len(pre_optimization.scaling_layers) == 0:
            rotation_position = i
        elif isinstance(pre_optimization, SmoothQuantConfig) and len(pre_optimization.scaling_layers) == 0:
            smooth_position = i

    # Check if AWQ configuration is needed
    if isinstance(config.algo_config, AWQConfig) and len(config.algo_config.scaling_layers) == 0:
        is_awq_needed = True

    return smooth_position, rotation_position, is_awq_needed


def add_auto_config(model: nn.Module, dummy_input: torch.Tensor, config: Config, smooth_position: int,
                    rotation_position: int, is_awq_needed: bool) -> Config:
    logger.info(
        "Lack of specific information of algorithm configuration, auto generating algorithms configuration. It may take several minutes..."
    )

    # Create EasyGraph instance for processing
    eg = EasyGraph(model, dummy_input, is_rotation_mode=(rotation_position >= 0))

    # Generate rotation configuration if needed
    if rotation_position >= 0:
        logger.info("Auto generating rotation configuration...")
        rotation_config_dict = eg.get_rotation_config()
        # config.pre_quant_opt_config[rotation_position] = RotationConfig.from_dict(rotation_config_dict)
        config.pre_quant_opt_config[rotation_position].update_from_dict(rotation_config_dict)
        # dump_config_to_json(model, "rotations_config", rotation_config_dict)

    # Generate smooth configuration if needed
    if smooth_position >= 0:
        logger.info("Auto generating smooth configuration...")
        smooth_config = eg.get_parameterized_pair_config()
        config.pre_quant_opt_config[smooth_position] = SmoothQuantConfig.from_dict(smooth_config)
        # dump_config_to_json(model, "parameterized_pair_config", smooth_config)

    # Generate AWQ configuration if needed
    if is_awq_needed:
        logger.info("Auto generating AWQ configuration...")
        awq_config = eg.get_parameterized_pair_config()

        # generate model_decoder_layers
        name_count_dict: Dict[str, Any] = {}
        for item in awq_config['scaling_layers']:
            result = re.split(r'\.\d+', item['prev_op'])
            if result[0] in name_count_dict:
                name_count_dict[result[0]] = name_count_dict[result[0]] + 1
            else:
                name_count_dict[result[0]] = 1
        model_decoder_layers = max(name_count_dict, key=lambda k: cast(int, name_count_dict.get(k)))

        awq_config['model_decoder_layers'] = model_decoder_layers
        # filter scaling_layers
        scaling_layers_set = set()
        for item in awq_config['scaling_layers']:
            item['prev_op'] = re.split(r'\.\d+\.', item['prev_op'], maxsplit=1)[-1]
            item['module2inspect'] = re.split(r'\.\d+\.', item['module2inspect'], maxsplit=1)[-1]
            item['layers'] = [re.split(r'\.\d+\.', layer, maxsplit=1)[-1] for layer in item['layers']]
            item['layers'].sort()
            scaling_layers_set.add(tuple([item['prev_op'], item['module2inspect'], *item['layers']]))
        awq_config['scaling_layers'] = [{
            'prev_op': item[0],
            'inp': item[2],
            'module2inspect': item[1],
            'layers': list(item[2:])
        } for item in scaling_layers_set]
        config.algo_config = cast(AlgoConfig, AWQConfig.from_dict(awq_config))
        # dump_config_to_json(model, "parameterized_pair_config", awq_config)

    return config


def dump_config_to_json(model: nn.Module, file_name: str, data_dict: Dict[str, Any]) -> None:
    # TODO: Move this function to quark.torch.quantization.config
    model_config_result = "model_config_result" + model.__class__.__name__ + '/'
    if not os.path.exists(model_config_result):
        os.makedirs(model_config_result)
    with open(os.path.join(model_config_result, (file_name + ".json")), 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)


class EasyGraph():

    @staticmethod
    def find_nearest_module_name(node: torch.fx.node.Node) -> str:
        module_info: str = ""
        if 'nn_module_stack' in node.meta.keys():
            name_info = [value for _, value in node.meta['nn_module_stack'].items()][-1][0]
            module_info = name_info.replace("L['self'].", "")

            # with torch 2.5, the original string looks like
            # "L['self']._modules['model']._modules['layers']._modules['0'].input_layernorm".
            module_info = re.sub(r"_modules\['([^']+)'\]\.", r"\1.", module_info)

            # with torch 2.3, the original string looks like "L['self'].model.layers[0].input_layernorm".
            module_info = re.sub(r'\[(\d+)\]\.', r'.\1.', module_info)
            module_info = re.sub(r'\[(\d+)\]', r'.\1', module_info)

            # The resulting string here would be "model.layers.0.input_layernorm".
        return module_info

    def __init__(self, model: torch.nn.Module, sample_input: torch.Tensor, is_rotation_mode: bool = False) -> None:
        self.name_2_module = {}
        for name, module in model.named_modules():
            self.name_2_module[name] = module
        self.parameterized_pair_list: Dict[str, List[str]] = {}
        self.rotation_pair_list: List[List[str]] = []

        def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Any:
            model_graph = "model_graph"
            if not os.path.exists(model_graph):
                os.makedirs(model_graph)
            graph_str = gm.print_readable()
            with open(os.path.join(model_graph, self.model.__class__.__name__ + ".txt"), 'w') as f:
                f.write(graph_str)
            self.gm = gm
            # ---------------------------------------------------------------------
            # make name convert
            param_dict: Dict[int, str] = {}
            self.parameters_convert: Dict[str, str] = {}
            for name, module in model.named_modules():
                setattr(module, "module_name", name)

            # `self.gm.named_modules()` is empty with torch 2.5, but contains modules as
            # `L__self___model_embed_tokens`, `L__self___model_layers_0_self_attn_q_proj`,
            # `L__self___model_layers_0_self_attn_k_proj`, etc. with torch 2.3.
            for name, module in self.gm.named_modules():
                if hasattr(module, "module_name"):
                    self.parameters_convert[name.lower()] = module.module_name
            # ---------------------------------------------------------------------
            self.is_rotation_mode = False
            self.find_nn_linear()
            if is_rotation_mode:
                self.is_rotation_mode = is_rotation_mode
                self.find_nn_linear()
            return gm.forward

        torch._dynamo.reset()
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

        model_graph_break_info = "model_graph_break_info"
        if not os.path.exists(model_graph_break_info):
            os.makedirs(model_graph_break_info)
        graph_info = torch._dynamo.explain(model, sample_input)
        with open(os.path.join(model_graph_break_info, model.__class__.__name__ + ".txt"), 'w') as f:
            f.write(str(graph_info))

        self.model = model
        torch.compile(model, backend=custom_backend)(sample_input, use_cache=False)

    def _is_weight_node(self, node: torch.fx.node.Node) -> bool:
        if node.op == 'get_attr':  # pragma: no cover
            # This path exists in torch<=2.3.
            return "source_fn_stack" not in node.meta.keys()
        elif node.op == "placeholder":
            # Weights are placeholders in more recent torch versions.
            return node.type == torch.nn.Parameter
        else:
            return False

    def _check_has_non_linear_sub_node(self, node: torch.fx.node.Node) -> bool:
        node_stack = [node]
        linear_node_list = [
            add, mul, "view", "getitem", "expand", "reshape", "contiguous", "transpose", "view", "permute", "to",
            "chunk", "sub", "int", "float", "cat", "matmul", "bmm", "truediv", "mean", "neg", "relu"
        ]

        if self.is_rotation_mode:
            linear_node_list.append("pow")

        while len(node_stack) > 0:
            tmp_node = node_stack.pop()
            if tmp_node.op == "output":
                continue
            node_type = [x for x in tmp_node.meta['source_fn_stack']][-1][-1]
            if any(keyword is node_type for keyword in CONST_PARAM_SOURCE_FN):
                continue
            if any(str(keyword) in str(node_type) for keyword in linear_node_list):
                node_stack.extend([k for k, v in tmp_node.users.items()])
            else:
                logger.info(f"non-linear: {node_type} {type(node_type)}")
                return False
        return True

    def _check_node(self, node: torch.fx.node.Node) -> bool:
        if all([self._check_has_non_linear_sub_node(k) for k, v in node.users.items()]):
            return True
        else:
            return False

    def get_module_name_by_node(self, node: torch.fx.node.Node) -> Any:
        if "source_fn_stack" in node.meta.keys():
            module_info = [x for x in node.meta['source_fn_stack']]
            node_name = module_info[-1][0]
            if node_name in self.parameters_convert.keys():
                return self.parameters_convert[node_name]
            elif (node_name.lower() + '.weight') in self.parameters_convert.keys():
                return self.parameters_convert[node_name + '.weight'][:-7]
            elif (node_name.lower() + '_weight') in self.parameters_convert.keys():
                return self.parameters_convert[node_name + '.weight'][:-7]
            else:
                return EasyGraph.find_nearest_module_name(node)
        else:
            return EasyGraph.find_nearest_module_name(node)

    @torch._dynamo.disable  # type: ignore[misc]
    def find_nn_linear(self) -> None:
        for node in self.gm.graph.nodes:
            if 'source_fn_stack' in node.meta.keys():
                module_info = [x for x in node.meta['source_fn_stack']]
                if module_info[-1][-1] is torch.nn.Linear or module_info[-1][-1] == torch.nn.functional.linear:
                    args_name_list = node.args
                    module_name = self.get_module_name_by_node(node)
                    self.find_merge_pair(args_name_list[0], [module_name])

    def _add_pair_list(self, node: torch.fx.node.Node, prefix_model: List[str]) -> None:
        prefix_model = [*prefix_model]
        module_name = self.get_module_name_by_node(node)

        if self.is_rotation_mode:
            if len(prefix_model) != 2:
                prefix_model.append(module_name)
                if len(node.args) == 0:
                    # const parameters
                    parent_node = [k for k in node.users.keys()][0]
                    for node_args in parent_node.args:
                        if isinstance(node_args, torch.fx.node.Node):
                            if node_args is not node:
                                return self.find_merge_pair(node_args, prefix_model)
                else:
                    if node.meta["source_fn_stack"][-1][-1] is not torch.nn.Linear and node.meta["source_fn_stack"][-1][
                            -1] != torch.nn.functional.linear:
                        if isinstance(node.args[0], torch.fx.node.Node):
                            return self.find_merge_pair(node.args[0], prefix_model)
            else:
                prefix_model.append(module_name)
                self.rotation_pair_list.append(prefix_model)
        else:
            if not self._check_node(node):
                return
            if module_name not in self.parameterized_pair_list.keys():
                self.parameterized_pair_list[module_name] = prefix_model
            else:
                self.parameterized_pair_list[module_name].extend(prefix_model)

    def _is_target_type(self, input_node_type: Type[Any], target_list: List[Union[type, str]]) -> bool:
        for target_node in target_list:
            if isinstance(target_node, str):
                input_node_type_str = str(input_node_type).replace("torch.nn.modules", "").replace("torch", "")
                if target_node in input_node_type_str:
                    return True
            else:
                if target_node is input_node_type:
                    return True

        return False

    def is_node_source_matching_target(self, node: torch.fx.node.Node, target_list: List[Type[nn.Module]]) -> bool:
        """
        Returns True if a node metadata has one of the modules from `target_list` as its last element in its modules stack.
        """
        if "nn_module_stack" not in node.meta:
            return False
        else:
            last_nn_module_stack_name = list(node.meta["nn_module_stack"].keys())[-1]

            # nn_module_stack appears to be a dictionary of string to tuples of length 2, but using `-1`
            # here just to be safe.
            last_module = node.meta["nn_module_stack"][last_nn_module_stack_name][-1]

            if any(last_module is target_module for target_module in target_list):
                return True
            else:
                return False

    def find_merge_pair(self, node: torch.fx.node.Node, prefix_model: List[str]) -> None:
        logger.debug("Processing node in find_merge_pair:", node, node.op, node.target, node.type, node.meta)

        node_has_double_input_tensor: List[Any] = [mul, add]
        node_has_double_input_tensor_only_left: List[Any] = [torch.matmul, torch.bmm]
        node_has_one_input_tensor: List[Any] = [
            "getitem", "expand", "reshape", "contiguous", "transpose", "view", "permute", "to", "chunk",
            torch.nn.modules.activation.ReLU, torch.nn.functional.relu
        ]

        if not isinstance(node, torch.fx.node.Node):  # pragma: no cover
            logger.info("Node in find_merge_pair is not a torch.fx.node.Node.")
            return

        # These nodes are the input of the graph.
        # Weights are stored as placeholder nodes in recent versions of PyTorch (at least in torch 2.5), hence
        # the checks on metadata.
        if node.op == "placeholder" and node.type == torch.Tensor:
            return

        if self._is_weight_node(node):
            return self._add_pair_list(node, prefix_model)

        node_type = [x for x in node.meta["source_fn_stack"]][-1][-1]

        if self.is_node_source_matching_target(node, MODULES_WITH_CONST_PARAM) and self._is_target_type(
                node_type, CONST_PARAM_SOURCE_FN):  # type: ignore
            return self._add_pair_list(node, prefix_model)
        elif self._is_target_type(node_type, node_has_double_input_tensor):
            # double input tensor
            self.find_merge_pair(node.args[0], prefix_model)
            self.find_merge_pair(node.args[1], prefix_model)

        elif self._is_target_type(node_type, node_has_double_input_tensor_only_left):
            # double input tensor
            self.find_merge_pair(node.args[1], prefix_model)

        elif self._is_target_type(node_type, node_has_one_input_tensor):
            # single input tensor
            self.find_merge_pair(node.args[0], prefix_model)

        else:
            logger.info(f"except: { node.name} {node_type}")

    def find_common_descendant_by_layers(self, node_list: List[torch.fx.node.Node]) -> torch.fx.node.Node:
        if len(node_list) == 1:
            return node_list[0]
        find_count = len(node_list)
        for idx in range(len(node_list)):
            setattr(node_list[idx], "bfs_flag", [idx])
        while len(node_list) > 0:
            node = node_list.pop()
            for sub_node, v in node.users.items():
                if hasattr(sub_node, "bfs_flag"):
                    sub_node.bfs_flag.extend(node.bfs_flag)  # type: ignore[attr-defined]
                    sub_node.bfs_flag = list(set(sub_node.bfs_flag))
                    if len(sub_node.bfs_flag) == find_count:
                        return sub_node
                else:
                    setattr(sub_node, "bfs_flag", node.bfs_flag)  # type: ignore[attr-defined]
                node_list.insert(0, sub_node)
        logger.info("no common node")
        assert (False)

    def generate_module2inspect(self, parameterized_pair_dict: Dict[str, Any]) -> None:
        self.module_name2node = {}
        for node in self.gm.graph.nodes:
            if node.name in self.parameters_convert:
                self.module_name2node[self.parameters_convert[node.name]] = node
            else:
                self.module_name2node[EasyGraph.find_nearest_module_name(node)] = node

        for scaling_layer in parameterized_pair_dict['scaling_layers']:
            module2inspect_node = self.find_common_descendant_by_layers(
                [self.module_name2node[layer_name] for layer_name in scaling_layer['layers']])
            if module2inspect_node.name in self.parameters_convert:
                module2inspect_module_name = self.parameters_convert[module2inspect_node.name]
            else:
                module2inspect_module_name = EasyGraph.find_nearest_module_name(module2inspect_node)
            scaling_layer["module2inspect"] = module2inspect_module_name

    def get_parameterized_pair_config(self) -> Dict[str, Any]:
        parameterized_pair_dict: Dict[str, Any] = {}
        scaling_layers = []
        for k, v in self.parameterized_pair_list.items():
            scaling_layers.append({"prev_op": k, "layers": v})
        parameterized_pair_dict['scaling_layers'] = scaling_layers
        parameterized_pair_dict['name'] = 'awq'
        self.generate_module2inspect(parameterized_pair_dict)
        return parameterized_pair_dict

    def get_rotation_config(self) -> Dict[str, List[Dict[str, Any]]]:
        data_dict: Dict[Any, Any] = {}
        result = []
        for k in self.rotation_pair_list:
            if (k[2], k[1]) not in data_dict.keys():
                data_dict[(k[2], k[1])] = [k[0]]
            else:
                data_dict[(k[2], k[1])].append(k[0])

        data_dict2: Dict[Any, Any] = {}

        for k, v in data_dict.items():
            v.sort()
            v = tuple(v)
            if (v, k[1]) not in data_dict2.keys():
                data_dict2[(v, k[1])] = [k[0]]
            else:
                data_dict2[(v, k[1])].append(k[0])

        for k, v in data_dict2.items():
            result.append({"prev_modules": v, "norm_module": k[1], "next_modules": k[0]})

        data_dict = {"scaling_layers": result}

        return data_dict
