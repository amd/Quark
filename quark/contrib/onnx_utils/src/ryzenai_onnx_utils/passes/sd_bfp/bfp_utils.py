from abc import ABC, abstractmethod

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import (
    copy_attributes,
    get_shape,
    is_initializer,
    set_attribute,
)
from ryzenai_onnx_utils.partitioner import get_dynamic_shape_candidate
from ryzenai_onnx_utils.passes.sd_bfp.bfp_cast import add_sd_cast_bf16_to_bfp, add_sd_cast_bfp_to_bf16
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_bfp_supported_shape(shape: tuple[int | str, ...], params: ryzenai_onnx_utils.ReplaceParams) -> bool:
    candidates = get_dynamic_shape_candidate([shape], params.attributes)
    for shape_group in candidates:
        for s in shape_group:
            if not (len(s) >= 2 and s[-2] % 8 == 0 and s[-1] % 8 == 0):
                return False
    return True


class BfpOpWrapper(ABC):
    def __init__(
        self, node: onnx.NodeProto, extractor, pass_id: int, domain: str, params: ryzenai_onnx_utils.ReplaceParams
    ):
        self.node = node
        self.extractor = extractor
        self.pass_id = pass_id
        self.domain = domain
        self.params = params

        self.tvis: list[onnx.ValueInfoProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.new_nodes: list[onnx.NodeProto] = []

    @property
    @abstractmethod
    def bfp_op_type(self) -> str:
        pass

    def add_pre_cast(self, input_name: str) -> str:
        input_shape = get_shape(input_name, self.extractor)
        output_name = f"{input_name}_bfp.out{self.pass_id}"
        nodes, inits, tvis = add_sd_cast_bf16_to_bfp(input_name, output_name, input_shape, self.domain)
        self.new_nodes.extend(nodes)
        self.initializers.extend(inits)
        self.tvis.extend(tvis)
        return output_name

    def add_post_cast(self, output_name: str):
        output_shape = get_shape(output_name, self.extractor)
        bfp_output_name = f"{output_name}_bfp.out{self.pass_id}"
        nodes, inits, tvis = add_sd_cast_bfp_to_bf16(bfp_output_name, output_name, output_shape, self.domain)
        self.new_nodes.extend(nodes)
        self.initializers.extend(inits)
        self.tvis.extend(tvis)

    def get_in_dtypes(self) -> list[str]:
        bfp_in_dtypes: list[str] = []
        for input_name in self.node.input:
            input_shape = get_shape(input_name, self.extractor)
            if is_initializer(input_name, self.extractor) or not is_bfp_supported_shape(input_shape, self.params):
                bfp_in_dtypes.append("bfloat16")
            else:
                bfp_in_dtypes.append("bfp16ebs8")
        return bfp_in_dtypes

    def wrap(self) -> PassOutputArgs:
        bfp_inputs: list[str] = []

        for input_name in self.node.input:
            input_shape = get_shape(input_name, self.extractor)
            if is_initializer(input_name, self.extractor) or not is_bfp_supported_shape(input_shape, self.params):
                bfp_inputs.append(input_name)
            else:
                bfp_output_name = self.add_pre_cast(input_name)
                bfp_inputs.append(bfp_output_name)

        output_shape = get_shape(self.node.output[0], self.extractor)
        sd_output_name = f"{self.node.output[0]}_bfp.out{self.pass_id}"
        sd_output_tvi = onnx.helper.make_tensor_value_info(sd_output_name, onnx.TensorProto.UINT8, output_shape)
        self.tvis.append(sd_output_tvi)

        bfp_node = onnx.helper.make_node(
            self.bfp_op_type,
            inputs=bfp_inputs,
            outputs=[sd_output_name],
            domain=self.domain,
            name=self.node.name,
        )
        copy_attributes(self.node, bfp_node)
        set_attribute(bfp_node, "in_dtypes", self.get_in_dtypes())
        set_attribute(bfp_node, "out_dtypes", ["bfp16ebs8"])
        self.new_nodes.append(bfp_node)

        self.add_post_cast(self.node.output[0])

        return self.new_nodes, self.initializers, self.tvis
