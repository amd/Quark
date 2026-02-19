# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils
import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.passes import global_pass
from ryzenai_onnx_utils.typing import PatternType

from .model_type import ModelType


def _get_graph(extractor: onnx.utils.Extractor) -> onnx.GraphProto:
    """Helper to retrieve the graph proto from an extractor.

    The extractor may either wrap a full ModelProto or just a GraphProto directly.
    """
    if extractor.model is not None:
        return extractor.model.graph
    return extractor.graph


def _remove_from_dynamic_dispatch(graph: onnx.GraphProto, removed_inputs: set[str]) -> None:
    """Remove LoRA inputs from DynamicDispatch nodes and renumber their input shape attributes.

    After LoRA inputs are removed from the graph, any DynamicDispatch nodes that previously
    referenced them must be updated to:
      1. Remove the LoRA input names from the node's input list
      2. Update the 'input_num' attribute to reflect the new count
      3. Renumber 'input_shape_i' attributes to match the new positional indices
      4. Delete any obsolete 'input_shape_i' attributes for removed or out-of-range indices

    Args:
        graph: The ONNX graph containing DynamicDispatch nodes
        removed_inputs: Set of input names that have been identified as LoRA inputs
    """
    if not removed_inputs:
        return

    for node in graph.node:
        # Only process DynamicDispatch nodes
        if node.op_type != "DynamicDispatch":
            continue

        # Record original input list and identify which positions are being removed
        original_inputs = list(node.input)
        removed_positions: list[int] = []
        kept_inputs: list[str] = []

        # Partition inputs into kept vs. removed
        for idx, input_name in enumerate(original_inputs):
            if input_name in removed_inputs:
                removed_positions.append(idx)
            else:
                kept_inputs.append(input_name)

        # Skip this node if no LoRA inputs were found
        if not removed_positions:
            continue

        # Update the node's input list to exclude removed inputs
        node.input[:] = kept_inputs

        # Build a mapping from old input indices to new indices
        # This is needed to renumber the input_shape_i attributes
        removed_indices = set(removed_positions)
        index_mapping: dict[int, int] = {}
        new_idx = 0
        for old_idx in range(len(original_inputs)):
            if old_idx in removed_indices:
                continue
            index_mapping[old_idx] = new_idx
            new_idx += 1

        # Update the 'input_num' attribute to reflect the new input count
        model_type = ModelType.LLM_TOKEN
        for attr in node.attribute:
            if attr.name == "model_type":
                model_type = attr.i
        for attr in node.attribute:  # -1 for eliminating the extra added input "sequence_length" in prefill
            if attr.name == "input_num":
                attr.i = (len(kept_inputs) - 1) if (model_type == ModelType.LLM_PREFILL) else len(kept_inputs)
            if attr.name == "seq_len":
                attr.i = (len(kept_inputs) - 1) if (model_type == ModelType.LLM_PREFILL) else len(kept_inputs)

        # Process all 'input_shape_i' attributes
        # We'll collect attributes to delete and rename the rest
        attrs_to_delete: list[onnx.AttributeProto] = []
        for attr in node.attribute:
            if not attr.name.startswith("input_shape_"):
                continue
            try:
                # Extract the index from the attribute name (e.g., "input_shape_5" -> 5)
                old_idx = int(attr.name.split("_")[-1])
            except ValueError:
                continue

            # If this index was removed, mark the attribute for deletion
            if old_idx in removed_indices:
                attrs_to_delete.append(attr)
                continue

            # If this index is somehow missing from our mapping, mark for deletion
            if old_idx not in index_mapping:
                attrs_to_delete.append(attr)
                continue

            # Rename the attribute to use the new index
            new_index = index_mapping[old_idx]
            attr.name = f"input_shape_{new_index}"

        # Remove attributes that were marked for deletion
        for attr in attrs_to_delete:
            node.attribute.remove(attr)

        # Final cleanup: remove any input_shape_i attributes where i >= new input count
        # This catches any edge cases where attributes might exist beyond the input list
        for attr in list(node.attribute):
            if not attr.name.startswith("input_shape_"):
                continue
            try:
                idx = int(attr.name.split("_")[-1])
            except ValueError:
                continue
            if idx >= len(kept_inputs):
                node.attribute.remove(attr)


def _remove_graph_inputs(graph: onnx.GraphProto, extractor: onnx.utils.Extractor, inputs_to_remove: list[int]) -> None:
    """Remove LoRA inputs from the graph's input list and clean up related metadata.

    This function removes the specified inputs (by index) from:
      1. graph.input - the model's input list
      2. extractor.vimap - the value info mapping used during graph transformation
      3. graph.value_info - intermediate tensor metadata

    Args:
        graph: The ONNX graph to modify
        extractor: The extractor containing vimap and other metadata
        inputs_to_remove: List of input indices to remove (will be sorted in reverse order)
    """
    # Process indices in reverse order to avoid shifting problems
    # (removing index 5 doesn't affect indices 6, 7, 8... if we go backwards)
    for index in sorted(inputs_to_remove, reverse=True):
        name = graph.input[index].name

        # Remove from the graph's input list
        del graph.input[index]

        # Remove from the extractor's value info map (used during transformation)
        if extractor.vimap is not None:
            extractor.vimap.pop(name, None)


@global_pass
def remove_lora_inputs(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    """Global pass to remove LoRA placeholder inputs after they're no longer needed.

    This pass runs after dd.llm_token_to_dd has fused all LoRA-enabled MatMul ops
    into DynamicDispatch nodes. At that point, the LoRA input placeholders are no
    longer consumed by any nodes in the graph, so we can safely remove them.

    The pass performs two main operations:
      1. Updates DynamicDispatch nodes to remove LoRA inputs and renumber their
         input_shape_i attributes to maintain consistency
      2. Removes the LoRA inputs from the graph's input list and cleans up
         associated metadata (vimap, value_info)

    This cleanup is important because:
      - It reduces the model's input signature to only what's actually needed
      - It prevents confusion when the model is loaded/executed
      - It saves memory by not requiring dummy LoRA buffers at runtime

    Args:
        extractor: The graph extractor containing the model
        pass_id: Unique identifier for this pass execution
        subgraph: Empty list (this is a global pass, not a pattern-based replacement)
        params: Pass parameters (unused in this pass)
    """
    graph = _get_graph(extractor)

    # Scan all graph inputs to identify LoRA placeholders
    # We identify them by checking if "lora" appears in the input name
    lora_input_indices: list[int] = []
    lora_input_names: list[str] = []

    for index, input_tvi in enumerate(graph.input):
        name = input_tvi.name
        # LoRA inputs follow naming patterns like "q_proj_lora.0", "v_proj_lora.1", etc.
        if "lora" not in name.lower():
            continue
        lora_input_indices.append(index)
        lora_input_names.append(name)

    # Early exit if no LoRA inputs were found
    if not lora_input_indices:
        return

    # Step 1: Update all DynamicDispatch nodes to remove LoRA inputs
    # and renumber their input_shape_i attributes
    _remove_from_dynamic_dispatch(graph, set(lora_input_names))

    # Step 2: Remove the LoRA inputs from the graph's input list
    # and clean up related metadata
    _remove_graph_inputs(graph, extractor, lora_input_indices)


PATTERN: PatternType = []
REPLACEMENT = remove_lora_inputs
