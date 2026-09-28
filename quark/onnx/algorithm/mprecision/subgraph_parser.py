#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import onnx

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


@dataclass
class SubgraphSpec:
    name: str
    start_nodes: list[str]
    end_nodes: list[str]
    resolved_nodes: list[str] = field(default_factory=list)


def _resolve_nodes(
    model: onnx.ModelProto,
    start_nodes: list[str],
    end_nodes: list[str],
) -> list[str]:
    """
    Walk the graph forward from every start_node and collect all reachable
    nodes up to and including any end_node.
    """
    node_map: dict[str, Any] = {n.name: n for n in model.graph.node}
    # Build output-tensor -> node map
    out_to_node: dict[str, str] = {}
    for n in model.graph.node:
        for o in n.output:
            out_to_node[o] = n.name

    # Input-tensor -> list of consumer node names
    in_to_nodes: dict[str, list[str]] = {}
    for n in model.graph.node:
        for inp in n.input:
            in_to_nodes.setdefault(inp, []).append(n.name)

    end_set = set(end_nodes)
    collected: list[str] = []
    visited: set[str] = set()
    queue: deque[str] = deque(start_nodes)

    while queue:
        name = queue.popleft()
        if name in visited or name not in node_map:
            continue
        visited.add(name)
        collected.append(name)
        if name in end_set:
            continue  # do not traverse beyond end nodes
        node = node_map[name]
        for out_tensor in node.output:
            for consumer_name in in_to_nodes.get(out_tensor, []):
                if consumer_name not in visited:
                    queue.append(consumer_name)

    return collected


def parse_subgraph_json(
    path: str | Path,
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
) -> list[SubgraphSpec]:
    """
    Load and validate a subgraph partition JSON. Returns SubgraphSpec list with
    resolved_nodes populated. An implicit '__ungrouped__' entry is appended for
    any nodes not mentioned in the JSON.

    Node traversal is performed on *float_model* because subgraph partitions are
    defined against the float graph topology. Each resolved subgraph is then
    filtered down to only the nodes that exist in *quant_model* (graph
    optimisation may have removed or renamed nodes). If any start or end node
    declared for a subgraph is absent from *quant_model*, the subgraph cannot be
    reliably identified in the quantized graph; all of its float-resolved nodes
    are moved to ``__ungrouped__`` and the subgraph is skipped with a warning.

    JSON schema:
      {
        "quantized":     bool (optional, default false),
        "num_subgraphs": int  (optional, must match len(subgraphs) if present),
        "subgraphs":     [{"name": str, "start_nodes": [str], "end_nodes": [str]}, ...]
      }

    ``quantized`` at the top level indicates the whole model is quantized: all
    boundary nodes are validated against *quant_model* and topology is resolved
    on *quant_model* directly.  When ``false`` (default), the float model is the
    source of truth and resolved nodes are filtered to those present in *quant_model*.

    ``num_subgraphs`` is informational; when present it must equal the number of
    entries in ``subgraphs`` or a ``ValueError`` is raised.
    """
    with open(path) as f:
        data = json.load(f)

    is_quantized: bool = bool(data.get("quantized", False))

    num_declared = data.get("num_subgraphs")
    actual_count = len(data.get("subgraphs", []))
    if num_declared is not None and num_declared != actual_count:
        raise ValueError(
            f"num_subgraphs={num_declared} does not match the actual number of subgraph entries ({actual_count})."
        )

    float_node_names = {n.name for n in float_model.graph.node}
    quant_node_names = {n.name for n in quant_model.graph.node}

    specs: list[SubgraphSpec] = []
    assigned: set[str] = set()
    ungrouped_extra: list[str] = []

    for entry in data.get("subgraphs", []):
        name = entry["name"]
        start_nodes: list[str] = entry["start_nodes"]
        end_nodes: list[str] = entry["end_nodes"]

        if is_quantized:
            # Whole model is quantized: validate boundary nodes against quant_model.
            for node_name in start_nodes + end_nodes:
                if node_name not in quant_node_names:
                    raise ValueError(f"Subgraph '{name}': node '{node_name}' not found in the quantized model.")
            resolved = _resolve_nodes(quant_model, start_nodes, end_nodes)
        else:
            # Validate that boundary nodes exist in the float model (source of truth).
            for node_name in start_nodes + end_nodes:
                if node_name not in float_node_names:
                    raise ValueError(f"Subgraph '{name}': node '{node_name}' not found in the float model.")

            # Resolve topology on the float model.
            resolved_float = _resolve_nodes(float_model, start_nodes, end_nodes)

            # Check that all resolved nodes survive graph optimisation in the quant model.
            # Only the absent nodes are excluded; the surviving nodes still form the subgraph.
            missing = [n for n in resolved_float if n not in quant_node_names]
            if missing:
                logger.warning(
                    f"Subgraph '{name}': {len(missing)} node(s) are absent from the quantized model "
                    f"(likely removed by graph optimisation): {missing}. "
                    f"Excluding them from the subgraph."
                )

            # Keep only nodes that actually exist in the quant model.
            resolved = [n for n in resolved_float if n in quant_node_names]

        overlap = set(resolved) & assigned
        if overlap:
            logger.warning(
                f"Subgraph '{name}' overlaps with a previously defined subgraph on nodes: {overlap}. "
                f"Removing overlapping nodes from '{name}'."
            )
            resolved = [n for n in resolved if n not in overlap]

        assigned.update(resolved)
        specs.append(SubgraphSpec(name=name, start_nodes=start_nodes, end_nodes=end_nodes, resolved_nodes=resolved))

    ungrouped = [n for n in quant_node_names if n not in assigned] + ungrouped_extra
    if ungrouped:
        specs.append(
            SubgraphSpec(
                name="__ungrouped__",
                start_nodes=[],
                end_nodes=[],
                resolved_nodes=ungrouped,
            )
        )

    return specs
