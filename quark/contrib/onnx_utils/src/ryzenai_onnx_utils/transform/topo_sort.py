# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import copy
import random
from collections import defaultdict, deque

import onnx
from onnx import ModelProto, NodeProto

import ryzenai_onnx_utils


class TopologicalPathPlanner:
    """
    This class generates optimized topological sorts via custom mode.

    Attributes
    ----------
    model : onnx.ModelProto
        The model extracted from the ONNX model.

    optimize_method: str
        Defines the method for optimizing topology order. Options include:
        1. Normal Methods: These methods use Python's sorted function to arrange a list of
        strings in alphabetical order (case-sensitive, with uppercase letters preceding
        lowercase). Options are:
        - Ascending
        - Descending
        - Random
        2. Custom Methods:
        - Aggregation_first (major): Optimizes the tensor lifecycle to address scratch buffer
          limitations.
        - PDI_first (minor): Groups candidates with the same PDI to minimize PDI swaps.

    adjacency_list : dict[str, list[str]]
        Defines the mapping from the current node to its output nodes.
    adjacency_list_trans : dict[str, list[str]]
        Defines the mapping from the current node to its input nodes.

    node_mapping : dict[str, str]
        Defines the mapping from each node name to its op type.
    indegree : dict[str, int]
        Defines the mapping from the current node to its indegree values.

    PDI_mapping: dict[str, list[str]]
        For the PDI_first optimization method, the dictionary defines the mapping from PDI_id to
        kernel type.
    PDI_order: list[str]
        For the PDI_first optimization method, the list records the order of used PDI_ids in the
        pipeline.

    aggregation_list : list[str]
        Stores the order of all aggregation nodes in the DAG model.
        Aggregation nodes have more than two input nodes, and their order determines the
        priority of parallel edges.
    aggregation_node : dict[str, list[str]]
        Defines the mapping from the current aggregation node to its input nodes.

    topological_sort_result: list[str]
        Stores the node names from the optimized topological sort.
    topological_sort_graph: list[onnx.NodeProto]
        Stores the onnx.NodeProto objects from the optimized topological sort.
    """

    def __init__(
        self, extractor: onnx.utils.Extractor, optimize_method: str, PDI_mapping: dict[str, list[str]] | None = None
    ) -> None:
        self.model: ModelProto = extractor.model
        self.optimize_method: str = optimize_method

        self.adjacency_list: defaultdict[str, list[str]] = defaultdict(list)  # node: output node
        self.adjacency_list_trans: defaultdict[str, list[str]] = defaultdict(list)  # node: input node
        self.node_mapping: dict[str, str] = {}
        self.indegree: dict[str, int] = {}

        self.PDI_mapping: dict[str, list[str]] = defaultdict(list) if PDI_mapping is None else PDI_mapping
        self.PDI_order: list[str] = []

        self.aggregation_list: list[str] = []
        self.aggregation_node: defaultdict[str, list[str]] = defaultdict(list)  # node: input node

        self.topological_sort_result: list[str] = []
        self.topological_sort_graph: list[NodeProto] = []

    def check_node_name(self) -> None:
        """
        Check the validity of all node names in the DAG.

        This function performs the following checks:
        1. Checks that each node has a name.
        2. Checks that no nodes have duplicate names.
        """
        graph = self.model.graph
        name_list = [node.name for node in graph.node]
        type_idx_map: dict[str, int] = {}

        # 1. Checks that each node has a name.
        for node in graph.node:
            if not node.name:
                node_type = node.op_type
                type_idx_map[node_type] = type_idx_map.get(node_type, 0)
                idx = type_idx_map[node_type]
                set_name = f"{node_type}_{idx}"
                while set_name in name_list:
                    type_idx_map[node_type] += 1
                    set_name = f"{node_type}_{type_idx_map[node_type]}"
                name_list.remove(node.name)
                node.name = set_name
                name_list.append(set_name)

        # 2. Checks that no nodes have duplicate names.
        for node in graph.node:
            if name_list.count(node.name) > 1:
                idx = 1
                set_name = f"{node.name}_repeat{idx}"
                while set_name in name_list:
                    idx += 1
                    set_name = f"{node.name}_repeat{idx}"
                name_list.remove(node.name)
                node.name = set_name
                name_list.append(set_name)

    def record_aggregation_node(self) -> None:
        graph = self.model.graph

        node_outputs: defaultdict[str, list[str]] = defaultdict(list)
        for node in graph.node:
            for output in node.output:  # output edge
                node_outputs[output].append(node.name)

        for node in graph.node:
            input_count = sum(True for item in node.input if item in node_outputs)
            if input_count:
                self.adjacency_list_trans[node.name] = []
                if input_count == 1:
                    input = node.input[0]
                    self.adjacency_list_trans[node.name] += node_outputs[input]
                elif input_count > 1:
                    self.aggregation_node[node.name] = []
                    self.aggregation_list.append(node.name)
                    for input in node.input:
                        if input in node_outputs:
                            self.aggregation_node[node.name] += node_outputs[input]
                            self.adjacency_list_trans[node.name] += node_outputs[input]

    def generate_node_mapping(self) -> None:
        graph = self.model.graph
        for node in graph.node:
            op_type = node.op_type
            self.node_mapping[node.name] = op_type

    def extract_graph(self) -> None:
        graph = self.model.graph

        node_inputs: defaultdict[str, list[str]] = defaultdict(list)
        for node in graph.node:
            for input in node.input:  # input edge
                node_inputs[input].append(node.name)

        for node in graph.node:
            for output_node in node.output:
                if not node_inputs[output_node]:
                    self.adjacency_list[node.name] = []
                for output in node_inputs[output_node]:
                    self.adjacency_list[node.name].append(output)

    def calculate_indegree(self) -> None:
        self.indegree = defaultdict(int)
        for _, neighbors in self.adjacency_list.items():
            for neighbor in neighbors:
                self.indegree[neighbor] += 1
        for node, _ in self.adjacency_list.items():
            if node not in self.indegree:
                self.indegree[node] = 0

    def fix_record_order(self) -> None:
        for node, output_node in self.adjacency_list.items():
            self.adjacency_list[node] = sorted(output_node)
        for node, input_node in self.adjacency_list_trans.items():
            self.adjacency_list_trans[node] = sorted(input_node)
        for node, input_node in self.aggregation_node.items():
            self.aggregation_node[node] = sorted(input_node)

    def subgraph_outdegree_test(self, queue: list[str]) -> dict[str, int]:
        outdegree_change = {}
        for candidate in queue:
            outdegree_change[candidate] = len(self.adjacency_list[candidate])
        return outdegree_change

    def reorder_queue(self, queue: deque[str], last_elem: str | None) -> deque[str]:
        if self.optimize_method == "Aggregation_first":
            candidate_times: dict[str, int] = {}
            for candidate in queue:
                outputs = self.adjacency_list[candidate]
                output_indegree_before = [self.indegree[output] for output in outputs]
                output_indegree_after = [self.indegree[output] - 1 for output in outputs]
                times = output_indegree_after.count(0) - output_indegree_before.count(0)
                candidate_times[candidate] = times

            sorted_queue: deque[str] = deque(sorted(queue, key=lambda x: candidate_times[x], reverse=True))

            current_aggregation = self.aggregation_list[0] if len(self.aggregation_list) else "NULL"

            # epsilon indicates how many layers will be searched before the current aggregation
            current_aggregation_inputs_checked: list[str] = []
            if current_aggregation in self.aggregation_node:
                current_aggregation_inputs = self.aggregation_node[current_aggregation]
                for input in current_aggregation_inputs:
                    epsilon = 5
                    not_find = True
                    current_input = copy.deepcopy(input)
                    while epsilon:
                        # The input has been passed.
                        if current_input in self.topological_sort_result:
                            not_find = False
                            break
                        # The input is one of the current candidates.
                        if current_input in sorted_queue:
                            current_aggregation_inputs_checked.append(current_input)
                            not_find = False
                            break
                        else:
                            # search input's last node
                            current_input = (
                                self.adjacency_list_trans[current_input][0]
                                if len(self.adjacency_list_trans[current_input])
                                else "NULL"
                            )
                            epsilon -= 1
                    if not_find:
                        current_aggregation_inputs_checked.append(current_input)  # input, need to check

            current_aggregation_inputs_checked = sorted(set(current_aggregation_inputs_checked))
            has_aggregation = all(item in sorted_queue for item in current_aggregation_inputs_checked)
            if has_aggregation:  # delete current_aggregation inputs together
                if len(current_aggregation_inputs_checked):
                    for input in current_aggregation_inputs_checked:
                        sorted_queue.remove(input)
                    sorted_queue = deque(list(current_aggregation_inputs_checked) + list(sorted_queue))
                if current_aggregation in sorted_queue:
                    sorted_queue.remove(current_aggregation)
                    sorted_queue.insert(len(current_aggregation_inputs_checked), current_aggregation)
            return sorted_queue

        elif self.optimize_method == "PDI_first":
            element_to_group: dict[str, list[str]] = {}
            for group, elems in self.PDI_mapping.items():
                for elem in elems:
                    if elem not in element_to_group:
                        element_to_group[elem] = [group]
                    else:
                        element_to_group[elem].append(group)

            grouped_elements: dict[str, list[str]] = {group: [] for group in self.PDI_mapping}
            grouped_elements["NULL"] = []
            for element in queue:
                if self.node_mapping[element] in element_to_group:
                    groups = element_to_group.get(self.node_mapping[element])
                    if groups is not None:
                        for group in groups:
                            grouped_elements[group].append(element)
                else:
                    grouped_elements["NULL"].append(element)

            sorted_groups = sorted(grouped_elements.items(), key=lambda x: len(x[1]), reverse=True)

            last_elem_group = self.PDI_order[-1] if len(self.PDI_order) else None

            result: deque[str] = deque()

            if "NULL" in grouped_elements:
                result.extend(grouped_elements["NULL"])

            if last_elem_group:
                result.extend(grouped_elements[last_elem_group])

            single_pdi: list[str] = []
            for group, elems in self.PDI_mapping.items():
                if len(elems) == 1:
                    single_pdi.append(group)
                    result.extend(grouped_elements[group])

            for group, _ in sorted_groups:
                if group != last_elem_group and group not in single_pdi:
                    result.extend(grouped_elements[group])

            pdi_id = sorted_groups[0][0]

            if last_elem_group in grouped_elements and len(grouped_elements[last_elem_group]) != 0:
                pdi_id = last_elem_group
            elif len(single_pdi) and len(grouped_elements[single_pdi[0]]) != 0:
                pdi_id = single_pdi[0]
            self.PDI_order.append(pdi_id)

            # clear repeat element
            idx = 0
            seen: set[str] = set()
            while idx < len(result):
                item = result[idx]
                if item in seen:
                    result.remove(item)
                else:
                    seen.add(item)
                    idx += 1

            return result

        else:
            raise ValueError(f"Unexpected optimization method: {self.optimize_method}")

    def choose_order(self, queue: deque[str]) -> deque[str]:
        if self.optimize_method == "Ascending":
            return deque(sorted(queue))
        elif self.optimize_method == "Descending":
            return deque(sorted(queue, reverse=True))
        elif self.optimize_method == "Random":
            random.shuffle(queue)
            return deque(queue)
        elif self.optimize_method in ["Aggregation_first", "PDI_first"]:
            if self.optimize_method == "PDI_first" and not self.PDI_mapping:
                raise ValueError("The PDI mapping hasn't been provided.")
            last_elem = self.topological_sort_result[-1] if self.topological_sort_result else None
            # The `sorted` function in Python:
            # Sort a list of strings in alphabetical order (case-sensitive, usually uppercase
            # letters come before lowercase letters).
            # fix candidate queue order.
            sorted_queue = deque(sorted(queue))
            return deque(self.reorder_queue(sorted_queue, last_elem))
        else:
            raise ValueError(f"Unexpected optimization method: {self.optimize_method}")

    def topological_sort(self) -> None:
        self.check_node_name()

        self.record_aggregation_node()

        self.generate_node_mapping()

        self.extract_graph()

        self.calculate_indegree()

        self.fix_record_order()

        zero_indegree_nodes: deque[str] = deque()

        for node in self.node_mapping:
            if self.indegree[node] == 0:
                zero_indegree_nodes.append(node)

        while zero_indegree_nodes:
            current_aggregation = self.aggregation_list[0] if len(self.aggregation_list) else "NULL"
            if current_aggregation in self.topological_sort_result:
                del self.aggregation_list[0]

            zero_indegree_nodes = self.choose_order(zero_indegree_nodes)

            node = zero_indegree_nodes.popleft()
            if node in self.topological_sort_result:
                continue
            self.topological_sort_result.append(node)

            for neighbor in self.adjacency_list[node]:
                self.indegree[neighbor] -= 1
                if self.indegree[neighbor] == 0 and neighbor not in zero_indegree_nodes:
                    zero_indegree_nodes.append(neighbor)

        # check with node_mapping
        if len(self.topological_sort_result) != len(self.node_mapping):
            raise ValueError("topological sort failure!!!")
        else:
            ref_sorted_order = []
            for node in self.model.graph.node:
                ref_sorted_order.append(node.name)
            trans_sorted_order = [item[0] for item in self.topological_sort_result]
            # check with onnx model
            if len(ref_sorted_order) != len(trans_sorted_order):
                raise ValueError("topological sort failure!!!")
            else:
                # topological sort success, generate the sorted graph list
                self.generate_topological_sort_graph()
                if self.optimize_method == "PDI_first":
                    self.set_pdi_id()
                    # self.clear_pdi_order()

    def set_pdi_id(self) -> None:
        for idx in range(len(self.topological_sort_graph)):
            node = self.topological_sort_graph[idx]
            pdi_id = self.PDI_order[idx]
            support_op_type = {item for op_type in self.PDI_mapping.values() for item in op_type}
            if node.op_type in support_op_type:
                ryzenai_onnx_utils.matcher.add_attribute(
                    node,
                    "pdi_id",
                    int(pdi_id[-1]),
                )

    def clear_pdi_order(self) -> None:
        cleared_pdi_list: list[str] = []
        for pdi in self.PDI_order:
            if pdi == "NULL":
                continue
            if len(cleared_pdi_list) == 0 or pdi != cleared_pdi_list[-1]:
                cleared_pdi_list.append(pdi)
        print(len(cleared_pdi_list) - 1, cleared_pdi_list)

    def generate_topological_sort_graph(self) -> None:
        name_node_mapping: dict[str, NodeProto] = {}
        for node in self.model.graph.node:
            name_node_mapping[node.name] = node
        for node in self.topological_sort_result:
            node_name = node
            self.topological_sort_graph.append(name_node_mapping[node_name])


def optimize_toposort(
    extractor: onnx.utils.Extractor,
    optimize_method: str,
    PDI_mapping: dict[str, list[str]] | None = None,
) -> list[NodeProto]:
    path_plan = TopologicalPathPlanner(extractor, optimize_method, PDI_mapping)
    path_plan.topological_sort()
    return path_plan.topological_sort_graph
