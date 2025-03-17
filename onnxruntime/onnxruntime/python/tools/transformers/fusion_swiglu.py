# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging

from fusion_base import Fusion
from onnx import helper
from onnx_model import OnnxModel

logger = logging.getLogger(__name__)


class FusionSwiGLU(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "SiLU", ["Mul"])

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        # Fuse the following subgraph to `SwiGLU`
        #
        #       root_input
        #      /          \
        #       \     Sigmoid       ----+
        #        \     /                |
        #         \   /                 |
        #          Mul              ----+
        #           |
        #       root_output

        if node.op_type != "Mul":
            logger.debug("fuse_swiglu: failed to match Mul node")
            return

        second_mul_node = node
        root_input = second_mul_node.input[0]

        sigmoid_node = self.model.match_parent_path(second_mul_node, ["Sigmoid"], [1])
        if sigmoid_node is None:
            logger.debug("fuse_swiglu: failed to match Sigmoid node")
            return
        sigmoid_node = sigmoid_node[0]

        # first_mul_node = self.model.match_parent_path(sigmoid_node, ["Mul"], [0])
        # if first_mul_node is None:
        #     logger.debug("fuse_swiglu: failed to match first Mul node")
        #     return
        # first_mul_node = first_mul_node[0]

        # approximation_value = self.model.get_constant_value(first_mul_node.input[1]).item()
        # if abs(approximation_value - 1.7021484375) >= 1e-3:
        #     logger.debug("fuse_quickgelu: failed to match approximation value")
        #     return

        if sigmoid_node.input[0] != root_input:
            logger.debug("fuse_swiglu: failed to match root input with Mul node's input")
            return

        new_node = helper.make_node(
            "SiLU",
            inputs=[root_input],
            outputs=[second_mul_node.output[0]],
            name=self.model.create_node_name("SiLU"),
        )
        new_node.domain = "com.microsoft"

        self.nodes_to_remove.extend([sigmoid_node, second_mul_node])
        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name
        self.increase_counter("SiLU")
