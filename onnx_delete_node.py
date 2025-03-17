import onnx
from onnx import helper
import onnx_graphsurgeon as gs

model_path = "./llama2-70b/llama2-70b.onnx"
model_output_path = "./llama2-70b/llama2-70b-delete.onnx"

onnx_model = onnx.load(model_path, load_external_data=False)
graph = onnx_model.graph

nodes_to_remove = [node for node in graph.node if node.op_type == 'Transpose']

for node in nodes_to_remove:
    input_name = node.input[0]
    output_name = node.output[0]
    
    for next_node in graph.node:
        for i, input_name in enumerate(next_node.input):
            if input_name == node.output[0]:
                next_node.input[i] = node.input[0]
    
    graph.node.remove(node)

nodes_to_remove = [node for node in graph.node if 'Reshape_4' in node.name or 'Reshape_3' in node.name]

for node in nodes_to_remove:
    input_name = node.input[0]
    output_name = node.output[0]
    
    for next_node in graph.node:
        for i, input_name in enumerate(next_node.input):
            if input_name == node.output[0]:
                next_node.input[i] = node.input[0]
    
    graph.node.remove(node)

nodes_to_remove = [node for node in graph.node if node.op_type == 'Expand']

for node in nodes_to_remove:
    input_name = node.input[0]
    output_name = node.output[0]
    
    for next_node in graph.node:
        for i, input_name in enumerate(next_node.input):
            if input_name == node.output[0]:
                next_node.input[i] = node.input[0]
    
    graph.node.remove(node)


onnx.save(onnx_model, model_output_path, save_as_external_data=False)


onnx_graph = onnx.load(model_output_path, load_external_data=False)
    
graph = gs.import_onnx(onnx_graph)
    
graph.cleanup().toposort()

onnx.save(gs.export_onnx(graph), model_output_path, save_as_external_data=False)