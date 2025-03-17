import onnx
from onnx import helper

import struct

model_path = "./llama2-70b/llama2-70b-delete.onnx"

onnx_model = onnx.load(model_path, load_external_data=False)

graph = onnx_model.graph

onnx_model.producer_name = 'yizhu'
onnx_model.producer_version = '1.0'

onnx_model.opset_import[1].domain = 'yznn'
onnx_model.opset_import[1].version = 1
    

for node_id, node in enumerate(graph.node):
    node.domain  = 'yznn'

    if node.op_type == 'SimplifiedLayerNormalization':
        node.op_type = 'RMSNorm'
        
        in_attr = helper.make_attribute('input_is_parallel', (True, False))
        node.attribute.append(in_attr)
        
        parallel_attr = helper.make_attribute('parallel_size', '(4 1 1), (1)', attr_type=onnx.AttributeProto.STRING)
        node.attribute.append(parallel_attr)
        
    
    if node.op_type == 'MatMul':    
        # if '/block/self_attn/q_proj/MatMul' == node.name:
        if 'self_attn/q_proj/MatMul' in node.name:
            in_attr = helper.make_attribute('input_is_parallel', (False, True))
            node.attribute.append(in_attr)
        
            parallel_attr = helper.make_attribute('parallel_size', '(1 1 1), (1 32)')
            node.attribute.append(parallel_attr)
            
            node.op_type = 'LinearQ'
        
        if 'self_attn/k_proj/MatMul' in node.name or 'self_attn/v_proj/MatMul' in node.name:
            in_attr = helper.make_attribute('input_is_parallel', (False, True))
            node.attribute.append(in_attr)
        
            parallel_attr = helper.make_attribute('parallel_size', '(1 1 1), (1 32)')
            node.attribute.append(parallel_attr)
            
            node.op_type = 'LinearKV'
        
        if 'self_attn/o_proj/MatMul' in node.name:
            in_attr = helper.make_attribute('input_is_parallel', (True, True))
            node.attribute.append(in_attr)
        
            parallel_attr = helper.make_attribute('parallel_size', '(1 1 8), (8 4)')
            node.attribute.append(parallel_attr)
            
            node.op_type = 'LinearO'
            
            output_name = node.output

            ccl_output_name = node.name.split('o_proj')[0] + "ccl_output"
            ccl_node_name = node.name.split('o_proj')[0] + "AllReduce_MHA"
            
            ccl_node_insert = helper.make_node("AllReduce", output_name, outputs=[ccl_output_name], name=ccl_node_name, input_is_parallel=(False,), parallel_size='(1 1 1)')
            
            add_node_name = node.name.split('self_attn/o_proj')[0] + "Add"
            for i in range(len(onnx_model.graph.node)):
                if onnx_model.graph.node[i].name == add_node_name:
                    onnx_model.graph.node[i].input[1] = ccl_output_name
                    
            onnx_model.graph.node.append(ccl_node_insert)
            
            value_info = helper.make_tensor_value_info(ccl_output_name, onnx.TensorProto.FLOAT, (16, 1, 8192))
            onnx_model.graph.value_info.extend([value_info])
        
        if 'mlp/up_proj/MatMul' in node.name or 'mlp/gate_proj/MatMul' in node.name:
            in_attr = helper.make_attribute('input_is_parallel', (False, True))
            node.attribute.append(in_attr)
        
            parallel_attr = helper.make_attribute('parallel_size', '(1 1 1), (1 32)')
            node.attribute.append(parallel_attr)
            
            node.op_type = 'LinearUp'
        
        if 'mlp/down_proj/MatMul' in node.name:
            in_attr = helper.make_attribute('input_is_parallel', (True, True))
            node.attribute.append(in_attr)
        
            parallel_attr = helper.make_attribute('parallel_size', '(1 1 8), (8 4)')
            node.attribute.append(parallel_attr)
            
            node.op_type = 'LinearDown'
            
            output_name = node.output

            ccl_output_name = node.name.split('down_proj')[0] + "ccl_output"
            ccl_node_name = node.name.split('down_proj')[0] + "AllReduce_MLP"
            
            ccl_node_insert = helper.make_node("AllReduce", output_name, outputs=[ccl_output_name], name=ccl_node_name, input_is_parallel=(False,), parallel_size='(1 1 1)')
            
            add_node_name = node.name.split('mlp/down_proj')[0] + "Add_1"
            for i in range(len(onnx_model.graph.node)):
                if onnx_model.graph.node[i].name == add_node_name:
                    onnx_model.graph.node[i].input[1] = ccl_output_name
                    
            onnx_model.graph.node.append(ccl_node_insert)
            
            value_info = helper.make_tensor_value_info(ccl_output_name, onnx.TensorProto.FLOAT, (16, 1, 8192))
            onnx_model.graph.value_info.extend([value_info])
        
        if 'self_attn/MatMul' in node.name and 'self_attn/MatMul_1' not in node.name:
            node.op_type = 'MatMulQK'
            
            in_attr = helper.make_attribute('input_is_parallel', (True, True))
            node.attribute.append(in_attr)
        
            parallel_attr = helper.make_attribute('parallel_size', '(4 8 1 1), (4 8 1 1)')
            node.attribute.append(parallel_attr)
            
        if 'self_attn/MatMul_1' in node.name:
            node.op_type = 'MatMulQKV'    
            
            in_attr = helper.make_attribute('input_is_parallel', (True, True))
            node.attribute.append(in_attr)
        
            parallel_attr = helper.make_attribute('parallel_size', '(4 8 1 1), (4 8 1 1)')
            node.attribute.append(parallel_attr)
            
    if node.op_type == 'SiLU':
        in_attr = helper.make_attribute('input_is_parallel', (True,))
        node.attribute.append(in_attr)
        
        parallel_attr = helper.make_attribute('parallel_size', '(1 1 32)')
        node.attribute.append(parallel_attr)
            
    if node.op_type == 'RotaryEmbedding':
        in_attr = helper.make_attribute('input_is_parallel', (True, False, False, False))
        node.attribute.append(in_attr)
        
        parallel_attr = helper.make_attribute('parallel_size', '(4 8 1 1), (1 1), (1 1), (1 1)')
        node.attribute.append(parallel_attr)
        
    if node.op_type == 'Add':
        in_attr = helper.make_attribute('input_is_parallel', (True, True))
        node.attribute.append(in_attr)
        
        parallel_attr = helper.make_attribute('parallel_size', '(4 1 1), (4 1 1)')
        node.attribute.append(parallel_attr)
            
    if node.op_type == 'Mul':
        in_attr = helper.make_attribute('input_is_parallel', (True, True))
        node.attribute.append(in_attr)
        
        parallel_attr = helper.make_attribute('parallel_size', '(1 1 32), (1 1 32)')
        node.attribute.append(parallel_attr)
        
    if node.op_type == 'Div':
        in_attr = helper.make_attribute('input_is_parallel', (True, False))
        node.attribute.append(in_attr)
        
        parallel_attr = helper.make_attribute('parallel_size', '(4 8 1 1), (1)')
        node.attribute.append(parallel_attr)
        
    if node.op_type == 'Softmax':
        in_attr = helper.make_attribute('input_is_parallel', (True,))
        node.attribute.append(in_attr)
        
        parallel_attr = helper.make_attribute('parallel_size', '(4 8 1 1)')
        node.attribute.append(parallel_attr)
    
    if node.op_type == 'Concat':
        in_attr = helper.make_attribute('input_is_parallel', (True, True))
        node.attribute.append(in_attr)
        
        parallel_attr = helper.make_attribute('parallel_size', '(4 8 1 1), (4 8 1 1)')
        node.attribute.append(parallel_attr)
    
    if node.op_type == 'Reshape':
        if 'Reshape_5'in node.name:
            in_attr = helper.make_attribute('input_is_parallel', (True, False))
            node.attribute.append(in_attr)
        
            parallel_attr = helper.make_attribute('parallel_size', '(4 8 1 1), (1)')
            node.attribute.append(parallel_attr)
        else:
            in_attr = helper.make_attribute('input_is_parallel', (True, False))
            node.attribute.append(in_attr)
        
            parallel_attr = helper.make_attribute('parallel_size', '(4 1 8), (1)')
            node.attribute.append(parallel_attr)


output_path = "llama2-70b-final.onnx"

# external_data_path = output_path + ".data"
# onnx.save(onnx_model, output_path, save_as_external_data=True, location=external_data_path)

onnx.save(onnx_model, output_path, save_as_external_data=False)




onnx_model = onnx.load(output_path, load_external_data=False)
        
for node_info in onnx_model.graph.value_info:

    if 'self_attn/Reshape_2_output_0' in node_info.name or 'self_attn/Reshape_1_output_0' in node_info.name:
        node_info.type.tensor_type.shape.dim[1].dim_value = 8
        node_info.type.tensor_type.shape.dim[2].dim_value = 1
    
    if 'self_attn/Reshape_output_0' in node_info.name:    
        node_info.type.tensor_type.shape.dim[1].dim_value = 64
        node_info.type.tensor_type.shape.dim[2].dim_value = 1
    if 'self_attn/MatMul_output_0' in node_info.name or 'self_attn/Div_2_output_0' in node_info.name or 'self_attn/Softmax_output_0' in node_info.name:
        node_info.type.tensor_type.shape.dim[3].dim_param = 'past_seq_len + 1'
        
    node_info.type.tensor_type.shape.dim[0].dim_value = 16

for input in onnx_model.graph.input:
    input.type.tensor_type.shape.dim[0].dim_value = 16
                
for output in onnx_model.graph.output:
    output.type.tensor_type.shape.dim[0].dim_value = 16
            
for init in onnx_model.graph.initializer:
    init_name = init.name
    if 'self_attn/Concat_1_output_0' in init_name or 'self_attn/Concat_2_output_0' in init_name: 
        if len(init.raw_data) > 0:
            shape = bytearray(init.raw_data)
            struct.pack_into('q', shape, 0, 16)
            struct.pack_into('q', shape, 8, 8)
            struct.pack_into('q', shape, 16, 1)
            init.raw_data = bytes(shape)
    
    if 'self_attn/Concat_output_0' in init_name: 
        if len(init.raw_data) > 0:
            shape = bytearray(init.raw_data)
            struct.pack_into('q', shape, 0, 16)
            struct.pack_into('q', shape, 8, 64)
            struct.pack_into('q', shape, 16, 1)
            init.raw_data = bytes(shape)
            
    if 'self_attn/Concat_7_output_0' in init_name:         
        if len(init.raw_data) > 0:
            shape = bytearray(init.raw_data)
            struct.pack_into('q', shape, 0, 16)
            init.raw_data = bytes(shape)
            
onnx.save(onnx_model, "llama2-70b-final-batch-reshape.onnx", save_as_external_data=False)