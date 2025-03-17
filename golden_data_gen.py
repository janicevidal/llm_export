import os
import onnxruntime as rt
import numpy as np
import onnx
import torch
import copy
from collections import OrderedDict


# /home/xiaoshuai.zhang/anaconda3/envs/yznn/lib/python3.10/site-packages/onnx/onnx_ml_pb2.pyi
def onnx_datatype_to_npType(data_type):
    if data_type == 1:
        return np.float32
    if data_type == 10:
        return np.float16
    if data_type == 3:
        return np.int8
    if data_type == 5:
        return np.int16
    if data_type == 6:
        return np.int32
    if data_type == 7:
        return np.int64
    else:
        raise TypeError("don't support data type")
    
    
def save_as_tile_layout(input):
    # 获取除了最后一维之外的维度乘积     
    first_dims_product = np.prod(input.shape[:-1])
    # 获取最后一维的大小
    last_dim = input.shape[-1]
    
    if len(input.shape) < 2:
        return input 
    
    if last_dim < 32:
        return input
    
    # 使用 reshape 方法转换数组
    reshaped_data = input.reshape(first_dims_product, last_dim)
    # reshaped_data = input.reshape(firsrt_dim, last_dims_product.astype(int))
    
    out_loop = (int)(reshaped_data.shape[0] / 4)
    inner_loop = (int)(reshaped_data.shape[1] / 32)
    
    blocks = [reshaped_data[i * 4: i * 4 + 4, j * 32:j * 32 + 32].flatten() for i in range(out_loop) for j in range(inner_loop)]     
    result_array = np.concatenate(blocks).astype(np.float16)
    
    return result_array



# Get position_ids from attention_mask
# onnxruntime/onnxruntime/python/tools/transformers/models/llama/llama_inputs.py
def get_position_ids(attention_mask: torch.Tensor, use_past_kv: bool):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    if use_past_kv:
        # Shape: (batch_size, 1)
        position_ids = position_ids[:, -1].unsqueeze(-1)

    # Shape: (batch_size, sequence_length)
    return position_ids

sess_options = rt.SessionOptions()
sess_options.graph_optimization_level = (rt.GraphOptimizationLevel.ORT_DISABLE_ALL)

refer_model_path = '/nas/zxs_onnx/llama2-70b-modified/llama2-70b-block-1-modified.onnx'
refer_onnx_model = onnx.load(refer_model_path)

refer_name_infos = []

for node in refer_onnx_model.graph.node:
    node_name = node.name
    
    refer_name_infos.append(node_name)
    
# print(refer_name_infos)

m_tile = 4
fp16_w_tile = 32

model_path = '/nas/zxs/llama2-70b-block-infer/block_1_opt_slim_fp16.onnx'
# output_folder = '/nas/zxs/dump_tensor_bin_kvlen_2047_n'

output_folder = '/nas/zxs/dump_tensor_bin_kvlen_2047_tile_mod'
# output_folder = '/nas/zxs/dump_tensor_bin_kvlen_5_o'
# output_folder = '/nas/zxs/dump_tensor_bin_kvlen_1_o'

output_folder_tiled = output_folder + '_tiled'

# kv_seq_length = 1
# kv_seq_length = 5
# kv_seq_length = 31
kv_seq_length = 2047

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
if not os.path.exists(output_folder_tiled):
    os.makedirs(output_folder_tiled)

onnx_model = onnx.load(model_path)

layer_infos = {}

for node in onnx_model.graph.node:
    node_output = node.output
    node_name = node.name
    
    layer_infos[node_output[0]] = node_name   

# print(layer_infos)

np.random.seed(0)

model = rt.InferenceSession(model_path)
input_names = [input.name for input in model.get_inputs()]
output_names = [output.name for output in model.get_outputs()]

inputs_embeds = np.random.normal(size=(16, 1, 8192)).astype(np.float16)

attention_mask = torch.ones(16, kv_seq_length + 1, dtype=torch.int64)

position_ids = get_position_ids(attention_mask, use_past_kv=True).numpy()

past_k = np.random.normal(1, 2, size=(16, 8, kv_seq_length, 128)).astype(np.float16)
past_v = np.random.normal(2, 4, size=(16, 8, kv_seq_length, 128)).astype(np.float16)

input = {input_names[0]:inputs_embeds, input_names[1]:position_ids, input_names[2]:past_k, input_names[3]:past_v}


shape_str = str(inputs_embeds.shape[0])

for d in inputs_embeds.shape[1:]:
    shape_str = shape_str + "_" + str(d)
        
type_str = str(inputs_embeds.dtype)
    
output_file_path = f"{output_folder}/inputs_embeds.bin"
inputs_embeds.flatten().tofile(output_file_path)

output_tiled_file_path = f"{output_folder_tiled}/inputs_embeds_tiled.bin"
save_as_tile_layout(inputs_embeds).flatten().tofile(output_tiled_file_path)

shape_str = str(position_ids.shape[0])

for d in position_ids.shape[1:]:
    shape_str = shape_str + "_" + str(d)
        
type_str = str(position_ids.dtype)
    
output_file_path = f"{output_folder}/position_ids.bin"
position_ids.flatten().tofile(output_file_path)

output_tiled_file_path = f"{output_folder_tiled}/position_ids_tiled.bin"
save_as_tile_layout(position_ids).flatten().tofile(output_tiled_file_path)



past_k_1 = np.random.normal(1, 2, size=(16, 8, 1, 128)).astype(np.float16)
past_v_1 = np.random.normal(2, 4, size=(16, 8, 1, 128)).astype(np.float16)


past_k_pad = np.concatenate((past_k, past_k_1), axis=2)
past_v_pad = np.concatenate((past_v, past_v_1), axis=2)
print(past_k_pad.shape)


output_file_path = f"{output_folder}/past_key_values.1.key.bin"
past_k.flatten().tofile(output_file_path)

output_tiled_file_path = f"{output_folder_tiled}/past_key_values.1.key_tiled.bin"

Batch, _, K, N  = past_k_pad.shape
            
golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8 = np.hsplit(past_k_pad, 8)
                 
b_row11 = golden_output1.reshape(Batch, K, N).reshape(1, Batch * N * K)
b_row22 = golden_output2.reshape(Batch, K, N).reshape(1, Batch * N * K)
b_row33 = golden_output3.reshape(Batch, K, N).reshape(1, Batch * N * K)
b_row44 = golden_output4.reshape(Batch, K, N).reshape(1, Batch * N * K)
b_row55 = golden_output5.reshape(Batch, K, N).reshape(1, Batch * N * K)
b_row66 = golden_output6.reshape(Batch, K, N).reshape(1, Batch * N * K)
b_row77 = golden_output7.reshape(Batch, K, N).reshape(1, Batch * N * K)
b_row88 = golden_output8.reshape(Batch, K, N).reshape(1, Batch * N * K)

golden_output_trow = np.concatenate((b_row11, b_row22, b_row33, b_row44, b_row55, b_row66, b_row77, b_row88))
golden_output_trow.flatten().tofile(output_tiled_file_path)
            
            
            
# golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8 = np.split(past_k, 8, axis=1)
    
# M = np.prod(golden_output1.shape[1:])
# N = golden_output1.shape[0] 

# tmp = golden_output1.reshape(N // m_tile, m_tile, M)
# # print("tmp", tmp.shape)
        
# golden_output1 = golden_output1.reshape(N // m_tile, m_tile, M)
# golden_output2 = golden_output2.reshape(N // m_tile, m_tile, M)
# golden_output3 = golden_output3.reshape(N // m_tile, m_tile, M)
# golden_output4 = golden_output4.reshape(N // m_tile, m_tile, M)
# golden_output5 = golden_output5.reshape(N // m_tile, m_tile, M)
# golden_output6 = golden_output6.reshape(N // m_tile, m_tile, M)
# golden_output7 = golden_output7.reshape(N // m_tile, m_tile, M)
# golden_output8 = golden_output8.reshape(N // m_tile, m_tile, M)

# golden_output_trow = np.concatenate((golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8))
# print(golden_output_trow.shape)


output_file_path = f"{output_folder}/past_key_values.1.value.bin"
past_v.flatten().tofile(output_file_path)

output_tiled_file_path = f"{output_folder_tiled}/past_key_values.1.value_tiled.bin"

Batch, _, K, N  = past_v_pad.shape
            
golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8 = np.hsplit(past_v_pad, 8)
                 
b_row11 = golden_output1.reshape(Batch, K, N).reshape(1, Batch * N * K)
b_row22 = golden_output2.reshape(Batch, K, N).reshape(1, Batch * N * K)
b_row33 = golden_output3.reshape(Batch, K, N).reshape(1, Batch * N * K)
b_row44 = golden_output4.reshape(Batch, K, N).reshape(1, Batch * N * K)
b_row55 = golden_output5.reshape(Batch, K, N).reshape(1, Batch * N * K)
b_row66 = golden_output6.reshape(Batch, K, N).reshape(1, Batch * N * K)
b_row77 = golden_output7.reshape(Batch, K, N).reshape(1, Batch * N * K)
b_row88 = golden_output8.reshape(Batch, K, N).reshape(1, Batch * N * K)

golden_output_trow = np.concatenate((b_row11, b_row22, b_row33, b_row44, b_row55, b_row66, b_row77, b_row88))
            
golden_output_trow.flatten().tofile(output_tiled_file_path)


def get_layer_output(model, ort_inputs):
    ori_output = copy.deepcopy(model.graph.output)
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
            
    ort_session = rt.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    outputs_ = [x.name for x in ort_session.get_outputs()]
    
    # outputs = list(set(outputs_))
    outputs = outputs_
    
    ort_outs = ort_session.run(outputs, ort_inputs)

    return outputs, ort_outs


layer_names, output_datas = get_layer_output(onnx_model, input)

for layer_name, output_data in zip(layer_names, output_datas):
    if len(output_data.shape) == 0:
        continue
    
    print(layer_infos[layer_name])
    
    if layer_infos[layer_name] == '/block/self_attn/Transpose' or layer_infos[layer_name] == '/block/self_attn/Transpose_1' or layer_infos[layer_name] == '/block/self_attn/Transpose_2':
        layer_name_ = layer_name.replace('/', '_') 
        layer_name_ = layer_name_.replace('Transpose', 'Reshape-') 
                
        output_file_path = f"{output_folder}/{layer_name_}.bin"
        output_data.flatten().tofile(output_file_path)
        
        golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8 = np.split(output_data, 8, axis=1)
            
        M = np.prod(output_data.shape[:-1])
        N = output_data.shape[-1]
        
        golden_output1 = golden_output1.reshape(int(M/8) // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, int(M/8) * N)
        golden_output2 = golden_output2.reshape(int(M/8) // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, int(M/8) * N)
        golden_output3 = golden_output3.reshape(int(M/8) // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, int(M/8) * N)
        golden_output4 = golden_output4.reshape(int(M/8) // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, int(M/8) * N)
        golden_output5 = golden_output5.reshape(int(M/8) // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, int(M/8) * N)
        golden_output6 = golden_output6.reshape(int(M/8) // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, int(M/8) * N)
        golden_output7 = golden_output7.reshape(int(M/8) // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, int(M/8) * N)
        golden_output8 = golden_output8.reshape(int(M/8) // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, int(M/8) * N)

        golden_output_trow = np.concatenate((golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8))
            
        output_tiled_file_path = f"{output_folder_tiled}/{layer_name_}_tiled.bin"
        golden_output_trow.flatten().tofile(output_tiled_file_path)

        
    if layer_infos[layer_name] == '/block/self_attn/Reshape_7':
        # layer_name_ = '/block/self_attn/Reshape_5'
        print("bingo")
        print(layer_infos[layer_name])
        
        layer_name_ = '/block/self_attn/Reshape_5_output_0'
            
        layer_name_ = layer_name_.replace('/', '_') 
                 
        output_file_path = f"{output_folder}/{layer_name_}.bin"
        output_data.flatten().tofile(output_file_path)  
        
        golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8 = np.split(output_data, 8, axis=2)
            
        M = np.prod(output_data.shape[:-1])
        N = output_data.shape[-1]
        
        golden_output1 = golden_output1.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
        golden_output2 = golden_output2.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
        golden_output3 = golden_output3.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
        golden_output4 = golden_output4.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
        golden_output5 = golden_output5.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
        golden_output6 = golden_output6.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
        golden_output7 = golden_output7.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
        golden_output8 = golden_output8.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))

        golden_output_trow = np.concatenate((golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8))
            
        output_tiled_file_path = f"{output_folder_tiled}/{layer_name_}_tiled.bin"
        golden_output_trow.flatten().tofile(output_tiled_file_path)
                
    
    if layer_infos[layer_name] == '/block/mlp/act_fn/Mul':
        layer_name_ = '/block/mlp/act_fn/Mul_output_0'
        layer_name_ = layer_name_.replace('/', '_')  
                 
        output_file_path = f"{output_folder}/{layer_name_}.bin"
        output_data.flatten().tofile(output_file_path)  
        
        golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8 = np.split(output_data, 8, axis=2)
            
        M = np.prod(output_data.shape[:-1])
        N = output_data.shape[-1]
        
        golden_output1 = golden_output1.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
        golden_output2 = golden_output2.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
        golden_output3 = golden_output3.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
        golden_output4 = golden_output4.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
        golden_output5 = golden_output5.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
        golden_output6 = golden_output6.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
        golden_output7 = golden_output7.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
        golden_output8 = golden_output8.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))

        golden_output_trow = np.concatenate((golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8))
            
        output_tiled_file_path = f"{output_folder_tiled}/{layer_name_}_tiled.bin"
        golden_output_trow.flatten().tofile(output_tiled_file_path)
        
    
    #################    
    if layer_infos[layer_name] in refer_name_infos:

        if layer_infos[layer_name] in ('/block/self_attn/q_proj/MatMul', '/block/self_attn/k_proj/MatMul', '/block/self_attn/v_proj/MatMul', '/block/mlp/gate_proj/MatMul', '/block/mlp/up_proj/MatMul', '/block/mlp/Mul'):
            layer_name_ = layer_name.replace('/', '_')
                
            output_file_path = f"{output_folder}/{layer_name_}.bin"
            output_data.flatten().tofile(output_file_path)
        
            golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8 = np.split(output_data, 8, axis=2)
            
            M = np.prod(output_data.shape[:-1])
            N = output_data.shape[-1]
        
            golden_output1 = golden_output1.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
            golden_output2 = golden_output2.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
            golden_output3 = golden_output3.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
            golden_output4 = golden_output4.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
            golden_output5 = golden_output5.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
            golden_output6 = golden_output6.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
            golden_output7 = golden_output7.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))
            golden_output8 = golden_output8.reshape(M // m_tile, m_tile, int(N/8) // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * int(N/8))

            golden_output_trow = np.concatenate((golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8))
            
            output_tiled_file_path = f"{output_folder_tiled}/{layer_name_}_tiled.bin"
            golden_output_trow.flatten().tofile(output_tiled_file_path)
        
        elif layer_infos[layer_name] in ('RotaryEmbedding_0', 'RotaryEmbedding_1', '/block/self_attn/MatMul', '/block/self_attn/Div_2', '/block/self_attn/Softmax', '/block/self_attn/MatMul_1'):
            layer_name_ = layer_name.replace('/', '_')
                
            output_file_path = f"{output_folder}/{layer_name_}.bin"
            output_data.flatten().tofile(output_file_path)
            
            golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8 = np.split(output_data, 8, axis=1)
            
            M = np.prod(output_data.shape[:-1])
            N = output_data.shape[-1]
            
            # print(output_data.shape)
            # print(layer_name_)
            # print(M, N)
        
            golden_output1 = golden_output1.reshape(int(M/8) // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, int(M/8) * N)
            golden_output2 = golden_output2.reshape(int(M/8) // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, int(M/8) * N)
            golden_output3 = golden_output3.reshape(int(M/8) // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, int(M/8) * N)
            golden_output4 = golden_output4.reshape(int(M/8) // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, int(M/8) * N)
            golden_output5 = golden_output5.reshape(int(M/8) // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, int(M/8) * N)
            golden_output6 = golden_output6.reshape(int(M/8) // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, int(M/8) * N)
            golden_output7 = golden_output7.reshape(int(M/8) // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, int(M/8) * N)
            golden_output8 = golden_output8.reshape(int(M/8) // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, int(M/8) * N)

            golden_output_trow = np.concatenate((golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8))
            
            output_tiled_file_path = f"{output_folder_tiled}/{layer_name_}_tiled.bin"
            golden_output_trow.flatten().tofile(output_tiled_file_path)
            
        elif layer_infos[layer_name] in ('/block/self_attn/Concat_5', '/block/self_attn/Concat_6'):
            layer_name_ = layer_name.replace('/', '_')
            print(layer_infos[layer_name], layer_name_)
            
            output_data_ = output_data.copy()
                
            output_file_path = f"{output_folder}/{layer_name_}.bin"
            output_data.flatten().tofile(output_file_path)
            
            Batch, _, K, N  = output_data_.shape
            
            golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8 = np.hsplit(output_data_, 8)
            print("golden_output1", golden_output1.shape)
            
            
            b_row11 = golden_output1.reshape(Batch, K, N).reshape(1, Batch * N * K)
            b_row22 = golden_output2.reshape(Batch, K, N).reshape(1, Batch * N * K)
            b_row33 = golden_output3.reshape(Batch, K, N).reshape(1, Batch * N * K)
            b_row44 = golden_output4.reshape(Batch, K, N).reshape(1, Batch * N * K)
            b_row55 = golden_output5.reshape(Batch, K, N).reshape(1, Batch * N * K)
            b_row66 = golden_output6.reshape(Batch, K, N).reshape(1, Batch * N * K)
            b_row77 = golden_output7.reshape(Batch, K, N).reshape(1, Batch * N * K)
            b_row88 = golden_output8.reshape(Batch, K, N).reshape(1, Batch * N * K)
            
            print("1111 b_row11.shape:", b_row11.shape)
            golden_output_trow = np.concatenate((b_row11, b_row22, b_row33, b_row44, b_row55, b_row66, b_row77, b_row88))

            
            # M = np.prod(golden_output1.shape[1:])
            # N = golden_output1.shape[0] 
                  
            # golden_output1 = golden_output1.reshape(N // m_tile, m_tile, M)
            # golden_output2 = golden_output2.reshape(N // m_tile, m_tile, M)
            # golden_output3 = golden_output3.reshape(N // m_tile, m_tile, M)
            # golden_output4 = golden_output4.reshape(N // m_tile, m_tile, M)
            # golden_output5 = golden_output5.reshape(N // m_tile, m_tile, M)
            # golden_output6 = golden_output6.reshape(N // m_tile, m_tile, M)
            # golden_output7 = golden_output7.reshape(N // m_tile, m_tile, M)
            # golden_output8 = golden_output8.reshape(N // m_tile, m_tile, M)

            # golden_output_trow = np.concatenate((golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8))
            # print("golden_output_trow", golden_output_trow.shape)
            
            output_tiled_file_path = f"{output_folder_tiled}/{layer_name_}_tiled.bin"
            golden_output_trow.flatten().tofile(output_tiled_file_path)
            
        else:
            # print(layer_infos[layer_name])
            
            output_data_ = output_data.copy()
            
            layer_name_ = layer_name.replace('/', '_') 
        
            output_file_path = f"{output_folder}/{layer_name_}.bin"
            output_data_.flatten().tofile(output_file_path)
            
            output_tiled_file_path = f"{output_folder_tiled}/{layer_name_}_tiled.bin"
            save_as_tile_layout(output_data_).flatten().tofile(output_tiled_file_path)
            
            
            if layer_infos[layer_name] == '/block/self_attn/o_proj/MatMul':
                layer_name_ = layer_name.replace('/', '__')
                
                for item in onnx_model.graph.initializer:
                    if item.name == "onnx::MatMul_344":
                        
                        # print("shape: ", item.dims)
                        weight = np.frombuffer(item.raw_data, dtype=onnx_datatype_to_npType(item.data_type)).reshape(*item.dims)
                        # print("weight shape: ", weight.shape)
                        
                        weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8 = np.split(weight, 8, axis=0)
                        
                        pre_layer_name_ = '/block/self_attn/Reshape_5_output_0'
                        pre_layer_name_ = pre_layer_name_.replace('/', '_') 
                 
                        input_file_path = f"{output_folder}/{pre_layer_name_}.bin"
                        
                        input = np.fromfile(input_file_path, dtype=np.float16).reshape(16, 1, 8192)
                        
                        input1, input2, input3, input4, input5, input6, input7, input8 = np.split(input, 8, axis=2)
                        
                        output1 = np.matmul(input1, weight1)
                        output2 = np.matmul(input2, weight2)
                        output3 = np.matmul(input3, weight3)
                        output4 = np.matmul(input4, weight4)
                        output5 = np.matmul(input5, weight5)
                        output6 = np.matmul(input6, weight6)
                        output7 = np.matmul(input7, weight7)
                        output8 = np.matmul(input8, weight8)
                        
                        M = np.prod(output1.shape[:-1])
                        N = output1.shape[-1]
                        
                        golden_output1 = output1.reshape(M // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * N)
                        golden_output2 = output2.reshape(M // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * N)
                        golden_output3 = output3.reshape(M // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * N)
                        golden_output4 = output4.reshape(M // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * N)
                        golden_output5 = output5.reshape(M // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * N)
                        golden_output6 = output6.reshape(M // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * N)
                        golden_output7 = output7.reshape(M // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * N)
                        golden_output8 = output8.reshape(M // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * N)
                        
                        golden_output_trow = np.concatenate((golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8))
            
                        output_tiled_file_path = f"{output_folder_tiled}/{layer_name_}_tiled.bin"
                        golden_output_trow.flatten().tofile(output_tiled_file_path)
                        
            elif layer_infos[layer_name] == '/block/mlp/down_proj/MatMul':
                layer_name_ = layer_name.replace('/', '__')
                
                for item in onnx_model.graph.initializer:
                    if item.name == "onnx::MatMul_347":
                        
                        # print("shape: ", item.dims)
                        weight = np.frombuffer(item.raw_data, dtype=onnx_datatype_to_npType(item.data_type)).reshape(*item.dims)
                        # print("weight shape: ", weight.shape)
                        
                        weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8 = np.split(weight, 8, axis=0)
                        
                        pre_layer_name_ = '/block/mlp/Mul_output_0'
                        pre_layer_name_ = pre_layer_name_.replace('/', '_') 
                 
                        input_file_path = f"{output_folder}/{pre_layer_name_}.bin"
                        
                        input = np.fromfile(input_file_path, dtype=np.float16).reshape(16, 1, 28672)
                        
                        input1, input2, input3, input4, input5, input6, input7, input8 = np.split(input, 8, axis=2)
                        
                        output1 = np.matmul(input1, weight1)
                        output2 = np.matmul(input2, weight2)
                        output3 = np.matmul(input3, weight3)
                        output4 = np.matmul(input4, weight4)
                        output5 = np.matmul(input5, weight5)
                        output6 = np.matmul(input6, weight6)
                        output7 = np.matmul(input7, weight7)
                        output8 = np.matmul(input8, weight8)
                        
                        M = np.prod(output1.shape[:-1])
                        N = output1.shape[-1]
                        
                        golden_output1 = output1.reshape(M // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * N)
                        golden_output2 = output2.reshape(M // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * N)
                        golden_output3 = output3.reshape(M // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * N)
                        golden_output4 = output4.reshape(M // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * N)
                        golden_output5 = output5.reshape(M // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * N)
                        golden_output6 = output6.reshape(M // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * N)
                        golden_output7 = output7.reshape(M // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * N)
                        golden_output8 = output8.reshape(M // m_tile, m_tile, N // fp16_w_tile, fp16_w_tile).transpose(0, 2, 1, 3).reshape(1, M * N)
                        
                        golden_output_trow = np.concatenate((golden_output1, golden_output2, golden_output3, golden_output4, golden_output5, golden_output6, golden_output7, golden_output8))
            
                        output_tiled_file_path = f"{output_folder_tiled}/{layer_name_}_tiled.bin"
                        golden_output_trow.flatten().tofile(output_tiled_file_path)    