import sys
import onnx
from onnxslim import slim

sys.path.append('/home/xiaoshuai.zhang/onnxruntime/onnxruntime/python/tools/transformers/')

from fusion_options import FusionOptions
from optimizer import optimize_model
from onnx_model import OnnxModel


temp_path = "/nas/zxs/llama2-70b-block/block_1.onnx"
output_path =temp_path.split('.onnx')[0]+'_opt.onnx'
output_path_fp16 =output_path.split('.onnx')[0]+'_fp16.onnx'


onnx_model = onnx.load_model(temp_path, load_external_data=True)

optimization_options = FusionOptions("gpt2")

optimization_options.enable_layer_norm = True
optimization_options.enable_skip_layer_norm = False

optimization_options.enable_attention = False
optimization_options.enable_rotary_embeddings = True
optimization_options.enable_shape_inference =False

optimization_options.enable_gelu =True


# llama-7b
# model_opt = optimize_model(
#     onnx_model,
#     model_type="gpt2",
#     num_heads=32,
#     hidden_size=4096,
#     opt_level=0,
#     optimization_options=optimization_options,
#     only_onnxruntime=False,
# )

# llama-70b
model_opt = optimize_model(
    onnx_model,
    model_type="gpt2",
    num_heads=64,
    hidden_size=8192,
    opt_level=0,
    optimization_options=optimization_options,
    only_onnxruntime=False,
)

model_opt.save_model_to_file(output_path, use_external_data_format=True)


# convert fp16
model = OnnxModel(onnx.load_model(output_path, load_external_data=True))
model.convert_float_to_float16(keep_io_types=False)
model.save_model_to_file(output_path_fp16, use_external_data_format=True)


# onnx slim
slim(output_path_fp16, output_model=output_path_fp16.split('.onnx')[0]+'_slim.onnx', save_as_external_data=True)