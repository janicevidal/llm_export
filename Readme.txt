llama2-70b huggingface模型存放路径 /nas/LLM/llama2-70b/


转换单个block  python llm_export.py --export_block 1 --path /nas/LLM/llama2-70b/ --onnx_path llama2-70b-block --type llama2-70b --skip_slim
转换完整模型  python llm_export.py --export --path /nas/LLM/llama2-70b/ --onnx_path llama2-70b-block --type llama2-70b --skip_slim

onnxruntime库在 10.10.70.12:/home/xiaoshuai.zhang/onnxruntime 源码做了修改，直接拷贝这边的文件夹
替换 symbolic_shape_infer.py 到 onnxslim 里面，位置在 /home/xiaoshuai.zhang/anaconda3/envs/yznn/lib/python3.10/site-packages/onnxslim/core/symbolic_shape_infer.py

模版匹配、算子简化 python onnx_fusion_optimize.py

算子手动删减 python onnx_delete_node.py
算子重命名、张量信息添加 python onnx_rename_paralle.py