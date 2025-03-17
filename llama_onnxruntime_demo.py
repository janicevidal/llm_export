# -*- coding: utf-8 -*-
import os

import numpy as np
from transformers import AutoConfig, AutoTokenizer

import time
import onnxruntime as rt


class OnnxRunner:
    def __init__(self, model_path, idx):
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = (
            rt.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        
        provider_options = {'device_id': '4'} 
        
        if idx > 16:
            provider_options = {'device_id': '5'}
            
        providers = [("CUDAExecutionProvider", provider_options)]
        
        # self.model = rt.InferenceSession(model_path, providers=providers)
        self.model = rt.InferenceSession(model_path)
        
        self.input_names = [input.name for input in self.model.get_inputs()]
        self.output_names = [output.name for output in self.model.get_outputs()]

    def __call__(self, *args):
        kwargs = {param: arg for param, arg in zip(self.input_names, args)}
        output = self.model.run(self.output_names, kwargs)
        if len(self.output_names) == 1:
            return output[0]

        return output
    
    
class LlamaForCausalLM:
    def __init__(self, model_path, max_length, **kwargs):
        self.context_len = 0
        self.token_len = 0
        self.max_length = max_length
        
        self.config = AutoConfig.from_pretrained(model_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        
        self.past_kv_shape = [2, 1, 0, self.config.num_key_value_heads, self.config.hidden_size// self.config.num_attention_heads]
        
        self.load(model_path)

    def load_module(self, path, idx):
        return OnnxRunner(path, idx)
    
    def load(self, model_path):
        self.block_nums = self.config.num_hidden_layers
        self.lm = self.load_module(os.path.join(model_path, 'lm.onnx'), 1)
        self.embed = self.load_module(os.path.join(model_path, 'embedding.onnx'), 0)
        self.blocks = []
        for i in range(self.block_nums):
            self.blocks.append(self.load_module(os.path.join(model_path, f'block_{i}.onnx'), i))

    def get_attention_mask(self):
        if self.token_len:
            return np.zeros((1, 1, 1, self.seq_len), dtype=np.float32)
        return (1 - np.tril(np.ones((1, 1, self.seq_len, self.seq_len), dtype=np.float32))) * np.finfo(np.float32).min

    def get_position_ids(self):
        if self.token_len:
            return np.array([[self.seq_len - 1]], dtype=np.int64)
        return np.array([np.arange(self.seq_len, dtype=np.int64)])

    def stop_id(self):
        return self.tokenizer.eos_token_id

    def build_prompt(self, query):
        if hasattr(self.tokenizer, 'build_prompt'):
            prompt = self.tokenizer.build_prompt(query)
        else:
            prompt = f'<s>Human: {query}\n</s><s>Assistant: '
            
        # print(prompt)
        return prompt

    def str_to_ids(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids']
        input_ids = input_ids.numpy()
        return input_ids

    def id_to_str(self, token_id):
        word = self.tokenizer._convert_id_to_token(int(token_id))
        word = self.tokenizer.convert_tokens_to_string([word])
        return word

    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        hidden_states = self.embed(input_ids)
        presents = []
        for i in range(self.block_nums):
            start_time = time.time()
            
            hidden_states, kv = self.blocks[i](hidden_states, attention_mask, position_ids, past_key_values[i])
            
            print(kv.shape)
            
            hidden_states = hidden_states.reshape(-1, 1, 4096)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            # print(f"运行时间 {i}-decoder : {elapsed_time}秒")

            presents.append(kv)
            
        hidden_states = hidden_states.squeeze(1)    
        logits = self.lm(hidden_states)
        token_id = np.argmax(logits)
        
        self.seq_len += 1
        self.token_len += 1
        return np.array([token_id]), presents

    def response(self, query, stream = False):
        prompt = self.build_prompt(query)
        input_ids = self.str_to_ids(prompt)
        self.seq_len = input_ids.size
        self.context_len = self.seq_len - 2
        self.token_len = 0
        past_key_values = [np.zeros(self.past_kv_shape, dtype=np.float32) for i in range(self.block_nums)]
        # token_id = input_ids
        token_id = input_ids.squeeze(0)
        res = ''
        
        tokens = []
        skip = 0
        REPLACEMENT_CHAR = '\ufffd'
        
        while self.token_len < self.max_length:
            print(f"------  token: {self.token_len}  ------")
            attention_mask = self.get_attention_mask()
            position_ids = self.get_position_ids()
            token_id, past_key_values = self.forward(token_id, attention_mask, position_ids, past_key_values)
            
            tokens.append(token_id[0])
            
            if token_id == self.stop_id():
                if stream:  print("", end='\n')
                res += '\n'
                break
            
            s = tokenizer.decode(tokens)
            if REPLACEMENT_CHAR not in s:               
                res = res + s[skip:]
                skip = len(s)
                print(res)

        return res

    def chat(self, tokenizer, query, history = None):
        self.tokenizer = tokenizer
        return self.response(query)

    def stream_chat(self, tokenizer, query, history = None):
        self.tokenizer = tokenizer
        return self.response(query, True), None

    def eval(self):
        pass

    def generate(self, **kwargs):
        input_ids = kwargs['input_ids'].numpy()
        self.seq_len = input_ids.size
        self.context_len = self.seq_len - 2
        self.token_len = 0
        past_key_values = [np.zeros(self.past_kv_shape, dtype=np.float32) for i in range(self.block_nums)]
        token_id = input_ids
        res = ''
        while self.token_len < self.max_length:
            attention_mask = self.get_attention_mask()
            position_ids = self.get_position_ids()
            token_id, past_key_values = self.forward(token_id, attention_mask, position_ids, past_key_values)
            if token_id == self.stop_id():
                res += '\n'
                break
            word = self.id_to_str(token_id)
            print("word", word, token_id)
            res += word
        return res
    
    
if __name__ == '__main__':
    # model_path = "/nas/zxs_onnx/Llama-Model-2"
    model_path = "/nas/zxs_onnx/Llama-Model-Chat-Chinese"
    # model_path = "/nas/zxs_onnx/Llama-2-7b-chat-hf-onnx"
    
    # prompt = '请问中国的首都是哪一个城市？'
    # prompt = '请介绍一下苏州亿铸科技 '
    prompt = '请介绍一下苏州市'
    max_length = 1024
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = LlamaForCausalLM(model_path, max_length, trust_remote_code=True)

    output = model.chat(tokenizer, prompt)
    
    print('\n')
    print(output)