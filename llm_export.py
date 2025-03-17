import os
import glob
import json
import shutil
import argparse
import torch
import onnx
import numpy as np
from onnxslim import slim
from itertools import chain
import onnxruntime as ort
import sentencepiece as spm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def save_onnx_model(onnx_model, output_path, data_path):
    onnx.save(
        onnx_model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path,
        size_threshold=1024,
        convert_attribute=False,
    )
    
# some wrapper class for export
class Embedding(torch.nn.Module):
    def __init__(self, embed, using_bf16: bool = False):
        super().__init__()
        self.bf16 = using_bf16
        self.embed_dim = embed.weight.shape[-1]
        if using_bf16:
            # using bf16 embedding weight
            self.embed = embed.bfloat16()
        else:
            self.embed = embed

    def forward(self, input_ids):
        res = self.embed(input_ids)
        if self.bf16:
            res = res.float()
        return res.view(-1, 1, self.embed_dim)

class Lm(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm 

    def forward(self, hidden_states):
        m_logits = self.lm(hidden_states)
        # token = torch.argmax(m_logits)
        return m_logits

class LLM(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.quant_bit = 4
        self.asymmetric = True
        self.onnx_path = args.onnx_path
        if not os.path.exists(self.onnx_path):
            os.makedirs(self.onnx_path)
        # default is False, just set True when using below command:
        # `python llm_export ../path --export --embed_bin` to export single model without embedding
        self.without_embed = False

        self.skip_slim = args.skip_slim
        tokenizer_model = os.path.join(args.path, 'tokenizer.model')
        ice_text_model = os.path.join(args.path, 'ice_text.model')
        try:
            if os.path.exists(tokenizer_model):
                self.sp_model = spm.SentencePieceProcessor(tokenizer_model)
            elif os.path.exists(ice_text_model):
                self.sp_model = spm.SentencePieceProcessor(ice_text_model)
            else:
                self.sp_model = None
        except:
            self.sp_model = None
        merge_file = os.path.join(args.path, 'merges.txt')
        if os.path.exists(merge_file):
            self.merge_txt = merge_file
        else:
            self.merge_txt = None
        self.stop_ids = []
        self.max_length = 1024
        self.hidden_size = 4096
        self.visual = None # defualt is not visual
        self.load_hf(args.path)
        self.load_model()

    def load_hf(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).float().eval()
        except:
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float().eval()
        self.config = self.model.config

    def load_model(self):
        raise NotImplementedError

    def get_attention_mask(self) -> torch.Tensor:
        raise NotImplementedError

    def get_position_ids(self) -> torch.Tensor:
        raise NotImplementedError

    def export_vocab(self):
        raise NotImplementedError

    def visual_embed(self, input_ids):
        raise NotImplementedError

    def __embedding(self, input_ids):
        if self.visual is not None and self.token_len == 0:
            input_embeds = self.visual_embed(input_ids)
        else:
            input_embeds = self.embed(input_ids)
        return input_embeds
    
    def __decode(self, hidden_states, attention_mask, position_ids, past_key_values):
        presents = []
        for i in range(self.block_nums):
            hidden_states, kv = self.blocks[i](hidden_states, attention_mask, position_ids, past_key_values[i])
            presents.append(kv)
        # presents = torch.stack(presents)
        self.seq_len += 1
        self.token_len += 1
        return hidden_states, presents

    def forward(self, inputs_embeds, attention_mask, position_ids, past_key_values):
        return self.__decode(inputs_embeds, attention_mask, position_ids, past_key_values)

    def build_prompt(self, query):
        if hasattr(self.tokenizer, 'build_prompt'):
            prompt = self.tokenizer.build_prompt(query)
        else:
            prompt = query
        return prompt

    def str_to_ids(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids']
        return input_ids

    def id_to_str(self, token_id):
        word = self.tokenizer._convert_id_to_token(int(token_id))
        word = self.tokenizer.convert_tokens_to_string([word])
        return word

    def response(self, query):
        prompt = self.build_prompt(query)
        input_ids = self.str_to_ids(prompt)
        self.seq_len = input_ids.numel()
        self.context_len = self.seq_len - 2
        self.token_len = 0
        past_key_values = [None for i in range(self.block_nums)]
        token_id = input_ids
        while self.token_len < self.max_length:
            attention_mask = self.get_attention_mask()
            position_ids = self.get_position_ids()
            logits, past_key_values = self.forward(token_id, attention_mask, position_ids, past_key_values)
            token_id = torch.argmax(logits)
            if token_id in self.stop_ids:
                print("", end='\n')
                break
            word = self.id_to_str(token_id)
            print(word, end="", flush=True)

    # some export functions
    def assert_equal(self, torch_outs, onnx_outs):
        if type(torch_outs) not in (list, tuple):
            torch_outs = (torch_outs, )
            onnx_outs = (onnx_outs, )
        same = True
        for orig, onnx in zip(torch_outs, onnx_outs):
            orig = orig.detach().numpy()
            if not np.allclose(orig, onnx, rtol=1e-3, atol=1e-3):
                print('Error: onnx outputs dont match original. [shape = {}] onnx: {}, original: {}'.format(onnx.shape, onnx, orig))
                same = False
                break
        if same:
            print('onnx test SUCCESS')

    def export_lm(self):
        model = self.lm
        hidden_states = torch.randn(1, self.hidden_size)
        onnx_model = f'./{self.onnx_path}/lm.onnx'
        torch.onnx.export(model, (hidden_states),
                        onnx_model,
                        input_names=['hidden_states'],
                        output_names=['logits'],
                        do_constant_folding=True,
                        opset_version=15)
        if not self.skip_slim:
            slim(onnx_model, output_model=onnx_model)
        # test lm
        if self.export_test:
            original_outs = model(hidden_states)
            ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
            inputs = {
                'hidden_states' : hidden_states.numpy(),
            }
            onnx_outs = ort_session.run(None, inputs)
            self.assert_equal(original_outs, onnx_outs)

    def export_visual(self):
        if self.visual is None:
            return
        input_images = torch.randn((1, 3, self.image_size, self.image_size))
        model = self.visual
        onnx_model = f'./{self.onnx_path}/visual.onnx'
        torch.onnx.export(model, (input_images),
                        onnx_model,
                        input_names=['input_images'],
                        output_names=['image_embeds'],
                        dynamic_axes={"input_images": {
                            0: "size"
                        }},
                        do_constant_folding=True,
                        opset_version=15)
        if not self.skip_slim:
            slim(onnx_model, output_model=onnx_model)
        # test
        if self.export_test:
            original_outs = model(input_images)
            ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
            inputs = {
                'input_images' : input_images.numpy(),
            }
            onnx_outs = ort_session.run(None, inputs)[0]
            self.assert_equal(original_outs, onnx_outs)

    def export_embed(self):
        model = self.embed
        if self.embed_bin:
            import ctypes
            tensor_data = model.embed.weight.data
            data_ptr = tensor_data.untyped_storage().data_ptr()
            buffer = (ctypes.c_byte * (tensor_data.numel() * 2)).from_address(data_ptr)
            with open(f'./{self.onnx_path}/embeddings_bf16.bin', 'wb') as f:
                f.write(buffer)
            return
        input_ids = torch.arange(3, dtype=torch.long)
        onnx_model = f'./{self.onnx_path}/embedding.onnx'
        torch.onnx.export(model, (input_ids),
                        onnx_model,
                        input_names=['input_ids'],
                        output_names=['inputs_embeds'],
                        dynamic_axes={"input_ids": {
                            0: "length"
                        }},
                        do_constant_folding=True,
                        opset_version=15)
        if not self.skip_slim:
            slim(onnx_model, output_model=onnx_model)
        # test
        if self.export_test:
            original_outs = model(input_ids)
            ort_session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
            inputs = {
                'input_ids' : input_ids.numpy(),
            }
            onnx_outs = ort_session.run(None, inputs)
            self.assert_equal(original_outs, onnx_outs)

    def export_block(self, block_id: int):
        self.seq_len = 1
        self.token_len = 1
        
        batch_size = 1
        
        inputs_embeds = torch.randn(batch_size, 1, self.hidden_size, dtype=torch.float32)
        position_ids = self.get_position_ids()
        
        model = self.blocks[block_id]
        
        if not os.path.exists(self.onnx_path):
            os.makedirs(self.onnx_path)
        
        onnx_model_path = f'./{self.onnx_path}/block_{block_id}.onnx'
        
        past_kv = [
                torch.rand(batch_size, 8, 0, self.head_dim, dtype=torch.float32),
                torch.rand(batch_size, 8, 0, self.head_dim, dtype=torch.float32)
        ]
        
        past_key_values = past_kv
        
        input_names = [
            "inputs_embeds",
            "position_ids",
            *list((f"past_key_values.{block_id}.key", f"past_key_values.{block_id}.value"))
        ]
        
        output_names = [
            "hidden_states",
            *list((f"present.{block_id}.key", f"present.{block_id}.value"))
        ]
        
        dynamic_axes = self.get_merged_model_dynamic_axes(input_names, output_names)
        
        temp_dir = "./temp_onnx"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        temp_path = os.path.join(temp_dir, "temp.onnx")
        
        torch.onnx.export(
            model, (inputs_embeds, None, position_ids, past_key_values),  
            temp_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=14)
        
        onnx_model = onnx.load_model(temp_path, load_external_data=True)
        
        location = f"{os.path.basename(onnx_model_path)}.data"
        if os.path.exists(location):
            os.remove(location)
                
        save_onnx_model(onnx_model, onnx_model_path, location)
        
        del onnx_model
        shutil.rmtree(temp_dir)
    
        if not self.skip_slim:
            slim(onnx_model_path, output_model=onnx_model_path.split('.onnx')[0]+'_slim.onnx')

    def export_blocks(self):
        for i in range(self.block_nums):
            self.export_block(i)

    def export(self):
        model = self
        
        self.seq_len = 1
        self.token_len = 1
        
        batch_size = 1
        
        inputs_embeds = torch.randn(batch_size, 1, self.hidden_size, dtype=torch.float32)
        position_ids = self.get_position_ids()
        
        past_kv = [
            (
                torch.rand(batch_size, 8, 0, self.head_dim, dtype=torch.float32),
                torch.rand(batch_size, 8, 0, self.head_dim, dtype=torch.float32)
            )
            for _ in range(self.block_nums)
        ]
        
        past_key_values = list(map(lambda kv: (kv[0], kv[1]), past_kv))
            
        input_names = [
            "inputs_embeds",
            "position_ids",
            *list(
                chain.from_iterable(
                    (f"past_key_values.{i}.key", f"past_key_values.{i}.value") for i in range(self.block_nums)
                )
            ),
        ]
        
        output_names = [
            "hidden_states",
            *list(
                chain.from_iterable((f"present.{i}.key", f"present.{i}.value") for i in range(self.block_nums))
            ),
        ]
        
        dynamic_axes = self.get_merged_model_dynamic_axes(input_names, output_names)
        
        # dynamic_axes = {}
        # for name in input_names + output_names:
        #     if name in {"inputs_embeds", "position_ids"}:
        #         dynamic_axes[name] = {0: "batch_size" }
        #     elif "past" in name:
        #         dynamic_axes[name] = {0: "batch_size", 2: "past_seq_len"}
        #     elif name == "hidden_states":
        #         dynamic_axes[name] = {0: "batch_size" }
        #     elif "present" in name:
        #         dynamic_axes[name] = {0: "batch_size", 2: "past_seq_len + 1"}

        temp_dir = "./temp_onnx"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        temp_path = os.path.join(temp_dir, "temp.onnx")
        
        if not os.path.exists(self.onnx_path):
            os.makedirs(self.onnx_path)
            
        onnx_model_path = f'./{self.onnx_path}/llm.onnx'

        print('export start ...')
        torch.onnx.export(
            model, (inputs_embeds, None, position_ids, past_key_values),
            temp_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=14)
        
        print('export done!')
        
        onnx_model = onnx.load_model(temp_path, load_external_data=True)
        
        location = f"{os.path.basename(onnx_model_path)}.data"
        if os.path.exists(location):
            os.remove(location)
                
        save_onnx_model(onnx_model, onnx_model_path, location)
        
        del onnx_model
        shutil.rmtree(temp_dir)
        
        if not self.skip_slim:
            slim(onnx_model_path, output_model=onnx_model_path.split('.onnx')[0]+'_slim.onnx')

class LLAMA2Block(torch.nn.Module):
    def __init__(self, block, block_id, hidden_size, head_dim, final_layernorm = None):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.head_dim = head_dim
        self.final_layernorm = final_layernorm
        self.hidden_size = hidden_size

    def forward(self, hidden_states, attention_mask, position_ids, past_kv):
        position_ids = position_ids.view(1, -1)
        hidden_states, presents = self.block(hidden_states,
                                             attention_mask,
                                             position_ids,
                                             past_kv,
                                            #  rotary_pos_emb=rotary_pos_emb,
                                             use_cache=True)
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # hidden_states = hidden_states.view(-1, self.hidden_size)[-1].view(1, 1, self.hidden_size)
        return hidden_states, presents

class Llama2_70b(LLM):
    def __init__(self, args):     
        super().__init__(args)

    def load_model(self):
        self.config = self.model.config
        transformer = self.model.model
        self.lm_ = self.model.lm_head
        self.embed_ = transformer.embed_tokens
        self.blocks_ = transformer.layers
        self.final_layernorm_ = transformer.norm
        # some wrapper
        self.hidden_size = self.embed_.weight.shape[-1]
        self.stop_ids.append(self.tokenizer.eos_token_id)
        if hasattr(self.model, 'generation_config'):
            self.stop_ids.append(self.model.generation_config.eos_token_id)
        self.block_nums = len(self.blocks_)
        self.embed = Embedding(self.embed_)
        self.lm = Lm(self.lm_)
        self.block_nums = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        self.num_attention_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        if hasattr(self.config, 'num_key_value_heads'):
            self.num_key_value_heads = self.config.num_key_value_heads
        else:
            self.num_key_value_heads = self.config.num_attention_heads
        
        self.blocks = [LLAMA2Block(self.blocks_[i], i, self.hidden_size, self.head_dim, self.final_layernorm_ if i == len(self.blocks_) - 1 else None) for i in range(self.block_nums)]
        self.past_kv_shape = [self.block_nums, 2, 1, 0, self.num_key_value_heads, self.head_dim]
        
        self.block_dynamic_axes = {
            "inputs_embeds" : { 0: "batch_size" },
            "position_ids" : { 0: "batch_size" },
            "past_key_values.1.key" : { 0: "batch_size", 2: "past_seq_len"},
            "past_key_values.1.value" : { 0: "batch_size", 2: "past_seq_len"},
            "hidden_states" : { 0: "batch_size" },
            "present.1.key" : { 0: "batch_size", 2: "past_seq_len + 1"},
            "present.1.value" : { 0: "batch_size", 2: "past_seq_len + 1"},
        }
        
    def get_merged_model_dynamic_axes(self, input_names, output_names):
        dynamic_axes = {}
        for name in input_names + output_names:
            if name in {"inputs_embeds", "position_ids"}:
                dynamic_axes[name] = {0: "batch_size" }
            elif "past" in name:
                dynamic_axes[name] = {0: "batch_size", 2: "past_seq_len"}
            elif name == "hidden_states":
                dynamic_axes[name] = {0: "batch_size" }
            elif "present" in name:
                dynamic_axes[name] = {0: "batch_size", 2: "past_seq_len + 1"}
            else:
                raise Exception("Unknown input or output name found")
        return dynamic_axes

    def build_prompt(self, query):
        return f'<s>[INST]{query}[/INST]'

    def get_attention_mask(self) -> torch.Tensor:
        if self.token_len:
            return torch.zeros([1, 1, 1, 1], dtype=torch.float32)
        return (1 - torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]))) * torch.finfo(torch.float32).min

    def get_position_ids(self) -> torch.Tensor:
        if self.token_len:
            return torch.zeros([1, self.seq_len], dtype=torch.long)
        return torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--path', type=str, default='THUDM/chatglm-6b', required=True,
                        help='path(`str` or `os.PathLike`):\nCan be either:'
                        '\n\t- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]'
                        '\n\t- A path to a *directory* clone from repo like `../chatglm-6b`.')
    parser.add_argument('--type', type=str, default=None, help='type(`str`, *optional*):'
                        '\n\tThe pretrain llm model type.'
                        )
    parser.add_argument('--onnx_path', type=str, default='./onnx', help='export onnx model path, defaut is `./onnx`.')
    parser.add_argument('--export', action='store_true', help='export model to an `onnx` model.')
    parser.add_argument('--export_split', action='store_true',
                        help='export model split to some `onnx` models:'
                        '\n\t- embedding model.'
                        '\n\t- block models.'
                        '\n\t- lm_head model.'
                        )
    parser.add_argument('--export_embed', action='store_true', help='export llm embedding to an `onnx` model.')
    parser.add_argument('--export_visual', action='store_true', help='export llm visual model to an `onnx` model.')
    parser.add_argument('--export_lm', action='store_true', help='export llm lm_head to an `onnx` model.')
    parser.add_argument('--export_block', type=int, help='export llm block [id] to an `onnx` model.')
    parser.add_argument('--export_blocks', action='store_true', help='export llm all blocks to `onnx` models.')
    parser.add_argument('--skip_slim', action='store_true', help='Whether or not to skip onnx-slim.')


    args = parser.parse_args()
    model_path = args.path
    model_type = args.type
    
    if model_type is None:
        raise RuntimeError('Please specify model type.')
    
    for file in glob.glob(f'./{model_type}/*'):
        shutil.copy2(file, model_path)

    llm_exporter = Llama2_70b(args)

    if args.export or args.export_split:
        llm_exporter.export_config(args.export)

    if args.export:
        llm_exporter.export()

    if args.export_embed or args.export_split:
        llm_exporter.export_embed()

    if args.export_visual or args.export_split:
        llm_exporter.export_visual()

    if args.export_lm or args.export_split:
        llm_exporter.export_lm()

    if args.export_blocks or args.export_split:
        llm_exporter.export_blocks()

    if args.export_block is not None:
        llm_exporter.export_block(args.export_block)