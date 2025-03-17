from pathlib import Path
from typing import List, Optional

import torch
import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

import json
from functools import partial    
from transformers import AutoTokenizer

from tensorrt_llm._utils import (mpi_barrier, mpi_rank, mpi_world_size)
from tensorrt_llm.builder import get_engine_version

QWEN_PROMPT_TEMPLATE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"

DEFAULT_PROMPT_TEMPLATES = {
    'QWenLMHeadModel': QWEN_PROMPT_TEMPLATE,
    'QWenForCausalLM': QWEN_PROMPT_TEMPLATE,
    'Qwen2ForCausalLM': QWEN_PROMPT_TEMPLATE,
    'Qwen2MoeForCausalLM': QWEN_PROMPT_TEMPLATE,
    'LlamaForCausalLM': "<s>Human: {input_text}\n</s><s>Assistant: "
}


# Load tokenizer impl, it will be called in external wrapper to avoid loading tokenizer bug under MPI env.
def _load_tokenizer(tokenizer_dir: Optional[str] = None,
                    vocab_file: Optional[str] = None,
                    model_name: str = 'GPTForCausalLM',
                    model_version: Optional[str] = None,
                    tokenizer_type: Optional[str] = None):
    if vocab_file is None:
        use_fast = True
        if tokenizer_type is not None and tokenizer_type == "llama":
            use_fast = False
        # Should set both padding_side and truncation_side to be 'left'
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            legacy=False,
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            tokenizer_type=tokenizer_type,
            use_fast=use_fast)

    if 'qwen' in model_name.lower() and model_version == 'qwen':
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        pad_id = gen_config['pad_token_id']
        end_id = gen_config['eos_token_id']
    elif 'GLM' in model_name and model_version == 'glm':
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id


def load_tokenizer(tokenizer_dir: Optional[str] = None,
                   vocab_file: Optional[str] = None,
                   model_name: str = 'GPTForCausalLM',
                   model_version: Optional[str] = None,
                   tokenizer_type: Optional[str] = None):
    func = partial(_load_tokenizer, tokenizer_dir, vocab_file, model_name,
                   model_version, tokenizer_type)
    if mpi_world_size() > 1:
        # Under MPI env, load tokenizer will result in multiple processes to download the same file to the same folder.
        # This will result some random bug. Force loading on rank0 to warmup the tokenizer to avoid this issue.
        if mpi_rank() == 0:
            func()
        mpi_barrier()
    return func()


def read_model_name(engine_dir: str):
    engine_version = get_engine_version(engine_dir)

    with open(Path(engine_dir) / "config.json", 'r') as f:
        config = json.load(f)

    if engine_version is None:
        return config['builder_config']['name'], None

    model_arch = config['pretrained_config']['architecture']
    model_version = None
    if 'GLM' in model_arch:
        model_version = config['pretrained_config']['chatglm_version']
    if 'qwen' in model_arch.lower():
        model_version = config['pretrained_config']['qwen_type']
    return model_arch, model_version


def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out
        
        
def parse_input(tokenizer,
                input_text=None,
                prompt_template=None,
                input_file=None,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
                num_prepend_vtokens=[],
                model_name=None,
                model_version=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    if input_file is None:
        if 'whisper' in model_name.lower():
            batch_input_ids.append(tokenizer.prefix_tokens)
        else:
            for curr_text in input_text:
                if prompt_template is not None:
                    curr_text = prompt_template.format(input_text=curr_text)
                input_ids = tokenizer.encode(
                    curr_text,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=max_input_length)
                batch_input_ids.append(input_ids)

    if num_prepend_vtokens:
        assert len(num_prepend_vtokens) == len(batch_input_ids)
        base_vocab_size = tokenizer.vocab_size - len(
            tokenizer.special_tokens_map.get('additional_special_tokens', []))
        for i, length in enumerate(num_prepend_vtokens):
            batch_input_ids[i] = list(
                range(base_vocab_size,
                      base_vocab_size + length)) + batch_input_ids[i]

    if input_file is None and 'GLM' in model_name and model_version == 'glm':
        for ids in batch_input_ids:
            ids.append(tokenizer.sop_token_id)

    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]

    logger.debug(f"Input token ids (batch_size = {len(batch_input_ids)}):")
    for i, input_ids in enumerate(batch_input_ids):
        logger.debug(f"Request {i}: {input_ids.tolist()}")

    return batch_input_ids


def print_output(tokenizer,
                 output_ids: torch.Tensor,
                 input_lengths: List[int],
                 sequence_lengths: torch.Tensor):
    num_output_sents, num_beams, _ = output_ids.size()
    batch_size = len(input_lengths)
    num_return_sequences = num_output_sents // batch_size
    
    REPLACEMENT_CHAR = '\ufffd'

    for i in range(batch_size * num_return_sequences):
        batch_idx = i // num_return_sequences
        seq_idx = i % num_return_sequences
        # inputs = output_ids[i][0][:input_lengths[batch_idx]].tolist()
        # input_text = tokenizer.decode(inputs)
        # if seq_idx == 0:
        #     print(f'Input [Text {batch_idx}]: \"{input_text}\"')

        for beam in range(num_beams):
            output_begin = input_lengths[batch_idx]
            output_end = sequence_lengths[i][beam]
            outputs = output_ids[i][beam][output_begin:output_end].tolist()
            output_text = tokenizer.decode(outputs)
            
            if REPLACEMENT_CHAR in output_text:  
                continue
                
            index_str = (f'Text {batch_idx} Seq {seq_idx} Beam {beam}'
                         if num_return_sequences > 1 else
                         f'Text {batch_idx} Beam {beam}')
            print(f'Output [{index_str}]: \"{output_text}\"')

    output_ids = output_ids.reshape((-1, output_ids.size(2)))


if __name__ == '__main__':
    
    promot = '请帮我简明扼要地介绍一下什么是大语言模型'
    engine_dir = '/nas/zxs_onnx/llama2_fp8_trtllm/trtllm_engine/'
    tokenizer_dir = '/nas/zxs_onnx/llama2_fp8_trtllm/tokenizer/'
    
    batch = 1
    
    if batch == 1:
        query = [promot.strip()]
    else:
        batch_prompt = [promot.strip() for _ in range(batch)]
        query = batch_prompt 
        
    streaming = True
    run_profiling = True
    max_output_len = 2048
    temperature = 0.9
    top_p = 0.6
    top_k = 40
    
    runtime_rank = tensorrt_llm.mpi_rank()

    model_name, model_version = read_model_name(engine_dir)

    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=tokenizer_dir,
        model_name=model_name,
        model_version=model_version
    )

    prompt_template = None
    if model_name in DEFAULT_PROMPT_TEMPLATES:
        prompt_template = DEFAULT_PROMPT_TEMPLATES[model_name]

    batch_input_ids = parse_input(tokenizer=tokenizer,
                                  input_text=query,
                                  prompt_template=prompt_template,
                                  max_input_length=512,
                                  pad_id=pad_id,
                                  model_name=model_name,
                                  model_version=model_version)
    
    stop_words_list = None
    bad_words_list = None

    input_lengths = [x.size(0) for x in batch_input_ids]
    
    use_py_session = False
    return_all_generated_tokens = False
    num_beams = 1

    logger.info(f"Using {'Python' if use_py_session else 'C++'} session")

   
    # Normal run
    runner_cls = ModelRunner if use_py_session else ModelRunnerCpp
    runner_kwargs = dict(
        engine_dir=engine_dir,
        rank=runtime_rank,
        max_output_len=max_output_len
    )
    
    if not use_py_session:
        runner_kwargs.update(
            is_enc_dec=False,
            max_batch_size=len(batch_input_ids),
            max_input_len=max(input_lengths),
            max_beam_width=num_beams,
            enable_chunked_context=False,
            multi_block_mode=True,
            cuda_graph_mode=False)
    
    runner_kwargs.update(enable_context_fmha_fp32_acc=False)
    runner = runner_cls.from_dir(**runner_kwargs)

    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids= batch_input_ids,
            max_new_tokens=max_output_len,
            end_id=end_id,
            pad_id=pad_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            stop_words_list=stop_words_list,
            bad_words_list=bad_words_list,
            streaming=streaming,
            output_sequence_lengths=True,
            return_dict=True,
            return_all_generated_tokens=return_all_generated_tokens)
        torch.cuda.synchronize()

    # Receive output, print to screen or save to file
    if streaming:
        streaming_interval = 1
        for curr_outputs in throttle_generator(outputs, streaming_interval):
            if runtime_rank == 0:
                output_ids = curr_outputs['output_ids']
                sequence_lengths = curr_outputs['sequence_lengths']
                
                print_output(
                    tokenizer,
                    output_ids,
                    input_lengths,
                    sequence_lengths)
    else:
        if runtime_rank == 0:
            output_ids = outputs['output_ids']
            sequence_lengths = outputs['sequence_lengths']

            print_output(tokenizer,
                         output_ids,
                         input_lengths,
                         sequence_lengths)

    # Profiling
    if run_profiling:
        ite = 10
        # warmup
        for _ in range(ite):
            with torch.no_grad():
                outputs = runner.generate(
                    batch_input_ids,
                    max_new_tokens=max_output_len,
                    end_id=end_id,
                    pad_id=pad_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_beams=num_beams,
                    stop_words_list=stop_words_list,
                    bad_words_list=bad_words_list,
                    streaming=streaming,
                    output_sequence_lengths=True,
                    return_dict=True,
                    return_all_generated_tokens=return_all_generated_tokens)
                torch.cuda.synchronize()

        tensorrt_llm.profiler.start("tmp")
        for _ in range(ite):
            with torch.no_grad():
                outputs = runner.generate(
                    batch_input_ids,
                    max_new_tokens=max_output_len,
                    end_id=end_id,
                    pad_id=pad_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_beams=num_beams,
                    stop_words_list=stop_words_list,
                    bad_words_list=bad_words_list,
                    streaming=streaming,
                    output_sequence_lengths=True,
                    return_dict=True,
                    return_all_generated_tokens=return_all_generated_tokens)
                torch.cuda.synchronize()
        tensorrt_llm.profiler.stop("tmp")

        print(
            f"batch_size: {len(batch_input_ids)}, avg latency of {ite} iterations: : {tensorrt_llm.profiler.elapsed_time_in_sec('tmp') / ite} sec"
        )
