from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generator, AsyncGenerator, Literal, Sequence, Any
from threading import Thread
from queue import Empty, Queue
import asyncio
import random
import time
from loguru import logger
from PIL import Image
from pynvml import nvmlInit, nvmlDeviceGetCount


VLQueryType = tuple[str, Image.Image] | tuple[str, list[Image.Image]] | tuple[str, str] | tuple[str, list[str]]


def random_uuid_int():
    """random_uuid 生成的 int uuid 会超出int64的范围,lmdeploy使用会报错"""
    return random.getrandbits(64)


@dataclass
class LmdeployConfig:
    model_path: str
    backend: Literal['turbomind', 'pytorch'] = 'turbomind'
    model_name: str = 'internlm2'
    model_format: Literal['hf', 'llama', 'awq'] = 'hf'
    tp: int = 1                         # Tensor Parallelism.
    max_batch_size: int = 128
    cache_max_entry_count: float = 0.8  # 调整 KV Cache 的占用比例为0.8
    quant_policy: int = 0               # KV Cache 量化, 0 代表禁用, 4 代表 4bit 量化, 8 代表 8bit 量化
    system_prompt: None| str = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
    """
    log_level: Literal['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'] = 'ERROR'
    deploy_method: Literal['local', 'serve'] = 'local'
    # for server
    server_name: str = '0.0.0.0'
    server_port: int = 23333
    api_keys: list[str] | str | None = None
    ssl: bool = False


class DeployEngine(ABC):

    @abstractmethod
    def chat_stream(
        self,
        *args,
        **kwargs,
    ) -> Generator[tuple[str, Sequence], None, None]:
        """流式返回对话

        Yields:
            Generator[tuple[str, Sequence], None, None]: 回答和历史记录
        """
        pass


class LmdeployEngine(DeployEngine):
    def __init__(self, config: LmdeployConfig) -> None:
        import lmdeploy
        from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig, ChatTemplateConfig

        logger.info(f"lmdeploy version: {lmdeploy.__version__}")
        logger.info(f"lmdeploy config: {config}")

        assert config.backend in ['turbomind', 'pytorch'], \
            f"backend must be 'turbomind' or 'pytorch', but got {config.backend}"
        assert config.model_format in ['hf', 'llama', 'awq'], \
            f"model_format must be 'hf' or 'llama' or 'awq', but got {config.model_format}"
        assert config.cache_max_entry_count >= 0.0 and config.cache_max_entry_count <= 1.0, \
            f"cache_max_entry_count must be >= 0.0 and <= 1.0, but got {config.cache_max_entry_count}"
        assert config.quant_policy in [0, 4, 8], f"quant_policy must be 0, 4 or 8, but got {config.quant_policy}"

        self.config = config

        if config.backend == 'turbomind':
            # 可以直接使用transformers的模型,会自动转换格式
            self.backend_config = TurbomindEngineConfig(
                model_format = config.model_format, # The format of input model. `hf` meaning `hf_llama`, `llama` meaning `meta_llama`, `awq` meaning the quantized model by awq. Default: None. Type: str
                tp = config.tp,                     # Tensor Parallelism.
                session_len = None,                 # the max session length of a sequence, default to None
                max_batch_size = config.max_batch_size,
                cache_max_entry_count = config.cache_max_entry_count,
                cache_block_seq_len = 64,
                enable_prefix_caching = False,
                quant_policy = config.quant_policy, # KV Cache 量化, 0 代表禁用, 4 代表 4bit 量化, 8 代表 8bit 量化
                rope_scaling_factor = 0.0,
                use_logn_attn = False,
                download_dir = None,
                revision = None,
                max_prefill_token_num = 8192,
                num_tokens_per_iter = 0,
                max_prefill_iters = 1,
            )
        else:
            self.backend_config = PytorchEngineConfig(
                tp = config.tp,                     # Tensor Parallelism.
                session_len = None,                 # the max session length of a sequence, default to None
                max_batch_size = config.max_batch_size,
                cache_max_entry_count = config.cache_max_entry_count,
                prefill_interval = 16,
                block_size = 64,
                num_cpu_blocks = 0,
                num_gpu_blocks = 0,
                adapters = None,
                max_prefill_token_num = 4096,
                thread_safe = False,
                enable_prefix_caching = False,
                download_dir = None,
                revision = None,
            )
        logger.info(f"lmdeploy backend_config: {self.backend_config}")

        self.chat_template_config = ChatTemplateConfig(
            model_name = config.model_name, # All the chat template names: `lmdeploy list`
            system = None,
            meta_instruction = config.system_prompt,
            eosys = None,
            user = None,
            eoh = None,
            assistant = None,
            eoa = None,
            separator = None,
            capability = None,
            stop_words = None,
        )
        logger.info(f"lmdeploy chat_template_config: {self.chat_template_config}")


class LmdeployLocalEngine(LmdeployEngine):
    def __init__(self, config: LmdeployConfig) -> None:
        super().__init__(config)

        from lmdeploy import pipeline
        from lmdeploy.serve.async_engine import AsyncEngine
        from lmdeploy.serve.vl_async_engine import VLAsyncEngine

        self.pipe: AsyncEngine | VLAsyncEngine = pipeline(
            model_path = config.model_path,
            model_name = None,
            backend_config = self.backend_config,
            chat_template_config = self.chat_template_config,
            log_level = config.log_level
        )
        self.use_vl_engine = isinstance(self.pipe, VLAsyncEngine)
        # logger.info(f"pipe: {self.pipe}")
        # logger.info(f"use_vl_engine: {self.use_vl_engine}")

    def __stream_infer_single(
        self,
        prompt: str | list[dict],
        session_id: int,
        gen_config = None,
        do_preprocess: bool = True,
        adapter_name: str | None = None,
        **kwargs
    ) -> Generator:
        """Inference a batch of prompts with stream mode.
        将输入的promot限制在一条

        Args:
            prompt (str | list[dict]): a prompt. It accepts: string prompt,
            a chat history in OpenAI format.
            session_id (int): a session id.
            gen_config (GenerationConfig | None): a instance of or a list of
                GenerationConfig. Default to None.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            adapter_name (str): the adapter name of slora for pytorch backend.
                Pick one from adapters. Default to None, using the base model.
        """
        from lmdeploy.messages import GenerationConfig, Response
        from lmdeploy.serve.async_engine import GenOut, _get_event_loop

        if gen_config is None:
            gen_config = GenerationConfig()
        # set random if it is not set
        if gen_config.random_seed is None:
            gen_config.random_seed = random.getrandbits(64)

        outputs = Queue()
        generator: AsyncGenerator[GenOut, Any] = self.pipe.generate(prompt,
                              session_id,
                              gen_config=gen_config,
                              stream_response=True,
                              sequence_start=True,
                              sequence_end=True,
                              do_preprocess=do_preprocess,
                              adapter_name=adapter_name,
                              **kwargs)

        async def _inner_call(i, generator) -> None:
            async for out in generator:
                outputs.put(
                    Response(out.response, out.generate_token_len,
                             out.input_token_len, i, out.finish_reason,
                             out.token_ids, out.logprobs))

        async def gather() -> None:
            await asyncio.gather(
                _inner_call(session_id, generator))
            outputs.put(None)

        loop: asyncio.AbstractEventLoop = _get_event_loop()
        proc = Thread(target=lambda: loop.run_until_complete(gather()))
        # 启动多线程
        proc.start()

        while True:
            try:
                out = outputs.get(timeout=0.001)
                if out is None:
                    break
                yield out
            except Empty:
                pass

        # 等待子线程执行完毕
        proc.join()

    async def chat_stream_local(
        self,
        prompt: str | list[dict],
        session_id: int,
        gen_config = None,
        do_preprocess: bool = True,
        adapter_name: str | None = None,
        **kwargs
    ) -> AsyncGenerator:
        """stream chat 异步实现

        Args:
            prompt (str | list[dict]): a prompt. It accepts: string prompt,
            a chat history in OpenAI format.
            session_id (int): a session id.
            gen_config (GenerationConfig | None): a instance of or a list of
                GenerationConfig. Default to None.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            adapter_name (str): the adapter name of slora for pytorch backend.
                Pick one from adapters. Default to None, using the base model.
        """
        from lmdeploy.messages import GenerationConfig, Response
        from lmdeploy.serve.async_engine import GenOut

        if gen_config is None:
            gen_config = GenerationConfig()
        # set random if it is not set
        if gen_config.random_seed is None:
            gen_config.random_seed = random.getrandbits(64)

        output: GenOut
        async for output in self.pipe.generate(
                prompt,
                session_id,
                gen_config=gen_config,
                stream_response=True,
                sequence_start=True,
                sequence_end=True,
                do_preprocess=do_preprocess,
                adapter_name=adapter_name,
            ):
            yield Response(
                text=output.response,
                generate_token_len=output.generate_token_len,
                input_token_len=output.input_token_len,
                session_id=session_id,
                finish_reason=output.finish_reason,
                token_ids=output.token_ids,
                logprobs=output.logprobs,
            )

    def chat_stream(
        self,
        query: str | VLQueryType,
        history: Sequence[Sequence] | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 40,
        session_id: int | None = None,
        **kwargs,
    ) -> Generator[tuple[str, Sequence], None, None]:
        from lmdeploy.messages import GenerationConfig, Response

        # session_id
        session_id = random_uuid_int() if session_id is None else session_id
        # logger.info(f"{session_id = }")

        messages = f'<s>Human: {query}\n</s><s>Assistant: '
        # logger.info(f"messages: {messages}")

        gen_config = GenerationConfig(
            n = 1,
            max_new_tokens = max_new_tokens,
            top_p = top_p,
            top_k = top_k,
            temperature = temperature,
            repetition_penalty = 1.0,
            ignore_eos = False,
            random_seed = None,
            stop_words = None,
            bad_words = None,
            min_new_tokens = None,
            skip_special_tokens = True,
            logprobs = None,
        )
        logger.info(f"gen_config: {gen_config}")

        response_text: str = ""
        # 放入 [{},{}] 格式返回一个response
        # 放入 [] 或者 [[{},{}]] 格式返回一个response列表
                
        start = time.perf_counter()
        
        n_token = 0
        
        response: Response

        for response in self.__stream_infer_single(
        # async for response in self.chat_stream_local(
            prompt = messages,
            session_id = session_id,
            gen_config = gen_config,
            do_preprocess = True,
            adapter_name = None
        ):
            # logger.info(f"response: {response}")
            # Response(text='很高兴', generate_token_len=10, input_token_len=111, session_id=0, finish_reason=None)
            # Response(text='认识', generate_token_len=11, input_token_len=111, session_id=0, finish_reason=None)
            # Response(text='你', generate_token_len=12, input_token_len=111, session_id=0, finish_reason=None)
            
            session_id = response.session_id
            n_token = response.generate_token_len  
            
            response_text += response.text
            yield response_text, history + [[query, response_text]]
        
        # logger.info(f"response_text: {response_text}")
        # logger.info(f"history: {history + [[query, response_text]]}")
        
        elapsed_time = time.perf_counter() - start
        
        completion_token_throughput = n_token / elapsed_time
        
        logger.info(f"tokens num: {n_token}, total time: {elapsed_time:.3f} s")
        logger.info(f'token throughput (completion token): {completion_token_throughput:.3f} token/s\n')


class InferEngine(DeployEngine):
    def __init__(
        self,
        lmdeploy_config: LmdeployConfig,
    ) -> None:
        
        assert lmdeploy_config is not None
        assert lmdeploy_config.deploy_method in ['local', 'serve'], f"deploy_method must be 'local' or 'serve', but got {lmdeploy_config.deploy_method}"
        if lmdeploy_config.deploy_method == 'local':
            self.engine = LmdeployLocalEngine(lmdeploy_config)
        logger.info("lmdeploy model loaded!")

    def chat_stream(
        self,
        query: str | VLQueryType,
        history: Sequence[Sequence] | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 40,
        session_id: int | None = None,
        **kwargs,
    ) -> Generator[tuple[str, Sequence], None, None]:
        """流式返回回答

        Args:
            query (str | VLQueryType): 查询语句,支持图片
            history (Sequence[Sequence], optional): 对话历史. Defaults to None.
                example: [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
            max_new_tokens (int, optional): 单次对话返回最大长度. Defaults to 1024.
            temperature (float, optional): temperature. Defaults to 0.8.
            top_p (float, optional): top_p. Defaults to 0.8.
            top_k (int, optional): top_k. Defaults to 40.
            session_id (int, optional): 会话id. Defaults to None.

        Yields:
            Generator[tuple[str, Sequence], None, None]: 回答和历史记录
        """
        history = [] if history is None else list(history)
        return self.engine.chat_stream(
            query = query,
            history = history,
            max_new_tokens = max_new_tokens,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            session_id = session_id,
            **kwargs
        )

if __name__ == "__main__":
    
    promot = '介绍一下中国'
    model_path = '/nas/zxs_onnx/Llama2-Chinese-7b-Chat-ms-4bit/'
    
    query = promot.strip()
    assert len(query) > 1 
    
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    
    LMDEPLOY_CONFIG = LmdeployConfig(
        model_path = model_path,
        backend = 'turbomind',
        model_name = 'llama2',
        model_format = 'awq', 
        tp = device_count,
        cache_max_entry_count = 0.5,
        system_prompt = None,
        deploy_method = 'local'
    )

    infer_engine = InferEngine(lmdeploy_config = LMDEPLOY_CONFIG)
    
    history = []
    
    for response, history in infer_engine.chat_stream(
        query = query,
        history = history,
        max_new_tokens = 4096,
        temperature = 0.9,
        top_p = 0.6,
        top_k = 40,
        session_id = None,
    ):
        logger.info(f"response: {response}")