from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM


def get_rotary_emb_ops(model):
    if isinstance(model, Qwen2ForCausalLM):
        return model.model.layers[0].self_attn.rotary_emb
    if isinstance(model, LlamaForCausalLM):
        return model.model.layers[0].self_attn.rotary_emb
    else:
        raise "get model rotary emb failed!  current not implemented for this model"


import functools
import time

from ucm.logger import init_logger

logger = init_logger(__name__)


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} exec time: {elapsed:.6f}s")
        return result

    return wrapper
