import contextlib
import os
import time
from dataclasses import asdict
# Third Party
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from unifiedcache.logger import init_logger

logger = init_logger(__name__)


def setup_environment_variables():
    os.environ["VLLM_USE_V1"] = "1"


@contextlib.contextmanager
def build_llm_with_uc(module_path: str, name: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=name,
        kv_connector_module_path=module_path,
        kv_role="kv_both",
        kv_connector_extra_config={"ucm_connector_name": "UcmDram",
                                   "ucm_connector_config": {"max_cache_size": 5368709120}}
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=8000,
        gpu_memory_utilization=0.8
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        logger.info("LLM engine is exiting.")


def print_output(
        llm: LLM,
        prompt: list[str],
        sampling_params: SamplingParams,
        req_str: str,
):
    start = time.time()
    outputs = llm.generate(prompt, sampling_params)
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def main():
    module_path = "unifiedcache.integration.vllm.uc_connector"
    name = "UnifiedCacheConnectorV1"
    model = "/home/models/Qwen2.5-14B-Instruct"

    setup_environment_variables()

    with build_llm_with_uc(module_path, name, model) as llm:
        prompts = [
            "Imagine you are an artificial intelligence developed in the year 2075, designed to assist humanity in "
            "navigating the complex ethical, philosophical, and technological challenges of a rapidly evolving world. "
            "You have access to vast historical records, scientfic data, and human literature, and your core "
            "directive is to promote sustainable development, social equity, and the flourishing of conscious beings. "
            "Write a detailed letter to the leaders of Earth, explaining the most urgent global issue of the 21st "
            "century, the root sauses behind it, and a set of scientifically grounded, morally sound, and globally "
            "cooperative solutions that transcend culturak and national boundaries. Include both immediate actions "
            "and long-term strategies."
        ]

        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=100)

        print_output(llm, prompts, sampling_params, "first")
        print_output(llm, prompts, sampling_params, "second")


if __name__ == "__main__":
    main()
