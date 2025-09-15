import vllm
from vllm.entrypoints.cli.main import main as vllm_main

from unifiedcache.integration.vllm import vllm_adapter

if __name__ == "__main__":
    vllm_main()
