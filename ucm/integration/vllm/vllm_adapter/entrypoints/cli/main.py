import vllm
from unifiedcache.integration.vllm import vllm_adapter
from vllm.entrypoints.cli.main import main as vllm_main

if __name__ == "__main__":
    vllm_main()
