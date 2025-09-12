from vllm.entrypoints.cli.main import main as vllm_main

import vllm
from unifiedcache.integration.vllm import vllm_adapter

if __name__ == "main":
    vllm_main()