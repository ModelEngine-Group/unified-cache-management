import vllm
from vllm.entrypoints.cli.main import main as vllm_main

# from ucm.integration.vllm import vllm_adapter
# import ucm.integration.vllm.vllm_adapter.adapter.vllm_adapter

if __name__ == "__main__":
    vllm_main()
