# Deployment

UCM integrates with the following inference engines by plugging in a KV
connector / patched store at runtime. Each engine has its own quick start guide
covering installation, configuration, and launching inference.

## Which guide should I follow?

Pick the guide that matches the engine you use. All guides assume UCM is
already installed; if it is not, see the [Installation](../installation.md)
guide first.

<div class="grid cards" markdown>

-   :material-home-account: **vLLM (CUDA)**

    ---

    Run UCM with vLLM on GPU (CUDA) platforms, including prefix caching and
    sparse attention. Docker, pip, and runtime integration steps.

    [:octicons-arrow-right-24: vLLM quick start](quickstart_vllm.md)

-   :material-chip: **vLLM-Ascend (NPU)**

    ---

    Run UCM with vLLM-Ascend on Ascend NPU platforms, including the upgrade
    deployment steps for the Ascend environment.

    [:octicons-arrow-right-24: vLLM-Ascend quick start](quickstart_vllm_ascend.md)

-   :material-tools: **SGLang (CUDA)**

    ---

    Run UCM with SGLang on GPU (CUDA) platforms via the hierarchical cache
    configuration.

    [:octicons-arrow-right-24: SGLang quick start](quickstart_sglang.md)

-   :material-server: **MindIE (NPU Ascend)**

    ---

    Run UCM with MindIE-LLM on Ascend NPU platforms, including MindIE-LLM
    patching and `mindie_llm_server` configuration.

    [:octicons-arrow-right-24: MindIE quick start](quickstart_mindie_llm.md)

</div>

## Building from source

If you prefer to build UCM from source code rather than using Docker images or
pip wheels, see [Building and Installing UCM from Source](../../developer-guide/build_from_source.md).