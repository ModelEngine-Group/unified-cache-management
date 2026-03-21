# Quickstart-MindIE-LLM

This guide shows how to install UCM with MindIE-LLM support, patch the required MindIE-LLM Python modules, configure `mindie_llm`, and launch `mindie_llm_server` with UCM as the KV cache backend on Ascend.

## Prerequisites

- MindIE-LLM 2.3.0
- Python >= 3.10
- Ascend runtime/toolkit installed and available in the environment. For details, refer to the [MindIE-LLM official documentation](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/docs/zh/user_guide/quick_start/quick_start.md).

## Step 1: Install UCM

UCM can be installed in one of the following ways.

### Option 1: Build from source

Clone the repository and install UCM with MindIE-LLM support enabled:

```bash
git clone --depth 1 https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management
export PLATFORM=ascend
export UCM_ENABLE_MINDIE=1
export UCM_CXX11_ABI=1  # Or 0. This must match the target MindIE/PyTorch ABI.
pip install -v -e . --no-build-isolation
cd ..
```

> **Note:** Packages built without `UCM_ENABLE_MINDIE=1` do **not** contain MindIE-LLM integration code.
>
> **ABI requirement:** When `UCM_ENABLE_MINDIE=1`, you must also set `UCM_CXX11_ABI=0` or `1`. The value must match the target MindIE/PyTorch ABI.

### Option 2: Build from Docker

Use the provided MindIE-LLM Dockerfile (Ascend base, MindIE-LLM 2.3.0):

```bash
docker build -t ucm-mindie:latest -f ./docker/Dockerfile.mindie_llm ./
```

This Dockerfile:

* uses `swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.3.0-800I-A2-py311-openeuler24.03-lts`
* sets `UCM_ENABLE_MINDIE=1`
* sets `UCM_CXX11_ABI` (default `1`, override with `--build-arg UCM_CXX11_ABI=0|1` to match your target environment)
* installs UCM with MindIE-LLM support
* applies the patch during image build

## Step 2: Prepare the UCM configuration

UCM for MindIE-LLM provides Prefix Cache integration through patched MindIE-LLM modules. You can use the packaged config at `ucm/integration/mindie/ucm_config.json` or provide your own config file.

Minimal example:

```json
{
  "storage_backends": ["/path/to/kvcache"],
  "mindie_config_path": "/usr/local/lib/python3.11/site-packages/mindie_llm/conf/config.json",
  "block_elem_size": 2
}
```

Save this file as `ucm_config.json` and replace the example paths with paths from your environment.

Key notes:

* By default, the patch is applied when `mindie_llm` is first imported after UCM is installed.
* The hook copies the patched `uc_utils.py`, `unifiedcache_mempool.py`, and `prefix_cache_plugin.py` files into the installed `mindie_llm` package.
* In the provided Docker image, the patch is already applied during image build, so runtime import uses the patched files directly.

## Step 3: Launch MindIE-LLM with UCM

Locate the installed `mindie_llm` package:

```bash
python -c "import mindie_llm, os; print(os.path.dirname(mindie_llm.__file__))"
```

Then locate `conf/config.json` under that directory.

Ensure the service user can read the configuration file. For example:

```bash
chmod 640 <path-to-mindie_llm>/conf/config.json
```

In `config.json`, add or update the `kvPoolConfig` section under `BackendConfig`:

```json
"BackendConfig": {
  "kvPoolConfig": {
    "backend": "unifiedcache",
    "configPath": "/path/to/your/ucm_config.json",
    "asyncWrite": true
  }
}
```

Update `config.json` with the service IP, port, model path, and any other required MindIE-LLM settings.

Run `mindie_llm_server` to start the service.

If the following message is displayed, the service has started successfully:

```bash
Daemon start success!
```

After startup, inspect the MindIE-LLM logs to confirm that the `unifiedcache` backend is loaded successfully.

## Troubleshooting

### MindIE-LLM code is not patched

* Confirm that `UCM_ENABLE_MINDIE=1` was set before installation.
* Confirm that `UCM_CXX11_ABI=0` or `1` was set correctly before installation.
* Reinstall UCM.
* Verify that `mindie_llm` is already installed.

### `configPath` not found

* Use an absolute path.
* Ensure the file is readable by the service user.

### Service starts but the UCM backend is not enabled

* Recheck `BackendConfig.kvPoolConfig`.
* Inspect MindIE-LLM startup logs.
