# How to Use UCM with Mooncake Store
This guide explains how to integrate UCM (Unified Cache Management) with the Mooncake framework.

## Build Mooncake
Follow the [official guide](https://github.com/kvcache-ai/Mooncake/blob/v0.3.4/doc/en/build.md) to build Mooncake:

📄 Mooncake Build Instructions

⚠️ Recommended: Compile inside a Ubuntu container to avoid environment issues.

## Start Mooncake Services
1. Start Metadata Service
Navigate to the metadata server directory:

```bash
cd $MOONCAKE_ROOT_DIR/mooncake-transfer-engine/example/http-metadata-server
```

Replace `$MOONCAKE_ROOT_DIR` with your Mooncake source root path.

2. Launch the service:

    - Make sure to unset any HTTP proxies to prevent networking issues.

    - Use appropriate port based on your environment.

```bash
unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY
go run . --addr=0.0.0.0:23790
```
3. Start Master Service
```bash
unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY
mooncake_master --port 50001
```
Same note as above: ensure no proxies are set, and adjust port as needed.

## Verify Integration
Run the Example Script
Set the Python working directory. Replace the path accordingly:

```bash
export UCM_WORK_DIR="/home/xxx/unified-cache-management" && cd $UCM_WORK_DIR
export PYTHONPATH="${PYTHONPATH}:$UCM_WORK_DIR"
```
Prepare the Mooncake configuration file:

```bash
vim $UCM_WORK_DIR/test/mooncake.json
```

Example config:

```json
{
    "local_hostname": "127.0.0.1",
    "metadata_server": "http://127.0.0.1:23790/metadata",
    "protocol": "tcp",
    "device_name": "mlx5_1",
    "master_server_address": "127.0.0.1:50001"
}
```
## Reference: Mooncake vLLM RDMA Integration

### Run the example:

```bash
unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY
export MOONCAKE_CONFIG_PATH=$UCM_WORK_DIR/test/mooncake.json
export MC_GID_INDEX=3 
python3 $UCM_WORK_DIR/examples/mooncake_kv_offload.py
```
- MOONCAKE_CONFIG_PATH: Path to your config file

- MC_GID_INDEX: (Optional) Required when using RDMA. Should match the RDMA device (rdma link show and ibv_devinfo -v -d mlx5_1 can help inspect).

### Run Unit Tests

```bash
unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY
export MOONCAKE_CONFIG_PATH=$UCM_WORK_DIR/test/mooncake.json
export MC_GID_INDEX=3 
pytest $UCM_WORK_DIR/test/test_mooncake.py
```