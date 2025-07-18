# 如何使用UCM对接Mooncake Store
本文档叙述了如何使用 UCM 对接 Mooncake 框架的方法。

## 编译Mooncake
按照 https://github.com/kvcache-ai/Mooncake/blob/v0.3.4/doc/en/build.md 的指引完成编译
注意：建议在ubuntu容器里进行编译

## 启动Mooncake服务

### 启动 Metadata Service

1. 进入如下目录，其中$MOONCAKE_ROOT_DIR是下载的mooncake源码根目录

```shell
cd $MOONCAKE_ROOT_DIR/mooncake-transfer-engine/example/http-metadata-server
```

2. 启动服务

   - 请先取消已有的代理，避免带来未知的网络问题

   - 端口号请根据实际情况填写

```shell
unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY
go run . --addr=0.0.0.0:23790
```

### 启动 Master Service

1. 启动服务

- 请先取消已有的代理，避免带来未知的网络问题

- 端口号请根据实际情况填写

```shell
unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY
mooncake_master --port 50001
```

## 验证效果

### 运行demo



1. 设置Python工作目录，其中 UCM_WORK_DIR 为本项目所在的根目录

```shell
export UCM_WORK_DIR="/home/xxx/unified-cache-management" && cd $UCM_WORK_DIR
export PYTHONPATH="${PYTHONPATH}:$UCM_WORK_DIR"
```

2. 准备mooncake.json（路径根据实际情况填写）

```shell
vim $UCM_WORK_DIR/test/mooncake.json
```

- json内容如下

```json
{
    "local_hostname": "127.0.0.1",
    "metadata_server": "http://127.0.0.1:23790/metadata",
    "protocol": "tcp",
    "device_name": "mlx5_1",
    "master_server_address": "127.0.0.1:50001"
}
```

- 参考： https://kvcache-ai.github.io/Mooncake/getting_started/examples/vllm-integration-v1.html#prepare-configuration-file-to-run-example-over-rdma

3. 执行example

```shell
unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY
export MOONCAKE_CONFIG_PATH=$UCM_WORK_DIR/test/mooncake.json
export MC_GID_INDEX=3 
python3 $UCM_WORK_DIR/examples/mooncake_kv_offload.py
```

- MOONCAKE_CONFIG_PATH：上文准备的mooncake.json路径
- MC_GID_INDEX：【可选】使用RDMA协议时必填，且应与RDMA device匹配(使用rdma link show可以查看状态，使用ibv_devinfo -v -d mlx5_1可以查看GID)

### 执行demo和单元测试

1. 【同上】设置Python工作目录，其中 UCM_WORK_DIR 为本项目所在的根目录
2. 【同上】准备mooncake.json（路径根据实际情况填写）
3. 执行example

```shell
unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY
export MOONCAKE_CONFIG_PATH=$UCM_WORK_DIR/test/mooncake.json
export MC_GID_INDEX=3 
pytest $UCM_WORK_DIR/test/test_mooncake.py
```

