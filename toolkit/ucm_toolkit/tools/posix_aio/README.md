# posix-aio

调用仓库中的 `ucm/store/test/e2e/posixstore_aio_test.py`，通过 `UcmPipelineStore` 和 `posix_io_engine=aio` 做 dump/load 性能测试，用于评估 UCM POSIX AIO store 的磁盘读写带宽。

← 返回 [UCM Toolkit 顶层文档](../../../README.md)

## 依赖

当前 UCM Python 包及其 native 扩展可用，`numpy` 可导入。

## 导入来源

`ucm-toolkit run posix-aio` 默认会优先使用当前 Python 环境中已经安装的 `ucm` 包；如果当前环境找不到安装版 `ucm`，才会把主仓源码根目录加入子进程 `PYTHONPATH`。当使用安装版 `ucm` 时，toolkit 会从子进程 `PYTHONPATH` 中移除主仓源码根目录，避免源码目录覆盖已安装包。如果需要显式切换导入来源，可以设置：

```bash
# 强制使用当前主仓源码中的 ucm 包
UCM_TOOLKIT_POSIX_AIO_IMPORT=source ucm-toolkit run posix-aio

# 强制使用当前 Python 环境中已安装的 ucm 包
UCM_TOOLKIT_POSIX_AIO_IMPORT=installed ucm-toolkit run posix-aio
```

## 示例

```bash
ucm-toolkit run posix-aio

ucm-toolkit run posix-aio \
  --worker-number 1 \
  --shard-size 8388608 \
  --shard-number 1 \
  --block-number 64 \
  --dump-epoch-number 32 \
  --load-epoch-number 32 \
  --storage-backend ./build/data
```

## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-w`, `--worker-number` | `1` | worker number: number of worker processes to start concurrently. |
| `-s`, `--shard-size` | `8388608` | shard size: POSIX store I/O size. In layerwise mode, this is the K/V tensor size for one layer of one block. In non-layerwise mode, this is the K/V tensor size for all layers of one block. |
| `-n`, `--shard-number` | `1` | shard number: number of layers in layerwise mode; use 1 in non-layerwise mode. |
| `-b`, `--block-number` | `64` | block number: total number of blocks. |
| `-d`, `--dump-epoch-number` | `32` | dump epoch number: number of dump epochs. |
| `-l`, `--load-epoch-number` | `32` | load epoch number: number of load epochs. |
| `-o`, `--storage-backend` | `./build/data` | storage backend: storage backend path; may be repeated. Passing this option replaces the default backend list with the provided values. |

## 资源估算

```text
单 worker 数据量约为 shard-size * shard-number * block-number
总数据量约为 worker-number * shard-size * shard-number * block-number
```
