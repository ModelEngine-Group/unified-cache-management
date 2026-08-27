# posix-aio

调用仓库中的 `ucm/store/test/e2e/posixstore_aio_test.py`，通过 `UcmPipelineStore` 做 dump/load 性能测试，用于评估 UCM POSIX store 的磁盘读写带宽。IO 引擎（`posix_io_engine`，psync/aio）、传输并发（`posix_data_trans_concurrency`）、是否走 O_DIRECT（`io_direct`）均可通过 CLI 配置。

支持两种用法：

- **手动模式**（默认）：直接指定 `--shard-size` / `--shard-number` / `--block-number` 等。
- **模型驱动模式**：传 `--model` 指向模型目录，自动读取 `config.json` 判定架构（GQA / MLA / DSA）并计算 `shard-size` / `shard-number` / `block-number`，再转发给同一脚本，实现自动化带宽测试。

← 返回 [UCM Toolkit 文档](../index.md)

## 依赖

当前 UCM Python 包及其 native 扩展可用，`numpy` 可导入。模型驱动模式仅使用标准库（`json` / `math`），不依赖 `transformers` / `vllm`。

## 示例

### 手动模式

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

### 模型驱动模式

```bash
# MLA, layerwise
ucm-toolkit run posix-aio --model /models/DeepSeek-V3 --tp 8 --input-len 4096 \
  --worker-number 8 --layerwise --storage-backend /mnt/ssd/ucm

# DSA (GLM-5.1 / DeepSeek-V3.2), 非 layerwise (--no-layerwise)
ucm-toolkit run posix-aio --model /models/GLM-5.1 --tp 8 --input-len 4096 \
  --worker-number 8 --no-layerwise --storage-backend /mnt/ssd/ucm

# GQA, layerwise
ucm-toolkit run posix-aio --model /models/Qwen3-32B --tp 8 --input-len 4096 \
  --worker-number 8 --layerwise --storage-backend /mnt/ssd/ucm

# 只打印 UCM Store IO Info（io size / io number），不跑脚本
ucm-toolkit run posix-aio --model /models/Qwen3-32B --tp 8 --input-len 4096 --layerwise --dry-run
```

不支持的架构会 warning 并退出：

```bash
ucm-toolkit run posix-aio --model /models/DeepSeek-V4-Pro --tp 8 --input-len 4096 --worker-number 8
# → warning: architecture 'hybrid' not supported, only GQA and MLA family (MLA/DSA) are supported now
```

## 参数

### 手动模式参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-w`, `--worker-number` | `1` | worker number: number of worker processes to start concurrently. |
| `-s`, `--shard-size` | `8388608` | shard size: POSIX store I/O size. In layerwise mode, this is the K/V tensor size for one layer of one block. In non-layerwise mode, this is the K/V tensor size for all layers of one block. |
| `-n`, `--shard-number` | `1` | shard number: number of layers in layerwise mode; use 1 in non-layerwise mode. |
| `-b`, `--block-number` | `64` | block number: total number of blocks. |
| `-d`, `--dump-epoch-number` | `32` | dump epoch number: number of dump epochs. |
| `-l`, `--load-epoch-number` | `32` | load epoch number: number of load epochs. |
| `-o`, `--storage-backend` | `./build/data` | storage backend: storage backend path; may be repeated. Passing this option replaces the default backend list with the provided values. |
| `--posix-data-trans-concurrency` | `32` | posix 数据传输并发（psync worker 数）。 |
| `--posix-io-engine` | `aio` | posix io 引擎：`psync` 或 `aio`。 |
| `--io-direct` | `True` | 是否走 O_DIRECT 做对齐文件 I/O；用 `--no-io-direct` 关闭。 |

### 模型驱动模式参数

传入 `--model` 即进入模型驱动模式。此时 `--shard-size` / `--shard-number` / `--block-number` 由模型 config 计算得出，会覆盖任何手动设置（并打印 warning）。`--storage-backend` / `--dump-epoch-number` / `--load-epoch-number` / `--worker-number` / `--posix-io-engine` / `--posix-data-trans-concurrency` / `--io-direct` 仍透传给脚本。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | — | 模型目录（含 `config.json`）或 `config.json` 文件路径；传入即进入模型驱动模式。 |
| `--tp` | `1` | tensor parallel size；GQA 下用于按 rank 切分 `num_kv_heads`。 |
| `--input-len` | `4096` | 请求输入长度；`block_number = ceil(input_len / block_size)`。 |
| `--layerwise` | `True` | layerwise 模式：一个 shard = 一层（默认 true）；用 `--no-layerwise` 切到非 layerwise（一个 shard = 全部层）。 |
| `--block-size` | `128` | vLLM paged block 的 token 数，用于 `input_len → block_number` 换算。 |
| `--kv-dtype` | config 的 `torch_dtype`，无则 `bfloat16` | 覆盖 KV dtype：`bfloat16`/`bf16`、`float16`/`fp16`、`float32`/`fp32`、`float8_e4m3fn`/`fp8`、`float8_e5m2`、`int8`。 |
| `--dry-run` | `False` | 只打印 `UCM Store IO Info` 摘要与转发命令，不启动脚本（用于核对 io size / io number）。 |

### 支持的架构与计算公式

检测逻辑参照 `docs/source/_static/calculator.js` 的 `detectArchitectureType`。设 `T` = 单层单 block 的 KV 字节数，`B` = `--block-size`，`elem` = dtype 字节数，`L` = `num_hidden_layers`：

| 架构 | 判定字段 | 单层单 block 字节 `T` | TP 处理 |
| --- | --- | --- | --- |
| GQA（Qwen、MiniMax） | 有 `num_key_value_heads` | `2 * (num_kv_heads // tp) * head_dim * B * elem` | 每 rank 存自己的 KV，`num_kv_heads/tp` |
| MLA（DeepSeek-V3/R1） | `kv_lora_rank` + `qk_rope_head_dim`，无 `index_head_dim` | `(kv_lora_rank + qk_rope_head_dim) * B * elem` | 不除 tp（latent 在 TP 间复制、仅 rank0 dump） |
| DSA（DeepSeek-V3.2、GLM-5/5.1） | 上述 + `index_head_dim` | `(kv_lora_rank + qk_rope_head_dim + index_head_dim) * B * elem` | 不除 tp（同 MLA 族） |

`head_dim` 推导：MLA→`kv_lora_rank + qk_rope_head_dim`；DSA→再加 `index_head_dim`；GQA→`config.head_dim`，否则 `hidden_size // num_attention_heads`。

layerwise 与非 layerwise（对齐真实 UCM store 的 shard/block 切分）：

| 模式 | shard size | shards per file | 说明 |
| --- | --- | --- | --- |
| layerwise | `align_up(T, 4096)` | `L` | 一个文件 = L 个 shard，逐层 I/O，可与 forward 流水重叠 |
| 非 layerwise | `align_up(L * T, 4096)` | `1` | 一个文件 = 1 个 shard，一次 I/O 传全部层 |

`file size = shard size * shards per file`；`shard size` 向上对齐到 4096，匹配真实 store 的 I/O 粒度并兼容 `io_direct`。一个 block 对应一个文件（`DataFilePath(blockId)`），故 `shards per file` 即一个文件里的 shard 数、`file size` 即该文件大小。

不支持（warning 并退出 1）：Hybrid / DeepSeek V4（`compress_ratios` 等指标）、Mamba / 线性注意力（`linear_attention` / `model_type` 含 mamba）、未知架构。

## 资源估算

```text
单 worker 数据量约为 shard-size * shard-number * block-number
总数据量约为 worker-number * shard-size * shard-number * block-number
```

模型驱动模式下，`shard-size` / `shard-number` / `block-number` 由上述公式计算，运行前会打印 `UCM Store IO Info` 摘要：architecture（大写 GQA/MLA/DSA）、num_hidden_layers、head_dim、dtype、per_layer/block（bytes+KB+公式）、shard size（bytes+KB+公式）、shards per file、file size（bytes+KB+MB）、block_number、单 worker 总数据量。加 `--dry-run` 只打印摘要与转发命令、不真正跑脚本。
