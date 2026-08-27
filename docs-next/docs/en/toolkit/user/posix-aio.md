# posix-aio

Invokes `ucm/store/test/e2e/posixstore_aio_test.py` from the repository to perform dump/load performance testing via `UcmPipelineStore`, for evaluating UCM POSIX store disk read/write bandwidth. IO engine (`posix_io_engine`, psync/aio), transfer concurrency (`posix_data_trans_concurrency`), and O_DIRECT (`io_direct`) are all configurable via CLI.

Two usage modes are supported:

- **Manual mode** (default): directly specify `--shard-size` / `--shard-number` / `--block-number`, etc.
- **Model-driven mode**: pass `--model` pointing to a model directory; automatically reads `config.json` to detect architecture (GQA / MLA / DSA) and computes `shard-size` / `shard-number` / `block-number`, then forwards to the same script for automated bandwidth testing.

← Back to [UCM Toolkit](../index.md)

## Dependencies

The UCM Python package and its native extensions must be available, and `numpy` must be importable. Model-driven mode only uses the standard library (`json` / `math`), with no dependency on `transformers` / `vllm`.

## Examples

### Manual Mode

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

### Model-Driven Mode

```bash
# MLA, layerwise
ucm-toolkit run posix-aio --model /models/DeepSeek-V3 --tp 8 --input-len 4096 \
  --worker-number 8 --layerwise --storage-backend /mnt/ssd/ucm

# DSA (GLM-5.1 / DeepSeek-V3.2), non-layerwise (--no-layerwise)
ucm-toolkit run posix-aio --model /models/GLM-5.1 --tp 8 --input-len 4096 \
  --worker-number 8 --no-layerwise --storage-backend /mnt/ssd/ucm

# GQA, layerwise
ucm-toolkit run posix-aio --model /models/Qwen3-32B --tp 8 --input-len 4096 \
  --worker-number 8 --layerwise --storage-backend /mnt/ssd/ucm

# Only print UCM Store IO Info (io size / io number), without running the script
ucm-toolkit run posix-aio --model /models/Qwen3-32B --tp 8 --input-len 4096 --layerwise --dry-run
```

Unsupported architectures will warn and exit:

```bash
ucm-toolkit run posix-aio --model /models/DeepSeek-V4-Pro --tp 8 --input-len 4096 --worker-number 8
# → warning: architecture 'hybrid' not supported, only GQA and MLA family (MLA/DSA) are supported now
```

## Parameters

### Manual Mode Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `-w`, `--worker-number` | `1` | worker number: number of worker processes to start concurrently. |
| `-s`, `--shard-size` | `8388608` | shard size: POSIX store I/O size. In layerwise mode, this is the K/V tensor size for one layer of one block. In non-layerwise mode, this is the K/V tensor size for all layers of one block. |
| `-n`, `--shard-number` | `1` | shard number: number of layers in layerwise mode; use 1 in non-layerwise mode. |
| `-b`, `--block-number` | `64` | block number: total number of blocks. |
| `-d`, `--dump-epoch-number` | `32` | dump epoch number: number of dump epochs. |
| `-l`, `--load-epoch-number` | `32` | load epoch number: number of load epochs. |
| `-o`, `--storage-backend` | `./build/data` | storage backend: storage backend path; may be repeated. Passing this option replaces the default backend list with the provided values. |
| `--posix-data-trans-concurrency` | `32` | posix data transfer concurrency (psync worker count). |
| `--posix-io-engine` | `aio` | posix IO engine: `psync` or `aio`. |
| `--io-direct` | `True` | Whether to use O_DIRECT for aligned file I/O; disable with `--no-io-direct`. |

### Model-Driven Mode Parameters

Passing `--model` enters model-driven mode. In this mode, `--shard-size` / `--shard-number` / `--block-number` are computed from the model config and override any manual settings (with a warning printed). `--storage-backend` / `--dump-epoch-number` / `--load-epoch-number` / `--worker-number` / `--posix-io-engine` / `--posix-data-trans-concurrency` / `--io-direct` are still passed through to the script.

| Parameter | Default | Description |
| --- | --- | --- |
| `--model` | — | Model directory (containing `config.json`) or `config.json` file path; passing it enters model-driven mode. |
| `--tp` | `1` | tensor parallel size; for GQA, used to split `num_kv_heads` by rank. |
| `--input-len` | `4096` | Request input length; `block_number = ceil(input_len / block_size)`. |
| `--layerwise` | `True` | Layerwise mode: one shard = one layer (default true); use `--no-layerwise` for non-layerwise (one shard = all layers). |
| `--block-size` | `128` | vLLM paged block token count, used for `input_len → block_number` conversion. |
| `--kv-dtype` | config's `torch_dtype`, or `bfloat16` if absent | Override KV dtype: `bfloat16`/`bf16`, `float16`/`fp16`, `float32`/`fp32`, `float8_e4m3fn`/`fp8`, `float8_e5m2`, `int8`. |
| `--dry-run` | `False` | Only print the `UCM Store IO Info` summary and forwarded command, without launching the script (for verifying io size / io number). |

### Supported Architectures and Formulas

Detection logic follows `docs/source/_static/calculator.js`'s `detectArchitectureType`. Let `T` = KV bytes per layer per block, `B` = `--block-size`, `elem` = dtype byte count, `L` = `num_hidden_layers`:

| Architecture | Detection Fields | Per-Layer-Per-Block Bytes `T` | TP Handling |
| --- | --- | --- | --- |
| GQA (Qwen, MiniMax) | Has `num_key_value_heads` | `2 * (num_kv_heads // tp) * head_dim * B * elem` | Each rank stores its own KV, `num_kv_heads/tp` |
| MLA (DeepSeek-V3/R1) | `kv_lora_rank` + `qk_rope_head_dim`, no `index_head_dim` | `(kv_lora_rank + qk_rope_head_dim) * B * elem` | No TP division (latent replicated across TP, only rank0 dumps) |
| DSA (DeepSeek-V3.2, GLM-5/5.1) | Above + `index_head_dim` | `(kv_lora_rank + qk_rope_head_dim + index_head_dim) * B * elem` | No TP division (same as MLA family) |

`head_dim` derivation: MLA → `kv_lora_rank + qk_rope_head_dim`; DSA → add `index_head_dim`; GQA → `config.head_dim`, otherwise `hidden_size // num_attention_heads`.

Layerwise vs non-layerwise (aligned with real UCM store shard/block partitioning):

| Mode | Shard Size | Shards Per File | Description |
| --- | --- | --- | --- |
| Layerwise | `align_up(T, 4096)` | `L` | One file = L shards, per-layer I/O, can overlap with forward pipeline |
| Non-layerwise | `align_up(L * T, 4096)` | `1` | One file = 1 shard, single I/O transfers all layers |

`file size = shard size * shards per file`; `shard size` is aligned up to 4096, matching the real store's I/O granularity and compatible with `io_direct`. One block corresponds to one file (`DataFilePath(blockId)`), so `shards per file` is the number of shards in a file and `file size` is the file size.

Not supported (warns and exits 1): Hybrid / DeepSeek V4 (`compress_ratios` etc.), Mamba / linear attention (`linear_attention` / `model_type` containing mamba), unknown architectures.

## Resource Estimation

```text
Per-worker data volume ≈ shard-size * shard-number * block-number
Total data volume ≈ worker-number * shard-size * shard-number * block-number
```

In model-driven mode, `shard-size` / `shard-number` / `block-number` are computed using the formulas above. Before running, a `UCM Store IO Info` summary is printed: architecture (uppercase GQA/MLA/DSA), num_hidden_layers, head_dim, dtype, per_layer/block (bytes+KB+formula), shard size (bytes+KB+formula), shards per file, file size (bytes+KB+MB), block_number, total per-worker data volume. With `--dry-run`, only the summary and forwarded command are printed without actually running the script.
