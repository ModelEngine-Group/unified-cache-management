# ttft-analyze

`ttft-analyze` 估算 KV Cache 完全命中 UCM SSD/DRAM Prefix Cache 时的 TTFT。
它基于用户给定（或实测）的 Full Prefill / Full HBM Prefix Cache TTFT，结合存储读带宽、
H2D 带宽与输入长度，输出预估 TTFT、相对增益/损失与瓶颈拆分，用于部署前 TTFT 预判。

纯计算，无外部依赖，不运行真实推理。

## 模型

`posix_bw` 为存储**总**读带宽（多卡均分），`h2d_bw` 为**单卡** H2D 带宽（独享）。
每张卡需加载进 HBM 的 kvcache 字节数因架构而异：

| 架构 | 每卡加载量 | t_read（总带宽均分） | t_h2d（单卡独享） |
| --- | --- | --- | --- |
| GQA/MHA | `cache_total / tp` | `cache_total / posix_bw` | `cache_total / (tp × h2d_bw)` |
| MLA/DSA | `cache_total`（latent 不切分） | `tp × cache_total / posix_bw` | `cache_total / h2d_bw` |

```
t_cache_load = t_read + t_h2d                                     (ms)

模式 layered（边加载边计算，对应 use_layerwise=true）:
    TTFT_ucm = max(TTFT_hbm, t_cache_load)

模式 full（先整段加载再计算，对应 use_layerwise=false）:
    TTFT_ucm = TTFT_hbm + t_cache_load
```

- `cache_total`：该输入长度下模型总 KV cache 字节数，由 `--model-dir/config.json`
  的架构参数推导，口径与 `docs/source/getting-started/kv_cache_calculator.md` 一致。
- `posix_bw`：存储总读带宽（GB/s）；SSD / DRAM 的实际读带宽分别代入即可得到对应介质结果。
- `h2d_bw`：单卡 H2D 搬移带宽（GB/s），可由 `dev-sandbox copy` 或 `bandwidth` 工具测得。
- `tp`：张量并行卡数，决定 GQA 的 KV 头切分与存储带宽均分。

## 用法

```bash
ucm-toolkit run ttft-analyze \
  --model-dir /home/models/Qwen2.5-14B-Instruct \
  --posix-bw 12 \
  --h2d-bw 60 \
  --input-len 2048 \
  --ttft-prefill 260 \
  --ttft-hbm 3.2 \
  --tp 8
```

两种加载模式（layered / full）的结果会同时输出，便于对比重叠收益。

## 参数

| 参数 | 说明 |
| --- | --- |
| `--model-dir` | 模型目录，读取 `config.json` 推导 kvcache 大小。 |
| `--posix-bw` | 存储总读带宽（GB/s，多卡均分）。 |
| `--h2d-bw` | 单卡 H2D 搬移带宽（GB/s）。 |
| `--input-len` | 输入序列长度（前缀命中长度假设）。 |
| `--ttft-prefill` | 该输入长度下 Full Prefill TTFT（ms）。 |
| `--ttft-hbm` | 该输入长度下 Full HBM Prefix Cache TTFT（ms）。 |
| `--tp` | 张量并行卡数（默认 1）。 |

## 说明

- 按单请求（batch=1）口径计算，dtype 取自 `config.json` 的 `torch_dtype`（默认 bfloat16）。
- 架构自动识别：DSA / MLA / GQA（含 MHA、MQA）。
- 边界输入（带宽 ≤ 0、输入长度 ≤ 0、tp < 1、缺模型配置字段）会明确报错。
