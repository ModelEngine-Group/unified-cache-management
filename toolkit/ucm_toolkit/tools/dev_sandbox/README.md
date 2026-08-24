# dev-sandbox

dev-sandbox 测量 Ascend 主机内存到设备显存的拷贝带宽，覆盖普通内存、O_DIRECT、共享内存等分配方式与多流 CE、FFTS direct H2D SDMA 等传输引擎，并支持 GQA（每卡各自本地内存）和 MLA（单块共享内存分发到所有卡）两种拓扑。

← 返回 [UCM Toolkit 顶层文档](../../../README.md)

## IO 流与 UCM KV cache 的关系

UCM 采用分层 KV cache：设备显存（HBM，最快）→ 主机内存（layer cache，次之）→ 磁盘（POSIX store，最慢）。当请求所需 KV 数据未命中显存时，需要从主机内存（或磁盘）加载回显存，这一步 **主机→设备（H2D）拷贝** 正是 dev-sandbox `copy` 测量的环节；`aio` 子功能则单独测量磁盘 dump/load 带宽。

```text
┌──────────────────── UCM 分层 KV cache ────────────────────┐
│                                                           │
│  ┌──────────────┐     H2D 拷贝       ┌────────────────┐  │
│  │  主机内存     │ ════════════════▶ │ 设备显存 (HBM)  │  │
│  │ (layer cache)│   CE 多流 / SDMA   │  (KV cache)    │  │
│  └──────┬───────┘                    └────────────────┘  │
│         │ dump/load                                       │
│  ┌──────▼───────┐                                       │
│  │ 磁盘 (POSIX)  │ ◀── aio 子功能测量磁盘带宽             │
│  └──────────────┘                                       │
└───────────────────────────────────────────────────────────┘
```

不同的传输引擎代表 H2D 阶段不同的硬件拷贝路径：

| 传输引擎 | 含义 | 对应 UCM 环节 |
| --- | --- | --- |
| 多流 CE（`--sdma false`） | Copy Engine 硬件 DMA，4 流并发提交以提升吞吐。 | 常规 H2D 加载路径：从主机 layer cache 把 KV 块拷回显存。 |
| FFTS direct H2D SDMA（`--sdma true`） | 走 SDMA 的 direct H2D 直通路径，配合 FFTS 快速任务调度，延迟更低。 | 对 direct H2D 直通路径的 KV 加载带宽评估。 |

主机内存分配方式对应不同的 KV 来源/拓扑：

| 分配方式 | 含义 | 拓扑 |
| --- | --- | --- |
| 普通主机内存（`--iodirect false`） | 锁页主机内存（pinned），每卡各自一份。 | GQA：每卡从本地 host buffer 加载各自 KV。 |
| O_DIRECT（`--iodirect true`） | O_DIRECT 风格 mmap 主机内存，绕过页缓存，贴近 UCM 直接读文件的路径。 | GQA：开启 O_DIRECT 后每卡的本地 KV 读入路径。 |
| 共享内存 | 一块 POSIX shared memory host buffer，经 fork fan-out 分发到所有卡。 | MLA：多卡共享同一份 host KV 数据。 |

两种拓扑的数据流：

```text
GQA：每卡各自本地 host buffer → 各自显存（每卡独立一路 H2D）

  host buf(卡0) ── 多流 CE / SDMA ──▶ device HBM(卡0)
  host buf(卡1) ── 多流 CE / SDMA ──▶ device HBM(卡1)
  host buf(卡2) ── 多流 CE / SDMA ──▶ device HBM(卡2)
   ...                                  ...
  host buf(卡N) ── 多流 CE / SDMA ──▶ device HBM(卡N)

MLA：一块共享 host buffer fan-out 到所有卡

                 ┌──▶ device HBM(卡0)
 shared host buf ┼──▶ device HBM(卡1)   (fork fan-out，每卡多流 CE / SDMA)
                 └──▶ device HBM(卡N)
```

## 快速使用

```bash
ucm-toolkit run dev-sandbox --model-type <gqa|mla> --iodirect <true|false> --sdma <true|false> [参数...]
```

三个选择参数：

| 参数 | 取值 | 含义 |
| --- | --- | --- |
| `--model-type` | `gqa` / `mla` | 模型类型。 |
| `--iodirect` | `true` / `false` | 是否开启 O_DIRECT（直接 IO，绕过页缓存）。 |
| `--sdma` | `true` / `false` | 是否走 SDMA（direct H2D SDMA 传输路径）。 |

三个参数组合决定测试场景：

| 模型 | IO-direct | SDMA | 测试场景 |
| --- | --- | --- | --- |
| `gqa` | `false` | `false` | 多卡各自从本地主机内存，用多流 CE 拷到显存。 |
| `gqa` | `true` | `false` | 多卡各自用 O_DIRECT 方式从本地主机内存，多流 CE 拷到显存。 |
| `gqa` | `false` | `true` | 多卡各自从主机内存，用 FFTS direct H2D SDMA 拷到显存。 |
| `gqa` | `true` | `true` | 多卡各自用 O_DIRECT 方式从主机内存，走 FFTS direct H2D SDMA 拷到显存。 |
| `mla` | `false` | `false` | 一块共享主机内存在多卡间分发，每卡用多流 CE 拷到各自显存。 |
| `mla` | `true` | `false` | 同上（MLA 不区分 iodirect，走同一 CE 路径）。 |
| `mla` | `false` | `true` | 一块共享主机内存，用 FFTS direct H2D SDMA 分发到各卡显存。 |
| `mla` | `true` | `true` | 同上（MLA 不区分 iodirect，走同一 SDMA 路径）。 |

## 运行参数

三个选择参数之后可以接以下参数（全部可选，不传则用默认值）：

| 参数 | 含义 | 默认值 | 示例 |
| --- | --- | --- | --- |
| `-s` | 单个数据块大小 | `512M` | `-s 16K`、`-s 1M` |
| `-n` | 数据块数量 | `8` | `-n 512` |
| `-i` | 迭代轮数 | `128` | `-i 128` |
| `-d` | 设备（卡）数量 | `8` | `-d 8` |
| `-f` | 分片数（仅 SDMA 场景相关） | `0` | `-f 4` |

## 示例

```bash
# 测 GQA + 多流 CE：16K 块 × 512 块 × 128 轮 × 8 卡
ucm-toolkit run dev-sandbox --model-type gqa --iodirect false --sdma false \
  -s 16K -n 512 -i 128 -d 8

# 用默认参数测 MLA 共享分发 + SDMA
ucm-toolkit run dev-sandbox --model-type mla --iodirect false --sdma true
```

### 输出示例

下面是两条命令在 8 卡 Ascend 910B3 上的真实输出（具体数值因机器而异）：

GQA + 多流 CE（每卡各自本地 host buffer → 各自显存）：

```text
[[ all_host_to_all_device_ce_multi_stream ]] memcpy from all host to all device with ce using multi stream and fork submit
  From              To                Method    Size(KB)  Count   Submit(us)-(Min/Max/Avg/P50/P90)        Copy(us)-(Min/Max/Avg/P50/P90)              BW(GB/s)
  acl::host::all    acl::device::all  CE-MS-FORK16        4096    878 / 2576 / 1117 / 1112 / 1269         3155 / 4101 / 3736 / 3759 / 3905            16.729
```

MLA + 多流 CE（一块共享 host buffer fan-out 到所有卡）：

```text
[[ one_share_host_to_all_device_ce_multi_stream ]] memcpy from one shared host to all device with ce using multi stream and fork submit
  From              To                Method    Size(KB)  Count   Submit(us)-(Min/Max/Avg/P50/P90)        Copy(us)-(Min/Max/Avg/P50/P90)              BW(GB/s)
  acl::shm::0       acl::device::all  CE-MS-FORK16        4096    1056 / 1270 / 1162 / 1161 / 1209        3914 / 5024 / 4471 / 4424 / 4852            13.979
```

MLA 用 `acl::shm::0` 作为数据源（共享内存），GQA 用 `acl::host::all`（多卡各自主机内存）；两者 `Count` 均为“卡数 × `-n`”（8 × 512 = 4096）。`--sdma true` 对应的 FFTS direct H2D SDMA 路径需要机器支持 FFTS 特性，输出格式与上表一致，仅 `Method`/`Copy` 数值不同。

输出字段释义：

| 字段 | 含义 |
| --- | --- |
| 标题行 `[[ ... ]]` | 实际执行的 case 名称与简介，可用来确认 `--model-type`/`--iodirect`/`--sdma` 映射到了哪个 case。 |
| `From` / `To` | 数据源 / 目的地内存类型。`acl::host::all` = 多卡各自主机内存，`acl::shm::0` = 共享内存，`acl::device::all` = 所有卡显存。 |
| `Method` | 传输引擎，如 `CE-MS-FORK` = 多流 CE + fork fan-out（列宽恰好被方法名填满，故输出中 `CE-MS-FORK` 与紧随其后的 `Size(KB)` 数值连写为 `CE-MS-FORK16`）。 |
| `Size(KB)` | 单个数据块大小，对应 `-s`（如 `-s 16K` → `16`）。 |
| `Count` | 实际拷贝的数据块总数。多卡 fork 场景下为“卡数 × `-n`”（如 8 卡 × 512 = 4096）。 |
| `Submit(us)` | 提交阶段耗时统计（微秒），按 Min/Max/Avg/P50/P90 给出。 |
| `Copy(us)` | 实际拷贝耗时统计（微秒），按 Min/Max/Avg/P50/P90 给出，是计算带宽的基准。 |
| `BW(GB/s)` | 聚合带宽 = `Size × Count × 1e6 / Copy.avg`，单位 GB/s，越大越好。 |

## 首次使用

首次运行前需先构建项目：

```bash
ucm-toolkit build dev-sandbox
```

构建细节（后端选择、自定义路径等）、底层原生子命令（`copy`/`trans`/`aio`）与完整 case 表见 [开发者文档](./developer.md)。
