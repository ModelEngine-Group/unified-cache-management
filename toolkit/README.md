# UCM Toolkit 用户文档

`ucm-toolkit` 是 UCM 仓库里的统一工具入口，用来集中调用性能测试、POSIX AIO 测试、物理网卡流量监控等辅助工具。它本身是一个独立 Python 包，不会随主 UCM 包自动安装。

当前顶层工具：

| 工具 | 别名 | 类型 | 功能 |
| --- | --- | --- | --- |
| `dev-sandbox` | `dev_sandbox` | 可构建、可运行 | 构建并运行 C++17 性能测试项目，包含 `copy`、`trans`、`aio` 三个子功能。 |
| `posix-aio` | `posix_aio` | 可运行 | 运行 `ucm/store/test/e2e/posixstore_aio_test.py`，测试 POSIX AIO store 的 dump/load 性能。 |
| `nic-monitor` | `nic_monitor` | 可运行 | 监控物理网卡实时流量、后台采样落盘，并生成阶段统计。 |
| `metrics-view` | `metrics_view`, `terminal-metrics`, `terminal_metrics` | 可运行 | 采集 Prometheus/OpenMetrics 样本到 SQLite，并在终端查询聚合指标。 |

## 安装

推荐在仓库根目录使用 editable 安装：

```bash
python -m pip install -e toolkit
```

安装后确认入口可用：

```bash
ucm-toolkit --help
ucm-toolkit list
```

如果希望隔离环境，可以先创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e toolkit
```

## 依赖

基础 CLI 只依赖 Python 标准库和 `setuptools`。不同工具还需要额外系统依赖：

| 功能 | 依赖 |
| --- | --- |
| `dev-sandbox` 构建 | CMake 3.18+、C++17 编译器。CUDA 后端需要 CUDA runtime；Ascend 后端需要 Ascend runtime。 |
| `dev-sandbox copy` 的 GDR case | CUDA 后端，并且系统能找到 `libibverbs` 头文件和库。 |
| `posix-aio` | 当前 UCM Python 包及其 native 扩展可用，`numpy` 可导入。 |
| `nic-monitor` | Linux、`bash`、`ethtool`，并且需要 root 或 sudo 权限读取网卡统计。 |
| NIC CSV 离线绘图 | `pandas`、`matplotlib`。 |
| `metrics-view` | 仅依赖 Python 标准库（`sqlite3` 内置）；采集需要可访问的 Prometheus/OpenMetrics `/metrics` HTTP 接口。 |

`dev-sandbox` 后端探测优先级：

1. CMake cache 参数：`-DCUDA_ROOT=...` 或 `-DASCEND_ROOT=...`
2. 环境变量：`CUDA_HOME` / `CUDA_PATH` 或 `ASCEND_HOME` / `ASCEND_TOOLKIT_HOME`
3. 默认路径：`/usr/local/cuda` 或 `/usr/local/Ascend/ascend-toolkit/latest`
4. 如果没有发现 GPU runtime，则使用 CPU simulation 后端

## 通用命令

列出顶层工具：

```bash
ucm-toolkit list
ucm-toolkit list --verbose
```

检查工具环境：

```bash
ucm-toolkit doctor
ucm-toolkit doctor dev-sandbox
ucm-toolkit doctor posix-aio
ucm-toolkit doctor nic-monitor
```

构建工具：

```bash
ucm-toolkit build TOOL [tool build args...]
```

目前只有 `dev-sandbox` 支持 `build`。

运行工具：

```bash
ucm-toolkit run TOOL [tool args...]
```

清理工具产物：

```bash
ucm-toolkit clean TOOL
ucm-toolkit clean TOOL --dry-run
```

目前 `clean dev-sandbox` 会删除配置的 build 目录；其他工具默认没有可清理产物。

## metrics-view

`metrics-view` 用于在没有 Prometheus/Grafana 的环境中，从 Prometheus/OpenMetrics
`/metrics` 接口采集原始样本到 SQLite，并在终端查询聚合后的 UCM/vLLM 指标。

默认数据库为 `/tmp/ucm_metrics.db`。采集端默认保存抓到的全部 metrics；查询端默认使用内置
`metrics_lite` 配置，只展示常用的请求数、延迟、cache hit、layerwise wait
blocking 时间和 cache/posix store load/dump 带宽等指标。当前内置 config 只保留
`metrics_lite`。

列出内置配置：

```bash
ucm-toolkit run metrics-view list-configs
```

前台采集一次：

```bash
ucm-toolkit run metrics-view collect \
  --url http://127.0.0.1:8000/metrics \
  --once
```

后台启动采集：

```bash
ucm-toolkit run metrics-view start \
  --url http://127.0.0.1:8000/metrics \
  --interval 5s
```

查看或停止后台采集：

```bash
ucm-toolkit run metrics-view status
ucm-toolkit run metrics-view stop
```

按时间窗口查询。推荐使用 `--aggr-by`，例如 `--window 10m --aggr-by 1m`
会展示这个 10 分钟窗口内每 1 分钟的聚合结果；histogram 指标会显示
`p50`、`p90`、`p99` 和 `avg`。

```bash
ucm-toolkit run metrics-view query \
  --window 10m \
  --aggr-by 1m
```

查询固定历史窗口并按 Prometheus label 过滤：

```bash
ucm-toolkit run metrics-view query \
  --start-time 2026-06-25T10:00:00 \
  --window 10m \
  --aggr-by 1m \
  --tag model_name=qwen \
  --tag worker_id=0
```

如果需要使用其它数据库文件，可以显式指定 `--db`：

```bash
ucm-toolkit run metrics-view query \
  --db /tmp/another_ucm_metrics.db \
  --window 10m \
  --aggr-by 1m
```

清空 metrics 数据库使用 `metrics-view` 自己的 `clean` 子命令：

```bash
ucm-toolkit run metrics-view clean
```

## dev-sandbox

`dev-sandbox` 是 CMake C++17 性能测试项目。toolkit 负责构建项目、定位二进制并转发子命令参数；`copy`、`trans`、`aio` 的业务参数由底层二进制解析。

`dev-sandbox` 在主仓中按 Git subtree 管理，源码路径为：

```text
toolkit/src/dev-sandbox
```

维护 subtree 时使用该路径作为 prefix：

```bash
git subtree pull --prefix=toolkit/src/dev-sandbox dev-sandbox main
git subtree push --prefix=toolkit/src/dev-sandbox dev-sandbox <branch>
```

如果从 `dev-sandbox` 源仓同步其他分支，将上面的 `main` 替换成对应分支名即可。

常见维护流程有两种。

从 `dev-sandbox` 源仓同步更新到主仓：

```bash
git fetch dev-sandbox
git subtree pull --prefix=toolkit/src/dev-sandbox dev-sandbox main
```

这会把 `dev-sandbox/main` 的更新合并到主仓的 `toolkit/src/dev-sandbox` 目录。完成后按主仓正常流程提交并向主仓发起 PR。

把主仓里对 subtree 源码的改动推回 `dev-sandbox` 源仓：

```bash
git subtree push --prefix=toolkit/src/dev-sandbox dev-sandbox <branch>
```

这会把主仓中 `toolkit/src/dev-sandbox` 目录的相关历史拆分出来，并推送到 `dev-sandbox` 远端的 `<branch>`。之后可以在 `dev-sandbox` 源仓中从该分支发起 PR。

如果需要同时做双向同步，推荐顺序是先 `git subtree pull` 保持本地 subtree 最新，再修改 `toolkit/src/dev-sandbox` 并提交到主仓；这些改动若也需要回贡献给独立 `dev-sandbox` 仓，再执行 `git subtree push` 到源仓分支。

### 构建

默认构建到：

```text
toolkit/src/dev-sandbox/build
```

常用命令：

```bash
ucm-toolkit build dev-sandbox
ucm-toolkit build dev-sandbox --build-type Debug
ucm-toolkit build dev-sandbox --build-type Release --jobs 16
```

指定 CUDA 或 Ascend runtime：

```bash
ucm-toolkit build dev-sandbox \
  --cmake-arg -DCUDA_ROOT=/usr/local/cuda

ucm-toolkit build dev-sandbox \
  --cmake-arg -DASCEND_ROOT=/usr/local/Ascend/ascend-toolkit/latest
```

指定构建目录：

```bash
ucm-toolkit build dev-sandbox \
  --build-dir toolkit/build/dev-sandbox/release \
  --build-type Release \
  --jobs 16
```

`--build-dir` 构建成功后会写回 adapter 中的 `build_dir` 字段；之后 `ucm-toolkit run dev-sandbox ...` 会从该目录查找二进制。

构建参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--build-type` | `Release` | 传给 CMake 的 `CMAKE_BUILD_TYPE`。 |
| `--jobs`, `-j` | 不设置 | 传给 `cmake --build` 的并行度。 |
| `--build-dir` | `toolkit/src/dev-sandbox/build` | 覆盖构建输出目录。 |
| `--cmake-arg` | 空 | 额外 CMake configure 参数，可重复传入。 |

### 运行子功能

查看子功能：

```bash
ucm-toolkit run dev-sandbox --help
```

可用子功能：

| 子功能 | 二进制 | 功能 |
| --- | --- | --- |
| `copy` | `module/copy/copy` | 设备/主机内存 copy 性能测试。测量不同内存类型（普通主机、锁页、匿名、设备）之间、不同传输引擎（CE、SM、GDR）之下的带宽，适用于评估 H2D/D2H/D2D 各路径的吞吐。 |
| `trans` | `module/trans/trans` | host/device 传输矩阵性能测试。以方向（H2D/D2H）× host buffer 类型 × 传输方法 构成组合矩阵，批量运行所有匹配 case，适用于快速扫描所有传输路径的带宽分布。 |
| `aio` | `module/aio/aio` | 异步 I/O 磁盘写读性能测试。在指定 workspace 中创建块文件，通过 Linux AIO 对磁盘执行 dump（写）和 load（读），测量磁盘带宽，适用于评估 UCM POSIX store 的磁盘吞吐。 |

### copy

示例：

```bash
ucm-toolkit run dev-sandbox copy -t host_to_device_ce -s 16K -n 512 -i 128 -d 8
ucm-toolkit run dev-sandbox copy -t host_to_device_ce -t device_to_host_ce -s 1M
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-t <name>` | 必填 | case 名称，可重复指定多个。 |
| `-s <size>` | `512M` | 单个数据块大小，只接受 `K/k` 或 `M/m` 后缀，例如 `16K`、`1M`。 |
| `-n <count>` | `8` | 每个 buffer 中的数据块数量。 |
| `-f`, `--frags`, `-frags <count>` | `0` | FFTS direct H2D 的每个 IO/task fragment 数。设置后 `-n` 表示 IO/task 数。 |
| `-i <count>` | `128` | 迭代次数。 |
| `-d <count>` | `8` | 设备数量。 |

当前 `copy` 原生程序没有把 `-h/--help` 做成成功返回的帮助参数。无参数运行会打印 usage 并非 0 退出；指定不存在的 case 会列出当前后端编译进来的全部 case：

```bash
ucm-toolkit run dev-sandbox copy -t unknown
```

常见 case：

| 后端 | case | 说明 |
| --- | --- | --- |
| **CUDA / Ascend** | `host_to_device_ce` | 单流 CE DMA 从普通主机内存拷到设备内存。**场景**：评估单卡 H2D CE 基础带宽，适合作为 H2D 性能基线。 |
| | `host_to_device_batch_ce` | 批量 CE DMA 从主机到设备。**场景**：评估批量提交 H2D 是否比逐个提交更高效，适合对比单流与 batch 启动开销。 |
| | `one_host_to_all_device_ce` | 同一份主机数据通过 CE 广播到所有设备。**场景**：评估模型加载时同一参数分发到多卡的性能。 |
| | `all_host_to_all_device_ce` | 多卡各自 host buffer 同时通过 CE 拷到各自设备。**场景**：评估多 worker 并发 H2D 的总吞吐，适合推测多卡并发加载的实际带宽。 |
| | `device_to_device_ce` | 同卡内 D2D CE 拷贝（src 与 dst 在同一张卡上）。**场景**：评估单卡内部设备内存搬移带宽，适合评估 GPU/Ascend 设备端数据重排或内存池内部搬移性能。跨卡 D2D 请使用 `one_device_to_all_device_ce`。 |
| | `one_device_to_all_device_ce` | 单卡数据通过 CE 广播到所有设备（含自身），跨卡 D2D。**场景**：评估跨卡 D2D 传输带宽，适合多卡 scatter 通信性能基线。 |
| | `anonymous_to_device_ce` | 匿名锁页内存（mmap 分配但未显式注册）通过 CE 拷到设备。**场景**：对比匿名锁页内存与普通主机内存的 H2D 性能差异，适合选择 host buffer 分配策略。 |
| **CUDA** | `device_to_host_ce` | 单流 CE DMA 从设备内存拷到普通主机内存。**场景**：评估单卡 D2H CE 基础带宽，适合作为 D2H 性能基线。 |
| | `device_to_host_batch_ce` | 批量 CE DMA 从设备到主机。**场景**：评估批量 D2H 是否比逐个回读更高效，适合结果回传优化。 |
| | `host_to_device_sm` | SM kernel 将主机数据拷到设备。**场景**：对比 SM kernel 传输与 CE DMA 的带宽差异，适合决定是否用 SM 替代 CE。 |
| | `device_to_host_sm` | SM kernel 将设备数据拷到主机。**场景**：对比 SM kernel 回读与 CE 回读的带宽差异。 |
| | `one_host_to_all_device_sm` | 同一份主机数据通过 SM 广播到所有设备。**场景**：对比 SM 广播与 CE 广播的多卡分发性能。 |
| | `device_to_anonymous_ce` | CE DMA 从设备拷到匿名锁页内存。**场景**：评估 D2H 到匿名锁页内存的带宽，适合回读到 mmap buffer 的场景。 |
| | `anonymous_to_device_sm` | SM kernel 将匿名锁页内存数据拷到设备。**场景**：评估匿名锁页 + SM 组合的 H2D 带宽，适合与 CE 版本对比。 |
| | `device_to_anonymous_sm` | SM kernel 将设备数据拷到匿名锁页内存。**场景**：评估匿名锁页 + SM 组合的 D2H 带宽。 |
| **Ascend** | `host_to_device_ce_multi_stream` | 4 流并发 CE DMA 从主机到设备。**场景**：评估 Ascend 多流并行传输是否能提升 H2D 吞吐，适合多流调度优化。 |
| | `one_share_host_to_all_device_ce_multi_stream` | 一块 POSIX shared memory host buffer 通过 fork fan-out 到所有 device，单卡内使用 4-stream CE。**场景**：模拟 MLA 模型中多卡同时读取同一份 shared host KV 数据并写入各自 device。 |
| | `all_host_to_all_device_ce_multi_stream` | 多卡各自 host buffer，通过 fork fan-out 并在每张卡内使用 4-stream CE。**场景**：评估多进程、多卡、multi-stream H2D 聚合吞吐。 |
| | `all_odirect_host_to_all_device_ce_multi_stream` | 多卡各自 UCM O_DIRECT 风格 mmap host buffer，通过 fork fan-out 和 4-stream CE 拷到设备。**场景**：更贴近开启 O_DIRECT 后 GQA 模型中每张卡从本地 host buffer 同时读入 KV 数据的路径。 |
| | `all_host_to_all_device_ffts_direct_h2d` | 多卡各自 mapped `aclrtMallocHost` buffer，通过 FFTS Plus direct H2D SDMA 拷到设备。**场景**：评估 direct H2D SDMA 的常规 pinned host 源。 |
| | `one_share_host_to_all_device_ffts_direct_h2d` | 一块 POSIX shared memory host buffer 在子进程中 mapped/pinned register 后通过 FFTS Plus direct H2D SDMA 分发到所有 device。**场景**：模拟 MLA 模型中多卡同时读取同一份 shared host KV 数据，验证 FFTS direct H2D 的共享源读入路径。 |
| | `all_odirect_host_to_all_device_ffts_direct_h2d` | 多卡各自 UCM O_DIRECT 风格 mmap host buffer，mapped + pinned register 后通过 FFTS Plus direct H2D SDMA 拷到设备。**场景**：更贴近开启 O_DIRECT 后 GQA 模型中每张卡从本地 host buffer 同时读入 KV 数据的 direct H2D 路径。 |
| **CUDA + libibverbs** | `host_to_device_gdr` | GPUDirect RDMA 直传主机数据到单卡设备内存。**场景**：评估 RDMA 直传到 GPU 是否比传统 CE 更快，适合 RDMA 通信基线。 |
| | `one_host_to_all_device_gdr` | 同一份主机数据通过 GDR 广播到所有设备。**场景**：评估多卡 RDMA 直传的分发性能，适合对比 GDR 广播与 CE 广播。 |
| | `all_host_to_all_device_gdr` | 多卡各自 host buffer 同时通过 GDR 拷到各自设备。**场景**：评估多 worker 并发 GDR 的总吞吐，适合大规模 RDMA 并行传输基线。 |
| **Simulation** | `host_to_anonymous_memcpy` | CPU memcpy 从主机拷到匿名内存。**场景**：在无 GPU 环境下模拟 H2D 传输，用于功能验证或 CPU memcpy 基准对照。 |
| | `shm_to_all_host_memcpy` | CPU memcpy 从共享内存拷到所有 host buffer。**场景**：评估共享内存到各 host 的拷贝带宽，模拟跨进程数据分发。 |

GDR case 使用 `GDR_NICS` 指定 device 与 RDMA 网卡映射，网卡数量需要与 `-d` 一致：

```bash
GDR_NICS=mlx5_0,mlx5_2,mlx5_4,mlx5_6,mlx5_8,mlx5_10,mlx5_12,mlx5_14 \
ucm-toolkit run dev-sandbox copy -t all_host_to_all_device_gdr -s 16K -n 512 -i 128 -d 8
```

### trans

`trans` 以方向（H2D/D2H）× host buffer 类型 × device buffer 类型 × 传输方法构成组合矩阵，批量运行所有匹配的 case。与 `copy` 的区别在于：`copy` 侧重单一路径的详细性能，`trans` 侧重快速扫描所有组合的带宽分布。

host buffer 类型含义：

| 类型 | 说明 | 场景 |
| --- | --- | --- |
| `normal` | 普通可分页主机内存（malloc 分配）。DMA 传输时可能发生页缺失，带宽通常最低。 | 评估最常见的 malloc 场景，作为对比基线。 |
| `anonymous` | 匿名锁页内存（mmap 分配，未显式注册）。页缺失减少，DMA 带宽高于 normal。 | 评估 mmap 锁页分配对传输的改善。 |
| `registered` | 显式注册锁页内存（cudaHostRegister / Ascend memory register）。DMA 传输最高效，带宽最高。 | 评估注册锁页内存的传输上限，适合高性能场景选择 host buffer 分配策略。 |

传输方法含义：

| 方法 | 后端 | 说明 | 场景 |
| --- | --- | --- | --- |
| `ce` | CUDA / Ascend | Copy Engine DMA 硬件搬移，不占用 SM/计算资源。 | 评估 DMA 基础带宽，适合后台搬移数据。 |
| `batch_ce` | CUDA / Ascend | 批量提交 Copy Engine，减少多次启动开销。 | 评估批量 DMA 是否比逐个 CE 更高效。 |
| `sm` | CUDA | SM kernel 传输，占用 GPU 计算资源。 | 对比 SM kernel 与 CE DMA 的带宽差异，适合决定传输引擎选择。 |
| `ms_48` | Ascend | 48 流并发 Copy Engine。 | 评估多流并行 DMA 是否能提升吞吐，适合多流调度优化。 |
| `memcpy` | Simulation | CPU memcpy。 | 在无 GPU 环境下模拟传输，用于功能验证或基准对照。 |

示例：

```bash
ucm-toolkit run dev-sandbox trans -h
ucm-toolkit run dev-sandbox trans -t H2D -H normal -D normal -M ce -s 32768 -n 1024 -d 8 -i 1024
ucm-toolkit run dev-sandbox trans -M ce -M batch_ce
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-H <host>` | 全部 | host buffer 类型，可重复。常见值：`normal`、`anonymous`、`registered`。 |
| `-D <device>` | 全部 | device buffer 类型，可重复。常见值：`normal`。 |
| `-M <method>` | 全部 | 传输方法，可重复。常见值：`ce`、`batch_ce`、`sm`、`ms_48`、`memcpy`，具体取决于后端。 |
| `-t <type>` | `ANY` | 传输方向，支持 `H2D` 或 `D2H`；不传则运行两个方向中匹配的 case。 |
| `-s <size>` | `32768` | 单个传输大小，单位 bytes。 |
| `-n <number>` | `1024` | 数据项数量。 |
| `-d <nDevice>` | `8` | 设备数量。 |
| `-i <nIter>` | `1024` | 迭代次数。 |
| `-h` | - | 显示原生帮助。 |

如果筛选条件没有匹配到 case，程序会打印当前后端全部可用 case。

### aio

`aio` 在指定 workspace 中创建/打开块文件，通过 Linux AIO 对磁盘执行异步写（dump）和异步读（load），测量磁盘 I/O 带宽。与 `copy`/`trans` 测量设备内存带宽不同，`aio` 测量的是磁盘吞吐。

host buffer 分配策略含义：

| 策略 | 说明 | 场景 |
| --- | --- | --- |
| `mmap` | 通过 `mmap` 分配主机内存，页对齐且可被 AIO 直接引用。适合大块 I/O，内存分配开销小。 | 评估 mmap 分配下的磁盘带宽，适合大块连续读写。 |
| `alloc` | 通过设备特定锁页分配（CUDA: `cudaMallocHost`，Ascend: `aclrtMallocHost`）分配主机内存。内存是锁页的，DMA 传输更高效但分配开销更大。 | 评估锁页内存下的磁盘带宽，适合需要将磁盘数据直接 DMA 搬移到设备的场景。 |

示例：

```bash
mkdir -p /tmp/ucm-aio
ucm-toolkit run dev-sandbox aio --workspace /tmp/ucm-aio

ucm-toolkit run dev-sandbox aio \
  --workspace /tmp/ucm-aio \
  --io-type mmap \
  --io-size 1048576 \
  --io-number 512 \
  --device-id 0 \
  --epoch-number 32
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--workspace <path>` | 必填 | AIO 测试工作目录。 |
| `--io-type mmap\|alloc` | `mmap` | host buffer 分配策略。 |
| `--io-size <bytes>` | `1048576` | 每个 I/O shard 大小，单位 bytes。 |
| `--io-number <n>` | `512` | I/O shard 数量。 |
| `--device-id <id>` | `0` | 使用的设备 ID。 |
| `--epoch-number <n>` | `32` | 写和读各自执行的轮数。 |
| `-h`, `--help` | - | 显示原生帮助。 |

## posix-aio

`posix-aio` 调用仓库中的 `ucm/store/test/e2e/posixstore_aio_test.py`，通过 `UcmPipelineStore` 和 `posix_io_engine=aio` 做 dump/load 性能测试。

`ucm-toolkit run posix-aio` 默认会优先使用当前 Python 环境中已经安装的 `ucm` 包；如果当前环境找不到安装版
`ucm`，才会把主仓源码根目录加入子进程 `PYTHONPATH`。当使用安装版 `ucm` 时，toolkit 会从子进程
`PYTHONPATH` 中移除主仓源码根目录，避免源码目录覆盖已安装包。如果需要显式切换导入来源，可以设置：

```bash
# 强制使用当前主仓源码中的 ucm 包
UCM_TOOLKIT_POSIX_AIO_IMPORT=source ucm-toolkit run posix-aio

# 强制使用当前 Python 环境中已安装的 ucm 包
UCM_TOOLKIT_POSIX_AIO_IMPORT=installed ucm-toolkit run posix-aio
```

示例：

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

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-w`, `--worker-number` | `1` | worker number: number of worker processes to start concurrently. |
| `-s`, `--shard-size` | `8388608` | shard size: POSIX store I/O size. In layerwise mode, this is the K/V tensor size for one layer of one block. In non-layerwise mode, this is the K/V tensor size for all layers of one block. |
| `-n`, `--shard-number` | `1` | shard number: number of layers in layerwise mode; use 1 in non-layerwise mode. |
| `-b`, `--block-number` | `64` | block number: total number of blocks. |
| `-d`, `--dump-epoch-number` | `32` | dump epoch number: number of dump epochs. |
| `-l`, `--load-epoch-number` | `32` | load epoch number: number of load epochs. |
| `-o`, `--storage-backend` | `./build/data` | storage backend: storage backend path; may be repeated. Passing this option replaces the default backend list with the provided values. |

资源估算：

```text
单 worker 数据量约为 shard-size * shard-number * block-number
总数据量约为 worker-number * shard-size * shard-number * block-number
```

## nic-monitor

`nic-monitor` 监控 Linux 物理网卡。脚本通过 `/sys/class/net` 找物理网卡，通过 `ethtool` 优先读取厂商统计计数器，失败时回退到 `/proc/net/dev`。

因为需要访问 `ethtool` 统计，通常需要 root 或 sudo：

```bash
sudo ucm-toolkit run nic-monitor fg
```

### 前台模式

前台模式实时刷新终端，不落盘：

```bash
sudo ucm-toolkit run nic-monitor fg
sudo ucm-toolkit run nic-monitor fg 5
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `fg [interval_sec]` | `2` | 前台刷新间隔，单位秒。 |

按 `Ctrl+C` 停止。

### 后台模式

后台模式会创建 `.log`、`.csv`、`.pid` 三类文件，并把采样数据持续写入 CSV：

```bash
sudo ucm-toolkit run nic-monitor bg
sudo ucm-toolkit run nic-monitor bg 24 5
sudo ucm-toolkit run nic-monitor bg 24 5 --log-dir /mnt/test/net_log
sudo ucm-toolkit run nic-monitor bg 24 5 --log-dir /mnt/test/net_log --stat-cycle-seconds 600
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `bg [duration_hours] [interval_sec]` | `12 10` | 后台运行时长和采样间隔。 |
| `--log-dir PATH` | 当前工作目录下的 `net_log` | 后台日志、CSV、PID 输出目录。 |
| `--stat-cycle-seconds SECONDS` | `3600` | 阶段统计周期，单位秒。 |

输出文件名格式：

```text
Eth_Perf_Monitor_YYYYmmdd_HHMMSS.log
Eth_Perf_Monitor_YYYYmmdd_HHMMSS.csv
Eth_Perf_Monitor_YYYYmmdd_HHMMSS.pid
```

后台启动时会检查同一 `--log-dir` 下是否已有存活的 `.pid` 进程，避免重复启动。

## NIC 结果可视化

`toolkit/src/nic_monitor` 还提供两个离线可视化入口。它们当前没有注册为 `ucm-toolkit run` 顶层工具。

### Python 绘图

安装依赖：

```bash
python -m pip install pandas matplotlib
```

生成 PNG 图表：

```bash
python toolkit/src/nic_monitor/visualize_traffic.py net_log/Eth_Perf_Monitor_*.csv -o net_log/charts
python toolkit/src/nic_monitor/visualize_traffic.py net_log/Eth_Perf_Monitor_*.csv -o net_log/charts -i eth0 eth1
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `csv` | 当前目录下 `*.csv` | 输入 CSV，可传多个。 |
| `-o`, `--output` | `.` | 图片输出目录。 |
| `-i`, `--interfaces` | 全部网卡 | 只绘制指定网卡，可传多个名称。 |

每个 CSV 会生成一个同名子目录，包含流量时序、利用率时序、总流量堆叠、统计摘要等 PNG。

### 浏览器页面

可以直接打开：

```text
toolkit/src/nic_monitor/index.html
```

页面支持上传或拖拽 `nic-monitor bg` 生成的 CSV，并在浏览器中查看交互式图表。

## 常见问题

### `command not found: ucm-toolkit`

确认已经安装：

```bash
python -m pip install -e toolkit
```

如果使用虚拟环境，确认虚拟环境已激活。

### `dev-sandbox` 找不到二进制

先构建：

```bash
ucm-toolkit build dev-sandbox
ucm-toolkit doctor dev-sandbox
```

如果之前用 `--build-dir` 指定了其他目录，`run` 会从 adapter 当前记录的 build 目录查找二进制。

### 想切换 CUDA 或 Ascend 后端

重新指定 runtime 并构建：

```bash
ucm-toolkit build dev-sandbox --cmake-arg -DCUDA_ROOT=/usr/local/cuda
ucm-toolkit build dev-sandbox --cmake-arg -DASCEND_ROOT=/usr/local/Ascend/ascend-toolkit/latest
```

也可以通过 `CUDA_HOME`、`CUDA_PATH`、`ASCEND_HOME`、`ASCEND_TOOLKIT_HOME` 环境变量影响 CMake 探测。

### `nic-monitor` 权限失败

用 root 或 sudo 运行：

```bash
sudo ucm-toolkit run nic-monitor fg
sudo ucm-toolkit run nic-monitor bg 12 10
```

同时确认系统安装了 `ethtool`。
