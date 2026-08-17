# dev-sandbox 开发者文档

> 面向开发者的参考文档：构建、原生子命令（`copy`/`trans`/`aio`）、完整 case 表与底层参数。日常使用见 [README.md](./README.md)。

CMake C++17 性能测试项目。toolkit 负责构建项目、定位二进制并转发子命令参数；`copy`、`trans`、`aio` 的业务参数由底层二进制解析。

← 返回 [UCM Toolkit 顶层文档](../../../README.md)

## 依赖

| 功能 | 依赖 |
| --- | --- |
| `dev-sandbox` 构建 | CMake 3.18+、C++17 编译器。CUDA 后端需要 CUDA runtime；Ascend 后端需要 Ascend runtime；`copy` 的 GDR case 还需要 `libibverbs` 头文件和库。 |

## 构建

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

通过 `--cmake-arg` 显式指定（优先级最高）：

```bash
ucm-toolkit build dev-sandbox \
  --cmake-arg -DCUDA_ROOT=/usr/local/cuda

ucm-toolkit build dev-sandbox \
  --cmake-arg -DASCEND_ROOT=/usr/local/Ascend/ascend-toolkit/latest
```

也可以通过环境变量指定 root，效果相同：

```bash
# CUDA：CUDA_HOME 或 CUDA_PATH
CUDA_HOME=/usr/local/cuda ucm-toolkit build dev-sandbox

# Ascend：ASCEND_HOME / ASCEND_TOOLKIT_HOME / ASCEND_HOME_PATH
ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest ucm-toolkit build dev-sandbox
```

如果既未用 `--cmake-arg` 指定、也未设置相应环境变量，且机器上探测不到有效的 CUDA/Ascend runtime，则自动回退到 CPU Simulation 后端。

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

## 运行子功能

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

### 快捷模式（model-type / iodirect / sdma）

如果不关心底层 `copy` 的 case 名称，可以直接用 `--model-type`、`--iodirect`、`--sdma` 三个语义参数，通过一张映射表选到对应的 Ascend 拷贝接口；这三个参数之后的其余参数会原样透传给底层 `copy` 二进制（即 `copy` 的全部原生参数 `-s`/`-n`/`-i`/`-d`/`-f`，见下文 [`### copy`](#copy) 一节）。原生的 `copy`/`trans`/`aio` 子命令用法不受影响。

```bash
ucm-toolkit run dev-sandbox \
  --model-type gqa --iodirect false --sdma false \
  -s 16K -n 512 -i 128 -d 8
```

`model-type` / `iodirect` / `sdma` 与 Ascend 拷贝接口的对应关系：

| model-type | iodirect | sdma | 拷贝接口（Ascend） | 场景 |
| --- | --- | --- | --- | --- |
| `gqa` | `false` | `false` | `all_host_to_all_device_ce_multi_stream` | 多卡各自 host buffer，4-stream CE H2D。 |
| `gqa` | `true` | `false` | `all_odirect_host_to_all_device_ce_multi_stream` | 多卡各自 O_DIRECT mmap host buffer，4-stream CE H2D。 |
| `gqa` | `false` | `true` | `all_host_to_all_device_ffts_direct_h2d` | 多卡各自 mapped host buffer，FFTS direct H2D SDMA。 |
| `gqa` | `true` | `true` | `all_odirect_host_to_all_device_ffts_direct_h2d` | 多卡各自 O_DIRECT mmap host buffer，FFTS direct H2D SDMA。 |
| `mla` | `false` | `false` | `one_share_host_to_all_device_ce_multi_stream` | 一块 shared memory host buffer fan-out 到所有卡，4-stream CE。 |
| `mla` | `true` | `false` | `one_share_host_to_all_device_ce_multi_stream` | MLA 不区分 iodirect，与 iodirect=false 走同一 CE 路径。 |
| `mla` | `false` | `true` | `one_share_host_to_all_device_ffts_direct_h2d` | 一块 shared memory host buffer，FFTS direct H2D SDMA 分发。 |
| `mla` | `true` | `true` | `one_share_host_to_all_device_ffts_direct_h2d` | MLA 不区分 iodirect，与 iodirect=false 走同一 SDMA 路径。 |

快捷模式只负责把三个选择器映射成 `-t <case>`，其余参数照搬 `copy` 原生语义（不传则沿用 `copy` 默认值）。例如下面两条命令完全等价：

```bash
# 快捷模式
ucm-toolkit run dev-sandbox --model-type gqa --iodirect false --sdma false -s 16K -n 512 -i 128 -d 8
# 原生模式
ucm-toolkit run dev-sandbox copy -t all_host_to_all_device_ce_multi_stream -s 16K -n 512 -i 128 -d 8
```

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

## 常见问题

### 找不到二进制

先构建：

```bash
ucm-toolkit build dev-sandbox
ucm-toolkit doctor dev-sandbox
```

如果之前用 `--build-dir` 指定了其他目录，`run` 会从 adapter 当前记录的 build 目录查找二进制。

### 切换 CUDA 或 Ascend 后端

重新指定 runtime 并构建：

```bash
ucm-toolkit build dev-sandbox --cmake-arg -DCUDA_ROOT=/usr/local/cuda
ucm-toolkit build dev-sandbox --cmake-arg -DASCEND_ROOT=/usr/local/Ascend/ascend-toolkit/latest
```

也可以通过 `CUDA_HOME`、`CUDA_PATH`、`ASCEND_HOME`、`ASCEND_TOOLKIT_HOME` 环境变量影响 CMake 探测。
