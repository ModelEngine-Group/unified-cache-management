# UCM Toolkit 用户文档

`ucm-toolkit` 是 UCM 仓库里的统一工具入口，用来集中调用性能测试、POSIX AIO 测试、物理网卡流量监控等辅助工具。它本身是一个独立 Python 包，不会随主 UCM 包自动安装。

当前顶层工具：

| 工具 | 别名 | 类型 | 功能 |
| --- | --- | --- | --- |
| `dev-sandbox` | `dev_sandbox` | 可构建、可运行 | 构建并运行 C++17 性能测试项目，包含 `copy`、`trans`、`aio` 三个子功能。 |
| `posix-aio` | `posix_aio` | 可运行 | 运行 `ucm/store/test/e2e/posixstore_aio_test.py`，测试 POSIX AIO store 的 dump/load 性能。 |
| `nic-monitor` | `nic_monitor` | 可运行 | 监控物理网卡实时流量、后台采样落盘，并生成阶段统计。 |

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
| `copy` | `module/copy/copy` | 设备/主机内存 copy 性能测试。 |
| `trans` | `module/trans/trans` | host/device 传输矩阵性能测试。 |
| `aio` | `module/aio/aio` | 异步 I/O 写读性能测试。 |

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
| `-i <count>` | `128` | 迭代次数。 |
| `-d <count>` | `8` | 设备数量。 |

当前 `copy` 原生程序没有把 `-h/--help` 做成成功返回的帮助参数。无参数运行会打印 usage 并非 0 退出；指定不存在的 case 会列出当前后端编译进来的全部 case：

```bash
ucm-toolkit run dev-sandbox copy -t unknown
```

常见 case：

| 后端 | case |
| --- | --- |
| CUDA / Ascend | `host_to_device_ce`、`host_to_device_batch_ce`、`one_host_to_all_device_ce`、`all_host_to_all_device_ce`、`device_to_device_ce`、`one_device_to_all_device_ce`、`anonymous_to_device_ce` |
| CUDA | `device_to_host_ce`、`device_to_host_batch_ce`、`host_to_device_sm`、`device_to_host_sm`、`one_host_to_all_device_sm`、`device_to_anonymous_ce`、`anonymous_to_device_sm`、`device_to_anonymous_sm` |
| Ascend | `host_to_device_ce_multi_stream` |
| CUDA + libibverbs | `host_to_device_gdr`、`one_host_to_all_device_gdr`、`all_host_to_all_device_gdr` |
| Simulation | `host_to_anonymous_memcpy`、`shm_to_all_host_memcpy` |

GDR case 使用 `GDR_NICS` 指定 device 与 RDMA 网卡映射，网卡数量需要与 `-d` 一致：

```bash
GDR_NICS=mlx5_0,mlx5_2,mlx5_4,mlx5_6,mlx5_8,mlx5_10,mlx5_12,mlx5_14 \
ucm-toolkit run dev-sandbox copy -t all_host_to_all_device_gdr -s 16K -n 512 -i 128 -d 8
```

### trans

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

`aio` 会在指定 workspace 中创建/打开块文件，先执行写任务，再执行读任务。

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
| `--worker-number` | `1` | worker 进程数量；脚本会为每个 worker 分配一个 `device_id`。 |
| `--shard-size` | `8388608` | 单个 shard 大小，单位 bytes。 |
| `--shard-number` | `1` | 每个 block 的 shard 数量。 |
| `--block-number` | `64` | block 数量。 |
| `--dump-epoch-number` | `32` | dump 轮数。 |
| `--load-epoch-number` | `32` | load 轮数。 |
| `--storage-backend` | `./build/data` | 存储路径；可重复指定多个。只要传入该参数，就会用传入列表替换默认列表。 |

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
