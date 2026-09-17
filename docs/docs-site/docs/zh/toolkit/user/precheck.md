# 部署环境预检

`precheck` 是 UCM 的部署环境检查工具。在目标宿主机上运行后，它会汇总软件版本，检查加速器驱动、内核、共享内存和 AIO 资源，并可对指定目录进行存储带宽测试，帮助提前发现可能影响启动或缓存读写的环境问题。

适合在以下情况下使用：

- 首次部署或迁移到新机器时，检查主机条件是否满足预检要求。
- 引擎启动失败或加载缓存异常时，先排查驱动、共享内存和内核资源。
- 准备使用某个缓存目录时，对其读写路径做初步带宽检查。

工具逐项展示检查结果与异常处理建议，也支持 JSON 输出，便于保存和对比环境。版本等信息项只记录现状，其他检查按配置阈值给出 `PASS`、`WARN` 或 `FAIL`。先按[工具集安装说明](../index.md)准备 CLI。

## 开始检查

```bash
ucm-toolkit run precheck --skip-bandwidth
ucm-toolkit run precheck --mount-path /mnt/ucm-cache-test
ucm-toolkit run precheck --only kernel --only accelerator_driver
ucm-toolkit run precheck --skip-bandwidth --json
```

带宽检查会在指定路径执行读写负载。使用独立测试目录并预留空间；未指定路径时跳过该项。纯环境检查依赖 Python 标准库，带宽检查还需要 Linux、NumPy 和可加载的 UCM 原生 Posix Store。

## 如何解释结果

| 检查 | 当前默认判定 | 说明 |
| --- | --- | --- |
| 引擎与 `uc-manager` 版本 | INFO，仅展示 | 记录实际安装环境 |
| 加速器驱动 | 缺少驱动时失败；CUDA 算力至少 8.0；Ascend HDK 高于 25.2.0，较低时警告 | 使用设备查询命令读取 |
| 内核 | 版本系列至少 5.10 | 按 major.minor 比较 |
| `/dev/shm` | 小于 512 GiB 时警告 | 工具的预检阈值，实际 UCM 容量仍按配置和模型计算 |
| AIO 资源 | INFO，仅展示 | 读取内核 AIO 上限和占用 |
| 存储带宽 | 最佳组合聚合带宽低于 8 GB/s 时警告 | 阈值可调整，结果受缓存与负载影响 |

`INFO` 不改变退出码。默认允许警告；`--strict` 将警告视为失败。退出码为 `0`（通过）、`1`（失败，或 strict 下的警告）、`2`（CLI 用法错误）。检查通过后仍需执行引擎启动和[外部缓存验证](../../user-guide/observability/verify-cache.md)。

## 配置阈值与测试负载

配置按以下顺序覆盖：代码后备值 → `precheck.defaults.json` → `--config FILE` → CLI 参数。JSON 无需额外解析库；YAML 配置需要 PyYAML。

示例覆盖文件：

```json
{
  "mount_path": "/mnt/ucm-cache-test",
  "kernel_min": "5.10",
  "cuda_min_compute_cap": 8.0,
  "ascend_min_hdk": "25.2.0",
  "bandwidth": {
    "shard_sizes": ["180k", "1m"],
    "worker_counts": [1, 16],
    "engines": ["psync", "aio"],
    "block_number": 64,
    "shard_number": 1,
    "dump_epochs": 32,
    "load_epochs": 32,
    "threshold_gb": 8.0
  }
}
```

此覆盖配置沿用原示例的负载设置，共享内存阈值使用配置中的默认值。运行时不加 `--quick`，以保留已指定的轮数：

```bash
ucm-toolkit run precheck --config /path/to/precheck.json
```

| 参数 | 用途 |
| --- | --- |
| `--mount-path` | 指定带宽测试目录 |
| `--only` / `--skip` | 选择或排除检查项，可重复使用 |
| `--skip-bandwidth` | 只做环境检查 |
| `--shard-sizes` / `--workers` / `--engines` | 控制分片大小、进程数量和 I/O 引擎 |
| `--modes` | 选择 `dump,read,mix` 中的阶段 |
| `--threshold` | 带宽警告阈值，单位 GB/s |
| `--block-number` / `--dump-epochs` / `--load-epochs` / `--mixed-epochs` / `--rw-ratio` | 控制测试规模和读写比例 |
| `--kernel-min` / `--cuda-min-compute-cap` / `--ascend-min-hdk` | 覆盖环境阈值 |
| `--quick` | 减少各阶段轮数 |
| `--strict` / `--json` / `--no-color` / `--verbose` | 退出码与输出格式 |

## 带宽结果的边界

当前默认扫描 180 KiB 和 8 MiB 分片、1/8/16 个 worker、psync/aio 两种引擎。每个组合执行保存、读取和读多写少的混合阶段；混合比例默认为一次写入对应四次读取，每阶段八轮。

工具报告样本均值、波动和聚合值，优先按混合带宽选择组合，缺少混合数据时使用保存与读取均值。默认非 Direct I/O 读取可能由操作系统页缓存满足，因此应把结果解释为该测试路径的吞吐。测量真实存储介质时需明确 I/O 模式和缓存状态。

原生依赖不可用时，带宽检查会报告跳过和警告。超时与 worker 同步由预检实现管理；默认 `combo_timeout` 为 120 秒。源码入口为 `toolkit/ucm_toolkit/tools/precheck/`，具体参数以 `ucm-toolkit run precheck --help` 为准。

## 开发验证

现有测试检查阈值、配置处理和 Toolkit 分派，不代替硬件带宽验收。

```bash
cd toolkit
python -m unittest tests.test_precheck tests.test_precheck_toolkit -v
```
