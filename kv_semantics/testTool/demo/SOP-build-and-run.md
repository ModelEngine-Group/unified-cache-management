# testTool 真机测试 SOP：构建引导与执行引导

适用场景：在一台新机器（目标环境）上构建 testTool 并**单独验证建链 + region 注册（regionRegister）链路**。
配套文件（本 demo 目录）：

| 文件 | 用途 |
| --- | --- |
| `conf/test_config_onlyconnect.conf` | 只建链模式的 testTool 测试配置（可直接使用） |
| `conf/kv_client_real.conf` | kv client 配置模板（按目标环境改端点） |
| `conf/kv_client_view_real.conf` | ASU 视图配置（与 kv_client_real.conf 联动修改） |

> region 并发使用安全性审查记录存放在文档工作区（D:/AI/testTool/region-concurrency-review.md），不在本目录。

---

## 第一部分：build 引导

### 1. 环境要求

- Linux（x86_64 或 ARM 均可，与工具可移植性约束一致）
- cmake >= 3.18，支持 C++17 的 g++（Ubuntu 24.04 的 build-essential 默认满足）
- git
- 网络：首次配置需在线拉取 fmt / spdlog（FetchContent）。目标机器无外网时，改用离线方式：`-DDOWNLOAD_DEPENDENCE=OFF` 并在 `testTool/` 下预置 `fmt/`、`spdlog/` 源码目录

### 2. 构建 kv client（必须先于 testTool）

testTool 链接 kv client 预构建库（头文件取 `kv_semantics/include`，库取 `kv_semantics/build-kv/src`），且 build-kv 是本机构建产物**不入 git**，新机器必须重新构建。

> CPU 建链断链测试（CPU-over-RoCE 链路）：可直接用一键脚本 `bash demo/scripts/build-cpu-roce.sh` 完成本节与第 3 节的全部构建（前置依赖检测 + RoCE provider 开关均内置于脚本；要求当前分支含 RoCE provider 代码，先参考 cpu-kv-over-roce-provider 分支，联调分支就绪后同样适用）。

以下命令均在**仓库根**（unified-cache-management）执行：

```bash
# 方式一：simu 环境（FAKE 后端，无需真实设备，用于流程走通）
cmake -S kv_semantics -B kv_semantics/build-kv -DRUNTIME_ENVIRONMENT=simu -DCMAKE_BUILD_TYPE=Debug

# 方式二：ascend 环境（真设备）——真机验证用
#   AIV provider 需外部 libumc.a，按需开启：
cmake -S kv_semantics -B kv_semantics/build-kv -DRUNTIME_ENVIRONMENT=ascend -DCMAKE_BUILD_TYPE=Debug -DBUILD_KV_CLIENT_PROVIDER_AIV=ON
#   如需 AICPU provider，改为 -DBUILD_KV_CLIENT_PROVIDER_AICPU=ON

# 编译 kv client（只构建该目标，避免全量构建）
cmake --build kv_semantics/build-kv --target kv_client -j
```

说明：

- `RUNTIME_ENVIRONMENT` 仅接受 `simu` 或 `ascend`（非法值直接 FATAL_ERROR）
- FAKE provider 默认编入（`BUILD_KV_CLIENT_PROVIDER_FAKE` 默认 ON）；AIV / AICPU 默认 OFF，真机传输按环境开启
- 产物：`kv_semantics/build-kv/src/libkv_client.so`（及 libkv_transport.so）

### 3. 构建 testTool

```bash
# 仓库根执行：
cmake -S kv_semantics/testTool -B kv_semantics/testTool/build -DCMAKE_BUILD_TYPE=Debug
cmake --build kv_semantics/testTool/build -j
# 产物：kv_semantics/testTool/build/testTool
```

或进入 `kv_semantics/testTool` 目录后 `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j`（与 README 一致）。

编译选项与主工程一致（`-Wall -Werror`），任何告警都会导致构建失败，属预期行为。

### 4. 备选：Docker 方式

目标机器装有 Docker 时，可用仓库自带开发镜像（Ubuntu 24.04 + 工具链，x86_64 / arm64 通用）：

```bash
cd <仓库根>/kv_semantics
docker build -t testtool-dev testTool/Dockerfile
docker run -d --name testtool-dev -v <仓库根>/kv_semantics:/workspace testtool-dev
docker exec -it testtool-dev bash
# 容器内落地即 /workspace/testTool，按第 2、3 节步骤构建（路径用 /workspace/...）
```

注意：真机验证建链时，若目标网络/设备需要直连，需在 docker run 时按需配置 `--network` 与设备直通参数；无法直通的场景建议直接在宿主机构建运行。

---

## 第二部分：执行引导（真机验证建链 + regionRegister）

### 5. 修改 kv client 配置（唯一需要按环境改的地方）

编辑 `demo/conf/kv_client_real.conf`，三处联动修改：

| 字段 | 说明 |
| --- | --- |
| `transport.provider_type` | `FAKE`（本地模拟，流程验证）→ 真实传输改 `AIV` 或 `AICPU` |
| `transport.asu_ids` / `asu_info.N` | ASU 节点编号与端点；同时改 `kv_client_view_real.conf` 的 `asuIds` / `asu_info.N`，两文件字段必须一致 |
| `transport.device_id` | 按目标机器实际设备编号 |

testTool 侧配置 `demo/conf/test_config_onlyconnect.conf` 一般无需修改；可选调整：

- `buffercount` / `buffersize`：region 注册的块数与单块大小（当前 256 块 × 1MB = 256MB host 内存，按目标环境内存余量调整）
- `buffertype`：`cpu`（主机内存注册，MemoryType=HOST）/ `gpu`（CUDA 显存注册，MemoryType=DEVICE）/ `npu`（ACL NPU 内存注册，MemoryType=DEVICE）。注意：testTool 未编入设备 SDK（`TESTTOOL_ENABLE_CUDA` / `TESTTOOL_ENABLE_ASCEND`）时，`gpu`/`npu` 会回退主机内存模拟（打 WARN 日志），注册类型仍按 DEVICE 走链路

### 6. 启动命令（只建链验证）

```bash
cd <仓库根>/kv_semantics/testTool
./build/testTool demo/conf/test_config_onlyconnect.conf demo/conf/kv_client_real.conf
```

- 只建链模式 **input 文件不传**（传入也会被忽略）
- 必须在 `testTool` 目录下启动：`view.config_path` 等相对路径相对进程工作目录

### 7. 验证观察点（按日志时序）

进程起来后依次确认（testTool 全量日志落盘 `log_demo/nds_*.log`；stdout 只有 kv client 库的建链日志与进程级 `[FAIL]` 错误输出）：

1. `======Test Starting=======` —— 启动标记（warn 级，任何 perf 配置下均可见）
2. kv client 建链日志（`AsuTransportImpl::Init ...` 系列）—— 真机端点连通性在此体现
3. `Scheduler: 256 value buffers registered (1048576 bytes each, type=host)` —— **regionRegister 成功的直接证据**（块数、单块字节数、注册内存类型）
4. `onlyconnect=on: connection established, waiting for Ctrl+C to exit...` —— 建链与注册全部完成，进程挂起等待
5. 需要结束时 `Ctrl+C`，日志出现 `onlyconnect=on: Ctrl+C received, exiting`，进程退出码 0

任一步缺失或进程提前退出：看 stdout 的 `[FAIL]` 行与 `log_demo/nds_error_*.log`；建链失败优先核查 kv_client_real.conf 端点字段与真实环境是否一致。

### 8. 进阶：全链路验证（建链 + IO）

建链验证通过后，可切换执行模式跑完整 IO：

```bash
# 复制 demo 配置改 onlyconnect = off 后使用（不要动仓库默认 conf/test_config.conf）
cd <仓库根>/kv_semantics/testTool
./build/testTool <改好的 test conf> demo/conf/kv_client_real.conf input/input.txt
# input 参数可缺省，缺省取启动目录下 input.txt（相对当前工作目录查找，仓库样例在 input/ 子目录，
# 用缺省值需先把 input.txt 放到启动目录平铺位置）
```

结束标记：`======Test Completed Successfully=======`；指标行含 `ops / traffic / tasks / notfound / mismatch`（notfound / mismatch 为观察指标，不算工具链失败）。

### 9. 常见问题

| 现象 | 处理 |
| --- | --- |
| 构建报 fmt/spdlog 拉取失败 | 检查网络；或 `-DDOWNLOAD_DEPENDENCE=OFF` + 预置源码目录 |
| 构建报 kv_client 库找不到 | kv client 未先构建，或构建目录不是 `kv_semantics/build-kv`（testTool 按此固定路径链接） |
| 建链失败（Init 报错） | 核对 kv_client_real.conf 与 kv_client_view_real.conf 的 asu_info 字段一致性、真实端点可达性 |
| `device` 类型打 WARN "falls back to host memory" | 当前二进制未编入设备 SDK（TESTTOOL_ENABLE_ASCEND），以主机内存模拟 device 语义，注册类型仍为 DEVICE |
| 进程起来后 stdout 无 testTool 日志 | 正常现象：testTool 日志只落盘 `log_demo/nds_*.log`，stdout 仅 kv client 库日志与 `[FAIL]` 错误 |
| 配置加载失败或视图文件找不到 | 确认在 testTool 目录下启动（conf / 视图等相对路径相对进程工作目录） |
