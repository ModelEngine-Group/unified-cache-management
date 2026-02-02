我来帮你重新润色这份技术文档，让它更加专业且富有视觉层次：

---

# 🔍 环境预检自动化测试套件 | Environment PreCheck Suite


> 💡 **核心目标**：在模型部署或训练任务执行前，对集群关键能力进行全面健康检查，提前发现 SSH 配置、设备状态、网络连通性、TLS 加密、模型完整性及存储性能等潜在问题。

---

## 📋 目录

- [🌟 核心特性](#-核心特性)
- [🎯 功能概述](#-功能概述)
- [🚀 快速开始](#-快速开始)
- [📁 项目结构](#-项目结构)
- [🧪 测试用例详解](#-测试用例详解)
- [⚙️ 配置说明](#️-配置说明)

---

## 🌟 核心特性

| 特性 | 描述 |
|------|------|
| 🎯 **智能平台识别** | 自动检测 NPU (Ascend) / GPU 环境并执行相应测试逻辑 |
| 🔧 **模块化架构** | 支持按阶段、平台、特性维度灵活组合测试用例 |
| 🛡️ **故障熔断机制** | 关键检查项失败立即中止，避免无效资源占用 |
| 📊 **全覆盖检测** | 6 大维度、12+ 细项检查，确保环境一致性 |

---

## 🎯 功能概述

本测试套件在任务启动前自动执行以下健康检查：

### 🔐 基础设施层

- **SSH 免密登录验证**：确保 Master ↔ Worker 双向无密码访问畅通
- **设备健康巡检**：NPU/GPU 在线状态、驱动版本、温度监控

### 🌐 网络通信层

- **节点连通性测试**：HCCN (NPU) / NVLink & InfiniBand (GPU) 链路丢包检测
- **TLS 加密状态**：校验 Ascend 设备间 TLS 开关配置是否符合安全基线

### 💾 数据完整性层

- **模型权重保险箱**：文件列表完整性扫描 → MD5/SHA256 哈希校验 → 权重格式有效性验证

### ⚡ 性能基线层

- **存储带宽压测**：Embedding/Fetch 操作实测带宽对比预期阈值（< 85% 触发告警）

---

## 🚀 快速开始

### 📁 项目结构

```bash
tests/
├── common/envPreCheck/
│   ├── run_env_preCheck.py      # 核心检测引擎
│   └── utils/                   # 辅助工具集
├── suites/E2E/
│   └── test_environment_precheck.py  # 测试入口
└── config.yaml                  # 预检阈值配置文件
```

### 🎮 运行方式

```bash
# 进入测试目录
cd tests/

# 1️⃣ 执行完整预检（阶段 2）
pytest --stage=2

# 2️⃣ 按硬件平台执行
pytest --platform=npu    # Ascend NPU 环境
pytest --platform=gpu    # NVIDIA GPU 环境

# 3️⃣ 按特性单独执行
pytest --feature=test_ssh_login
pytest --feature=test_check_bandwidth

# 4️⃣ 直接运行特定文件
pytest suites/E2E/test_environment_precheck.py -v
```

---

## 🧪 测试用例详解

### 🔐 SSH 连通性检查

```python
test_ssh_login()
```

- **验证内容**：Master → Worker 双向免密登录
- **失败策略**：❌ **立即中止**后续所有测试（阻塞性问题）

### 🖥️ 设备状态检查

```python
# NPU 环境
test_hccn_check_device_status()

# GPU 环境  
test_nvidia_check_device_status()
```

- **检测项**：设备在线状态、驱动加载、显存健康、温度阈值

### 🌐 节点间网络质量

```python
test_check_hccn_ping()      # NPU: HCCN 链路
test_check_nvidia_ping()    # GPU: NCCL 网络
```

**生成全链路拓扑报告**：

```
✅ local_card_0  →  local_card_1        [0.02ms, 0% loss]
✅ local_card_0  →  remote_ip:192.168.1.10  [0.15ms, 0% loss]
⚠️  remote_card_1 →  local_ip:192.168.1.5   [2.34ms, 3% loss]  ← 异常链路
```

### 🔒 TLS 安全配置

```python
test_check_tls()
```

- **检查目标**：每张卡的 `tls_switch` 状态
- **通过标准**：所有设备 TLS 开关一致且符合安全策略（通常为 0 或 1）

### 📦 模型权重完整性

```python
test_check_model_weights()
```

**三层防护体系**：

1. **文件树扫描**：确认所有 `.bin`, `.safetensors`, `.json` 存在
2. **哈希值校验**：比对预计算的 checksum，防止传输损坏
3. **格式有效性**：使用 `torch.load` / `safetensors` 快速加载验证

### ⚡ 存储带宽基准测试

```python
test_check_bandwidth()
```

- **测试场景**：大规模 Embedding 读取 / Checkpoint Fetch 写入
- **判定逻辑**：
  ```python
  if actual_bandwidth < expected_threshold * 0.85:
      raise PerformanceWarning("存储带宽不足，可能影响训练效率")
  ```

---

## ⚙️ 配置说明（`config.yaml`）

| 配置项 | 类型 | 说明 | 示例值 |
|--------|------|------|--------|
| `master_ip` | string | Master 节点 SSH IP | `192.168.1.10` |
| `worker_ip` | list | Worker 节点 IP 列表 | `["192.168.1.11", "192.168.1.12"]` |
| `ascend_rt_visible_devices` | string | NPU 可见设备序号 | `"0,1,2,3,4,5,6,7"` |
| `node_num` | int | 集群总节点数 | `2` |
| `model_path` | string | 模型权重根目录 | `/data/models/llama-7b` |
| `hf_model_name` | string | HuggingFace 模型标识 | `meta-llama/Llama-2-7b` |
| `middle_page` | string | 中间页/组织名称 | `model_storage` |
| `expected_embed_bandwidth` | float | 预期 Embedding 带宽 (GB/s) | `12.0` |
| `expected_fetch_bandwidth` | float | 预期 Fetch 带宽 (GB/s) | `8.0` |
| `kvCache_block_number` | int | KV Cache 预分配块数 | `4096` |
| `storage_backends` | list | 存储后端挂载路径 | `["/data", "/mnt/nfs"]` |

---

## 🎨 输出示例

```diff
🚀 启动环境预检套件 (Platform: NPU)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ [PASS] SSH 免密登录 (2/2 nodes)
✅ [PASS] NPU 设备状态 (8/8 cards online)
✅ [PASS] HCCN 链路连通性 (56/56 links)
⚠️  [WARN] TLS 配置 (card_3: tls_switch=1, expected=0)
✅ [PASS] 模型权重完整性 (hash verified)
❌ [FAIL] 存储带宽检测 (6.5 GB/s < 12.0 GB/s * 0.85)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 预检未通过，请修复高优问题后再启动训练任务
```

---