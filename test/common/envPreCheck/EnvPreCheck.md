 # 🔍 Environment PreCheck Automation Test Suite | Environment PreCheck Suite

> 💡 **Core Objective**: Prior to model deployment or training task execution, conduct comprehensive health checks on critical cluster capabilities to proactively identify potential issues in SSH configuration, device status, network connectivity, TLS encryption, model integrity, and storage performance.

---

## 📋 Table of Contents

- [🌟 Core Features](#-core-features)
- [🎯 Functional Overview](#-functional-overview)
- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [🧪 Test Case Details](#-test-case-details)
- [⚙️ Configuration](#️-configuration)

---

## 🌟 Core Features

| Feature | Description |
|------|------|
| 🎯 **Intelligent Platform Detection** | Automatically detect NPU (Ascend) / GPU environments and execute corresponding test logic |
| 🔧 **Modular Architecture** | Support flexible combination of test cases by stage, platform, and feature dimensions |
| 🛡️ **Fail-Fast Mechanism** | Immediately terminate upon critical check failures to avoid invalid resource consumption |
| 📊 **Comprehensive Coverage** | 6 major dimensions, 12+ detailed checks to ensure environmental consistency |

---

## 🎯 Functional Overview

This test suite automatically executes the following health checks prior to task initiation:

### 🔐 Infrastructure Layer

- **SSH Passwordless Login Verification**: Ensure bidirectional passwordless access between Master ↔ Worker is functional
- **Device Health Inspection**: NPU/GPU online status, driver version, and temperature monitoring

### 🌐 Network Communication Layer

- **Inter-Node Connectivity Testing**: HCCN (NPU) / NVLink & InfiniBand (GPU) link packet loss detection
- **TLS Encryption Status**: Verify Ascend device TLS switch configuration complies with security baselines

### 💾 Data Integrity Layer

- **Model Weight Vault**: File list integrity scan → MD5/SHA256 hash verification → weight format validity validation

### ⚡ Performance Baseline Layer

- **Storage Bandwidth Stress Test**: Compare measured Embedding/Fetch operation bandwidth against expected thresholds (< 85% triggers warning)

---

## 🚀 Quick Start

### 📁 Project Structure

```bash
tests/
├── common/envPreCheck/
│   ├── run_env_preCheck.py      # Core detection engine
│   └── utils/                   # Auxiliary utilities
├── suites/E2E/
│   └── test_environment_precheck.py  # Test entry point
└── config.yaml                  # PreCheck threshold configuration file
```

### 🎮 Execution Methods

```bash
# Enter test directory
cd tests/

# 1️⃣ Execute full precheck (stage 2)
pytest --stage=2

# 2️⃣ Execute by hardware platform
pytest --platform=npu    # Ascend NPU environment
pytest --platform=gpu    # NVIDIA GPU environment

# 3️⃣ Execute by individual feature
pytest --feature=test_ssh_login
pytest --feature=test_check_bandwidth

# 4️⃣ Run specific file directly
pytest suites/E2E/test_environment_precheck.py -v
```

---

## 🧪 Test Case Details

### 🔐 SSH Connectivity Check

```python
test_ssh_login()
```

- **Verification Content**: Master → Worker bidirectional passwordless login
- **Failure Strategy**: ❌ **Immediate termination** of all subsequent tests (blocking issue)

### 🖥️ Device Status Check

```python
# NPU environment
test_hccn_check_device_status()

# GPU environment  
test_nvidia_check_device_status()
```

- **Check Items**: Device online status, driver loading, memory health, temperature thresholds

### 🌐 Inter-Node Network Quality

```python
test_check_hccn_ping()      # NPU: HCCN links
test_check_nvidia_ping()    # GPU: NCCL networks
```

**Generate full-link topology report**:

```
✅ local_card_0  →  local_card_1        [0.02ms, 0% loss]
✅ local_card_0  →  remote_ip:192.168.1.10  [0.15ms, 0% loss]
⚠️  remote_card_1 →  local_ip:192.168.1.5   [2.34ms, 3% loss]  ← Abnormal link
```

### 🔒 TLS Security Configuration

```python
test_check_tls()
```

- **Check Target**: `tls_switch` status of each card
- **Pass Criteria**: All device TLS switches are consistent and comply with security policies (usually 0 or 1)

### 📦 Model Weight Integrity

```python
test_check_model_weights()
```

**Three-layer protection system**:

1. **File Tree Scanning**: Confirm existence of all `.bin`, `.safetensors`, `.json` files
2. **Hash Verification**: Compare against pre-computed checksums to prevent transmission corruption
3. **Format Validity**: Rapid loading validation using `torch.load` / `safetensors`

### ⚡ Storage Bandwidth Benchmark Test

```python
test_check_bandwidth()
```

- **Test Scenario**: Large-scale Embedding reads / Checkpoint Fetch writes
- **Decision Logic**:
  ```python
  if actual_bandwidth < expected_threshold * 0.85:
      raise PerformanceWarning("Insufficient storage bandwidth may impact training efficiency")
  ```

---

## ⚙️ Configuration (`config.yaml`)

| Configuration Item | Type | Description | Example Value |
|--------|------|------|--------|
| `master_ip` | string | SSH login IP of the cluster master node | `192.168.1.10` |
| `worker_ip` | list | IP addresses of cluster worker nodes | `["192.168.1.11", "192.168.1.12"]` |
| `ascend_rt_visible_devices` | string | Visible device IDs for Ascend/NPU | `"0,1,2,3,4,5,6,7"` |
| `node_num` | int | Total number of cluster nodes | `2` |
| `model_path` | string | Root directory of model weights | `/data/models/llama-7b` |
| `hf_model_name` | string | HuggingFace model identifier | `meta-llama/Llama-2-7b` |
| `middle_page` | string | Intermediate page/organization name corresponding to the model | `model_storage` |
| `expected_embed_bandwidth` | float | Expected embedding bandwidth (GB/s) | `12.0` |
| `expected_fetch_bandwidth` | float | Expected fetch bandwidth (GB/s) | `8.0` |
| `kvCache_block_number` | int | Number of KV Cache pre-allocated blocks | `4096` |
| `storage_backends` | list | Storage backend mount paths | `["/data", "/mnt/nfs"]` |

---

## 🎨 Output Example

```diff
🚀 Launching Environment PreCheck Suite (Platform: NPU)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ [PASS] SSH Passwordless Login (2/2 nodes)
✅ [PASS] NPU Device Status (8/8 cards online)
✅ [PASS] HCCN Link Connectivity (56/56 links)
⚠️  [WARN] TLS Configuration (card_3: tls_switch=1, expected=0)
✅ [PASS] Model Weight Integrity (hash verified)
❌ [FAIL] Storage Bandwidth Check (6.5 GB/s < 12.0 GB/s * 0.85)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 PreCheck failed, please fix high-priority issues before launching training tasks
```