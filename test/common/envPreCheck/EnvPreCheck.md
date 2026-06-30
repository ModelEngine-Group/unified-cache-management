---

# 🔍 Environment PreCheck Suite

> 💡 **Core Objective**: Before model deployment or training tasks begin, perform comprehensive health checks on key cluster capabilities to proactively identify potential issues in SSH configuration, device status, network connectivity, TLS encryption, model integrity, and storage performance.

---

## 📋 Table of Contents

* [🌟 Key Features](#-key-features)
* [🎯 Overview](#-overview)
* [⚙️ Configuration](#️-configuration)
* [🚀 Quick Start](#-quick-start)
* [🧪 Test Case Details](#-test-case-details)

---

## 🌟 Key Features

| Feature                               | Description                                                                                 |
| ------------------------------------- | ------------------------------------------------------------------------------------------- |
| 🎯 **Intelligent Platform Detection** | Automatically detects NPU (Ascend) / GPU environments and executes corresponding test logic |
| 🔧 **Modular Architecture**           | Supports flexible combination of test cases by stage, platform, and feature                 |
| 🛡️ **Failure Fast Mechanism**        | Critical check failures immediately terminate execution to avoid wasted resources           |
| 📊 **Comprehensive Coverage**         | Covers 6 major dimensions and 12+ checks to ensure environment consistency                  |

---

## 🎯 Overview

This test suite automatically performs the following health checks before task execution:

### 🔐 Infrastructure Layer

* **SSH Passwordless Login Verification**: Ensures bidirectional password-free access between Master ↔ Worker
* **Device Health Inspection**: NPU/GPU status, driver versions, temperature monitoring

### 🌐 Network Communication Layer

* **Node Connectivity Testing**: Packet loss detection for HCCN (NPU) / NVLink & InfiniBand (GPU)
* **TLS Encryption Status**: Verifies whether TLS settings between Ascend devices meet security baselines

### 💾 Data Integrity Layer

* **Model Weights Vault**: File list integrity scan → MD5/SHA256 hash verification → weight format validation

### ⚡ Performance Baseline Layer

* **Storage Bandwidth Benchmarking**: Compare measured Dump/Load bandwidth against expected thresholds (< 85% triggers warning)

---

## ⚙️ Configuration (`config.yaml`)

| Parameter                   | Type   | Description                                    | Example                            |
| --------------------------- | ------ | ---------------------------------------------- | ---------------------------------- |
| `master_ip`                 | string | Master node SSH IP                             | `192.168.1.10`                     |
| `worker_ip`                 | list   | Worker node IP list (empty = single-node test) | `["192.168.1.11", "192.168.1.12"]` |
| `ascend_rt_visible_devices` | string | Visible NPU device indices                     | `"0,1,2,3,4,5,6,7"`                |
| `node_num`                  | int    | Total number of nodes (master + workers)       | `2`                                |
| `model_path`                | string | Root directory of model weights                | `/data/models/llama-7b`            |
| `hf_model_name`             | string | HuggingFace model identifier                   | `meta-llama/Llama-2-7b`            |
| `middle_page`               | string | Model intermediate page / organization name    | `model_storage`                    |
| `expected_embed_bandwidth`  | float  | Expected Dump bandwidth (GB/s)                 | `12.0`                             |
| `expected_fetch_bandwidth`  | float  | Expected Load bandwidth (GB/s)                 | `8.0`                              |
| `storage_backends`          | list   | Mounted storage paths in current environment   | `["/mnt/nfs"]`                     |

---

## 🚀 Quick Start

### 📁 Project Structure

```bash
tests/
├── common/envPreCheck/
│   ├── run_env_preCheck.py      # Core check engine
├── suites/E2E/
│   └── test_environment_precheck.py  # Test entry point
└── config.yaml                  # Precheck threshold configuration
```

### 🎮 How to Run

```bash
# Navigate to test directory
cd tests/

# 1️⃣ Run by hardware platform
pytest --platform=npu    # Ascend NPU environment
pytest --platform=gpu    # NVIDIA GPU environment

# 2️⃣ Run by specific feature
pytest --feature=test_ssh_login
pytest --feature=test_check_bandwidth

# 3️⃣ Run full precheck (stage 2)
pytest --stage=2

# 4️⃣ Run a specific test file directly
pytest suites/E2E/test_environment_precheck.py -v
```

---

## 🧪 Test Case Details

### 🔐 SSH Connectivity Check

```python
test_ssh_login()
```

* **Validation**: Bidirectional passwordless login between Master → Worker
* **Failure Strategy**: ❌ **Immediately aborts** all subsequent tests (blocking issue)

### 🖥️ Device Status Check

```python
# NPU environment
test_hccn_check_device_status()

# GPU environment  
test_nvidia_check_device_status()
```

* **Checks**: Device availability, driver status, memory health, temperature thresholds

### 🌐 Inter-Node Network Quality

```python
test_check_hccn_ping()      # NPU: HCCN link
test_check_nvidia_ping()    # GPU: NCCL network
```

**Generates full topology report**:

```
✅ local_card_0  →  local_card_1        [0.02ms, 0% loss]
✅ local_card_0  →  remote_ip:192.168.1.10  [0.15ms, 0% loss]
⚠️  remote_card_1 →  local_ip:192.168.1.5   [2.34ms, 3% loss]  ← abnormal link
```

### 🔒 TLS Security Configuration

```python
test_check_tls()
```

* **Check Target**: `tls_switch` status on each device
* **Pass Criteria**: All devices have consistent TLS settings and comply with security policy (typically 0 or 1)

### 📦 Model Weight Integrity

```python
test_check_model_weights()
```

**Three-layer protection system**:

1. **File Tree Scan**: Ensure all `.bin`, `.safetensors`, `.json` files exist
2. **Hash Verification**: Compare precomputed checksums to prevent corruption
3. **Format Validation**: Quick load validation using `torch.load` / `safetensors`

### ⚡ Storage Bandwidth Benchmark

```python
test_check_bandwidth()
```

* **Test Scenario**: Large-scale embedding read / checkpoint fetch write
* **Evaluation Logic**:

  ```python
  if actual_bandwidth < expected_threshold * 0.85:
      raise PerformanceWarning("Insufficient storage bandwidth, may impact training efficiency")
  ```

---
