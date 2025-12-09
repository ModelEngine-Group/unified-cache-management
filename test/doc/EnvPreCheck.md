# 环境预检（Environment PreCheck）自动化测试套件

基于 pytest 的环境预检自动化测试，用于在大规模训练或推理任务执行前，对集群环境的关键能力进行全面验证。该预检流程覆盖 SSH 连通性、设备健康状态、节点间通信、TLS 配置、模型权重完整性以及存储带宽等关键组件，确保集群处于可用且一致的运行状态。

# 测试框架具备以下特性：

1、针对 NPU（Ascend）/ GPU 自动识别执行相应测试

2、所有测试项均动态输出，便于快速识别检测

3、全面覆盖环境依赖的核心组件


# 功能概述

本测试套件在模型部署或训练前自动执行以下预检内容：

✔ SSH 登录能力检查：确保 Master/Worker 的免密登录配置正确。

✔ 设备状态检查（NPU / GPU）：检测各节点的设备是否在线、驱动是否正常。

✔ 节点间 Ping 连通性检查：验证 HCCN（NPU）或 NVIDIA（GPU）网络链路是否可达、是否存在丢包或故障链路。

✔ TLS 配置检查：检查 Ascend 集群中设备间 TLS 加密链路开关状态。

✔ 模型权重完整性检查：包括权重文件列表扫描、哈希值校验和权重有效性验证

✔ 存储点带宽检查：检查 embedding / fetch 操作带宽，并与配置文件中的限制进行比较。


# 🚀 ###########如何运行测试（test/下执行）#############
# 目录结构说明
├── tests/
├── common/
│   └── envPreCheck/
│       ├── run_env_preCheck.py # 所有检查逻辑
│       └── ...
├── suites
│   └── E2E/
│       ├── test_environment_precheck.py # 执行逻辑
│       └── ...
├── config.yaml                  # 预检的阈值配置

# 运行完整的阶段预检
pytest --stage=2

# 按平台运行预检（公共检测项默认NPU平台）
# NPU环境
pytest --platform=npu
# GPU环境
pytest --platform=gpu

# 按特性类别运行
pytest --feature=test_ssh_login

# 按文件运行
pytest suites/E2E/test_environment_precheck.py


# 🛠️ #########以下为各测试的示例说明##########
# ● SSH 登录检查：
test_ssh_login()
校验 master/worker SSH 免密登录

任何失败将立即中止后续测试

# ● 设备状态检查（NPU/GPU）
test_hccn_check_device_status()
test_nvidia_check_device_status()

检查设备是否在线、是否可用

# ● HCCN / NVIDIA Ping 连通性检查
test_check_hccn_ping()
test_check_nvidia_ping()

生成所有链路状态信息，例如：

local_card_0 → local_card_1

local_card_0 → remote_IP

remote_card_1 → local_IP

# ● TLS 开关检查
test_check_tls()

校验每张卡的tls switch是否为0

# ● 模型权重完整性检查
test_check_model_weights()

权重文件列表扫描、哈希值校验和权重有效性验证

# ● 存储带宽检测
test_check_bandwidth()

检查存储实际待遇是否达到预期带宽：实际 bandwidth < 阈值 * 0.85，否则判定为异常


# ⚙️##########配置项说明（config.yaml）##########
master_ip：集群主节点（Master）的 SSH 登录 IP，用于执行预检中的主节点检查。

worker_ip：集群工作节点（Worker）的 IP 地址，用于检测 Worker 的设备状态、网络连通性等。

ascend_rt_visible_devices：设置 Ascend/NPU 的可见设备序号，用于设备状态和带宽测试。

node_num：集群节点数量，用于预检逻辑判断是否需要跨节点检查。

model_path：模型权重所在目录，用于权重文件检查和哈希校验。

hf_model_name：Hugging Face 模型名称，用于识别模型类型与路径结构。

middle_page：模型对应的中间页面/组织名称，用于存储结构或路径判断

expected_embed_bandwidth：预期 embedding 带宽（GB/s），用于对比实际带宽是否满足阈值。

expected_fetch_bandwidth：预期 fetch 带宽（GB/s），与实际带宽比较判断读写性能是否正常。

kvCache_block_number：KV Cache 预分配 block 数量，用于环境容量检查或推理配置校验。

storage_backends：存储后端挂载路径，用于带宽测试和路径可用性检查。
将这些内容复制打出来