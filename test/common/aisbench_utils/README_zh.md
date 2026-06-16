# AISBench 工具模块

本模块提供 AISBench Prefix Cache 性能测试的工具函数。

## 模块结构

```
aisbench_utils/
├── __init__.py           # 模块初始化和导出
├── data_class.py         # 配置和结果数据类
├── data_picker.py        # GSM8K数据选择器（支持自动重置）
├── generate_dataset.py   # 数据集生成（普通/prefix_cache，支持GSM8K/随机token）
├── api_config.py         # AISBench API配置生成
├── prefix_hit_rate.py    # Prefix Cache命中率计算
├── result_parser.py      # AISBench日志解析和结果保存
├── README.md             # 英文文档
└── README_zh.md          # 中文文档
```

## 功能特性

### 数据选择器 (`data_picker.py`)
- **不重复模式**: 从未使用的数据中随机选择，数据耗尽时自动重置
- **可重复模式**: 从所有数据中随机选择
- **纯内存追踪**: 无文件依赖，已选ID仅在内存中维护

```python
from common.aisbench_utils import DataPicker

# 不重复模式（数据耗尽时自动重置）
picker = DataPicker("GSM8K.jsonl", prefix_flag=1)

# 可重复模式
picker = DataPicker("GSM8K.jsonl", prefix_flag=0)

# 手动重置
picker.reset()
```

### 数据集生成 (`generate_dataset.py`)

支持两种数据来源模式：
- **GSM8K模式** (`use_gsm8k=True`): 使用GSM8K数据集作为内容源
- **随机token模式** (`use_gsm8k=False`): 直接生成随机token（无数据集依赖）

```python
from common.aisbench_utils import create_multi_prefix_dataset, create_dataset_from_random_tokens, parse_prefix_ratio

# 解析重复率（支持"50%"、"0.5"或整数50）
repeat_rate = parse_prefix_ratio("50%")  # -> 0.5

# 使用GSM8K创建Prefix Cache数据集
prefix_path, dataset_path = create_multi_prefix_dataset(
    tokenizer_path="/path/to/model",
    input_len=2048,
    number=160,
    save_path="/path/to/save",
    prefix_flag=1,          # 1=prefix_cache, 0=normal
    dp=2,
    repeat_rate=0.5,
    seed=1,
    prefix_num=1,
    use_gsm8k=True          # 使用GSM8K数据集
)

# 使用随机token创建Prefix Cache数据集（无需GSM8K）
prefix_path, dataset_path = create_multi_prefix_dataset(
    tokenizer_path="/path/to/model",
    input_len=2048,
    number=160,
    save_path="/path/to/save",
    prefix_flag=1,
    dp=2,
    repeat_rate=0.5,
    seed=1,
    prefix_num=1,
    use_gsm8k=False         # 直接生成随机token
)

# 创建简单的随机token数据集
samples = create_dataset_from_random_tokens(
    tokenizer_path="/path/to/model",
    input_len=512,
    number=10,
    seed=42,
    use_time_offset=True    # 添加时间偏移确保前缀唯一
)
```

#### prefix_num 参数效果

`prefix_num` 参数控制生成多少种不同的前缀模式：
- `prefix_num=1`: 所有请求共用同一个前缀（最大缓存命中率）
- `prefix_num=2`: 请求轮流使用2个不同前缀（50%命中率）
- `prefix_num=4`: 请求在4个前缀间轮转（25%命中率）

#### 时间偏移机制

使用随机token模式时，会自动添加基于时间的偏移量，确保不同测试运行的前缀不重复：

```python
# 不同时间的测试运行会生成不同的前缀
time_offset = int(time.time()) % 1000000
effective_seed = seed + time_offset
```

### 配置数据类 (`data_class.py`)
- **AisbenchConfig**: 包含所有测试参数的配置类
- **AisbenchResult**: 包含性能指标的结果类
- **自动类型转换**: `request_rate`和`repeat_rate`接受多种格式

```python
from common.aisbench_utils import AisbenchConfig, AisbenchResult

# 创建配置 - 类型自动转换
config = AisbenchConfig(
    input_len=2048,
    output_len=2048,
    data_num=160,
    concurrency=40,
    dataset_type="prefix_cache",
    repeat_rate=50,         # 接受: 50, 0.5, "50%", "0.5" -> 都转换为 "0.5"
    request_rate=10,        # 接受: 10, 0, "10", "0" -> 都转换为字符串
    prefix_test=True,
    dp=2,
    test_name="prefix_50pct_dp2"
)

# 转换为字典
config_dict = config.to_dict()
```

### API配置生成 (`api_config.py`)
- 动态生成AISBench API配置
- 支持stream和text测试类型
- 通过`url`参数支持域名访问（绕过IP验证）
- 支持DeepSeek V3.1思考模式

```python
from common.aisbench_utils import generate_api_config

# 使用IP地址（验证为IPv4/IPv6）
generate_api_config(
    model_path="/path/to/model",
    model_name="Qwen3-4B",
    concurrency=40,
    output_len=2048,
    request_rate="10",
    host_ip="127.0.0.1",
    host_port="8000",
    test_type="stream",
    work_path="/home/benchmark"
)

# 使用域名URL（绕过IP验证）
generate_api_config(
    model_path="/path/to/model",
    model_name="Qwen3-4B",
    concurrency=40,
    output_len=2048,
    request_rate="10",
    url="http://api.example.com:8080",  # 支持域名
    test_type="stream",
    work_path="/home/benchmark"
)
```

### 命中率计算 (`prefix_hit_rate.py`)
- 查询vLLM metrics API
- 计算HBM和external cache命中率

```python
from common.aisbench_utils import get_pod_metrics_info, cal_prefix_hit_info

pod_info = ["127.0.0.1:8000"]

# 测试前获取metrics
query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = get_pod_metrics_info(pod_info)

# 运行测试...

# 测试后获取metrics
query_tokens_new, query_tokens_external_new, hit_tokens_new, hit_tokens_external_new = get_pod_metrics_info(pod_info)

# 计算命中率
hit_info = cal_prefix_hit_info(
    query_tokens, query_tokens_external, hit_tokens, hit_tokens_external,
    query_tokens_new, query_tokens_external_new, hit_tokens_new, hit_tokens_external_new
)
```

### 结果解析 (`result_parser.py`)
- 解析AISBench日志文件
- 提取性能指标（TTFT、TPOT、吞吐量等）
- 保存结果到CSV

```python
from common.aisbench_utils import get_data, save_log, save_csv

# 解析日志
ans, log_dir = get_data("aisbench.log", "10", 1)

# 保存日志和CSV
save_log("aisbench.log", log_dir)
save_csv(ans, "aisbench_result.csv")
```

## 依赖

| 依赖 | 是否必需 | 说明 |
|------|----------|------|
| `transformers` | 可选 | 用于tokenizer（备用） |
| `tokenizers` | 可选 | Rust tokenizer（推荐，无torch依赖） |
| `pandas` | 是 | 用于CSV输出 |
| `torch` | 否 | 已移除（使用`random`替代） |
| `tqdm` | 否 | 可选（进度条） |

## 错误处理

不重复模式下数据耗尽时会自动重置，无需手动干预。

```python
# 手动重置（可选）
picker = DataPicker("GSM8K.jsonl")
picker.reset()
```

## 测试配置

完整测试示例参见 `test_aisbench_prefix_cache.py`。

### 参数传递方式

1. **环境变量**: `AISBENCH_TEST_CASE`（JSON格式）
2. **配置文件**: `config.yaml` 中的 `aisbench_prefix_cache.test_scenarios`
3. **默认配置**: 测试文件中内置的测试场景

```bash
# 使用环境变量传递多个测试配置
export AISBENCH_TEST_CASE='[
    {"input_len": 2048, "output_len": 2048, "data_num": 160, "concurrency": 40, "test_name": "2k_perf"},
    {"input_len": 4096, "output_len": 1024, "dataset_type": "prefix_cache", "repeat_rate": "50%", "test_name": "prefix_50pct"}
]'

pytest --feature=aisbench_prefix_cache
```

### 配置文件示例

```yaml
# config.yaml
aisbench_prefix_cache:
  dataset:
    base_path: "/mnt/host/d/Dataset/gsm"
    gsm8k_source: "/mnt/host/d/Dataset/gsm/GSM8K.jsonl"
    use_gsm8k: false  # false = 随机token（无数据集依赖）

  model:
    name: "qwen3:0.6b"
    path: "/mnt/host/d/Models/Qwen3-32B"
    tokenizer_path: "/mnt/host/d/Models/Qwen3-32B"

  server:
    url: "http://192.168.65.254:9090"  # 支持域名

  test:
    default_perf: "default_perf"
    test_type: "stream"
```

## AisbenchConfig 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_len` | int | 2048 | 输入token长度 |
| `output_len` | int | 2048 | 输出token长度 |
| `data_num` | int | 160 | 数据集条数 |
| `concurrency` | int | 40 | 系统最大并发数 |
| `request_rate` | int/float/str | 0 | 请求频率（自动转换: 0, 10, "10" -> "0", "10"） |
| `test_type` | str | "stream" | stream或text |
| `repeat` | int | 1 | 测试重复次数 |
| `dataset` | str | "" | 指定数据集路径 |
| `dataset_type` | str | "normal" | normal或prefix_cache |
| `prefix_num` | int | 1 | 前缀个数 |
| `repeat_rate` | int/float/str | 0.5 | 前缀重复率（自动转换: 50, 0.5, "50%" -> "0.5"） |
| `prefix_test` | bool | False | 是否预热前缀 |
| `dp` | int | 1 | DP域数量 |
| `seed` | int | 1 | 随机种子 |
| `length_mean` | int | None | 输入长度均值（高斯分布） |
| `length_std` | float | None | 输入长度标准差 |
| `length_min` | int | None | 输入长度最小值 |
| `length_max` | int | None | 输入长度最大值 |
| `test_accuracy` | bool | False | 是否测试精度 |
| `enable_think` | bool | False | DeepSeek V3.1思考模式 |
| `npu_num` | int | 1 | NPU卡数 |
| `test_name` | str | "Default" | 测试名称 |