# AISBench Utilities

This module provides utilities for AISBench Prefix Cache performance testing.

## Module Structure

```
aisbench_utils/
├── __init__.py           # Module initialization and exports
├── data_class.py         # Configuration and result data classes
├── data_picker.py        # GSM8K dataset picker with auto-reset
├── generate_dataset.py   # Dataset generation (normal/prefix_cache, GSM8K/random tokens)
├── api_config.py         # AISBench API configuration generation
├── prefix_hit_rate.py    # Prefix cache hit rate calculation
├── result_parser.py      # AISBench log parsing and result saving
├── README.md             # English documentation
└── README_zh.md          # Chinese documentation
```

## Features

### Data Picker (`data_picker.py`)
- **Non-repeat mode**: Pick from unused data randomly, auto-reset when data exhausted
- **Repeatable mode**: Pick from all data randomly
- **In-memory tracking**: No file dependency, picked IDs tracked in memory only

```python
from common.aisbench_utils import DataPicker

# Non-repeat mode (auto-reset when exhausted)
picker = DataPicker("GSM8K.jsonl", prefix_flag=1)

# Repeatable mode
picker = DataPicker("GSM8K.jsonl", prefix_flag=0)

# Manual reset
picker.reset()
```

### Dataset Generation (`generate_dataset.py`)

Supports two data source modes:
- **GSM8K mode** (`use_gsm8k=True`): Use GSM8K dataset as content source
- **Random token mode** (`use_gsm8k=False`): Generate random tokens directly (no dataset dependency)

```python
from common.aisbench_utils import create_multi_prefix_dataset, create_dataset_from_random_tokens, parse_prefix_ratio

# Parse repeat rate (supports "50%", "0.5", or integer 50)
repeat_rate = parse_prefix_ratio("50%")  # -> 0.5

# Create prefix cache dataset using GSM8K
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
    use_gsm8k=True          # Use GSM8K dataset
)

# Create prefix cache dataset using random tokens (no GSM8K dependency)
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
    use_gsm8k=False         # Generate random tokens directly
)

# Create simple random token dataset
samples = create_dataset_from_random_tokens(
    tokenizer_path="/path/to/model",
    input_len=512,
    number=10,
    seed=42,
    use_time_offset=True    # Add time-based offset for unique prefixes
)
```

#### Prefix Num Effect

The `prefix_num` parameter controls how many different prefix patterns are generated:
- `prefix_num=1`: All requests share the same prefix (maximum cache hit rate)
- `prefix_num=2`: Requests alternate between 2 different prefixes (50% hit rate)
- `prefix_num=4`: Requests rotate among 4 prefixes (25% hit rate)

#### Time-based Offset

When using random token mode, a time-based offset is automatically added to the seed to ensure different prefixes across test runs:

```python
# Same test run at different times will generate different prefixes
time_offset = int(time.time()) % 1000000
effective_seed = seed + time_offset
```

### Configuration Data Classes (`data_class.py`)
- **AisbenchConfig**: Test configuration with all parameters
- **AisbenchResult**: Test result with performance metrics
- **Auto type conversion**: `request_rate` and `repeat_rate` accept multiple formats

```python
from common.aisbench_utils import AisbenchConfig, AisbenchResult

# Create config - type conversion is automatic
config = AisbenchConfig(
    input_len=2048,
    output_len=2048,
    data_num=160,
    concurrency=40,
    dataset_type="prefix_cache",
    repeat_rate=50,         # Accepts: 50, 0.5, "50%", "0.5" -> all convert to "0.5"
    request_rate=10,        # Accepts: 10, 0, "10", "0" -> all convert to string
    prefix_test=True,
    dp=2,
    test_name="prefix_50pct_dp2"
)

# Convert to dict
config_dict = config.to_dict()
```

### API Configuration (`api_config.py`)
- Generate AISBench API configuration dynamically
- Support stream and text test types
- Support domain names via `url` parameter (bypasses IP validation)
- Support DeepSeek V3.1 thinking mode

```python
from common.aisbench_utils import generate_api_config

# Using IP address (validated as IPv4/IPv6)
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

# Using domain name URL (bypasses IP validation)
generate_api_config(
    model_path="/path/to/model",
    model_name="Qwen3-4B",
    concurrency=40,
    output_len=2048,
    request_rate="10",
    url="http://api.example.com:8080",  # Supports domain names
    test_type="stream",
    work_path="/home/benchmark"
)
```

### Prefix Hit Rate (`prefix_hit_rate.py`)
- Query vLLM metrics API
- Calculate HBM and external cache hit rates

```python
from common.aisbench_utils import get_pod_metrics_info, cal_prefix_hit_info

pod_info = ["127.0.0.1:8000"]

# Get metrics before test
query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = get_pod_metrics_info(pod_info)

# Run test...

# Get metrics after test
query_tokens_new, query_tokens_external_new, hit_tokens_new, hit_tokens_external_new = get_pod_metrics_info(pod_info)

# Calculate hit rate
hit_info = cal_prefix_hit_info(
    query_tokens, query_tokens_external, hit_tokens, hit_tokens_external,
    query_tokens_new, query_tokens_external_new, hit_tokens_new, hit_tokens_external_new
)
```

### Result Parser (`result_parser.py`)
- Parse AISBench log file
- Extract performance metrics (TTFT, TPOT, throughput, etc.)
- Save results to CSV

```python
from common.aisbench_utils import get_data, save_log, save_csv

# Parse log
ans, log_dir = get_data("aisbench.log", "10", 1)

# Save log and CSV
save_log("aisbench.log", log_dir)
save_csv(ans, "aisbench_result.csv")
```

## Dependencies

| Dependency | Required | Notes |
|------------|----------|-------|
| `transformers` | Optional | For tokenizer (fallback) |
| `tokenizers` | Optional | Rust tokenizer (preferred, no torch) |
| `pandas` | Yes | For CSV output |
| `torch` | No | Removed (uses `random` instead) |
| `tqdm` | No | Optional (progress bar) |

## Error Handling

Data exhaustion in non-repeat mode is handled automatically — the DataPicker auto-resets when all GSM8K items have been used, so no manual intervention is needed.

```python
# Manual reset (optional)
picker = DataPicker("GSM8K.jsonl")
picker.reset()
```

## Test Configuration

See `test_aisbench_prefix_cache.py` for complete test examples.

### Parameter Passing Methods

1. **Environment variable**: `AISBENCH_TEST_CASE` (JSON format)
2. **Config file**: `config.yaml` under `aisbench_prefix_cache.test_scenarios`
3. **Default**: Built-in test scenarios in test file

```bash
# Using environment variable
export AISBENCH_TEST_CASE='[
    {"input_len": 2048, "output_len": 2048, "data_num": 160, "concurrency": 40, "test_name": "2k_perf"},
    {"input_len": 4096, "output_len": 1024, "dataset_type": "prefix_cache", "repeat_rate": "50%", "test_name": "prefix_50pct"}
]'

pytest --feature=aisbench_prefix_cache
```

### Configuration File Example

```yaml
# config.yaml
aisbench_prefix_cache:
  dataset:
    base_path: "/mnt/host/d/Dataset/gsm"
    gsm8k_source: "/mnt/host/d/Dataset/gsm/GSM8K.jsonl"
    use_gsm8k: false  # false = random tokens (no dataset dependency)

  model:
    name: "qwen3:0.6b"
    path: "/mnt/host/d/Models/Qwen3-32B"
    tokenizer_path: "/mnt/host/d/Models/Qwen3-32B"

  server:
    url: "http://192.168.65.254:9090"  # Supports domain names

  test:
    default_perf: "default_perf"
    test_type: "stream"
```

## AisbenchConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_len` | int | 2048 | Input token length |
| `output_len` | int | 2048 | Output token length |
| `data_num` | int | 160 | Dataset sample count |
| `concurrency` | int | 40 | Maximum concurrency |
| `request_rate` | int/float/str | 0 | Request rate (auto-converts: 0, 10, "10" -> "0", "10") |
| `test_type` | str | "stream" | stream or text |
| `repeat` | int | 1 | Test repeat count |
| `dataset` | str | "" | Specified dataset path |
| `dataset_type` | str | "normal" | normal or prefix_cache |
| `prefix_num` | int | 1 | Number of prefix types |
| `repeat_rate` | int/float/str | 0.5 | Prefix repeat rate (auto-converts: 50, 0.5, "50%" -> "0.5") |
| `prefix_test` | bool | False | Whether to warmup prefix |
| `dp` | int | 1 | Number of DP domains |
| `seed` | int | 1 | Random seed |
| `length_mean` | int | None | Input length mean (Gaussian) |
| `length_std` | float | None | Input length std dev |
| `length_min` | int | None | Input length minimum |
| `length_max` | int | None | Input length maximum |
| `test_accuracy` | bool | False | Whether to test accuracy |
| `enable_think` | bool | False | DeepSeek V3.1 thinking mode |
| `npu_num` | int | 1 | NPU card count |
| `test_name` | str | "Default" | Test name |