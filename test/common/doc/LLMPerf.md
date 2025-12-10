# 📝 LLM 性能测试使用说明

## 🔧 功能概述  
本测试用于评估 LLM 推理服务在不同负载下的性能表现，涵盖延迟、吞吐量、请求成功率等关键指标。

## 📌 测试参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `mean_input_tokens` | 平均输入 token 数 | `[2000, 3000]` |
| `mean_output_tokens` | 平均输出 token 数 | `[200, 500]` |
| `max_num_completed_requests` | 最大完成请求数 | `[8, 4]` |
| `concurrent_requests` | 并发请求数 | `[8, 4]` |
| `additional_sampling_params` | 额外采样参数（如 temperature） | `["{}", "{}"]` |
| `hit_rate` | 缓存命中率 | `[0, 50]` |

> ✅ 支持多组参数组合运行，自动执行多轮推理并收集统计结果。

## 📊 输出结果

测试完成后，将输出以下性能指标的统计值（每轮结果均记录）：

- **延迟指标**：  
  - `inter_token_latency_s`（token 间延迟）  
  - `ttft_s`（首个 token 延迟）  
  - `end_to_end_latency_s`（端到端延迟）  
  - 各项包含：P50、P90、P99、平均值

- **吞吐量指标**：  
  - `total_throughput`（总吞吐量）  
  - `incremental_throughput`（增量吞吐量）

- **其他指标**：  
  - `num_completed_requests`（完成请求数）  
  - `elapsed_time`（总耗时）  
  - `incremental_time_delay`（增量时间延迟）

## ✅ 验证规则

- 所有数值必须 > 0
- 若出现 `None` 或 ≤ 0 的值，测试将标记为失败，并输出异常详情

## 📤 输出格式

返回一个字典，包含：
```python
{
    "_name": "llmperf",
    "_data": {  # 所有指标的列表
        "results_inter_token_latency_s_quantiles_p50": [...],
        "results_ttft_s_mean": [...],
        # ...
    }
}
```

## 🚀 使用方式 test/下运行

# 按文件运行
pytest test_uc_performance.py

# 按阶段运行
pytest --stage=0

# 按特性运行
pytest --feature=uc_performance_test

> ⚠️ 确保已安装依赖：`pytest` 等模块。