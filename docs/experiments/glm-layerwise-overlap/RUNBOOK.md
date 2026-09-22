# GLM layerwise 两组实验

直接复用服务器现有的 vLLM / vLLM-Ascend 镜像、GLM 模型和成功启动命令。两组使用同一份原生 UCM 及其可用构建库。

## 配置

沿用全层 GLM、TP8×DP2 和现有存储配置，两组均设置：

```yaml
use_layerwise: true
enable_early_prefetch: false
```

测试服务启用 `VLLM_SERVER_DEV_MODE=1`，供本机调用 HBM 缓存重置接口。下文以测试端口18180为例。不操作其他服务或缓存。

## 准备一次 L2 种子

准备8条独立的40000-token输入，保存 token IDs。每条前36000 tokens作为缓存前缀，后4000 tokens作为新增内容。

只发送这8条 **36000-token前缀请求**，等UCM写盘完成，正常停止本任务种子服务，保留其L2作为原始种子。不要预跑完整40000-token请求。

## 跑两组

两组都用无关输入做相同的内核预热；测量时C8、`temperature=0`、`max_tokens=1`，每条完整请求只发一次，记录首个生成token的时间。

| 步骤 | A：冷CPU L1 | B：热CPU L1 |
| --- | --- | --- |
| 启动 | 新测试容器、独立私有IPC，L2从原始种子复制 | 同A，从原始种子另复制一份L2 |
| 准备L1 | 不访问测试前缀 | 先发送8条36000-token前缀请求，等加载完成 |
| 清HBM | 服务空闲后调用下方命令 | 同A，只清HBM；之后不要重启服务 |
| 测量 | C8发送8条完整40000-token请求 | 发送完全相同的8条请求 |

**B不能继承A测量后的L2，也不能先预热完整40000-token请求**，否则新增的4000 tokens也可能被缓存。

服务空闲时，仅对本任务端口执行：

```bash
curl --fail --show-error --silent -X POST \
  'http://127.0.0.1:18180/reset_prefix_cache?reset_external=false&reset_running_requests=false'
```

## 只保留三项有效性检查

1. 正式请求的HBM前缀命中均为0；不能仅看重置接口返回200。
2. 两组外部命中与实际prefill tokens相同。命中按原生block边界计算，不强求恰好36000。
3. A确实从SSD读取测试KV，B的测试KV读取为0。用UCM实际backend读取记录，不用外部命中或native_get_fallback次数代替。

使用现有指标/日志即可；窗口统计排除预热。两组输出应正常，加载出错的轮次不纳入比较。

按A1/B1、B2/A2、A3/B3重复三对，每次恢复原始种子和对应L1状态。比较TTFT mean/p50与轮间波动。先拿到这张结果表，不把全套profiler、版本清单或逐层审计作为开跑前置条件。

B稳定更快，说明提前准备CPU KV存在端到端收益空间；若差距落入波动，只报告本次未测出可辨差异，不宣称已经证明完全掩盖。需要解释差距时，再针对性采样设备时间线。
