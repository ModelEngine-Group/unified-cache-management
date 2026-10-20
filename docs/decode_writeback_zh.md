# UCM Decode KV 回写与下一轮 Prefill 复用

## 范围与验证状态

基于 develop `3317a8280d7406e96cf6466c306ffa479f0bd518`，针对 vLLM / vLLM-Ascend
0.30 的非 LayerWise 路径实现。目标模型为 GLM5.2（`glm_moe_dsa`）、DeepSeek V4、
DeepSeek V4.1。覆盖 MRV1/MRV2 请求恢复及 MTP 的 committed-token 保存范围。
PP=PCP=DCP=1；DP 实例独立，不把 DP 编号写进共享对象 key。

本地验证是 CPU 计划/地址/内存 Store 测试，**不是 NPU 端到端验证**。
Dense GQA 异构 TP、DSpark、HLA/Mamba Decode 回写不在本次范围。
未宣称全量测试通过：本机原有 `test_ucm_connector_metrics.py`、`test_ucm_request_hasher.py`
联合收集因旧测试 stub 缺少 `vllm.v1.request` 而失败；这些测试未计入下面的独立 CPU 回归。

## 配置与角色

唯一新增功能开关：`save_decode_cache`，布尔值，默认 `false`。
可放入 UCM YAML，也可通过 `kv_connector_extra_config` 覆盖 YAML 中的同名值。

| 开关 | 角色 | Prefill 保存 | Decode 保存 |
|---|---|---|---|
| false | 任意 | 保持已有行为 | 不启用本功能 |
| true | kv_consumer | 保留实际计算的 Prefill/重新 Prefill 保存 | 保存可持久化完整单元 |
| true | kv_producer / kv_both | 保留 | 保存可持久化完整单元 |

角色不是物理节点类型：这里 Prefill/Decode 是每个请求当前阶段，而不是根据角色猜测。
consumer 故障接管时实际执行的 Prefill，以及抢占恢复后 prompt＋已生成历史的重新计算，
均允许保存有效完整单元；不因角色跳过。保存仍受源页面安全、阈值、成功确认和去重限制。
本功能不实现故障路由切换，也不要求对直接接收而未重新计算的 KV 强制重复 PUT。
`save_decode_cache=false` 不禁止读取 KV。Prefill 端无需打开该开关即可读 Decode 对象。
不要添加 AscendStore 专用的 `consumer_is_to_put`。

两个节点均使用现有 UCM 存储配置（例如 `examples/ucm_mooncake_config.yaml`），
确认 `use_layerwise: false`，相同权重/版本、KV dtype/量化、逻辑块和共享存储。
Decode 的配置示例：

```json
{
  "kv_connector": "UCMConnector",
  "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
  "kv_role": "kv_consumer",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "UCM_CONFIG_FILE": "/path/to/ucm_mooncake_config.yaml",
    "save_decode_cache": true
  }
}
```

Prefill 使用其原有角色与 PD 传输配置，`save_decode_cache` 可为 false。
MultiConnector 下把上述内容作为 Store 子连接器，不能代替原有 P→D 传输连接器。
使用 Ascend 原生 UCM 注册名时，本次适配补丁也会转发 `on_new_request`。
多组缓存继续要求 `kv_load_failure_policy=fail`，不宣称已支持多组失败重计算。

https://github.com/ModelEngine-Group/unified-cache-management/tree/develop
即使 Decode 已写成功。性能测试应恢复或调优该阈值，并明确记录配置。

## 实现要点

- 注册时持有真实 Request；根据 `request.num_computed_tokens + 本轮调度量` 跟踪计算范围，
  并以 `len(request.all_token_ids)` 限制 committed 范围。无新物理 block 也更新。
- 增量 hash 与 Prefill 的父 hash、extra keys、多模态游标一致。拒绝的 draft token 不进入 key。
  MRV1/MRV2 恢复时创建新请求代次，重新记录完整 re-prefill 范围，忽略旧任务回执。
- 短 prompt 或 `load_tokens_threshold` 跳过查询，不会丢失后续 Decode 的跟踪。
- 一个请求只允许一个在途逻辑任务。Worker 确认完成后回传；所有 TP 参与者成功，
  Scheduler 才更新 saved 集合。失败任务不推进，重试前补查已存在对象。
- GLM MLA 和 Indexer 等分量按各自 group 的 block ID 寻址；不使用 group 0 的地址代替所有组。
  复制型目标缓存使用 TP/MTP 无关的版本化 namespace；draft/private/transient 不参与。
  namespace 仍隔离不兼容 dtype、量化、压缩、缓存角色及实际记录格式。旧格式为冷缓存。
- FAWA 将 FA 前缀和最终 WA checkpoint 作为同一逻辑任务。FA 分摊到 TP，最终 WA 由一个 rank 保存；
  FA/WA 后端 BlockID 另外按 family 隔离，避免同一个全局 Mooncake pool 中同 logical key 的对象冲突。
  这一隔离同时用于查询、加载、保存和 prefetch，不仅是 Posix 子目录隔离。
  不把历史 WA checkpoint 当作可随时补写的数据。MTP 越过边界时，只取仍保留的边界完整页，
  不复制当前正在写的下一页；按包含 draft 写入的实际调度末端检查：若跨越余量已达到任意 WA 页跨度，
  或 block 表显示源页失效，则跳过。MTP 调度跨度较大时可能持续无法取得安全 checkpoint，需缩小跨度验证。
  因此部分步骤可能不保存，这是正确性保护，不是逐 token PUT，也不要求 MTP 恰好落在 hash 边界。
- GLM Direct 的异步 PUT 在下次覆盖 HBM 前 drain；FAWA 回写在本轮返回前等待 FA/WA，
  保护滚动窗口和压缩状态的源内存。该等待有 TPOT 成本，不能只测 TTFT。
- V4/V4.1 复用当前 FAWA 的逻辑/物理页换算，不硬编码 16384。
  V4.1 沿用已支持的整页传输格式，包含 Indexer scale 等必要字节；页记录大小不兼容时隔离，
  不声称支持任意 padding/打包格式转换。异构 TP 必须仍满足复制型、兼容记录格式约束。

## DEBUG 证据链

```bash
export UCM_LOG_LEVEL=debug
# vLLM 日志需要时再开启；不是本功能的开关
export VLLM_LOGGING_LEVEL=DEBUG
```

默认 INFO 仅输出初始化配置；DEBUG 才逐 key 输出。无需新增功能环境变量。

| 标识 | 意义 |
|---|---|
| UCM_DECODE_SAVE_CONFIG | 开关、连接器和角色 |
| UCM_DECODE_SAVE_SKIP | 阶段、完整单元/边界不足、在途任务等跳过原因 |
| UCM_DECODE_SAVE_PLAN | 计划的实际 key、token 范围、generation/job ID |
| UCM_DECODE_STORE_RESULT | FA/WA 具体 PUT 等待后的成功或失败 |
| UCM_DECODE_STORE_PRESENT | 重试补查确认对象已存在，不是新的 PUT 成功 |
| UCM_DECODE_SAVE_RESULT | 本 rank 的逻辑任务结果；非写入者标记 not_writer |
| UCM_PREFILL_LOOKUP | 下一轮查询的 key 和可复用前缀命中判断 |
| UCM_PREFILL_LOOKUP_SKIP | 既有 persist/load 阈值等导致未执行外部查询 |
| UCM_PREFILL_LOAD_RESULT | 实际加载完成结果 |

PLAN 不是写入成功。LOOKUP 的 FA+WA hit 表示可恢复前缀，不表示每个历史 key 都有独立 WA。
排查时要比较相同 key，而不是跨轮 request ID；请求 ID 本来就会不同。
key 日志会显著增量，最终性能测试关闭 DEBUG。
FAWA 日志同时给出 `backend_key_fa`、`backend_key_wa`（后端 BlockID 的十六进制）；
Mooncake 原生日志还可能追加 shard 后缀 `_0`。

## 两轮验证与验收

```bash
python benchmarks/verify_decode_writeback.py \
  --url http://ROUTER:8000/v1/chat/completions --model MODEL \
  --min-tokens 512 --max-tokens 2048 --report /tmp/decode_reuse.json
```

脚本保留第一轮实际 content 和 reasoning 字段，第二轮真实携带历史，区分首 delta 和首 content TTFT。
功能排查需要等待异步写完成时可加 `--settle-seconds 1`，该间隔不计入第二轮 TTFT；性能测试不加人为等待。
`--chat-template-kwargs` 可传模型所需 JSON。不能假设保存 reasoning 字段就一定被模板保留：
需检查所用模板的完整 token 前缀。不要使用 `ignore_eos=true` 强制长输出；其隐藏特殊 token
可能在文本回放中丢失。不要使用 ground truth 代替模型第一轮回答。

三模型分别执行 TP1↔TP2、TP2↔TP4 两方向和同 TP；组合无 MTP、D 开 MTP、P 无 MTP 而 D 开 MTP。
长输出需跨多个实际保存单元。并验证零接受/部分接受/全接受、抢占与取消、短输入、FA/WA 或 rank 故障。

成功标准：

1. Decode 有成功确认，key 对应的 token 范围超出第一轮原始 prompt，而不只是回存 prompt。
2. 下一轮 Prefill 查询并成功加载同一回答 key，外部命中超过原 prompt 边界。
   关闭本地 prefix caching，或只重启 Prefill、保留 pool，排除 HBM 命中；勿清空共享池。
3. 与相同读取端 TP、相同输入的无缓存参考比较输出正确性。异构 TP 不要求天然逐 token 相同，
   先区分浮点计算差异与 KV 错误；可比较同一读取端的 logits/输出参考。
4. 成对比较开关关闭/开启后的第二轮以后 TTFT，并报告 TPOT、写流量、实际回答命中量。
   客户端 TTFT/usage 或 key 存在单独都不足以证明数据正确、性能收益或真实加载。

CPU 回归：

本地记录（2026-10-20）：下列四个测试文件共 **169 项通过、22 项子用例通过**；
Python 编译检查、Black 24.4.2 / isort 6.0.1 格式检查及 `git diff --check` 通过。
测试包含真实 CPU 地址的字节 PUT/GET、共享后端 FA/WA 隔离，以及源内存覆盖后的回读。
这些结果不能替代 Linux/NPU 服务上的模型输出、MTP 和 TTFT/TPOT 验证。

```bash
python -m pytest --confcutdir=test/suites/Unit \
  test/suites/Unit/test_decode_writeback.py \
  test/suites/Unit/test_hma_block_geometry.py \
  test/suites/Unit/test_ucm_v41_compat.py \
  test/suites/Unit/test_kv_cache_layout.py
```
