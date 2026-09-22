# GLM 原生 UCM layerwise：两组核心对照

状态：2026-09-21，按用户要求精简。只保留一组缓存状态对照，尚未运行。不修改推理源码，不启动服务。

问题：原生 layerwise 开启时，如果 CPU KV 已经提前准备好，还能降低多少 TTFT？这里的掩盖指计算隐藏 KV 加载等待。UCM L1 是 CPU buffer，HBM prefix cache 是另一层缓存。

## 对照组

| 组 | UCM | layerwise | Early | 测量前 CPU L1 | SSD L2 | 测试前缀 HBM 命中 |
| --- | --- | --- | --- | --- | --- | --- |
| A：按需读盘 | baseline/native | 开 | 关 | 冷 | 前缀已完整保存 | 0 |
| B：CPU KV提前就绪 | 同一源码和库 | 开 | 关 | 热 | 与A相同逻辑内容 | 0 |

B保留原生 lookup、CPU→HBM和计算，只移除测量阶段按需从SSD准备CPU KV的工作。B的预热时间不计入测量，故它是提前准备CPU KV的参考，不是实际Early收益，也不是所有Early优化的严格数学上界。

本轮不加入 layerwise关闭、Early开启、其他源码版本、多档suffix或W32压力矩阵。

## 固定配置与负载

- UCM：`baseline/native @ 5e1427d6b89f69a2bff518dc59fbf5b6a52bc3c1`。工作树：`/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/src/ucm-native`；运行前核验A3库与Python实际加载路径。
- 镜像：`quay.io/ascend/vllm-ascend@sha256:d03346e4826118aaade06b4425e35dd7a7270f5a54322691e051e93d34897a0e`。
- vLLM：`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`；vLLM-Ascend：`f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db`。
- GLM-5.1-w4a8全78层，TP8×DP2、local DP2、EP开启；eager、max-model-len=131072、max-num-seqs=16、batch tokens=4096、HBM utilization=0.95；APC与chunked prefill保持原配置。
- 原生MLA shared L1 96GiB，L2配置1024GiB，AIO、io_direct=true、既有并发1、timeout=300000。dtype、共享范围、复制实现和失败策略两组相同，以有效配置为准。
- 只用一档负载：C8，输入40000 tokens，其中预埋前缀36000、未预埋suffix4000；greedy、输出1 token。使用实际token ID，配对输入完全一致。36000不是128的整数倍，接受原生完整block和最终block边界规则，实际external hit tokens必须两组一致，不强行声称命中恰好36000。
- 每个窗口8个互不共享前缀的请求，只访问一次；两组至少3个配对窗口，交替先后次序。以实际KV字节核验这8个前缀及新增dump工作空间能装入原96GiB L1，不能增加容量来构造B。

本轮用这一档C8窗口回答机制问题，不把它当作W32、长输入或稳定容量压力的结论。若差距落入轮间波动，只报告证据不足，不自动扩展实验矩阵。

## 缓存准备

1. 用同一模型和layerwise布局预埋窗口前缀到本任务L2，等待dump真实完成并核验种子。测量前不能清空这些L2文件，否则测试会变成计算miss而非SSD加载。
2. A使用本任务新L1实例，保留L2种子；采用不相干identity的输入完成内核/通信预热。正式窗口以实际SSD读取确认测试前缀不驻留。不能在同一窗口反复访问前缀，把后续热请求仍算作冷L1。
3. B在同一测试服务中仅用36000-token前缀请求预热，不能先发送完整40000-token测量请求；等待读写完成，再仅重置本服务HBM APC。保留UCM L1/L2；重置后不运行可能淘汰测试前缀的其他负载。A/B使用相同内核预热标准。
4. 已核验的vLLM `/reset_prefix_cache` 保持 `reset_external=false`，无在途请求时操作，并核对两个DP。接口HTTP200不证明重置成功，必须从正式请求记录确认HBM matched tokens为0。
5. 两组必须输入相同、external hit tokens和实际prefill tokens相同、输出正确；A有测试KV实际SSD读取，B的测试KV SSD读取为0。未满足就判定缓存状态构造无效，不能解释成重叠结果。

## 只保留必要观测

- 主结果：TTFT mean/p50、逐请求配对差值、各窗口差异。样本较少，不对p95/p99下结论。
- 有效性：HBM/external命中、实际prefill tokens、SSD读取字节、输出和错误。
- 机制：抽样关联request、DP/rank、step、layer/chunk、native task ID，查看SSD完成、H2D完成与计算依赖，确认加载是否真正推迟了首token。保留同一时段的计算与通信事件，避免将通信或CPU调度停顿归因于SSD。

源码依据：原生 `UCMLayerWiseConnector.start_load_kv`（ucm_connector.py:2267）提交首层，`wait_for_layer_load`（2307）等待当前层后提交下一层。它有约一层前瞻窗口，但不保证完全隐藏加载。`_record_layerwise_load_duration`（2158）是提交到等待返回的时间，包含已经被计算覆盖的部分。

主性能运行不启用重型profiler；从相同两组额外抽短窗口做设备时间线采样，不增加实验配置。不得为打点增加全设备同步、逐shard INFO日志，或改变任务粒度。CPU wait不等于NPU停顿，不累计各rank的重复等待，不用不同指标的p50相减推算阶段时间。若无法取得可靠设备关联，就只给端到端差距，不声称已证明因果。

## 判定

- B稳定更快，且时间线显示A的SSD依赖延迟了计算：存在提前准备CPU KV的收益空间，报告实际毫秒数及轮间不确定性。
- A/B差距小、配对区间足够窄且无持续SSD关键路径等待：在该负载下，SSD预取空间小。可以预先用TTFT的1%作为工程阈值，同时报告绝对时间；不显著不等于证明完全掩盖。
- B仍有H2D/同步等待：这是SSD预取没有覆盖的成本，不能宣称全部KV加载已隐藏。

## 执行边界

只测GLM，不停止其他服务、不清无关L1/L2/IPC、不全局drop_caches。设备不足不启动完整拓扑。核验库身份及SSD→CPU→HBM正确性后才采性能；既有SDMA Direct冷加载字节异常未定案，不能用后续一次通过关闭，也不能暗中给两组使用不同复制实现。失败日志保留。

两组均留存配置/源码/库manifest、输入token与种子身份、缓存有效性核验、逐请求结果和抽样时间线。当前只更新设计文档，没有启动实验。

具体准备、执行及APC命令见同目录RUNBOOK.md。每臂从原始前缀L2种子独立启动，不继承上一臂测量时写入的suffix缓存。
