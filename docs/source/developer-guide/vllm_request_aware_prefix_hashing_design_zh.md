# vLLM 集成的请求语义感知 Prefix Hash 设计

## 文档状态

- 状态：已实施（Implemented）
- 范围：UCM vLLM 集成
- 核心目标：使外部 KV Prefix Cache Key 感知完整请求语义，并且无需依赖
  `PYTHONHASHSEED` 即可跨进程复现

## 1. 背景

UCM 当前独立于 vLLM 计算外部 KV Cache Block ID。现有算法将父 block hash
与当前 block 的 token ID 组成链式哈希：

```text
block_hash[i] = H(block_hash[i - 1], block_token_ids[i])
```

对于 KV 状态完全由 token ID 决定的纯文本请求，这种方式是充分的。但如果相同
token 序列可能产生不同 KV 状态，该算法就不能正确区分请求。

多模态模型通常会把图片展开成重复的媒体占位 token。两张不同图片展开后的
placeholder token prefix 可能完全相同。因此，如果 UCM 只对 token ID 做哈希，
之前写入的小图 KV prefix 可能在后续大图请求中被错误命中。剩余的大图 block
随后又使用新图片 embedding 计算，最终导致一次推理同时包含新旧两张图片的 KV
特征。

vLLM 通过向 block hash 中加入请求语义数据避免该问题。当前 extra keys 包括：

- 多模态特征 identifier，以及它相对于当前 block 的位置；
- LoRA 标识；
- 请求 cache salt；
- prompt embedding 内容哈希。

UCM 必须纳入相同的请求语义，同时保留自身确定性的、面向外部 Store 的 key 链。

## 2. 目标

本设计需要实现以下目标：

1. 不同多模态输入不能仅仅因为 placeholder token 相同而共享 KV block。
2. 相同请求和部署配置在不同进程及服务重启后必须生成相同的 UCM Block ID。
3. UCM 不得依赖 vLLM 进程内的 `NONE_HASH`，也不能要求用户设置
   `PYTHONHASHSEED`。
4. UCM 必须复用 vLLM 对 request hash extra keys 的定义，而不是只复制一份
   多模态字段逻辑。
5. Direct、LayerWise、CP、Mock、Lite、HLA/Mamba 和 FAWA 使用同一套请求语义
   感知哈希实现。
6. 现有 block size、缓存查询、load、dump、rank 隔离和物理布局行为保持不变。
7. 对不支持的请求语义必须 fail closed：可以 miss，但不能为 KV 状态依赖额外
   数据的请求加载或写入 token-only key。

## 3. 非目标

以下内容不属于本次修改范围：

- 直接复用 `request.block_hashes` 作为 UCM Store key；
- 修改 UCM Store `BlockId` 的宽度；
- 替换 `RequestHasher` 当前使用的 MD5/pickle 编码；
- 修改 `RequestHasher` 当前构造的模型、配置和 rank namespace；
- 将 HLA 优化为所有 Full Attention group 共享一条细粒度语义 hash 链；
- 为 CacheBlend 的 chunk-local key 增加请求语义感知哈希；
- 实现尚未完成的 PD Connector；
- 修改 Inference Monitor 行为；
- 从 `ucm.utils` 迁移通用 `Config` 类。

FAWA 会参与公共 API 重构，但本次不增加 FAWA 专属的多模态行为。目前支持的
DeepSeek V4 配置尚未提供多模态请求路径。

## 4. 设计原则

### 4.1 请求语义与进程状态分离

UCM 继续计算自己的确定性 hash 链，不复用 vLLM 最终 block hash。原因是：未设置
`PYTHONHASHSEED` 时，vLLM block hash 以进程级随机值为根，不能作为持久化 Store
key。

### 4.2 由 vLLM 定义影响 KV 的请求语义

UCM 针对每个 UCM hash block 调用 vLLM 的
`generate_block_hash_extra_keys()`。这样多模态、LoRA、cache salt 和 prompt
embedding 的语义可以持续与 vLLM 对齐。

### 4.3 由 UCM 维护物理缓存身份

`RequestHasher` 继续纳入现有的 UCM 模型、dtype、并行配置、rank、推测解码和
稀疏注意力元数据。Worker 侧的 rank key 派生逻辑保持不变。

### 4.4 一套实现支持多种 block size

请求哈希循环归属于 `RequestHasher`。每个 Connector 使用自身需要的 block size
和初始 parent hash，绑定一个可复用闭包。

## 5. 组件位置

将 `RequestHasher` 从 `ucm_connector.py` 移至：

```text
ucm/integration/vllm/request_hasher.py
```

该类会使用 vLLM `Request` 对象以及 vLLM block extra-key helper，因此属于 vLLM
专属能力，应保留在 vLLM integration 下，而不是放入框架无关的
`ucm.integration.utils` 包。

通用 `Config` 类继续保留在：

```text
ucm/utils.py
```

如果将 `Config` 移入 integration 包，会导致 core/sparse 模块反向依赖框架适配
层，破坏当前分层关系。

## 6. RequestHasher API

`RequestHasher` 保留现有通用对象哈希接口，并新增 request block hasher 工厂。

```python
class RequestHasher:
    def __init__(self, vllm_config, rank_id):
        # 保留现有模型、配置和rank namespace构造逻辑。
        ...
        self.seed = self("UCM_HASH_SEED")

    def __call__(self, input_data) -> bytes:
        # 保留现有meta + pickle + MD5实现。
        ...

    def make_request_block_hasher(
        self,
        block_size: int,
        initial_hash: bytes | None = None,
    ) -> Callable[["Request"], list[bytes]]:
        ...
```

返回的闭包按以下公式计算：

```text
parent[0] = initial_hash or RequestHasher.seed

block_hash[i] = RequestHasher(
    parent[i],
    block_token_ids[i],
    block_extra_keys[i],
)

parent[i + 1] = block_hash[i]
```

参考实现：

```python
def make_request_block_hasher(self, block_size, initial_hash=None):
    root = initial_hash if initial_hash is not None else self.seed

    def hash_request(request):
        token_ids = request.all_token_ids
        parent = root
        curr_mm_idx = 0
        hashes = []

        for start in range(0, len(token_ids), block_size):
            end = start + block_size
            if end > len(token_ids):
                break

            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request,
                start,
                end,
                curr_mm_idx,
            )
            parent = self(
                (
                    parent,
                    tuple(token_ids[start:end]),
                    extra_keys,
                )
            )
            hashes.append(parent)

        return hashes

    return hash_request
```

`parent` 和 `curr_mm_idx` 必须在 `hash_request()` 内部初始化。如果将其中任何
变量放到外层工厂作用域，可复用闭包就会在不同请求之间泄漏状态。

与 UCM 当前行为一致，不对不完整的尾部 block 计算 hash。

## 7. Extra-key 兼容层

`request_hasher.py` 负责封装一个较薄的 vLLM
`generate_block_hash_extra_keys()` 兼容层。

兼容层必须遵守以下规则：

1. vLLM helper 可用时直接委托给 vLLM。
2. 保留并返回多模态 cursor，保证每个请求或 group 内的扫描复杂度为线性。
3. helper 不可用时，只有当请求不存在多模态、LoRA、cache salt 和 prompt
   embedding 时，才允许返回 `None`。
4. 对包含额外语义但 helper 不可用的请求，抛出专用 request-hash 异常。
5. identifier 校验委托给 vLLM；当 helper 扫描到 identifier 缺失的多模态 feature
   时，将其异常转换为 request-hash 错误，禁止回退到 token-only。

Connector 捕获该专用异常后，返回 external miss，并且不创建 dump metadata。
请求正常重新计算，但不会写入无效的外部 KV 条目。

## 8. Hash seed 与缓存兼容性

seed 保持不变：

```python
self.seed = self("UCM_HASH_SEED")
```

不需要升级 seed 版本。block 输入会从：

```text
(parent_hash, block_token_ids)
```

变为：

```text
(parent_hash, block_token_ids, extra_keys)
```

即使 `extra_keys` 为 `None`，序列化后的 tuple 也不同，因此新 Block ID 不会命中
旧 token-only Block ID。现有缓存会自然变成 cold miss。

本设计在 UCM 当前运行约束下保持可复现性：相同 UCM/Python 软件版本、模型配置
和 rank 配置生成相同 key。跨 Python 版本或 pickle 协议变更的稳定性不属于本次
范围。

## 9. Connector 变更

### 9.1 Direct Connector

删除现有基于 token list 的 `generate_hash()` 实现。在 `hash_block_size` 和
`RequestHasher` 初始化完成后绑定 request 闭包：

```python
self.request_block_hasher = self.request_hasher.make_request_block_hasher(
    self.hash_block_size
)
```

lookup 改为：

```python
ucm_block_ids = self.request_block_hasher(request)
```

Store lookup、请求 metadata、load/dump 规划和 rank-specific key 派生不变。

### 9.2 LayerWise 与 Mock Connector

二者继承 Direct 的 lookup 和 request hasher，不需要独立 hash 实现。

### 9.3 CP Connector

CP 保持当前关系：

```text
hash_block_size = 配置的基础block size
物理/调度block size = 基础block size * cp_world_size
```

CP 使用归一化后的 TP/rank 配置重新构造 `RequestHasher` 后，必须重新绑定
`request_block_hasher`。现有 `[current_rank::cp_world_size]` key 切片逻辑不变。

### 9.4 Lite Connector

删除 Lite 中重复的 token-only `generate_hash()`。Lite 与 Direct 一样创建并调用
RequestHasher 闭包。

### 9.5 HLA Full Attention group

将 group API 从 token-list 输入改成 Request 输入：

```python
compute_block_hashes(group, request)
compute_all_group_block_ids(request)
```

每个 Full Attention group 使用自身 block size 和 group seed 绑定闭包：

```python
group.block_hasher = request_hasher.make_request_block_hasher(
    block_size=group.block_size,
    initial_hash=group.seed,
)
```

每个 group 独立扫描 Request。这样可以保持现有 group hash 结构和 block 映射。
当前实现本身已经针对每个 Full Attention group 重复序列化 token，本次只是在这些
已有遍历中增加 extra-key 生成。

### 9.6 HLA Mamba align group

Mamba align group 继续生成空的 per-block 占位项。持久化 state key 仍由以下内容
派生：

```text
Mamba group seed
+ state tag
+ sequence length
+ primary Full Attention prefix hash
```

Mamba group 不需要再次扫描 extra keys。一旦 primary Full Attention prefix hash
包含完整请求 extra keys，不同图片、LoRA、salt 或 prompt embedding 就会自然
生成不同的 Mamba state key。

LCM 对齐、两阶段 lookup、反向 state lookup 和 HLA load/dump dispatch 不变。

### 9.7 FAWA

FAWA 在确定 canonical hash block size 后绑定公共闭包：

```text
GPU默认值：256
Ascend：配置的基础block size * C4压缩率
```

随后将原继承调用替换为：

```python
canonical_hashes = self.request_block_hasher(request)
```

对于当前 DeepSeek V4 纯文本请求，`extra_keys` 通常为 `None`。canonical block
边界、FA/WA Store 隔离、prefix lookup、WA reverse lookup 和 load/dump 映射均不
改变。由于 hash tuple 增加了第三个元素，具体 hash 值会变化，因此会发生预期的
cold-cache 切换。

### 9.8 CacheBlend 兼容处理

本次明确将 CacheBlend 限制为纯文本请求语义。普通 prefix 路径使用公共
request-aware 闭包；chunk-local key 虽然会重置 parent 链，但同样使用三元组输入，
并固定 `extra_keys=None`。这样纯文本 chunk 的构建和查询 key 完全一致。具备多模态
能力的模型仍可正常启动并处理纯文本请求，CacheBlend 只在启动时记录 warning，不再
拒绝模型。包含多模态特征、LoRA、cache salt 或 prompt embedding 的请求会绕过
CacheBlend external cache，因为这些输入的可复用 chunk 相对语义需要单独设计。

## 10. 错误处理

Request hash 错误在 Scheduler 侧 external lookup 入口处理。

对于不支持或无效的语义请求：

1. 记录 request ID 和错误原因；
2. 返回零 external hit tokens；
3. 不创建 Connector request metadata；
4. 不向外部 Store dump KV。

HLA 任意一个 group 发生 hash 错误时，整个 external lookup 都应失败，不能使用
不完整的 group key 集合继续运行。

## 11. 性能考虑

多模态路径不会重新加载图片、运行图片 processor 或重新 hash 原始像素，只使用
vLLM input processing 阶段已经生成的 identifier。

对于纯文本请求，额外的 helper 检查和对 `None` tuple 元素的哈希，相比现有 tuple
序列化及 block hash 开销可以忽略。

对于具有多个 Full Attention group 的 HLA，请求 token 和 extra keys 会按 group
分别处理。这与当前架构一致，第一版以正确性为优先，接受该开销。共享细粒度语义
链作为后续优化。

当 UCM 与 vLLM 使用不同 block 边界时，prompt embedding 可能需要额外 hash。
如果边界相同，则可以复用 vLLM Request 内的 prompt-embedding hash 缓存。

计算得到的 UCM Block ID 继续保存在 Connector request metadata 中，因此 lookup、
load 和 dump 不会重复计算 request hash。

## 12. 测试方案

### 12.1 RequestHasher 公共测试

- 相同 token 和相同 extra keys 生成相同 Block ID；
- placeholder token 相同但多模态 identifier 不同时，从首个图片覆盖 block 开始
  key 必须不同；
- 第一张图片之前的纯文本 block 仍可复用；
- 相同多模态内容出现在不同位置时生成不同 key；
- 任一 block 分叉后，后续所有链式 block 都必须分叉；
- 不同 LoRA、cache salt 和 prompt embedding 生成不同 key；
- helper 扫描到 identifier 缺失且与 hash block 相交的多模态 feature 时 fail
  closed；UCM 不预扫描未参与哈希的尾部；
- 不对不完整尾部 block 计算 hash；
- 同一闭包重复调用时不能共享 parent 或多模态 cursor 状态。

### 12.2 Connector 测试

- Direct、LayerWise、Mock、Lite、CP 和 FAWA 均调用公共 request 闭包；
- CP 保持细粒度 hash 数量和 rank slicing 不变；
- FAWA 在不存在 extra keys 时保持 canonical block 数量和 FA/WA dispatch 映射
  不变；
- 单 Full Attention group 的 HLA 请求在多模态 identifier 不同时生成不同 Mamba
  state key；
- 多 Full Attention group 的 HLA 按每个 group 自身 block size 生成语义 key；
- 任一 HLA group hash 失败时禁用整个 external hit。

### 12.3 可复现性与迁移测试

- `PYTHONHASHSEED` 不同或未设置的进程，在相同部署配置下生成相同 UCM key；
- 旧二元组 token-only 输入和新三元组 request-aware 输入生成不同 Block ID；
- 同一 UCM/Python 版本及配置下，新进程可以读取另一进程写入的持久化 key。

## 13. 上线方案

新 hash tuple 会使所有已有 token-only key 失效。上线时必须按 cold-cache 切换
处理。

推荐顺序：

1. 禁止多模态 external-cache 流量进入旧实例；
2. 将所有缓存生产者和消费者升级到 request-aware 实现；
3. 重新开启多模态 external cache；
4. 监控 semantic hash failure、hash 延迟和 external hit rate；
5. 由 Store 正常 GC 回收不可达的 token-only 条目。

混合版本部署不适用于多模态 external cache，因为旧实例仍可能查询 token-only
key。

## 14. 实现文件

实际修改：

```text
ucm/integration/vllm/request_hasher.py      新RequestHasher模块
ucm/integration/vllm/ucm_connector.py      Direct/CP/Lite接入
ucm/integration/vllm/hla_connector.py      group改用Request和闭包
ucm/integration/vllm/hma_connector.py      FAWA绑定公共闭包
ucm/integration/vllm/blend_connector.py    保持chunk-local兼容行为
ucm/default_metrics_config.py              request-hash失败计数器
docs/source/user-guide/metrics/metrics_list.md  计数器文档
test/test_ucm_request_hasher.py             语义hash测试
test/test_ucm_hla_hash.py                   HLA/Mamba测试
test/test_ucm_blend_hash.py                 纯文本chunk hash兼容测试
test/test_ucm_connector_metrics.py          计数器注册测试
```

现有从 `ucm_connector.py` 导入 `RequestHasher` 的代码必须改为从新模块导入。
`Config` 的 import 保持不变。

## 15. 后续扩展点

### 15.1 稳定序列化和更强摘要算法

将 pickle/MD5 替换为 canonical encoding 和 SHA-256 或 BLAKE3，并截断为 Store
要求的16字节 `BlockId`。这样可以降低持久化 key 协议对 Python 实现细节的依赖。

### 15.2 更强的模型与布局指纹

将当前模型 basename namespace 替换为稳定的模型 revision 或显式 cache
namespace，并纳入所有影响 KV layout 的配置。

### 15.3 正式的 vLLM Connector API

将 request extra-key 生成或稳定语义 block descriptor 提升为公开的 vLLM KV
Connector API，避免依赖内部 helper 的代码位置。

### 15.4 HLA 共享语义 hash 链

按所有 Full Attention group block size 的最大公约数生成一次请求语义链，再在
每个 group 边界派生 group key，减少多 group HLA 的重复 token 序列化和 extra-key
扫描。

### 15.5 FAWA 多模态支持

如果 DeepSeek V4 后续支持多模态，需要验证 FAWA canonical hash block size 与
多模态 feature range 是否能保持正确的 FA/WA 边界语义，并补充专项回归测试。

### 15.6 CacheBlend 请求语义

设计 chunk-local 语义 hash：重置 parent 链、使用原请求 feature range、使用 chunk
相对位置，并保证 cache-salt 隔离。

### 15.7 Store 侧语义校验

在每个16字节 Store key 旁持久化完整 semantic digest 和 schema metadata。load KV
前验证 digest，使未来 key 构造错误退化为 miss，而不是加载语义不兼容的 KV。

### 15.8 vLLM helper 不可用时截断到安全文本前缀

当 `generate_block_hash_extra_keys()` 不可用时，从 `request.mm_features` 获取最早的
多模态插入 offset，并向下对齐到 UCM hash block 边界，只计算该位置之前的完整文本
block。与多模态插入位置相交的 block 及其链式后继全部禁止 lookup 和 dump。HLA 还
需要将所有 group 的公共截止点向下对齐到 LCM，并且只允许复用截止点之前的 Mamba
state。

这种部分降级只适用于能够可靠确定影响位置的语义，例如多模态 feature。LoRA、
cache salt 等请求级语义仍必须从第一个 block 开始 fail closed；prompt embedding
在无法安全确定影响范围时也必须整体 fail closed。

## 16. 验收标准

满足以下条件时，本次修改完成：

1. 所有纳入范围的 Connector 都不再保留独立 token-only request hash 循环；
2. 所有纳入范围的 Connector Block ID 都由 `RequestHasher` 闭包生成；
3. 不同多模态输入不能共享图片覆盖 block 及其后续 prefix Block ID；
4. HLA Mamba state key 继承修复后的 primary prefix语义；
5. UCM external key 在不设置 `PYTHONHASHSEED` 时仍可跨进程复现；
6. 不支持的语义请求 fail closed；
7. 现有 block size 和物理 load/dump 映射保持不变；
8. 所列单元测试和回归测试全部通过。
