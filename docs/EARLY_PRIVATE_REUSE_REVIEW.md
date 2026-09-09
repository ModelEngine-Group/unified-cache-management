# 保持私有属性的 Early 复用修改

本轮 UCM 源码分支为 `codex/ucm-minimal-subclass-reuse`，起点为
`d03a13423cde9091a7d97b56d71d95751dc92a10`，原生比较基线为
`5e1427d6b89f69a2bff518dc59fbf5b6a52bc3c1`。起点与此前 `d0090af`
代码树相同。本文件不把管理仓库提交当作 UCM 提交。

## 本轮提交

| 提交 | 实际修改 |
|---|---|
| `4478e57` | 删除 TransBuffer/Handle 的两个 WaitReady；Early 使用原生状态接口等待 |
| `957c249` | Cache 配置改用原生私有 ParseConfig；删除 GetBufferManager；StorageBackend 恢复 private |
| `07b7d58` | 三个 Early 操作移出原生 C++ PipelineStore，使用同一个 Self 入口 |
| `f045970` | Python Pipeline 通过 super 和原生 builder 初始化，只适配工厂名 |
| `b5fd87f` | 删除 TryAlloc 副本；原生私有 Alloc 支持可选尝试上限 |
| `be6137d` | 绑定缺失时在创建 Pipeline 前失败，避免混用旧库后静默回退 |

本轮没有向原生类新增方法，移除了此前添加的七个方法：
TransBuffer::WaitReady、TransBuffer::Handle::WaitReady、CacheStore::GetBufferManager、
PipelineStore 的 ReservePrefix/PrefetchReserved/ReleaseReservation、TransBuffer::TryAlloc。
不计纯声明与定义的重复，仍有 19 个此前新增的原生类方法，见下表。

原有 private 变量没有改成 public/protected，原生 StorageBackend 函数恢复为
private。新增的两条 friend 声明授予 EarlyCacheStore、PosixReadLease 类级访问权；
这是访问授权变更，不是成员访问属性变更，也不能计作原生零修改。

## 相对原生仍存在的类和函数差异

| 原生类型 | 剩余差异 | 原因及边界 |
|---|---|---|
| UCMConnector | 修改构造选择和 PENDING 返回；新增 on_new_request、shutdown | 保留已有启动入口；外层生命周期可以进一步隔离，但选择 Early 外壳仍需合适的工厂接入，不能靠复制初始化或先创建再替换资源 |
| CacheStore | 一条 EarlyCacheStore friend 声明 | 复用既有 BufferManager 和私有 ParseConfig；原生方法主体不变 |
| TransBuffer | TryGet、PinReady；私有 Alloc 增加可选 maxAttempts | 原子 READY pin 和有界获取需要底层支持；分配主体复用原生，无新 TryAlloc |
| BufferStrategy | NodeCount、CanReserve | 从底层真实容量判断可保护范围 |
| LocalBufferStrategy / SharedBufferStrategy | 各一个 NodeCount | 分别读取原生实际 node 数，不复制共享布局 |
| StoreV1 | AcquireReadLease、ReservePrefix、PrefetchReserved、ReleaseReservation | 当前存储链的可选接口；独立扩展接口是可研究的替代，尚未实现 |
| HealthBreakerStore | 上述四个接口的转发 | 保留 health gating，释放始终透传；不能直接绕开包装 |
| PipelineStore（C++） | 类本身无差异；模块新增 include 和注册调用 | 通过既有 Self 访问同一入口，Early 绑定独立维护 |
| PosixStore | Setup 中 Configure 调用；Configure、GetLayout | 在 GC 启动前安装 guard，并访问同一 SpaceManager/SpaceLayout |
| Posix Config | RemoveGuard 和 acquireRemoveGuard | 通过原生初始化链传递可选删除保护 |
| SpaceLayout | Setup、RemoveFile、guard/延期标记、TakeDeferredDeletion；friend PosixReadLease | 在实际 unlink 处互斥；StorageBackend 保持 private |
| ShardGarbageCollector | RunGcCycle 中检查延期标记 | 保护冲突回到原生周期等待，不在同轮反复重试 |

仍直接修改的原生既有函数为 UCMConnector.__init__、
UCMConnector.get_num_new_matched_tokens、TransBuffer::Alloc、PosixStore::Setup、
SpaceLayout::Setup、SpaceLayout::RemoveFile、ShardGarbageCollector::RunGcCycle。
Alloc 的签名有扩展，但访问属性保持 private；默认没有尝试次数上限。

仍有 12 个原生文件存在差异；其中 cache_store.cc 和 posix_store.cc 还容纳新增
Early 子类及工厂，不能将文件内所有新增行都算作原生父类改写。
原生 C++ PipelineStore 类和 TransBuffer::Handle 类恢复为基线内容。
SpaceManager、BufferManager、Direct/LayerWise Connector 的原生主流程没有修改。

本轮没有修改 vLLM/vLLM-Ascend、推理参数、共享身份、HRRN 或控制通道，没有恢复
producer prefix、持久化观察或等待其他请求产出。历史 GLM 冷请求异常仍未解决。

## 检查与证据范围

用户暂停实验的要求继续有效。本轮未运行 Python/C++ 测试、编译、NPU 启动或请求。

已做的静态检查：

- Python 文件只执行 AST 解析，没有导入或执行项目代码。
- `git diff --check` 检查修改中的空白错误。
- 对照原生源码，核对 C++ PipelineStore 类内容、原生 Alloc 的分配主体及私有访问位置。
- 核对新旧函数调用点，确认 Python Early 操作使用新模块绑定。

`test/test_early_ready_reuse.py` 现在有 14 个测试函数。本轮新增覆盖同一 Pipeline
实例、绑定错误透传、原生 builder 配置与错误、缺失绑定时提前失败。
这些使用外部环境 mock 的 Python 用例尚未执行，不能证明 C++ ABI 或 NPU 行为。

`cache_early_store_test.cc` 新增全容量被 pin 时有界失败、释放后原生分配可继续
工作的用例。`posix_read_lease_test.cc` 改为通过 Early 操作和原生公开文件路径
验证行为，不再调用公开的 StorageBackend。C++、真实多进程 GC 和 A3 证据本轮
均为“未验证”；原有测试文件存在不等于这些场景已经通过。

## 后续运行前要求

必须在目标 A3 环境用本源码重新构建 UCM 的全部相关库。新 Python 代码调用
EarlyReservePrefix、EarlyPrefetchReserved、EarlyReleaseReservation 模块函数；
旧 PipelineStore 的成员绑定不再是当前接口。缺失新绑定会在初始化时明确失败。
这项检查只能排除缺失接口，不能单独证明所有动态库的 commit 一致。

恢复实验后，需先验证绑定导入、异常映射与生命周期，再运行现有 GLM/Qwen
关闭/开启、冷 L1 的 L2 命中、全 L1、真实多 rank、handoff 和异常清理对照。
本轮没有修改管理仓库的源码锁定或启动入口，不能将其原有构建产物当成本分支证据。

## 本地恢复与完整 diff

本轮文档提交后生成的恢复材料位于：

- `/data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/recovery/ucm-private-reuse.bundle`：
  当前 UCM 分支及其完整祖先历史，不依赖旧工作树的未提交文件。
- 同目录的 `ucm-private-reuse-vs-native.patch`：相对 `5e1427d` 的完整差异。
- 同目录的 `ucm-private-reuse-vs-start.patch`：相对本轮起点 `d03a134` 的完整差异。
- 同目录的 `ucm-private-reuse-manifest.json`：精确提交、文件 SHA-256 和 Git bundle 验证结果。

恢复到一个尚不存在的目录时，在 `/data/liyuhang` 下执行：

```bash
git clone --branch codex/ucm-minimal-subclass-reuse /data/liyuhang/glm51-vllm-ascend-ucm-minimal-review/recovery/ucm-private-reuse.bundle /data/liyuhang/ucm-private-reuse-restored
```

bundle 只恢复 UCM 源码及历史，不包含本轮构建产物；本轮没有构建。
管理仓库没有新增提交，已有未提交改动保持原状；未向远端推送。
