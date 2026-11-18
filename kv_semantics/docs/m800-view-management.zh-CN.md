# M800 场景视图管理

## 1. 目的与链路

M800 场景视图管理用于把 Ceph/OSDMap 中的 pool（namespace）和 ASU 节点信息发布给多个 Worker：

```text
Ceph MON / DPU Agent
        │
        ▼
Master：读取 namespace、in-state OSD、epoch，构造不可变多 namespace 快照
        │  GET /v1/views?since_epoch=N
        ▼
Worker/Slave：顺序尝试多个 Master，校验文本协议并发布到 ViewUniverse
        │
        ▼
后续 KvClient/Router 集成消费多 namespace 视图
```

`ViewUniverse` 保存完整、不可变、原子替换的视图快照。Master 负责候选生成和失败 namespace 回填；Slave 负责 HTTP 拉取、解码、旧 epoch 过滤和 `PublishAll`。完整的常驻调度、KvClientImpl/IO 路由和最终 `view.provider` 工厂接入不属于当前 Shard 2/3/4/5 的完成范围。

## 2. 部署角色

- `master`：连接 MON/Agent，提供 HTTP 视图服务。
- `worker`：实现代码中的 `M800ViewSlaveServer`，从 Master 拉取视图。
- 多 Master 是顺序 primary/backup：返回第一个可接受结果，不保证选择所有 Master 中的最大 epoch。
- `ViewServer` 提供双向兼容适配：旧实现只覆盖 `GetGlobalView` 时，默认 `GetAllViews` 将其包装为单 namespace；M800 实现覆盖 `GetAllViews`，继承的默认 `GetGlobalView` 返回按 nsId 排序的首个 namespace，空 Universe 返回 `NOT_FOUND`。
- Worker 的 Master 地址只允许数字 IPv4/IPv6，使用 `inet_pton` 和 `AI_NUMERICHOST`，禁止域名，避免 DNS 阻塞。
- 原 Router 的不可用节点重载已移除；在真正实现备用哈希/故障转移之前，不暴露会静默忽略不可用节点的接口。

## 3. 配置项

| 配置键 | 默认值 | 说明 |
|---|---|---|
| `view.role` | `master` | `master` 或 `worker` |
| `view.listen_addr` | `0.0.0.0:9100` | Master HTTP 监听地址 |
| `view.master_addrs` | 空 | Worker 的 Master 地址列表，逗号分隔；键存在时优先，即使值为空 |
| `view.master_addr` | 空 | 单地址兼容键，仅在未配置 `master_addrs` 时读取 |
| `view.ceph.mon_host` | 空 | Master 连接 MON 的地址，Master 必填 |
| `view.ceph.client_name` | `admin` | Ceph 客户端名 |
| `view.ceph.keyring` | `/etc/ceph/ceph.client.admin.keyring` | keyring 文件；无法读取时失败 |
| `view.ceph.agent_lib` | `/usr/lib/libkv_agent.so` | Agent 动态库路径 |
| `view.ceph.agent_fetch_timeout_ms` | `10000` | Agent 拉取超时，毫秒；优先于非 Ceph 命名空间键 |
| `view.agent_fetch_timeout_ms` | `10000` | 上述超时的兼容键 |
| `view.http_timeout_ms` | `2000` | Worker 整轮拉取和 Master 单连接请求的超时 |
| `view.refresh_interval_ms` | `30000` | Agent 视图刷新间隔 |
| `view.endpoint_protocol` | `ub` | 下发 ASU endpoint 协议 |
| `view.endpoint_port` | `6000` | 下发 ASU endpoint 默认端口 |
| `view.ceph.filter` | `in` | 当前仅支持 in-state 过滤；其他值告警并回退 |

超时和间隔解析拒绝嵌入 NUL、非法数字和超出实现范围的值，并按对应配置策略回退或报错。示例见 `examples/m800_view.conf`。

## 4. HTTP 与文本协议

### HTTP

```text
GET /v1/views?since_epoch=N
Connection: close
```

- `200`：返回完整文本快照。
- `304`：调用方传入的非空对象已经持有对应 epoch，可保持 no-op。
- `400`：请求行、HTTP 版本或接收结构非法。
- `404`：路径不是 `/v1/views`。
- `405`：方法不是 GET，并返回 `Allow: GET`。
- `500`：handler 或响应体生成失败。

空输出对象始终使用 `since_epoch=0` 全量拉取，即使该 Worker 之前已为另一个对象获取过视图。非空对象使用自己的 epoch 发送条件请求。Worker 不维护 `cachedEpoch_`，也不与本地缓存做 epoch 比较；Master 的完整、可解码、非空响应直接交给 `ViewUniverse::PublishAll`，由它统一决定发布、旧快照 no-op 或相等 no-op。

### 文本快照

- 统一 `globalEpoch`，每个 namespace 的 `viewEpoch` 等于该值，`viewId = nsId`。
- `KvNsId = XXH32(poolName raw bytes, len, 5381)`；Master 检测不同 pool 名的哈希冲突。
- 每个 namespace 包含 `viewEpoch`、`viewId`、`asuIds` 和对应 `asuInfo.N`。
- 可选 `localKvNsId.N` 必须紧跟 `asuInfo.N`，默认 0 不输出，保持 M800 wire format 兼容。
- 编解码顺序确定，严格校验字段、数字、CR/LF、Content-Length、截断和大小上限；Decode 失败不修改输出对象。
- `PublishAll` 按 nsId 输出逐 namespace 日志：`ns={} view published, epocj={}->{}, asu_count={}` 或 `ns={} view not published, epocj unchanged={}`，使用不限频 INFO 记录。

## 5. Agent ABI 兼容策略

当前 ABI 版本为 **v1**，导出 12 个 `kv_client_agent_*` 符号，保留既有 `kv_client_agent_get_all_ns_name` 名称。

加载过程：

1. 使用 `dlopen(..., RTLD_NOW | RTLD_LOCAL)` 打开 `view.ceph.agent_lib`。
2. 解析全部必需符号，包括 `kv_client_agent_get_abi_version`。
3. 要求库返回的版本等于 `KV_CLIENT_AGENT_ABI_VERSION`。
4. init 同时传入 ABI 版本和 `sizeof(kv_client_agent_config)`，阻止结构体布局不一致的调用。
5. 缺库、缺符号、版本不匹配或初始化失败均使 Master 启动失败；不静默回退到不兼容实现。
6. 运行中刷新失败时保留 Master 的 last-known-good 快照，不以空视图或部分候选覆盖。

销毁顺序为停止刷新、完成进行中的调用、释放 ABI 输出、destroy、dlclose。Stop 会请求取消，但第三方 ABI 永久不返回且忽略 stop 的情况仍需供应商契约或进程隔离。

## 6. 容量与可用性约束

- 每个快照最多 **64 个 namespace**。
- 每个 namespace 最多 **4096 个 ASU**。
- Master/Codec 对超限输入执行告警、拒绝或确定性截断，具体以对应路径实现为准。
- HTTP v1 在 accept 线程中**串行处理连接**，listen backlog 为 **16**。
- 单连接受绝对 deadline 限制，默认 **2 秒**；Stop 会 shutdown 活动连接。
- 串行处理不等于支持任意规模 Worker。慢连接会占用当前服务位置，大规模部署应：
  - 限制单个 Master 的 Worker 数和轮询频率；
  - 部署多个 Master 并分散 Worker 地址列表；
  - 需要更高并发时引入有界线程池和连接队列，而不是直接无限创建线程。

## 7. 失败与生命周期语义

- Master 首图超时、Agent 错误、namespace 失败回填、空/旧/相等快照 no-op 均有明确处理。
- Worker 先启动时返回连接错误；无效 Master 地址失败后尝试下一个地址。
- 所有 Master/地址/connect/send/read 共用一轮 deadline；Stop 可中断活动 socket。
- 空输出强制全量拉取，避免“缓存 epoch 非零但新输出为空却接受 304”。
- 旧 Master 结果不会覆盖较新的调用方/Worker epoch；全部有效但过旧时按既定约定 `OK + no-op`。

## 8. 验证

```bash
./build-m800-next/tests/kv.test \
  --gtest_filter='ViewServerTest.*:ViewUniverseTest.*:M800View*:ViewHttpServerTest.*'
ctest --test-dir build-m800-next --output-on-failure -j2
```

覆盖配置、XXH32、ABI/dlopen、Master 首图与部分失败、Codec 非法输入、HTTP 400/404/405/500、空输出全量拉取、多 Master、旧 epoch、304、超时、Stop/Start 和不可变快照。

真实环境另验证了 Ceph MON/OSD、真实 `libkv_agent.so`、Master→HTTP→Slave、坏地址回退和 namespace 动态增删。DPU Agent 必须先就绪，不能仅靠 MON/OSD 证明持续刷新可用。
