# 连接 OD 盘框通信组件（CPU Product 版本）

- 接口：`NDS_client.h`

## 编译

```bash
# 自动检测 agent/release：两库齐全用 release，都没有用非集群 stub
make

# 非集群构建：不依赖厂商 agent 库，运行时配置 sceneMode = 0
make AGENT_MODE=none

# 集群构建：目录中必须同时有 libkv_agent.so 和 libceph-common.so.2
make AGENT_MODE=release AGENT_LIB_DIR=/path/from/vendor

# 测试 mock：含固定测试地址，不可用于生产
make AGENT_MODE=mock
```

`AGENT_MODE` 支持 `auto`（默认）、`release`、`none`、`mock`。`auto` 只找到一个厂商库时报错。`release` 会把两个厂商库复制到 `NDS_bin/`；`none` 编译同 ABI stub，不链接厂商库，误用集群模式时 agent 初始化会明确失败。旧 `AGENT_MOCK=true/false` 分别兼容映射到 `mock/release`。

`BUILD=debug` / `BUILD=release` 控制编译优化，`make clean` 清理生成物。

## 测试样例

`Test/test.c` 为简易一写一读样例：

```bash
g++ Test/test.c -o Test/test -L./NDS_bin -lndsclient -Wl,-rpath=./NDS_bin
```

使用前配置 `nds_config.conf` 中的地址；非集群构建设置 `sceneMode = 0`。

## 参数配置注意

- `QP_NUM_T`（`NDS_client.c`）配置 QP 个数，实际使用为 QP-1（QP0 为 Admin）。
- `MAX_QP_WR`（`common.h`）配置每个 QP 最大在飞请求数量。
- 以上参数修改后需要重新 `make`。
- Test 中建议 `Batch 数 × QP_NUM_T × MAX_QP_WR < 8192`，避免超过盘框在飞资源上限。
