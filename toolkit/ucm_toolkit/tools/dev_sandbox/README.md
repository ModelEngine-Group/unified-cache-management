# dev-sandbox

dev-sandbox 测量 Ascend 主机内存到设备显存的拷贝带宽，覆盖普通内存、O_DIRECT、共享内存等分配方式与多流 CE、FFTS direct H2D SDMA 等传输引擎，并支持 GQA（每卡各自本地内存）和 MLA（单块共享内存分发到所有卡）两种拓扑。

← 返回 [UCM Toolkit 顶层文档](../../../README.md)

## 快速使用

```bash
ucm-toolkit run dev-sandbox --model-type <gqa|mla> --iodirect <true|false> --sdma <true|false> [参数...]
```

三个选择参数：

| 参数 | 取值 | 含义 |
| --- | --- | --- |
| `--model-type` | `gqa` / `mla` | 模型类型。 |
| `--iodirect` | `true` / `false` | 是否开启 O_DIRECT（直接 IO，绕过页缓存）。 |
| `--sdma` | `true` / `false` | 是否走 SDMA（direct H2D SDMA 传输路径）。 |

三个参数组合决定测试场景：

| 模型 | IO-direct | SDMA | 测试场景 |
| --- | --- | --- | --- |
| `gqa` | `false` | `false` | 多卡各自从本地主机内存，用多流 CE 拷到显存。 |
| `gqa` | `true` | `false` | 多卡各自用 O_DIRECT 方式从本地主机内存，多流 CE 拷到显存。 |
| `gqa` | `false` | `true` | 多卡各自从主机内存，用 FFTS direct H2D SDMA 拷到显存。 |
| `gqa` | `true` | `true` | 多卡各自用 O_DIRECT 方式从主机内存，走 FFTS direct H2D SDMA 拷到显存。 |
| `mla` | `false` | `false` | 一块共享主机内存在多卡间分发，每卡用多流 CE 拷到各自显存。 |
| `mla` | `true` | `false` | 同上（MLA 不区分 iodirect，走同一 CE 路径）。 |
| `mla` | `false` | `true` | 一块共享主机内存，用 FFTS direct H2D SDMA 分发到各卡显存。 |
| `mla` | `true` | `true` | 同上（MLA 不区分 iodirect，走同一 SDMA 路径）。 |

## 运行参数

三个选择参数之后可以接以下参数（全部可选，不传则用默认值）：

| 参数 | 含义 | 默认值 | 示例 |
| --- | --- | --- | --- |
| `-s` | 单个数据块大小 | `512M` | `-s 16K`、`-s 1M` |
| `-n` | 数据块数量 | `8` | `-n 512` |
| `-i` | 迭代轮数 | `128` | `-i 128` |
| `-d` | 设备（卡）数量 | `8` | `-d 8` |
| `-f` | 分片数（仅 SDMA 场景相关） | `0` | `-f 4` |

## 示例

```bash
# 测 GQA + 多流 CE：16K 块 × 512 块 × 128 轮 × 8 卡
ucm-toolkit run dev-sandbox --model-type gqa --iodirect false --sdma false \
  -s 16K -n 512 -i 128 -d 8

# 用默认参数测 MLA 共享分发 + SDMA
ucm-toolkit run dev-sandbox --model-type mla --iodirect false --sdma true
```

## 首次使用

首次运行前需先构建项目：

```bash
ucm-toolkit build dev-sandbox
```

构建细节（后端选择、自定义路径等）、底层原生子命令（`copy`/`trans`/`aio`）与完整 case 表见 [开发者文档](./developer.md)。
