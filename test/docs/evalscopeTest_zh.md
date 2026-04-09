# EvalScope 精度评测指南

本测试case基于 **EvalScope (v1.5.2)** 封装了自动化评测能力，用于便捷地测试大语言模型在主流学术基准及长上下文检索任务上的表现。

## 支持的评测类型

| 类型 | 说明 | 示例数据集 |
|------|------|------------|
| **主流数据集评测** | 覆盖数学、推理、知识、代码等能力的标准问答任务 | `aime24`、`aime25`、`aime26`、`gsm8k`、`longbench_v2`、`ceval`、`cmmlu`、`humaneval`、`mmlu`、`mmlu_pro` 等 |
| **大海捞针评测** | 评估模型在超长上下文中定位特定信息的能力（Needle In A Haystack） | - |

> **注意**：除大海捞针测试外，当前仅支持简单问答形式的数据集。需要额外运行环境或裁判模型介入的数据集暂未适配。

---

## 快速开始

### 1. 环境准备

- 推荐使用虚拟环境安装依赖：
  ```bash
  cd test
  pip install -r requirements.txt
  ```

### 2. 数据集准备

#### 在线环境（有网络）
- 框架会自动从 ModelScope 下载所需数据集，**无需手动操作**。

#### 离线环境（无网络）
- 需提前将数据集下载至统一目录。
- 确保子目录名称与任务列表中的标识完全一致。

**下载方式一：克隆单个数据集**
```bash
git clone https://www.modelscope.cn/datasets/evalscope/aime26.git
git clone https://www.modelscope.cn/datasets/ZhipuAI/LongBench-v2.git   # 注意克隆后需将目录重命名为 longbench_v2
git clone https://www.modelscope.cn/datasets/AI-ModelScope/Needle-in-a-Haystack-Corpus.git
```

**下载方式二：使用打包好的数据集压缩包**
- 访问 [ModelScope 数据集仓库](https://modelscope.cn/datasets/keriko/UCM_tools/files/dataset) 下载全量压缩包并解压至目标路径。

---

## 配置说明

### 通用参数

| 环境变量 | 默认值 | 说明 |
|----------|------|------|
| `SCOPE_DATASET_ROOT` |  | 数据集存放根目录 |
| `SCOPE_TREST_LIST` | `aime24,gsm8k`（示例） | 待评测数据集列表，逗号分隔 |

### 大海捞针专用参数

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `SCOPE_NEEDLE_MIN` | `1000` | 最小上下文长度（token 数） |
| `SCOPE_NEEDLE_MAX` | `32000` | 最大上下文长度（token 数） |

### 本地手动测试
直接修改 `test_evalscope.py` 中的以下常量即可：
```python
DEFAULT_DATASET_ROOT = "/mnt/data/evalscope/dataset"          # 数据集路径，联网环境下可为空
DEFAULT_TASK_LIST = ["aime24", "gsm8k"]      # 待测数据集
```

---

## 运行测试

### 单任务执行

```bash
cd test

# 主流数据集评测
pytest suites/E2E/test_evalscope.py::test_eval_accuracy

# 大海捞针评测
pytest suites/E2E/test_evalscope.py::test_needle_task
```

### 按标签批量执行

```bash
cd test
pytest --feature=evalscope
```

---

## 结果输出

### 1. EvalScope 原生输出
所有运行记录均保存在 `test/results/evalscope_outputs/` 目录下，按时间戳分子目录，包含：
- 评测配置文件
- 详细请求/响应日志
- 汇总指标文件（JSON）
- 可视化报告（HTML）

具体格式说明请参阅 [EvalScope 官方文档](https://evalscope.readthedocs.io/)。

### 2. 数据库持久化存储
评测结果会被自动解析并存入配置的数据库后端，便于集中查询与对比。

`test/results/` 目录下会生成以下文件：
- `eval_scope.jsonl`
- `eval_scope.csv`

如需自定义数据库连接，可修改配置中的 `results` 段落（支持 PostgreSQL、MongoDB 等）：

```yaml
results:
  localFile:
    path: "./results"
  # postgresql:
  #   host: "localhost"
  #   ...
  # mongodb:
  #   host: "127.0.0.1"
  #   ...
```

---

## 注意事项

1. 部分数据集名称需与 ModelScope 仓库名严格对应（如 `longbench_v2` 而非 `LongBench-v2`），离线使用时请留意目录重命名。
2. 若使用远程 API 进行评测，请确保 `llm_connection` 配置正确且服务可访问(示例：http://127.0.0.1:8080/)。
3. 大海捞针任务会使用**被测模型自身**作为裁判模型，请确保模型具备基本的指令遵循能力；且在`llm_connection`中配置模型路径作为`tokenizer_path`

测试过程
![](assets/pic1.png)
测试结果
```json
{
	"aime25": {
		"pretty_name": "AIME-2025",
		"model": "Qwen3-32B",
		"score": 0.0,
		"metrics": [{
			"name": "mean_acc",
			"score": 0.0,
			"macro_score": 0.0,
			"num": 30,
			"categories": [{
				"name": ["default"],
				"score": 0.0,
				"macro_score": 0.0,
				"num": 30,
				"subsets": [{
					"name": "default",
					"score": 0.0,
					"num": 30
				}]
			}]
		}],
		"analysis": "N/A"
	},
	"aime25.score": 0.0,
	"model_name": "Qwen3-32B",
	"test_id": "ad9ba909-1646-47b3-89d6-9240c6497593",
	"test_items": "pytestall_cases",
	"create_at": "2026-04-09 17:00:05.910252",
	"extra_info": ""
}
```
HTML测试报告
![](assets/pic2.png)
大海捞针测试热力图
![](assets/pic3.png)

注：使用Mock模型进行测试，所以得分均为0
