# Pytest 自动化测试框架使用指南

本指南旨在介绍基于 **pytest 7.0+** 构建的自动化测试框架，该框架集成了 **配置管理、数据库集成、性能测试** 与 **HTML 报告生成** 等核心能力，适用于单元测试、功能测试及端到端（E2E）测试场景。

---

## 📋 框架核心特性

- **现代化测试架构**：基于 pytest 7.0+ 构建，兼容 Python 3.11+，支持丰富的插件生态。
- **集中式配置管理**：通过 YAML 配置文件实现线程安全的单例模式配置加载。
- **数据库集成能力**：内置 postgresql 支持，可自动将测试结果持久化至数据库；若未配置数据库，可将结果保存至本地目录。
- **可视化测试报告**：集成 pytest-html 插件，自动生成结构清晰、信息完整的 HTML 测试报告。
- **多维测试标记体系**：支持按测试阶段（stage）、功能模块（feature）、运行平台（platform）等维度对测试用例进行分类与筛选。

---

## 🗂️ 项目目录结构

```text
pytest_demo/
├── common/                          # 公共工具模块
│   ├── __init__.py
│   ├── config_utils.py              # 配置加载与管理
│   ├── db_utils.py                  # 数据库操作封装
│   └── capture_utils.py             # 测试数据捕获与导出工具
├── results/                         # 测试结果与日志输出目录
├── suites/                          # 测试套件目录
│   ├── UnitTest/                    # 单元测试
│   ├── Feature/                     # 功能测试
│   └── E2E/                         # 端到端测试
├── config.yaml                      # 主配置文件（YAML 格式）
├── conftest.py                      # pytest 共享配置与 fixture 定义
├── pytest.ini                       # pytest 运行参数配置
├── requirements.txt                 # 项目依赖列表
└── README.md                        # 项目说明文档（本文档）
```

---

## 🚀 快速入门

### 环境要求

- Python 3.11 或更高版本

### 安装与配置

1. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

2. **（可选）配置数据库**

   编辑 `config.yaml` 中的数据库配置段落：

   ```yaml
   database:
     enabled: true                     # 启用数据库结果存储（设为 false 则落盘保存至配置目录）
     backup: "results/"
     host: "127.0.0.1"
     port: 5432
     name: "ucm_test"
     user: "postgres"
     password: "123456"
   ```

3. **执行测试**

   ```bash
   # 进入测试项目根目录
   cd /test

   # 运行全部测试
   pytest

   # 按标记筛选执行（示例）
   pytest --stage=0                      # 仅运行单元测试
   pytest --feature=test_uc_performance  # 运行指定功能模块
   ```

---

## ⚙️ 开发规范与最佳实践

### 1. 测试用例组织约定

所有测试用例必须置于 `suites/` 目录下，并遵循以下命名规范：

- **文件名**：必须以 `test_` 开头（如 `test_login.py`）
- **类名**：必须以 `Test` 开头（如 `TestClassA`）
- **函数名**：必须以 `test_` 开头（如 `test_valid_user_login`）

> pytest 配置（`pytest.ini`）已预设以下发现规则：
>
> ```ini
> testpaths = suites
> python_files = test_*.py
> python_classes = Test*
> python_functions = test_*
> ```

---

### 2. 多维标记（Markers）体系

当前框架支持以下三类标记：

| 标记类型   | 取值说明                                   |
|------------|----------------------------------------|
| `stage`    | `0`=单元测试, `1`=冒烟测试, `2`=回归测试, `3`=版本测试 |
| `feature`  | 功能模块标识（如 `"uc_performance"`）           |
| `platform` | 运行平台（如 `"Ascend"`, `"CUDA"`）           |

**用法示例：**

```python
import pytest

@pytest.mark.stage(0)
@pytest.mark.feature("uc_unit_test")
@pytest.mark.platform("Ascend")
def test_unit_case():
    assert True
```

**运行指定标记的测试：**

```bash
pytest --stage=0 --feature=capture_demo
```

---

### 3. 配置与参数管理

- **静态配置**（如数据库连接、API 地址等）应统一维护在 `config.yaml` 中，并通过 `config_utils` 加载：

  ```python
  from common.config_utils import config_utils

  db_config = config_utils.get_config("database")
  api_url = config_utils.get_nested_config("easyPerf.api.url")
  ```

- **动态测试参数**（如输入长度、并发数等）应使用 `@pytest.mark.parametrize` 进行参数化：

  ```python
  perf_scenarios = [
      (4000, 1024, 80),
      (2000, 512, 40)
  ]
  scenario_ids = [f"in_{s[0]}-out_{s[1]}-con_{s[2]}" for s in perf_scenarios]

  @pytest.mark.feature("uc_performance_test")
  @pytest.mark.parametrize("in_tokens, out_tokens, concurrent", perf_scenarios, ids=scenario_ids)
  def test_performance(in_tokens, out_tokens, concurrent):
      res = run_case(in_tokens, out_tokens, concurrent)
      assert res is not None
  ```

---

### 4. 测试数据自动采集与导出

框架支持通过 `@export_vars` 装饰器自动捕获并持久化测试结果数据。

**使用要求：**
- 装饰器必须置于测试函数最内层（即最靠近函数定义）
- 函数返回值必须为字典，包含以下字段之一：
  - `_name`：目标数据库表名（必填）
  - `_data`：字典或字典列表（键值对将转为表字段）
  - `_proj`：字典列表（用于结构化投影数据）

**示例：**

```python
from common.capture_utils import export_vars
import pytest

@pytest.mark.feature("capture_demo")
@export_vars
def test_capture_scalar():
    return {"_name": "demo_case", "_data": {"accuracy": 0.95, "loss": 0.05}}

@pytest.mark.feature("capture_demo")
@export_vars
def test_capture_list():
    return {"_name": "demo_case", "_data": {"accuracy": [0.9, 0.95], "loss": [0.1, 0.05]}}

@pytest.mark.feature("demo")
@export_vars
def test_proj_data():
    return {
        "_name": "demo_case",
        "_proj": [
            {"accuracy": 0.88, "loss": 0.12},
            {"accuracy": 0.92, "loss": 0.08}
        ]
    }
```

> 数据将根据 `config.yaml` 中的 `database.enabled` 设置，自动写入数据库或本地文件。

---

### 5. Fixture 使用规范

`@pytest.fixture` 用于提供可复用的测试依赖（如数据库连接、服务实例等）。

**示例：**

```python
import pytest

class Calculator:
    def add(self, a, b): return a + b
    def divide(self, a, b): return a / b

@pytest.fixture(scope="module")
def calc():
    return Calculator()

@pytest.mark.feature("calculator")
class TestCalculator:
    def test_add(self, calc):
        assert calc.add(1, 2) == 3

    def test_divide_by_zero(self, calc):
        with pytest.raises(ZeroDivisionError):
            calc.divide(6, 0)
```

> **提示**：Fixture 的作用域（`scope`）可设为 `function`（默认）、`class`、`module` 或 `session`，以优化资源复用效率。  

