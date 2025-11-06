# Pytest Demo 项目
 Pytest 测试框架，包括配置管理、数据库集成、性能测试和 HTML 报告生成。

## 📋 项目特性

- **现代化测试框架**: 基于 Pytest 7.0+ 的完整测试解决方案
- **配置管理**: 支持 YAML 配置文件，线程安全的单例模式配置管理
- **数据库集成**: 内置 MySQL 数据库支持，自动结果存储
- **性能测试**: 集成 EasyPerfBenchmark 性能测试工具
- **HTML 报告**: 自动生成带时间戳的 HTML 测试报告
- **标记系统**: 支持多维度测试标记（阶段、功能、平台等）

## 🗂️ 项目结构

```
pytest_demo/
├── common/                          # 公共模块
│   ├── __init__.py
│   ├── config_utils.py              # 配置管理工具
│   ├── db_utils.py                  # 数据库工具
│   ├── EasyPerfBenchmark/           # 性能测试模块
│   │   ├── __init__.py
│   │   ├── EasyPerfBenchmark.py     # 性能测试实现
│   └── └── requirements.txt
├── results/                         # 结果存储目录
├── suites/                          # 测试套件
│   ├── demo/                        # 示例测试
│   │   └── test_demo.py
│   └── E2E/                         # 端到端测试
│       └── test_performance.py
├── config.yaml                      # 主配置文件
├── conftest.py                      # Pytest 配置文件
├── pytest.ini                       # Pytest 配置
├── requirements.txt                 # 项目依赖
└── readme.md                        # 本文档
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- MySQL 5.7+ (可选，用于数据库功能)
- Git

### 安装步骤

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **配置数据库**（可选）

   编辑 `config.yaml` 文件中的数据库配置：
   ```yaml
   database:
     enabled: true
     host: "127.0.0.1"
     port: 3306
     name: "ucm_pytest"
     user: "root"
     password: "123456"
     charset: "utf8mb4"
   ```

3. **运行测试**
   ```bash
   # 运行所有测试
   pytest

   # 运行特定标记的测试
   pytest --stage=1
   pytest --feature=performance

   ```

## ⚙️ 配置说明


### config.yaml 配置

项目支持完整的 YAML 配置管理，主要配置项包括：

- **reports**: 报告配置（HTML 报告、时间戳等）
- **database**: 数据库连接配置
- **easyPerf**: 性能测试配置参考（API、模型、实验参数）

## 🧪 测试示例

### 基础功能测试

```python
# suites/E2E/test_demo_performance.py
import pytest

@pytest.fixture(scope="module", name="calc")
def calculator():
    return Calculator()

@pytest.mark.feature("mark")
class TestCalculator:
    def test_add(self, calc):
        assert calc.add(1, 2) == 3

    def test_divide_by_zero(self, calc):
        with pytest.raises(ZeroDivisionError):
            calc.divide(6, 0)
```

### 性能测试

```python
# suites/E2E/test_demo_performance.py
import pytest
from common.EasyPerfBenchmark.EasyPerfBenchmark import EasyPerfBenchmark

@pytest.mark.performance("performance1")
def test_easyperf_benchmark(easyperf_config=config_instance.get_config("easyPerf")):
    benchmark = EasyPerfBenchmark(easyperf_config)
    results = benchmark.run_all()
    assert len(results) == len(easyperf_config["experiments"])
```

## 🏷️ 测试标记系统

项目支持多维度的测试标记：

### 测试阶段标记
- `stage(0)`: 单元测试
- `stage(1)`: 冒烟测试
- `stage(2)`: 回归测试
- `stage(3)`: 发布测试

### 功能标记
- `feature`: 功能模块标记
- `platform`: 平台标记（GPU/NPU）

### 使用示例

```bash
# 运行冒烟测试及以上的所有测试
pytest --stage=1+

# 运行特定功能的测试
pytest --feature=performance
pytest --feature=performance, reliability
# 运行特定平台的测试
pytest --platform=gpu
```

## 📊 报告系统

### HTML 报告

项目自动生成带时间戳的 HTML 测试报告：
- 报告位置：`reports/pytest_YYYYMMDD_HHMMSS/report.html`
- 包含详细的测试结果、错误信息和执行时间
- 支持自定义报告标题和样式

### 数据库存储

如果启用数据库功能，测试结果会自动存储到 MySQL 数据库：
- 测试用例信息表：`test_case_info`
- 自动添加测试构建 ID 用于结果追踪

## 🔧 高级功能

### 配置管理

使用线程安全的单例模式配置管理：

```python
from common.config_utils import config_utils

# 获取配置
db_config = config_utils.get_config("database")
api_config = config_utils.get_nested_config("easyPerf.api")
```

### 数据库工具

内置数据库连接和操作工具：

```python
from common.db_utils import write_to_db, get_db

# 写入数据
# 若表不存在，则会以第一次写入的字段建表
data = {"name": "test", "value": 123}
success = write_to_db("test_table", data)
```


## 🛠️ 开发指南

### 添加新测试

1. 在 `suites/` 目录下创建新的测试文件
2. 使用适当的测试标记
3. 遵循命名规范：`test_*.py`
4. 使用 fixture 进行测试数据管理

### 扩展配置

1. 编辑 `config.yaml` 添加新的配置项
2. 在代码中使用 `config_utils` 访问配置
3. 确保配置项有合理的默认值