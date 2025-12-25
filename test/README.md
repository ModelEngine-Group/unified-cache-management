# Pytest Automation Testing Framework Guide
[English](README.md) | [简体中文](README_zh.md)

This guide introduces an automation testing framework built on **pytest 7.0+**, integrating core capabilities such as **configuration management**, **database integration**, **performance testing**, and **HTML report generation**. It is suitable for unit testing, functional testing, and end-to-end (E2E) testing scenarios.

---

## 📋 Core Framework Features

- **Modern Testing Architecture**: Built on pytest 7.0+, compatible with Python 3.11+, and supports a rich plugin ecosystem.
- **Centralized Configuration Management**: Thread-safe singleton-pattern configuration loading via YAML files.
- **Database Integration**: Built-in PostgreSQL support for automatically persisting test results to a database; if no database is configured, results are saved locally.
- **Visual Test Reporting**: Integrated with the pytest-html plugin to auto-generate clear and comprehensive HTML test reports.
- **Multi-dimensional Test Tagging**: Supports categorizing and filtering test cases by dimensions such as test stage, feature module, and execution platform.

---

## 🗂️ Project Directory Structure

```text
pytest_demo/
├── common/                          # Shared utility modules
│   ├── __init__.py
│   ├── config_utils.py              # Configuration loading and management
│   ├── db_utils.py                  # Database operation utilities
│   └── capture_utils.py             # Test data capture and export utilities
├── results/                         # Output directory for test results and logs
├── suites/                          # Test suite directory
│   ├── UnitTest/                    # Unit tests
│   ├── Feature/                     # Functional tests
│   └── E2E/                         # End-to-end tests
├── config.yaml                      # Main configuration file (YAML format)
├── conftest.py                      # Shared pytest configuration and fixture definitions
├── pytest.ini                       # pytest runtime parameter configuration
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation (this document)
```

---

## 🚀 Quick Start

### Environment Requirements

- Python 3.11 or higher

### Installation and Setup

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Configure Database**

   Edit the database section in `config.yaml`:

   ```yaml
   database:
     enabled: true                     # Enable database result storage (set to false to save to local directory)
     backup: "results/"
     host: "127.0.0.1"
     port: 5432
     name: "ucm_test"
     user: "postgres"
     password: "123456"
   ```

3. **Run Tests**

   ```bash
   # Navigate to the project root directory
   cd /test

   # Run all tests
   pytest

   # Run selected tests by markers (examples)
   pytest --stage=0                      # Run unit tests only
   pytest --feature=test_uc_performance  # Run a specific feature module
   ```

---

## ⚙️ Development Guidelines and Best Practices

### 1. Test Case Organization Conventions

All test cases must reside under the `suites/` directory and follow these naming conventions:

- **File Names**: Must start with `test_` (e.g., `test_login.py`)
- **Class Names**: Must start with `Test` (e.g., `TestClassA`)
- **Function Names**: Must start with `test_` (e.g., `test_valid_user_login`)

> pytest configuration (`pytest.ini`) has pre-set these discovery rules:
>
> ```ini
> testpaths = suites
> python_files = test_*.py
> python_classes = Test*
> python_functions = test_*
> ```

---

### 2. Multi-dimensional Marker System

The framework supports the following marker types:

| Marker Type | Value Description |
|------------|-------------------|
| `stage`    | `0`=Unit Test, `1`=Smoke Test, `2`=Regression Test, `3`=Release Test |
| `feature`  | Feature module identifier (e.g., `"uc_performance"`) |
| `platform` | Execution platform (e.g., `"Ascend"`, `"CUDA"`) |

**Usage Example:**

```python
import pytest

@pytest.mark.stage(0)
@pytest.mark.feature("uc_unit_test")
@pytest.mark.platform("Ascend")
def test_unit_case():
    assert True
```

**Run tests with specific markers:**

```bash
pytest --stage=0 --feature=capture_demo
```

---

### 3. Configuration and Parameter Management

- **Static configurations** (e.g., database connections, API endpoints) should be maintained in `config.yaml` and loaded via `config_utils`:

  ```python
  from common.config_utils import config_utils

  db_config = config_utils.get_config("database")
  api_url = config_utils.get_nested_config("easyPerf.api.url")
  ```

- **Dynamic test parameters** (e.g., input length, concurrency count) should use `@pytest.mark.parametrize`:

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

### 4. Automatic Test Data Collection and Export

The framework supports automatic capture and persistence of test result data using the `@export_vars` decorator.

**Usage Requirements:**
- The decorator must be the innermost decorator (closest to the function definition).
- The function’s return value must be a dictionary containing at least one of the following fields:
  - `_name`: Target database table name (**required**)
  - `_data`: A dictionary or list of dictionaries (key-value pairs become table columns)
  - `_proj`: A list of dictionaries (for structured projection data)

**Examples:**

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

> Data will be automatically written to the database or a local file based on the `database.enabled` setting in `config.yaml`.

---

### 5. Fixture Usage Guidelines

`@pytest.fixture` is used to provide reusable test dependencies (e.g., database connections, service instances).

**Example:**

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

> **Tip**: Fixture scope (`scope`) can be set to `function` (default), `class`, `module`, or `session` to optimize resource reuse efficiency.