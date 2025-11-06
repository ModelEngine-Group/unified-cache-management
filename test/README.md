# Pytest Demo Project
A comprehensive testing framework based on Pytest, featuring configuration management, database integration, performance testing, and HTML report generation.

## 📋 Project Features

- **Modern Testing Framework**: Complete testing solution built on Pytest 7.0+
- **Configuration Management**: YAML configuration file support with thread-safe singleton pattern
- **Database Integration**: Built-in MySQL support with automatic result storage
- **Performance Testing**: Integrated EasyPerfBenchmark performance testing tool
- **HTML Reporting**: Automatically generates timestamped HTML test reports
- **Tagging System**: Supports multi-dimensional test markers (stage, feature, platform, etc.)

## 🗂️ Project Structure

```
pytest_demo/
├── common/                          # Common modules
│   ├── __init__.py
│   ├── config_utils.py              # Configuration utilities
│   ├── db_utils.py                  # Database utilities
│   ├── EasyPerfBenchmark/           # Performance testing module
│   │   ├── __init__.py
│   │   ├── EasyPerfBenchmark.py     # Performance testing implementation
│   └── └── requirements.txt
├── results/                         # Results storage directory
├── suites/                          # Test suites
│   ├── demo/                        # Example tests
│   │   └── test_demo.py
│   └── E2E/                         # End-to-end tests
│       └── test_performance.py
├── config.yaml                      # Main configuration file
├── conftest.py                      # Pytest configuration file
├── pytest.ini                       # Pytest configuration
├── requirements.txt                 # Project dependencies
└── readme.md                        # This document
```

## 🚀 Quick Start

### Requirements

- Python 3.8+
- MySQL 5.7+ (optional, for database functionality)
- Git

### Installation Steps

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Database** (Optional)

   Edit the database configuration in `config.yaml`:
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

3. **Run Tests**
   ```bash
   # Run all tests
   pytest

   # Run tests with specific markers
   pytest --stage=1
   pytest --feature=performance
   ```

## ⚙️ Configuration Guide

### config.yaml Configuration

The project supports full YAML configuration management. Main configuration items include:

- **reports**: Report configuration (HTML reports, timestamps, etc.)
- **database**: Database connection settings
- **easyPerf**: Performance testing configuration reference (API, models, experiment parameters)

## 🧪 Test Examples

### Basic Functional Test

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

### Performance Test

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

## 🏷️ Test Tagging System

The project supports multi-dimensional test tagging:

### Test Stage Tags
- `stage(0)`: Unit tests
- `stage(1)`: Smoke tests
- `stage(2)`: Regression tests
- `stage(3)`: Release tests

### Feature Tags
- `feature`: Feature module tags
- `platform`: Platform tags (GPU/NPU)

### Usage Examples

```bash
# Run smoke tests and above
pytest --stage=1+

# Run tests for specific features
pytest --feature=performance
pytest --feature=performance,reliability

# Run tests for specific platforms
pytest --platform=gpu
```

## 📊 Reporting System

### HTML Reports

The project automatically generates timestamped HTML test reports:
- Location: `reports/pytest_YYYYMMDD_HHMMSS/report.html`
- Contains detailed test results, error information, and execution times
- Supports custom report titles and styling

### Database Storage

When database functionality is enabled, test results are automatically stored in MySQL:
- Test case information table: `test_case_info`
- Automatically adds test build ID for result tracking

## 🔧 Advanced Features

### Configuration Management

Uses thread-safe singleton pattern for configuration management:

```python
from common.config_utils import config_utils

# Get configuration
db_config = config_utils.get_config("database")
api_config = config_utils.get_nested_config("easyPerf.api")
```

### Database Utilities

Built-in database connection and operation tools:

```python
from common.db_utils import write_to_db, get_db

# Write data
# If table doesn't exist, it will be created using the fields from first write
data = {"name": "test", "value": 123}
success = write_to_db("test_table", data)
```

## 🛠️ Development Guide

### Adding New Tests

1. Create a new test file under the `suites/` directory
2. Use appropriate test markers
3. Follow naming convention: `test_*.py`
4. Use fixtures for test data management

### Extending Configuration

1. Edit `config.yaml` to add new configuration items
2. Access configuration in code using `config_utils`
3. Ensure configuration items have reasonable default values