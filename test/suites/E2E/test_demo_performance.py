import pytest
from common.config_utils import config_utils as config_instance
from common.EasyPerfBenchmark.EasyPerfBenchmark import EasyPerfBenchmark


# ---------------- Usage Example ----------------
@pytest.mark.feature("performance")
def test_easy_perf_benchmark(easy_perf_config=config_instance.get_config("easyPerf")):
    benchmark = EasyPerfBenchmark(easy_perf_config)
    results = benchmark.run_all()

    assert len(results) == len(easy_perf_config["experiments"])
    for result in results:
        assert result["ttft"] == 0
        assert result["tpot"] == 0
        assert result["avg_tps"] == 0


# ---------------- Fixture Example ----------------
class Calculator:
    def __init__(self):
        print("[Calculator Initialization]")
        pass

    def add(self, a, b):
        return a + b

    def divide(self, a, b):
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b


@pytest.fixture(scope="module", name="calc")
def calculator():
    return Calculator()


@pytest.mark.feature("mark")
class TestCalculator:
    # The calc instance will only be initialized on the first call, see the pytest documentation for more usage
    def test_add(self, calc):
        assert calc.add(1, 2) == 3

    def test_divide(self, calc):
        assert calc.divide(6, 2) == 3

    def test_divide_by_zero(self, calc):
        with pytest.raises(ZeroDivisionError):
            calc.divide(6, 0)


# ---------------- Write to DB Example ----------------
from common.db_utils import write_to_db


@pytest.mark.feature("db")
def test_db():
    # If the table does not exist, it will be created with the fields written for the first time.1
    data1 = {"name": "boss", "age": 30}
    data2 = {"name": "alice", "age": 30, "sex": "male"}
    data3 = {"name": "yang"}
    success = write_to_db("user_info", data1)
    write_to_db("user_info", data2)
    write_to_db("user_info", data3)
    print("Write to DB success" if success else "Write to DB Failure")


# ---------------- Read Config Example ----------------
from common.config_utils import config_utils as config_instance


@pytest.mark.feature("config")
def test_config():
    assert (
        config_instance.get_nested_config("database.host", "localhost") == "127.0.0.1"
    )
    easy_perf_config = config_instance.get_config("easyPerf")
    assert easy_perf_config["api"]["api_key"] == "sk-123456"
