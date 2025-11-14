import pytest
from common.config_utils import config_utils as config_instance


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
from common.capture_utils import *


@pytest.mark.feature("capture")  # pytest must be the top
@export_vars
def test_capture_mix():
    """Mixed single + lists via '_name' + '_data'"""
    assert 1 == 1
    return {
        "_name": "demo",
        "_data": {
            "length": 10086,  # single value
            "accuracy": [0.1, 0.2, 0.3],  # list
            "loss": [0.1, 0.2, 0.3],  # list
        },
    }


# ---------------- Read Config Example ----------------
from common.config_utils import config_utils as config_instance


@pytest.mark.feature("config")
def test_config():
    assert (
        config_instance.get_nested_config("database.host", "localhost") == "127.0.0.1"
    )
