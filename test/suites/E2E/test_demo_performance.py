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
class TestCapture:
    @export_vars
    def test_capture_mix(self):
        """Mixed single + lists via '_name' + '_data'"""
        assert 1 == 1
        return {
            "_name": "capture_demo",
            "_data": {
                "length": 1,  # single value
                "accuracy": [0.1, 0.2, 0.3],  # list
                "loss": [0.1, 0.2, 0.3],  # list
            },
        }

    @export_vars
    def test_capture_dict(self):
        """Mixed single + lists via '_name' + '_proj'"""
        return {
            "_name": "capture_demo",
            "_proj": {"length": 2, "accuracy": 0.1, "loss": 0.1},
        }

    @export_vars
    def test_capture_list_dict(self):
        """Mixed single + lists via '_name' + '_proj'"""
        return {
            "_name": "capture_demo",
            "_proj": [
                {"length": 3, "accuracy": 0.1, "loss": 0.1},
                {"length": 3, "accuracy": 0.2, "loss": 0.2},
                {"length": 3, "accuracy": 0.3, "loss": 0.3},
            ],
        }

    @export_vars
    def test_capture_proj(self):
        """Mixed single + lists via '_name' + '_proj'"""

        class Result:
            def __init__(self, length, accuracy, loss):
                self.length = length
                self.accuracy = accuracy
                self.loss = loss

        return {
            "_name": "capture_demo",
            "_proj": Result(4, 0.1, 0.1),
        }

    @export_vars
    def test_capture_list_proj(self):
        """Mixed single + lists via '_name' + '_proj'"""

        class Result:
            def __init__(self, length, accuracy, loss):
                self.length = length
                self.accuracy = accuracy
                self.loss = loss

        return {
            "_name": "capture_demo",
            "_proj": [
                Result(5, 0.1, 0.1),
                Result(5, 0.2, 0.2),
                Result(5, 0.3, 0.3),
            ],
        }


# ---------------- Read Config Example ----------------
from common.config_utils import config_utils as config_instance


@pytest.mark.feature("config")
def test_config():
    assert (
        config_instance.get_nested_config("database.host", "localhost") == "127.0.0.1"
    )
