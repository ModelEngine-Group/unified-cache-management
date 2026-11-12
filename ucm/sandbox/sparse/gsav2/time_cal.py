import time
from functools import wraps


def time_us(func):
    """简化的微秒计时装饰器"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1_000_000
        print(f"[TIMING] {func.__name__}: {elapsed:.2f} μs")
        return result

    return wrapper
