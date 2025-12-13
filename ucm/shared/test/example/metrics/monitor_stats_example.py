# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from functools import wraps

from ucm.shared.metrics import ucmmonitor


def test_wrap(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"========>> Running in {func.__name__}:")
        result = func(*args, **kwargs)
        print()
        return result

    return wrapper


@test_wrap
def metrics_with_update_stats():
    ucmmonitor.create_stats("PyStats")
    ucmmonitor.update_stats(
        "PyStats",
        {
            "save_duration": 1.2,
            "save_speed": 300.5,
            "load_duration": 0.8,
            "load_speed": 450.0,
            "interval_lookup_hit_rates": 0.95,
        },
    )

    data = ucmmonitor.get_stats("PyStats").data
    assert data["save_duration"][0] == 1.2
    assert len(data) == 5
    print(f"Get PyStats stats: {data}")

    data = ucmmonitor.get_stats_and_clear("PyStats").data
    assert data["save_duration"][0] == 1.2
    assert len(data) == 5
    print(f"Get PyStats stats and clear: {data}")

    data = ucmmonitor.get_stats_and_clear("PyStats").data
    assert len(data) == 0
    print(f"After clear then get PyStats: {data}")


@test_wrap
def metrics_with_update_all_stats():
    ucmmonitor.create_stats("PyStats1")
    ucmmonitor.create_stats("PyStats2")
    ucmmonitor.update_stats(
        "PyStats1",
        {
            "save_duration": 1.2,
            "save_speed": 300.5,
        },
    )

    ucmmonitor.update_stats(
        "PyStats2",
        {
            "load_duration": 0.8,
            "load_speed": 450.0,
        },
    )

    data = ucmmonitor.get_stats("PyStats1").data
    assert data["save_duration"][0] == 1.2
    assert len(data) == 2
    print(f"Only get PyStats1 stats: {data}")

    data = ucmmonitor.get_all_stats_and_clear().data
    assert data["save_duration"][0] == 1.2
    assert data["load_duration"][0] == 0.8
    assert len(data) == 4
    print(f"Get all stats and clear: {data}")

    data = ucmmonitor.get_stats("PyStats2").data
    assert len(data) == 0
    print(f"After clear then get PyStats2: {data}")


if __name__ == "__main__":
    metrics_with_update_stats()
    metrics_with_update_all_stats()
