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
import time
from functools import wraps

import cupy
import numpy as np

from ucm.shared.trans import ucmtrans


def test_wrap(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"========>> Running in {func.__name__}:")
        result = func(*args, **kwargs)
        print()
        return result

    return wrapper


def make_host_memory(size, number, dtype, fill=False):
    host = cupy.cuda.alloc_pinned_memory(size * number)
    host_np = np.frombuffer(host, dtype=dtype)
    if fill:
        fixed_len = min(1024, number)
        host_np[:fixed_len] = np.arange(fixed_len, dtype=dtype)
    print("make:", host_np.shape, host_np.itemsize, host_np)
    return host


def compare(host1, host2, dtype):
    host1_np = np.frombuffer(host1, dtype=dtype)
    host2_np = np.frombuffer(host2, dtype=dtype)
    print("compare[1]:", host1_np.shape, host1_np.itemsize, host1_np)
    print("compare[2]:", host2_np.shape, host2_np.itemsize, host2_np)
    return np.array_equal(host1_np, host2_np)


@test_wrap
def trans_with_ce(d, size, number, dtype):
    s = d.MakeStream()
    host1 = make_host_memory(size, number, dtype, True)
    device = [cupy.empty(size, dtype=np.uint8) for _ in range(number)]
    device_ptr = np.array([d.data.ptr for d in device], dtype=np.uint64)
    host2 = make_host_memory(size, number, dtype)
    tp = time.perf_counter()
    s.HostToDeviceScatter(host1.ptr, device_ptr, size, number)
    s.DeviceToHostGather(device_ptr, host2.ptr, size, number)
    cost = time.perf_counter() - tp
    print(f"cost: {cost}s")
    print(f"bandwidth: {size * number / cost / 1e9}GB/s")
    assert compare(host1, host2, dtype)


@test_wrap
def trans_with_sm(d, size, number, dtype):
    s = d.MakeSMStream()
    host1 = make_host_memory(size, number, dtype, True)
    device = [cupy.empty(size, dtype=np.uint8) for _ in range(number)]
    device_ptr = np.array([d.data.ptr for d in device], dtype=np.uint64)
    device_ptr_cupy = cupy.empty(number, dtype=np.uint64)
    device_ptr_cupy.set(device_ptr)
    host2 = make_host_memory(size, number, dtype)
    tp = time.perf_counter()
    s.HostToDeviceScatter(host1.ptr, device_ptr_cupy.data.ptr, size, number)
    s.DeviceToHostGather(device_ptr_cupy.data.ptr, host2.ptr, size, number)
    cost = time.perf_counter() - tp
    print(f"cost: {cost}s")
    print(f"bandwidth: {size * number / cost / 1e9}GB/s")
    assert compare(host1, host2, dtype)


@test_wrap
def trans_with_ce_async(d, size, number, dtype):
    s = d.MakeStream()
    host1 = make_host_memory(size, number, dtype, True)
    device = [cupy.empty(size, dtype=np.uint8) for _ in range(number)]
    device_ptr = np.array([d.data.ptr for d in device], dtype=np.uint64)
    host2 = make_host_memory(size, number, dtype)
    tp = time.perf_counter()
    s.HostToDeviceScatterAsync(host1.ptr, device_ptr, size, number)
    s.DeviceToHostGatherAsync(device_ptr, host2.ptr, size, number)
    s.Synchronized()
    cost = time.perf_counter() - tp
    print(f"cost: {cost}s")
    print(f"bandwidth: {size * number / cost / 1e9}GB/s")
    assert compare(host1, host2, dtype)


@test_wrap
def trans_with_sm_async(d, size, number, dtype):
    s = d.MakeSMStream()
    host1 = make_host_memory(size, number, dtype, True)
    device = [cupy.empty(size, dtype=np.uint8) for _ in range(number)]
    device_ptr = np.array([d.data.ptr for d in device], dtype=np.uint64)
    device_ptr_cupy = cupy.empty(number, dtype=np.uint64)
    device_ptr_cupy.set(device_ptr)
    host2 = make_host_memory(size, number, dtype)
    tp = time.perf_counter()
    s.HostToDeviceScatterAsync(host1.ptr, device_ptr_cupy.data.ptr, size, number)
    s.DeviceToHostGatherAsync(device_ptr_cupy.data.ptr, host2.ptr, size, number)
    s.Synchronized()
    cost = time.perf_counter() - tp
    print(f"cost: {cost}s")
    print(f"bandwidth: {size * number / cost / 1e9}GB/s")
    assert compare(host1, host2, dtype)


def main():
    device_id = 0
    size = 36 * 1024
    number = 61 * 64
    dtype = np.float16
    print(f"ucmtrans: {ucmtrans.commit_id}-{ucmtrans.build_type}")
    cupy.cuda.Device(device_id).use()
    d = ucmtrans.Device()
    d.Setup(device_id)
    trans_with_ce(d, size, number, dtype)
    trans_with_sm(d, size, number, dtype)
    trans_with_ce_async(d, size, number, dtype)
    trans_with_sm_async(d, size, number, dtype)


if __name__ == "__main__":
    main()
