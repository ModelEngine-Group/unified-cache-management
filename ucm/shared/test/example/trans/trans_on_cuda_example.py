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
    element_size = np.dtype(dtype).itemsize
    num_elements = size // element_size
    host = cupy.cuda.alloc_pinned_memory(size * number)
    host_np = np.frombuffer(host, dtype=dtype, count=num_elements)
    if fill:
        fixed_len = min(1024, number)
        host_np[:fixed_len] = np.arange(fixed_len, dtype=dtype)
    print("make:", host_np.shape, host_np.itemsize, host_np)
    return host


def make_batch_host_memory(size, number, dtype, fill=False):
    element_size = np.dtype(dtype).itemsize
    num_elements = size // element_size
    host = []
    for i in range(number):
        pinned_mem = cupy.cuda.alloc_pinned_memory(size)
        np_array = np.frombuffer(pinned_mem, dtype=dtype, count=num_elements)
        if fill:
            value = np.uint64(1023 + i)
            np_array[0] = value
            np_array[-1] = value
        host.append(pinned_mem)
        if i == 0:
            print("make:", np_array.shape, np_array.itemsize, np_array)
    return host


def compare(host1, host2, size, dtype, show_detail=True):
    element_size = np.dtype(dtype).itemsize
    num_elements = size // element_size
    host1_np = np.frombuffer(host1, dtype=dtype, count=num_elements)
    host2_np = np.frombuffer(host2, dtype=dtype, count=num_elements)
    if show_detail:
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
    assert compare(host1, host2, size, dtype)


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
    assert compare(host1, host2, size, dtype)


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
    assert compare(host1, host2, size, dtype)


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
    assert compare(host1, host2, size, dtype)


@test_wrap
def trans_batch_with_ce(d, size, number, dtype):
    s = d.MakeStream()
    host1 = make_batch_host_memory(size, number, dtype, True)
    host1_ptr = np.array([h.ptr for h in host1], dtype=np.uint64)
    device = [cupy.empty(size, dtype=np.uint8) for _ in range(number)]
    device_ptr = np.array([d.data.ptr for d in device], dtype=np.uint64)
    host2 = make_batch_host_memory(size, number, dtype)
    host2_ptr = np.array([h.ptr for h in host2], dtype=np.uint64)
    tp = time.perf_counter()
    s.HostToDeviceBatch(host1_ptr, device_ptr, size, number)
    s.DeviceToHostBatch(device_ptr, host2_ptr, size, number)
    cost = time.perf_counter() - tp
    print(f"cost: {cost}s")
    print(f"bandwidth: {size * number / cost / 1e9}GB/s")
    for h1, h2 in zip(host1, host2):
        assert compare(h1, h2, size, dtype, False)


@test_wrap
def trans_batch_with_sm(dev, size, number, dtype):
    s = dev.MakeSMStream()
    h1 = make_batch_host_memory(size, number, dtype, True)
    h1_ptr = np.array([h.ptr for h in h1], dtype=np.uint64)
    h1_ptr_cupy = cupy.empty(number, dtype=np.uint64)
    h1_ptr_cupy.set(h1_ptr)
    d = [cupy.empty(size, dtype=np.uint8) for _ in range(number)]
    d_ptr = np.array([d.data.ptr for d in d], dtype=np.uint64)
    d_ptr_cupy = cupy.empty(number, dtype=np.uint64)
    d_ptr_cupy.set(d_ptr)
    h2 = make_batch_host_memory(size, number, dtype)
    h2_ptr = np.array([h.ptr for h in h2], dtype=np.uint64)
    h2_ptr_cupy = cupy.empty(number, dtype=np.uint64)
    h2_ptr_cupy.set(h2_ptr)
    tp = time.perf_counter()
    s.HostToDeviceBatch(h1_ptr_cupy.data.ptr, d_ptr_cupy.data.ptr, size, number)
    s.DeviceToHostBatch(d_ptr_cupy.data.ptr, h2_ptr_cupy.data.ptr, size, number)
    cost = time.perf_counter() - tp
    print(f"cost: {cost}s")
    print(f"bandwidth: {size * number / cost / 1e9}GB/s")
    for x, y in zip(h1, h2):
        assert compare(x, y, size, dtype, False)


@test_wrap
def trans_batch_with_ce_async(d, size, number, dtype):
    s = d.MakeStream()
    host1 = make_batch_host_memory(size, number, dtype, True)
    host1_ptr = np.array([h.ptr for h in host1], dtype=np.uint64)
    device = [cupy.empty(size, dtype=np.uint8) for _ in range(number)]
    device_ptr = np.array([d.data.ptr for d in device], dtype=np.uint64)
    host2 = make_batch_host_memory(size, number, dtype)
    host2_ptr = np.array([h.ptr for h in host2], dtype=np.uint64)
    tp = time.perf_counter()
    s.HostToDeviceBatchAsync(host1_ptr, device_ptr, size, number)
    s.DeviceToHostBatchAsync(device_ptr, host2_ptr, size, number)
    s.Synchronized()
    cost = time.perf_counter() - tp
    print(f"cost: {cost}s")
    print(f"bandwidth: {size * number / cost / 1e9}GB/s")
    for h1, h2 in zip(host1, host2):
        assert compare(h1, h2, size, dtype, False)


@test_wrap
def trans_batch_with_sm_async(dev, size, number, dtype):
    s = dev.MakeSMStream()
    h1 = make_batch_host_memory(size, number, dtype, True)
    h1_ptr = np.array([h.ptr for h in h1], dtype=np.uint64)
    h1_ptr_cupy = cupy.empty(number, dtype=np.uint64)
    h1_ptr_cupy.set(h1_ptr)
    d = [cupy.empty(size, dtype=np.uint8) for _ in range(number)]
    d_ptr = np.array([d.data.ptr for d in d], dtype=np.uint64)
    d_ptr_cupy = cupy.empty(number, dtype=np.uint64)
    d_ptr_cupy.set(d_ptr)
    h2 = make_batch_host_memory(size, number, dtype)
    h2_ptr = np.array([h.ptr for h in h2], dtype=np.uint64)
    h2_ptr_cupy = cupy.empty(number, dtype=np.uint64)
    h2_ptr_cupy.set(h2_ptr)
    tp = time.perf_counter()
    s.HostToDeviceBatchAsync(h1_ptr_cupy.data.ptr, d_ptr_cupy.data.ptr, size, number)
    s.DeviceToHostBatchAsync(d_ptr_cupy.data.ptr, h2_ptr_cupy.data.ptr, size, number)
    s.Synchronized()
    cost = time.perf_counter() - tp
    print(f"cost: {cost}s")
    print(f"bandwidth: {size * number / cost / 1e9}GB/s")
    for x, y in zip(h1, h2):
        assert compare(x, y, size, dtype, False)


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
    trans_batch_with_ce(d, size, number, dtype)
    trans_batch_with_sm(d, size, number, dtype)
    trans_batch_with_ce_async(d, size, number, dtype)
    trans_batch_with_sm_async(d, size, number, dtype)


if __name__ == "__main__":
    main()
