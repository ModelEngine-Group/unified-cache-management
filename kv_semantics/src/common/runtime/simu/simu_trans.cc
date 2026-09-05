/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include "simu_trans.h"
#include <cstdint>
#include <cstring>
#include "types.h"

namespace kv::runtime {

Status SimuTrans::DeviceToHost(void* device, void* host, size_t size)
{
    std::memcpy(host, device, size);
    return Status::OK();
}

Status SimuTrans::DeviceToHost(void* device[], void* host[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto s = this->DeviceToHost(device[i], host[i], size);
        if (!s.ok()) { return s; }
    }
    return Status::OK();
}

Status SimuTrans::DeviceToHost(void* device[], void* host, size_t size, size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto pDevice = device[i];
        auto pHost = static_cast<void*>(static_cast<int8_t*>(host) + size * i);
        auto s = this->DeviceToHost(pDevice, pHost, size);
        if (!s.ok()) { return s; }
    }
    return Status::OK();
}

Status SimuTrans::HostToDevice(void* host, void* device, size_t size)
{
    std::memcpy(device, host, size);
    return Status::OK();
}

Status SimuTrans::HostToDevice(void* host[], void* device[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto s = this->HostToDevice(host[i], device[i], size);
        if (!s.ok()) { return s; }
    }
    return Status::OK();
}

Status SimuTrans::HostToDevice(void* host, void* device[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto pHost = static_cast<void*>(static_cast<int8_t*>(host) + size * i);
        auto s = this->HostToDevice(pHost, device[i], size);
        if (!s.ok()) { return s; }
    }
    return Status::OK();
}

}  // namespace kv::runtime
