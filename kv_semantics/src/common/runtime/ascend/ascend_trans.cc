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
#include "ascend_trans.h"
#include <acl/acl.h>
#include <cstdint>
#include <string>
#include "types.h"

namespace kv::runtime {

Status AscendTrans::DeviceToHost(void* device, void* host, size_t size)
{
    auto ret = aclrtMemcpy(host, size, device, size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) { return Status::OK(); }
    return Status::Error(StatusCode::INTERNAL_ERROR, std::to_string(ret));
}

Status AscendTrans::DeviceToHost(void* device[], void* host[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto s = DeviceToHost(device[i], host[i], size);
        if (!s.ok()) { return s; }
    }
    return Status::OK();
}

Status AscendTrans::DeviceToHost(void* device[], void* host, size_t size, size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto pHost = static_cast<void*>(static_cast<int8_t*>(host) + size * i);
        auto s = DeviceToHost(device[i], pHost, size);
        if (!s.ok()) { return s; }
    }
    return Status::OK();
}

Status AscendTrans::HostToDevice(void* host, void* device, size_t size)
{
    auto ret = aclrtMemcpy(device, size, host, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret == ACL_SUCCESS) { return Status::OK(); }
    return Status::Error(StatusCode::INTERNAL_ERROR, std::to_string(ret));
}

Status AscendTrans::HostToDevice(void* host[], void* device[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto s = HostToDevice(host[i], device[i], size);
        if (!s.ok()) { return s; }
    }
    return Status::OK();
}

Status AscendTrans::HostToDevice(void* host, void* device[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto pHost = static_cast<void*>(static_cast<int8_t*>(host) + size * i);
        auto s = HostToDevice(pHost, device[i], size);
        if (!s.ok()) { return s; }
    }
    return Status::OK();
}

}  // namespace kv::runtime
