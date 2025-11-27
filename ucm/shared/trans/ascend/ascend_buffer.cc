/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#include "ascend_buffer.h"
#include <acl/acl.h>

namespace UC::Trans {

std::shared_ptr<void> Trans::AscendBuffer::MakeDeviceBuffer(size_t size)
{
    void* device = nullptr;
    auto ret = aclrtMalloc(&device, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    if (ret == ACL_SUCCESS) { return std::shared_ptr<void>(device, aclrtFree); }
    return nullptr;
}

std::shared_ptr<void> Trans::AscendBuffer::MakeHostBuffer(size_t size)
{
    void* host = nullptr;
    auto ret = aclrtMallocHost(&host, size);
    if (ret == ACL_SUCCESS) { return std::shared_ptr<void>(host, aclrtFreeHost); }
    return nullptr;
}

Status Buffer::RegisterHostBuffer(void* host, size_t size, void** pDevice)
{
    void* device = nullptr;
    auto ret = aclrtHostRegister(host, size, ACL_HOST_REGISTER_MAPPED, &device);
    if (ret != ACL_SUCCESS) [[unlikely]] { return Status{ret, std::to_string(ret)}; }
    if (pDevice) { *pDevice = device; }
    return Status::OK();
}

void Buffer::UnregisterHostBuffer(void* host) { aclrtHostUnregister(host); }

} // namespace UC::Trans
