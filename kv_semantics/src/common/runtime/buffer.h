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
#ifndef KV_SEMANTICS_RUNTIME_BUFFER_H
#define KV_SEMANTICS_RUNTIME_BUFFER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include "types.h"

namespace UC::Trans {

class Buffer {
public:
    virtual ~Buffer() = default;

    virtual std::shared_ptr<void> MakeDeviceBuffer(size_t size) = 0;
    virtual std::shared_ptr<void> MakeHostBuffer(size_t size) = 0;

    virtual bool SupportsHostMappedDeviceBuffer() const { return false; }
    virtual std::shared_ptr<void> MakeHostMappedDeviceBuffer(size_t size,
                                                            void** pDevice = nullptr) = 0;

    virtual bool SupportsDeviceMappedHostBuffer() const { return false; }
    virtual std::shared_ptr<void> MakeDeviceMappedHostBuffer(size_t size) = 0;
};

UC::ASU::Status Memset(void* ptr, std::size_t size, std::int32_t value);

}  // namespace UC::Trans

#endif  // KV_SEMANTICS_RUNTIME_BUFFER_H
