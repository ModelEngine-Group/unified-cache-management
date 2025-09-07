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
#ifndef UNIFIEDCACHE_IDEVICE_H
#define UNIFIEDCACHE_IDEVICE_H

#include <cstddef>
#include <memory>
#include "status/status.h"

namespace UC {

class IDevice {
public:
    IDevice(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber)
        : deviceId{deviceId}, bufferSize{bufferSize}, bufferNumber{bufferNumber}
    {
    }
    virtual ~IDevice() = default;
    virtual Status Setup() = 0;
    virtual void* GetBuffer(const size_t idx) = 0;
    virtual void PutBuffer(const size_t idx, void* ptr) = 0;
    virtual Status H2DBatch(const uintptr_t* from, uintptr_t* to, const size_t number, const size_t size) = 0;
    virtual Status D2HBatch(const uintptr_t* from, uintptr_t* to, const size_t number, const size_t size) = 0;

protected:
    int32_t deviceId;
    size_t bufferSize;
    size_t bufferNumber;
};

class DeviceFactory {
public:
    static std::unique_ptr<IDevice> Make(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber);
};

} // namespace UC

#endif
