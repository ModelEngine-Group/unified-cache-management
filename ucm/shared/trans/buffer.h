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
#ifndef UNIFIEDCACHE_TRANS_BUFFER_H
#define UNIFIEDCACHE_TRANS_BUFFER_H

#include <memory>
#include "status/status.h"

namespace UC::Trans {

class Buffer {
public:
    virtual ~Buffer() = default;

    virtual std::shared_ptr<void> MakeDeviceBuffer(size_t size) = 0;
    virtual Status MakeDeviceBuffers(size_t size, size_t number) = 0;
    virtual std::shared_ptr<void> GetDeviceBuffer(size_t size) = 0;

    virtual std::shared_ptr<void> MakeHostBuffer(size_t size) = 0;
    virtual Status MakeHostBuffers(size_t size, size_t number) = 0;
    virtual std::shared_ptr<void> GetHostBuffer(size_t size) = 0;

    static Status RegisterHostBuffer(void* host, size_t size, void** pDevice = nullptr);
    static void UnregisterHostBuffer(void* host);
};

} // namespace UC::Trans

#endif
