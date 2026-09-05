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
#ifndef KV_SEMANTICS_RUNTIME_TRANS_H
#define KV_SEMANTICS_RUNTIME_TRANS_H

#include <cstddef>
#include "types.h"

namespace UC::Trans {

class Trans {
public:
    virtual ~Trans() = default;

    virtual UC::ASU::Status DeviceToHost(void* device, void* host, size_t size) = 0;
    virtual UC::ASU::Status DeviceToHost(void* device[], void* host[], size_t size,
                                         size_t number) = 0;
    virtual UC::ASU::Status DeviceToHost(void* device[], void* host, size_t size,
                                         size_t number) = 0;

    virtual UC::ASU::Status HostToDevice(void* host, void* device, size_t size) = 0;
    virtual UC::ASU::Status HostToDevice(void* host[], void* device[], size_t size,
                                          size_t number) = 0;
    virtual UC::ASU::Status HostToDevice(void* host, void* device[], size_t size,
                                          size_t number) = 0;
};

}  // namespace UC::Trans

#endif  // KV_SEMANTICS_RUNTIME_TRANS_H
