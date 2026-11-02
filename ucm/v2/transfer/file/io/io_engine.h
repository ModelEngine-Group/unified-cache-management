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
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include "status/status.h"

namespace UC::Transfer {

class IoEngine {
public:
    struct InitParam {
        std::size_t ioDepth;
    };

    struct IoJob {
        std::vector<int32_t> fileHandles;
        std::vector<std::size_t> offsets;
        std::vector<std::size_t> lengths;
        std::vector<void*> addresses;
    };

public:
    virtual ~IoEngine() = default;

    virtual Status Setup(const InitParam& param) = 0;

    virtual Status SubmitReadV(std::size_t jobId, const IoJob& job) = 0;
    virtual Status SubmitWriteV(std::size_t jobId, const IoJob& job) = 0;

    virtual Expected<std::vector<std::pair<std::size_t, bool>>> GetFinished() = 0;
    virtual Status Wait(std::size_t jobId) = 0;
};

}  // namespace UC::Transfer
