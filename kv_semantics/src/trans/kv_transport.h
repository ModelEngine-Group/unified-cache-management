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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "trans_provider.h"
#include "types.h"

namespace UC::ASU {

struct TransportTask;

template <typename T>
struct BatchView {
    const T* data{nullptr};
    std::size_t size{0};

    const T& operator[](std::size_t i) const noexcept { return data[i]; }
    bool empty() const noexcept { return size == 0; }
};

class AsuTransport {
public:
    virtual ~AsuTransport() = default;

    virtual Status Init(const TransportConfig& config,
                        std::shared_ptr<TransProvider> transProvider) = 0;
    virtual Status Init(const std::string& configPath,
                        std::shared_ptr<TransProvider> transProvider) = 0;
    virtual Status Shutdown() = 0;
    virtual Status CheckHealth() = 0;

    virtual Status Submit(const std::shared_ptr<TransportTask>& task) = 0;

    // Best-effort cancellation, does not interrupt underlying UB/RoCE IO
    virtual Status Cancel(TaskId taskId) = 0;
};

std::unique_ptr<AsuTransport> CreateAsuTransport();

}  // namespace UC::ASU
