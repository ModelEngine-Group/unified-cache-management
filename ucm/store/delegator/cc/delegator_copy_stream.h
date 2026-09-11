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
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "status/status.h"
#include "trans/stream.h"

namespace UC::Delegator {

class CopyStream {
public:
    CopyStream() = default;
    ~CopyStream();

    CopyStream(const CopyStream&) = delete;
    CopyStream& operator=(const CopyStream&) = delete;

    // Selects the configured runtime device and creates device copy streams on it.
    Status Setup(std::int32_t deviceId, std::size_t streamNumber);
    std::shared_ptr<Trans::Stream> NextStream() noexcept;
    Status DeviceToDeviceAsync(const std::shared_ptr<Trans::Stream>& stream, void* destination,
                               std::size_t destinationCapacity, void* source, std::size_t size);
    Status Synchronize(const std::shared_ptr<Trans::Stream>& stream);
    Status SynchronizeAll();
    std::size_t Size() const { return streams_.size(); }

private:
    void Reset();

    std::vector<std::shared_ptr<Trans::Stream>> streams_;
    std::size_t nextStream_{0};
};

}  // namespace UC::Delegator
