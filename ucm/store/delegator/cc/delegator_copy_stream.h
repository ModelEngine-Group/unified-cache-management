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

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "status/status.h"

namespace UC::Delegator {

class CopyStream {
public:
    CopyStream() = default;
    ~CopyStream();

    CopyStream(const CopyStream&) = delete;
    CopyStream& operator=(const CopyStream&) = delete;

    // Validates that the caller's current ACL context belongs to device_id. The caller must keep
    // that context current for this object's lifetime.
    Status Setup(std::int32_t device_id, std::size_t stream_number);
    aclrtStream NextStream() noexcept;
    Status DeviceToDeviceAsync(aclrtStream stream, void* destination,
                               std::size_t destination_capacity, const void* source,
                               std::size_t size);
    Status Synchronize(aclrtStream stream);
    Status SynchronizeAll();
    std::size_t Size() const { return streams_.size(); }

private:
    void Reset();

    std::vector<aclrtStream> streams_;
    std::size_t next_stream_{0};
};

}  // namespace UC::Delegator
