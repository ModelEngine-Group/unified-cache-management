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
#ifndef UNIFIEDCACHE_YUANRONG_STORE_CC_COPY_STREAM_H
#define UNIFIEDCACHE_YUANRONG_STORE_CC_COPY_STREAM_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "logger/logger.h"
#include "status/status.h"
#include "trans/device.h"

namespace UC::YuanRongStore {

class CopyStream {
    int32_t deviceId_{-1};
    size_t next_{0};
    std::vector<std::shared_ptr<Trans::Stream>> streams_;

public:
    Status Setup(int32_t deviceId, size_t streamCount)
    {
        if (streamCount == 0) { return Status::InvalidParam("H2D stream count must be positive"); }
        Trans::Device device;
        auto status = device.Setup(deviceId);
        if (status.Failure()) { return status; }
        streams_.reserve(streamCount);
        for (size_t i = 0; i < streamCount; ++i) {
            auto stream = device.MakeSharedStream();
            if (!stream) { return Status::Error("failed to create YuanRong H2D stream"); }
            streams_.push_back(std::move(stream));
        }
        deviceId_ = deviceId;
        return Status::OK();
    }

    std::shared_ptr<Trans::Stream> NextStream()
    {
        if (streams_.empty()) { return nullptr; }
        auto stream = streams_[next_];
        next_ = (next_ + 1) % streams_.size();
        return stream;
    }

    Status Synchronize()
    {
        auto result = Status::OK();
        for (auto& stream : streams_) {
            auto status = stream->Synchronized();
            if (status.Failure()) {
                UC_ERROR("Failed({}) to synchronize YuanRong H2D stream on device({}).", status,
                         deviceId_);
                if (result.Success()) { result = status; }
            }
        }
        return result;
    }
};

}  // namespace UC::YuanRongStore

#endif
