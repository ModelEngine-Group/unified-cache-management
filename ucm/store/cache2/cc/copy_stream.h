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
#include <vector>
#include "logger/logger.h"
#include "status/status.h"
#include "trans/device.h"

namespace UC::Cache2 {

class CopyStream {
    int32_t deviceId_{-1};
    size_t streamNumber_{0};
    size_t streamIndex_{0};
    std::vector<std::shared_ptr<Trans::Stream>> streams_;

public:
    Status Setup(const int32_t deviceId, const size_t streamNumber)
    {
        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
            return s;
        }
        streams_.clear();
        streams_.reserve(streamNumber);
        for (size_t i = 0; i < streamNumber; ++i) {
            auto stream = device.MakeSharedStream();
            if (!stream) {
                UC_ERROR("Failed to make stream on device({}).", deviceId);
                return Status::Error();
            }
            streams_.push_back(std::move(stream));
        }
        deviceId_ = deviceId;
        streamNumber_ = streamNumber;
        streamIndex_ = 0;
        return Status::OK();
    }

    Status DeviceToDeviceScatterAsync(void* src, void* dst[],
                                      const std::vector<size_t>& sizes) noexcept
    {
        auto stream = NextStream();
        size_t offset = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (sizes[i] != 0 && dst[i] != nullptr) {
                auto* pSrc = static_cast<void*>(static_cast<int8_t*>(src) + offset);
                auto s = stream->DeviceToDeviceAsync(pSrc, dst[i], sizes[i]);
                if (s.Failure()) {
                    auto syncS = stream->Synchronized();
                    if (syncS.Failure()) {
                        UC_ERROR("Failed({}) to synchronize stream on device({}).", syncS,
                                 deviceId_);
                    }
                    return s;
                }
            }
            offset += sizes[i];
        }
        return Status::OK();
    }

    Status DeviceToDeviceGatherAsync(void* src[], void* dst,
                                     const std::vector<size_t>& sizes) noexcept
    {
        auto stream = NextStream();
        size_t offset = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (sizes[i] != 0 && src[i] != nullptr) {
                auto* pDst = static_cast<void*>(static_cast<int8_t*>(dst) + offset);
                auto s = stream->DeviceToDeviceAsync(src[i], pDst, sizes[i]);
                if (s.Failure()) {
                    auto syncS = stream->Synchronized();
                    if (syncS.Failure()) {
                        UC_ERROR("Failed({}) to synchronize stream on device({}).", syncS,
                                 deviceId_);
                    }
                    return s;
                }
            }
            offset += sizes[i];
        }
        return Status::OK();
    }

    Status HostToDeviceScatterAsync(void* src, void* dst[],
                                    const std::vector<size_t>& sizes) noexcept
    {
        auto stream = NextStream();
        size_t offset = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (sizes[i] != 0 && dst[i] != nullptr) {
                auto* pSrc = static_cast<void*>(static_cast<int8_t*>(src) + offset);
                auto s = stream->HostToDeviceAsync(pSrc, dst[i], sizes[i]);
                if (s.Failure()) {
                    auto syncS = stream->Synchronized();
                    if (syncS.Failure()) {
                        UC_ERROR("Failed({}) to synchronize stream on device({}).", syncS,
                                 deviceId_);
                    }
                    return s;
                }
            }
            offset += sizes[i];
        }
        return Status::OK();
    }

    Status DeviceToHostGatherAsync(void* src[], void* dst,
                                   const std::vector<size_t>& sizes) noexcept
    {
        auto stream = NextStream();
        size_t offset = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (sizes[i] != 0 && src[i] != nullptr) {
                auto* pDst = static_cast<void*>(static_cast<int8_t*>(dst) + offset);
                auto s = stream->DeviceToHostAsync(src[i], pDst, sizes[i]);
                if (s.Failure()) {
                    auto syncS = stream->Synchronized();
                    if (syncS.Failure()) {
                        UC_ERROR("Failed({}) to synchronize stream on device({}).", syncS,
                                 deviceId_);
                    }
                    return s;
                }
            }
            offset += sizes[i];
        }
        return Status::OK();
    }

    Status WaitEvent(const Trans::Event& event) noexcept
    {
        auto status = Status::OK();
        for (auto& stream : streams_) {
            auto s = stream->WaitEvent(event);
            if (s.Success()) { continue; }
            UC_ERROR("Failed({}) to wait event on stream on device({}).", s, deviceId_);
            if (status.Success()) { status = s; }
        }
        return status;
    }

    Status Synchronize() noexcept
    {
        auto status = Status::OK();
        for (auto& stream : streams_) {
            auto s = stream->Synchronized();
            if (s.Success()) { continue; }
            UC_ERROR("Failed({}) to synchronize stream on device({}).", s, deviceId_);
            if (status.Success()) { status = s; }
        }
        return status;
    }

private:
    std::shared_ptr<Trans::Stream> NextStream() noexcept
    {
        auto& stream = streams_[streamIndex_];
        streamIndex_ = (streamIndex_ + 1) % streamNumber_;
        return stream;
    }
};

}  // namespace UC::Cache2
