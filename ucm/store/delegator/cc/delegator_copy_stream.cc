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
#include "delegator_copy_stream.h"
#include <algorithm>
#include "trans/device.h"

namespace UC::Delegator {

CopyStream::~CopyStream() { Reset(); }

Status CopyStream::Setup(std::int32_t deviceId, std::size_t streamNumber)
{
    if (!streams_.empty()) { return Status::Error(); }
    if (deviceId < 0 || streamNumber == 0) {
        return Status::InvalidParam("invalid delegator copy stream config");
    }
    Trans::Device device;
    auto status = device.Setup(deviceId);
    if (status.Failure()) { return status; }

    streams_.reserve(streamNumber);
    for (std::size_t i = 0; i < streamNumber; ++i) {
        auto stream = device.MakeSharedStream();
        if (!stream) [[unlikely]] {
            Reset();
            return Status::Error("failed to create delegator copy stream");
        }
        streams_.push_back(std::move(stream));
    }
    return Status::OK();
}

std::shared_ptr<Trans::Stream> CopyStream::NextStream() noexcept
{
    if (streams_.empty()) { return nullptr; }

    auto stream = streams_[nextStream_];
    nextStream_ = (nextStream_ + 1) % streams_.size();
    return stream;
}

Status CopyStream::DeviceToDeviceAsync(const std::shared_ptr<Trans::Stream>& stream,
                                       void* destination, std::size_t destinationCapacity,
                                       void* source, std::size_t size)
{
    if (stream == nullptr ||
        std::find(streams_.begin(), streams_.end(), stream) == streams_.end() ||
        destination == nullptr || source == nullptr || size == 0 || size > destinationCapacity) {
        return Status::InvalidParam("invalid delegator D2D copy");
    }
    return stream->DeviceToDeviceAsync(source, destination, size);
}

Status CopyStream::Synchronize(const std::shared_ptr<Trans::Stream>& stream)
{
    if (stream == nullptr ||
        std::find(streams_.begin(), streams_.end(), stream) == streams_.end()) {
        return Status::InvalidParam("stream is not owned by CopyStream");
    }

    return stream->Synchronized();
}

Status CopyStream::SynchronizeAll()
{
    auto result = Status::OK();
    for (const auto& stream : streams_) {
        auto status = Synchronize(stream);
        if (result.Success() && status.Failure()) { result = status; }
    }
    return result;
}

void CopyStream::Reset()
{
    for (const auto& stream : streams_) {
        if (stream != nullptr) { (void)stream->Synchronized(); }
    }
    streams_.clear();
    nextStream_ = 0;
}

}  // namespace UC::Delegator
