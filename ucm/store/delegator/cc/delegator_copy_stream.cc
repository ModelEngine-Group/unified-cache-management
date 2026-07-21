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
#include <string>

namespace UC::Delegator {
namespace {

Status AclStatus(const char* operation, aclError error)
{
    return Status::Error(std::string(operation) + " failed: " + std::to_string(error));
}

}  // namespace

CopyStream::~CopyStream() { Reset(); }

Status CopyStream::Setup(std::int32_t device_id, std::size_t stream_number)
{
    if (!streams_.empty()) { return Status::DuplicateKey(); }
    if (device_id < 0 || stream_number == 0) {
        return Status::InvalidParam("invalid delegator copy stream config");
    }

    device_id_ = device_id;
    auto status = BindDevice();
    if (status.Failure()) { return status; }

    streams_.resize(stream_number, nullptr);
    for (auto& stream : streams_) {
        auto ret = aclrtCreateStreamWithConfig(&stream, 0,
                                          ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC);
        if (ret != ACL_SUCCESS) [[unlikely]] {
            Reset();
            return AclStatus("aclrtCreateStreamWithConfig", ret);
        }
    }
    return Status::OK();
}

Status CopyStream::BindDevice() const
{
    const auto ret = aclrtSetDevice(device_id_);
    return ret == ACL_SUCCESS ? Status::OK() : AclStatus("aclrtSetDevice", ret);
}

aclrtStream CopyStream::NextStream() noexcept
{
    if (streams_.empty()) { return nullptr; }

    auto stream = streams_[next_stream_];
    next_stream_ = (next_stream_ + 1) % streams_.size();
    return stream;
}

Status CopyStream::DeviceToDeviceAsync(aclrtStream stream, void* destination,
                                       std::size_t destination_capacity,
                                       const void* source, std::size_t size)
{
    if (stream == nullptr || destination == nullptr || source == nullptr || size == 0 ||
        size > destination_capacity) {
        return Status::InvalidParam("invalid delegator D2D copy");
    }
    const auto ret = aclrtMemcpyAsync(destination, destination_capacity, source, size,
                                      ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    return ret == ACL_SUCCESS ? Status::OK() : AclStatus("aclrtMemcpyAsync", ret);
}

Status CopyStream::Synchronize()
{
    auto result = Status::OK();
    for (const auto stream : streams_) {
        const auto ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            result = AclStatus("aclrtSynchronizeStream", ret);
        }
    }
    return result;
}

void CopyStream::Reset()
{
    if (device_id_ >= 0) { (void)BindDevice(); }
    for (const auto stream : streams_) {
        if (stream != nullptr) { (void)aclrtDestroyStream(stream); }
    }
    streams_.clear();
    device_id_ = -1;
    next_stream_ = 0;
}

}  // namespace UC::Delegator
