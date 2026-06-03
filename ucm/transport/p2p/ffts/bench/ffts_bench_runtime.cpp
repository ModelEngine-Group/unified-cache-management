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
#include "ffts_bench_runtime.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace UC::Transport::Ffts::Bench {

void Samples::Add(double value) { values_.push_back(value); }

double Samples::Average() const
{
    if (values_.empty()) { return 0.0; }
    return std::accumulate(values_.begin(), values_.end(), 0.0) / static_cast<double>(values_.size());
}

double Samples::Min() const
{
    if (values_.empty()) { return 0.0; }
    return *std::min_element(values_.begin(), values_.end());
}

double Samples::Median() const
{
    if (values_.empty()) { return 0.0; }
    auto sorted = values_;
    std::sort(sorted.begin(), sorted.end());
    const auto mid = sorted.size() / 2;
    if (sorted.size() % 2 == 0) { return (sorted[mid - 1] + sorted[mid]) / 2.0; }
    return sorted[mid];
}

AclSession::~AclSession()
{
    if (deviceSet_) { (void)aclrtResetDevice(deviceId_); }
    if (initialized_) { (void)aclFinalize(); }
}

void AclSession::Init(int32_t deviceId)
{
    CheckAcl(aclInit(nullptr), "aclInit");
    initialized_ = true;

    CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice");
    deviceId_ = deviceId;
    deviceSet_ = true;
}

DeviceBuffer::DeviceBuffer(size_t bytes) : bytes_(bytes)
{
    CheckAcl(aclrtMalloc(&ptr_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
}

DeviceBuffer::~DeviceBuffer()
{
    if (ptr_ != nullptr) { (void)aclrtFree(ptr_); }
}

void* DeviceBuffer::Get() const { return ptr_; }

size_t DeviceBuffer::Size() const { return bytes_; }

AclStream::AclStream()
{
    CheckAcl(aclrtCreateStreamWithConfig(&stream_, 0, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC),
             "aclrtCreateStreamWithConfig");
}

AclStream::~AclStream()
{
    if (stream_ != nullptr) {
        (void)aclrtSynchronizeStream(stream_);
        (void)aclrtDestroyStream(stream_);
    }
}

aclrtStream AclStream::Get() const { return stream_; }

void CheckAcl(aclError code, const char* call)
{
    if (code == ACL_SUCCESS) { return; }
    throw std::runtime_error(std::string(call) + " failed: " + std::to_string(code));
}

void CheckStatus(const UC::Status& status, const char* call)
{
    if (status.Success()) { return; }
    throw std::runtime_error(std::string(call) + " failed: " + status.ToString());
}

}  // namespace UC::Transport::Ffts::Bench
