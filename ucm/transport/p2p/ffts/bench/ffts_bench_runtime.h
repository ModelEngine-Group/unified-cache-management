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
#ifndef UNIFIEDCACHE_TRANSPORT_FFTS_BENCH_RUNTIME_H
#define UNIFIEDCACHE_TRANSPORT_FFTS_BENCH_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include <acl/acl.h>

#include "status/status.h"

namespace UC::Transport::Ffts::Bench {

class Samples {
public:
    void Add(double value);
    double Average() const;
    double Min() const;
    double Median() const;

private:
    std::vector<double> values_;
};

class AclSession {
public:
    AclSession() = default;
    ~AclSession();

    AclSession(const AclSession&) = delete;
    AclSession& operator=(const AclSession&) = delete;

    void Init(int32_t deviceId);

private:
    int32_t deviceId_{0};
    bool initialized_{false};
    bool deviceSet_{false};
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* Get() const;
    size_t Size() const;

private:
    void* ptr_{nullptr};
    size_t bytes_{0};
};

class AclStream {
public:
    AclStream();
    ~AclStream();

    AclStream(const AclStream&) = delete;
    AclStream& operator=(const AclStream&) = delete;

    aclrtStream Get() const;

private:
    aclrtStream stream_{nullptr};
};

void CheckAcl(aclError code, const char* call);
void CheckStatus(const UC::Status& status, const char* call);

}  // namespace UC::Transport::Ffts::Bench

#endif  // UNIFIEDCACHE_TRANSPORT_FFTS_BENCH_RUNTIME_H
