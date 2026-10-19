/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_GC_LIVENESS_H
#define UNIFIEDCACHE_POSIX_STORE_CC_GC_LIVENESS_H

#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "status/status.h"

namespace UC::PosixStore {

namespace GcClock {

Status Touch(const std::string& path, time_t& stamp, bool create);
std::string LocalHostName();
uint32_t Nonce();

}  // namespace GcClock

struct GcMemberInfo {
    std::string name;
    size_t lag{0};
};

Expected<std::vector<GcMemberInfo>> ScanMembers(const std::string& dir, const std::string& prefix,
                                                const std::string& checkTimePath);

class GcHeartbeat {
public:
    enum class OnMissing { Recreate, Stop };

    GcHeartbeat() = default;
    GcHeartbeat(const GcHeartbeat&) = delete;
    GcHeartbeat& operator=(const GcHeartbeat&) = delete;
    ~GcHeartbeat();

    void Setup(std::string path, std::string threadName, size_t intervalSec, OnMissing onMissing);
    Status Start();
    void RequestStop();
    void Stop();

private:
    void Loop();

    std::string path_;
    std::string threadName_;
    size_t intervalSec_{5};
    OnMissing onMissing_{OnMissing::Stop};
    std::thread worker_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stop_{false};
};

}  // namespace UC::PosixStore

#endif
