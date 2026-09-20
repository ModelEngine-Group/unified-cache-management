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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_GC_CONFIG_GUARD_H
#define UNIFIEDCACHE_POSIX_STORE_CC_GC_CONFIG_GUARD_H

#include <atomic>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "gc_liveness.h"
#include "global_config.h"
#include "status/status.h"

namespace UC::PosixStore {

class GcConfigGuard {
public:
    GcConfigGuard() = default;
    GcConfigGuard(const GcConfigGuard&) = delete;
    GcConfigGuard& operator=(const GcConfigGuard&) = delete;
    ~GcConfigGuard();

    Status Setup(const Config& config);

private:
    static std::vector<std::pair<std::string, std::string>> Entries(const Config& config);
    static std::string Serialize(const Config& config);
    static std::map<std::string, std::string> Parse(const std::string& content);
    static std::vector<std::string> Diff(const std::string& mine, const std::string& published);

    Status Join(const std::string& payload);
    Status Publish(const std::string& payload) const;
    Status Verify(const std::string& payload) const;
    Status Compare(const std::string& payload, const std::string& published) const;
    Expected<std::string> ReadPublished() const;
    Status AcquireGate();
    void ReleaseGate();
    Expected<size_t> CountLiveMembers() const;
    Status RegisterMember();
    void UnregisterMember() const;

    std::string backend_;
    std::string configPath_;
    std::string gateDir_;
    std::string membersDir_;
    std::string checkTimePath_;
    std::string identity_;
    std::string memberPath_;
    size_t heartbeatIntervalSec_{0};
    size_t staleThresholdSec_{0};
    bool gateHeld_{false};
    std::atomic<bool> registered_{false};
    GcHeartbeat heartbeat_;
};

}  // namespace UC::PosixStore

#endif
