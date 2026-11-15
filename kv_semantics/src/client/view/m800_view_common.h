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

#include <cstdint>
#include <string>
#include "kv_types.h"

namespace kv {

struct KvClientConfig;

// The deployment configuration uses "worker"; this contract uses Slave to
// match the implementation name M800ViewSlaveServer.
enum class M800ViewRole : std::uint8_t {
    MASTER = 0,
    SLAVE = 1,
};

struct M800ViewServerConfig {
    M800ViewRole role{M800ViewRole::MASTER};

    std::string listenAddr{"0.0.0.0:9100"};
    std::string masterAddr;

    std::string monHost;
    std::string clientName;
    std::string keyringPath;
    std::string clientKey;
    std::string agentLibPath;

    // Timeout and interval values are milliseconds. The Agent C ABI accepts
    // seconds; convert explicitly before setting its view timeout/interval
    // fields.
    std::uint64_t agentFetchTimeoutMs{10000};
    std::uint64_t httpTimeoutMs{10000};
    std::uint64_t refreshIntervalMs{30000};

    // Parses and validates role-specific attributes. The implementation is
    // delivered by the M800ViewServerConfig shard.
    static Status Build(const KvClientConfig& config, M800ViewServerConfig& m800Config);
};

struct M800ViewHTTPContract {
    static constexpr const char* kViewsPath = "/v1/views";
    static constexpr const char* kSinceEpochParam = "since_epoch";
    static constexpr const char* kContentType = "text/plain";
    static constexpr const char* kConnection = "close";
    static constexpr int kStatusOk = 200;
    static constexpr int kStatusNotModified = 304;
    static constexpr int kStatusServerError = 500;
};

// The view body is line-oriented text. Every line ends with '\n'. Values
// cannot contain CR, LF, or NUL in v1; no escaping is defined. The first two
// fields are globalEpoch and nsCount; each namespace block starts at ns and
// must include viewEpoch, viewId, asuIds, and one asuInfo.<nodeId> entry per
// ASU. Pool names are consumed by the master while deriving KvNsId and are
// not part of the client runtime contract.
struct M800ViewTextContract {
    static constexpr const char* kGlobalEpochField = "globalEpoch";
    static constexpr const char* kNsCountField = "nsCount";
    static constexpr const char* kNsField = "ns";
    static constexpr const char* kViewEpochField = "viewEpoch";
    static constexpr const char* kViewIdField = "viewId";
    static constexpr const char* kAsuIdsField = "asuIds";
    static constexpr const char* kAsuInfoFieldPrefix = "asuInfo.";
};

}  // namespace kv
