/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include "m800_view_common.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>

#include "logger.h"
#include "parser_common.h"

namespace kv {
namespace {

constexpr std::uint32_t kXxh32Prime1 = 0x9E3779B1U;
constexpr std::uint32_t kXxh32Prime2 = 0x85EBCA77U;
constexpr std::uint32_t kXxh32Prime3 = 0xC2B2AE3DU;
constexpr std::uint32_t kXxh32Prime4 = 0x27D4EB2FU;
constexpr std::uint32_t kXxh32Prime5 = 0x165667B1U;

std::uint32_t RotateLeft32(std::uint32_t value, unsigned int count)
{
    return (value << count) | (value >> (32U - count));
}

std::uint32_t Read32LittleEndian(const std::uint8_t* data)
{
    return static_cast<std::uint32_t>(data[0]) |
           (static_cast<std::uint32_t>(data[1]) << 8U) |
           (static_cast<std::uint32_t>(data[2]) << 16U) |
           (static_cast<std::uint32_t>(data[3]) << 24U);
}

std::uint32_t Xxh32Round(std::uint32_t accumulator, std::uint32_t input)
{
    accumulator += input * kXxh32Prime2;
    accumulator = RotateLeft32(accumulator, 13);
    return accumulator * kXxh32Prime1;
}

std::uint32_t Xxh32Raw(const std::string& value, std::uint32_t seed)
{
    const auto* data = reinterpret_cast<const std::uint8_t*>(value.data());
    const std::size_t length = value.size();
    std::size_t index = 0;
    std::uint32_t hash = 0;

    if (length >= 16) {
        std::uint32_t v1 = seed + kXxh32Prime1 + kXxh32Prime2;
        std::uint32_t v2 = seed + kXxh32Prime2;
        std::uint32_t v3 = seed;
        std::uint32_t v4 = seed - kXxh32Prime1;
        while (length - index >= 16) {
            v1 = Xxh32Round(v1, Read32LittleEndian(data + index));
            index += 4;
            v2 = Xxh32Round(v2, Read32LittleEndian(data + index));
            index += 4;
            v3 = Xxh32Round(v3, Read32LittleEndian(data + index));
            index += 4;
            v4 = Xxh32Round(v4, Read32LittleEndian(data + index));
            index += 4;
        }
        hash = RotateLeft32(v1, 1) + RotateLeft32(v2, 7) + RotateLeft32(v3, 12) +
               RotateLeft32(v4, 18);
    } else {
        hash = seed + kXxh32Prime5;
    }

    hash += static_cast<std::uint32_t>(length);
    while (length - index >= 4) {
        hash += Read32LittleEndian(data + index) * kXxh32Prime3;
        hash = RotateLeft32(hash, 17) * kXxh32Prime4;
        index += 4;
    }
    while (index < length) {
        hash += static_cast<std::uint32_t>(data[index]) * kXxh32Prime5;
        hash = RotateLeft32(hash, 11) * kXxh32Prime1;
        ++index;
    }

    hash ^= hash >> 15U;
    hash *= kXxh32Prime2;
    hash ^= hash >> 13U;
    hash *= kXxh32Prime3;
    hash ^= hash >> 16U;
    return hash;
}

std::string ReadStringAttr(const std::unordered_map<std::string, std::string>& attrs,
                           const std::string& key, const std::string& defaultValue)
{
    std::string value;
    if (!GetAttr(attrs, key, value)) { return defaultValue; }
    return TrimConfigValue(value);
}

std::uint64_t ParseUint64OrDefault(const std::string& value, std::uint64_t defaultValue)
{
    const auto normalized = TrimConfigValue(value);
    if (normalized.find('\0') != std::string::npos) { return defaultValue; }
    // strtoull/stoull accepts a leading minus sign and wraps it into the
    // unsigned result. Configuration values treat every negative number as
    // invalid rather than as a very large timeout or interval.
    if (!normalized.empty() && normalized.front() == '-') { return defaultValue; }
    return ParseConfigUint64(normalized, defaultValue);
}

std::uint64_t ParseTimeoutOrDefault(const std::string& value, std::uint64_t defaultValue)
{
    const auto parsed = ParseUint64OrDefault(value, defaultValue);
    constexpr std::uint64_t kMaxSupportedTimeoutMs =
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
    return parsed > kMaxSupportedTimeoutMs ? defaultValue : parsed;
}

Protocol ParseEndpointProtocolOrDefault(const std::string& value, Protocol defaultValue)
{
    const auto normalized = ToLower(TrimConfigValue(value));
    if (normalized == "ub" || normalized == "uboe" || normalized == "ubg") {
        return Protocol::UB;
    }
    if (normalized == "roce") { return Protocol::ROCE; }
    if (normalized == "tcp") { return Protocol::TCP; }
    return defaultValue;
}

std::string ReadKeyring(const std::string& keyringPath)
{
    std::ifstream keyringFile{keyringPath};
    if (!keyringFile.is_open()) {
        KV_WARN("Failed to open M800 keyring; clientKey will be empty, path={}", keyringPath);
        return {};
    }

    std::string line;
    while (std::getline(keyringFile, line)) {
        line = TrimConfigValue(line);
        if (line.empty() || line.front() == '#' || line.front() == ';' || line.front() == '[') {
            continue;
        }
        const auto separator = line.find('=');
        if (separator == std::string::npos) { continue; }
        if (TrimConfigValue(line.substr(0, separator)) != "key") { continue; }
        return TrimConfigValue(line.substr(separator + 1));
    }

    KV_WARN("M800 keyring contains no key; clientKey will be empty, path={}", keyringPath);
    return {};
}

Status InvalidConfig(const std::string& message)
{
    return Status::Error(StatusCode::INVALID_ARGUMENT, message);
}

}  // namespace

KvNsId M800PoolNameToKvNsId(const std::string& poolName)
{
    return Xxh32Raw(poolName, kKvNsIdHashSeed);
}

Status M800ViewServerConfig::Build(const KvClientConfig& config,
                                   M800ViewServerConfig& m800Config)
{
    m800Config = M800ViewServerConfig{};
    const auto& attrs = config.attrs;

    std::string roleValue;
    if (GetAttr(attrs, "view.role", roleValue)) {
        roleValue = ToLower(TrimConfigValue(roleValue));
        if (roleValue == "master") {
            m800Config.role = M800ViewRole::MASTER;
        } else if (roleValue == "worker") {
            m800Config.role = M800ViewRole::SLAVE;
        } else {
            return InvalidConfig("view.role must be master or worker, value=" + roleValue);
        }
    }

    m800Config.listenAddr = ReadStringAttr(attrs, "view.listen_addr", m800Config.listenAddr);
    m800Config.monHost = ReadStringAttr(attrs, "view.ceph.mon_host", m800Config.monHost);
    m800Config.clientName =
        ReadStringAttr(attrs, "view.ceph.client_name", m800Config.clientName);
    m800Config.keyringPath =
        ReadStringAttr(attrs, "view.ceph.keyring", m800Config.keyringPath);
    m800Config.agentLibPath =
        ReadStringAttr(attrs, "view.ceph.agent_lib", m800Config.agentLibPath);

    const auto masterAddrs = ReadStringAttr(attrs, "view.master_addrs", "");
    m800Config.masterAddrs = SplitConfigValue(masterAddrs, ',');
    if (!m800Config.masterAddrs.empty()) {
        m800Config.masterAddr = m800Config.masterAddrs.front();
    }

    // Prefer correctly spelled keys when both forms are present, while keeping
    // compatibility with the names already deployed by the M800 view prompt.
    const auto refreshInterval = TrimConfigValue(GetAttr(
        attrs, {"view.refresh_interval_ms", "view.refresh_in_terval_ms"}));
    m800Config.refreshIntervalMs =
        ParseTimeoutOrDefault(refreshInterval, m800Config.refreshIntervalMs);
    const auto agentFetchTimeout = TrimConfigValue(GetAttr(
        attrs, {"view.agent_fetch_timeout_ms", "view.agent_fetch_timeoout_ms"}));
    m800Config.agentFetchTimeoutMs =
        ParseTimeoutOrDefault(agentFetchTimeout, m800Config.agentFetchTimeoutMs);
    m800Config.httpTimeoutMs = ParseTimeoutOrDefault(
        ReadStringAttr(attrs, "view.http_timeout_ms", ""), m800Config.httpTimeoutMs);

    m800Config.endpointProtocol = ParseEndpointProtocolOrDefault(
        ReadStringAttr(attrs, "view.endpoint_protocol", ""), m800Config.endpointProtocol);
    constexpr std::uint16_t kDefaultEndpointPort = 6000;
    const auto endpointPort = ParseUint64OrDefault(
        ReadStringAttr(attrs, "view.endpoint_port", ""), kDefaultEndpointPort);
    m800Config.endpointPort = endpointPort > std::numeric_limits<std::uint16_t>::max()
                                  ? kDefaultEndpointPort
                                  : static_cast<std::uint16_t>(endpointPort);

    const auto filter = ToLower(ReadStringAttr(attrs, "view.ceph.filter", "in"));
    if (filter != "in") {
        KV_WARN("Unsupported M800 view filter; continuing with in-only filtering, value={}",
                filter);
    }

    if (m800Config.role == M800ViewRole::SLAVE) {
        if (m800Config.masterAddrs.empty()) {
            return InvalidConfig("view.master_addrs is required for worker role");
        }
        return Status::OK();
    }

    if (m800Config.monHost.empty()) {
        return InvalidConfig("view.ceph.mon_host is required for master role");
    }
    m800Config.clientKey = ReadKeyring(m800Config.keyringPath);
    return Status::OK();
}

}  // namespace kv
