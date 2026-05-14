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
#include "gdr_config.h"
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>
#include "gdr_mr_buffer.h"
#include "logger/logger.h"

namespace {

constexpr const char* kDeviceGdrNicsEnv = "DEVICE_GDR_NICS";
constexpr const char* kDefaultNicName = "mlx5_0";

std::string Trim(std::string value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) { return ""; }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::vector<std::string> ParseNicList(const char* value)
{
    std::vector<std::string> nics;
    std::string text{value};
    size_t pos = 0;
    while (pos <= text.size()) {
        const auto next = text.find(',', pos);
        const auto len = next == std::string::npos ? std::string::npos : next - pos;
        nics.push_back(Trim(text.substr(pos, len)));
        if (next == std::string::npos) { break; }
        pos = next + 1;
    }
    return nics;
}

UC::Expected<std::string> ResolveFromNicList(const std::vector<std::string>& nicNames,
                                             int32_t deviceId)
{
    for (size_t i = 0; i < nicNames.size(); ++i) {
        if (nicNames[i].empty()) {
            return UC::Status::InvalidParam("invalid GDR NIC name at index({})", i);
        }
    }
    if (deviceId < 0 || static_cast<size_t>(deviceId) >= nicNames.size()) {
        return UC::Status::InvalidParam("missing GDR NIC name for device({})", deviceId);
    }
    return std::string{nicNames[deviceId]};
}

}  // namespace

namespace UC::Trans {

Expected<std::string> GdrNicConfig::ResolveNicName(int32_t deviceId)
{
    const auto* deviceGdrNics = std::getenv(kDeviceGdrNicsEnv);
    if (deviceGdrNics && deviceGdrNics[0] != '\0') {
        auto nicNames = ParseNicList(deviceGdrNics);
        return ResolveFromNicList(nicNames, deviceId);
    }
    return std::string{kDefaultNicName};
}

GdrKVBufferConfig::~GdrKVBufferConfig()
{
#if UCM_ENABLE_GDR_STREAM
    for (auto it = buffers_.rbegin(); it != buffers_.rend(); ++it) {
        GdrMrBuffer::GdrUnregisterDeviceBuffer(reinterpret_cast<void*>(it->addr));
    }
#endif
}

Status GdrKVBufferConfig::Validate(const std::vector<uintptr_t>& addrs,
                                   const std::vector<size_t>& sizes)
{
#if UCM_ENABLE_GDR_STREAM
    if (addrs.size() != sizes.size()) {
        return Status::InvalidParam("mismatched GPU KV buffer ranges({},{})", addrs.size(),
                                    sizes.size());
    }
    for (size_t i = 0; i < addrs.size(); ++i) {
        if (addrs[i] == 0) {
            return Status::InvalidParam("invalid GPU KV buffer address at index({})", i);
        }
        if (sizes[i] == 0) {
            return Status::InvalidParam("invalid GPU KV buffer size at index({})", i);
        }
    }
    return Status::OK();
#else
    (void)addrs;
    (void)sizes;
    return Status::OK();
#endif
}

Status GdrKVBufferConfig::Register(const std::vector<uintptr_t>& addrs,
                                   const std::vector<size_t>& sizes)
{
#if UCM_ENABLE_GDR_STREAM
    auto status = Validate(addrs, sizes);
    if (status.Failure()) { return status; }
    for (size_t i = 0; i < addrs.size(); ++i) {
        auto s = GdrMrBuffer::GdrRegisterDeviceBuffer(reinterpret_cast<void*>(addrs[i]), sizes[i]);
        if (s.Success()) {
            buffers_.push_back({static_cast<uint64_t>(addrs[i]), sizes[i]});
            continue;
        }
        UC_WARN("Failed({}) to pre-register GPU KV buffer at addr(0x{:x}) with size({}).", s,
                addrs[i], sizes[i]);
    }
    return Status::OK();
#else
    (void)addrs;
    (void)sizes;
    return Status::OK();
#endif
}

}  // namespace UC::Trans
