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
#include "gdr_config.h"

#include <cstddef>
#include <cstdlib>
#include <mutex>

#include "gdr_mr_buffer.h"
#include "logger/logger.h"

namespace {

std::mutex gGdrNicConfigMutex;
std::vector<std::string> gDeviceNicNames;

std::string ParseStringEnv(const char* name, const char* defaultValue)
{
    const auto* value = std::getenv(name);
    if (!value || value[0] == '\0') { return defaultValue; }
    return value;
}

}  // namespace

namespace UC::Trans {

Status GdrNicConfig::SetDeviceNicNames(const std::vector<std::string>& nicNames)
{
    if (nicNames.empty()) { return Status::OK(); }
    auto status = ValidateDeviceNicNames(nicNames, -1);
    if (status.Failure()) { return status; }

    std::lock_guard<std::mutex> lock{gGdrNicConfigMutex};
    if (gDeviceNicNames.empty()) {
        gDeviceNicNames = nicNames;
        return Status::OK();
    }
    if (gDeviceNicNames == nicNames) { return Status::OK(); }
    return Status::InvalidParam("conflicting GDR NIC name mappings");
}

Status GdrNicConfig::ValidateDeviceNicNames(const std::vector<std::string>& nicNames,
                                            int32_t deviceId)
{
    if (nicNames.empty()) { return Status::OK(); }
    for (size_t i = 0; i < nicNames.size(); ++i) {
        if (nicNames[i].empty()) {
            return Status::InvalidParam("invalid GDR NIC name at index({})", i);
        }
    }
    if (deviceId < 0) { return Status::OK(); }
    if (static_cast<size_t>(deviceId) >= nicNames.size()) {
        return Status::InvalidParam("missing GDR NIC name for device({})", deviceId);
    }
    return Status::OK();
}

Expected<std::string> GdrNicConfig::ResolveNicName(int32_t deviceId)
{
    std::lock_guard<std::mutex> lock{gGdrNicConfigMutex};
    if (gDeviceNicNames.empty()) {
        return std::string{ParseStringEnv("UCM_GDR_NIC_NAME", "mlx5_0")};
    }
    if (deviceId < 0 || static_cast<size_t>(deviceId) >= gDeviceNicNames.size()) {
        return Status::InvalidParam("missing GDR NIC name for device({})", deviceId);
    }
    if (gDeviceNicNames[deviceId].empty()) {
        return Status::InvalidParam("invalid GDR NIC name for device({})", deviceId);
    }
    return std::string{gDeviceNicNames[deviceId]};
}

void GdrNicConfig::ClearForTest()
{
    std::lock_guard<std::mutex> lock{gGdrNicConfigMutex};
    gDeviceNicNames.clear();
}

GdrKVBufferConfig::~GdrKVBufferConfig()
{
    for (auto it = buffers_.rbegin(); it != buffers_.rend(); ++it) {
        GdrMrBuffer::GdrUnregisterDeviceBuffer(reinterpret_cast<void*>(it->addr));
    }
}

Status GdrKVBufferConfig::Validate(const std::vector<uintptr_t>& addrs,
                                   const std::vector<size_t>& sizes)
{
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
}

Status GdrKVBufferConfig::Register(const std::vector<uintptr_t>& addrs,
                                   const std::vector<size_t>& sizes)
{
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
}

}  // namespace UC::Trans
