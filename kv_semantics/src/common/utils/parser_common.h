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
#include <unordered_map>
#include <vector>
#include "kv_transport.h"

namespace kv {

std::string TrimConfigValue(const std::string& value);
std::vector<std::string> SplitConfigValue(const std::string& value, char delimiter);
std::uint64_t ParseConfigUint64(const std::string& value);
Protocol ParseConfigProtocol(std::string value);
TransProviderType ParseConfigTransProviderType(std::string value);

std::string NormalizeAttrValue(std::string value);
std::string ToLower(std::string value);
bool GetAttr(const std::unordered_map<std::string, std::string>& attrs, const std::string& key,
             std::string& value);
bool GetUint64Attr(const std::unordered_map<std::string, std::string>& attrs, const std::string& key,
                   std::uint64_t& value);
bool GetBoolAttr(const std::unordered_map<std::string, std::string>& attrs, const std::string& key,
                 bool& value);

bool ApplyTransportBufferConfigField(TransportConfig& config, const std::string& key,
                                     const std::string& value);
bool ApplyTransportIoNumConfigField(TransportConfig& config, const std::string& key,
                                    const std::string& value);
bool ApplyTransportCompletionConfigField(TransportConfig& config, const std::string& key,
                                         const std::string& value);
bool ApplyTransportProviderConfigField(TransportConfig& config, const std::string& key,
                                       const std::string& value);
bool ApplyTransportDeviceConfigField(TransportConfig& config, const std::string& key,
                                     const std::string& value);

AsuEndpoint ParseTransportEndpoint(const std::string& value);

}  // namespace kv
