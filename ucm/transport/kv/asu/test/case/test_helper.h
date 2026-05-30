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
#include <vector>
#include "asu_transport/asu_transport.h"
#include "asu_transport/types.h"

namespace UC::ASU::test {

inline AsuEndpoint MakeEndpoint(const std::string& ip, std::uint16_t port = 9559)
{
    AsuEndpoint ep;
    ep.ip = ip;
    ep.port = port;
    ep.protocol = Protocol::UB;
    return ep;
}

inline std::vector<ConnectionHandle> StubCreateConnection(const AsuEndpoint& endpoint,
                                                          std::uint32_t qp_num)
{
    (void)endpoint;
    return std::vector<ConnectionHandle>(qp_num, nullptr);
}

inline std::vector<Status> StubDeleteConnections(const std::vector<ConnectionHandle>& handles)
{
    return std::vector<Status>(handles.size(), Status::OK());
}

inline std::vector<KVBuffer> MakeKVEntries(std::size_t count)
{
    std::vector<std::uint8_t> payload(64, 0xAB);
    MemoryRegion region;
    region.memoryType = MemoryType::HOST;
    region.addr = reinterpret_cast<std::uint64_t>(payload.data());
    region.size = payload.size();

    Buffer buffer;
    buffer.region = region;

    std::vector<KVBuffer> entries;
    for (std::size_t i = 0; i < count; ++i) {
        entries.push_back(KVBuffer{"key_" + std::to_string(i), buffer});
    }
    return entries;
}

inline std::vector<CacheKey> MakeKeys(std::size_t count)
{
    std::vector<CacheKey> keys;
    for (std::size_t i = 0; i < count; ++i) { keys.push_back("key_" + std::to_string(i)); }
    return keys;
}

}  // namespace UC::ASU::test
