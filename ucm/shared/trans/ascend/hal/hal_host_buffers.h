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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace UC::Trans {

class HalHostBuffers {
    struct Mapping;
    std::vector<Mapping> mappings_;
    void* base_{nullptr};
    size_t owner_{};
    int32_t deviceId_{-1};
    size_t rankStride_{};

    void LocalSetup(size_t dataBytes, size_t nRanks, int32_t numaNode, uint32_t pgType);
    void Reset();

public:
    HalHostBuffers();
    ~HalHostBuffers();
    HalHostBuffers(const HalHostBuffers&) = delete;
    HalHostBuffers& operator=(const HalHostBuffers&) = delete;

    // Allocates the owner's Host slice; leaves peer slices reserved for Device imports.
    // Throws on failure and releases resources acquired by this call.
    void Setup(int32_t deviceId, size_t dataBytes, size_t nRanks, size_t rank);
    uint64_t ExportHandle();
    void ImportHandle(size_t rank, uint64_t peerHandle);

    size_t Owner() const { return owner_; }
    int32_t DeviceId() const { return deviceId_; }
    size_t RankCount() const;
    // Returns only mapped slices of the requested address kind; otherwise nullptr.
    void* HostData(size_t rank) const;
    void* DeviceData(size_t rank) const;
};

}  // namespace UC::Trans
