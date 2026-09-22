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
#include <memory>
#include "ctrl_layout.h"

namespace UC::Trans {
class HalHostBuffers;
}

namespace UC::Cache2 {

class DataStrategy {
    size_t slotSize_{};
    size_t nSlotsPerRank_{};
#if UCM_RUNTIME_ASCEND_HAL
    std::unique_ptr<Trans::HalHostBuffers> hostBuffers_;
#endif

    void LocalSetup(int32_t deviceId, size_t dataBytes, size_t nRanks, size_t rank);
    void CrossRankSetup(CtrlLayout& ctrl, size_t timeoutMs);

public:
    DataStrategy();
    ~DataStrategy();
    DataStrategy(const DataStrategy&) = delete;
    DataStrategy& operator=(const DataStrategy&) = delete;

    // Throws on failure after releasing resources acquired by this call.
    // All peer handles share timeoutMs; zero allows one read attempt per peer.
    void Setup(CtrlLayout& ctrl, int32_t deviceId, size_t slotSize, size_t nSlotsPerRank,
               size_t rank, size_t timeoutMs = 1800 * 1000);

    // Only locally allocated slots have a CPU/IO-accessible address.
    void* DataAt(size_t slotIdx);
    // Only peer slots have a Device mapping; local slots use DataAt with H2D/D2H.
    void* DeviceDataAt(size_t slotIdx);
};

}  // namespace UC::Cache2
