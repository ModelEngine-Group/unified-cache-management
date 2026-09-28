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

#include "data_backend.h"

#if UCM_RUNTIME_ASCEND_HAL
#include <vector>
#include "trans/ascend/hal/hal_memory.h"

namespace UC::Cache2 {

/* Ascend-a5 backend: host DDR physical allocations exported as driver share
 * handles and mapped into one reserved VA region, rank-strided. Only the local
 * rank's segment is host-accessible; peer segments are device-mapped only. */
class AscendHalDataBackend : public DataBackend {
    struct Mapping {
        Trans::Hal::MemHandle handle{nullptr};
        bool mapped{false};
    };

    std::vector<Mapping> mappings_{};
    void* base_{nullptr};
    size_t owner_{};
    int32_t deviceId_{-1};
    size_t nRanks_{0};
    size_t rankStride_{0};
    size_t rankBytes_{0};

public:
    AscendHalDataBackend() = default;
    ~AscendHalDataBackend() override;

    const char* Name() const override { return "ascend-hal"; }
    Status Setup(int32_t deviceId, size_t nRanks, size_t rankBytes) override;
    Status BindLocal(size_t rank) override;
    Status ExportLocal(uint64_t* handle) override;
    Status ImportPeer(size_t rank, uint64_t handle) override;
    void FinalizeSetup() override;
    void* HostAddrOf(size_t rank) const override;
    void* DeviceAddrOf(size_t rank) const override;
    void Reset() override;

private:
    Status Allocate(Trans::Hal::PageType pageType);
    std::byte* RankAddr(size_t rank) const;
};

}  // namespace UC::Cache2
#endif  // UCM_RUNTIME_ASCEND_HAL
