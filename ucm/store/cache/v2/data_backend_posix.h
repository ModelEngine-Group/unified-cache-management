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
#include "data_backend.h"

namespace UC::Cache2 {

/* Generic backend: one named POSIX shared-memory segment per rank. Every
 * participant maps all segments, so all slots are host-accessible and no
 * device mapping is provided (DeviceAddrOf is always nullptr; transfers use
 * plain host-memory copies). Once every rank has mapped, the segment names
 * are released (shm_unlink); the objects live on through the mappings. */
class PosixShmDataBackend : public DataBackend {
    struct Segment {
        void* addr{nullptr};
        bool owner{false};
    };

    std::string baseName_{};
    std::vector<Segment> segments_{};
    size_t rankStride_{0};
    size_t ownerRank_{0};
    int32_t deviceId_{-1};
    bool nameUnlinked_{false};

public:
    explicit PosixShmDataBackend(const std::string& uniqueId);
    ~PosixShmDataBackend() override;

    const char* Name() const override { return "posix-shm"; }
    Status Setup(int32_t deviceId, size_t nRanks, size_t rankBytes) override;
    Status BindLocal(size_t rank) override;
    Status ExportLocal(uint64_t* handle) override;
    Status ImportPeer(size_t rank, uint64_t handle) override;
    void FinalizeSetup() override;
    void* HostAddrOf(size_t rank) const override;
    void* DeviceAddrOf(size_t rank) const override;
    void Reset() override;

private:
    std::string SegmentName(size_t rank) const;
    Status MapSegment(size_t rank, bool create, void*& addr);
};

}  // namespace UC::Cache2
