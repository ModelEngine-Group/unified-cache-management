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

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "data_backend.h"
#include "ipc/fd_socket.h"
#include "ipc/mem_fd.h"

namespace UC::Cache2 {

/* Generic backend for platforms without a vendor host-memory path
 * (ascend/ascend-a3/cuda/simu): one memfd segment per rank, imported over an
 * abstract Unix socket with SCM_RIGHTS. Every participant maps all segments, so
 * all slots are host-accessible and DeviceAddrOf stays nullptr. Every segment is
 * registered with the device so the device can reach it on any copy path; the
 * device-visible alias is fetched only when the transfer mode needs it (Memfd
 * SDMA Direct) and exposed through HostMappedDeviceAddrOf -- the alias is for
 * descriptors only, never for CPU access. */
class MemfdDataBackend : public DataBackend {
    struct Segment {
        std::unique_ptr<MemFd> mem{}; /* host mapping and fd; null until mapped */
        void* deviceAddr{nullptr};    /* device-visible alias, nullptr unless required */
        bool owner{false};
        bool registered{false};
    };

    std::string domainId_{};
    std::vector<Segment> segments_{};
    size_t rankStride_{0};
    size_t ownerRank_{0};
    int32_t deviceId_{-1};
    bool requireHostDeviceAddress_{false};
    Numa::Plan localPlacement_{};
    std::chrono::steady_clock::time_point deadline_{};
    FdSocket fdServer_{};
    std::thread fdThread_{};

public:
    explicit MemfdDataBackend(const std::string& domainId);
    ~MemfdDataBackend() override;

    const char* Name() const override { return "memfd"; }
    Status Setup(const BackendOptions& options) override;
    Status BindLocal(size_t rank) override;
    Status ExportLocal(uint64_t* handle) override;
    Status ImportPeer(size_t rank, uint64_t handle) override;
    void FinalizeSetup() override;
    void* HostAddrOf(size_t rank) const override;
    void* DeviceAddrOf(size_t rank) const override;
    void* HostMappedDeviceAddrOf(size_t rank) const override;
    void Reset() override;

private:
    /* memfd_create -> ftruncate -> mmap, without MAP_POPULATE. */
    Status CreateLocalSegment(size_t rank);
    /* First touch, then device registration. NUMA binding runs between them once
     * Numa::BindBeforeTouch is wired: registering pins pages, so it must stay
     * last. */
    Status PlaceAndRegister(size_t rank);
    Status RegisterSegment(Segment& segment);
    /* memfd name, and the abstract socket a peer uses to fetch the fd. */
    std::string SegmentName(size_t rank) const;
    std::string SocketName(size_t rank) const;
    Status StartFdServer();
    void AcceptLoop(size_t peerCount);
    void StopFdServer();
};

}  // namespace UC::Cache2
