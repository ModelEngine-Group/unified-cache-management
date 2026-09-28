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
#include "status/status.h"

namespace UC::Cache2 {

/* One shared segment per rank. A backend creates the local rank's segment,
 * exports it as a handle published through CtrlLayout::RankDataDesc, and
 * imports every peer segment. A rank's host and device addresses are mutually
 * exclusive: callers pick via DataStrategy::HostAccessibleOf and never query
 * the device address of a host-accessible slot (or vice versa). */
class DataBackend {
public:
    virtual ~DataBackend() = default;
    DataBackend(const DataBackend&) = delete;
    DataBackend& operator=(const DataBackend&) = delete;

    virtual const char* Name() const = 0;
    /* Prepares backend-wide state (device binding, geometry bookkeeping). */
    virtual Status Setup(int32_t deviceId, size_t nRanks, size_t rankBytes) = 0;
    /* Creates and maps the local rank's segment. */
    virtual Status BindLocal(size_t rank) = 0;
    /* Returns the handle to publish for the local segment. */
    virtual Status ExportLocal(uint64_t* handle) = 0;
    /* Maps a peer segment previously published by its owner. */
    virtual Status ImportPeer(size_t rank, uint64_t handle) = 0;
    /* Called once after every rank finished importing; backends release
     * name-based resources here (e.g. unlink the local segment name). */
    virtual void FinalizeSetup() = 0;
    /* Returns the CPU/IO-accessible segment address, nullptr if inaccessible. */
    virtual void* HostAddrOf(size_t rank) const = 0;
    /* Returns the device-accessible segment address, nullptr if unmapped. */
    virtual void* DeviceAddrOf(size_t rank) const = 0;
    /* Releases every mapping and the local segment; idempotent. */
    virtual void Reset() = 0;

protected:
    DataBackend() = default;
};

}  // namespace UC::Cache2
