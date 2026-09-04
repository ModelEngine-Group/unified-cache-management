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
#include <string>
#include "trans_provider.h"
#include "types.h"
#include "index_pool.h"

namespace UC::ASU {

struct ScatterGatherEntry {
    // Local-side address. CPU-accessible for HOST/HOST_PINNED and a local
    // device address for DEVICE.
    std::uint64_t local_addr{0};
    // Device-visible address used by HCOMM/HIXL and remote RDMA operations.
    std::uint64_t device_addr{0};
    std::uint32_t length{0};
    std::uint32_t tokenId{0};
    std::uint32_t slot_index{UINT32_MAX};
    MemoryType memory_type{MemoryType::HOST};
};

class BufferManager {
public:
    BufferManager() = default;
    ~BufferManager();

    BufferManager(const BufferManager&) = delete;
    BufferManager& operator=(const BufferManager&) = delete;

    Status Init(std::string name, MemoryType type, std::size_t slot_capacity, std::size_t slot_num);
    void Shutdown();

    Status Allocate(std::size_t size, ScatterGatherEntry& sge);
    Status Free(std::uint32_t slot_index);

    bool IsValidPointer(const void* ptr) const;

    std::uint32_t GetTokenId() const { return tokenId_; }
    void SetTokenId(std::uint32_t tokenId) { tokenId_ = tokenId; }
    Status GetRegisterMemoryDesc(TransProvider::RegisterMemoryDesc& desc) const;

private:
    struct BufferRegion {
        static Status Create(MemoryType type, std::size_t size, BufferRegion& region);

        explicit operator bool() const { return owner != nullptr; }
        void Reset();

        std::shared_ptr<void> owner;
        void* localAddr{nullptr};
        void* deviceAddr{nullptr};
        TransProvider::MemType providerMemType{TransProvider::MemType::MEM_HOST};
    };

    std::string name_;
    std::size_t slot_capacity_{0};
    std::size_t slot_stride_{0};
    std::size_t slot_num_{0};
    MemoryType memory_type_{MemoryType::HOST};

    BufferRegion region_;
    IndexPool index_pool_;

    std::uint32_t tokenId_{0};
};

}  // namespace UC::ASU
