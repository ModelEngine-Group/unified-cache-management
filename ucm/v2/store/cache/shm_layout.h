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
#include <string>
#include <vector>
#include "status/status.h"

namespace UC::Store {

/**
 * @brief Total byte size of the shared pool: capacity x blockSize bytes,
 *        rounded up to a whole huge page.
 *
 * @return Expected the size on success; InvalidParam when the geometry is
 *         degenerate (zero capacity or block size) or overflows the
 *         addressable range.
 */
[[nodiscard]] Expected<std::size_t> ShmTotalSize(std::uint32_t capacity, std::uint64_t blockSize);

/**
 * @brief Address of one slot inside a mapped pool base.
 *
 * The address may cross huge-page boundaries when blockSize is not a multiple
 * of the page size; the mapping is one contiguous region, so crossing is
 * harmless. Precondition: slot < capacity of the pool being addressed.
 */
[[nodiscard]] void* SlotAddress(void* shmBase, std::uint32_t slot, std::uint64_t blockSize);

/**
 * @brief Payload of an AddressedToken minted over the shared pool: a bare
 *        little-endian u32 slot array. Pure translation with no policy --
 *        the recipient resolves slots through its own geometry, and slot
 *        validity is the minter's contract (see SlotAddress's precondition).
 */
[[nodiscard]] std::vector<std::byte> EncodeTokenPayload(const std::vector<std::uint32_t>& slots);

/**
 * @brief Decodes a token payload: slot count is the payload byte length
 *        divided by four, rounded down; one to three trailing bytes are
 *        ignored. No semantic checks -- a payload that did not come from
 *        EncodeTokenPayload is out of contract.
 */
[[nodiscard]] std::vector<std::uint32_t> DecodeTokenPayload(const std::vector<std::byte>& data);

/**
 * @brief Owns one mmap over a shared memory object. Move-only; destruction
 *        munmaps. The descriptor used for attaching is not kept: a mapping
 *        stays valid after its fd is closed.
 */
class ShmMapping {
public:
    ShmMapping() noexcept = default;
    ~ShmMapping();
    ShmMapping(ShmMapping&& other) noexcept;
    ShmMapping& operator=(ShmMapping&& other) noexcept;
    ShmMapping(const ShmMapping&) = delete;
    ShmMapping& operator=(const ShmMapping&) = delete;

    /**
     * @brief fstats the descriptor and maps it shared read-write.
     *
     * @param fd            Open descriptor of a regular shared memory object.
     * @param expectedSize  Byte length the object must report; a mismatch
     *                      means the peer runs a different geometry and the
     *                      attach is refused before any mapping happens.
     *
     * @return Expected the mapping on success; InvalidParam on a size
     *         mismatch, OsApiError when fstat or mmap fails.
     */
    [[nodiscard]] static Expected<ShmMapping> Attach(int fd, std::size_t expectedSize);

    /** @brief Mapped base address, nullptr when not attached. */
    [[nodiscard]] void* Base() const noexcept { return base_; }

    /** @brief Mapped byte length, 0 when not attached. */
    [[nodiscard]] std::size_t Size() const noexcept { return size_; }

    /** @brief Tells whether a mapping is held. */
    [[nodiscard]] bool IsValid() const noexcept { return base_ != nullptr; }

    /** @brief Releases the mapping; idempotent. */
    void Reset() noexcept;

private:
    ShmMapping(void* base, std::size_t size) noexcept : base_{base}, size_{size} {}

    void* base_{nullptr};
    std::size_t size_{0};
};

/**
 * @brief A created shared pool: the memfd to hand to peers over SCM_RIGHTS,
 *        the locally attached mapping, and whether huge pages back it.
 *        Move-only; destruction closes the descriptor and unmaps.
 */
class MemfdRegion {
public:
    MemfdRegion() noexcept = default;
    ~MemfdRegion();
    MemfdRegion(MemfdRegion&& other) noexcept;
    MemfdRegion& operator=(MemfdRegion&& other) noexcept;
    MemfdRegion(const MemfdRegion&) = delete;
    MemfdRegion& operator=(const MemfdRegion&) = delete;

    /** @brief Descriptor of the memfd, -1 when not holding a region. */
    [[nodiscard]] int Memfd() const noexcept { return memfd_; }

    /** @brief The attached mapping. */
    [[nodiscard]] const ShmMapping& Mapping() const noexcept { return mapping_; }

    /** @brief Tells whether the region is backed by huge pages. */
    [[nodiscard]] bool HugePage() const noexcept { return hugePage_; }

    /**
     * @brief Creates the shared pool of one cache domain.
     *
     * Steps: try to create and size a 2MB hugetlb memfd -- with hugetlb the
     * allocation gate is ftruncate, so the attempt spans both calls and
     * either failure falls back to normal pages unless forceHugePage
     * demands the error -- then interleave the mapping over every NUMA
     * node this process may allocate from (a single such node makes
     * interleaving vacuous; an indeterminable or rejected NUMA policy
     * fails the region rather than silently mis-placing it), and finally
     * walk the region touching one byte per page so the whole pool is
     * resident before anybody maps it.
     *
     * @param capacity      Slot count of the pool.
     * @param blockSize     Byte length of one slot.
     * @param forceHugePage Demands huge pages instead of falling back to
     *                      normal pages when the pool is unavailable.
     * @param name          Debugging label embedded in the memfd, visible
     *                      under /proc/<pid>/fd.
     *
     * @return Expected the region on success; InvalidParam for overflowing
     *         or degenerate geometry; OsApiError when a required syscall
     *         fails, including the forced-huge-page case with no pool
     *         available and a NUMA interleave that cannot be established.
     */
    [[nodiscard]] static Expected<MemfdRegion> Create(std::uint32_t capacity,
                                                      std::uint64_t blockSize, bool forceHugePage,
                                                      const std::string& name);

private:
    /** @brief Releases descriptor and mapping; idempotent. */
    void Reset() noexcept;

    MemfdRegion(int memfd, ShmMapping&& mapping, bool hugePage) noexcept
        : memfd_{memfd}, mapping_{std::move(mapping)}, hugePage_{hugePage}
    {
    }

    int memfd_{-1};
    ShmMapping mapping_;
    bool hugePage_{false};
};

}  // namespace UC::Store
