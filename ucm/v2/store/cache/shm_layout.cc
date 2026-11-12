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
#include "shm_layout.h"
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

/* Kernel UAPI constants that user-space libc headers do not carry. */
#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
#ifndef MFD_HUGETLB
#define MFD_HUGETLB 0x0004U
#endif
#ifndef MFD_HUGE_2MB
#define MFD_HUGE_2MB (21U << 26)
#endif

namespace UC::Store {
namespace {

/** Backing page size of the shared pool: 2MB transparent huge pages. */
constexpr std::size_t kShmHugePageSize = std::size_t{2} << 20;

/** MPOL_INTERLEAVE policy of the kernel memory-policy UAPI. */
constexpr int kMpolInterleave = 2;

/** Fallback page size of a non-huge mapping. */
constexpr std::size_t kNormalPageSize = 4096;

/** Where the online NUMA node list is published. */
constexpr const char* kNodeOnlinePath = "/sys/devices/system/node/online";

void PutU32(std::byte* dst, std::uint32_t value)
{
    for (int shift = 0; shift <= 24; shift += 8) {
        dst[shift / 8] = static_cast<std::byte>((value >> shift) & 0xFF);
    }
}

std::uint32_t GetU32(const std::byte* src)
{
    std::uint32_t value = 0;
    for (int shift = 0; shift <= 24; shift += 8) {
        value |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(src[shift / 8])) << shift;
    }
    return value;
}

bool IsPadding(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

/** Parses a decimal node id of at most four digits; anything else fails. */
bool ParseNodeNumber(const std::string& text, std::uint32_t& value)
{
    if (text.empty() || text.size() > 4) { return false; }
    std::uint32_t parsed = 0;
    for (char c : text) {
        if (c < '0' || c > '9') { return false; }
        parsed = parsed * 10 + static_cast<std::uint32_t>(c - '0');
    }
    value = parsed;
    return true;
}

/** Parses one comma-separated element of the online list: "N" or "A-B". */
bool ParseNodeRange(const std::string& part, std::uint32_t& first, std::uint32_t& last)
{
    const std::size_t dash = part.find('-');
    if (dash == std::string::npos) {
        if (!ParseNodeNumber(part, first)) { return false; }
        last = first;
        return true;
    }
    return ParseNodeNumber(part.substr(0, dash), first) &&
           ParseNodeNumber(part.substr(dash + 1), last) && first <= last;
}

/**
 * @brief Reads the online node list, e.g. "0" or "0-3" or "0,2-3". Returns
 *        an empty vector on any read or parse failure; the caller then skips
 *        the interleave step rather than binding against a guessed mask.
 */
std::vector<std::uint32_t> ParseOnlineNodes()
{
    std::vector<std::uint32_t> nodes;
    const int fd = ::open(kNodeOnlinePath, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { return nodes; }

    std::string text;
    char chunk[128];
    for (;;) {
        const ssize_t n = ::read(fd, chunk, sizeof(chunk));
        if (n < 0) {
            if (errno == EINTR) { continue; }
            ::close(fd);
            return {};
        }
        if (n == 0) { break; }
        text.append(chunk, static_cast<std::size_t>(n));
    }
    ::close(fd);

    while (!text.empty() && IsPadding(text.back())) { text.pop_back(); }
    std::size_t pos = 0;
    while (pos <= text.size()) {
        const std::size_t comma = text.find(',', pos);
        const std::size_t len = comma == std::string::npos ? std::string::npos : comma - pos;
        std::uint32_t first = 0;
        std::uint32_t last = 0;
        if (!ParseNodeRange(text.substr(pos, len), first, last)) { return {}; }
        for (std::uint32_t node = first;; ++node) {
            nodes.push_back(node);
            if (node == last) { break; }
        }
        if (comma == std::string::npos) { break; }
        pos = comma + 1;
    }
    return nodes;
}

/**
 * @brief Spreads the mapping over every online NUMA node. A single node (or
 *        an unreadable list) makes interleaving a no-op. The mask is padded
 *        to the word count the kernel reads for the given bit width: mbind
 *        rounds maxnode up to whole unsigned longs and copies that many
 *        words from user space.
 */
Status InterleaveOverOnlineNodes(void* base, std::size_t size)
{
    const std::vector<std::uint32_t> nodes = ParseOnlineNodes();
    if (nodes.size() <= 1) { return Status::Ok(); }

    std::uint32_t highest = 0;
    for (std::uint32_t node : nodes) { highest = std::max(highest, node); }
    const std::size_t maxnode = static_cast<std::size_t>(highest) + 2;
    const std::size_t words = (maxnode + 63) / 64;
    std::vector<unsigned long> mask(words, 0UL);
    for (std::uint32_t node : nodes) { mask[node / 64] |= 1UL << (node % 64); }

    if (::syscall(SYS_mbind, base, size, kMpolInterleave, mask.data(), maxnode, 0) != 0) {
        return Status::OsApiError("mbind interleave over {} online nodes failed: {}", nodes.size(),
                                  std::strerror(errno));
    }
    return Status::Ok();
}

/**
 * @brief Faults in every page with a one-byte store so the region is fully
 *        resident before peers map it. Huge mappings step per huge page,
 *        normal mappings per 4KB page; a coarser stride on normal pages
 *        would only fault one page in 512.
 */
void TouchEveryPage(void* base, std::size_t size, bool hugePage)
{
    const std::size_t stride = hugePage ? kShmHugePageSize : kNormalPageSize;
    auto* cursor = static_cast<volatile std::uint8_t*>(base);
    for (std::size_t offset = 0; offset < size; offset += stride) { cursor[offset] = 1; }
}

/**
 * @brief One attempt at creating and sizing a memfd. A descriptor of -1
 *        with errNo set means the attempt failed; for hugetlb attempts the
 *        failure surfaces at ftruncate time (the actual allocation moment),
 *        not at memfd_create.
 */
struct FdAttempt {
    int fd{-1};
    int errNo{0};
};

FdAttempt OpenMemfd(const std::string& name, std::size_t size, bool huge)
{
    const unsigned int flags =
        MFD_CLOEXEC | (huge ? static_cast<unsigned int>(MFD_HUGETLB | MFD_HUGE_2MB) : 0U);
    int fd = ::memfd_create(name.c_str(), flags);
    if (fd < 0) { return {-1, errno}; }
    if (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
        const int err = errno;
        ::close(fd);
        return {-1, err};
    }
    return {fd, 0};
}

/**
 * @brief Byte offset of one slot inside the shared pool.
 *
 * The result may cross huge-page boundaries when blockSize is not a multiple
 * of the page size; the mapping is one contiguous region, so crossing is
 * harmless. Precondition: slot < capacity of the pool the offset addresses.
 */
std::size_t SlotOffset(std::uint32_t slot, std::uint64_t blockSize)
{
    return static_cast<std::size_t>(static_cast<std::uint64_t>(slot) * blockSize);
}

}  // namespace

Expected<std::size_t> ShmTotalSize(std::uint32_t capacity, std::uint64_t blockSize)
{
    constexpr std::uint64_t kMaxU64 = std::numeric_limits<std::uint64_t>::max();
    if (capacity == 0 || blockSize == 0) {
        return Status::InvalidParam("shared pool geometry is degenerate: {} blocks of {} bytes",
                                    capacity, blockSize);
    }
    if (blockSize > (kMaxU64 - (kShmHugePageSize - 1)) / capacity) {
        return Status::InvalidParam("shared pool geometry overflows: {} blocks of {} bytes",
                                    capacity, blockSize);
    }
    const std::uint64_t raw = static_cast<std::uint64_t>(capacity) * blockSize;
    const std::uint64_t rounded =
        (raw + kShmHugePageSize - 1) / kShmHugePageSize * kShmHugePageSize;
    return static_cast<std::size_t>(rounded);
}

void* SlotAddress(void* shmBase, std::uint32_t slot, std::uint64_t blockSize)
{
    return static_cast<std::uint8_t*>(shmBase) + SlotOffset(slot, blockSize);
}

std::vector<std::byte> EncodeTokenPayload(const std::vector<std::uint32_t>& slots)
{
    std::vector<std::byte> data(slots.size() * 4);
    for (std::size_t i = 0; i < slots.size(); ++i) { PutU32(data.data() + i * 4, slots[i]); }
    return data;
}

std::vector<std::uint32_t> DecodeTokenPayload(const std::vector<std::byte>& data)
{
    std::vector<std::uint32_t> slots(data.size() / 4);
    for (std::size_t i = 0; i < slots.size(); ++i) { slots[i] = GetU32(data.data() + i * 4); }
    return slots;
}

ShmMapping::~ShmMapping() { Reset(); }

ShmMapping::ShmMapping(ShmMapping&& other) noexcept
    : base_{std::exchange(other.base_, nullptr)}, size_{std::exchange(other.size_, 0)}
{
}

ShmMapping& ShmMapping::operator=(ShmMapping&& other) noexcept
{
    if (this != &other) {
        Reset();
        base_ = std::exchange(other.base_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

void ShmMapping::Reset() noexcept
{
    if (base_ != nullptr) {
        ::munmap(base_, size_);
        base_ = nullptr;
        size_ = 0;
    }
}

Expected<ShmMapping> ShmMapping::Attach(int fd, std::size_t expectedSize)
{
    struct ::stat st{};
    if (::fstat(fd, &st) != 0) {
        return Status::OsApiError("fstat of the shared memory object failed: {}",
                                  std::strerror(errno));
    }
    if (st.st_size < 0 || static_cast<std::uint64_t>(st.st_size) != expectedSize) {
        return Status::InvalidParam("shared memory object is {} bytes, expected {}", st.st_size,
                                    expectedSize);
    }

    void* base = ::mmap(nullptr, expectedSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        return Status::OsApiError("mmap of {} bytes of shared memory failed: {}", expectedSize,
                                  std::strerror(errno));
    }
    return ShmMapping{base, expectedSize};
}

MemfdRegion::~MemfdRegion() { Reset(); }

MemfdRegion::MemfdRegion(MemfdRegion&& other) noexcept
    : memfd_{std::exchange(other.memfd_, -1)},
      mapping_{std::move(other.mapping_)},
      hugePage_{std::exchange(other.hugePage_, false)}
{
}

MemfdRegion& MemfdRegion::operator=(MemfdRegion&& other) noexcept
{
    if (this != &other) {
        Reset();
        memfd_ = std::exchange(other.memfd_, -1);
        mapping_ = std::move(other.mapping_);
        hugePage_ = std::exchange(other.hugePage_, false);
    }
    return *this;
}

void MemfdRegion::Reset() noexcept
{
    if (memfd_ >= 0) {
        ::close(memfd_);
        memfd_ = -1;
    }
    mapping_.Reset();
    hugePage_ = false;
}

Expected<MemfdRegion> MemfdRegion::Create(std::uint32_t capacity, std::uint64_t blockSize,
                                          bool forceHugePage, const std::string& name)
{
    auto total = ShmTotalSize(capacity, blockSize);
    if (!total.HasValue()) { return total.Error(); }

    FdAttempt attempt = OpenMemfd(name, total.Value(), true);
    bool hugePage = attempt.fd >= 0;
    if (attempt.fd < 0) {
        if (forceHugePage) {
            return Status::OsApiError("hugetlb memfd '{}' of {} bytes failed: {}", name,
                                      total.Value(), std::strerror(attempt.errNo));
        }
        attempt = OpenMemfd(name, total.Value(), false);
        if (attempt.fd < 0) {
            return Status::OsApiError("memfd '{}' of {} bytes failed: {}", name, total.Value(),
                                      std::strerror(attempt.errNo));
        }
    }
    const int memfd = attempt.fd;

    auto mapping = ShmMapping::Attach(memfd, total.Value());
    if (!mapping.HasValue()) {
        ::close(memfd);
        return mapping.Error();
    }

    if (Status st = InterleaveOverOnlineNodes(mapping.Value().Base(), total.Value());
        st.Failure()) {
        ::close(memfd);
        return st;
    }

    TouchEveryPage(mapping.Value().Base(), total.Value(), hugePage);
    return MemfdRegion{memfd, std::move(mapping.Value()), hugePage};
}

}  // namespace UC::Store
