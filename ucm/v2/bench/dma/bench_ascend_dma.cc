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

/**
 * Ascend host<->device DMA bandwidth benchmark on top of the DmaEngine
 * abstraction.
 *
 * A round submits --tasks batch jobs at once -- each carrying
 * --io-count segments whose sizes cycle through the --io-size list, so a
 * multi-entry list measures a mixed workload -- then drains them with
 * Wait and reports the wall time of the whole round as a GB/s figure.
 * The working set never aliases: task t owns a contiguous host and
 * device slice, so the engine may run concurrent jobs on any stream
 * assignment it picks.
 *
 * The bench owns the Ascend runtime lifecycle -- aclInit, aclrtSetDevice,
 * aclFinalize; the engines take none of it -- and allocates both sides
 * itself: the host side as an anonymous mmap, optionally --huge 2 MiB
 * hugetlb or --interleave spread over the allowed NUMA nodes (a bench
 * must not measure a fallback it did not ask for, so a refused hugetlb
 * mmap or interleave policy is fatal), or as aclrtMallocHost; the device
 * side as one aclrtMalloc block. The host region is handed to the
 * engine at Setup, which pins it best effort.
 */

#include <acl/acl.h>
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <limits>
#include <linux/mempolicy.h>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>
#include "memory/dma/dma_engine.h"
#include "memory/dma/dma_engine_registry.h"
#include "status/status.h"

/* Older headers may lack the hugetlb mmap flags; fall back to the
 * canonical Linux ABI values (dev-sandbox parity). */
#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif
#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif

namespace {

using UC::Expected;
using UC::Status;
using UC::Transfer::DmaEngine;
using DmaJob = DmaEngine::DmaJob;

/** @brief The one huge page size --huge asks for: 2 MiB hugetlb. */
constexpr std::size_t kHugePageSize = 2UL << 20;
/** @brief mmap flag selecting 2 MiB hugetlb pages (huge page size 21). */
constexpr int kHugePageFlag = 21 << MAP_HUGE_SHIFT;
/** @brief Where the NUMA nodes this process may allocate from are published. */
constexpr const char* kStatusPath = "/proc/self/status";

/** @brief Host working set backing of the bench. */
enum class HostMem {
    Mmap,  /* anonymous private mapping */
    Aclrt, /* aclrtMallocHost allocation */
};

/** @brief Transfer direction of the measured traffic. */
enum class Direction {
    H2D, /* host -> device */
    D2H, /* device -> host */
};

/** @brief Command line state of the bench, one field per option; the
 *         defaults mirror the --help text and must stay in sync with it. */
struct Options {
    std::string engine{}; /* empty takes this build's default engine */
    HostMem hostMem{HostMem::Mmap};
    bool huge{false};
    bool interleave{false};
    std::size_t warmup{3};
    std::size_t rounds{10};
    Direction direction{Direction::H2D};
    std::vector<std::size_t> ioSizes{1024 * 1024};
    std::size_t ioCount{1024};
    std::size_t tasks{1};
    std::size_t streams{1};
    int32_t device{0};
    bool help{false};
};

/**
 * @brief Parses one decimal unsigned integer; blanks, signs, any other
 *        character and out-of-range values reject the input.
 */
Expected<std::size_t> ParseUnsigned(std::string_view text)
{
    std::size_t value = 0;
    const char* const begin = text.data();
    const auto [ptr, ec] = std::from_chars(begin, begin + text.size(), value);
    if (text.empty() || ec != std::errc{} || ptr != begin + text.size()) {
        return Status::InvalidParam("expected an unsigned integer, got '{}'", text);
    }
    return value;
}

/**
 * @brief Parses one io size: a byte count with an optional binary unit
 *        suffix -- k/K, m/M or g/G, each optionally followed by b/B --
 *        so 512, 4k, 64KB, 1M and 2G all parse; zero rejects.
 */
Expected<std::size_t> ParseIoSize(std::string_view text)
{
    while (!text.empty() && text.front() == ' ') { text.remove_prefix(1); }
    while (!text.empty() && text.back() == ' ') { text.remove_suffix(1); }
    std::size_t numberEnd = 0;
    while (numberEnd < text.size() && text[numberEnd] >= '0' && text[numberEnd] <= '9') {
        ++numberEnd;
    }
    const auto number = ParseUnsigned(text.substr(0, numberEnd));
    if (!number.HasValue()) { return Status::InvalidParam("io size '{}' has no byte count", text); }
    std::string_view suffix = text.substr(numberEnd);
    unsigned unitShift = 0;
    if (!suffix.empty()) {
        const char unit = suffix.front();
        if (unit == 'k' || unit == 'K') {
            unitShift = 10;
        } else if (unit == 'm' || unit == 'M') {
            unitShift = 20;
        } else if (unit == 'g' || unit == 'G') {
            unitShift = 30;
        } else {
            return Status::InvalidParam("io size '{}' carries an unknown unit", text);
        }
        suffix.remove_prefix(1);
        if (!suffix.empty() && (suffix.front() == 'b' || suffix.front() == 'B')) {
            suffix.remove_prefix(1);
        }
        if (!suffix.empty()) {
            return Status::InvalidParam("io size '{}' carries trailing characters", text);
        }
    }
    const std::size_t sizeMax = std::numeric_limits<std::size_t>::max();
    if (number.Value() > (sizeMax >> unitShift)) {
        return Status::InvalidParam("io size '{}' overflows size_t", text);
    }
    const std::size_t size = number.Value() << unitShift;
    if (size == 0) { return Status::InvalidParam("io size '{}' is zero", text); }
    return size;
}

/**
 * @brief Splits the comma-separated io size list and parses every entry;
 *        blank entries reject, blanks around an entry are tolerated.
 */
Expected<std::vector<std::size_t>> ParseIoSizes(std::string_view text)
{
    std::vector<std::size_t> sizes;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
        const auto size = ParseIoSize(text.substr(start, end - start));
        if (!size.HasValue()) { return size.Error(); }
        sizes.push_back(size.Value());
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    return sizes;
}

/**
 * @brief Parses one device ordinal: a non-negative decimal that fits
 *        an int32.
 */
Expected<int32_t> ParseDevice(std::string_view text)
{
    int64_t value = 0;
    const char* const begin = text.data();
    const auto [ptr, ec] = std::from_chars(begin, begin + text.size(), value);
    if (text.empty() || ec != std::errc{} || ptr != begin + text.size()) {
        return Status::InvalidParam("expected a device ordinal, got '{}'", text);
    }
    if (value < 0 || value > std::numeric_limits<int32_t>::max()) {
        return Status::InvalidParam("device ordinal '{}' is out of the int32 range", text);
    }
    return static_cast<int32_t>(value);
}

/**
 * @brief Parses one positive count, wrapping rejects with the option
 *        name that wants it.
 */
Expected<std::size_t> ParseCount(std::string_view text, std::string_view option)
{
    const auto parsed = ParseUnsigned(text);
    if (!parsed.HasValue()) { return Status::InvalidParam("{}: {}", option, parsed.Error()); }
    if (parsed.Value() == 0) { return Status::InvalidParam("{} must be at least 1", option); }
    return parsed;
}

/** @brief Prints the usage; the defaults cited live in Options and must
 *         stay in sync with this text. */
void PrintHelp(std::string_view program)
{
    fmt::println("Ascend DMA bandwidth benchmark (host <-> device).");
    fmt::println("");
    fmt::println("Usage: {} [options]", program);
    fmt::println("");
    fmt::println("Options:");
    fmt::println("  --engine <name>    dma engine name; default is this build's default engine");
    fmt::println("  --host-mem <type>  host memory type: mmap | aclrt (default: mmap)");
    fmt::println(
        "  --huge             mmap host memory on 2 MiB hugetlb pages (mmap only; fails hard");
    fmt::println("                     when no huge pages are reserved; default: off)");
    fmt::println(
        "  --interleave       interleave mmap host memory over the allowed NUMA nodes (mmap");
    fmt::println("                     only; fails hard when mbind is rejected; default: off)");
    fmt::println("  --warmup <n>       warmup rounds before measuring (default: 3)");
    fmt::println("  --rounds <n>       measured rounds (default: 10)");
    fmt::println("  --direction <dir>  transfer direction: h2d | d2h (default: h2d)");
    fmt::println(
        "  --io-size <list>   comma-separated io sizes with k/m/g suffixes, e.g. 4K,64K,1M");
    fmt::println("                     (default: 1M); mixed entries measure a mixed workload");
    fmt::println("  --io-count <n>     io count per task (default: 1024)");
    fmt::println(
        "  --tasks <n>        concurrent dma tasks -- batch jobs in flight per round (default: 1)");
    fmt::println("  --streams <n>      dma stream count handed to the engine (default: 1)");
    fmt::println("  --device <id>      device ordinal (default: 0)");
    fmt::println("  --help             print this help");
}

/**
 * @brief Parses the command line; every reject names the offending
 *        option, and --help prints the syntax instead of running.
 */
Expected<Options> ParseOptions(int argc, char const* const argv[])
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        const auto takeValue = [&argv, &i, argc, &arg]() -> Expected<std::string_view> {
            if (i + 1 >= argc) { return Status::InvalidParam("option '{}' needs a value", arg); }
            return std::string_view{argv[++i]};
        };
        if (arg == "--help") {
            options.help = true;
        } else if (arg == "--engine") {
            const auto value = takeValue();
            if (!value.HasValue()) { return value.Error(); }
            options.engine.assign(value.Value());
        } else if (arg == "--host-mem") {
            const auto value = takeValue();
            if (!value.HasValue()) { return value.Error(); }
            if (value.Value() == "mmap") {
                options.hostMem = HostMem::Mmap;
            } else if (value.Value() == "aclrt") {
                options.hostMem = HostMem::Aclrt;
            } else {
                return Status::InvalidParam("unknown host memory type '{}' (valid: mmap, aclrt)",
                                            value.Value());
            }
        } else if (arg == "--huge") {
            options.huge = true;
        } else if (arg == "--interleave") {
            options.interleave = true;
        } else if (arg == "--warmup") {
            const auto value = takeValue();
            if (!value.HasValue()) { return value.Error(); }
            const auto parsed = ParseUnsigned(value.Value());
            if (!parsed.HasValue()) { return Status::InvalidParam("--warmup: {}", parsed.Error()); }
            options.warmup = parsed.Value();
        } else if (arg == "--rounds") {
            const auto value = takeValue();
            if (!value.HasValue()) { return value.Error(); }
            const auto parsed = ParseCount(value.Value(), "--rounds");
            if (!parsed.HasValue()) { return parsed.Error(); }
            options.rounds = parsed.Value();
        } else if (arg == "--direction") {
            const auto value = takeValue();
            if (!value.HasValue()) { return value.Error(); }
            if (value.Value() == "h2d") {
                options.direction = Direction::H2D;
            } else if (value.Value() == "d2h") {
                options.direction = Direction::D2H;
            } else {
                return Status::InvalidParam("unknown direction '{}' (valid: h2d, d2h)",
                                            value.Value());
            }
        } else if (arg == "--io-size") {
            const auto value = takeValue();
            if (!value.HasValue()) { return value.Error(); }
            const auto parsed = ParseIoSizes(value.Value());
            if (!parsed.HasValue()) {
                return Status::InvalidParam("--io-size: {}", parsed.Error());
            }
            options.ioSizes = std::move(parsed.Value());
        } else if (arg == "--io-count") {
            const auto value = takeValue();
            if (!value.HasValue()) { return value.Error(); }
            const auto parsed = ParseCount(value.Value(), "--io-count");
            if (!parsed.HasValue()) { return parsed.Error(); }
            options.ioCount = parsed.Value();
        } else if (arg == "--tasks") {
            const auto value = takeValue();
            if (!value.HasValue()) { return value.Error(); }
            const auto parsed = ParseCount(value.Value(), "--tasks");
            if (!parsed.HasValue()) { return parsed.Error(); }
            options.tasks = parsed.Value();
        } else if (arg == "--streams") {
            const auto value = takeValue();
            if (!value.HasValue()) { return value.Error(); }
            const auto parsed = ParseCount(value.Value(), "--streams");
            if (!parsed.HasValue()) { return parsed.Error(); }
            options.streams = parsed.Value();
        } else if (arg == "--device") {
            const auto value = takeValue();
            if (!value.HasValue()) { return value.Error(); }
            const auto parsed = ParseDevice(value.Value());
            if (!parsed.HasValue()) { return parsed.Error(); }
            options.device = parsed.Value();
        } else {
            return Status::InvalidParam("unknown option '{}' (try --help)", arg);
        }
    }
    return options;
}

/** @brief Address-independent round layout derived from the options. */
struct Layout {
    std::size_t bytesPerJob{0};
    std::size_t bytesPerRound{0};
    std::size_t maxIoSize{0};
};

/**
 * @brief Computes the byte count each task moves, the byte count the
 *        whole round moves, and the largest single io -- the engine's
 *        max segment length. Every product is overflow checked.
 */
Expected<Layout> ComputeLayout(const Options& options)
{
    Layout layout;
    layout.maxIoSize = *std::max_element(options.ioSizes.begin(), options.ioSizes.end());
    const std::size_t sizeMax = std::numeric_limits<std::size_t>::max();
    /* ioCount >= 1 holds from parsing; the bound below keeps the sum loop
     * overflow free, since the sum cannot exceed ioCount * maxIoSize */
    if (layout.maxIoSize > sizeMax / options.ioCount) {
        return Status::InvalidParam("the io size list cannot shape {} ios without overflow",
                                    options.ioCount);
    }
    for (std::size_t i = 0; i < options.ioCount; ++i) {
        layout.bytesPerJob += options.ioSizes[i % options.ioSizes.size()];
    }
    if (layout.bytesPerJob > sizeMax / options.tasks) {
        return Status::InvalidParam("{} tasks x {} bytes per task overflows size_t", options.tasks,
                                    layout.bytesPerJob);
    }
    layout.bytesPerRound = layout.bytesPerJob * options.tasks;
    return layout;
}

/**
 * @brief Process-wide Ascend runtime guard: aclInit on demand,
 *        aclFinalize at teardown. The dma engines deliberately leave
 *        this lifecycle to their caller.
 */
class AscendRuntime {
public:
    AscendRuntime() = default;
    AscendRuntime(const AscendRuntime&) = delete;
    AscendRuntime& operator=(const AscendRuntime&) = delete;
    ~AscendRuntime()
    {
        if (initialized_) { aclFinalize(); }
    }

    Status Initialize()
    {
        const aclError code = aclInit(nullptr);
        if (code != ACL_SUCCESS) {
            return Status::OsApiError("aclInit failed: acl error {}", code);
        }
        initialized_ = true;
        return Status::Ok();
    }

private:
    bool initialized_{false};
};

/**
 * @brief Parses a decimal node id of at most four digits; anything else
 *        fails.
 */
bool ParseNodeNumber(std::string_view text, std::uint32_t& value)
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

/** @brief Parses one comma-separated element of the allowed list: "N" or "A-B". */
bool ParseNodeRange(std::string_view part, std::uint32_t& first, std::uint32_t& last)
{
    const std::size_t dash = part.find('-');
    if (dash == std::string_view::npos) {
        if (!ParseNodeNumber(part, first)) { return false; }
        last = first;
        return true;
    }
    return ParseNodeNumber(part.substr(0, dash), first) &&
           ParseNodeNumber(part.substr(dash + 1), last) && first <= last;
}

/**
 * @brief Reads the NUMA nodes this process may allocate from, published
 *        as the Mems_allowed_list line of /proc/self/status. The kernel
 *        reports the same set through getmempolicy(MPOL_F_MEMS_ALLOWED),
 *        but that syscall is filtered behind CAP_SYS_NICE by the default
 *        container seccomp profiles, while the file always reads.
 *
 * @return Expected the node ids on success; OsApiError when the file is
 *         unreadable or the line is missing or malformed -- an
 *         indeterminable set is never guessed.
 */
Expected<std::vector<std::uint32_t>> ReadAllowedNodes()
{
    const int fd = ::open(kStatusPath, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return Status::OsApiError("open of {} failed: {}", kStatusPath, std::strerror(errno));
    }

    std::string text;
    char chunk[512];
    for (;;) {
        const ssize_t n = ::read(fd, chunk, sizeof(chunk));
        if (n < 0) {
            if (errno == EINTR) { continue; }
            const int err = errno;
            ::close(fd);
            return Status::OsApiError("read of {} failed: {}", kStatusPath, std::strerror(err));
        }
        if (n == 0) { break; }
        text.append(chunk, static_cast<std::size_t>(n));
    }
    ::close(fd);

    const std::string key = "Mems_allowed_list:";
    const std::size_t keyAt = text.find(key);
    if (keyAt == std::string::npos) {
        return Status::OsApiError("{} carries no Mems_allowed_list line", kStatusPath);
    }
    std::size_t pos = keyAt + key.size();
    std::size_t end = text.find('\n', pos);
    if (end == std::string::npos) { end = text.size(); }
    while (pos < end && (text[pos] == ' ' || text[pos] == '\t')) { ++pos; }
    while (end > pos && (text[end - 1] == ' ' || text[end - 1] == '\t' || text[end - 1] == '\r')) {
        --end;
    }
    const std::string list = text.substr(pos, end - pos);

    std::vector<std::uint32_t> nodes;
    pos = 0;
    while (pos <= list.size()) {
        const std::size_t comma = list.find(',', pos);
        const std::size_t len = comma == std::string::npos ? std::string::npos : comma - pos;
        std::uint32_t first = 0;
        std::uint32_t last = 0;
        if (!ParseNodeRange(list.substr(pos, len), first, last)) {
            return Status::OsApiError("malformed Mems_allowed_list value '{}'", list);
        }
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
 * @brief Interleaves the mapping over every NUMA node this process may
 *        allocate from (see ReadAllowedNodes) -- in a cpuset-restricted
 *        container that is the container's own node set, never the
 *        host-wide online list. Strict: a single allocatable node makes
 *        interleaving vacuous, but an unreadable allowed set or a
 *        rejected mbind fails the caller rather than silently
 *        mis-placing the working set. The mask is padded to the word
 *        count the kernel reads for the given bit width: mbind rounds
 *        maxnode up to whole unsigned longs and copies that many words
 *        from user space.
 *
 * @return Ok on success; OsApiError when the allowed-node set cannot be
 *         read or the interleave policy is rejected.
 */
Status InterleaveOverAllowedNodes(void* base, std::size_t size)
{
    auto nodes = ReadAllowedNodes();
    if (!nodes.HasValue()) { return nodes.Error(); }
    const std::vector<std::uint32_t>& allowed = nodes.Value();
    if (allowed.size() <= 1) { return Status::Ok(); }

    std::uint32_t highest = 0;
    for (std::uint32_t node : allowed) { highest = std::max(highest, node); }
    const std::size_t maxnode = static_cast<std::size_t>(highest) + 2;
    const std::size_t words = (maxnode + 63) / 64;
    std::vector<unsigned long> mask(words, 0UL);
    for (std::uint32_t node : allowed) { mask[node / 64] |= 1UL << (node % 64); }

    if (::syscall(SYS_mbind, base, size, MPOL_INTERLEAVE, mask.data(), maxnode, 0) != 0) {
        return Status::OsApiError("mbind interleave over {} allowed NUMA nodes failed: {}",
                                  allowed.size(), std::strerror(errno));
    }
    return Status::Ok();
}

/**
 * @brief Host working set of the bench: one contiguous region released
 *        by the path it was allocated on -- munmap for mmap,
 *        aclrtFreeHost for aclrt -- optionally interleaved over the
 *        allowed NUMA nodes before its pages are touched, then fully
 *        touched at allocation so the measured rounds never fault.
 *        mappedBytes_ carries the engine-facing region length: the
 *        2 MiB-aligned mapping length on the hugetlb path, the plain
 *        byte count otherwise.
 */
class HostBuffer {
public:
    HostBuffer() = default;
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    ~HostBuffer() { Release(); }

    Status Allocate(std::size_t bytes, HostMem type, bool huge, bool interleave);
    void Release() noexcept;

    void* Base() const noexcept { return base_; }
    std::size_t MappedBytes() const noexcept { return mappedBytes_; }

private:
    Status AllocateMmap(std::size_t bytes, bool huge, bool interleave);
    Status AllocateAclrt(std::size_t bytes);

    void* base_{nullptr};
    std::size_t mappedBytes_{0};
    HostMem type_{HostMem::Mmap};
};

Status HostBuffer::Allocate(std::size_t bytes, HostMem type, bool huge, bool interleave)
{
    Release();
    /* aclrtMallocHost has no page-size or NUMA knob, so huge and
     * interleave only shape the mmap path; the config echo shows the
     * effective backing */
    return type == HostMem::Aclrt ? AllocateAclrt(bytes) : AllocateMmap(bytes, huge, interleave);
}

Status HostBuffer::AllocateMmap(std::size_t bytes, bool huge, bool interleave)
{
    std::size_t mapped = bytes;
    int pageFlags = 0;
    if (huge) {
        const std::size_t sizeMax = std::numeric_limits<std::size_t>::max();
        if (bytes > sizeMax - (kHugePageSize - 1)) {
            return Status::InvalidParam("{} bytes cannot align to a 2 MiB page without overflow",
                                        bytes);
        }
        mapped = (bytes + kHugePageSize - 1) / kHugePageSize * kHugePageSize;
        pageFlags = MAP_HUGETLB | kHugePageFlag;
    }
    void* base = mmap(nullptr, mapped, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | pageFlags, -1, 0);
    if (base == MAP_FAILED) {
        if (huge) {
            /* a bench must not measure a fallback it did not ask for */
            return Status::OsApiError(
                "mapping {} bytes with 2 MiB hugetlb failed (errno {}: {}); reserve huge pages "
                "first, e.g. echo N > /proc/sys/vm/nr_hugepages",
                mapped, errno, std::strerror(errno));
        }
        return Status::OsApiError("mapping {} bytes failed (errno {}: {})", bytes, errno,
                                  std::strerror(errno));
    }
    if (interleave) {
        /* the policy must sit before the first fault -- pages touched
         * earlier keep their node -- and a rejected policy is fatal:
         * the bench would otherwise measure a placement it did not ask
         * for */
        const Status spread = InterleaveOverAllowedNodes(base, mapped);
        if (spread.Failure()) {
            munmap(base, mapped);
            return spread;
        }
    }
    std::memset(base, 0, mapped); /* fault every page in before the rounds */
    mlock(base, mapped);          /* best effort: keep the pages resident */
    base_ = base;
    mappedBytes_ = mapped;
    type_ = HostMem::Mmap;
    return Status::Ok();
}

Status HostBuffer::AllocateAclrt(std::size_t bytes)
{
    void* base = nullptr;
    const aclError code = aclrtMallocHost(&base, bytes);
    if (code != ACL_SUCCESS || base == nullptr) {
        return Status::OsApiError("aclrtMallocHost of {} bytes failed: acl error {}", bytes, code);
    }
    std::memset(base, 0, bytes);
    base_ = base;
    mappedBytes_ = bytes;
    type_ = HostMem::Aclrt;
    return Status::Ok();
}

void HostBuffer::Release() noexcept
{
    if (base_ == nullptr) { return; }
    if (type_ == HostMem::Mmap) {
        munmap(base_, mappedBytes_);
    } else {
        aclrtFreeHost(base_);
    }
    base_ = nullptr;
    mappedBytes_ = 0;
}

/**
 * @brief Device working set of the bench: one contiguous aclrtMalloc
 *        allocation, ACL_MEM_MALLOC_HUGE_FIRST so the runtime prefers
 *        device huge pages, released with aclrtFree -- and fully
 *        touched at allocation, like the host side, so the first
 *        measured round never pays the cold-page cost.
 */
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { Release(); }

    Status Allocate(std::size_t bytes);
    void Release() noexcept;

    void* Base() const noexcept { return base_; }

private:
    void* base_{nullptr};
};

Status DeviceBuffer::Allocate(std::size_t bytes)
{
    void* base = nullptr;
    const aclError code = aclrtMalloc(&base, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (code != ACL_SUCCESS || base == nullptr) {
        return Status::OsApiError("aclrtMalloc of {} bytes failed: acl error {}", bytes, code);
    }
    /* first touch: whether a CANN build commits device pages eagerly or
     * lazily, the round that first reaches a page pays for it, and
     * --warmup 0 opts out of the warmup rounds covering that */
    const aclError touched = aclrtMemset(base, bytes, 0, bytes);
    if (touched != ACL_SUCCESS) {
        aclrtFree(base);
        return Status::OsApiError("first-touching {} bytes of device memory failed: acl error {}",
                                  bytes, touched);
    }
    base_ = base;
    return Status::Ok();
}

void DeviceBuffer::Release() noexcept
{
    if (base_ == nullptr) { return; }
    aclrtFree(base_);
    base_ = nullptr;
}

/**
 * @brief Prebuilt transfer plan of one round: tasks DmaJob batches, job
 *        t covering its own contiguous host and device slice so the
 *        jobs of a round never alias and the engine may run them all in
 *        parallel. Segment sizes cycle through the io size list. The
 *        same plan is resubmitted under the same ids every round.
 */
struct Workload {
    std::vector<DmaJob> jobs;
    std::size_t bytesPerRound{0};
};

Workload BuildWorkload(const Options& options, const Layout& layout, const HostBuffer& host,
                       const DeviceBuffer& device)
{
    Workload workload;
    workload.jobs.resize(options.tasks);
    workload.bytesPerRound = layout.bytesPerRound;
    auto* const hostCursor = static_cast<std::byte*>(host.Base());
    auto* const deviceCursor = static_cast<std::byte*>(device.Base());
    std::size_t offset = 0;
    for (std::size_t t = 0; t < options.tasks; ++t) {
        DmaJob& job = workload.jobs[t];
        job.sources.reserve(options.ioCount);
        job.destinations.reserve(options.ioCount);
        job.sizes.reserve(options.ioCount);
        for (std::size_t i = 0; i < options.ioCount; ++i) {
            const std::size_t size = options.ioSizes[i % options.ioSizes.size()];
            void* const hostAddress = hostCursor + offset;
            void* const deviceAddress = deviceCursor + offset;
            if (options.direction == Direction::H2D) {
                job.sources.push_back(hostAddress);
                job.destinations.push_back(deviceAddress);
            } else {
                job.sources.push_back(deviceAddress);
                job.destinations.push_back(hostAddress);
            }
            job.sizes.push_back(size);
            offset += size;
        }
    }
    return workload;
}

/** @brief One measured quantity of a round: bytes moved and seconds spent. */
struct RoundResult {
    std::size_t bytes{0};
    double seconds{0.0};
};

/**
 * @brief Runs one round: submits every task of the workload at once,
 *        then consumes every completion in id order. Submissions
 *        overlap in flight, so the round measures dma bandwidth, not
 *        per-call latency.
 */
Expected<RoundResult> RunRound(DmaEngine& engine, const Workload& workload, Direction direction)
{
    const auto begin = std::chrono::steady_clock::now();
    for (std::size_t t = 0; t < workload.jobs.size(); ++t) {
        const Status submitted = direction == Direction::H2D
                                     ? engine.SubmitH2DBatch(t, workload.jobs[t])
                                     : engine.SubmitD2HBatch(t, workload.jobs[t]);
        if (submitted.Failure()) { return submitted; }
    }
    for (std::size_t t = 0; t < workload.jobs.size(); ++t) {
        const Status settled = engine.Wait(t);
        if (settled.Failure()) { return settled; }
    }
    const auto end = std::chrono::steady_clock::now();
    return RoundResult{workload.bytesPerRound, std::chrono::duration<double>(end - begin).count()};
}

/** @brief GB/s -- 10^9 bytes per second -- of one round. */
double Bandwidth(const RoundResult& round)
{
    return static_cast<double>(round.bytes) / round.seconds / 1e9;
}

/** @brief Echoes the effective configuration before the first byte moves. */
void PrintConfig(const Options& options, const Layout& layout)
{
    std::string hostMem =
        options.hostMem == HostMem::Aclrt ? "aclrt (aclrtMallocHost)" : "mmap (anonymous)";
    if (options.hostMem == HostMem::Mmap && options.huge) { hostMem += ", 2 MiB hugetlb"; }
    if (options.hostMem == HostMem::Mmap && options.interleave) { hostMem += ", NUMA interleaved"; }
    fmt::println("engine           : {}",
                 options.engine.empty() ? "(build default)" : options.engine);
    fmt::println("device           : {}", options.device);
    fmt::println("direction        : {}", options.direction == Direction::H2D ? "h2d" : "d2h");
    fmt::println("host memory      : {}", hostMem);
    fmt::println("streams          : {}", options.streams);
    fmt::println("tasks            : {}", options.tasks);
    fmt::println("io count         : {} per task", options.ioCount);
    fmt::println("io sizes (bytes) : {}", fmt::join(options.ioSizes, ", "));
    fmt::println("bytes per task   : {}", layout.bytesPerJob);
    fmt::println("bytes per round  : {}", layout.bytesPerRound);
    fmt::println("warmup rounds    : {}", options.warmup);
    fmt::println("test rounds      : {}", options.rounds);
    fmt::println("");
}

/**
 * @brief Aggregates the measured rounds into the headline numbers: the
 *         byte total, the wall total and the bandwidth spread. The
 *         average weighs rounds by their byte count -- here all rounds
 *         carry the same count, so it is the plain mean.
 */
void PrintSummary(const std::vector<RoundResult>& results)
{
    std::size_t bytes = 0;
    double seconds = 0.0;
    double best = 0.0;
    double worst = std::numeric_limits<double>::max();
    for (const RoundResult& round : results) {
        bytes += round.bytes;
        seconds += round.seconds;
        best = std::max(best, Bandwidth(round));
        worst = std::min(worst, Bandwidth(round));
    }
    const double average = static_cast<double>(bytes) / seconds / 1e9;
    fmt::println("");
    fmt::println("total    : {} bytes ({:.2f} GiB) in {:.3f} ms", bytes,
                 static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), seconds * 1000.0);
    fmt::println("bandwidth: {:.2f} GB/s avg, {:.2f} GB/s min, {:.2f} GB/s max", average, worst,
                 best);
}

/** @brief Reports a fatal status on stderr; the bench has no recovery path. */
int Fail(const Status& status)
{
    fmt::println(stderr, "bench_ascend_dma: {}", status);
    return EXIT_FAILURE;
}

/** @brief Benchmark body: parse, prepare, warm up, measure, report. */
int Run(int argc, char const* const argv[])
{
    const auto parsed = ParseOptions(argc, argv);
    if (!parsed.HasValue()) {
        fmt::println(stderr, "bench_ascend_dma: {}", parsed.Error());
        fmt::println(stderr, "run with --help for usage");
        return EXIT_FAILURE;
    }
    const Options& options = parsed.Value();
    if (options.help) {
        PrintHelp(argv[0]);
        return EXIT_SUCCESS;
    }

    const auto layoutCreated = ComputeLayout(options);
    if (!layoutCreated.HasValue()) { return Fail(layoutCreated.Error()); }
    const Layout layout = layoutCreated.Value();

    /* reject an unknown engine name before any resource moves */
    if (!options.engine.empty()) {
        const auto kind = UC::Transfer::DmaEngineKindFromName(options.engine);
        if (!kind.HasValue()) { return Fail(kind.Error()); }
    }

    AscendRuntime runtime;
    if (const Status st = runtime.Initialize(); st.Failure()) { return Fail(st); }
    const aclError bound = aclrtSetDevice(options.device);
    if (bound != ACL_SUCCESS) {
        return Fail(
            Status::OsApiError("binding to device {} failed: acl error {}", options.device, bound));
    }

    PrintConfig(options, layout);

    HostBuffer host;
    if (const Status st =
            host.Allocate(layout.bytesPerRound, options.hostMem, options.huge, options.interleave);
        st.Failure()) {
        return Fail(st);
    }
    DeviceBuffer device;
    if (const Status st = device.Allocate(layout.bytesPerRound); st.Failure()) { return Fail(st); }

    const Workload workload = BuildWorkload(options, layout, host, device);

    auto created = UC::Transfer::DmaEngineRegistry::Instance().Create(options.engine);
    if (!created.HasValue()) { return Fail(created.Error()); }
    std::unique_ptr<DmaEngine> engine{created.Value()};
    DmaEngine::InitParam param;
    param.deviceId = options.device;
    param.maxLength = layout.maxIoSize;
    param.hostBase = host.Base();
    param.hostLength = host.MappedBytes();
    param.nStream = options.streams;
    if (const Status st = engine->Setup(param); st.Failure()) { return Fail(st); }

    for (std::size_t w = 1; w <= options.warmup; ++w) {
        const auto round = RunRound(*engine, workload, options.direction);
        if (!round.HasValue()) { return Fail(round.Error()); }
        fmt::println("warmup {}/{}: {} bytes in {:.3f} ms", w, options.warmup, round.Value().bytes,
                     round.Value().seconds * 1000.0);
    }

    std::vector<RoundResult> results;
    results.reserve(options.rounds);
    for (std::size_t r = 1; r <= options.rounds; ++r) {
        const auto round = RunRound(*engine, workload, options.direction);
        if (!round.HasValue()) { return Fail(round.Error()); }
        results.push_back(round.Value());
        fmt::println("round {}/{}: {} bytes in {:.3f} ms, {:.2f} GB/s", r, options.rounds,
                     round.Value().bytes, round.Value().seconds * 1000.0, Bandwidth(round.Value()));
    }
    PrintSummary(results);
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char const* argv[])
{
    try {
        return Run(argc, argv);
    } catch (const std::bad_alloc&) {
        fmt::println(stderr, "bench_ascend_dma: out of memory");
        return EXIT_FAILURE;
    } catch (const std::exception& error) {
        fmt::println(stderr, "bench_ascend_dma: {}", error.what());
        return EXIT_FAILURE;
    }
}
