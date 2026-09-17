// Included once, after the native private BufferStrategy definitions.
// Keep Early operations out of native methods; reuse native Hash and metadata/locks.
#include "early_buffer_pins.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/sysmacros.h>

namespace UC::CacheStore {
// Same native identity, reference and allocation operations as Get, with the
// existing Alloc attempt bound. No waiting for another task to return capacity.
TransBuffer::Handle EarlyBufferPins::Acquire(TransBuffer& buffer,
    const Detail::BlockId& block, size_t shard, bool load)
{
    auto& strategy = *buffer.strategy_;
    const auto bucket = Hash(block, shard);
    bool owner = false;
    strategy.BucketLock(bucket);
    auto node = buffer.FindAt(bucket, block, shard, owner);
    if (node == invalidIndex && (load ? strategy.NodeCount() : ReservableSlots(buffer)) != 0) {
        node = buffer.Alloc(block, shard, bucket, load, 2 * strategy.NodeCount());
        owner = true;
    } else if (load && buffer.bypassHitOnLoad_ && owner && buffer.Ready(node)) {
        buffer.MarkNotReady(node);
    }
    strategy.BucketUnlock(bucket);
    return TransBuffer::Handle(&buffer, node, owner);
}

Early::Domain EarlyBufferPins::DescribeDomain(TransBuffer& buffer)
{
    Early::Domain result;
    auto& strategy = *buffer.strategy_;
    result.nodes = strategy.NodeCount();
    result.reservable = ReservableSlots(buffer);
    if (!dynamic_cast<SharedBufferStrategy*>(&strategy) || !result.nodes) { return result; }
    // Resolve addresses only inside their owning process. Export inode/offsets,
    // never an address, filename-based guess, or a reconstructed shard layout.
    const auto data = reinterpret_cast<uintptr_t>(strategy.DataAt(0));
    const auto meta = reinterpret_cast<uintptr_t>(strategy.MetaAt(0));
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        unsigned long long start, end, offset, inode;
        unsigned major, minor;
        char permissions[5]{};
        int consumed = 0;
        if (std::sscanf(line.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n",
                        &start, &end, permissions, &offset, &major, &minor, &inode, &consumed) != 7) {
            continue;
        }
        if (permissions[3] != 's' || !inode || data < start || data >= end || meta < start || meta >= end) {
            continue;
        }
        auto path = line.substr(consumed);
        struct stat stat{};
        // An unlinked/replaced object is unsuitable for admitting a cohort.
        if (::stat(path.c_str(), &stat) || static_cast<uint64_t>(stat.st_ino) != inode ||
            ::major(stat.st_dev) != major || ::minor(stat.st_dev) != minor) { continue; }
        std::ostringstream identity;
        identity << major << ':' << minor << ':' << inode << ':' << stat.st_size
                 << ':' << offset + meta - start << ':' << offset + data - start;
        result.backing = identity.str();
        break;
    }
    return result;
}

size_t EarlyBufferPins::ReservableSlots(TransBuffer& buffer)
{
    // Native CanReserve is the contiguous prefix before its load-exclusive area.
    auto& strategy = *buffer.strategy_;
    size_t low = 0, high = strategy.NodeCount();
    while (low < high) {
        const auto middle = low + (high - low) / 2;
        if (strategy.CanReserve(middle)) { low = middle + 1; }
        else { high = middle; }
    }
    return low;
}

size_t EarlyBufferPins::HintBlockCapacity(TransBuffer& buffer, size_t shardsPerBlock)
{
    return shardsPerBlock ? std::max(size_t{1}, buffer.strategy_->NodeCount() / shardsPerBlock) : 0;
}

std::vector<size_t> EarlyBufferPins::Locations(const std::vector<TransBuffer::Handle>& pins)
{
    std::vector<size_t> result;
    result.reserve(pins.size());
    for (const auto& pin : pins) { result.push_back(pin.pos_); }
    return result;
}

std::vector<TransBuffer::Handle> EarlyBufferPins::Recall(TransBuffer& buffer,
    const Detail::BlockId& block, const std::vector<size_t>& locations, size_t& checks)
{
    auto& strategy = *buffer.strategy_;
    std::vector<TransBuffer::Handle> pins(locations.size());
    for (size_t shard = 0; shard < locations.size(); ++shard) {
        const auto node = locations[shard];
        if (node >= strategy.NodeCount() || !strategy.CanReserve(node)) { return {}; }
        strategy.NodeLock(node);
        ++checks;
        auto* meta = strategy.MetaAt(node);
        // Alloc changes identity under this same node lock and cannot recycle
        // a referenced node. A hint alone never establishes identity/readiness.
        const bool valid = meta->block == block && meta->shard == shard &&
            meta->state.load(std::memory_order_acquire) == TransBuffer::State::READY;
        if (valid) {
            ++meta->reference;
            strategy.MarkAccessed(node);
            pins[shard] = TransBuffer::Handle(&buffer, node, false);
        }
        strategy.NodeUnlock(node);
        if (!valid) { return {}; }  // prior acquisitions are released by RAII
    }
    return pins;
}

void EarlyBufferPins::Touch(TransBuffer& buffer, const std::vector<TransBuffer::Handle>& pins)
{
    // Existing references protect each position. Preserve native CLOCK hotness
    // without another bucket lookup, node lock or node-reference increment.
    for (const auto& pin : pins) {
        if (pin && pin.buf_ == &buffer) { buffer.strategy_->MarkAccessed(pin.pos_); }
    }
}

TransBuffer::Handle EarlyBufferPins::PinNode(TransBuffer& buffer, size_t node,
                                            bool& conflict)
{
    auto& strategy = *buffer.strategy_;
    if (!strategy.CanReserve(node)) {
        conflict = true;
        return {};
    }
    strategy.NodeLock(node);
    auto* meta = strategy.MetaAt(node);
    const auto state = meta->state.load(std::memory_order_acquire);
    TransBuffer::Handle result;
    if (state == TransBuffer::State::READY ||
        (state == TransBuffer::State::LOADING && meta->reference != 0)) {
        ++meta->reference;
        strategy.MarkAccessed(node);
        result = TransBuffer::Handle(&buffer, node, false);
    }
    strategy.NodeUnlock(node);
    return result;
}

EarlyBufferPins::Batch EarlyBufferPins::PinBlocks(TransBuffer& buffer,
    const Detail::BlockId* blocks, size_t count, size_t shardsPerBlock)
{
    Batch result;
    result.pins.resize(count * shardsPerBlock);
    if (result.pins.empty()) { return result; }
    // Temporary grouping uses the native hash; persistent hints are validated by Recall.
    std::vector<size_t> heads(nHashTableBucket, invalidIndex), next(result.pins.size());
    std::vector<size_t> used;
    used.reserve(std::min(result.pins.size(), nHashTableBucket));
    for (size_t q = 0; q < result.pins.size(); ++q) {
        const auto bucket = Hash(blocks[q / shardsPerBlock], q % shardsPerBlock);
        if (heads[bucket] == invalidIndex) { used.push_back(bucket); }
        next[q] = heads[bucket];
        heads[bucket] = q;
    }
    auto& strategy = *buffer.strategy_;
    for (auto bucket : used) {
        strategy.BucketLock(bucket);
        ++result.bucketLocks;
        auto remaining = heads[bucket];
        for (auto node = strategy.FirstAt(bucket); node != invalidIndex && remaining != invalidIndex;) {
            const auto* meta = strategy.MetaAt(node);
            auto* link = &remaining;
            while (*link != invalidIndex) {
                const auto q = *link;
                if (meta->block == blocks[q / shardsPerBlock] && meta->shard == q % shardsPerBlock) {
                    result.pins[q] = PinNode(buffer, node, result.capacityConflict);
                    *link = next[q];
                } else {
                    link = &next[q];
                }
            }
            node = meta->next;
        }
        strategy.BucketUnlock(bucket);
    }
    return result;
}
}  // namespace UC::CacheStore
