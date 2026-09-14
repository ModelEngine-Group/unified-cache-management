// Included once, after the native private BufferStrategy definitions.
// Keep Early operations out of native methods; reuse native Hash and metadata/locks.
#include "early_buffer_pins.h"
#include <algorithm>

namespace UC::CacheStore {
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
                                            bool* conflict, bool includeLoading)
{
    auto& strategy = *buffer.strategy_;
    if (conflict && !strategy.CanReserve(node)) {
        *conflict = true;
        return {};
    }
    strategy.NodeLock(node);
    auto* meta = strategy.MetaAt(node);
    const auto state = meta->state.load(std::memory_order_acquire);
    TransBuffer::Handle result;
    if (state == TransBuffer::State::READY ||
        (includeLoading && state == TransBuffer::State::LOADING && meta->reference != 0)) {
        ++meta->reference;
        strategy.MarkAccessed(node);
        result = TransBuffer::Handle(&buffer, node, false);
    }
    strategy.NodeUnlock(node);
    return result;
}

TransBuffer::Handle EarlyBufferPins::PinOne(TransBuffer& buffer, const Detail::BlockId& block,
                                           size_t shard, bool* conflict, bool includeLoading)
{
    auto& strategy = *buffer.strategy_;
    const auto bucket = Hash(block, shard);
    strategy.BucketLock(bucket);
    TransBuffer::Handle result;
    for (auto node = strategy.FirstAt(bucket); node != invalidIndex;) {
        const auto* meta = strategy.MetaAt(node);
        if (meta->block == block && meta->shard == shard) {
            result = PinNode(buffer, node, conflict, includeLoading);
            break;
        }
        node = meta->next;
    }
    strategy.BucketUnlock(bucket);
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
                    result.pins[q] = PinNode(buffer, node, &result.capacityConflict, true);
                    *link = next[q];
                } else {
                    link = &next[q];
                }
            }
            node = meta->next;
        }
        strategy.BucketUnlock(bucket);
        if (result.capacityConflict) { break; }
    }
    return result;
}
}  // namespace UC::CacheStore
