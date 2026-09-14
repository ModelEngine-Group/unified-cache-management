#ifndef UCM_RESERVED_LOCALIZE_IMPL_H
#define UCM_RESERVED_LOCALIZE_IMPL_H
namespace UC::CacheStore {
inline Expected<std::shared_ptr<ReservedPrefetch::BlockPins>> ReservedPrefetch::NewGroup()
{
    auto group = std::make_shared<BlockPins>();
    group->charge = std::make_shared<Charge>();
    group->charge->budget = budget_;
    const auto bytes = shardsPerBlock_ * shardBytes_;
    auto used = budget_->used.load();
    for (;;) {
        if (bytes > budget_->limit - used) {
            // Fail instead of waiting while other partial requests hold the
            // credits needed to complete this one. Native fallback can compute.
            return Status::NoSpace();
        }
        if (budget_->used.compare_exchange_weak(used, used + bytes)) { break; }
    }
    group->charge->bytes = bytes;
    return group;
}

inline Expected<std::vector<std::shared_ptr<ReservedPrefetch::BlockPins>>>
ReservedPrefetch::Locate(const Detail::BlockId* blocks, size_t count, EarlyStats& stats)
{
    std::vector<std::shared_ptr<BlockPins>> groups(count);
    struct Leader {
        size_t offset;
        std::shared_ptr<ProbeFlight> flight;
        std::shared_ptr<const std::vector<size_t>> hint;
    };
    std::vector<Leader> leaders;
    std::vector<std::pair<size_t, std::shared_ptr<ProbeFlight>>> followers;
    std::vector<size_t> reused;
    leaders.reserve(count);
    followers.reserve(count);
    reused.reserve(count);
    auto localize = [&]() -> Status {
        {
            std::lock_guard guard(mutex_);
            for (size_t i = 0; i < count; ++i) {
                auto hint = readyBlocks_.find(blocks[i]);
                if (hint != readyBlocks_.end()) {
                    groups[i] = hint->second.live.lock();
                    hintAge_.splice(hintAge_.begin(), hintAge_, hint->second.age);
                }
                if (groups[i]) {
                    reused.push_back(i);
                    ++stats.reusedBlocks;
                    continue;
                }
                auto pending = locating_.find(blocks[i]);
                if (pending != locating_.end()) {
                    followers.emplace_back(i, pending->second);
                    ++stats.joinedBlocks;
                    continue;
                }
                auto flight = std::make_shared<ProbeFlight>();
                // Record ownership before map insertion can throw. Every
                // published promise is resolved, including partial failures.
                leaders.push_back({i, flight, hint == readyBlocks_.end() ? nullptr : hint->second.locations});
                locating_.emplace(blocks[i], std::move(flight));
            }
        }
        for (auto index : reused) { EarlyBufferPins::Touch(*buffer_, groups[index]->pins); }
        std::vector<Detail::BlockId> probe;
        std::vector<size_t> offsets;
        for (const auto& leader : leaders) {
            auto created = NewGroup();
            if (!created) { return created.Error(); }
            auto& group = groups[leader.offset];
            group = std::move(created.Value());
            if (leader.hint && leader.hint->size() == shardsPerBlock_) {
                group->pins = EarlyBufferPins::Recall(*buffer_, blocks[leader.offset], *leader.hint, stats.hintChecks);
                if (!group->pins.empty()) {
                    group->ready = true;
                    ++stats.recalledBlocks;
                    continue;
                }
            }
            probe.push_back(blocks[leader.offset]);
            offsets.push_back(leader.offset);
        }
        auto batch = EarlyBufferPins::PinBlocks(*buffer_, probe.data(), probe.size(), shardsPerBlock_);
        stats.shards += batch.pins.size();
        stats.bucketLocks += batch.bucketLocks;
        if (batch.capacityConflict) { return Status::Error("prefix occupies native load-exclusive buffer"); }
        for (size_t j = 0; j < offsets.size(); ++j) {
            auto& group = groups[offsets[j]];
            group->pins.resize(shardsPerBlock_);
            group->ready = true;
            for (size_t shard = 0; shard < shardsPerBlock_; ++shard) {
                auto& pin = group->pins[shard];
                pin = std::move(batch.pins[j * shardsPerBlock_ + shard]);
                if (!pin || !pin.Ready()) { group->ready = false; }
            }
        }
        return Status::OK();
    };
    auto status = Status::OK();
    try { status = localize(); }
    catch (const std::bad_alloc&) { status = Status::OutOfMemory(); }
    catch (const std::exception& error) { status = Status::Error(fmt::format("localization failed: {}", error.what())); }
    {
        std::lock_guard guard(mutex_);
        // Resolve our leaders BEFORE awaiting followers. Opposite block order
        // across two requests cannot introduce an A->B->A wait cycle.
        for (const auto& leader : leaders) {
            if (status.Success()) { Remember(blocks[leader.offset], groups[leader.offset]); }
            leader.flight->promise.set_value({Status{status.Underlying(), {}},
                status.Success() ? groups[leader.offset] : nullptr});
            auto found = locating_.find(blocks[leader.offset]);
            if (found != locating_.end() && found->second == leader.flight) { locating_.erase(found); }
        }
    }
    if (status.Failure()) { return status; }
    for (const auto& [index, flight] : followers) {
        EarlyStats::Phase phase{stats.joinMs};
        // These are concurrent CPU lookups, never waits for producer KV or SSD
        // persistence. No global/bucket lock is held during this bounded wait.
        if (flight->future.wait_for(std::chrono::milliseconds(timeoutMs_)) != std::future_status::ready) {
            return Status::Error("concurrent localization timeout");
        }
        const auto& outcome = flight->future.get();
        if (outcome.status.Failure()) { return outcome.status; }
        groups[index] = outcome.group;
    }
    leaders.clear();
    followers.clear();
    for (auto& group : groups) {
        if (!group->ready && group.use_count() > 1) {
            // Published localization is immutable. Each data task owns its
            // missing positions; TryGet revalidates negatives and coalesces I/O.
            // Known handles and the block's byte claim remain shared/protected.
            auto copy = std::make_shared<BlockPins>();
            copy->charge = group->charge;
            copy->pins = group->pins;
            group = std::move(copy);
        }
    }
    return groups;
}
}  // namespace UC::CacheStore
#endif
