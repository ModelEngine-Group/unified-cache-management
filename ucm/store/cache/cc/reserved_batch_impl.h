// Included after ReservedPrefetch; this is its Early-only task state.
#ifndef UCM_RESERVED_BATCH_IMPL_H
#define UCM_RESERVED_BATCH_IMPL_H
namespace UC::CacheStore {
class ReservedPrefetch::Batch final : public Early::Batch {
    ReservedPrefetch& owner_;
    std::weak_ptr<Reservation> reservation_;
    // Only accepted I/O keeps a cancelled reservation alive between Step calls.
    std::shared_ptr<Reservation> active_;
    const uint64_t handle_;
    mutable std::mutex mutex_;
    std::unique_ptr<EarlyStats> stats_;
    Status status_{Status::OK()};
    Early::Progress progress_;
    bool started_{false};
    bool discard_{false};
    bool activated_{false};
    bool cutoff_{false};
    size_t cursor_{0};
    size_t prefixShards_{0};  // capacity cutoff, kept separate until accepted I/O drains
    std::deque<size_t> waiting_;
    struct Transfer {
        Detail::TaskHandle handle{0};
        std::vector<size_t> owners;
    };
    std::deque<Transfer> transfers_;

    void Fail(Status status)
    { if (status_.Success() && status.Failure()) { status_ = std::move(status); } }

    void Complete(Reservation& r)
    {
        auto& transfer = transfers_.front();
        EarlyStats::Phase phase{stats_->waitMs};
        // Check(true) is followed by native Wait to collect status and resources.
        // On abandonment/error Run/destruction also drains unfinished tasks here.
        auto status = CompleteEarlyRead(*owner_.backend_, transfer.handle, transfer.owners,
            [&](size_t index) -> TransBuffer::Handle& { return r.Pin(index, owner_.shardsPerBlock_); });
        if (status.Success()) { progress_.completedReadBytes += transfer.owners.size() * shardBytes; }
        progress_.inflightBytes -= transfer.owners.size() * shardBytes;
        progress_.remainingL2Work -= std::min(progress_.remainingL2Work, transfer.owners.size());
        stats_->lastIoNs = EarlyStats::NowNs();
        transfers_.pop_front();
        progress_.advanced = true;
        Fail(status);
    }

    Expected<Early::Progress> Advance(size_t availableBytes, bool blocking)
    {
        progress_.advanced = false;
        if (progress_.done) {
            if (status_.Failure()) { return status_; }
            return Early::Progress{progress_};
        }
        auto r = active_ ? active_ : reservation_.lock();
        if (!r) { return Status::Error("reserved prefetch cancelled"); }
        if (!started_) {
            std::lock_guard lock(r->mutex);
            if (r->started) { return Status::Error("reservation already prefetched"); }
            stats_ = std::make_unique<EarlyStats>("prefetch");
            stats_->handle = handle_;
            r->started = started_ = true;
        }
        if (r->deferred && !activated_) { return Status::Error("deferred batch not activated"); }
        if (cutoff_) {
            prefixShards_ = cursor_ / owner_.shardsPerBlock_ * owner_.shardsPerBlock_;
            progress_.result.capacityTruncated = true;
            progress_.waitingCapacity = false;
        }
        if (progress_.waitingCapacity &&
            EarlyBufferPins::ReleaseEpoch(*owner_.buffer_) != progress_.waitEpoch) {
            progress_.waitingCapacity = false;
        }
        if (r->cancelled.load()) { Fail(Status::Error("reserved prefetch cancelled")); }
        if (!healthy_()) { Fail(Status::StoreUnhealthy()); }
        while (!transfers_.empty()) {
            if (blocking && status_.Success() && !discard_ && progress_.inflightBytes < windowBytes &&
                cursor_ < prefixShards_) { break; }
            if (!blocking) {
                auto check = owner_.backend_->Check(transfers_.front().handle);
                if (check && !check.Value()) { break; }
                if (!check) {
                    // Unknown completion must not block the data executor or
                    // release destinations. Retain this batch for later drain.
                    if (status_.Success()) {
                        UC_ERROR("UCM_EARLY completion_unknown reservation={} read={}; "
                                 "retaining pins/lease until proven completion or domain recovery",
                                 handle_, transfers_.front().handle);
                    }
                    Fail(check.Error());
                    progress_.canSubmit = false;
                    return Early::Progress{progress_};
                }
            }
            Complete(*r);
            if (blocking) { break; }
        }
        try {
            if (status_.Success() && !discard_) {
                Detail::TaskDesc task;
                task.brief = "EarlyL2ToL1";
                std::vector<size_t> owners;
                const size_t total = prefixShards_;
                const size_t end = std::min(total, cursor_ + size_t{64});
                // Bound localization per turn as well as submitted bytes. A
                // previous negative result is never trusted for allocation.
                {
                EarlyStats::Phase positioning{stats_->pinMs};
                while (!progress_.waitingCapacity && cursor_ < end && availableBytes >= shardBytes) {
                    if (r->cancelled.load()) { Fail(Status::Error("reserved prefetch cancelled")); break; }
                    auto& pin = r->Pin(cursor_, owner_.shardsPerBlock_);
                    const auto& block = r->blocks[cursor_ / owner_.shardsPerBlock_];
                    const auto shard = cursor_ % owner_.shardsPerBlock_;
                    bool allocated = false;
                    if (!pin) {
                        ++stats_->shards;
                        const auto epoch = EarlyBufferPins::ReleaseEpoch(*owner_.buffer_);
                        // Native zero reservable slots must not reach FetchNode's modulo.
                        if (protectionLimitBytes) { pin = owner_.buffer_->TryGet(block, shard); }
                        if (!pin) {
                            if (r->deferred && !cutoff_) {
                                progress_.waitingCapacity = true;
                                progress_.waitEpoch = epoch;
                            } else {
                                prefixShards_ = cursor_ / owner_.shardsPerBlock_ * owner_.shardsPerBlock_;
                                progress_.result.capacityTruncated = true;
                                progress_.advanced = true;
                            }
                            break;
                        }
                        allocated = pin.Owner() && !pin.Ready();
                        if (pin.Ready() && progress_.remainingL2Work) { --progress_.remainingL2Work; }
                    }
                    if (allocated) {
                        owners.push_back(cursor_);
                        task.push_back({block, shard, {pin.Data()}});
                        availableBytes -= shardBytes;
                    } else if (!pin.Ready()) {
                        waiting_.push_back(cursor_);
                        ++progress_.result.coalescedShards;
                    }
                    ++cursor_;
                    progress_.advanced = true;
                }
                }
                if (!task.empty() && status_.Success() && !r->cancelled.load()) {
                    // All allocating bookkeeping precedes Load. The reservation
                    // is retained before the backend can start writing to it.
                    transfers_.push_back({0, std::move(owners)});
                    active_ = r;
                    auto& transfer = transfers_.back();
                    if (stats_->firstIoMs < 0) { stats_->firstIoMs = EarlyStats::Milliseconds(stats_->elapsed); }
                    if (stats_->firstIoNs == 0) { stats_->firstIoNs = EarlyStats::NowNs(); }
                    const StopWatch submitting;
                    auto result = owner_.backend_->Load(std::move(task));
                    stats_->submitMs += EarlyStats::Milliseconds(submitting);
                    if (!result) { transfers_.pop_back(); Fail(result.Error()); }
                    else {
                        transfer.handle = result.Value();
                        const auto bytes = transfer.owners.size() * shardBytes;
                        progress_.result.l2ReadShards += transfer.owners.size();
                        progress_.result.l2ReadBytes += bytes;
                        progress_.inflightBytes += bytes;
                        ++stats_->ioBatches;
                        stats_->inflightPeakBytes = std::max(stats_->inflightPeakBytes, progress_.inflightBytes);
                    }
                }
                // Only peer-owned LOADING handles need state observation. Owner
                // completions above set READY directly; no full-prefix rescans.
                const auto checks = std::min(waiting_.size(), size_t{64});
                for (size_t i = 0; i < checks; ++i) {
                    const auto index = waiting_.front();
                    waiting_.pop_front();
                    if (index >= prefixShards_) { continue; }
                    auto& pin = r->Pin(index, owner_.shardsPerBlock_);
                    const auto state = pin.GetState();
                    if (state == TransBuffer::State::FAILED) { Fail(pin.FailureStatus()); }
                    else if (state != TransBuffer::State::READY) { waiting_.push_back(index); }
                    else {
                        progress_.advanced = true;
                        if (progress_.remainingL2Work) { --progress_.remainingL2Work; }
                    }
                }
                if (!waiting_.empty() && EarlyStats::Milliseconds(stats_->elapsed) >= owner_.timeoutMs_) {
                    Fail(Status::Error("shared shard readiness timeout"));
                }
            }
        } catch (const std::exception& error) {
            if (!transfers_.empty() && transfers_.back().handle == 0) { transfers_.pop_back(); }
            Fail(Status::Error(fmt::format("prefetch exception: {}", error.what())));
        }
        if (r->cancelled.load()) { Fail(Status::Error("reserved prefetch cancelled")); }
        if (transfers_.empty() && (status_.Failure() || discard_ ||
            (cursor_ >= prefixShards_ && waiting_.empty()))) {
            if (status_.Failure()) {
                for (auto& group : r->groups) {
                    for (auto& pin : group->pins) {
                        if (pin && pin.Owner() && !pin.Ready()) { pin.MarkFailed(status_); }
                    }
                }
            } else if (!discard_) {
                std::lock_guard lock(r->mutex);
                if (r->cancelled.load()) { Fail(Status::Error("reserved prefetch cancelled")); }
                else {
                    // Every submitted tail destination has drained. Only whole
                    // blocks survive; partial tail nodes remain native evictable cache.
                    r->blocks.resize(prefixShards_ / owner_.shardsPerBlock_);
                    r->groups.resize(r->blocks.size());
                    progress_.result.totalHitBlocks = r->blocks.size();
                    for (auto& group : r->groups) {
                        if (!group->ready) { group->ready = true; }
                    }
                    r->ready = true;
                }
            }
            if (status_.Success() && !discard_) { stats_->hintEntries = owner_.PublishReady(r); }
            progress_.discarded = discard_;
            progress_.done = true;
            progress_.remainingL2Work = 0;
            stats_->protectedBytes = owner_.ProtectedBytes(*r);
            stats_->protectionLimitBytes = protectionLimitBytes;
            stats_->success = status_.Success();
            stats_.reset();
        }
        if (transfers_.empty()) { active_.reset(); }
        progress_.canSubmit = status_.Success() && !discard_ && !progress_.done &&
            !progress_.waitingCapacity && cursor_ < prefixShards_;
        if (progress_.done && status_.Failure()) { return status_; }
        return Early::Progress{progress_};
    }

public:
    Batch(ReservedPrefetch& owner, const std::shared_ptr<Reservation>& r, uint64_t handle)
        : Early::Batch(owner.shardBytes_, 128 * owner.shardBytes_,
                       EarlyBufferPins::ReservableSlots(*owner.buffer_) * owner.shardBytes_),
          owner_(owner), reservation_(r), handle_(handle)
    {
        prefixShards_ = r->blocks.size() * owner.shardsPerBlock_;
        progress_.remainingL2Work = r->initialMissing;
        if (r->ready) { progress_.result.totalHitBlocks = r->blocks.size(); }
    }

    Expected<Early::Progress> Activate() override
    {
        std::lock_guard lock(mutex_);
        auto r = reservation_.lock();
        if (!r || r->cancelled.load()) { return Status::Error("reservation cancelled"); }
        if (!r->deferred || activated_ || started_) { return Status::Error("invalid activation"); }
        if (!healthy_()) { return Status::StoreUnhealthy(); }
        EarlyStats stats{"activate"};
        auto located = owner_.Locate(r->blocks.data(), r->blocks.size(), stats);
        if (!located) { return located.Error(); }
        auto groups = std::move(located.Value());
        // A formerly L1-only block may have been evicted. L2-positive results
        // remain protected by the existing lease, negative locations do not.
        std::vector<Detail::BlockId> uncertain;
        std::vector<size_t> indices;
        for (size_t i = 0; i < groups.size(); ++i) {
            if (!groups[i]->ready && !r->inL2[i]) {
                uncertain.push_back(r->blocks[i]); indices.push_back(i);
            }
        }
        if (!uncertain.empty()) {
            auto hit = owner_.backend_->LookupOnPrefix(uncertain.data(), uncertain.size());
            if (!hit) { return hit.Error(); }
            const auto miss = static_cast<size_t>(hit.Value() + 1);
            if (miss < indices.size()) {
                groups.resize(indices[miss]); r->blocks.resize(indices[miss]);
            }
        }
        if (r->cancelled.load()) { return Status::Error("reservation cancelled"); }
        r->groups = std::move(groups);
        r->initialMissing = 0;
        for (auto& group : r->groups) {
            for (auto& pin : group->pins) { if (!pin || !pin.Ready()) { ++r->initialMissing; } }
        }
        activated_ = true;
        prefixShards_ = r->blocks.size() * owner_.shardsPerBlock_;
        progress_.remainingL2Work = r->initialMissing;
        progress_.result.totalHitBlocks = r->blocks.size();
        r->ready = r->initialMissing == 0;
        if (r->ready) {
            progress_.done = true;
            progress_.canSubmit = false;
            owner_.PublishReady(r);
        }
        return Early::Progress{progress_};
    }
    void Limit(size_t count) override
    {
        std::lock_guard lock(mutex_);
        auto r = reservation_.lock();
        if (!r || !r->deferred || activated_ || started_ || count > r->blocks.size()) {
            throw std::runtime_error("invalid deferred prefix limit");
        }
        r->blocks.resize(count);
        r->inL2.resize(count);
        prefixShards_ = count * owner_.shardsPerBlock_;
    }
    void Cutoff() override
    {
        std::lock_guard lock(mutex_);
        cutoff_ = true;
    }

    size_t ProtectedBytes() const override
    {
        std::lock_guard lock(mutex_);
        auto r = active_ ? active_ : reservation_.lock();
        return r ? owner_.ProtectedBytes(*r) : 0;
    }

    ~Batch() override
    {
        if (!active_) { return; }
        // Abandoned capability: accepted I/O cannot outlive destination pins or
        // leases. Normal worker cancellation keeps stepping to avoid blocking it.
        Fail(Status::Error("prefetch batch abandoned"));
        while (!transfers_.empty()) { Complete(*active_); }
        (void)Advance(0, true);
    }

    Expected<Early::Progress> Step(size_t availableBytes) override
    {
        std::lock_guard lock(mutex_);
        return Advance(std::min(availableBytes, windowBytes - progress_.inflightBytes), false);
    }

    void Discard() override
    {
        std::lock_guard lock(mutex_);
        discard_ = true;
    }

    Early::Progress Snapshot() override
    {
        std::lock_guard lock(mutex_);
        return progress_;
    }

    Expected<PrefetchResult> Run() override
    {
        std::lock_guard lock(mutex_);
        if (started_) { return Status::Error("reservation already prefetched"); }
        for (;;) {
            auto next = Advance(windowBytes - progress_.inflightBytes, true);
            if (!next) { return next.Error(); }
            if (next.Value().done) { return PrefetchResult{next.Value().result}; }
            if (!next.Value().advanced) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        }
    }
};

inline std::shared_ptr<Early::Batch> ReservedPrefetch::MakeBatch(
    const std::shared_ptr<Reservation>& r, uint64_t handle)
{ return std::make_shared<Batch>(*this, r, handle); }
}  // namespace UC::CacheStore
#endif
