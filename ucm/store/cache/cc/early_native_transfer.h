#ifndef UCM_EARLY_NATIVE_TRANSFER_H
#define UCM_EARLY_NATIVE_TRANSFER_H
#include <condition_variable>
#include <deque>
#include <unordered_set>
#include "early_read_completion.h"
#include "trans_manager.h"

namespace UC::CacheStore {
struct EarlyNativeTask {
    TransTask task;
    const Detail::TaskHandle id;
    // Extra references cover native error paths whose stream drain is unproven.
    // These are plain handles: they never claim ownership or publish FAILED.
    std::vector<TransBuffer::Handle> guards;
    std::vector<size_t> owners, followers;
    size_t cursor{0};
    struct Read { Detail::TaskHandle id; std::vector<size_t> indices; };
    std::deque<Read> reads;
    std::shared_ptr<ReadLease> lease;
    bool submitted{false}, quarantined{false};
    Status status{Status::OK()};
    explicit EarlyNativeTask(TransTask prepared)
        : task(std::move(prepared)), id(task.id),
          guards(task.loadHandles.begin(), task.loadHandles.end())
    {
        if (task.type == TransTask::Type::LOAD) {
            for (size_t i = 0; i < guards.size(); ++i) {
                if (!guards[i].Ready()) {
                    (guards[i].Owner() ? owners : followers).push_back(i);
                }
            }
        }
    }
    void Fail(Status error) { if (status.Success()) { status = std::move(error); } }
};

// Owns no buffer, queue, or shared identity. Prepares only the handles acquired
// by EarlyCacheStore, then delegates the original complete task to TransManager.
class EarlyNativeTransfer final : public Detail::TaskWrapper<EarlyNativeTask, Detail::TaskHandle> {
    StoreV1& backend_;
    TransManager& native_;
    const size_t shardBytes_;
    std::mutex pendingMutex_;
    std::condition_variable wake_;
    std::deque<TaskPair> pending_;
    std::vector<TaskPtr> quarantined_;
    bool stopping_{false};
    size_t inflight_{0};
    std::thread thread_;

    void Dispatch(TaskPtr task, WaiterPtr waiter) override
    {
        waiter->Set(1);
        // The all-READY path still submits immediately, without data queueing.
        if (task->owners.empty() && task->followers.empty()) { SubmitNative(*task); }
        {
            std::lock_guard lock(pendingMutex_);
            pending_.emplace_back(std::move(task), std::move(waiter));
        }
        wake_.notify_one();
    }
    void Cancel(TaskPtr) override { wake_.notify_one(); } // failureSet_ stops new reads
    Status FailureStatus(const TaskPtr& task) const override { return task->status; }

    void SubmitNative(EarlyNativeTask& t)
    {
        auto submitted = native_.Submit(std::move(t.task));
        if (!submitted) { t.Fail(submitted.Error()); }
        else { t.submitted = true; }
    }

    bool Advance(EarlyNativeTask& t, bool stopping)
    {
        if (stopping || failureSet_.Contains(t.id)) { t.Fail(Status::Timeout()); }
        // Inspect every accepted read: no FIFO dependency between owners.
        for (auto it = t.reads.begin(); it != t.reads.end();) {
            auto check = backend_.Check(it->id);
            if (!check) { t.Fail(check.Error()); ++it; continue; }
            if (!check.Value()) { ++it; continue; }
            auto status = CompleteEarlyRead(backend_, it->id, it->indices,
                [&](size_t i) -> TransBuffer::Handle& { return t.guards[i]; });
            inflight_ -= it->indices.size();
            if (status.Failure()) { t.Fail(status); }
            it = t.reads.erase(it);
        }
        if (t.submitted) {
            auto check = native_.Check(t.id);
            if (!check) {
                t.Fail(check.Error());
                t.quarantined = true;
                return true;
            }
            if (!check.Value()) { return false; }
            auto status = native_.Wait(t.id);
            if (status.Failure()) {
                t.Fail(status);
                // A failed H2D/D2H task latch does not prove stream completion.
                t.quarantined = true;
            }
            return true;
        }
        if (t.status.Failure()) { return t.reads.empty(); }
        if (!t.lease && !t.owners.empty()) {
            std::unordered_set<Detail::BlockId, Detail::BlockIdHasher> unique;
            for (auto i : t.owners) { unique.insert(t.task.desc[i].owner); }
            std::vector<Detail::BlockId> blocks(unique.begin(), unique.end());
            auto lease = backend_.AcquireReadLease(blocks.data(), blocks.size());
            if (!lease) { t.Fail(lease.Error()); return true; }
            t.lease = std::move(lease.Value());
        }
        if (t.cursor < t.owners.size() && inflight_ < 128) {
            const auto count = std::min({size_t{64}, size_t{128} - inflight_,
                                         t.owners.size() - t.cursor});
            Detail::TaskDesc desc;
            desc.brief = "EarlyNativeRead";
            std::vector<size_t> indices;
            indices.reserve(count);
            desc.reserve(count);
            for (size_t n = 0; n < count; ++n) {
                auto i = t.owners[t.cursor + n];
                const auto& shard = t.task.desc[i];
                indices.push_back(i);
                desc.push_back({shard.owner, shard.index, {t.guards[i].Data()}});
            }
            // Allocate bookkeeping before Load can accept any I/O.
            t.reads.push_back({0, std::move(indices)});
            auto read = backend_.Load(std::move(desc));
            if (!read) { t.reads.pop_back(); t.Fail(read.Error()); }
            else {
                t.reads.back().id = read.Value();
                t.cursor += count;
                inflight_ += count;
                UC_DEBUG("UCM_EARLY native_prepare task={} read={} shards={} inflight_bytes={}",
                         t.id, read.Value(), count, inflight_ * shardBytes_);
            }
        }
        // Followers never obstruct submission or completion of our own reads.
        for (auto it = t.followers.begin(); it != t.followers.end();) {
            const auto state = t.guards[*it].GetState();
            if (state == TransBuffer::State::FAILED) {
                t.Fail(t.guards[*it].FailureStatus()); break;
            }
            if (state == TransBuffer::State::READY) { it = t.followers.erase(it); }
            else { ++it; }
        }
        if (t.status.Failure()) { return t.reads.empty(); }
        if (t.cursor == t.owners.size() && t.reads.empty() && t.followers.empty()) {
            t.lease.reset(); // physical reads completed; L1 guards cover H2D
            SubmitNative(t);
            if (!t.submitted) { return true; }
        }
        return false;
    }

    void Run()
    {
        std::deque<TaskPair> active;
        for (;;) {
            bool stopping;
            {
                std::unique_lock lock(pendingMutex_);
                if (active.empty() && pending_.empty() && !stopping_) {
                    wake_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                }
                while (!pending_.empty()) {
                    active.push_back(std::move(pending_.front())); pending_.pop_front();
                }
                stopping = stopping_;
            }
            if (stopping && active.empty()) { break; }
            bool progress = false;
            const auto count = active.size();
            for (size_t n = 0; n < count; ++n) {
                auto [task, waiter] = std::move(active.front()); active.pop_front();
                if (timeoutMs_ && waiter->IsTimeout(timeoutMs_)) { task->Fail(Status::Timeout()); }
                if (Advance(*task, stopping)) {
                    if (task->status.Failure()) { failureSet_.Insert(task->id); }
                    if (task->quarantined) {
                        UC_ERROR("UCM_EARLY native task={} drain unproven; retaining {} pins; "
                                 "shared domain restart required", task->id, task->guards.size());
                        quarantined_.push_back(task);
                    } else {
                        if (task->status.Failure() && !task->submitted) {
                            for (auto& pin : task->guards) {
                                if (pin.Owner() && pin.GetState() == TransBuffer::State::LOADING) {
                                    pin.MarkFailed(task->status);
                                }
                            }
                        }
                        task->task.loadHandles.clear();
                        task->guards.clear();
                        task->lease.reset();
                    }
                    waiter->Done();
                    progress = true;
                } else { active.emplace_back(std::move(task), std::move(waiter)); }
            }
            if (!progress) {
                std::unique_lock lock(pendingMutex_);
                wake_.wait_for(lock, std::chrono::milliseconds(1));
            }
        }
    }

public:
    EarlyNativeTransfer(StoreV1& backend, TransManager& native, const Config& config)
        : backend_(backend), native_(native), shardBytes_(config.shardSize)
    {
        timeoutMs_ = config.timeoutMs;
        thread_ = std::thread([this] { Run(); });
    }
    ~EarlyNativeTransfer()
    {
        { std::lock_guard lock(pendingMutex_); stopping_ = true; }
        wake_.notify_one();
        thread_.join();
        if (!quarantined_.empty()) {
            UC_ERROR("UCM_EARLY shutdown blocked: native stream drain unknown; "
                     "stop all shared-domain workers before resetting this domain");
            // Do not destroy the native buffer beneath unproven DMA. Only an
            // external whole-domain/process recovery may cross this boundary.
            std::unique_lock lock(pendingMutex_);
            wake_.wait(lock, [] { return false; });
        }
    }
};
}
#endif
