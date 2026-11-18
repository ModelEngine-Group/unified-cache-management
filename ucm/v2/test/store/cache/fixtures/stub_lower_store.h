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
#include <optional>
#include <utility>
#include <vector>
#include "status/status.h"
#include "store.h"
#include "type/block.h"
#include "type/request.h"
#include "type/token.h"

namespace UC::Store {

/** One Dump the stub was asked to fly: its job id, the keys, and the
 *  flattened sources and sizes as the relay built them. */
struct StubDumpCall {
    std::size_t jobId;              /* The submitted sub-batch id. */
    std::vector<BlockId> keys;      /* The sub-batch's blocks. */
    std::vector<void*> sources;     /* The flattened source pointers. */
    std::vector<std::size_t> sizes; /* The flattened source sizes. */
};

/** One Load the stub was asked to fly: its job id, the keys, and the
 *  flattened destinations and sizes as the relay built them. */
struct StubLoadCall {
    std::size_t jobId;               /* The submitted sub-batch id. */
    std::vector<BlockId> keys;       /* The sub-batch's blocks. */
    std::vector<void*> destinations; /* The flattened destination pointers. */
    std::vector<std::size_t> sizes;  /* The flattened destination sizes. */
};

/**
 * @brief Scriptable stand-in for the wired lower tier: records the
 *        admission and read protocols a relay drives (Alloc, Dump,
 *        Commit; Pin, Load, Unpin) and settles completions on command
 *        instead of moving any bytes.
 *
 * The records are public and in call order: allocs holds every
 * requested reservation's keys, dumps every Dump submission (admitted
 * or refused, flattened to its source pointers and sizes), commits
 * every concluded Commit with its verdict; pins holds every requested
 * pin's keys, loads every Load submission (admitted or refused,
 * flattened to its destination pointers and sizes), unpins every
 * Unpin's keys. Completions are scripted: Complete() parks a verdict
 * for the next GetFinished() drain; FailAllocNth()/FailDumpNth()/
 * FailPinNth()/FailLoadNth() turn the n-th call of that entry point
 * from now on into the given error; SettleOnWait() has Wait() conclude
 * a still-flying job on the spot instead of blocking, so a
 * single-threaded test can drive a blocking pump path.
 */
class StubLowerStore final : public Store {
public:
    /* Keys of every requested reservation, in call order. */
    std::vector<std::vector<BlockId>> allocs;
    /* Every Dump submission, admitted or refused. */
    std::vector<StubDumpCall> dumps;
    /* Every concluded Commit, with its verdict. */
    std::vector<std::pair<std::vector<BlockId>, bool>> commits;
    /* Keys of every requested pin, in call order. */
    std::vector<std::vector<BlockId>> pins;
    /* Every Load submission, admitted or refused. */
    std::vector<StubLoadCall> loads;
    /* Keys of every Unpin, in call order. */
    std::vector<std::vector<BlockId>> unpins;

    /** Makes the n-th Alloc from now on answer the given error instead of
     *  reserving. */
    void FailAllocNth(std::size_t nth, Status error)
    {
        failAllocNth_ = nth;
        failAlloc_ = std::move(error);
    }

    /** Makes the n-th Dump from now on answer the given error instead of
     *  flying. */
    void FailDumpNth(std::size_t nth, Status error)
    {
        failDumpNth_ = nth;
        failDump_ = std::move(error);
    }

    /** Makes the n-th Pin from now on answer the given error instead of
     *  pinning. */
    void FailPinNth(std::size_t nth, Status error)
    {
        failPinNth_ = nth;
        failPin_ = std::move(error);
    }

    /** Makes the n-th Load from now on answer the given error instead of
     *  flying. */
    void FailLoadNth(std::size_t nth, Status error)
    {
        failLoadNth_ = nth;
        failLoad_ = std::move(error);
    }

    /** Parks the job's verdict for the next GetFinished() drain. */
    void Complete(std::size_t jobId, bool ok)
    {
        for (auto it = inFlight_.begin(); it != inFlight_.end(); ++it) {
            if (*it == jobId) {
                inFlight_.erase(it);
                finished_.emplace_back(jobId, ok);
                return;
            }
        }
    }

    /** Has Wait() conclude a still-flying job on the spot with the given
     *  verdict, instead of blocking. */
    void SettleOnWait(bool ok)
    {
        settleOnWait_ = true;
        settleVerdict_ = ok;
    }

    Status Setup(const Dictionary&) override { return Status::Ok(); }

    Expected<std::vector<std::uint8_t>> Lookup(const std::vector<BlockId>&) override
    {
        return Status::Unsupported("the stub lower tier is not probeable");
    }

    Expected<AddressedToken> Alloc(const std::vector<BlockId>& blockIds,
                                   const RequestContext&) override
    {
        allocs.push_back(blockIds);
        if (failAllocNth_ != 0 && --failAllocNth_ == 0) { return *failAlloc_; }
        AddressedToken token;
        token.keys = blockIds; /* Keys-only addressing: the stub resolves ids on its own. */
        return token;
    }

    Expected<AddressedToken> Pin(const std::vector<BlockId>& blockIds) override
    {
        pins.push_back(blockIds);
        if (failPinNth_ != 0 && --failPinNth_ == 0) { return *failPin_; }
        AddressedToken token;
        token.keys = blockIds;
        return token;
    }

    Status Dump(std::size_t jobId, JobDescriptor job, AddressedToken) override
    {
        StubDumpCall call;
        call.jobId = jobId;
        call.keys = std::move(job.blockIds);
        for (const BlockPayload& payload : job.payloads) {
            for (std::size_t i = 0; i < payload.pointers.size(); ++i) {
                call.sources.push_back(payload.pointers[i]);
                call.sizes.push_back(payload.sizes[i]);
            }
        }
        dumps.push_back(std::move(call));
        if (failDumpNth_ != 0 && --failDumpNth_ == 0) { return *failDump_; }
        inFlight_.push_back(jobId);
        return Status::Ok();
    }

    void Commit(const std::vector<BlockId>& blockIds, bool success) override
    {
        commits.emplace_back(blockIds, success);
    }

    void Unpin(const std::vector<BlockId>& blockIds) override { unpins.push_back(blockIds); }

    Status Load(std::size_t jobId, JobDescriptor job, AddressedToken) override
    {
        StubLoadCall call;
        call.jobId = jobId;
        call.keys = std::move(job.blockIds);
        for (const BlockPayload& payload : job.payloads) {
            for (std::size_t i = 0; i < payload.pointers.size(); ++i) {
                call.destinations.push_back(payload.pointers[i]);
                call.sizes.push_back(payload.sizes[i]);
            }
        }
        loads.push_back(std::move(call));
        if (failLoadNth_ != 0 && --failLoadNth_ == 0) { return *failLoad_; }
        inFlight_.push_back(jobId);
        return Status::Ok();
    }

    std::vector<std::pair<std::size_t, bool>> GetFinished() override
    {
        std::vector<std::pair<std::size_t, bool>> out{finished_.begin(), finished_.end()};
        finished_.clear();
        return out;
    }

    Status Wait(std::size_t jobId) override
    {
        for (auto it = finished_.begin(); it != finished_.end(); ++it) {
            if (it->first == jobId) {
                const bool ok = it->second;
                finished_.erase(it);
                return Verdict(jobId, ok);
            }
        }
        if (settleOnWait_) {
            for (auto it = inFlight_.begin(); it != inFlight_.end(); ++it) {
                if (*it == jobId) {
                    inFlight_.erase(it);
                    return Verdict(jobId, settleVerdict_);
                }
            }
        }
        return Status::InvalidParam("stub lower store: job {} is unknown or already consumed",
                                    jobId);
    }

private:
    static Status Verdict(std::size_t jobId, bool ok)
    {
        return ok ? Status::Ok() : Status::General("stub lower store: job {} failed", jobId);
    }

    std::vector<std::size_t> inFlight_;                  /* Dumped or loaded, not yet settled. */
    std::vector<std::pair<std::size_t, bool>> finished_; /* Settled, not yet drained. */
    std::size_t failAllocNth_{0};     /* Alloc calls left before the scripted error fires. */
    std::optional<Status> failAlloc_; /* The scripted Alloc error. */
    std::size_t failDumpNth_{0};      /* Dump calls left before the scripted error fires. */
    std::optional<Status> failDump_;  /* The scripted Dump error. */
    std::size_t failPinNth_{0};       /* Pin calls left before the scripted error fires. */
    std::optional<Status> failPin_;   /* The scripted Pin error. */
    std::size_t failLoadNth_{0};      /* Load calls left before the scripted error fires. */
    std::optional<Status> failLoad_;  /* The scripted Load error. */
    bool settleOnWait_{false};        /* Wait() settles a still-flying job instead of blocking. */
    bool settleVerdict_{true};        /* The verdict an on-wait settle concludes with. */
};

}  // namespace UC::Store
