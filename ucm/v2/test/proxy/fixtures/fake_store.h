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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "status/status.h"
#include "store.h"
#include "store_entry.h"
#include "type/block.h"
#include "type/request.h"
#include "type/token.h"

namespace UC::Proxy {

using UC::Store::JobDescriptor;

struct RecordedJob {
    std::size_t jobId;
    JobDescriptor job;
    AddressedToken token;
};

inline AddressedToken MakeAddressedToken(std::size_t tag)
{
    AddressedToken token;
    token.payload.push_back(static_cast<std::byte>(tag));
    return token;
}

class FakeStore : public Store::Store, public Store::Prefetchable {
public:
    /* tokened_: true mints payload-bearing tokens (real addressing); false
     * mints keys-only tokens, emulating the base Store's default addressing
     * while still recording every call. */
    explicit FakeStore(bool prefetchable, bool tokened)
        : prefetchable_(prefetchable), tokened_(tokened)
    {
    }

    /*== Store ==*/

    Status Setup(const Dictionary& /*config*/) override { return Status::Ok(); }

    Expected<std::vector<std::uint8_t>> Lookup(const std::vector<BlockId>& blockIds) override
    {
        plainLookups.push_back(blockIds);
        if (plainLookupFails) { return Status::NotFound("plain lookup failed"); }
        if (!plainHits.empty()) { return plainHits; }
        return std::vector<std::uint8_t>(blockIds.size(), 1);
    }

    void Touch(const std::vector<BlockId>& blockIds, const RequestContext& context) override
    {
        touches.push_back({blockIds, context});
    }

    Status Load(std::size_t jobId, JobDescriptor job, AddressedToken token) override
    {
        loads.push_back({jobId, std::move(job), std::move(token)});
        return loadStatus;
    }

    Status Dump(std::size_t jobId, JobDescriptor job, AddressedToken token) override
    {
        dumps.push_back({jobId, std::move(job), std::move(token)});
        return dumpStatus;
    }

    std::vector<std::pair<std::size_t, bool>> GetFinished() override
    {
        ++getFinishedCalls;
        auto finished = finishedQueue;
        finishedQueue.clear();
        return finished;
    }

    std::vector<std::pair<std::size_t, bool>> Wait(const std::vector<std::size_t>& jobIds) override
    {
        waits.push_back(jobIds);
        std::vector<std::pair<std::size_t, bool>> results;
        for (const auto& jobId : jobIds) {
            for (auto it = finishedQueue.begin(); it != finishedQueue.end(); ++it) {
                if (it->first == jobId) {
                    results.push_back(*it);
                    finishedQueue.erase(it);
                    break;
                }
            }
        }
        return results;
    }

    Prefetchable* AsPrefetchable() override { return prefetchable_ ? this : nullptr; }

    /*== Addressing & admission ==*/

    Expected<AddressedToken> Pin(const std::vector<BlockId>& blockIds) override
    {
        pins.push_back(blockIds);
        if (pinFails) { return Status::NotFound("pin failed"); }
        AddressedToken token = tokened_ ? MakeAddressedToken(blockIds.size()) : AddressedToken{};
        token.keys = blockIds;
        return token;
    }

    void Unpin(const std::vector<BlockId>& blockIds) override { unpins.push_back(blockIds); }

    Expected<AddressedToken> Alloc(const std::vector<BlockId>& blockIds,
                                   const RequestContext& context) override
    {
        allocs.push_back({blockIds, context});
        if (allocFails) { return Status::NoSpace("alloc failed"); }
        AddressedToken token = tokened_ ? MakeAddressedToken(blockIds.size()) : AddressedToken{};
        token.keys = blockIds;
        return token;
    }

    void Commit(const std::vector<BlockId>& blockIds, bool success) override
    {
        commits.push_back({blockIds, success});
    }

    /*== Prefetchable ==*/

    Expected<std::optional<std::vector<std::uint8_t>>> Lookup(
        const RequestId& requestId, const std::vector<BlockId>& blockIds) override
    {
        prefetchLookups.push_back({requestId, blockIds});
        if (prefetchLookupFails) { return Status::NotFound("prefetch lookup failed"); }
        if (signalPrefetchable) { return std::nullopt; }
        return std::vector<std::uint8_t>(blockIds.size(), 1);
    }

    Expected<AddressedToken> PreparePrefetch(const RequestId& requestId,
                                             const RequestContext& context) override
    {
        preparedPrefetches.push_back({requestId, context});
        AddressedToken token;
        token.keys =
            prefetchLookups.empty() ? std::vector<BlockId>{} : prefetchLookups.back().second;
        token.payload = {std::byte{1}, std::byte{2}};
        return token;
    }

    Status CompletePrefetch(const AddressedToken& plan, bool success) override
    {
        completedPrefetches.push_back({plan, success});
        return Status::Ok();
    }

    Status Prefetch(std::size_t jobId, AddressedToken plan) override
    {
        prefetches.push_back({jobId, std::move(plan)});
        return prefetchStatus;
    }

    /*== Test knobs ==*/

    bool prefetchable_ = true;
    bool tokened_ = true;
    bool plainLookupFails = false;
    bool prefetchLookupFails = false;
    bool signalPrefetchable = false;
    bool pinFails = false;
    bool allocFails = false;
    Status loadStatus = Status::Ok();
    Status dumpStatus = Status::Ok();
    Status prefetchStatus = Status::Ok();
    std::size_t getFinishedCalls = 0;
    std::vector<std::uint8_t> plainHits;
    std::vector<std::pair<std::size_t, bool>> finishedQueue;

    std::vector<std::vector<BlockId>> plainLookups;
    std::vector<std::pair<std::vector<BlockId>, RequestContext>> touches;
    std::vector<RecordedJob> loads;
    std::vector<RecordedJob> dumps;
    std::vector<std::vector<std::size_t>> waits;
    std::vector<std::vector<BlockId>> pins;
    std::vector<std::vector<BlockId>> unpins;
    std::vector<std::pair<std::vector<BlockId>, RequestContext>> allocs;
    std::vector<std::pair<std::vector<BlockId>, bool>> commits;
    std::vector<std::pair<RequestId, std::vector<BlockId>>> prefetchLookups;
    std::vector<std::pair<RequestId, RequestContext>> preparedPrefetches;
    std::vector<std::pair<AddressedToken, bool>> completedPrefetches;
    std::vector<std::pair<std::size_t, AddressedToken>> prefetches;
};

struct EntryHandle {
    FakeStore* fake;
    std::unique_ptr<StoreEntry> entry;
};

inline EntryHandle MakeEntry(bool prefetchable, bool tokened)
{
    auto* fake = new FakeStore(prefetchable, tokened);
    EntryHandle handle;
    handle.fake = fake;
    handle.entry =
        std::make_unique<StoreEntry>(LoadedStore::Adopt(std::unique_ptr<Store::Store>(fake)));
    return handle;
}

inline BlockId MakeKey(std::uint8_t value)
{
    BlockId id;
    id.bytes[0] = static_cast<std::byte>(value);
    return id;
}

inline std::vector<BlockId> MakeKeys(std::uint8_t first, std::uint8_t count)
{
    std::vector<BlockId> keys;
    for (std::uint8_t i = 0; i < count; i++) { keys.push_back(MakeKey(first + i)); }
    return keys;
}

inline RequestContext MakeContext(const RequestId& requestId)
{
    RequestContext context;
    context.requestId = requestId;
    context.ttl = std::chrono::milliseconds(500);
    context.totalBlockCount = 3;
    return context;
}

inline BlockPayload MakePayload(std::size_t size)
{
    BlockPayload payload;
    payload.pointers = {nullptr};
    payload.sizes = {size};
    return payload;
}

inline bool Contains(const std::vector<IoResult>& results, std::size_t jobId, bool success)
{
    for (const auto& result : results) {
        if (result.jobId == jobId && result.success == success) { return true; }
    }
    return false;
}

}  // namespace UC::Proxy
