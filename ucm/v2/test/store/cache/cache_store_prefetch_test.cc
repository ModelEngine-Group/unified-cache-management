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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>
#include "cache/cache_store.h"
#include "cache/token_codec.h"
#include "fixtures/stub_lower_store.h"
#include "prefetchable.h"
#include "status/status.h"
#include "type/block.h"
#include "type/dictionary.h"
#include "type/token.h"

namespace UC::Store {

/*
 * Rendezvous path helpers are defined in bootstrap.cc with external linkage
 * but declared in no header; the signatures must stay identical (mismatch
 * turns into a link error, never silent drift).
 */
[[nodiscard]] std::string LockPathOf(const std::string& uniqueId);
[[nodiscard]] std::string SocketPathOf(const std::string& uniqueId);

namespace {

/*
 * One gigabyte per domain -- the capacity floor -- with 256MB blocks yields
 * exactly four pool slots, the geometry the multiprocess suite runs.
 */
constexpr std::int64_t kBlockSize = 268435456;

BlockId MakeBlock(std::uint64_t tag)
{
    BlockId id;
    std::memcpy(id.bytes.data(), "ost", 3);
    std::memcpy(id.bytes.data() + 8, &tag, sizeof(tag));
    return id;
}

/** A plan token in the engine half's encoding: the receipt handle plus
 *  one destination slot per key. */
AddressedToken MakePlan(const std::vector<BlockId>& ids, const std::vector<std::uint32_t>& slots,
                        std::uint64_t batchId = 1)
{
    AddressedToken plan;
    plan.keys = ids;
    plan.payload = EncodePlanPayload(PlanPayload{batchId, slots});
    return plan;
}

Dictionary MakeDomainDict(const std::string& uniqueId, const char* role, Store* lower)
{
    Dictionary dict;
    dict.Set("unique_id", uniqueId);
    dict.Set("llm_role", std::string(role));
    dict.SetInt64("worker_number", 1);
    dict.SetInt64("worker_index", 0);
    dict.SetInt64("device_id", 0);
    dict.SetInt64("block_size", kBlockSize);
    dict.SetInt64("cache_buffer_capacity_gb", 1);
    dict.Set("cache_replacement_policy", std::string("lru"));
    if (lower != nullptr) {
        /* The assembler stores a Store*, never the derived pointer. */
        dict.Set("__ucm_lower_store__", lower);
    }
    return dict;
}

/** One Prefetchable wiring of the cache store, over one in-process domain per
 *  unique id; the fixture unlinks each domain's rendezvous artifacts at the
 *  end of its test. */
class CacheStorePrefetchTest : public ::testing::Test {
protected:
    void TearDown() override
    {
        for (const std::string& id : usedIds_) {
            ::unlink(LockPathOf(id).c_str());
            ::unlink(SocketPathOf(id).c_str());
        }
    }

    /** Unique per test process and per call, so parallel runs never collide. */
    std::string FreshId()
    {
        std::string id = "gtest-prefetch-";
        id += std::to_string(::getpid());
        id += "-";
        id += std::to_string(nextId_++);
        usedIds_.push_back(id);
        return id;
    }

    /** Brings one role up over the given domain; a failed Setup answers
     *  nullptr, already reported. */
    std::unique_ptr<CacheStore> MakeStore(const std::string& uniqueId, const char* role,
                                          Store* lower)
    {
        auto store = std::make_unique<CacheStore>();
        const Status setup = store->Setup(MakeDomainDict(uniqueId, role, lower));
        if (setup.Failure()) {
            ADD_FAILURE() << "setup of role " << role << " failed: " << setup.ToString();
            return nullptr;
        }
        return store;
    }

    std::vector<std::string> usedIds_;
    std::uint32_t nextId_{0};
};

/*== The executor half, as served by the worker role ==*/

TEST_F(CacheStorePrefetchTest, WorkerExposesTheExecutorHalfWhenWired)
{
    StubLowerStore lower;
    std::unique_ptr<CacheStore> leader = MakeStore(FreshId(), "worker", &lower);
    ASSERT_NE(leader, nullptr);
    Prefetchable* facet = leader->AsPrefetchable();
    ASSERT_NE(facet, nullptr);

    /* A leaf of the same domain, identically wired: its facet appears too. */
    const std::string id = usedIds_.back();
    std::unique_ptr<CacheStore> leaf = MakeStore(id, "worker", &lower);
    ASSERT_NE(leaf, nullptr);
    EXPECT_NE(leaf->AsPrefetchable(), nullptr);

    /* The engine half is structural idle: no engine lives in a worker, so
     * the lower tier stays untouched through its whole call set. */
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2)};
    EXPECT_EQ(facet->Lookup("r1", ids).Error().Underlying(), Status::Unsupported().Underlying());
    auto idle = facet->PreparePrefetch(ids, RequestContext{});
    ASSERT_TRUE(idle.HasValue());
    EXPECT_TRUE(idle.Value().keys.empty());
    facet->CompletePrefetch(AddressedToken{}, true);
    facet->ReleaseRequest("r1");
    EXPECT_TRUE(lower.lookups.empty());

    /* The executor half drives the wired tier's read protocol, and the
     * outer job shares the store's wait. */
    EXPECT_EQ(facet->Prefetch(9, MakePlan(ids, {0, 1})).Underlying(), 0);
    ASSERT_EQ(lower.pins.size(), 1U);
    EXPECT_EQ(lower.pins[0], ids);
    ASSERT_EQ(lower.loads.size(), 1U);
    EXPECT_EQ(lower.loads[0].jobId, 1U);
    EXPECT_EQ(lower.loads[0].keys, ids);

    lower.Complete(1, true);
    EXPECT_EQ(leader->Wait(9).Underlying(), 0);
    EXPECT_EQ(lower.unpins.size(), 1U);
    EXPECT_EQ(lower.unpins[0], ids);
    EXPECT_TRUE(leader->GetFinished().empty());
    EXPECT_EQ(leader->Wait(9).Underlying(), Status::InvalidParam().Underlying());
}

/*== The engine half, as served by the scheduler role ==*/

TEST_F(CacheStorePrefetchTest, SchedulerServesTheEngineHalfLocally)
{
    StubLowerStore lower;
    lower.present.insert(MakeBlock(1));
    std::unique_ptr<CacheStore> worker = MakeStore(FreshId(), "worker", &lower);
    ASSERT_NE(worker, nullptr);

    const std::string id = usedIds_.back();
    std::unique_ptr<CacheStore> scheduler = MakeStore(id, "scheduler", &lower);
    ASSERT_NE(scheduler, nullptr);
    Prefetchable* facet = scheduler->AsPrefetchable();
    ASSERT_NE(facet, nullptr);

    /* The probe answers the engine's snapshot beside the lower tier's
     * signal, and reaches the wired tier's lookup. */
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2)};
    auto probe = facet->Lookup("r1", ids);
    ASSERT_TRUE(probe.HasValue());
    EXPECT_EQ(probe.Value().first, std::vector<std::uint8_t>({0, 0}));
    EXPECT_TRUE(probe.Value().second);
    ASSERT_EQ(lower.lookups.size(), 1U);
    EXPECT_EQ(lower.lookups[0], ids);

    /* The mint: planned keys, one destination slot per key, a non-zero
     * receipt handle. */
    RequestContext context;
    context.requestId = "r1";
    auto plan = facet->PreparePrefetch(ids, context);
    ASSERT_TRUE(plan.HasValue());
    EXPECT_EQ(plan.Value().keys, ids);
    const PlanPayload payload = DecodePlanPayload(plan.Value().payload);
    EXPECT_NE(payload.batchId, 0U);
    ASSERT_EQ(payload.slots.size(), ids.size());

    /* The receipt concludes the window; a re-probe reports the admission
     * as present, with nothing left to fetch. */
    facet->CompletePrefetch(plan.Value(), true);
    auto after = facet->Lookup("r1", ids);
    ASSERT_TRUE(after.HasValue());
    EXPECT_EQ(after.Value().first, std::vector<std::uint8_t>({1, 1}));
    EXPECT_FALSE(after.Value().second);

    /* The executor half is refused: no data plane runs here. */
    EXPECT_EQ(facet->Prefetch(1, plan.Value()).Underlying(), Status::Unsupported().Underlying());
    EXPECT_TRUE(lower.pins.empty());
    EXPECT_TRUE(scheduler->GetFinished().empty());
}

/*== Both halves, closing the loop inside one primary ==*/

TEST_F(CacheStorePrefetchTest, PrimaryClosesTheLoopInternally)
{
    StubLowerStore lower;
    lower.present.insert(MakeBlock(1));
    lower.present.insert(MakeBlock(2));
    std::unique_ptr<CacheStore> store = MakeStore(FreshId(), "both", &lower);
    ASSERT_NE(store, nullptr);
    Prefetchable* facet = store->AsPrefetchable();
    ASSERT_NE(facet, nullptr);

    /* The probe: the engine holds nothing, the tier below holds it all. */
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2)};
    auto probe = facet->Lookup("r1", ids);
    ASSERT_TRUE(probe.HasValue());
    EXPECT_EQ(probe.Value().first, std::vector<std::uint8_t>({0, 0}));
    EXPECT_TRUE(probe.Value().second);

    /* The mint rides the pump; the execution drives the wired tier; the
     * verdict surfaces exactly once through the store's stream. */
    RequestContext context;
    context.requestId = "r1";
    auto plan = facet->PreparePrefetch(ids, context);
    ASSERT_TRUE(plan.HasValue());
    ASSERT_EQ(plan.Value().keys, ids);
    EXPECT_EQ(facet->Prefetch(7, plan.Value()).Underlying(), 0);
    ASSERT_EQ(lower.loads.size(), 1U);
    EXPECT_EQ(lower.loads[0].jobId, 1U);
    lower.Complete(1, true);
    const auto finished = store->GetFinished();
    ASSERT_EQ(finished.size(), 1U);
    EXPECT_EQ(finished[0].first, 7U);
    EXPECT_TRUE(finished[0].second);
    EXPECT_EQ(lower.unpins.size(), 1U);

    /* The receipt rides the pump too; a re-probe is a full hit with the
     * signal off. */
    facet->CompletePrefetch(plan.Value(), true);
    auto after = facet->Lookup("r1", ids);
    ASSERT_TRUE(after.HasValue());
    EXPECT_EQ(after.Value().first, std::vector<std::uint8_t>({1, 1}));
    EXPECT_FALSE(after.Value().second);
    EXPECT_EQ(lower.lookups.size(), 1U);
    EXPECT_EQ(store->Wait(7).Underlying(), Status::InvalidParam().Underlying());
}

/*== The join key: the mint must name the request the probe built ==*/

TEST_F(CacheStorePrefetchTest, TheMintJoinsTheRequestTheProbeBuilt)
{
    StubLowerStore lower;
    std::unique_ptr<CacheStore> store = MakeStore(FreshId(), "both", &lower);
    ASSERT_NE(store, nullptr);
    Prefetchable* facet = store->AsPrefetchable();
    ASSERT_NE(facet, nullptr);

    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2)};
    lower.present.insert(ids.begin(), ids.end());
    RequestContext context;
    context.requestId = "r1";

    /* No probe, no state: the mint names a request that never looked. */
    EXPECT_EQ(facet->PreparePrefetch(ids, context).Error().Underlying(),
              Status::InvalidParam().Underlying());

    /* The probe builds the state its requestId addresses; a mint under
     * another name joins nothing. */
    ASSERT_TRUE(facet->Lookup("r1", ids).HasValue());
    context.requestId = "r2";
    EXPECT_EQ(facet->PreparePrefetch(ids, context).Error().Underlying(),
              Status::InvalidParam().Underlying());

    /* The mint under the probed request carries its keys; the release
     * drops the state, so the next mint finds nothing again. */
    context.requestId = "r1";
    auto plan = facet->PreparePrefetch(ids, context);
    ASSERT_TRUE(plan.HasValue());
    EXPECT_EQ(plan.Value().keys, ids);
    facet->ReleaseRequest("r1");
    EXPECT_EQ(facet->PreparePrefetch(ids, context).Error().Underlying(),
              Status::InvalidParam().Underlying());
}

/*== A replica spanning the two planes of a both-led domain ==*/

TEST_F(CacheStorePrefetchTest, ReplicaDrivesTheDomainEngineAndItsOwnRelay)
{
    StubLowerStore lower;
    lower.present.insert(MakeBlock(1));
    lower.present.insert(MakeBlock(2));
    std::unique_ptr<CacheStore> primary = MakeStore(FreshId(), "both", &lower);
    ASSERT_NE(primary, nullptr);

    const std::string id = usedIds_.back();
    std::unique_ptr<CacheStore> replica = MakeStore(id, "both", &lower);
    ASSERT_NE(replica, nullptr);
    Prefetchable* facet = replica->AsPrefetchable();
    ASSERT_NE(facet, nullptr);

    /* The probe rides the replica's planes: the snapshot from the primary's
     * engine over the control connection, the signal from the wired tier. */
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2)};
    auto probe = facet->Lookup("r1", ids);
    ASSERT_TRUE(probe.HasValue());
    EXPECT_EQ(probe.Value().first, std::vector<std::uint8_t>({0, 0}));
    EXPECT_TRUE(probe.Value().second);

    /* The mint comes from the primary's engine; the execution goes to the
     * replica's own wired tier; the receipt travels back. */
    RequestContext context;
    context.requestId = "r1";
    auto plan = facet->PreparePrefetch(ids, context);
    ASSERT_TRUE(plan.HasValue());
    ASSERT_EQ(plan.Value().keys, ids);
    EXPECT_EQ(facet->Prefetch(5, plan.Value()).Underlying(), 0);
    ASSERT_EQ(lower.loads.size(), 1U);
    lower.Complete(1, true);
    const auto finished = replica->GetFinished();
    ASSERT_EQ(finished.size(), 1U);
    EXPECT_EQ(finished[0].first, 5U);
    EXPECT_TRUE(finished[0].second);

    facet->CompletePrefetch(plan.Value(), true);
    auto after = facet->Lookup("r1", ids);
    ASSERT_TRUE(after.HasValue());
    EXPECT_EQ(after.Value().first, std::vector<std::uint8_t>({1, 1}));
}

/*== The wiring gate: no lower tier, no facet ==*/

TEST_F(CacheStorePrefetchTest, RolesHideTheFacetWithoutALowerTier)
{
    /* A both-led domain: the primary and a replica. */
    const std::string bothId = FreshId();
    std::unique_ptr<CacheStore> primary = MakeStore(bothId, "both", nullptr);
    ASSERT_NE(primary, nullptr);
    EXPECT_EQ(primary->AsPrefetchable(), nullptr);
    std::unique_ptr<CacheStore> replica = MakeStore(bothId, "both", nullptr);
    ASSERT_NE(replica, nullptr);
    EXPECT_EQ(replica->AsPrefetchable(), nullptr);

    /* A worker-led domain: its leader and the scheduler. */
    const std::string workerId = FreshId();
    std::unique_ptr<CacheStore> worker = MakeStore(workerId, "worker", nullptr);
    ASSERT_NE(worker, nullptr);
    EXPECT_EQ(worker->AsPrefetchable(), nullptr);
    std::unique_ptr<CacheStore> scheduler = MakeStore(workerId, "scheduler", nullptr);
    ASSERT_NE(scheduler, nullptr);
    EXPECT_EQ(scheduler->AsPrefetchable(), nullptr);
}

}  // namespace
}  // namespace UC::Store
