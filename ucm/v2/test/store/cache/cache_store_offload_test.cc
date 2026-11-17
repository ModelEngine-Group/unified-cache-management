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
#include "offloadable.h"
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
 *  one source slot per key. */
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

/** One Offloadable wiring of the cache store, over one in-process domain per
 *  unique id; the fixture unlinks each domain's rendezvous artifacts at the
 *  end of its test. */
class CacheStoreOffloadTest : public ::testing::Test {
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
        std::string id = "gtest-offload-";
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

TEST_F(CacheStoreOffloadTest, WorkerExposesTheExecutorHalfWhenWired)
{
    StubLowerStore lower;
    std::unique_ptr<CacheStore> leader = MakeStore(FreshId(), "worker", &lower);
    ASSERT_NE(leader, nullptr);
    Offloadable* facet = leader->AsOffloadable();
    ASSERT_NE(facet, nullptr);

    /* A leaf of the same domain, identically wired: its facet appears too. */
    const std::string id = usedIds_.back();
    std::unique_ptr<CacheStore> leaf = MakeStore(id, "worker", &lower);
    ASSERT_NE(leaf, nullptr);
    EXPECT_NE(leaf->AsOffloadable(), nullptr);

    /* The engine half is structural idle: no engine lives in a worker. */
    EXPECT_TRUE(facet->PrepareOffload().keys.empty());
    facet->CompleteOffload(AddressedToken{});

    /* The executor half drives the wired lower tier, and the outer job
     * shares the store's wait. */
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2)};
    EXPECT_EQ(facet->Offload(9, MakePlan(ids, {0, 1})).Underlying(), 0);
    ASSERT_EQ(lower.allocs.size(), 1U);
    EXPECT_EQ(lower.allocs[0], ids);
    ASSERT_EQ(lower.dumps.size(), 1U);
    EXPECT_EQ(lower.dumps[0].jobId, 1U);

    lower.Complete(1, true);
    EXPECT_EQ(leader->Wait(9).Underlying(), 0);
    EXPECT_TRUE(leader->GetFinished().empty());
    EXPECT_EQ(leader->Wait(9).Underlying(), Status::InvalidParam().Underlying());
}

/*== The worker index: an out-of-domain index is refused ==*/

TEST_F(CacheStoreOffloadTest, WorkerRefusesAnIndexOutsideTheDomainWorkers)
{
    StubLowerStore lower;
    /* The domain declares one worker: index 1 is outside it. */
    Dictionary dict = MakeDomainDict(FreshId(), "worker", &lower);
    dict.SetInt64("worker_index", 1);
    CacheStore store;
    EXPECT_EQ(store.Setup(dict).Underlying(), Status::InvalidParam().Underlying());
}

/*== The engine half, as served by the scheduler role ==*/

TEST_F(CacheStoreOffloadTest, SchedulerServesTheEngineHalfLocally)
{
    StubLowerStore lower; /* Never driven here. */
    std::unique_ptr<CacheStore> worker = MakeStore(FreshId(), "worker", &lower);
    ASSERT_NE(worker, nullptr);

    const std::string id = usedIds_.back();
    std::unique_ptr<CacheStore> scheduler = MakeStore(id, "scheduler", &lower);
    ASSERT_NE(scheduler, nullptr);
    Offloadable* facet = scheduler->AsOffloadable();
    ASSERT_NE(facet, nullptr);

    /* The local engine serves the admission protocol; a fully successful
     * conclusion fills the offload pending set. */
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2)};
    ASSERT_TRUE(scheduler->Alloc(ids, RequestContext{}).HasValue());
    scheduler->Commit(ids, true);

    /* The mint: planned keys, one slot per key, a non-zero handle. */
    AddressedToken plan = facet->PrepareOffload();
    ASSERT_EQ(plan.keys, ids);
    const PlanPayload payload = DecodePlanPayload(plan.payload);
    EXPECT_NE(payload.batchId, 0U);
    ASSERT_EQ(payload.slots.size(), ids.size());

    /* Receipt releases; the next mint is idle. */
    facet->CompleteOffload(plan);
    EXPECT_TRUE(facet->PrepareOffload().keys.empty());

    /* The executor half is refused: no data plane runs here. */
    EXPECT_EQ(facet->Offload(1, std::move(plan)).Underlying(), Status::Unsupported().Underlying());
    EXPECT_TRUE(lower.allocs.empty());
    EXPECT_TRUE(scheduler->GetFinished().empty());
}

/*== Both halves, closing the loop inside one primary ==*/

TEST_F(CacheStoreOffloadTest, PrimaryClosesTheLoopInternally)
{
    StubLowerStore lower;
    std::unique_ptr<CacheStore> store = MakeStore(FreshId(), "both", &lower);
    ASSERT_NE(store, nullptr);
    Offloadable* facet = store->AsOffloadable();
    ASSERT_NE(facet, nullptr);

    /* An admission concludes fully successfully: the pending set fills. */
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2)};
    ASSERT_TRUE(store->Alloc(ids, RequestContext{}).HasValue());
    store->Commit(ids, true);

    /* The mint rides the pump; the execution drives the wired tier; the
     * verdict surfaces exactly once through the store's stream. */
    AddressedToken plan = facet->PrepareOffload();
    ASSERT_EQ(plan.keys, ids);
    EXPECT_EQ(facet->Offload(7, plan).Underlying(), 0);
    ASSERT_EQ(lower.dumps.size(), 1U);
    lower.Complete(1, true);
    const auto finished = store->GetFinished();
    ASSERT_EQ(finished.size(), 1U);
    EXPECT_EQ(finished[0].first, 7U);
    EXPECT_TRUE(finished[0].second);

    /* The receipt rides the pump too; the next mint is idle. */
    facet->CompleteOffload(plan);
    EXPECT_TRUE(facet->PrepareOffload().keys.empty());
    EXPECT_EQ(store->Wait(7).Underlying(), Status::InvalidParam().Underlying());
}

/*== A replica spanning the two planes of a both-led domain ==*/

TEST_F(CacheStoreOffloadTest, ReplicaDrivesTheDomainEngineAndItsOwnRelay)
{
    StubLowerStore lower;
    std::unique_ptr<CacheStore> primary = MakeStore(FreshId(), "both", &lower);
    ASSERT_NE(primary, nullptr);

    const std::string id = usedIds_.back();
    std::unique_ptr<CacheStore> replica = MakeStore(id, "both", &lower);
    ASSERT_NE(replica, nullptr);
    Offloadable* facet = replica->AsOffloadable();
    ASSERT_NE(facet, nullptr);

    /* The replica's admission lands in the primary's engine over the
     * control connection. */
    const std::vector<BlockId> ids{MakeBlock(1), MakeBlock(2)};
    ASSERT_TRUE(replica->Alloc(ids, RequestContext{}).HasValue());
    replica->Commit(ids, true);

    /* The mint comes from the primary's engine; the execution goes to the
     * replica's own wired tier. */
    AddressedToken plan = facet->PrepareOffload();
    ASSERT_EQ(plan.keys, ids);
    EXPECT_EQ(facet->Offload(5, plan).Underlying(), 0);
    ASSERT_EQ(lower.dumps.size(), 1U);
    lower.Complete(1, true);
    const auto finished = replica->GetFinished();
    ASSERT_EQ(finished.size(), 1U);
    EXPECT_EQ(finished[0].first, 5U);
    EXPECT_TRUE(finished[0].second);

    /* The receipt travels back to the primary's engine. */
    facet->CompleteOffload(plan);
    EXPECT_TRUE(facet->PrepareOffload().keys.empty());
}

/*== The wiring gate: no lower tier, no facet ==*/

TEST_F(CacheStoreOffloadTest, RolesHideTheFacetWithoutALowerTier)
{
    /* A both-led domain: the primary and a replica. */
    const std::string bothId = FreshId();
    std::unique_ptr<CacheStore> primary = MakeStore(bothId, "both", nullptr);
    ASSERT_NE(primary, nullptr);
    EXPECT_EQ(primary->AsOffloadable(), nullptr);
    std::unique_ptr<CacheStore> replica = MakeStore(bothId, "both", nullptr);
    ASSERT_NE(replica, nullptr);
    EXPECT_EQ(replica->AsOffloadable(), nullptr);

    /* A worker-led domain: its leader and the scheduler. */
    const std::string workerId = FreshId();
    std::unique_ptr<CacheStore> worker = MakeStore(workerId, "worker", nullptr);
    ASSERT_NE(worker, nullptr);
    EXPECT_EQ(worker->AsOffloadable(), nullptr);
    std::unique_ptr<CacheStore> scheduler = MakeStore(workerId, "scheduler", nullptr);
    ASSERT_NE(scheduler, nullptr);
    EXPECT_EQ(scheduler->AsOffloadable(), nullptr);
}

}  // namespace
}  // namespace UC::Store
