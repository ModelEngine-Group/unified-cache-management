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
#include "store_entry.h"
#include <gtest/gtest.h>
#include "fixtures/fake_store.h"

namespace UC::Proxy {
namespace {

TEST(StoreEntryTest, OnNewRequestDuplicateIsRejected)
{
    auto handle = MakeEntry(true, true);
    const RequestContext context = MakeContext("r");
    EXPECT_TRUE(handle.entry->OnNewRequest(context).Success());
    auto duplicate = handle.entry->OnNewRequest(context);
    ASSERT_TRUE(duplicate.Failure());
    EXPECT_EQ(duplicate, Status::InvalidParam());
}

TEST(StoreEntryTest, OperationsOnUnknownRequestFail)
{
    auto handle = MakeEntry(true, true);
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    EXPECT_FALSE(handle.entry->Lookup("missing", keys));
    EXPECT_FALSE(handle.entry->LookupOnPrefix("missing", keys));
    EXPECT_FALSE(handle.entry->LookupOnReverse("missing", keys));
    EXPECT_FALSE(handle.entry->PreparePrefetch("missing"));
    EXPECT_FALSE(handle.entry->PrepareGet("missing", keys));
    EXPECT_FALSE(handle.entry->PreparePut("missing", keys));
    handle.entry->Touch("missing", keys);
    handle.entry->CompletePrefetch("missing", true);
    handle.entry->CompleteGet("missing");
    handle.entry->CompletePut("missing", true);
    EXPECT_TRUE(handle.fake->plainLookups.empty());
    EXPECT_TRUE(handle.fake->prefetchLookups.empty());
    EXPECT_TRUE(handle.fake->pins.empty());
    EXPECT_TRUE(handle.fake->unpins.empty());
    EXPECT_TRUE(handle.fake->allocs.empty());
    EXPECT_TRUE(handle.fake->commits.empty());
    EXPECT_TRUE(handle.fake->completedPrefetches.empty());
}

TEST(StoreEntryTest, LookupDispatchesToPrefetchableFacet)
{
    auto handle = MakeEntry(true, true);
    const std::vector<BlockId> keys = MakeKeys(1, 3);
    handle.entry->OnNewRequest(MakeContext("r"));
    handle.fake->signalPrefetchable = true;
    auto signal = handle.entry->Lookup("r", keys);
    ASSERT_TRUE(signal);
    EXPECT_FALSE(signal.Value().has_value());
    auto prefixSignal = handle.entry->LookupOnPrefix("r", keys);
    ASSERT_TRUE(prefixSignal);
    EXPECT_FALSE(prefixSignal.Value().has_value());
    handle.fake->signalPrefetchable = false;
    auto hits = handle.entry->Lookup("r", keys);
    ASSERT_TRUE(hits);
    ASSERT_TRUE(hits.Value().has_value());
    EXPECT_EQ(hits.Value().value(), std::vector<std::uint8_t>(3, 1));
    auto prefix = handle.entry->LookupOnPrefix("r", keys);
    ASSERT_TRUE(prefix);
    EXPECT_EQ(prefix.Value(), 2);
    ASSERT_EQ(handle.fake->prefetchLookups.size(), 4);
    for (const auto& recorded : handle.fake->prefetchLookups) {
        EXPECT_EQ(recorded.first, "r");
        EXPECT_EQ(recorded.second, keys);
    }
    EXPECT_TRUE(handle.fake->plainLookups.empty());
}

TEST(StoreEntryTest, LookupDispatchesToPlainStore)
{
    auto handle = MakeEntry(false, false);
    const std::vector<BlockId> keys = MakeKeys(1, 4);
    handle.entry->OnNewRequest(MakeContext("r"));
    handle.fake->plainHits = {1, 1, 0, 1};
    auto hits = handle.entry->Lookup("r", keys);
    ASSERT_TRUE(hits);
    ASSERT_TRUE(hits.Value().has_value());
    EXPECT_EQ(hits.Value().value(), (std::vector<std::uint8_t>{1, 1, 0, 1}));
    auto prefix = handle.entry->LookupOnPrefix("r", keys);
    ASSERT_TRUE(prefix);
    EXPECT_EQ(prefix.Value(), 1);
    auto reverse = handle.entry->LookupOnReverse("r", keys);
    ASSERT_TRUE(reverse);
    EXPECT_EQ(reverse.Value(), 3);
    ASSERT_EQ(handle.fake->plainLookups.size(), 3);
    EXPECT_TRUE(handle.fake->prefetchLookups.empty());
}

TEST(StoreEntryTest, PrepareGetPinsAndUnpinsTokenedBlocks)
{
    auto handle = MakeEntry(true, true);
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    handle.entry->OnNewRequest(MakeContext("r"));
    auto token = handle.entry->PrepareGet("r", keys);
    ASSERT_TRUE(token);
    EXPECT_EQ(token.Value().keys, keys);
    EXPECT_EQ(token.Value().payload, MakeAddressedToken(2).payload);
    ASSERT_EQ(handle.fake->pins.size(), 1);
    EXPECT_EQ(handle.fake->pins[0], keys);
    handle.entry->CompleteGet("r");
    ASSERT_EQ(handle.fake->unpins.size(), 1);
    EXPECT_EQ(handle.fake->unpins[0], keys);
    handle.entry->CompleteGet("r");
    ASSERT_EQ(handle.fake->unpins.size(), 2);
    EXPECT_TRUE(handle.fake->unpins[1].empty());
}

TEST(StoreEntryTest, PrepareGetOnNonTokenedStoreMintsKeysOnlyToken)
{
    auto handle = MakeEntry(true, false);
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    handle.entry->OnNewRequest(MakeContext("r"));
    auto token = handle.entry->PrepareGet("r", keys);
    ASSERT_TRUE(token);
    EXPECT_EQ(token.Value().keys, keys);
    EXPECT_TRUE(token.Value().payload.empty());
    EXPECT_TRUE(handle.fake->pins.empty());
    handle.entry->CompleteGet("r");
    EXPECT_TRUE(handle.fake->unpins.empty());
}

TEST(StoreEntryTest, PrepareGetPropagatesPinFailure)
{
    auto handle = MakeEntry(true, true);
    handle.entry->OnNewRequest(MakeContext("r"));
    handle.fake->pinFails = true;
    auto token = handle.entry->PrepareGet("r", MakeKeys(1, 2));
    ASSERT_FALSE(token);
    EXPECT_EQ(token.Error(), Status::NotFound());
    handle.entry->CompleteGet("r");
    ASSERT_EQ(handle.fake->unpins.size(), 1);
    EXPECT_TRUE(handle.fake->unpins[0].empty());
}

TEST(StoreEntryTest, PreparePutAllocsWithRequestContext)
{
    auto handle = MakeEntry(true, true);
    const std::vector<BlockId> keys = MakeKeys(5, 3);
    handle.entry->OnNewRequest(MakeContext("r"));
    auto token = handle.entry->PreparePut("r", keys);
    ASSERT_TRUE(token);
    EXPECT_EQ(token.Value().keys, keys);
    EXPECT_EQ(token.Value().payload, MakeAddressedToken(3).payload);
    ASSERT_EQ(handle.fake->allocs.size(), 1);
    EXPECT_EQ(handle.fake->allocs[0].first, keys);
    EXPECT_EQ(handle.fake->allocs[0].second.ttl, std::chrono::milliseconds(500));
    EXPECT_EQ(handle.fake->allocs[0].second.totalBlockCount, 3);
    handle.entry->CompletePut("r", true);
    ASSERT_EQ(handle.fake->commits.size(), 1);
    EXPECT_EQ(handle.fake->commits[0].first, keys);
    EXPECT_TRUE(handle.fake->commits[0].second);
    handle.entry->CompletePut("r", true);
    ASSERT_EQ(handle.fake->commits.size(), 2);
    EXPECT_TRUE(handle.fake->commits[1].first.empty());
}

TEST(StoreEntryTest, CompletePutForwardsSuccessFlag)
{
    auto handle = MakeEntry(true, true);
    const std::vector<BlockId> keys = MakeKeys(1, 1);
    handle.entry->OnNewRequest(MakeContext("r"));
    ASSERT_TRUE(handle.entry->PreparePut("r", keys));
    handle.entry->CompletePut("r", false);
    ASSERT_EQ(handle.fake->commits.size(), 1);
    EXPECT_EQ(handle.fake->commits[0].first, keys);
    EXPECT_FALSE(handle.fake->commits[0].second);
}

TEST(StoreEntryTest, PreparePutOnNonTokenedStoreMintsKeysOnlyToken)
{
    auto handle = MakeEntry(true, false);
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    handle.entry->OnNewRequest(MakeContext("r"));
    auto token = handle.entry->PreparePut("r", keys);
    ASSERT_TRUE(token);
    EXPECT_EQ(token.Value().keys, keys);
    EXPECT_TRUE(token.Value().payload.empty());
    EXPECT_TRUE(handle.fake->allocs.empty());
    handle.entry->CompletePut("r", true);
    EXPECT_TRUE(handle.fake->commits.empty());
}

TEST(StoreEntryTest, PreparePutPropagatesAllocFailure)
{
    auto handle = MakeEntry(true, true);
    handle.entry->OnNewRequest(MakeContext("r"));
    handle.fake->allocFails = true;
    auto token = handle.entry->PreparePut("r", MakeKeys(1, 2));
    ASSERT_FALSE(token);
    EXPECT_EQ(token.Error(), Status::NoSpace());
    handle.entry->CompletePut("r", true);
    ASSERT_EQ(handle.fake->commits.size(), 1);
    EXPECT_TRUE(handle.fake->commits[0].first.empty());
}

TEST(StoreEntryTest, SubmitPrefetchForwardsPlanToPrefetchableStore)
{
    auto handle = MakeEntry(true, true);
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    handle.entry->OnNewRequest(MakeContext("r"));
    handle.fake->signalPrefetchable = true;
    ASSERT_FALSE(handle.entry->Lookup("r", keys).Value().has_value());
    auto token = handle.entry->PreparePrefetch("r");
    ASSERT_TRUE(token);
    EXPECT_EQ(token.Value().keys, keys);
    EXPECT_TRUE(handle.entry->SubmitPrefetch(7, token.Value()).Success());
    ASSERT_EQ(handle.fake->prefetches.size(), 1);
    EXPECT_EQ(handle.fake->prefetches[0].first, 7);
    EXPECT_EQ(handle.fake->prefetches[0].second.payload, token.Value().payload);
    auto duplicate = handle.entry->SubmitPrefetch(7, token.Value());
    ASSERT_TRUE(duplicate.Failure());
    EXPECT_EQ(duplicate, Status::InvalidParam());
    handle.fake->finishedQueue = {
        {7, true}
    };
    auto finished = handle.entry->GetFinished();
    ASSERT_EQ(finished.size(), 1);
    EXPECT_EQ(finished[0].jobId, 7);
    EXPECT_TRUE(finished[0].success);
    EXPECT_TRUE(handle.entry->SubmitPrefetch(7, token.Value()).Success());
    handle.entry->CompletePrefetch("r", true);
    ASSERT_EQ(handle.fake->completedPrefetches.size(), 1);
    EXPECT_TRUE(handle.fake->completedPrefetches[0].second);
    handle.entry->CompletePrefetch("r", true);
    ASSERT_EQ(handle.fake->completedPrefetches.size(), 2);
    EXPECT_TRUE(handle.fake->completedPrefetches[1].first.keys.empty());
}

TEST(StoreEntryTest, SubmitPrefetchPropagatesFailure)
{
    auto handle = MakeEntry(true, true);
    AddressedToken token;
    handle.fake->prefetchStatus = Status::General("prefetch rejected");
    EXPECT_EQ(handle.entry->SubmitPrefetch(7, token), Status::General());
    handle.fake->prefetchStatus = Status::Ok();
    EXPECT_TRUE(handle.entry->SubmitPrefetch(7, token).Success());
}

TEST(StoreEntryTest, SubmitPrefetchOnNonPrefetchableStoreSucceedsImmediately)
{
    auto handle = MakeEntry(false, true);
    AddressedToken token;
    EXPECT_TRUE(handle.entry->SubmitPrefetch(7, token).Success());
    EXPECT_TRUE(handle.fake->prefetches.empty());
    auto duplicate = handle.entry->SubmitPrefetch(7, token);
    ASSERT_TRUE(duplicate.Failure());
    EXPECT_EQ(duplicate, Status::InvalidParam());
    auto finished = handle.entry->GetFinished();
    ASSERT_EQ(finished.size(), 1);
    EXPECT_EQ(finished[0].jobId, 7);
    EXPECT_TRUE(finished[0].success);
    EXPECT_TRUE(handle.entry->GetFinished().empty());
    EXPECT_TRUE(handle.entry->SubmitPrefetch(7, token).Success());
}

TEST(StoreEntryTest, WaitDeliversSyntheticCompletionOnce)
{
    auto handle = MakeEntry(false, true);
    AddressedToken token;
    EXPECT_TRUE(handle.entry->SubmitPrefetch(7, token).Success());
    auto first = handle.entry->Wait({7});
    ASSERT_EQ(first.size(), 1);
    EXPECT_EQ(first[0].jobId, 7);
    EXPECT_TRUE(first[0].success);
    EXPECT_TRUE(handle.entry->Wait({7}).empty());
    EXPECT_TRUE(handle.entry->GetFinished().empty());
    ASSERT_EQ(handle.fake->waits.size(), 1);
    EXPECT_EQ(handle.fake->waits[0], std::vector<std::size_t>{7});
}

TEST(StoreEntryTest, WaitCombinesSyntheticAndStoreCompletions)
{
    auto handle = MakeEntry(false, true);
    const std::vector<BlockId> keys = MakeKeys(1, 1);
    handle.entry->OnNewRequest(MakeContext("r"));
    auto token = handle.entry->PrepareGet("r", keys);
    ASSERT_TRUE(token);
    AddressedToken synthetic;
    EXPECT_TRUE(handle.entry->SubmitPrefetch(7, synthetic).Success());
    EXPECT_TRUE(
        handle.entry->SubmitGet(5, {MakePayload(16)}, token.Value(), IoContext{0}).Success());
    handle.fake->finishedQueue = {
        {5, false}
    };
    auto results = handle.entry->Wait({5, 7});
    EXPECT_EQ(results.size(), 2);
    EXPECT_TRUE(Contains(results, 7, true));
    EXPECT_TRUE(Contains(results, 5, false));
    ASSERT_EQ(handle.fake->waits.size(), 1);
    EXPECT_EQ(handle.fake->waits[0], std::vector<std::size_t>{5});
    EXPECT_TRUE(handle.entry->Wait({5, 7}).empty());
    EXPECT_TRUE(handle.entry->GetFinished().empty());
}

TEST(StoreEntryTest, SubmitGetUsesTokenBearingLoadOnTokenedStore)
{
    auto handle = MakeEntry(true, true);
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    handle.entry->OnNewRequest(MakeContext("r"));
    auto token = handle.entry->PrepareGet("r", keys);
    ASSERT_TRUE(token);
    const std::vector<BlockPayload> payloads = {MakePayload(16), MakePayload(32)};
    const IoContext context{0xABCD};
    EXPECT_TRUE(handle.entry->SubmitGet(9, payloads, token.Value(), context).Success());
    ASSERT_EQ(handle.fake->tokenedLoads.size(), 1);
    EXPECT_EQ(handle.fake->tokenedLoads[0].jobId, 9);
    EXPECT_EQ(handle.fake->tokenedLoads[0].job.blockIds, keys);
    ASSERT_EQ(handle.fake->tokenedLoads[0].job.payloads.size(), 2);
    EXPECT_EQ(handle.fake->tokenedLoads[0].job.payloads[0].sizes, payloads[0].sizes);
    EXPECT_EQ(handle.fake->tokenedLoads[0].job.payloads[1].sizes, payloads[1].sizes);
    EXPECT_EQ(handle.fake->tokenedLoads[0].job.prerequisiteHandle, 0xABCD);
    EXPECT_TRUE(handle.fake->tokenedLoads[0].token.keys.empty());
    EXPECT_EQ(handle.fake->tokenedLoads[0].token.payload, MakeAddressedToken(2).payload);
    EXPECT_TRUE(handle.fake->baseLoads.empty());
}

TEST(StoreEntryTest, SubmitGetUsesPlainLoadOnNonTokenedStore)
{
    auto handle = MakeEntry(true, false);
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    handle.entry->OnNewRequest(MakeContext("r"));
    auto token = handle.entry->PrepareGet("r", keys);
    ASSERT_TRUE(token);
    const std::vector<BlockPayload> payloads = {MakePayload(16), MakePayload(32)};
    const IoContext context{0xABCD};
    EXPECT_TRUE(handle.entry->SubmitGet(9, payloads, token.Value(), context).Success());
    ASSERT_EQ(handle.fake->baseLoads.size(), 1);
    EXPECT_EQ(handle.fake->baseLoads[0].jobId, 9);
    EXPECT_EQ(handle.fake->baseLoads[0].job.blockIds, keys);
    ASSERT_EQ(handle.fake->baseLoads[0].job.payloads.size(), 2);
    EXPECT_EQ(handle.fake->baseLoads[0].job.payloads[1].sizes, payloads[1].sizes);
    EXPECT_EQ(handle.fake->baseLoads[0].job.prerequisiteHandle, 0xABCD);
    EXPECT_TRUE(handle.fake->tokenedLoads.empty());
}

TEST(StoreEntryTest, SubmitGetRejectsMismatchedInputs)
{
    auto handle = MakeEntry(true, true);
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    handle.entry->OnNewRequest(MakeContext("r"));
    auto token = handle.entry->PrepareGet("r", keys);
    ASSERT_TRUE(token);
    auto payloadMismatch =
        handle.entry->SubmitGet(9, {MakePayload(16)}, token.Value(), IoContext{0});
    ASSERT_TRUE(payloadMismatch.Failure());
    EXPECT_EQ(payloadMismatch, Status::InvalidParam());
    EXPECT_TRUE(handle.fake->tokenedLoads.empty());

    auto plain = MakeEntry(true, false);
    plain.entry->OnNewRequest(MakeContext("r"));
    auto plainToken = plain.entry->PrepareGet("r", keys);
    ASSERT_TRUE(plainToken);
    AddressedToken stray = MakeAddressedToken(1);
    stray.keys = keys;
    auto strayRejected =
        plain.entry->SubmitGet(9, {MakePayload(16), MakePayload(16)}, stray, IoContext{0});
    ASSERT_TRUE(strayRejected.Failure());
    EXPECT_EQ(strayRejected, Status::InvalidParam());
    EXPECT_TRUE(plain.fake->baseLoads.empty());
}

TEST(StoreEntryTest, SubmitGetPropagatesStoreFailure)
{
    auto handle = MakeEntry(true, true);
    const std::vector<BlockId> keys = MakeKeys(1, 1);
    handle.entry->OnNewRequest(MakeContext("r"));
    auto token = handle.entry->PrepareGet("r", keys);
    ASSERT_TRUE(token);
    handle.fake->loadStatus = Status::General("load rejected");
    auto status = handle.entry->SubmitGet(9, {MakePayload(16)}, token.Value(), IoContext{0});
    EXPECT_EQ(status, Status::General());
    handle.fake->loadStatus = Status::Ok();
    EXPECT_TRUE(
        handle.entry->SubmitGet(9, {MakePayload(16)}, token.Value(), IoContext{0}).Success());
}

TEST(StoreEntryTest, SubmitPutUsesTokenBearingDumpOnTokenedStore)
{
    auto handle = MakeEntry(true, true);
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    handle.entry->OnNewRequest(MakeContext("r"));
    auto token = handle.entry->PreparePut("r", keys);
    ASSERT_TRUE(token);
    const std::vector<BlockPayload> payloads = {MakePayload(16), MakePayload(32)};
    const IoContext context{0xABCD};
    EXPECT_TRUE(handle.entry->SubmitPut(4, payloads, token.Value(), context).Success());
    ASSERT_EQ(handle.fake->tokenedDumps.size(), 1);
    EXPECT_EQ(handle.fake->tokenedDumps[0].jobId, 4);
    EXPECT_EQ(handle.fake->tokenedDumps[0].job.blockIds, keys);
    EXPECT_EQ(handle.fake->tokenedDumps[0].job.prerequisiteHandle, 0xABCD);
    EXPECT_TRUE(handle.fake->tokenedDumps[0].token.keys.empty());
    EXPECT_EQ(handle.fake->tokenedDumps[0].token.payload, MakeAddressedToken(2).payload);
    EXPECT_TRUE(handle.fake->baseDumps.empty());
    auto duplicate = handle.entry->SubmitPut(4, payloads, token.Value(), context);
    ASSERT_TRUE(duplicate.Failure());
    EXPECT_EQ(duplicate, Status::InvalidParam());
}

TEST(StoreEntryTest, SubmitPutUsesPlainDumpOnNonTokenedStore)
{
    auto handle = MakeEntry(true, false);
    const std::vector<BlockId> keys = MakeKeys(1, 1);
    handle.entry->OnNewRequest(MakeContext("r"));
    auto token = handle.entry->PreparePut("r", keys);
    ASSERT_TRUE(token);
    EXPECT_TRUE(
        handle.entry->SubmitPut(4, {MakePayload(16)}, token.Value(), IoContext{0}).Success());
    ASSERT_EQ(handle.fake->baseDumps.size(), 1);
    EXPECT_EQ(handle.fake->baseDumps[0].jobId, 4);
    EXPECT_EQ(handle.fake->baseDumps[0].job.blockIds, keys);
    EXPECT_TRUE(handle.fake->tokenedDumps.empty());
}

TEST(StoreEntryTest, SubmitPutPropagatesStoreFailure)
{
    auto handle = MakeEntry(true, true);
    const std::vector<BlockId> keys = MakeKeys(1, 1);
    handle.entry->OnNewRequest(MakeContext("r"));
    auto token = handle.entry->PreparePut("r", keys);
    ASSERT_TRUE(token);
    handle.fake->dumpStatus = Status::General("dump rejected");
    auto status = handle.entry->SubmitPut(4, {MakePayload(16)}, token.Value(), IoContext{0});
    EXPECT_EQ(status, Status::General());
    handle.fake->dumpStatus = Status::Ok();
    EXPECT_TRUE(
        handle.entry->SubmitPut(4, {MakePayload(16)}, token.Value(), IoContext{0}).Success());
}

TEST(StoreEntryTest, GetFinishedConvertsStoreCompletionsAndFreesJobIds)
{
    auto handle = MakeEntry(true, true);
    const std::vector<BlockId> keys = MakeKeys(1, 1);
    handle.entry->OnNewRequest(MakeContext("r"));
    auto token = handle.entry->PrepareGet("r", keys);
    ASSERT_TRUE(token);
    EXPECT_TRUE(
        handle.entry->SubmitGet(5, {MakePayload(16)}, token.Value(), IoContext{0}).Success());
    handle.fake->finishedQueue = {
        {5, false},
        {6, true }
    };
    auto finished = handle.entry->GetFinished();
    ASSERT_EQ(finished.size(), 2);
    EXPECT_EQ(finished[0].jobId, 5);
    EXPECT_FALSE(finished[0].success);
    EXPECT_EQ(finished[1].jobId, 6);
    EXPECT_TRUE(finished[1].success);
    EXPECT_TRUE(handle.entry->GetFinished().empty());
    EXPECT_TRUE(
        handle.entry->SubmitGet(5, {MakePayload(16)}, token.Value(), IoContext{0}).Success());
}

TEST(StoreEntryTest, WaitConvertsStoreCompletionsAndIgnoresUnknownIds)
{
    auto handle = MakeEntry(true, true);
    const std::vector<BlockId> keys = MakeKeys(1, 1);
    handle.entry->OnNewRequest(MakeContext("r"));
    auto token = handle.entry->PrepareGet("r", keys);
    ASSERT_TRUE(token);
    EXPECT_TRUE(
        handle.entry->SubmitGet(5, {MakePayload(16)}, token.Value(), IoContext{0}).Success());
    handle.fake->finishedQueue = {
        {5, false}
    };
    auto first = handle.entry->Wait({5, 404});
    ASSERT_EQ(first.size(), 1);
    EXPECT_EQ(first[0].jobId, 5);
    EXPECT_FALSE(first[0].success);
    EXPECT_TRUE(handle.entry->Wait({5}).empty());
    EXPECT_TRUE(handle.entry->GetFinished().empty());
    EXPECT_TRUE(
        handle.entry->SubmitGet(5, {MakePayload(16)}, token.Value(), IoContext{0}).Success());
}

TEST(StoreEntryTest, PrepareGetRejectsOverlappingWindow)
{
    auto handle = MakeEntry(true, true);
    handle.entry->OnNewRequest(MakeContext("r"));
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    ASSERT_TRUE(handle.entry->PrepareGet("r", keys));
    auto overlapping = handle.entry->PrepareGet("r", MakeKeys(5, 2));
    ASSERT_FALSE(overlapping);
    EXPECT_EQ(overlapping.Error(), Status::InvalidParam());
    ASSERT_EQ(handle.fake->pins.size(), 1);  // the rejected Prepare never reached the store
    EXPECT_EQ(handle.fake->pins[0], keys);
}

TEST(StoreEntryTest, PreparePutRejectsOverlappingWindow)
{
    auto handle = MakeEntry(true, true);
    handle.entry->OnNewRequest(MakeContext("r"));
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    ASSERT_TRUE(handle.entry->PreparePut("r", keys));
    auto overlapping = handle.entry->PreparePut("r", MakeKeys(5, 2));
    ASSERT_FALSE(overlapping);
    EXPECT_EQ(overlapping.Error(), Status::InvalidParam());
    ASSERT_EQ(handle.fake->allocs.size(), 1);
    EXPECT_EQ(handle.fake->allocs[0].first, keys);
}

TEST(StoreEntryTest, PreparePrefetchRejectsOverlappingWindow)
{
    auto handle = MakeEntry(true, true);
    handle.entry->OnNewRequest(MakeContext("r"));
    handle.fake->signalPrefetchable = true;
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    ASSERT_FALSE(handle.entry->Lookup("r", keys).Value().has_value());
    ASSERT_TRUE(handle.entry->PreparePrefetch("r"));
    auto overlapping = handle.entry->PreparePrefetch("r");
    ASSERT_FALSE(overlapping);
    EXPECT_EQ(overlapping.Error(), Status::InvalidParam());
    ASSERT_EQ(handle.fake->preparedPrefetches.size(), 1);
}

TEST(StoreEntryTest, WindowReopensAfterComplete)
{
    auto handle = MakeEntry(true, true);
    handle.entry->OnNewRequest(MakeContext("r"));
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    ASSERT_TRUE(handle.entry->PrepareGet("r", keys));
    handle.entry->CompleteGet("r");
    ASSERT_TRUE(handle.entry->PrepareGet("r", keys));
    EXPECT_EQ(handle.fake->pins.size(), 2);
    handle.entry->CompleteGet("r");
    ASSERT_TRUE(handle.entry->PreparePut("r", keys));
    handle.entry->CompletePut("r", true);
    ASSERT_TRUE(handle.entry->PreparePut("r", keys));
    EXPECT_EQ(handle.fake->allocs.size(), 2);
    handle.entry->CompletePut("r", true);
}

TEST(StoreEntryTest, FailedPrepareDoesNotOpenWindow)
{
    auto handle = MakeEntry(true, true);
    handle.entry->OnNewRequest(MakeContext("r"));
    handle.fake->pinFails = true;
    ASSERT_FALSE(handle.entry->PrepareGet("r", MakeKeys(1, 2)));
    handle.fake->pinFails = false;
    // The retry succeeding proves the failed attempt opened no window; the
    // fake records every Pin attempt, failures included.
    ASSERT_TRUE(handle.entry->PrepareGet("r", MakeKeys(1, 2)));
    EXPECT_EQ(handle.fake->pins.size(), 2);
}

TEST(StoreEntryTest, OnRequestFinishedUnpinsOpenGetWindow)
{
    auto handle = MakeEntry(true, true);
    handle.entry->OnNewRequest(MakeContext("r"));
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    ASSERT_TRUE(handle.entry->PrepareGet("r", keys));
    handle.entry->OnRequestFinished("r");
    ASSERT_EQ(handle.fake->unpins.size(), 1);
    EXPECT_EQ(handle.fake->unpins[0], keys);
    EXPECT_FALSE(handle.entry->Lookup("r", keys));  // the request is gone
}

TEST(StoreEntryTest, OnRequestFinishedRollsBackUnsubmittedPutWindow)
{
    auto handle = MakeEntry(true, true);
    handle.entry->OnNewRequest(MakeContext("r"));
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    ASSERT_TRUE(handle.entry->PreparePut("r", keys));
    handle.entry->OnRequestFinished("r");
    ASSERT_EQ(handle.fake->commits.size(), 1);
    EXPECT_EQ(handle.fake->commits[0].first, keys);
    EXPECT_FALSE(handle.fake->commits[0].second);
}

TEST(StoreEntryTest, OnRequestFinishedCancelsOpenPrefetchPlan)
{
    auto handle = MakeEntry(true, true);
    handle.entry->OnNewRequest(MakeContext("r"));
    handle.fake->signalPrefetchable = true;
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    ASSERT_FALSE(handle.entry->Lookup("r", keys).Value().has_value());
    auto plan = handle.entry->PreparePrefetch("r");
    ASSERT_TRUE(plan);
    handle.entry->OnRequestFinished("r");
    ASSERT_EQ(handle.fake->completedPrefetches.size(), 1);
    EXPECT_FALSE(handle.fake->completedPrefetches[0].second);
    EXPECT_EQ(handle.fake->completedPrefetches[0].first.keys, keys);
}

TEST(StoreEntryTest, OnRequestFinishedWithoutWindowsLeavesStoreUntouched)
{
    auto handle = MakeEntry(true, true);
    handle.entry->OnNewRequest(MakeContext("r"));
    ASSERT_TRUE(handle.entry->Lookup("r", MakeKeys(1, 2)));
    handle.entry->OnRequestFinished("r");
    EXPECT_TRUE(handle.fake->unpins.empty());
    EXPECT_TRUE(handle.fake->commits.empty());
    EXPECT_TRUE(handle.fake->completedPrefetches.empty());
}

TEST(StoreEntryTest, CompleteAfterRequestFinishedIsNoOp)
{
    auto handle = MakeEntry(true, true);
    handle.entry->OnNewRequest(MakeContext("r"));
    const std::vector<BlockId> keys = MakeKeys(1, 2);
    ASSERT_TRUE(handle.entry->PrepareGet("r", keys));
    handle.entry->OnRequestFinished("r");  // the teardown already unpinned
    handle.entry->CompleteGet("r");        // state is gone: silent no-op
    ASSERT_EQ(handle.fake->unpins.size(), 1);
}

TEST(StoreEntryTest, GetFinishedSkipsStoreWhenIdle)
{
    auto handle = MakeEntry(true, true);
    EXPECT_TRUE(handle.entry->GetFinished().empty());
    EXPECT_EQ(handle.fake->getFinishedCalls, 0u);
}

TEST(StoreEntryTest, WaitDeliversManySyntheticCompletionsWithoutStoreRoundTrip)
{
    auto handle = MakeEntry(false, true);
    AddressedToken token;
    std::vector<std::size_t> jobIds;
    for (std::size_t id = 1; id <= 64; id++) {
        ASSERT_TRUE(handle.entry->SubmitPrefetch(id, token).Success());
        jobIds.push_back(id);
    }
    auto results = handle.entry->Wait(jobIds);
    ASSERT_EQ(results.size(), jobIds.size());
    for (std::size_t i = 0; i < results.size(); i++) {
        EXPECT_EQ(results[i].jobId, jobIds[i]);
        EXPECT_TRUE(results[i].success);
    }
    EXPECT_TRUE(handle.fake->waits.empty());
    EXPECT_TRUE(handle.entry->Wait(jobIds).empty());
    EXPECT_TRUE(handle.entry->GetFinished().empty());
    EXPECT_TRUE(handle.entry->SubmitPrefetch(7, token).Success());
}

TEST(StoreEntryTest, WaitDeliversSyntheticCompletionsInWaitedOrder)
{
    auto handle = MakeEntry(false, true);
    AddressedToken token;
    EXPECT_TRUE(handle.entry->SubmitPrefetch(3, token).Success());
    EXPECT_TRUE(handle.entry->SubmitPrefetch(7, token).Success());
    auto results = handle.entry->Wait({7, 3});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].jobId, 7u);
    EXPECT_TRUE(results[0].success);
    EXPECT_EQ(results[1].jobId, 3u);
    EXPECT_TRUE(handle.fake->waits.empty());
    EXPECT_TRUE(handle.entry->Wait({7, 3}).empty());
    EXPECT_TRUE(handle.entry->GetFinished().empty());
}

}  // namespace
}  // namespace UC::Proxy
