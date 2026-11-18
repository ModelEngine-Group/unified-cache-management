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
#include "cache/cache_config.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include "status/status.h"
#include "store.h"
#include "type/dictionary.h"

namespace UC::Store {
namespace {

constexpr std::int64_t kMb = 1048576;

Dictionary MakeDict(const std::string& omitted = {})
{
    Dictionary dict;
    if (omitted != "unique_id") {
        dict.Set("unique_id", std::string("0123456789abcdef0123456789abcdef"));
    }
    if (omitted != "llm_role") { dict.Set("llm_role", std::string("worker")); }
    if (omitted != "worker_number") { dict.SetInt64("worker_number", 8); }
    if (omitted != "worker_index") { dict.SetInt64("worker_index", 0); }
    if (omitted != "device_id") { dict.SetInt64("device_id", 3); }
    if (omitted != "block_size") { dict.SetInt64("block_size", kMb); }
    if (omitted != "cache_buffer_capacity_gb") { dict.SetInt64("cache_buffer_capacity_gb", 2); }
    if (omitted != "cache_replacement_policy") {
        dict.Set("cache_replacement_policy", std::string("lru"));
    }
    if (omitted != "cache_force_huge_page") { dict.Set("cache_force_huge_page", true); }
    if (omitted != "cache_dma_stream_number") { dict.SetInt64("cache_dma_stream_number", 8); }
    if (omitted != "cache_tier_batch_size") { dict.SetInt64("cache_tier_batch_size", 512); }
    return dict;
}

Dictionary MakeRequiredOnlyDict()
{
    Dictionary dict;
    dict.Set("unique_id", std::string("0123456789abcdef0123456789abcdef"));
    dict.Set("llm_role", std::string("both"));
    dict.SetInt64("worker_number", 1);
    dict.SetInt64("worker_index", 0);
    dict.SetInt64("device_id", 0);
    dict.SetInt64("block_size", kMb);
    return dict;
}

TEST(CacheConfigTest, FromAcceptsFullConfig)
{
    auto res = CacheConfig::From(MakeDict());
    ASSERT_TRUE(res.HasValue());
    const auto& cfg = res.Value();
    EXPECT_EQ(cfg.uniqueId, "0123456789abcdef0123456789abcdef");
    EXPECT_EQ(cfg.llmRole, "worker");
    EXPECT_EQ(cfg.nWorker, std::size_t{8});
    EXPECT_EQ(cfg.deviceId, 3);
    EXPECT_EQ(cfg.capacityGb, std::int64_t{2});
    EXPECT_EQ(cfg.blockSize, std::size_t{kMb});
    EXPECT_EQ(cfg.blockTotalNumber, std::size_t{2048});
    EXPECT_EQ(cfg.replacementPolicy, "lru");
    EXPECT_TRUE(cfg.forceHugePage);
    EXPECT_EQ(cfg.nDmaStream, std::size_t{8});
    EXPECT_EQ(cfg.tierBatchSize, std::size_t{512});
}

TEST(CacheConfigTest, FromAcceptsAllRoles)
{
    for (const auto* role : {"scheduler", "worker", "both"}) {
        Dictionary dict = MakeDict();
        dict.Set("llm_role", std::string(role));
        auto res = CacheConfig::From(dict);
        ASSERT_TRUE(res.HasValue()) << role;
        EXPECT_EQ(res.Value().llmRole, role);
    }
}

TEST(CacheConfigTest, FromDerivesBlockTotalNumberWithFloor)
{
    Dictionary dict = MakeDict();
    dict.SetInt64("cache_buffer_capacity_gb", 1);
    dict.SetInt64("block_size", 3 * kMb);
    auto res = CacheConfig::From(dict);
    ASSERT_TRUE(res.HasValue());
    EXPECT_EQ(res.Value().blockTotalNumber, std::size_t{341});
}

TEST(CacheConfigTest, FromAppliesDefaultsWhenOptionalKeysAbsent)
{
    auto res = CacheConfig::From(MakeRequiredOnlyDict());
    ASSERT_TRUE(res.HasValue());
    const auto& cfg = res.Value();
    EXPECT_EQ(cfg.capacityGb, std::int64_t{128});
    EXPECT_EQ(cfg.replacementPolicy, "clock");
    EXPECT_FALSE(cfg.forceHugePage);
    EXPECT_EQ(cfg.nDmaStream, std::size_t{16});
    EXPECT_EQ(cfg.tierBatchSize, std::size_t{1024});
    EXPECT_EQ(cfg.blockTotalNumber, (std::size_t{128} << 30) / std::size_t{kMb});
}

TEST(CacheConfigTest, FromRejectsMissingRequiredKeys)
{
    for (const auto* key :
         {"unique_id", "llm_role", "worker_number", "worker_index", "device_id", "block_size"}) {
        auto res = CacheConfig::From(MakeDict(key));
        ASSERT_FALSE(res.HasValue()) << key;
        EXPECT_EQ(res.Error(), Status::InvalidParam()) << key;
    }
}

TEST(CacheConfigTest, FromRejectsEmptyUniqueId)
{
    Dictionary dict = MakeDict();
    dict.Set("unique_id", std::string(""));
    auto res = CacheConfig::From(dict);
    ASSERT_FALSE(res.HasValue());
    EXPECT_EQ(res.Error(), Status::InvalidParam());
}

TEST(CacheConfigTest, FromRejectsInvalidRole)
{
    Dictionary dict = MakeDict();
    dict.Set("llm_role", std::string("master"));
    auto res = CacheConfig::From(dict);
    ASSERT_FALSE(res.HasValue());
    EXPECT_EQ(res.Error(), Status::InvalidParam());
}

TEST(CacheConfigTest, FromRejectsNonPositiveWorkerNumber)
{
    Dictionary dict = MakeDict();
    dict.SetInt64("worker_number", 0);
    ASSERT_FALSE(CacheConfig::From(dict).HasValue());
    dict.SetInt64("worker_number", -1);
    ASSERT_FALSE(CacheConfig::From(dict).HasValue());
}

TEST(CacheConfigTest, FromReadsTheWorkerIndex)
{
    /* Zero is a legal index; any other non-negative value reads as is. */
    Dictionary dict = MakeDict();
    dict.SetInt64("worker_index", 0);
    auto zero = CacheConfig::From(dict);
    ASSERT_TRUE(zero.HasValue());
    EXPECT_EQ(zero.Value().workerIndex, std::size_t{0});

    dict.SetInt64("worker_index", 3);
    auto read = CacheConfig::From(dict);
    ASSERT_TRUE(read.HasValue());
    EXPECT_EQ(read.Value().workerIndex, std::size_t{3});

    /* Negative is refused. */
    dict.SetInt64("worker_index", -1);
    EXPECT_FALSE(CacheConfig::From(dict).HasValue());

    /* A value of the wrong type is refused. */
    dict.Set("worker_index", std::string("1"));
    EXPECT_FALSE(CacheConfig::From(dict).HasValue());
}

TEST(CacheConfigTest, FromRejectsDeviceIdOutOfInt32Range)
{
    Dictionary dict = MakeDict();
    dict.SetInt64("device_id", std::int64_t{std::numeric_limits<std::int32_t>::max()} + 1);
    ASSERT_FALSE(CacheConfig::From(dict).HasValue());
    dict.SetInt64("device_id", std::int64_t{std::numeric_limits<std::int32_t>::min()} - 1);
    ASSERT_FALSE(CacheConfig::From(dict).HasValue());
}

TEST(CacheConfigTest, FromRejectsNonPositiveBlockSize)
{
    Dictionary dict = MakeDict();
    dict.SetInt64("block_size", 0);
    ASSERT_FALSE(CacheConfig::From(dict).HasValue());
    dict.SetInt64("block_size", -kMb);
    ASSERT_FALSE(CacheConfig::From(dict).HasValue());
}

TEST(CacheConfigTest, FromRejectsZeroCapacity)
{
    Dictionary dict = MakeDict();
    dict.SetInt64("cache_buffer_capacity_gb", 0);
    ASSERT_FALSE(CacheConfig::From(dict).HasValue());
}

TEST(CacheConfigTest, FromRejectsCapacitySmallerThanOneBlock)
{
    Dictionary dict = MakeDict();
    dict.SetInt64("cache_buffer_capacity_gb", 1);
    dict.SetInt64("block_size", std::int64_t{8} << 30);
    ASSERT_FALSE(CacheConfig::From(dict).HasValue());
}

TEST(CacheConfigTest, FromRejectsCapacityOfWrongType)
{
    Dictionary dict = MakeDict();
    dict.Set("cache_buffer_capacity_gb", std::string("256"));
    auto res = CacheConfig::From(dict);
    ASSERT_FALSE(res.HasValue());
    EXPECT_EQ(res.Error(), Status::InvalidParam());
}

TEST(CacheConfigTest, FromRejectsPolicyOfWrongType)
{
    Dictionary dict = MakeDict();
    dict.SetInt64("cache_replacement_policy", 1);
    auto res = CacheConfig::From(dict);
    ASSERT_FALSE(res.HasValue());
    EXPECT_EQ(res.Error(), Status::InvalidParam());
}

TEST(CacheConfigTest, FromRejectsEmptyPolicy)
{
    Dictionary dict = MakeDict();
    dict.Set("cache_replacement_policy", std::string(""));
    ASSERT_FALSE(CacheConfig::From(dict).HasValue());
}

TEST(CacheConfigTest, FromRejectsHugePageFlagOfWrongType)
{
    Dictionary dict = MakeDict();
    dict.SetInt64("cache_force_huge_page", 1);
    auto res = CacheConfig::From(dict);
    ASSERT_FALSE(res.HasValue());
    EXPECT_EQ(res.Error(), Status::InvalidParam());
}

TEST(CacheConfigTest, FromRejectsNonPositiveDmaStreamNumber)
{
    Dictionary dict = MakeDict();
    dict.SetInt64("cache_dma_stream_number", 0);
    ASSERT_FALSE(CacheConfig::From(dict).HasValue());
    dict.SetInt64("cache_dma_stream_number", -4);
    ASSERT_FALSE(CacheConfig::From(dict).HasValue());
}

TEST(CacheConfigTest, FromRejectsDmaStreamNumberOfWrongType)
{
    Dictionary dict = MakeDict();
    dict.Set("cache_dma_stream_number", std::string("16"));
    auto res = CacheConfig::From(dict);
    ASSERT_FALSE(res.HasValue());
    EXPECT_EQ(res.Error(), Status::InvalidParam());
}

TEST(CacheConfigTest, FromRejectsBadTierBatchSize)
{
    Dictionary dict = MakeDict();
    dict.SetInt64("cache_tier_batch_size", 0);
    ASSERT_FALSE(CacheConfig::From(dict).HasValue());
    dict.SetInt64("cache_tier_batch_size", -4);
    ASSERT_FALSE(CacheConfig::From(dict).HasValue());
    dict.Set("cache_tier_batch_size", std::string("1024"));
    auto res = CacheConfig::From(dict);
    ASSERT_FALSE(res.HasValue());
    EXPECT_EQ(res.Error(), Status::InvalidParam());
}

/*== Assembler-injected lower-tier wiring ==*/

/** A minimal live Store: the lower-tier extraction needs a real Store
 *  pointer to carry, not just the type name. */
class NullStore final : public Store {
public:
    Status Setup(const Dictionary&) override { return Status::Ok(); }
    Expected<std::vector<std::uint8_t>> Lookup(const std::vector<BlockId>&) override
    {
        return Status::Unsupported();
    }
    Status Load(std::size_t, JobDescriptor, AddressedToken) override
    {
        return Status::Unsupported();
    }
    Status Dump(std::size_t, JobDescriptor, AddressedToken) override
    {
        return Status::Unsupported();
    }
    std::vector<std::pair<std::size_t, bool>> GetFinished() override { return {}; }
    Status Wait(std::size_t) override { return Status::InvalidParam(); }
};

TEST(CacheConfigTest, FromCarriesTheWiredLowerStore)
{
    NullStore store;
    Dictionary dict = MakeDict();
    Store* wired = &store; /* The assembler stores a Store*, never the derived pointer. */
    dict.Set("__ucm_lower_store__", wired);
    auto res = CacheConfig::From(dict);
    ASSERT_TRUE(res.HasValue());
    EXPECT_EQ(res.Value().lowerStore, wired);
}

TEST(CacheConfigTest, FromLeavesLowerStoreNullWhenNotWired)
{
    /* Absent wiring: nothing carried. */
    auto res = CacheConfig::From(MakeDict());
    ASSERT_TRUE(res.HasValue());
    EXPECT_EQ(res.Value().lowerStore, nullptr);
}

TEST(CacheConfigTest, FromRejectsAnIllTypedLowerStore)
{
    Dictionary dict = MakeDict();
    dict.Set("__ucm_lower_store__", std::string("not a store"));
    auto res = CacheConfig::From(dict);
    ASSERT_FALSE(res.HasValue());
    EXPECT_EQ(res.Error(), Status::InvalidParam());
}

}  // namespace
}  // namespace UC::Store
