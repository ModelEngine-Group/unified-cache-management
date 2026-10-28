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
#include <gtest/gtest.h>
#include <string>
#include "cache/v2/global_config.h"

namespace {

UC::Detail::Dictionary MakeWorkerConfig()
{
    UC::Detail::Dictionary dict;
    dict.Set("unique_id", std::string{"cache2-copy-config-test"});
    dict.SetNumber("device_id", 0);
    dict.SetNumber("tensor_size", 4096);
    dict.SetNumber("shard_size", 4096);
    dict.SetNumber("block_size", 4096);
    return dict;
}

TEST(UCCache2CopyConfigTest, DefaultsFollowRuntime)
{
    const auto config = UC::Cache2::Config::From(MakeWorkerConfig());

    EXPECT_EQ(config.bufferCapacity, 160ULL << 30);
    EXPECT_FALSE(config.ioAggregation);
    EXPECT_EQ(config.sdmaDirect, bool(UCM_RUNTIME_ASCEND_SDMA_DIRECT));
    EXPECT_TRUE(config.Validate().Success());
}

TEST(UCCache2CopyConfigTest, HmaFaAggregationDefaultRespectsRuntime)
{
    auto dict = MakeWorkerConfig();
    // HMA injects this default even when the user's configuration omits it.
    dict.Set("cache_io_aggregation", true);
    const auto config = UC::Cache2::Config::From(dict);

    EXPECT_EQ(config.ioAggregation, bool(UCM_RUNTIME_ASCEND_IO_AGGREGATION));
    EXPECT_EQ(config.sdmaDirect, bool(UCM_RUNTIME_ASCEND_SDMA_DIRECT));
    EXPECT_TRUE(config.Validate().Success());
}

TEST(UCCache2CopyConfigTest, ExplicitDisableKeepsPlainCopy)
{
    auto dict = MakeWorkerConfig();
    dict.Set("cache_io_aggregation", false);
    dict.Set("cache_sdma_direct", false);
    const auto config = UC::Cache2::Config::From(dict);

    EXPECT_FALSE(config.ioAggregation);
    EXPECT_FALSE(config.sdmaDirect);
    EXPECT_TRUE(config.Validate().Success());
}

TEST(UCCache2CopyConfigTest, AggregationRequestPreservesExplicitSdmaDisable)
{
    auto dict = MakeWorkerConfig();
    dict.Set("cache_io_aggregation", true);
    dict.Set("cache_sdma_direct", false);
    const auto config = UC::Cache2::Config::From(dict);

    EXPECT_EQ(config.ioAggregation, bool(UCM_RUNTIME_ASCEND_IO_AGGREGATION));
    EXPECT_FALSE(config.sdmaDirect);
    EXPECT_TRUE(config.Validate().Success());
}

TEST(UCCache2CopyConfigTest, ValidationStillRejectsConflictingCopyModes)
{
    auto config = UC::Cache2::Config::From(MakeWorkerConfig());
    config.ioAggregation = true;
    config.sdmaDirect = true;
    const auto status = config.Validate();

    EXPECT_TRUE(status.Failure());
#if UCM_RUNTIME_ASCEND_SDMA_DIRECT
    EXPECT_NE(status.ToString().find("incompatible"), std::string::npos);
#endif
}

}  // namespace
