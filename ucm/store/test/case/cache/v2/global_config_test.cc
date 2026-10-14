/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "cache/v2/global_config.h"
#include <gtest/gtest.h>
#include <string>

namespace UC::Cache2 {
namespace {

TEST(Cache2ConfigTest, UsesUniqueIdAsDefaultInstanceId)
{
    Detail::Dictionary input;
    input.Set("unique_id", std::string{"shared-engine"});

    const auto config = Config::From(input);

    EXPECT_EQ(config.uniqueId, "shared-engine");
    EXPECT_EQ(config.InstanceId(), "shared-engine");
}

TEST(Cache2ConfigTest, UsesDedicatedInstanceIdWhenConfigured)
{
    Detail::Dictionary input;
    input.Set("unique_id", std::string{"shared-engine"});
    input.Set("cache2_instance_id", std::string{"engine_dp1"});

    const auto config = Config::From(input);

    EXPECT_EQ(config.uniqueId, "shared-engine");
    EXPECT_EQ(config.InstanceId(), "engine_dp1");
}

}  // namespace
}  // namespace UC::Cache2
