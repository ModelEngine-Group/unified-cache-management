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
#include <cstdint>
#include <gtest/gtest.h>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <utility>
#include "router/config.h"

namespace kv::test {
namespace {

using Attrs = std::unordered_map<std::string, std::string>;

Attrs MakeAttrs(std::initializer_list<std::pair<const char*, const char*>> items)
{
    Attrs attrs;
    for (const auto& item : items) { attrs.emplace(item.first, item.second); }
    return attrs;
}

TEST(RouterConfigTest, EmptyAttrsResetConfigToDefaults)
{
    RouterConfig config;
    // 预置脏值，确认每次构建都会重置为默认值
    config.type = RouterType::MAGLEV_FULL_SPREAD;
    config.ringHash.virtualNodeCount = 1;
    config.maglev.tableSize = 3;
    config.contiguousBlockAffinity.blockCount = 9;
    config.contiguousBlockAffinity.fullSpreadType = RouterType::MAGLEV_FULL_SPREAD;
    config.contiguousBlockAffinity.dynamicAdjustEnabled = true;
    config.batchTopKAffinity.topK = 9;
    config.batchTopKAffinity.dynamicAdjustEnabled = true;

    ASSERT_TRUE(BuildRouterConfigFromAttrs(Attrs{}, config).ok());

    EXPECT_EQ(config.type, RouterType::RING_HASH_FULL_SPREAD);
    EXPECT_EQ(config.ringHash.virtualNodeCount, kDefaultVirtualNodeCount);
    EXPECT_EQ(config.maglev.tableSize, kDefaultMaglevTableSize);
    EXPECT_EQ(config.contiguousBlockAffinity.blockCount, std::uint64_t{1});
    EXPECT_EQ(config.contiguousBlockAffinity.fullSpreadType, RouterType::RING_HASH_FULL_SPREAD);
    EXPECT_FALSE(config.contiguousBlockAffinity.dynamicAdjustEnabled);
    EXPECT_EQ(config.batchTopKAffinity.topK, std::uint64_t{1});
    EXPECT_FALSE(config.batchTopKAffinity.dynamicAdjustEnabled);
}

TEST(RouterConfigTest, ParsesRouterTypeAliases)
{
    struct Case {
        const char* value;
        RouterType expected;
    };
    const Case cases[] = {
        {"RING_HASH",                 RouterType::RING_HASH_FULL_SPREAD    },
        {"ring-hash",                 RouterType::RING_HASH_FULL_SPREAD    },
        {"ring_hash_full_spread",     RouterType::RING_HASH_FULL_SPREAD    },
        {"MAGLEV",                    RouterType::MAGLEV_FULL_SPREAD       },
        {"maglev-full-spread",        RouterType::MAGLEV_FULL_SPREAD       },
        {"CONTIGUOUS_BLOCK_AFFINITY", RouterType::CONTIGUOUS_BLOCK_AFFINITY},
        {"contiguous-block-affinity", RouterType::CONTIGUOUS_BLOCK_AFFINITY},
        {"BATCH_TOPK_AFFINITY",       RouterType::BATCH_TOPK_AFFINITY      },
        {"batch-topk-affinity",       RouterType::BATCH_TOPK_AFFINITY      },
    };

    for (const auto& item : cases) {
        RouterConfig config;
        const auto status = BuildRouterConfigFromAttrs(MakeAttrs({
                                                           {"hash_table.type", item.value}
        }),
                                                       config);
        ASSERT_TRUE(status.ok()) << item.value << ": " << status.message;
        EXPECT_EQ(config.type, item.expected) << item.value;
    }
}

TEST(RouterConfigTest, UnknownRouterTypeKeepsDefault)
{
    RouterConfig config;
    config.type = RouterType::MAGLEV_FULL_SPREAD;

    ASSERT_TRUE(BuildRouterConfigFromAttrs(MakeAttrs({
                                               {"hash_table.type", "NOT_A_ROUTER"}
    }),
                                           config)
                    .ok());

    EXPECT_EQ(config.type, RouterType::RING_HASH_FULL_SPREAD);
}

TEST(RouterConfigTest, ParsesNumericAttributes)
{
    RouterConfig config;

    const auto status =
        BuildRouterConfigFromAttrs(MakeAttrs({
                                       {"ring_hash.virtual_node_count",               "0x10"  },
                                       {"maglev.table_size",                          "257"   },
                                       {"contiguous_block_affinity.block_count",      "3"     },
                                       {"batch_topk_affinity.top_k",                  "5"     },
                                       {"contiguous_block_affinity.full_spread_type", "MAGLEV"},
    }),
                                   config);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(config.ringHash.virtualNodeCount, std::uint64_t{16});
    EXPECT_EQ(config.maglev.tableSize, std::uint64_t{257});
    EXPECT_EQ(config.contiguousBlockAffinity.blockCount, std::uint64_t{3});
    EXPECT_EQ(config.contiguousBlockAffinity.fullSpreadType, RouterType::MAGLEV_FULL_SPREAD);
    EXPECT_EQ(config.batchTopKAffinity.topK, std::uint64_t{5});
}

TEST(RouterConfigTest, ParsesOctalNumericAttribute)
{
    RouterConfig config;

    ASSERT_TRUE(BuildRouterConfigFromAttrs(MakeAttrs({
                                               {"ring_hash.virtual_node_count", "010"}
    }),
                                           config)
                    .ok());

    EXPECT_EQ(config.ringHash.virtualNodeCount, std::uint64_t{8});
}

TEST(RouterConfigTest, RejectsInvalidNumericAttributes)
{
    const std::pair<const char*, const char*> cases[] = {
        {"ring_hash.virtual_node_count",          "abc"                 },
        {"ring_hash.virtual_node_count",          "12abc"               },
        {"ring_hash.virtual_node_count",          ""                    },
        {"maglev.table_size",                     "18446744073709551616"},
        {"contiguous_block_affinity.block_count", "1.5"                 },
        {"batch_topk_affinity.top_k",             "0x"                  },
    };

    for (const auto& item : cases) {
        const std::string key{item.first};
        const std::string value{item.second};
        RouterConfig config;

        const auto status = BuildRouterConfigFromAttrs(MakeAttrs({
                                                           {item.first, item.second}
        }),
                                                       config);

        EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT) << key << "=" << value;
        EXPECT_NE(status.message.find(key), std::string::npos) << status.message;
        EXPECT_NE(status.message.find(value), std::string::npos) << status.message;
    }
}

TEST(RouterConfigTest, UnknownFullSpreadTypeKeepsDefault)
{
    RouterConfig config;
    config.contiguousBlockAffinity.fullSpreadType = RouterType::MAGLEV_FULL_SPREAD;

    const auto status = BuildRouterConfigFromAttrs(
        MakeAttrs({
            {"contiguous_block_affinity.full_spread_type", "NOT_A_ROUTER"}
    }),
        config);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(config.contiguousBlockAffinity.fullSpreadType, RouterType::RING_HASH_FULL_SPREAD);
}

TEST(RouterConfigTest, ParsesBooleanAttributes)
{
    for (const char* value : {"1", "true", "TRUE", "on", "Yes"}) {
        RouterConfig config;
        const auto status = BuildRouterConfigFromAttrs(
            MakeAttrs({
                {"contiguous_block_affinity.dynamic_adjust_enabled", value},
                {"batch_topk_affinity.dynamic_adjust_enabled",       value}
        }),
            config);
        ASSERT_TRUE(status.ok()) << value << ": " << status.message;
        EXPECT_TRUE(config.contiguousBlockAffinity.dynamicAdjustEnabled) << value;
        EXPECT_TRUE(config.batchTopKAffinity.dynamicAdjustEnabled) << value;
    }

    for (const char* value : {"0", "false", "OFF", "no"}) {
        RouterConfig config;
        config.contiguousBlockAffinity.dynamicAdjustEnabled = true;
        config.batchTopKAffinity.dynamicAdjustEnabled = true;
        const auto status = BuildRouterConfigFromAttrs(
            MakeAttrs({
                {"contiguous_block_affinity.dynamic_adjust_enabled", value},
                {"batch_topk_affinity.dynamic_adjust_enabled",       value}
        }),
            config);
        ASSERT_TRUE(status.ok()) << value << ": " << status.message;
        EXPECT_FALSE(config.contiguousBlockAffinity.dynamicAdjustEnabled) << value;
        EXPECT_FALSE(config.batchTopKAffinity.dynamicAdjustEnabled) << value;
    }
}

TEST(RouterConfigTest, RejectsInvalidBooleanAttributes)
{
    for (const char* key : {"contiguous_block_affinity.dynamic_adjust_enabled",
                            "batch_topk_affinity.dynamic_adjust_enabled"}) {
        RouterConfig config;

        const auto status = BuildRouterConfigFromAttrs(MakeAttrs({
                                                           {key, "maybe"}
        }),
                                                       config);

        EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT) << key;
        EXPECT_NE(status.message.find(key), std::string::npos) << status.message;
        EXPECT_NE(status.message.find("maybe"), std::string::npos) << status.message;
    }
}

}  // namespace
}  // namespace kv::test
