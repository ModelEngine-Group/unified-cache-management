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
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include "config.h"

namespace UC::Dram {
namespace {

Detail::Dictionary BaseConfig(bool includeIoLengths = true)
{
    Detail::Dictionary config;
    config.Set("local_control_endpoint", std::string{"127.0.0.1:6000"});
    config.Set("local_host", std::string{"127.0.0.1"});
    config.Set("local_transport_manager_id", std::string{"127.0.0.1:6100"});
    config.Set("node_ids", std::vector<ssize_t>{7, 9});
    config.Set("node_control_endpoints",
               std::vector<std::string>{"127.0.0.1:7000", "127.0.0.1:9000"});
    config.Set("node_transport_manager_ids",
               std::vector<std::string>{"127.0.0.1:7100", "127.0.0.1:9100"});
    if (includeIoLengths) { config.Set("io_lengths", std::vector<ssize_t>{4096}); }
    return config;
}

TEST(DramConfigTest, KeepsOnlyTaskAndEntryBudgets)
{
    auto input = BaseConfig();
    input.SetNumber("max_tasks", 17);
    input.SetNumber("max_io_items", 23);
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    const auto& config = parsed.Value();
    EXPECT_EQ(config.taskLimits.maxLiveTasks, std::size_t{17});
    EXPECT_EQ(config.taskLimits.maxLiveEntries, std::size_t{23});
}

TEST(DramConfigTest, DerivesNodeAndReplyCapacitiesFromGlobalEntryBudget)
{
    auto input = BaseConfig();
    input.SetNumber("max_io_items", 7);
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    const auto& config = parsed.Value();
    EXPECT_EQ(config.nodeScheduler.limits.maxPendingRequests, std::size_t{7});
    EXPECT_EQ(config.nodeScheduler.limits.maxInflightRequests, std::size_t{7});
    EXPECT_EQ(config.nodeScheduler.limits.maxPendingEntries, std::size_t{7});
    EXPECT_EQ(config.nodeScheduler.limits.maxBatchEntries, std::size_t{7});
    EXPECT_EQ(config.replySlotCount, std::size_t{14});
    EXPECT_EQ(config.replySlotSize, std::uint32_t{5});
}

TEST(DramConfigTest, CapsDerivedNodeAndReplyCapacities)
{
    auto input = BaseConfig();
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    const auto& config = parsed.Value();
    EXPECT_EQ(config.nodeScheduler.limits.maxPendingRequests, std::size_t{1024});
    EXPECT_EQ(config.nodeScheduler.limits.maxInflightRequests, std::size_t{128});
    EXPECT_EQ(config.nodeScheduler.limits.maxPendingEntries, std::size_t{65536});
    EXPECT_EQ(config.nodeScheduler.limits.maxBatchEntries, std::size_t{128});
    EXPECT_EQ(config.replySlotCount, std::size_t{256});
    EXPECT_EQ(config.replySlotSize, std::uint32_t{65});
}

TEST(DramConfigTest, DerivesRuntimeWorkersFromHardwareAndNodeCount)
{
    auto input = BaseConfig();
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    const auto hardwareThreads =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::thread::hardware_concurrency()));
    const auto expected = std::min<std::size_t>(2, std::max<std::size_t>(1, hardwareThreads / 2));
    EXPECT_EQ(parsed.Value().nodeScheduler.runnerCount, expected);
    EXPECT_EQ(parsed.Value().transportRuntime.workerCount, expected);
}

TEST(DramConfigTest, ParsesFixedReconnectInterval)
{
    auto input = BaseConfig();
    input.SetNumber("reconnect_interval_ms", 37);
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed.Value().nodeScheduler.reconnectInterval.count(), 37);
}

TEST(DramConfigTest, RequiresIoLengths) { EXPECT_FALSE(DramConfig::Parse(BaseConfig(false))); }

TEST(DramConfigTest, ParsesRouterTypeIntoStrongConfiguration)
{
    auto input = BaseConfig();
    input.Set("router_type", std::string{"maglev"});
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed.Value().routerType, UC::KV::RouterType::MAGLEV_FULL_SPREAD);

    input.Set("router_type", std::string{"unsupported"});
    EXPECT_FALSE(DramConfig::Parse(input));
}

TEST(DramConfigTest, RejectsMalformedControlEndpointsAndEmptyManagerIds)
{
    auto local = BaseConfig();
    local.Set("local_control_endpoint", std::string{"client"});
    EXPECT_FALSE(DramConfig::Parse(local));

    auto manager = BaseConfig();
    manager.Set("local_transport_manager_id", std::string{});
    EXPECT_FALSE(DramConfig::Parse(manager));

    auto remote = BaseConfig();
    remote.Set("node_transport_manager_ids", std::vector<std::string>{"", "127.0.0.1:9100"});
    EXPECT_FALSE(DramConfig::Parse(remote));
}

TEST(DramConfigTest, ParsesControlEndpointsAndStoresManagerIds)
{
    auto input = BaseConfig();
    input.Set("local_control_endpoint", std::string{"127.0.0.1:06000"});
    auto parsed = DramConfig::Parse(input);
    ASSERT_TRUE(parsed);
    const auto& config = parsed.Value();
    EXPECT_EQ(config.localControlHost, "127.0.0.1");
    EXPECT_EQ(config.localControlPort, std::uint16_t{6000});
    EXPECT_EQ(config.localTransportManagerId, "127.0.0.1:6100");
    ASSERT_EQ(config.nodeScheduler.nodes.size(), std::size_t{2});
    EXPECT_EQ(config.nodeScheduler.nodes[0].controlHost, "127.0.0.1");
    EXPECT_EQ(config.nodeScheduler.nodes[0].controlPort, std::uint16_t{7000});
    EXPECT_EQ(config.nodeScheduler.nodes[0].transportManagerId, "127.0.0.1:7100");
}

TEST(DramConfigTest, RejectsIoLengthOutsideProtocolRange)
{
    if (sizeof(ssize_t) <= sizeof(std::uint32_t)) {
        GTEST_SKIP() << "ssize_t cannot represent an oversized IO length";
    }
    auto input = BaseConfig();
    input.Set("io_lengths",
              std::vector<ssize_t>{static_cast<ssize_t>(
                  static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1)});
    EXPECT_FALSE(DramConfig::Parse(input));
}

}  // namespace
}  // namespace UC::Dram
